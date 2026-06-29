#!/usr/bin/env python3
import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from tqdm.auto import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from utils.system_prompts import (
    SYSTEM_PROMPT_NO_REASONING,
    SYSTEM_PROMPT_NO_REASONING_NUMERIC,
    SYSTEM_PROMPT_WITH_REASONING,
    SYSTEM_PROMPT_WITH_BRIEF_REASONING,
)

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Thinking"
DEFAULT_DATA_JSON = "data/SpatiaLab/data.json"
DEFAULT_IMG_DIR = "data/SpatiaLab/img"
DEFAULT_LOG_DIR = "logs"

# Direct-text answer: model outputs the option text verbatim instead of a digit.
SYSTEM_PROMPT_TEXT_OUTPUT = (
    "You are a spatial reasoning assistant. "
    "Given an image and a multiple-choice question, output only the exact text of the correct option. "
    "Do not output any explanation, punctuation, or extra words — just the option text verbatim."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen3-VL over SpatiaLab test samples.")
    parser.add_argument("--data-json", default=DEFAULT_DATA_JSON)
    parser.add_argument("--img-dir", default=DEFAULT_IMG_DIR)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--adapter-path", default=None, help="LoRA adapter to merge into base model.")
    parser.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help="Categories to run (space or comma separated). If omitted, runs all.",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--with-reasoning", action="store_true")
    parser.add_argument(
        "--prompt-variant",
        choices=["default", "brief"],
        default="default",
        help="'default': full bbox grounding prompt. 'brief': 2-3 line thinking, no bbox (for base thinking models).",
    )
    parser.add_argument(
        "--answer-mode",
        choices=["letter", "text", "numeric"],
        default="letter",
        help="'letter': options labeled A/B/C/D, model outputs a letter. "
             "'text': options listed as bare text, model outputs the option text. "
             "'numeric': options labeled 1/2/3/4, model outputs a digit.",
    )
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--vllm-max-model-len", type=int, default=16384)
    parser.add_argument("--print-generated-text", action="store_true")
    return parser.parse_args()


def normalize_category_args(raw: Optional[Sequence[str]]) -> Optional[List[str]]:
    if not raw:
        return None
    items: List[str] = []
    for chunk in raw:
        for part in chunk.split(","):
            part = part.strip()
            if part:
                items.append(part)
    return items or None


def load_data(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_image_path(img_dir: Path, filename: str) -> Path:
    path = img_dir / filename
    if path.exists():
        return path
    raise FileNotFoundError(f"Image not found: {path}")


def build_user_text(question: str, options: List[str], answer_mode: str = "letter") -> str:
    lines = [question.strip(), "", "Options:"]
    for i, opt in enumerate(options):
        if answer_mode == "text":
            lines.append(f"- {opt}")
        elif answer_mode == "numeric":
            lines.append(f"{i + 1}. {opt}")
        else:
            lines.append(f"{chr(ord('A') + i)}. {opt}")
    if answer_mode == "letter":
        lines.append("")
        lines.append("Select exactly one option letter based on the image.")
    return "\n".join(lines)


def build_messages(image_path: Path, system_prompt: str, prompt: str) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        },
    ]


def infer_batch_hf(
    model,
    processor: AutoProcessor,
    image_paths: Sequence[Path],
    system_prompt: str,
    prompts: Sequence[str],
    max_new_tokens: int,
    temperature: float,
    enable_thinking: bool,
) -> List[str]:
    all_messages = [
        build_messages(ip, system_prompt, p) for ip, p in zip(image_paths, prompts)
    ]
    texts = [
        processor.apply_chat_template(
            m, tokenize=False, add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": enable_thinking},
        )
        for m in all_messages
    ]
    image_inputs, _ = process_vision_info(all_messages)
    processor.tokenizer.padding_side = "left"
    inputs = processor(text=texts, images=image_inputs, padding=True, return_tensors="pt")
    inputs = {k: v.to("cuda:0") if torch.is_tensor(v) else v for k, v in inputs.items()}

    gen_kwargs: Dict[str, Any] = {"max_new_tokens": max_new_tokens}
    if temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature

    with torch.inference_mode():
        outputs = model.generate(**inputs, **gen_kwargs)

    generated = [
        out[len(inp):] for inp, out in zip(inputs["input_ids"], outputs)
    ]
    return [processor.decode(g, skip_special_tokens=True).strip() for g in generated]


def infer_batch_vllm(
    llm,
    processor: AutoProcessor,
    image_paths: Sequence[Path],
    system_prompt: str,
    prompts: Sequence[str],
    max_new_tokens: int,
    temperature: float,
    enable_thinking: bool,
) -> List[str]:
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=temperature if temperature > 0 else 0.0,
        max_tokens=max_new_tokens,
    )
    vllm_inputs = []
    for image_path, prompt in zip(image_paths, prompts):
        messages = build_messages(image_path, system_prompt, prompt)
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": enable_thinking},
        )
        image_inputs, _ = process_vision_info(messages)
        vllm_inputs.append({"prompt": text, "multi_modal_data": {"image": image_inputs}})

    outputs = llm.generate(vllm_inputs, sampling_params)
    return [o.outputs[0].text.strip() for o in outputs]


def extract_answer_digit(text: str, options: Optional[List[str]] = None) -> Optional[str]:
    # Text-match mode: match the model output against the option strings.
    if options:
        t = text.strip().lower()
        for i, opt in enumerate(options):
            if t == opt.strip().lower():
                return str(i + 1)
        for i, opt in enumerate(options):
            opt_lower = opt.strip().lower()
            if opt_lower and (opt_lower in t or t in opt_lower):
                return str(i + 1)

    # Tagged answer: <result>N</result>
    m = re.search(r"<result>\s*([1-4])\s*</result>", text, re.IGNORECASE)
    if m:
        return m.group(1)

    # Restrict the answer search to the text after </think> so we do not
    # accidentally pick up digits/letters inside the reasoning trace.
    think_end = text.rfind("</think>")
    answer_text = text[think_end + len("</think>"):].strip() if think_end >= 0 else text

    digit_matches = re.findall(r"\b([1-4])\b", answer_text)
    if digit_matches:
        return digit_matches[0]

    # Letter A-D mapped to digit 1-4 (handles models that emit letter answers).
    letter_matches = re.findall(r"\b([A-D])\b", answer_text, re.IGNORECASE)
    if letter_matches:
        return str(ord(letter_matches[0].upper()) - ord("A") + 1)

    return None


def default_output_path() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"spatiallab_results_{ts}.jsonl"


def main() -> int:
    args = parse_args()

    data = load_data(args.data_json)
    img_dir = Path(args.img_dir)

    all_categories = sorted({str(s["Category"]) for s in data})
    requested = normalize_category_args(args.categories)

    if requested is None:
        selected = set(all_categories)
    else:
        unknown = sorted(set(requested) - set(all_categories))
        if unknown:
            print(f"[ERROR] Unknown categories: {unknown}. Available: {all_categories}", file=sys.stderr)
            return 2
        selected = set(requested)

    filtered = [s for s in data if str(s["Category"]) in selected]
    if args.limit:
        filtered = filtered[: args.limit]

    print(f"Loaded samples: {len(data)}")
    print(f"Selected categories: {sorted(selected)}")
    print(f"Samples to run: {len(filtered)}")

    log_dir = Path(DEFAULT_LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    output_path = log_dir / (Path(args.output).name if args.output else default_output_path())

    print(f"Loading model: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id)

    if args.with_reasoning:
        if args.prompt_variant == "brief":
            system_prompt = SYSTEM_PROMPT_WITH_BRIEF_REASONING
        else:
            system_prompt = SYSTEM_PROMPT_WITH_REASONING
    elif args.answer_mode == "text":
        system_prompt = SYSTEM_PROMPT_TEXT_OUTPUT
    elif args.answer_mode == "numeric":
        system_prompt = SYSTEM_PROMPT_NO_REASONING_NUMERIC
    else:
        system_prompt = SYSTEM_PROMPT_NO_REASONING
    print(f"Reasoning mode: {'on' if args.with_reasoning else 'off'}, prompt variant: {args.prompt_variant}, answer mode: {args.answer_mode}")

    if args.use_vllm:
        from vllm import LLM
        llm = LLM(
            model=args.model_id,
            dtype="bfloat16",
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_model_len=args.vllm_max_model_len,
            limit_mm_per_prompt={"image": 1},
        )
        model = None
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_id, torch_dtype=torch.bfloat16, device_map="cuda:0"
        )
        if args.adapter_path:
            from peft import PeftModel
            print(f"Loading adapter: {args.adapter_path}")
            model = PeftModel.from_pretrained(model, args.adapter_path)
            model = model.merge_and_unload()
            print("Adapter merged.")
        llm = None

    succeeded = failed = 0

    with output_path.open("w", encoding="utf-8") as out_f:
        for batch_start in tqdm(range(0, len(filtered), args.batch_size), desc="Inference", unit="batch"):
            batch = filtered[batch_start : batch_start + args.batch_size]
            results, image_paths, prompts = [], [], []

            for sample in batch:
                options = [
                    str(sample["Option_1"]), str(sample["Option_2"]),
                    str(sample["Option_3"]), str(sample["Option_4"]),
                ]
                gold_digit = str(sample["Answer"]).strip()
                gold_idx = int(gold_digit) - 1
                gold_text = options[gold_idx] if 0 <= gold_idx < 4 else gold_digit

                result: Dict[str, Any] = {
                    "id": str(sample["id"]),
                    "category": str(sample["Category"]),
                    "sub_category": str(sample["Sub_Category"]),
                    "question": str(sample["Question"]),
                    "options": options,
                    "gold_answer": gold_digit,
                    "gold_answer_text": gold_text,
                    "status": "ok",
                    "prediction": None,
                    "raw_generation": None,
                    "image_path": None,
                    "error": None,
                }

                try:
                    image_path = resolve_image_path(img_dir, str(sample["Image_Filename"]))
                    result["image_path"] = str(image_path)
                    prompts.append(build_user_text(str(sample["Question"]), options, args.answer_mode))
                    image_paths.append(image_path)
                except Exception as exc:
                    result["status"] = "error"
                    result["error"] = str(exc)
                    failed += 1

                results.append(result)

            ready = [i for i, r in enumerate(results) if r["status"] == "ok"]
            if ready:
                try:
                    if args.use_vllm:
                        generations = infer_batch_vllm(
                            llm, processor, image_paths, system_prompt, prompts,
                            args.max_new_tokens, args.temperature, args.with_reasoning,
                        )
                    else:
                        generations = infer_batch_hf(
                            model, processor, image_paths, system_prompt, prompts,
                            args.max_new_tokens, args.temperature, args.with_reasoning,
                        )
                    for gen, idx in zip(generations, ready):
                        r = results[idx]
                        r["raw_generation"] = gen
                        opts = r.get("options") if args.answer_mode == "text" else None
                        digit = extract_answer_digit(gen, opts)
                        r["prediction"] = digit or ""
                        if args.print_generated_text:
                            print(f"[GEN][id={r['id']}] pred={digit} | {gen[:120]}", flush=True)
                        succeeded += 1
                except Exception as exc:
                    for idx in ready:
                        results[idx]["status"] = "error"
                        results[idx]["error"] = str(exc)
                        failed += 1

            for r in results:
                out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
            out_f.flush()

    print(f"Done. Output: {output_path}")
    print(f"success={succeeded}, failed={failed}, total={len(filtered)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
