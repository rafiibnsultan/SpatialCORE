#!/usr/bin/env python3
import argparse
import importlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from tqdm.auto import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

DEFAULT_MODEL_ID_WITH_REASONING = "Qwen/Qwen3-VL-8B-Thinking"
DEFAULT_MODEL_ID_WITHOUT_REASONING = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_DATA_JSON = "data/OmniSpatial-test/data.json"
DEFAULT_ROOT_DIR = "data/OmniSpatial-test"
DEFAULT_LOG_DIR = "logs"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qwen3-VL over OmniSpatial test samples."
    )
    parser.add_argument("--data-json", default=DEFAULT_DATA_JSON)
    parser.add_argument("--root-dir", default=DEFAULT_ROOT_DIR)
    parser.add_argument(
        "--task-types",
        nargs="*",
        default=None,
        help=(
            "Task types to run. Can be passed as space-separated values or comma-separated"
            " groups. If omitted, all task types in data.json are used."
        ),
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help=(
            "Optional explicit model id. If omitted, uses "
            f"{DEFAULT_MODEL_ID_WITH_REASONING} with --with-reasoning, else "
            f"{DEFAULT_MODEL_ID_WITHOUT_REASONING}."
        ),
    )
    parser.add_argument("--output", default=None, help="Output JSONL file path.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--print-generated-text",
        action="store_true",
        help="Print raw generated text to console for each sample (default: off).",
    )
    parser.add_argument(
        "--print-input-messages",
        action="store_true",
        help="Print the chat messages (system/user) before generation (default: off).",
    )
    parser.add_argument(
        "--with-reasoning",
        action="store_true",
        help="Enable Qwen3 thinking mode.",
    )
    parser.add_argument(
        "--prompt-variant",
        default="default",
        choices=["default", "forced-bbox", "generic"],
        help=(
            "System prompt variant. 'default' uses SYSTEM_PROMPT_WITH_REASONING (bboxes optional). "
            "'forced-bbox' uses SYSTEM_PROMPT_WITH_FORCED_BBOX (bboxes required every sample). "
            "'generic' uses SYSTEM_PROMPT_WITH_GENERIC_REASONING (no bbox instructions, pure spatial reasoning test)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate filtering and file mapping without loading/running the model.",
    )
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="Path to a PEFT/LoRA adapter checkpoint to load on top of the base model.",
    )
    parser.add_argument(
        "--use-vllm",
        action="store_true",
        help="Use vLLM for fast batched inference instead of HF generate.",
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.85,
        help="vLLM gpu_memory_utilization (default 0.85).",
    )
    parser.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=16384,
        help="vLLM max_model_len (default 16384).",
    )
    return parser.parse_args()


def normalize_task_args(raw: Optional[Sequence[str]]) -> Optional[List[str]]:
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
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("data", "samples", "items"):
            if key in data and isinstance(data[key], list):
                return data[key]

    raise ValueError(f"Unsupported JSON format in {path}")


def resolve_image_path(root_dir: Path, task_type: str, sample_id: str) -> Path:
    image_idx = str(sample_id).split("_", 1)[0]
    task_dir = root_dir / task_type

    for ext in IMAGE_EXTENSIONS:
        candidate = task_dir / f"{image_idx}{ext}"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"No image found for id={sample_id} under {task_dir} (tried: {IMAGE_EXTENSIONS})"
    )


def build_user_text(question: str, options: Iterable[str], with_reasoning: bool) -> str:
    options_list = list(options)
    lines = [question.strip(), ""]
    if options_list:
        lines.append("Options:")
        for i, option in enumerate(options_list):
            letter = chr(ord("A") + i)
            lines.append(f"{letter}. {option}")
        lines.append("")
        lines.append("Select exactly one option letter based on the image.")
    else:
        lines.append(
            "The answer choices are shown inside the image itself. Select exactly one option "
            "letter from A, B, C, or D based on the visual candidates in the image."
        )
    # The system prompt defines output format; the user message is task-only.
    _ = with_reasoning
    return "\n".join(lines)


def _first_existing_attr(module: Any, names: Sequence[str]) -> Optional[str]:
    for name in names:
        value = getattr(module, name, None)
        if isinstance(value, str) and value.strip():
            return value
    return None


def load_system_prompt(with_reasoning: bool) -> str:
    mod = importlib.import_module("utils.system_prompts")

    if with_reasoning:
        candidates = (
            "SYSTEM_PROMPT_WITH_REASONING",
            "REASONING_SYSTEM_PROMPT",
            "SYSTEM_PROMPT_REASONING",
            "WITH_REASONING_SYSTEM_PROMPT",
        )
    else:
        candidates = (
            "SYSTEM_PROMPT_NO_REASONING",
            "NO_REASONING_SYSTEM_PROMPT",
            "SYSTEM_PROMPT_WITHOUT_REASONING",
            "SYSTEM_PROMPT",
        )

    prompt = _first_existing_attr(mod, candidates)
    if prompt is None:
        mode = "with reasoning" if with_reasoning else "without reasoning"
        raise ValueError(
            f"Could not find a {mode} prompt in utils/system_prompts.py. "
            f"Tried names: {list(candidates)}"
        )
    return prompt


def prepare_inputs_with_fallbacks(
    processor: AutoProcessor,
    image_path: Path,
    system_prompt: Optional[str],
    prompt: str,
    enable_thinking: bool,
    print_input_messages: bool = False,
) -> Dict[str, torch.Tensor]:
    messages = []
    if system_prompt and system_prompt.strip():
        messages.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            }
        )
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    )
    if print_input_messages:
        print("[INPUT_MESSAGES]")
        print(json.dumps(messages, ensure_ascii=False, indent=2), flush=True)
    return processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        chat_template_kwargs={"enable_thinking": enable_thinking},
    )


def build_messages(
    image_path: Path,
    system_prompt: Optional[str],
    prompt: str,
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    if system_prompt and system_prompt.strip():
        messages.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            }
        )
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    )
    return messages


def prepare_batched_inputs(
    processor: AutoProcessor,
    image_paths: Sequence[Path],
    system_prompt: Optional[str],
    prompts: Sequence[str],
    enable_thinking: bool,
    print_input_messages: bool = False,
) -> Dict[str, torch.Tensor]:
    all_messages = [
        build_messages(image_path=image_path, system_prompt=system_prompt, prompt=prompt)
        for image_path, prompt in zip(image_paths, prompts)
    ]
    if print_input_messages:
        print("[INPUT_MESSAGES_BATCH]")
        print(json.dumps(all_messages, ensure_ascii=False, indent=2), flush=True)

    texts = [
        processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": enable_thinking},
        )
        for messages in all_messages
    ]
    image_inputs, _ = process_vision_info(all_messages)
    processor.tokenizer.padding_side = "left"
    return processor(
        text=texts,
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )


def infer_one(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    image_path: Path,
    system_prompt: Optional[str],
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    enable_thinking: bool,
    print_input_messages: bool = False,
) -> str:
    inputs = prepare_inputs_with_fallbacks(
        processor=processor,
        image_path=image_path,
        system_prompt=system_prompt,
        prompt=prompt,
        enable_thinking=enable_thinking,
        print_input_messages=print_input_messages,
    )
    inputs = {key: value.to("cuda:0") if torch.is_tensor(value) else value for key, value in inputs.items()}

    gen_kwargs: Dict[str, Any] = {"max_new_tokens": max_new_tokens}
    if temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature

    with torch.inference_mode():
        outputs = model.generate(**inputs, **gen_kwargs)

    input_len = inputs["input_ids"].shape[-1]
    text = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
    return text.strip()


def infer_batch(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    image_paths: Sequence[Path],
    system_prompt: Optional[str],
    prompts: Sequence[str],
    max_new_tokens: int,
    temperature: float,
    enable_thinking: bool,
    print_input_messages: bool = False,
) -> List[str]:
    inputs = prepare_batched_inputs(
        processor=processor,
        image_paths=image_paths,
        system_prompt=system_prompt,
        prompts=prompts,
        enable_thinking=enable_thinking,
        print_input_messages=print_input_messages,
    )
    inputs = {
        key: value.to("cuda:0") if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }

    gen_kwargs: Dict[str, Any] = {"max_new_tokens": max_new_tokens}
    if temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature

    with torch.inference_mode():
        outputs = model.generate(**inputs, **gen_kwargs)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], outputs)
    ]
    decoded = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
    return [text.strip() for text in decoded]


def infer_batch_vllm(
    llm,
    processor: AutoProcessor,
    image_paths: Sequence[Path],
    system_prompt: Optional[str],
    prompts: Sequence[str],
    max_new_tokens: int,
    temperature: float,
    enable_thinking: bool,
    lora_request=None,
) -> List[str]:
    from vllm import SamplingParams
    from PIL import Image

    sampling_params = SamplingParams(
        temperature=temperature if temperature > 0 else 0.0,
        max_tokens=max_new_tokens,
    )

    vllm_inputs = []
    for image_path, prompt in zip(image_paths, prompts):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        })
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": enable_thinking},
        )
        image_inputs, _ = process_vision_info(messages)
        vllm_inputs.append({
            "prompt": text,
            "multi_modal_data": {"image": image_inputs},
        })

    outputs = llm.generate(vllm_inputs, sampling_params, lora_request=lora_request)
    return [o.outputs[0].text.strip() for o in outputs]


def infer_option_letter(text: str, options: Iterable[str]) -> Optional[str]:
    matches = re.findall(r"\b([A-D])\b", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].upper()

    normalized_text = text.strip().lower()
    for i, option in enumerate(list(options)):
        if normalized_text == str(option).strip().lower():
            return chr(ord("A") + i)
    return None


def build_extraction_prompt(question: str, options: Iterable[str], raw_generation: str) -> str:
    options_list = list(options)
    lines = [
        "Extract the final multiple-choice answer.",
        "Return ONLY one letter: A, B, C, or D.",
        "",
        f"Question: {question.strip()}",
    ]
    if options_list:
        lines.append("Options:")
        for i, option in enumerate(options_list):
            letter = chr(ord("A") + i)
            lines.append(f"{letter}. {option}")
    else:
        lines.append("The options are visual choices shown in the image.")
    lines.append("")
    lines.append("Previous model output:")
    lines.append(raw_generation.strip()[:1200])
    lines.append("")
    lines.append("Final letter only:")
    return "\n".join(lines)


def format_gold_answer(sample: Dict[str, Any]) -> Optional[str]:
    answer = sample.get("answer")
    options = sample.get("options", [])

    if isinstance(answer, int) and isinstance(options, list) and 0 <= answer < len(options):
        letter = chr(ord("A") + answer)
        return f"{letter}. {options[answer]}"

    if isinstance(answer, int) and not options and 0 <= answer < 4:
        return chr(ord("A") + answer)

    if answer is None:
        return None

    return str(answer)


def default_output_path(with_reasoning: bool) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "with_reasoning" if with_reasoning else "without_reasoning"
    return f"omnispatial_results_{ts}_{mode}.jsonl"


def resolve_output_path(output_arg: Optional[str], with_reasoning: bool) -> Path:
    log_dir = Path(DEFAULT_LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(output_arg).name if output_arg else default_output_path(with_reasoning)
    return log_dir / filename


def main() -> int:
    args = parse_args()

    data_json = Path(args.data_json)
    root_dir = Path(args.root_dir)

    data = load_data(str(data_json))
    requested_tasks = normalize_task_args(args.task_types)

    all_tasks = sorted({str(sample.get("task_type")) for sample in data if sample.get("task_type")})

    if requested_tasks is None:
        selected_tasks = set(all_tasks)
    else:
        unknown = sorted(set(requested_tasks) - set(all_tasks))
        if unknown:
            print(
                f"[ERROR] Unknown task type(s): {unknown}. Available: {all_tasks}",
                file=sys.stderr,
            )
            return 2
        selected_tasks = set(requested_tasks)

    filtered: List[Dict[str, Any]] = [
        s for s in data if str(s.get("task_type")) in selected_tasks
    ]

    if args.limit is not None:
        filtered = filtered[: args.limit]

    print(f"Loaded samples: {len(data)}")
    print(f"Available task types: {all_tasks}")
    print(f"Selected task types: {sorted(selected_tasks)}")
    print(f"Samples to run: {len(filtered)}")

    output_path = resolve_output_path(args.output, args.with_reasoning)

    if args.dry_run:
        missing = 0
        for sample in filtered:
            try:
                _ = resolve_image_path(root_dir, str(sample.get("task_type")), str(sample.get("id")))
            except Exception:
                missing += 1
        print(f"[DRY-RUN] Missing image mappings: {missing}/{len(filtered)}")
        return 0

    if args.with_reasoning and args.prompt_variant == "forced-bbox":
        mod = importlib.import_module("utils.system_prompts")
        system_prompt = getattr(mod, "SYSTEM_PROMPT_WITH_FORCED_BBOX")
        print("System prompt mode: FORCED BBOX — bboxes required every sample")
    elif args.with_reasoning and args.prompt_variant == "generic":
        mod = importlib.import_module("utils.system_prompts")
        system_prompt = getattr(mod, "SYSTEM_PROMPT_WITH_GENERIC_REASONING")
        print("System prompt mode: GENERIC — no bbox instructions, pure spatial reasoning test")
    else:
        system_prompt = load_system_prompt(with_reasoning=args.with_reasoning)
    extraction_system_prompt = None
    if system_prompt and args.prompt_variant == "default":
        print("System prompt mode: custom reasoning prompt from utils/system_prompts.py")
    else:
        print("System prompt mode: model default (no custom system prompt)")
    if args.with_reasoning:
        print("Reasoning mode: enabled")

    model_id = args.model_id or (
        DEFAULT_MODEL_ID_WITH_REASONING
        if args.with_reasoning
        else DEFAULT_MODEL_ID_WITHOUT_REASONING
    )

    print(f"Loading model: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)

    if args.use_vllm:
        from vllm import LLM
        vllm_kwargs = dict(
            model=model_id,
            dtype="bfloat16",
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_model_len=args.vllm_max_model_len,
            limit_mm_per_prompt={"image": 1},
        )
        lora_request = None
        if args.adapter_path:
            from vllm.lora.request import LoRARequest
            vllm_kwargs["enable_lora"] = True
            vllm_kwargs["max_lora_rank"] = 64
            lora_request = LoRARequest("adapter", 1, args.adapter_path)
            print(f"vLLM LoRA adapter: {args.adapter_path}")
        llm = LLM(**vllm_kwargs)
        model = None
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
        )
        if args.adapter_path:
            from peft import PeftModel
            print(f"Loading adapter: {args.adapter_path}")
            model = PeftModel.from_pretrained(model, args.adapter_path)
            model = model.merge_and_unload()
            print("Adapter merged.")
        llm = None
        lora_request = None

    succeeded = 0
    failed = 0

    with output_path.open("w", encoding="utf-8") as out_f:
        progress = tqdm(
            range(0, len(filtered), args.batch_size),
            desc="Running inference",
            unit="batch",
        )
        for batch_start in progress:
            batch_samples = filtered[batch_start : batch_start + args.batch_size]
            batch_results: List[Dict[str, Any]] = []
            batch_image_paths: List[Path] = []
            batch_prompts: List[str] = []
            batch_options: List[List[Any]] = []

            for offset, sample in enumerate(batch_samples):
                idx = batch_start + offset
                sample_id = str(sample.get("id"))
                task_type = str(sample.get("task_type"))
                question = str(sample.get("question", "")).strip()
                options = sample.get("options", [])

                result: Dict[str, Any] = {
                    "index": idx,
                    "id": sample_id,
                    "task_type": task_type,
                    "sub_task_type": str(sample.get("sub_task_type", "")),
                    "question": question,
                    "options": options,
                    "gold_answer": format_gold_answer(sample),
                    "status": "ok",
                    "prediction": None,
                    "raw_generation": None,
                    "image_path": None,
                    "error": None,
                }

                try:
                    image_path = resolve_image_path(root_dir, task_type, sample_id)
                    result["image_path"] = str(image_path)
                    prompt = build_user_text(
                        question=question,
                        options=options,
                        with_reasoning=args.with_reasoning,
                    )
                    batch_image_paths.append(image_path)
                    batch_prompts.append(prompt)
                    batch_options.append(list(options))
                except Exception as exc:  # noqa: BLE001
                    result["status"] = "error"
                    result["error"] = str(exc)
                    failed += 1

                batch_results.append(result)

            ready_indexes = [
                i for i, result in enumerate(batch_results) if result["status"] == "ok"
            ]
            if ready_indexes:
                try:
                    if args.use_vllm:
                        batch_generations = infer_batch_vllm(
                            llm=llm,
                            processor=processor,
                            image_paths=batch_image_paths,
                            system_prompt=system_prompt,
                            prompts=batch_prompts,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            enable_thinking=args.with_reasoning,
                            lora_request=lora_request,
                        )
                    else:
                        batch_generations = infer_batch(
                            model=model,
                            processor=processor,
                            image_paths=batch_image_paths,
                            system_prompt=system_prompt,
                            prompts=batch_prompts,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            enable_thinking=args.with_reasoning,
                            print_input_messages=args.print_input_messages,
                        )
                    for generation, result_idx, options in zip(
                        batch_generations, ready_indexes, batch_options
                    ):
                        result = batch_results[result_idx]
                        if args.print_generated_text:
                            print(
                                f"[GEN][{result['index']}][id={result['id']}] {generation}",
                                flush=True,
                            )
                        if args.with_reasoning:
                            result["raw_generation"] = generation
                            letter = infer_option_letter(generation, options)
                            if letter is not None:
                                option_idx = ord(letter) - ord("A")
                                if 0 <= option_idx < len(options):
                                    result["prediction"] = f"{letter}. {options[option_idx]}"
                                else:
                                    result["prediction"] = letter
                            else:
                                result["prediction"] = ""
                        else:
                            result["raw_generation"] = generation
                            result["prediction"] = generation
                        succeeded += 1
                except Exception as exc:  # noqa: BLE001
                    for result_idx in ready_indexes:
                        batch_results[result_idx]["status"] = "error"
                        batch_results[result_idx]["error"] = str(exc)
                        failed += 1

            for result in batch_results:
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()

            processed = min(batch_start + len(batch_samples), len(filtered))
            if processed % 10 == 0 or processed == len(filtered):
                print(
                    f"Progress: {processed}/{len(filtered)} | success={succeeded} | failed={failed}",
                    flush=True,
                )

    print(f"Done. Output: {output_path}")
    print(f"Final counts: success={succeeded}, failed={failed}, total={len(filtered)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
