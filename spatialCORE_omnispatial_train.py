#!/usr/bin/env python3
"""GRPO training entry point for SpatialCORE on the OmniSpatial dataset."""
import importlib
import inspect
import json
import math
import pathlib
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from datasets import Dataset, Features, Sequence as HFSequence, Value
from transformers import AutoProcessor, PreTrainedModel, Qwen3VLForConditionalGeneration
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import unwrap_model_for_generation
from trl.generation import VLLMGeneration


REPO_ROOT = Path(__file__).resolve().parent
repo_root_str = str(REPO_ROOT)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

from trl.trainer.grpo_config import GRPOConfig
from spatial_grpo import VLMGRPOTrainer, VLMBaseModule


DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Thinking"
DEFAULT_DATA_JSON = "data/OmniSpatial-train/data.json"
DEFAULT_ROOT_DIR = "data/OmniSpatial-train"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
# Used only when utils.system_prompts is missing SYSTEM_PROMPT_WITH_REASONING.
DEFAULT_REASONING_SYSTEM_PROMPT = (
    "You are a spatial-reasoning assistant for visual multiple-choice questions.\n\n"
    "You are given one image and a question about the image. Answer options may be provided in the text, or they may appear "
    "inside the image itself. If they are not fully provided in the text, identify them from the image. Use the image as the "
    "primary source of truth. Do not hallucinate objects, text, or relations.\n\n"
    "At the beginning of the thinking content, output bounding boxes for visible objects or regions that the question or "
    "answer options refer to and that you could reasonably localize—short noun phrases a detector could target, keeping "
    "useful descriptive modifiers when needed (e.g. \"cyclist\", \"blue truck\", \"man in red\").\n"
    "Each box must be on its own line in this exact JSON format:\n"
    '{"bbox_2d": [x_min, y_min, x_max, y_max], "label": "descriptive noun phrase"}\n'
    "Coordinates must be integers from 0 to 1000 (x: left→right, y: top→bottom).\n"
    "Only include boxes for entities that are actually visible. Do not invent boxes for things that are not in the image.\n"
    "If there is nothing in the question or options that corresponds to a visible, boxable entity, output no bbox lines and "
    "proceed with reasoning only.\n\n"
    "After any bbox lines (or immediately, if there are none), continue reasoning using visible evidence such as position, "
    "distance, depth, ordering, overlap, perspective, and text in the image. Refer to grounded objects when you have them.\n\n"
    "After </think>, output exactly one final answer as a single capital letter: A, B, C, or D.\n"
    "Do not output punctuation, spaces, option text, or anything after that answer letter.\n\n"
    "Do not refuse. If uncertain, choose the most plausible answer based on the image."
)
DEFAULT_NO_REASONING_SYSTEM_PROMPT = (
    "You are a spatial-reasoning assistant.\n"
    "Use only visible evidence from the image.\n"
    "If the task is multiple choice, output exactly one option letter only: A, B, C, or D."
)


script_args = None
# Curriculum phase: 1 = cold start (format only), 2 = grounding active, 3 = full reward.
_current_training_phase = 3
_total_training_steps = None
_reward_call_count = 0


def get_training_phase(global_step: int) -> int:
    """Return the current curriculum phase based on the optimizer step.

    Phase 1: format reward + scaled accuracy + optional bbox-format bonus.
    Phase 2: format reward + full accuracy + grounding reward (no uncertainty weighting).
    Phase 3: phase 2 plus uncertainty weighting on grounding.
    """
    if not script_args.cold_start or _total_training_steps is None:
        return 3
    p1_end = int(_total_training_steps * script_args.cold_start_phase1_ratio)
    p2_end = int(_total_training_steps * script_args.cold_start_phase2_ratio)
    if global_step < p1_end:
        return 1
    elif global_step < p2_end:
        return 2
    return 3


_original_get_train_sampler = VLMGRPOTrainer._get_train_sampler


def _compat_get_train_sampler(self, dataset=None):
    return _original_get_train_sampler(self)


VLMGRPOTrainer._get_train_sampler = _compat_get_train_sampler


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
    """Build the user turn (question + options). Format rules live in the system prompt."""
    _ = with_reasoning
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
        fallback = DEFAULT_REASONING_SYSTEM_PROMPT
    else:
        candidates = (
            "SYSTEM_PROMPT_NO_REASONING",
            "NO_REASONING_SYSTEM_PROMPT",
            "SYSTEM_PROMPT_WITHOUT_REASONING",
            "SYSTEM_PROMPT",
        )
        fallback = DEFAULT_NO_REASONING_SYSTEM_PROMPT

    prompt = _first_existing_attr(mod, candidates)
    return prompt if prompt is not None else fallback


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


def extract_tag(text: str, tag: str) -> Optional[str]:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    value = match.group(1).strip()
    return value or None


def extract_thinking_content(text: str) -> Optional[str]:
    """Return the reasoning body (text inside <think>...</think> or before </think>).

    Handles both full wrapped spans and the common case where the prompt already
    opens <think> and the completion only contains the closing tag.
    """
    inner = extract_tag(text, "think")
    if inner is not None:
        return inner if inner else None
    m = re.search(r"(.*?)</think>", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        body = m.group(1).strip()
        return body if body else None
    return None


def normalize_answer(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).rstrip(".").lower()


def normalize_bbox_label(text: str) -> str:
    text = str(text or "").lower().strip()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def infer_option_letter(text: str, options: Iterable[str]) -> Optional[str]:
    matches = re.findall(r"\b([A-D])\b", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].upper()

    normalized_text = text.strip().lower()
    for i, option in enumerate(list(options)):
        if normalized_text == str(option).strip().lower():
            return chr(ord("A") + i)
    return None


def normalize_prediction_for_scoring(raw_text: str, options: Iterable[str]) -> str:
    if re.search(r"</think>", raw_text, flags=re.IGNORECASE):
        match = re.search(r"</think>\s*([A-D])\s*$", raw_text, flags=re.IGNORECASE)
        candidate = match.group(1) if match else re.split(r"</think>", raw_text, flags=re.IGNORECASE, maxsplit=1)[-1]
    else:
        candidate = raw_text
    letter = infer_option_letter(candidate, options)
    if letter is not None:
        return letter.lower()
    return normalize_answer(candidate)


def normalize_target_for_scoring(answer: str, options: Iterable[str]) -> str:
    if answer is None:
        return ""
    letter = infer_option_letter(answer, options)
    if letter is not None:
        return letter.lower()
    return normalize_answer(answer)


def accuracy_reward(completions, **kwargs):
    global _reward_call_count
    answers = kwargs.get("answer")
    options_batch = kwargs.get("options")
    gt_bboxes_batch = kwargs.get("gt_bboxes")
    rewards = []

    # Periodic diagnostic: log the fraction of unique completions in this batch.
    # A low rate suggests duplicate samples from the generation backend.
    _reward_call_count += 1
    if _reward_call_count % 20 == 1:
        contents = [c[0]["content"] for c in completions]
        unique_rate = len(set(contents)) / max(len(contents), 1)
        print(f"[unique_completion_rate] call={_reward_call_count} "
              f"n={len(contents)} unique={len(set(contents))} rate={unique_rate:.3f}")

    weight = float(script_args.accuracy_reward_weight)
    if _current_training_phase == 1:
        weight *= script_args.cold_start_accuracy_scale

    coverage_floor = float(script_args.grounding_coverage_floor)
    coverage_iou_thresh = float(script_args.grounding_coverage_iou_threshold)
    coverage_gate_active = coverage_floor < 1.0 and gt_bboxes_batch is not None

    for idx, (completion, answer, options) in enumerate(zip(completions, answers, options_batch)):
        content = completion[0]["content"]
        pred = normalize_prediction_for_scoring(content, options)
        gold = normalize_target_for_scoring(answer, options)
        reward = 1.0 if pred and gold and pred == gold else 0.0

        if coverage_gate_active and reward > 0.0:
            gt_raw = gt_bboxes_batch[idx]
            gt_list = json.loads(gt_raw) if gt_raw else []
            if gt_list:
                think_content = extract_thinking_content(content)
                pred_bboxes = extract_bboxes(think_content) if think_content else []
                valid_preds = [b for b in pred_bboxes if is_valid_bbox(b["bbox_2d"])]
                coverage = compute_grounding_coverage(valid_preds, gt_list, coverage_iou_thresh)
                reward *= (coverage_floor + (1.0 - coverage_floor) * coverage)

        rewards.append(reward * weight)

    return rewards


def format_reward(completions, **kwargs):
    # In Qwen3 thinking mode the prompt already opens <think>, so the completion is the
    # closing tag plus the answer letter. The base format score requires those tags and
    # a single A/B/C/D after </think>; bbox quality adjustments are applied below.
    if script_args.with_reasoning:
        basic_pattern = r"^\s*.*?</think>\s*[A-D]\s*$"
    else:
        basic_pattern = r"^\s*[A-D]\s*$"

    gt_bboxes_batch = kwargs.get("gt_bboxes")
    rewards = []
    for idx, completion in enumerate(completions):
        content = completion[0]["content"].strip()
        # Truncated completions (no </think>) get a neutral 0 so they neither shrink
        # group variance nor drag down the group mean when truncation is common.
        if script_args.with_reasoning and not re.search(r"</think>", content, re.IGNORECASE):
            rewards.append(0.0)
            continue
        if re.fullmatch(basic_pattern, content, flags=re.IGNORECASE | re.DOTALL) is None:
            rewards.append(0.0)
            continue

        if script_args.with_reasoning:
            if len(re.findall(r"</think>", content, flags=re.IGNORECASE)) != 1:
                rewards.append(0.0)
                continue

        if script_args.with_reasoning:
            tail = re.split(r"</think>", content, flags=re.IGNORECASE, maxsplit=1)[-1].strip()
            if re.fullmatch(r"[A-D]", tail, flags=re.IGNORECASE) is None:
                rewards.append(0.0)
                continue
        else:
            if re.fullmatch(r"[A-D]", content, flags=re.IGNORECASE) is None:
                rewards.append(0.0)
                continue

        score = float(script_args.format_reward_weight)

        if script_args.with_reasoning:
            think_part, tail_part = re.split(
                r"</think>", content, flags=re.IGNORECASE, maxsplit=1
            )

            # Structural violation: bbox JSON must not appear after </think>.
            if extract_bboxes(tail_part):
                score -= float(script_args.format_reward_weight)

            # Penalise repeated labels inside thinking (one bbox per object).
            think_bboxes = extract_bboxes(think_part)
            normalized_labels = [
                normalize_bbox_label(b.get("label", "")) for b in think_bboxes
            ]
            normalized_labels = [label for label in normalized_labels if label]
            label_counts = Counter(normalized_labels)
            extra_duplicates = sum(max(0, count - 1) for count in label_counts.values())
            if extra_duplicates > 0:
                score -= min(0.05 * extra_duplicates, 0.15)

            # Penalise identical coordinates across predictions (same box copied multiple times).
            coord_tuples = [tuple(b["bbox_2d"]) for b in think_bboxes if is_valid_bbox(b["bbox_2d"])]
            duplicate_coords = len(coord_tuples) - len(set(coord_tuples))
            if duplicate_coords > 0:
                score -= 0.1 * duplicate_coords

            # Text-anchor signal during Phase 1: penalise bbox labels whose tokens do not
            # appear in the question/options text. The grounding reward is disabled in
            # Phase 1, so this guards against off-topic boxes during the cold start.
            if _current_training_phase == 1 and script_args.cold_start and think_bboxes:
                questions_batch_fa = kwargs.get("question")
                options_batch_fa = kwargs.get("options")
                question_text_fa = questions_batch_fa[idx] if questions_batch_fa else ""
                if isinstance(question_text_fa, list):
                    question_text_fa = " ".join(str(q) for q in question_text_fa)
                options_fa = options_batch_fa[idx] if options_batch_fa else []
                anchor_text = (question_text_fa + " " + " ".join(str(o) for o in options_fa)).lower()
                anchor_tokens = set(re.findall(r"\w+", anchor_text))
                if anchor_tokens:
                    off_topic = sum(
                        1 for b in think_bboxes
                        if not set(re.findall(r"\w+", b.get("label", "").lower())).intersection(anchor_tokens)
                    )
                    if off_topic > 0:
                        score -= min(0.05 * off_topic, 0.15)

        # Optional Phase-1 bonus for emitting any valid bbox. Off by default.
        if _current_training_phase == 1 and script_args.cold_start_bbox_bonus > 0:
            think_content = extract_thinking_content(content)
            if think_content:
                bboxes = extract_bboxes(think_content)
                valid = [b for b in bboxes if is_valid_bbox(b["bbox_2d"])]
                if valid:
                    score += script_args.cold_start_bbox_bonus

        # bbox_attempt_bonus: small reward when ground-truth boxes exist AND the model
        # produced at least one valid bbox whose label overlaps the question/option text.
        # The token-overlap gate prevents rewarding off-topic predictions.
        if script_args.bbox_attempt_bonus > 0 and gt_bboxes_batch is not None:
            gt_list = json.loads(gt_bboxes_batch[idx]) if gt_bboxes_batch[idx] else []
            if gt_list:
                think_content = extract_thinking_content(content)
                if think_content:
                    bboxes = extract_bboxes(think_content)
                    valid = [b for b in bboxes if is_valid_bbox(b["bbox_2d"])]
                    if valid:
                        questions_batch_ba = kwargs.get("question")
                        options_batch_ba = kwargs.get("options")
                        q_text_ba = questions_batch_ba[idx] if questions_batch_ba else ""
                        if isinstance(q_text_ba, list):
                            q_text_ba = " ".join(str(q) for q in q_text_ba)
                        opts_ba = options_batch_ba[idx] if options_batch_ba else []
                        anchor_ba = (q_text_ba + " " + " ".join(str(o) for o in opts_ba)).lower()
                        anchor_tokens_ba = set(re.findall(r"\w+", anchor_ba))
                        on_topic = any(
                            set(re.findall(r"\w+", b.get("label", "").lower())).intersection(anchor_tokens_ba)
                            for b in valid
                        )
                        if on_topic or not anchor_tokens_ba:
                            score += script_args.bbox_attempt_bonus

        rewards.append(score)

    return rewards


BBOX_PATTERN = re.compile(
    r'\{\s*"bbox_2d"\s*:\s*\[(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*)\]'
    r'\s*,\s*"label"\s*:\s*"([^"]+)"\s*\}',
    re.IGNORECASE,
)


def extract_bboxes(text: str) -> list[dict]:
    results = []
    for match in BBOX_PATTERN.finditer(text):
        coords = [int(c.strip()) for c in match.group(1).split(",")]
        label = match.group(2).strip()
        results.append({"bbox_2d": coords, "label": label})
    return results


def is_valid_bbox(coords: list[int]) -> bool:
    if len(coords) != 4:
        return False
    x_min, y_min, x_max, y_max = coords
    if not all(0 <= c <= 1000 for c in coords):
        return False
    if x_max <= x_min or y_max <= y_min:
        return False
    return True


def compute_iou(box_a: list[int], box_b: list[int]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def centered_iou_score(box_a: list[int], box_b: list[int]) -> float:
    """IoU shifted to [0, 0.75]: zero below IoU=0.25, positive above.

    Boxes with low overlap (IoU < 0.25) are treated as neutral rather than
    penalised, so the model is not punished for imprecise early attempts.
    """
    iou = compute_iou(box_a, box_b)
    return max(0.0, iou - 0.25)


def compute_grounding_coverage(pred_bboxes: list[dict], gt_list: list[dict], iou_threshold: float = 0.1) -> float:
    """Fraction of ground-truth bboxes covered by at least one predicted bbox.

    Returns 1.0 when there are no ground-truth boxes (nothing to ground).
    """
    if not gt_list:
        return 1.0
    covered = 0
    for gt in gt_list:
        gt_box = gt.get("bbox_2d", [])
        if not is_valid_bbox(gt_box):
            covered += 1  # malformed annotation — count as covered to avoid spurious penalty
            continue
        for pred in pred_bboxes:
            pred_box = pred.get("bbox_2d", [])
            if is_valid_bbox(pred_box) and compute_iou(pred_box, gt_box) >= iou_threshold:
                covered += 1
                break
    return covered / len(gt_list)


def _tokenize_label(text: str) -> list[str]:
    text = str(text or "").lower()
    return re.findall(r"[a-z0-9]+", text)


def cosine_label_similarity(label_a: str, label_b: str) -> float:
    toks_a = _tokenize_label(label_a)
    toks_b = _tokenize_label(label_b)
    if not toks_a or not toks_b:
        return 0.0
    cnt_a = Counter(toks_a)
    cnt_b = Counter(toks_b)
    common = set(cnt_a.keys()) & set(cnt_b.keys())
    dot = float(sum(cnt_a[t] * cnt_b[t] for t in common))
    norm_a = math.sqrt(sum(v * v for v in cnt_a.values()))
    norm_b = math.sqrt(sum(v * v for v in cnt_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def label_in_text_score(pred_label: str, question: str, options: list) -> float:
    """Fraction of tokens in pred_label that appear in the question + options.

    Returns 1.0 if every token of the predicted label is found in the question
    or options text, and 0.0 if none are. Used as a text-anchor signal so the
    model does not invent labels from visible text in the image.
    """
    text = (question + " " + " ".join(str(o) for o in options if o)).lower()
    tokens = _tokenize_label(pred_label)
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in text) / len(tokens)


def hungarian_assign_pairs(score_matrix: list[list[float]]) -> list[tuple[int, int]]:
    """Return a one-to-one assignment that maximises the total score.

    Uses scipy's Hungarian algorithm when available, with a greedy fallback.
    """
    if not score_matrix or not score_matrix[0]:
        return []
    try:
        from scipy.optimize import linear_sum_assignment
    except Exception:
        used_rows = set()
        used_cols = set()
        pairs: list[tuple[int, int]] = []
        flat = [
            (score_matrix[r][c], r, c)
            for r in range(len(score_matrix))
            for c in range(len(score_matrix[0]))
        ]
        flat.sort(reverse=True, key=lambda x: x[0])
        for s, r, c in flat:
            if r in used_rows or c in used_cols:
                continue
            used_rows.add(r)
            used_cols.add(c)
            pairs.append((r, c))
            if len(pairs) == min(len(score_matrix), len(score_matrix[0])):
                break
        return pairs

    # Hungarian minimizes cost; convert max-score problem to min-cost.
    max_score = max(max(row) for row in score_matrix)
    cost = [[max_score - s for s in row] for row in score_matrix]
    row_ind, col_ind = linear_sum_assignment(cost)
    return list(zip(row_ind.tolist(), col_ind.tolist()))


def best_iou_for_pred(pred_bbox: list[int], gt_bboxes: list[dict]) -> tuple[float, float]:
    best_iou = 0.0
    best_conf = 0.0
    for gt in gt_bboxes:
        iou = compute_iou(pred_bbox, gt["bbox_2d"])
        if iou > best_iou:
            best_iou = iou
            best_conf = gt.get("confidence", 1.0)
    return best_iou, best_conf


def compute_soft_recall(
    valid_preds: list[dict],
    gt_list: list[dict],
    *,
    iou_weight: float,
    label_weight: float,
) -> float:
    total_conf = sum(gt.get("confidence", 1.0) for gt in gt_list)
    if total_conf == 0:
        return 0.0

    weighted_sum = 0.0
    for gt in gt_list:
        best_score = float("-inf")
        for pred in valid_preds:
            centered_iou = centered_iou_score(pred["bbox_2d"], gt["bbox_2d"])
            label_sim = cosine_label_similarity(pred.get("label", ""), gt.get("label", ""))
            pair_score = iou_weight * centered_iou + label_weight * label_sim
            best_score = max(best_score, pair_score)
        if best_score == float("-inf"):
            best_score = 0.0
        weighted_sum += gt.get("confidence", 1.0) * best_score

    return weighted_sum / total_conf


def f_beta_score(precision: float, recall: float, beta: float) -> float:
    beta_sq = beta * beta
    numerator = (1 + beta_sq) * precision * recall
    denominator = beta_sq * precision + recall
    if denominator < 1e-9:
        return 0.0
    return numerator / denominator


def find_bbox_coordinate_positions(token_ids: list[int], tokenizer) -> list[list[list[int]]]:
    """Locate token positions of bbox coordinate digits inside a completion.

    For every bbox in the decoded text, returns a 4-element list of
    coordinate groups; each group is a list of token-position indices that
    together encode that coordinate's digits.
    """
    text_tokens = [tokenizer.decode([tid]) for tid in token_ids]

    full_text = tokenizer.decode(token_ids)
    bbox_iter = re.finditer(
        r'\{\s*"bbox_2d"\s*:\s*\[(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*)\]',
        full_text,
    )

    all_bboxes_positions = []
    for match in bbox_iter:
        coord_str = match.group(1)
        coord_start = match.start(1)

        coord_texts = coord_str.split(",")
        coord_positions = []
        cursor = coord_start
        for ct in coord_texts:
            stripped = ct.strip()
            offset_in_group = ct.index(stripped)
            abs_start = cursor + offset_in_group
            abs_end = abs_start + len(stripped)

            positions = char_span_to_token_indices(
                token_ids, tokenizer, abs_start, abs_end
            )
            coord_positions.append(positions)
            cursor += len(ct) + 1  # +1 for comma

        if len(coord_positions) == 4 and all(coord_positions):
            all_bboxes_positions.append(coord_positions)

    return all_bboxes_positions


def char_span_to_token_indices(
    token_ids: list[int], tokenizer, char_start: int, char_end: int
) -> list[int]:
    """Map a character span in the decoded text to token indices."""
    indices = []
    current_char = 0
    for i, tid in enumerate(token_ids):
        token_text = tokenizer.decode([tid])
        token_start = current_char
        token_end = current_char + len(token_text)
        if token_end > char_start and token_start < char_end:
            indices.append(i)
        current_char = token_end
        if current_char >= char_end:
            break
    return indices


def compute_bbox_entropies(
    logits: torch.Tensor,
    completion_ids: torch.Tensor,
    tokenizer,
) -> list[list[float]]:
    """Compute the normalized entropy of each bbox's coordinates per sample.

    Qwen tokenizes numbers digit-by-digit, so each coordinate digit has exactly
    ten valid values (0-9). We normalise by log(10) so each per-token entropy
    falls in [0, 1].

    Args:
        logits: (B, L, V) logits over completion tokens.
        completion_ids: (B, L) completion token ids.
        tokenizer: the tokenizer used to decode token positions.

    Returns:
        A list of length B. Each element is a list of per-bbox uncertainty
        values U(b), one per bbox in that sample. Empty when no bbox is found.
    """
    log_v = math.log(10)
    batch_entropies = []

    for b in range(completion_ids.size(0)):
        sample_ids = completion_ids[b].tolist()
        bbox_positions = find_bbox_coordinate_positions(sample_ids, tokenizer)

        if not bbox_positions:
            batch_entropies.append([])
            continue

        sample_bbox_entropies = []
        for coord_groups in bbox_positions:
            coord_entropies = []
            for positions in coord_groups:
                token_entropies = []
                for pos in positions:
                    if pos >= logits.size(1):
                        continue
                    probs = torch.softmax(logits[b, pos], dim=-1)
                    h = -(probs * torch.log(probs + 1e-12)).sum().item()
                    h_norm = h / log_v
                    token_entropies.append(h_norm)
                if token_entropies:
                    coord_entropies.append(sum(token_entropies) / len(token_entropies))

            if len(coord_entropies) == 4:
                u_b = sum(coord_entropies) / 4.0
                sample_bbox_entropies.append(u_b)

        batch_entropies.append(sample_bbox_entropies)

    return batch_entropies


def compute_uncertainty_weights(
    bbox_entropies_json: str | None, num_bboxes: int, beta: float
) -> list[float]:
    """Compute the per-bbox confidence weight omega_j = beta + (1 - beta) * (1 - U_tilde_j).

    Returns uniform weights of 1.0 when entropy data is missing.
    """
    if not bbox_entropies_json:
        return [1.0] * num_bboxes
    entropies = json.loads(bbox_entropies_json)
    beta = max(0.0, min(1.0, float(beta)))
    weights = []
    for j in range(num_bboxes):
        if j < len(entropies):
            u_tilde = max(0.0, min(1.0, float(entropies[j])))
            confidence = 1.0 - u_tilde
            weights.append(beta + (1.0 - beta) * confidence)
        else:
            weights.append(1.0)
    return weights


def grounding_reward(completions, **kwargs):
    if _current_training_phase == 1:
        return [0.0] * len(completions)

    gt_bboxes_batch = kwargs.get("gt_bboxes")
    bbox_entropies_batch = kwargs.get("bbox_entropies")
    answers = kwargs.get("answer")
    options_batch = kwargs.get("options")
    questions_batch = kwargs.get("question")
    use_entropy = _current_training_phase >= 3 and script_args.use_uncertainty_weighting
    gamma = float(script_args.spatial_answer_gate)
    rewards = []

    for idx, completion in enumerate(completions):
        content = completion[0]["content"]

        # Answer-correctness gate: γ if the answer is wrong, 1.0 if correct.
        answer = answers[idx] if answers else None
        options = options_batch[idx] if options_batch else []
        question_text = questions_batch[idx] if questions_batch else ""
        if isinstance(question_text, list):
            question_text = " ".join(str(q) for q in question_text)
        pred_ans = normalize_prediction_for_scoring(content, options)
        gold_ans = normalize_target_for_scoring(answer, options)
        answer_correct = bool(pred_ans and gold_ans and pred_ans == gold_ans)
        gate = 1.0 if answer_correct else gamma

        think_content = extract_thinking_content(content)
        if think_content is None:
            # Truncated rollout: return a neutral 0 to avoid collapsing group variance.
            rewards.append(0.0)
            continue

        pred_bboxes = extract_bboxes(think_content)
        valid_preds = [b for b in pred_bboxes if is_valid_bbox(b["bbox_2d"])]

        gt_list = json.loads(gt_bboxes_batch[idx]) if gt_bboxes_batch else []
        if not valid_preds:
            if gt_list:
                # Ground-truth boxes exist but the model emitted none — penalise once per missed entity.
                penalty = script_args.over_prediction_penalty * sum(gt.get("confidence", 1.0) for gt in gt_list)
                rewards.append(max(-1.0, -penalty) * script_args.grounding_reward_weight)
            else:
                # No ground truth and no prediction: correct behaviour, no signal.
                rewards.append(0.0)
            continue

        if not gt_list:
            # No ground-truth boxes for this sample: return a neutral 0 rather than
            # penalising. Penalising here would create an anti-grounding signal that
            # leaks through the shared LoRA into samples that do need grounding.
            rewards.append(0.0)
            continue

        if use_entropy:
            entropy_json = bbox_entropies_batch[idx] if bbox_entropies_batch else None
            omegas = compute_uncertainty_weights(
                entropy_json, len(valid_preds), script_args.uncertainty_beta
            )
        else:
            omegas = [1.0] * len(valid_preds)

        # One-to-one matching between predicted and ground-truth boxes via Hungarian
        # assignment. Pair score = weighted IoU + weighted label cosine similarity.
        iou_w = float(script_args.grounding_iou_weight)
        label_w = float(script_args.grounding_label_weight)
        norm = iou_w + label_w
        if norm <= 0:
            iou_w, label_w = 1.0, 0.0
            norm = 1.0
        iou_w /= norm
        label_w /= norm

        score_matrix: list[list[float]] = []
        for pred in valid_preds:
            row: list[float] = []
            for gt in gt_list:
                centered_iou = centered_iou_score(pred["bbox_2d"], gt["bbox_2d"])
                label_sim = cosine_label_similarity(pred.get("label", ""), gt.get("label", ""))
                pair_score = (iou_w * centered_iou + label_w * label_sim) * gt.get("confidence", 1.0)
                row.append(pair_score)
            score_matrix.append(row)

        assigned = hungarian_assign_pairs(score_matrix)
        matched_score_sum = 0.0
        for pred_idx, gt_idx in assigned:
            pair = score_matrix[pred_idx][gt_idx]
            matched_score_sum += omegas[pred_idx] * pair

        # Normalize precision by the matched-pair count, not total predictions.
        precision = matched_score_sum / max(len(assigned), 1)
        recall = compute_soft_recall(
            valid_preds,
            gt_list,
            iou_weight=iou_w,
            label_weight=label_w,
        )
        precision = max(-1.0, min(1.0, precision))
        recall = max(-1.0, min(1.0, recall))

        if script_args.grounding_use_recall:
            if precision >= 0.0 and recall >= 0.0:
                score = f_beta_score(precision, recall, script_args.grounding_fbeta)
            else:
                # F_β is undefined when either input is negative; fall back to
                # precision so the negative signal is preserved.
                score = precision
        else:
            score = precision

        # Penalise predictions beyond the ground-truth count: each unmatched bbox
        # has no entity to match against and is therefore over-prediction.
        n_unmatched = len(valid_preds) - len(assigned)
        if n_unmatched > 0:
            score -= script_args.over_prediction_penalty * n_unmatched

        # Text-anchor penalty: down-weight predictions whose label tokens do not
        # appear in the question or options text.
        if script_args.grounding_text_anchor:
            for pred in valid_preds:
                ta = label_in_text_score(pred.get("label", ""), question_text, options)
                score -= script_args.grounding_text_anchor_weight * (1.0 - ta)

        # Gate only positive scores: a wrong answer scales the reward down by γ.
        # Negative scores (bad grounding) are never softened.
        gated_score = gate * score if score > 0 else score
        rewards.append(max(-1.0, gated_score) * script_args.grounding_reward_weight)

    return rewards


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "grounding": grounding_reward,
}


class Qwen3VLModule(VLMBaseModule):
    def __init__(self, enable_thinking: bool = True):
        super().__init__()
        self.enable_thinking = enable_thinking

    def get_vlm_key(self):
        return "qwen3"

    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        model_cls = Qwen3VLForConditionalGeneration

        class Qwen3VLCompatModel(model_cls):
            @classmethod
            def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
                # Qwen3VL does not accept use_cache via __init__; pop it and apply to
                # the config after loading so gradient checkpointing can disable it.
                use_cache = kwargs.pop("use_cache", None)
                print(
                    "[Qwen3VLCompatModel] from_pretrained kwargs: "
                    f"{sorted(kwargs.keys())}"
                )
                model = model_cls.from_pretrained(
                    pretrained_model_name_or_path, *args, **kwargs
                )
                if use_cache is not None:
                    model.config.use_cache = use_cache
                return model

        return Qwen3VLCompatModel

    def post_model_init(self, model, processing_class):
        if model is not None:
            for candidate in (
                model,
                getattr(model, "base_model", None),
                getattr(getattr(model, "base_model", None), "model", None),
            ):
                if candidate is not None and not hasattr(candidate, "warnings_issued"):
                    candidate.warnings_issued = {}

        original_apply_chat_template = processing_class.apply_chat_template

        @wraps(original_apply_chat_template)
        def patched_apply_chat_template(*args, **kwargs):
            chat_template_kwargs = dict(kwargs.pop("chat_template_kwargs", {}) or {})
            chat_template_kwargs.setdefault("enable_thinking", self.enable_thinking)
            return original_apply_chat_template(
                *args,
                **kwargs,
                chat_template_kwargs=chat_template_kwargs,
            )

        processing_class.apply_chat_template = patched_apply_chat_template
        if getattr(processing_class, "tokenizer", None) is not None:
            processing_class.tokenizer.padding_side = "left"

    def is_embeds_input(self):
        return False

    def get_processing_class(self):
        return AutoProcessor

    def get_vision_modules_keywords(self):
        return ["visual"]

    def get_custom_multimodal_keywords(self):
        return ["pixel_values", "image_grid_thw"]

    def get_non_generate_params(self):
        return []

    def get_custom_processing_keywords(self):
        return [("image_processor", "max_pixels"), ("image_processor", "min_pixels")]

    def prepare_prompt(self, processing_class, inputs: dict[str, Any]):
        return [maybe_apply_chat_template(example, processing_class)["prompt"] for example in inputs]

    def prepare_model_inputs(
        self,
        processing_class,
        prompts_text,
        images,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        add_special_tokens=False,
    ):
        processing_class.tokenizer.padding_side = padding_side
        return processing_class(
            text=prompts_text,
            images=images if images else None,
            return_tensors=return_tensors,
            padding=padding,
            add_special_tokens=add_special_tokens,
        )


class Qwen3GRPOTrainer(VLMGRPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vllm_gen = None
        self._last_vllm_step = -1

    def _init_vllm(self):
        """Initialise the colocated vLLM generation backend.

        Must run after _patch_generation_config so the generation config has
        its final temperature / top_p / top_k before being passed to vLLM.
        """
        import os
        # Reuse the existing torchrun process group instead of letting vLLM
        # initialise its own (which can deadlock under colocate mode).
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        os.environ.setdefault("VLLM_USE_V1", "0")

        args = self.args
        gc = self.generation_config
        self._vllm_gen = VLLMGeneration(
            model=self.model,
            accelerator=self.accelerator,
            is_fsdp_enabled=getattr(self, "is_fsdp_enabled", False),
            processing_class=self.processing_class,
            mode=getattr(args, "vllm_mode", "colocate"),
            tensor_parallel_size=getattr(args, "vllm_tensor_parallel_size", 1),
            gpu_memory_utilization=getattr(args, "vllm_gpu_memory_utilization", 0.7),
            max_model_length=getattr(args, "vllm_max_model_length", None),
            max_num_seqs=args.per_device_train_batch_size * self.num_generations,
            enable_sleep_mode=False,
            temperature=float(gc.temperature) if gc.temperature is not None else 1.0,
            top_p=float(gc.top_p) if gc.top_p is not None else 1.0,
            top_k=int(gc.top_k) if gc.top_k is not None else 0,
            max_completion_length=self.max_completion_length,
            logprobs=0,
        )
        print(
            f"[vLLM] Initialized colocate backend — "
            f"gpu_memory_utilization={getattr(args, 'vllm_gpu_memory_utilization', 0.7)}, "
            f"max_num_seqs={args.per_device_train_batch_size * self.num_generations}"
        )

    def _patch_generation_config(self):
        """Merge the model's pretrained generation config into the trainer's config
        so model-specific settings (top_k, top_p, eos_token_id, ...) are not lost."""
        from transformers import GenerationConfig as GC
        try:
            model_gc = GC.from_pretrained(
                self.args._name_or_path
                if hasattr(self.args, "_name_or_path")
                else self.model.config._name_or_path
            )
        except Exception:
            model_gc = getattr(self.model, "generation_config", None)
        if model_gc is None:
            return
        for attr in ("top_k", "top_p", "eos_token_id", "bos_token_id", "repetition_penalty"):
            trainer_val = getattr(self.generation_config, attr, None)
            model_val = getattr(model_gc, attr, None)
            if trainer_val is None and model_val is not None:
                setattr(self.generation_config, attr, model_val)

    def _generate_and_score_completions(self, inputs: dict[str, Any], model) -> dict[str, Any]:
        global _current_training_phase, _total_training_steps
        if _total_training_steps is None and hasattr(self, "state") and self.state.max_steps > 0:
            _total_training_steps = self.state.max_steps
        step = self.state.global_step if hasattr(self, "state") else 0
        prev_phase = _current_training_phase
        _current_training_phase = get_training_phase(step)
        if _current_training_phase != prev_phase:
            print(
                f"[Curriculum] Step {step}: Phase {prev_phase} -> Phase {_current_training_phase} "
                f"({'format+acc' if _current_training_phase == 1 else 'grounding' if _current_training_phase == 2 else 'full+uncertainty'})"
            )

        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]

        from qwen_vl_utils import process_vision_info

        all_texts = []
        all_images = []
        for example in inputs:
            cleaned_msg = []
            for message in example["prompt"]:
                cleaned_message = {"role": message["role"], "content": []}
                for item in message.get("content", []):
                    item_type = item.get("type")
                    if item_type == "image" and item.get("image") is not None:
                        cleaned_message["content"].append(
                            {"type": "image", "image": item["image"]}
                        )
                    elif item_type == "text" and item.get("text") is not None:
                        cleaned_message["content"].append(
                            {"type": "text", "text": item["text"]}
                        )
                if cleaned_message["content"]:
                    cleaned_msg.append(cleaned_message)

            all_texts.append(
                self.processing_class.apply_chat_template(
                    cleaned_msg,
                    tokenize=False,
                    add_generation_prompt=True,
                    chat_template_kwargs={"enable_thinking": self.vlm_module.enable_thinking},
                )
            )
            images, _ = process_vision_info(cleaned_msg)
            all_images.extend(images if images else [])

        all_images = all_images if all_images else None

        self.processing_class.tokenizer.padding_side = "left"
        prompt_inputs = self.processing_class(
            text=all_texts,
            images=all_images,
            padding=True,
            return_tensors="pt",
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_inputs = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in prompt_inputs.items()
        }
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if step == 0 and self.accelerator.is_main_process:
            print(f"[DEBUG-PROMPT] prompt_inputs keys: {list(prompt_inputs.keys())}")
            print(f"[DEBUG-PROMPT] input_ids shape: {prompt_ids.shape}")
            if "pixel_values" in prompt_inputs:
                print(f"[DEBUG-PROMPT] pixel_values shape: {prompt_inputs['pixel_values'].shape}")
            else:
                print("[DEBUG-PROMPT] WARNING: pixel_values NOT in prompt_inputs!")
            gc = self.generation_config
            print(f"[DEBUG-GEN-CONFIG] top_k={gc.top_k}, top_p={gc.top_p}, temp={gc.temperature}, "
                  f"eos_token_id={gc.eos_token_id}, do_sample={gc.do_sample}, "
                  f"max_new_tokens={gc.max_new_tokens}")

        if self._vllm_gen is not None:
            # vLLM colocate generation path.
            # Sync LoRA weights into the vLLM model copy once per optimizer step.
            if step != self._last_vllm_step:
                self._vllm_gen.sync_weights()
                self._last_vllm_step = step

            # vLLM expects unpadded token-id lists.
            attn_cpu = prompt_inputs["attention_mask"].cpu()
            prompt_ids_unpadded = [
                [int(t) for t, m in zip(row_ids.tolist(), row_mask.tolist()) if m == 1]
                for row_ids, row_mask in zip(prompt_inputs["input_ids"], attn_cpu)
            ]

            # Single image per sample.
            images_for_vllm = [[img] for img in all_images] if all_images else None

            _, completion_ids_list, _, _ = self._vllm_gen.generate(
                prompts=prompt_ids_unpadded,
                images=images_for_vllm,
                num_generations=1,
            )

            # Pad completions into a tensor; right-pad with EOS.
            eos_id = self.processing_class.eos_token_id
            max_comp_len = max((len(ids) for ids in completion_ids_list), default=1)
            completion_ids = torch.full(
                (len(completion_ids_list), max_comp_len), eos_id, dtype=torch.long, device=device
            )
            for i, ids in enumerate(completion_ids_list):
                if ids:
                    completion_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)

            prompt_length = prompt_inputs["input_ids"].size(1)
            prompt_completion_ids = torch.cat([prompt_inputs["input_ids"].to(device), completion_ids], dim=1)
        else:
            # Hugging Face autoregressive generation path.
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                was_training = unwrapped_model.training
                unwrapped_model.eval()
                try:
                    generate_returned_result = unwrapped_model.generate(
                        **{k: v for k, v in prompt_inputs.items() if k not in self.vlm_module.get_non_generate_params()},
                        generation_config=self.generation_config,
                    )
                finally:
                    if was_training:
                        unwrapped_model.train()
                prompt_length = prompt_ids.size(1)
                prompt_completion_ids = generate_returned_result
                prompt_ids = prompt_completion_ids[:, :prompt_length]
                completion_ids = prompt_completion_ids[:, prompt_length:]

        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        multimodal_keywords = self.vlm_module.get_custom_multimodal_keywords()
        multimodal_inputs = {k: prompt_inputs[k] if k in prompt_inputs else None for k in multimodal_keywords}

        if script_args.cold_start:
            need_uncertainty = (
                script_args.use_uncertainty_weighting
                and _current_training_phase >= 3
            )
        else:
            need_uncertainty = (
                script_args.use_uncertainty_weighting
                and step >= script_args.uncertainty_warmup_steps
            )
        need_policy_logits = self.num_iterations > 1 or need_uncertainty
        bbox_entropies = None

        with torch.no_grad():
            if need_policy_logits:
                full_logits = model(
                    input_ids=prompt_completion_ids,
                    attention_mask=attention_mask,
                    **multimodal_inputs,
                ).logits  # (B, L, V)

                if self.num_iterations > 1:
                    shifted_logits = full_logits[:, :-1, :]
                    shifted_ids = prompt_completion_ids[:, 1:]
                    per_token_logps = []
                    for logits_row, ids_row in zip(shifted_logits, shifted_ids):
                        log_probs = logits_row.log_softmax(dim=-1)
                        token_log_prob = torch.gather(
                            log_probs, dim=1, index=ids_row.unsqueeze(1)
                        ).squeeze(1)
                        per_token_logps.append(token_log_prob)
                    old_per_token_logps = torch.stack(per_token_logps)[:, prompt_length - 1:]
                else:
                    old_per_token_logps = None

                if need_uncertainty:
                    completion_logits = full_logits[:, prompt_length - 1:-1, :]
                    bbox_entropies = compute_bbox_entropies(
                        completion_logits,
                        completion_ids,
                        self.processing_class.tokenizer,
                    )
                    mean_entropy = [
                        sum(es) / len(es) for es in bbox_entropies if es
                    ]
                    if mean_entropy:
                        self._metrics["spatial_entropy"].append(
                            sum(mean_entropy) / len(mean_entropy)
                        )
                    bbox_rate = sum(1 for es in bbox_entropies if es) / len(bbox_entropies) if bbox_entropies else 0.0
                    self._metrics["bbox_rate"].append(bbox_rate)

                del full_logits
                torch.cuda.empty_cache()
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, **multimodal_inputs
                )
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        model, prompt_completion_ids, attention_mask, **multimodal_inputs
                    )
        if ref_per_token_logps is not None:
            ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]

        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if step == 0 and self.accelerator.is_main_process:
            raw_first = self.processing_class.tokenizer.decode(completion_ids[0], skip_special_tokens=False)
            prompt_msgs = inputs[0].get("prompt", [])
            user_msgs = [m for m in prompt_msgs if m.get("role") == "user"]
            if user_msgs:
                last_user_content = user_msgs[-1].get("content", "")
                if isinstance(last_user_content, list):
                    _q = " ".join(p["text"] for p in last_user_content if isinstance(p, dict) and p.get("type") == "text")
                else:
                    _q = str(last_user_content)
            else:
                _q = "(no user message found)"
            print(
                f"[DEBUG-RAW] Step 0 sample\n"
                f"  Question: {_q[:400]}\n"
                f"  Completion (raw, first 500 chars):\n{raw_first[:500]}"
            )
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        bbox_entropies_json = None
        if bbox_entropies is not None:
            bbox_entropies_json = [json.dumps(es) for es in bbox_entropies]

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else:
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        reward_kwargs[key].extend([example[key]])
                if bbox_entropies_json is not None:
                    reward_kwargs["bbox_entropies"] = bbox_entropies_json
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        rewards_per_func = self.accelerator.gather(rewards_per_func)
        rewards = rewards_per_func.sum(dim=1)
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)
        self._metrics["training_phase"].append(float(_current_training_phase))

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            reward_func_name = (
                reward_func.config._name_or_path.split("/")[-1]
                if isinstance(reward_func, PreTrainedModel)
                else reward_func.__name__
            )
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        if script_args.debug_completions and step % 10 == 0 and self.accelerator.is_main_process:
            sample_text = (
                completions[0][0]["content"] if is_conversational(inputs[0])
                else completions[0]
            )
            sample_rewards = rewards_per_func[0].tolist()
            reward_names = [
                rf.__name__ if not isinstance(rf, PreTrainedModel)
                else rf.config._name_or_path.split("/")[-1]
                for rf in self.reward_funcs
            ]
            reward_str = ", ".join(
                f"{n}={v:.3f}" for n, v in zip(reward_names, sample_rewards)
            )
            # Extract the question from the last user message in the prompt.
            prompt_msgs = inputs[0].get("prompt", [])
            user_msgs = [m for m in prompt_msgs if m.get("role") == "user"]
            if user_msgs:
                last_user_content = user_msgs[-1].get("content", "")
                if isinstance(last_user_content, list):
                    question_text = " ".join(
                        p["text"] for p in last_user_content if isinstance(p, dict) and p.get("type") == "text"
                    )
                else:
                    question_text = str(last_user_content)
            else:
                question_text = "(no user message found)"
            question_preview = question_text[:300] + ("..." if len(question_text) > 300 else "")
            preview = sample_text[:500] + ("..." if len(sample_text) > 500 else "")
            print(
                f"\n{'='*60}\n"
                f"[Step {step} | Phase {_current_training_phase}] Question:\n"
                f"{'-'*60}\n"
                f"{question_preview}\n"
                f"{'-'*60}\n"
                f"Sample completion:\n"
                f"{'-'*60}\n"
                f"{preview}\n"
                f"{'-'*60}\n"
                f"Rewards: {reward_str} | total={sum(sample_rewards):.3f}\n"
                f"{'='*60}\n"
            )

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "multimodal_inputs": multimodal_inputs,
        }


@dataclass
class OmniSpatialGRPOScriptArguments(ScriptArguments):
    data_json: str = field(
        default=DEFAULT_DATA_JSON,
        metadata={"help": "Path to the OmniSpatial training data.json"},
    )
    root_dir: str = field(
        default=DEFAULT_ROOT_DIR,
        metadata={"help": "Root directory containing the task-specific image folders"},
    )
    task_types: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated task types to train on (default: all task types in data.json)"},
    )
    val_split_ratio: float = field(
        default=0.0,
        metadata={"help": "Optional validation split fraction"},
    )
    dataset_limit: Optional[int] = field(
        default=None,
        metadata={"help": "Optional cap on the number of training samples for quick runs"},
    )
    balance_tasks: bool = field(
        default=False,
        metadata={
            "help": "Oversample minority tasks (with replacement) so every task has the same count as the largest one."
        },
    )
    with_reasoning: bool = field(
        default=True,
        metadata={"help": "Enable Qwen3 thinking mode (require </think> then a single answer letter)"},
    )
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "Override the system prompt (default: load from utils.system_prompts)"},
    )
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "grounding"],
        metadata={"help": "Reward functions to apply"},
    )
    accuracy_reward_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for the answer-correctness reward"},
    )
    format_reward_weight: float = field(
        default=0.2,
        metadata={"help": "Weight for the output-format reward"},
    )
    grounding_reward_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for the spatial grounding reward"},
    )
    grounding_iou_threshold: float = field(
        default=0.5,
        metadata={"help": "Reserved IoU threshold (kept for backward compatibility)"},
    )
    grounding_iou_weight: float = field(
        default=0.8,
        metadata={"help": "Weight of the IoU term in the Hungarian pair score"},
    )
    grounding_label_weight: float = field(
        default=0.2,
        metadata={"help": "Weight of the label cosine-similarity term in the Hungarian pair score"},
    )
    grounding_annotations: Optional[str] = field(
        default=None,
        metadata={"help": "Path to precomputed grounding annotations (phrases + pseudo ground-truth bboxes)"},
    )
    grounding_use_recall: bool = field(
        default=True,
        metadata={"help": "Combine precision and soft recall via F-beta (False = precision only)"},
    )
    grounding_fbeta: float = field(
        default=2.0,
        metadata={"help": "Beta for F-beta. β=1 is balanced; β>1 favours recall; β<1 favours precision."},
    )
    use_uncertainty_weighting: bool = field(
        default=True,
        metadata={"help": "Weight the grounding reward by per-bbox confidence (1 - normalised entropy)"},
    )
    uncertainty_beta: float = field(
        default=0.5,
        metadata={"help": "Floor for omega = beta + (1 - beta) * (1 - U_tilde)"},
    )
    uncertainty_warmup_steps: int = field(
        default=0,
        metadata={"help": "Steps before uncertainty weighting activates (used when cold_start is False)"},
    )
    spatial_answer_gate: float = field(
        default=0.3,
        metadata={
            "help": (
                "Soft gate γ ∈ (0, 1) on the spatial reward when the answer is wrong. "
                "Returns 1.0 when the answer is correct and γ otherwise."
            )
        },
    )
    over_prediction_penalty: float = field(
        default=0.05,
        metadata={
            "help": (
                "Penalty per predicted bbox that has no ground-truth match. Also applied "
                "per-bbox when ground truth is empty. Subtracted before clipping the reward to -1."
            )
        },
    )
    missing_bbox_accuracy_mult: float = field(
        default=1.0,
        metadata={
            "help": (
                "Multiplier applied to the accuracy reward when ground truth has bboxes but the "
                "completion emits none. Default 1.0 disables this gate (use grounding_coverage_floor instead)."
            )
        },
    )
    grounding_coverage_floor: float = field(
        default=1.0,
        metadata={
            "help": (
                "Minimum accuracy multiplier when ground truth has bboxes. Default 1.0 disables gating. "
                "Set < 1.0 to scale accuracy by (floor + (1 - floor) * coverage), where coverage is the "
                "fraction of ground-truth bboxes matched by a prediction (IoU >= grounding_coverage_iou_threshold)."
            )
        },
    )
    grounding_coverage_iou_threshold: float = field(
        default=0.1,
        metadata={
            "help": (
                "IoU threshold for considering a ground-truth bbox 'covered' when computing "
                "the coverage gate. Lenient by default since we reward effort, not precision."
            )
        },
    )
    grounding_text_anchor: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, penalise each predicted bbox whose label tokens do not appear in "
                "the question or option text. Per-prediction penalty = "
                "grounding_text_anchor_weight * (1 - fraction of label tokens found in text)."
            )
        },
    )
    grounding_text_anchor_weight: float = field(
        default=0.1,
        metadata={"help": "Weight applied per prediction when grounding_text_anchor is True."},
    )
    cold_start: bool = field(
        default=False,
        metadata={"help": "Enable the three-phase curriculum: format → +grounding → +uncertainty"},
    )
    cold_start_phase1_ratio: float = field(
        default=0.10,
        metadata={"help": "Fraction of total steps spent in Phase 1 (format only, scaled accuracy)"},
    )
    cold_start_phase2_ratio: float = field(
        default=0.20,
        metadata={"help": "Fraction of total steps marking the end of Phase 2 (grounding without uncertainty)"},
    )
    cold_start_accuracy_scale: float = field(
        default=0.2,
        metadata={"help": "Multiplier applied to the accuracy reward during Phase 1"},
    )
    cold_start_bbox_bonus: float = field(
        default=0.0,
        metadata={
            "help": (
                "Phase-1 only bonus when at least one valid bbox is emitted. Default 0.0. "
                "A non-zero value rewards any valid bbox regardless of question relevance."
            )
        },
    )
    bbox_attempt_bonus: float = field(
        default=0.05,
        metadata={
            "help": (
                "Small bonus added to the format reward when ground-truth bboxes exist and "
                "the completion produced at least one valid bbox. Prevents the model from "
                "settling on a no-prediction policy to avoid IoU penalties."
            )
        },
    )
    debug_completions: bool = field(
        default=False,
        metadata={"help": "Print a sample completion with reward breakdown every 10 steps."},
    )
    no_resume: bool = field(
        default=False,
        metadata={"help": "Start a fresh run even if checkpoints exist in output_dir."},
    )
    max_pixels: Optional[int] = field(
        default=128 * 28 * 28,
        metadata={"help": "Maximum image pixels forwarded to the processor"},
    )
    min_pixels: Optional[int] = field(
        default=16 * 28 * 28,
        metadata={"help": "Minimum image pixels forwarded to the processor"},
    )


@dataclass
class OmniSpatialGRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False
    train_vision_lora: bool = False
    vision_lora_r: Optional[int] = None


def build_dataset(local_script_args: OmniSpatialGRPOScriptArguments) -> Dataset:
    data_json = Path(local_script_args.data_json)
    root_dir = Path(local_script_args.root_dir)
    task_types = normalize_task_args(
        local_script_args.task_types.split(",") if local_script_args.task_types else None
    )

    grounding_annots = {}
    if local_script_args.grounding_annotations:
        annot_path = Path(local_script_args.grounding_annotations)
        if annot_path.exists():
            with open(annot_path, "r", encoding="utf-8") as f:
                grounding_annots = json.load(f)
            print(f"Loaded {len(grounding_annots)} grounding annotations from {annot_path}")

    raw_samples = load_data(str(data_json))
    selected_tasks = set(task_types) if task_types else None

    filtered = []
    for sample in raw_samples:
        task_type = str(sample.get("task_type", "")).strip()
        if not task_type:
            continue
        if selected_tasks is not None and task_type not in selected_tasks:
            continue
        filtered.append(sample)

    if local_script_args.dataset_limit is not None:
        filtered = filtered[: local_script_args.dataset_limit]

    if local_script_args.balance_tasks:
        import random
        from collections import defaultdict
        grouped: dict = defaultdict(list)
        for s in filtered:
            grouped[str(s.get("task_type", ""))].append(s)
        target = max(len(v) for v in grouped.values())
        balanced = []
        for task, rows in grouped.items():
            if len(rows) < target:
                balanced += random.choices(rows, k=target)
            else:
                balanced += rows
        random.shuffle(balanced)
        filtered = balanced
        print(f"[balance_tasks] target={target} per task, total={len(filtered)}")
        for task, rows in sorted(grouped.items()):
            print(f"  {task}: {len(rows)} → {target}")

    all_rows = []
    for sample in filtered:
        sample_id = str(sample.get("id"))
        task_type = str(sample.get("task_type"))
        image_path = resolve_image_path(root_dir, task_type, sample_id)
        options = sample.get("options") or []
        question = str(sample.get("question", "")).strip()
        answer = format_gold_answer(sample) or ""
        user_text = build_user_text(question, options, local_script_args.with_reasoning)
        system_prompt = local_script_args.system_prompt or load_system_prompt(
            local_script_args.with_reasoning
        )

        messages = []
        if system_prompt:
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
                    {"type": "text", "text": user_text},
                ],
            }
        )

        annot_key = f"{task_type}/{sample_id}"
        annot_entry = grounding_annots.get(annot_key, {})
        gt_bboxes_json = json.dumps(annot_entry.get("bboxes", []))

        all_rows.append(
            {
                "sample_id": sample_id,
                "task_type": task_type,
                "question": question,
                "options": [str(option) for option in options],
                "answer": answer,
                "image_path": [str(image_path)],
                "data_type": "single_image",
                "gt_bboxes": gt_bboxes_json,
                "prompt": messages,
            }
        )

    features = Features(
        {
            "sample_id": Value("string"),
            "task_type": Value("string"),
            "question": Value("string"),
            "options": HFSequence(Value("string")),
            "answer": Value("string"),
            "image_path": HFSequence(Value("string")),
            "data_type": Value("string"),
            "gt_bboxes": Value("string"),
            "prompt": [
                {
                    "role": Value("string"),
                    "content": [
                        {
                            "type": Value("string"),
                            "text": Value("string"),
                            "image": Value("string"),
                        }
                    ],
                }
            ],
        }
    )

    return Dataset.from_list(all_rows, features=features)


def main(local_script_args, training_args, model_args):
    import os
    os.environ.setdefault("WANDB_PROJECT", "spatialcore")

    print(f"[DEBUG] dataset_limit={local_script_args.dataset_limit}, task_types={local_script_args.task_types}")
    dataset = build_dataset(local_script_args)
    print(f"[DEBUG] dataset size: {len(dataset)}")
    reward_funcs = [reward_funcs_registry[name] for name in local_script_args.reward_funcs]

    train_dataset = dataset
    eval_dataset = None
    if local_script_args.val_split_ratio > 0:
        split = dataset.train_test_split(test_size=local_script_args.val_split_ratio)
        train_dataset = split["train"]
        eval_dataset = split["test"]

    trainer = Qwen3GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        vlm_module=Qwen3VLModule(enable_thinking=local_script_args.with_reasoning),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        train_vision_lora=model_args.train_vision_lora,
        vision_lora_r=model_args.vision_lora_r,
        attn_implementation=model_args.attn_implementation,
        max_pixels=local_script_args.max_pixels,
        min_pixels=local_script_args.min_pixels,
    )
    trainer._patch_generation_config()

    if getattr(training_args, "use_vllm", False):
        trainer._init_vllm()

    if not local_script_args.no_resume and list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    parser = TrlParser((OmniSpatialGRPOScriptArguments, GRPOConfig, OmniSpatialGRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
