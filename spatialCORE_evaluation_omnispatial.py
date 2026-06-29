#!/usr/bin/env python3
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

LETTER_RE = re.compile(r"\b([A-D])\b", re.IGNORECASE)
ANSWER_TAG_RE = re.compile(r"<ANSWER>\s*(.*?)\s*</ANSWER>", re.IGNORECASE | re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate multiple-choice accuracy from a JSONL predictions file."
    )
    parser.add_argument("--input", required=True, help="Path to predictions JSONL file.")
    parser.add_argument(
        "--denominator",
        choices=["all", "valid"],
        default="all",
        help="Use all rows or only rows with extractable labels for denominator.",
    )
    parser.add_argument(
        "--show-errors",
        type=int,
        default=0,
        help="How many mismatches/unparsed examples to print.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {i}: {exc}") from exc
    return rows


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def extract_answer_content(text: str) -> str:
    m = ANSWER_TAG_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def extract_choice_letter(text: str, options: Optional[List[str]] = None) -> Optional[str]:
    if not text:
        return None

    text = extract_answer_content(text)

    m = LETTER_RE.search(text)
    if m:
        return m.group(1).upper()

    if options:
        normalized_options = [str(x).strip().lower() for x in options]
        t = text.strip().lower()
        for idx, opt in enumerate(normalized_options):
            if t == opt:
                return chr(ord("A") + idx)
    else:
        numeric = text.strip()
        if numeric.isdigit():
            idx = int(numeric)
            if 0 <= idx < 4:
                return chr(ord("A") + idx)

    return None


def evaluate(rows: List[Dict[str, Any]], denominator: str) -> Tuple[int, int, int, int, List[str]]:
    correct = 0
    total = len(rows)
    valid = 0
    unparsed = 0
    examples: List[str] = []

    for row in rows:
        sample_id = row.get("id")
        gold = _normalize_text(row.get("gold_answer"))
        pred = _normalize_text(row.get("prediction"))
        options = row.get("options") if isinstance(row.get("options"), list) else None

        gold_letter = extract_choice_letter(gold, options)
        pred_letter = extract_choice_letter(pred, options)

        if gold_letter is not None and pred_letter is not None:
            valid += 1
            if gold_letter == pred_letter:
                correct += 1
            elif len(examples) < 1000:
                examples.append(
                    f"id={sample_id} mismatch: gold={gold_letter} pred={pred_letter} | pred_text={pred!r}"
                )
        else:
            unparsed += 1
            if len(examples) < 1000:
                examples.append(
                    f"id={sample_id} unparsed: gold={gold!r}->{gold_letter}, pred={pred!r}->{pred_letter}"
                )

    if denominator == "valid":
        denom = valid
    else:
        denom = total

    return correct, denom, total, unparsed, examples


def evaluate_by_task(
    rows: List[Dict[str, Any]], denominator: str
) -> Dict[str, Tuple[int, int, int, int]]:
    grouped_rows: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)

    for row in rows:
        task_type = _normalize_text(row.get("task_type")) or "unknown"
        grouped_rows[task_type].append(row)

    return {
        task_type: evaluate(task_rows, denominator)[:4]
        for task_type, task_rows in sorted(grouped_rows.items())
    }


def evaluate_by_sub_task(
    rows: List[Dict[str, Any]], denominator: str
) -> Dict[str, Tuple[int, int, int, int]]:
    grouped_rows: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)

    for row in rows:
        sub_task_type = _normalize_text(row.get("sub_task_type")) or "unknown"
        grouped_rows[sub_task_type].append(row)

    return {
        sub: evaluate(sub_rows, denominator)[:4]
        for sub, sub_rows in sorted(grouped_rows.items())
    }


def main() -> int:
    args = parse_args()
    path = Path(args.input)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    rows = load_jsonl(path)
    correct, denom, total, unparsed, examples = evaluate(rows, args.denominator)

    accuracy = (correct / denom) if denom > 0 else 0.0

    print(f"Input: {path}")
    print(f"Rows: {total}")
    print(f"Denominator mode: {args.denominator}")
    print(f"Unparsed rows: {unparsed}")
    print(f"Correct: {correct}/{denom}")
    print(f"Accuracy: {accuracy:.4%}")

    per_task = evaluate_by_task(rows, args.denominator)
    if per_task:
        print("\nPer-task accuracy:")
        for task_type, (task_correct, task_denom, task_total, task_unparsed) in per_task.items():
            task_accuracy = (task_correct / task_denom) if task_denom > 0 else 0.0
            print(
                f"  {task_type}: {task_accuracy:.4%} "
                f"({task_correct}/{task_denom}, rows={task_total}, unparsed={task_unparsed})"
            )

    per_sub_task = evaluate_by_sub_task(rows, args.denominator)
    has_sub_tasks = any(k != "unknown" for k in per_sub_task)
    if has_sub_tasks:
        print("\nPer-sub-task accuracy:")
        for sub, (sub_correct, sub_denom, sub_total, sub_unparsed) in per_sub_task.items():
            sub_accuracy = (sub_correct / sub_denom) if sub_denom > 0 else 0.0
            print(
                f"  {sub}: {sub_accuracy:.4%} "
                f"({sub_correct}/{sub_denom}, rows={sub_total}, unparsed={sub_unparsed})"
            )

        # Sub-task accuracy weighted by the number of test samples per sub-task.
        SUBTASK_WEIGHTS = {
            "Manipulation": 74, "Motion_Analysis": 346, "Traffic_Analysis": 85,
            "Localization": 105, "Geospatial_Strategy": 110, "Pattern_Recognition": 97,
            "Geometric_Reasoning": 155, "Egocentric": 102, "Allocentric": 376,
            "Hypothetical": 83,
        }
        TOTAL = 1533
        weighted_sum = sum(
            SUBTASK_WEIGHTS[sub] * (sub_correct / sub_denom)
            for sub, (sub_correct, sub_denom, _, _) in per_sub_task.items()
            if sub in SUBTASK_WEIGHTS and sub_denom > 0
        )
        print(f"\nWeighted average (sample-weighted): {weighted_sum / TOTAL:.4%}")

    n = max(0, args.show_errors)
    if n > 0 and examples:
        print(f"\nExamples (up to {n}):")
        for line in examples[:n]:
            print(f"- {line}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
