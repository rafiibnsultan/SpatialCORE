#!/usr/bin/env python3
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DIGIT_RE = re.compile(r"\b([1-4])\b")
LETTER_RE = re.compile(r"\b([A-D])\b", re.IGNORECASE)

CATEGORY_WEIGHTS = {
    "3D Geometry":          238,
    "Depth & Occlusion":    259,
    "Orientation":          202,
    "Relative Positioning": 212,
    "Size & Scale":         252,
    "Spatial Navigation":   237,
}
TOTAL_SAMPLES = 1400


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SpatiaLab predictions.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--show-errors", type=int, default=0)
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {i}: {e}")
    return rows


def extract_digit(text: Optional[str], options: Optional[List[str]] = None) -> Optional[str]:
    if not text:
        return None
    text = str(text).strip()

    # Text-match mode: try to find which option the prediction matches
    if options:
        t = text.lower()
        for i, opt in enumerate(options):
            if t == opt.strip().lower():
                return str(i + 1)
        for i, opt in enumerate(options):
            opt_lower = opt.strip().lower()
            if opt_lower and (opt_lower in t or t in opt_lower):
                return str(i + 1)

    # Tagged answer format: <result>N</result>
    m = re.search(r"<result>\s*([1-4])\s*</result>", text, re.IGNORECASE)
    if m:
        return m.group(1)

    # Standalone digit 1-4
    m = DIGIT_RE.search(text)
    if m:
        return m.group(1)

    # Letter A-D mapped to digit (handles models that emit letter answers)
    m = LETTER_RE.search(text)
    if m:
        return str(ord(m.group(1).upper()) - ord("A") + 1)

    return None


def evaluate(rows: List[Dict[str, Any]]) -> Tuple[int, int, int, List[str]]:
    correct = total = unparsed = 0
    examples = []

    for row in rows:
        total += 1
        options = row.get("options") if isinstance(row.get("options"), list) else None
        gold = extract_digit(str(row.get("gold_answer", "")))
        pred = extract_digit(str(row.get("prediction", "")), options)

        if gold is not None and pred is not None:
            if gold == pred:
                correct += 1
            elif len(examples) < 1000:
                examples.append(
                    f"id={row.get('id')} mismatch: gold={gold} pred={pred} | {str(row.get('prediction',''))!r}"
                )
        else:
            unparsed += 1
            if len(examples) < 1000:
                examples.append(
                    f"id={row.get('id')} unparsed: gold={row.get('gold_answer')!r}->{gold}, pred={row.get('prediction')!r}->{pred}"
                )

    return correct, total, unparsed, examples


def main() -> int:
    args = parse_args()
    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    rows = load_jsonl(path)
    correct, total, unparsed, examples = evaluate(rows)
    accuracy = correct / total if total else 0.0

    print(f"Input: {path}")
    print(f"Rows: {total}")
    print(f"Unparsed: {unparsed}")
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy:.4%}")

    # Per-category breakdown
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("category", "unknown"))].append(row)

    print("\nPer-category accuracy:")
    weighted_sum = weighted_n = 0
    for cat in sorted(grouped):
        cat_correct, cat_total, cat_unparsed, _ = evaluate(grouped[cat])
        cat_acc = cat_correct / cat_total if cat_total else 0.0
        n = CATEGORY_WEIGHTS.get(cat, cat_total)
        weighted_sum += n * cat_acc * 100
        weighted_n += n
        print(f"  {cat}: {cat_acc:.4%} ({cat_correct}/{cat_total}, unparsed={cat_unparsed})")

    if weighted_n > 0:
        weighted_avg = weighted_sum / weighted_n
        print(f"\nWeighted average (sample-weighted): {weighted_avg:.2f}%")

    if args.show_errors > 0 and examples:
        print(f"\nExamples (up to {args.show_errors}):")
        for line in examples[: args.show_errors]:
            print(f"  {line}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
