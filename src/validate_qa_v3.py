"""Validate challenge-facing curated QA v3 artifacts before GPU work."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from curate_surgmotion_vqa import case_number, segment_key
from generate_qa import (
    ALL_TARGET_TOOLS,
    COMMERCIAL_VARIANTS,
    FORCEPS_TOOLS,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN = PROJECT_ROOT / "src" / "train_vqa_surgmotion_curated_v3.json"
DEFAULT_VAL = PROJECT_ROOT / "src" / "val_vqa_surgmotion_curated_v3.json"
DEFAULT_TEST = PROJECT_ROOT / "src" / "test_vqa_surgmotion_curated_v3.json"
DEFAULT_ALL = PROJECT_ROOT / "src" / "all_vqa_surgmotion_curated_v3.json"
DEFAULT_SEGMENTS = PROJECT_ROOT / "src" / "surgvu_segments_v2.json"
DEFAULT_REPORT = PROJECT_ROOT / "runs" / "dataset_audit" / "qa_v3_validation.json"

REQUIRED_FIELDS = {"id", "video", "t_start", "t_end", "question", "answers"}
FORBIDDEN_GRAMMAR = (
    re.compile(r"\bforcepss\b", re.IGNORECASE),
    re.compile(r"\ba forceps\b", re.IGNORECASE),
)
PRESENCE_VISUAL_WORDS = re.compile(
    r"\b(being used|was used|present|involved|utilized)\b", re.IGNORECASE
)
DESCRIPTION_MARKERS = (
    "activity outside of the structured training",
    "training activity is other",
    "annotator",
    "annotation note",
)
COMMERCIAL_TOOLS = {
    variant
    for variants in COMMERCIAL_VARIANTS.values()
    for variant in variants
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize(text):
    return re.sub(r"\s+", " ", str(text).strip().lower())


def validate_split(name, rows, expected_cases, segment_map, errors):
    counts = Counter()
    semantic = {}
    for index, row in enumerate(rows):
        prefix = f"{name}[{index}]/{row.get('id', '<missing>')}"
        missing = REQUIRED_FIELDS.difference(row)
        if missing:
            errors.append(f"{prefix}: missing fields {sorted(missing)}")
            continue

        counts[row.get("question_type", "missing")] += 1
        number = case_number(row)
        if number not in expected_cases:
            errors.append(f"{prefix}: case {number} outside expected range")

        answers = row["answers"]
        if not isinstance(answers, list) or len(answers) != 5:
            errors.append(f"{prefix}: answers must contain exactly five references")
            continue
        normalized_answers = [normalize(answer) for answer in answers]
        if any(not answer for answer in normalized_answers):
            errors.append(f"{prefix}: empty reference answer")
        if len(set(normalized_answers)) != 5:
            errors.append(f"{prefix}: reference answers are not unique")

        combined = " ".join([row["question"], *map(str, answers)])
        for pattern in FORBIDDEN_GRAMMAR:
            if pattern.search(combined):
                errors.append(f"{prefix}: forbidden grammar pattern {pattern.pattern}")

        key = (
            row["video"],
            float(row["t_start"]),
            float(row["t_end"]),
            normalize(row["question"]),
        )
        signature = tuple(sorted(normalized_answers))
        if key in semantic:
            if semantic[key] != signature:
                errors.append(f"{prefix}: conflicting semantic supervision")
            else:
                errors.append(f"{prefix}: duplicate semantic supervision")
        else:
            semantic[key] = signature

        metadata = segment_map.get(segment_key(row))
        if metadata is None:
            errors.append(f"{prefix}: segment metadata not found")
            continue

        qtype = row.get("question_type")
        question = normalize(row["question"])
        active_tools = set(metadata.get("tools", []))
        active_commercial = set(metadata.get("commercial_tools", []))

        if qtype == "tool_presence":
            if not ("listed" in question or "installation record" in question):
                errors.append(f"{prefix}: tool-presence question lacks listed/installation wording")
            if PRESENCE_VISUAL_WORDS.search(question):
                errors.append(f"{prefix}: tool-presence question makes a visual-use claim")
            target = row.get("tool")
            if target == "forceps":
                expected = any(tool in active_tools for tool in FORCEPS_TOOLS)
            elif target in COMMERCIAL_TOOLS:
                expected = target in active_commercial
            else:
                expected = target in active_tools
            actual = normalized_answers[0] == "yes"
            if actual != expected:
                errors.append(
                    f"{prefix}: tool-presence label mismatch target={target} "
                    f"expected={expected} actual={actual}"
                )

        elif qtype == "forceps_type":
            expected_forceps = [tool for tool in FORCEPS_TOOLS if tool in active_tools]
            for answer in normalized_answers:
                for tool in expected_forceps:
                    if tool not in answer:
                        errors.append(f"{prefix}: reference omits active subtype {tool}")
                        break
            if len(expected_forceps) >= 2 and "types" not in question:
                errors.append(f"{prefix}: multi-forceps question is not explicitly plural")

        elif qtype == "tool_purpose":
            if not active_tools:
                errors.append(f"{prefix}: tool-purpose generated without an active tool")
            elif not any(tool in question for tool in ALL_TARGET_TOOLS if tool in active_tools):
                errors.append(f"{prefix}: tool-purpose does not name an active tool exactly")

        elif qtype == "organ" and row.get("grounding_status") == "semantic_task_label":
            if len(set(normalized_answers)) != 5:
                errors.append(f"{prefix}: semantic organ item does not have five unique references")

        elif qtype == "description":
            lowered = combined.lower()
            if any(marker in lowered for marker in DESCRIPTION_MARKERS):
                errors.append(f"{prefix}: placeholder or annotation text leaked into description")

    return {
        "examples": len(rows),
        "cases": sorted({case_number(row) for row in rows}),
        "question_types": dict(counts),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=str(DEFAULT_TRAIN))
    parser.add_argument("--val", default=str(DEFAULT_VAL))
    parser.add_argument("--test", default=str(DEFAULT_TEST))
    parser.add_argument("--all", dest="all_path", default=str(DEFAULT_ALL))
    parser.add_argument("--segments", default=str(DEFAULT_SEGMENTS))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    return parser.parse_args()


def main():
    args = parse_args()
    splits = {
        "train": load_json(args.train),
        "val": load_json(args.val),
        "test": load_json(args.test),
    }
    all_rows = load_json(args.all_path)
    segment_rows = load_json(args.segments)
    segment_map = {segment_key(row): row for row in segment_rows}
    errors = []

    expected = {
        "train": set(range(0, 140)),
        "val": set(range(140, 147)),
        "test": set(range(147, 155)),
    }
    summary = {
        name: validate_split(name, rows, expected[name], segment_map, errors)
        for name, rows in splits.items()
    }

    ids_by_split = {name: {row["id"] for row in rows} for name, rows in splits.items()}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = ids_by_split[left].intersection(ids_by_split[right])
        if overlap:
            errors.append(f"{left}/{right}: {len(overlap)} overlapping ids")

    concatenated_ids = [
        row["id"]
        for name in ("train", "val", "test")
        for row in splits[name]
    ]
    all_ids = [row["id"] for row in all_rows]
    if concatenated_ids != all_ids:
        errors.append("all split is not the ordered train+val+test concatenation")
    if len(all_ids) != len(set(all_ids)):
        errors.append("ids are not globally unique")

    report = {
        "status": "pass" if not errors else "fail",
        "splits": summary,
        "all_examples": len(all_rows),
        "errors": errors,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if errors:
        raise RuntimeError(f"QA v3 validation failed with {len(errors)} errors")


if __name__ == "__main__":
    main()
