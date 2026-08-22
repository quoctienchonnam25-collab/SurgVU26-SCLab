"""Create quality-aware SurgMotion VQA train/validation files.

The original SurgVU QA files are preserved. This script joins the visual-signal
audit with the generated QA metadata, removes unanswerable supervision on fully
black clips, rewrites metadata-derived organ questions as simulation semantics,
and emits diagnostic validation slices.
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN = PROJECT_ROOT / "src" / "train_vqa.json"
DEFAULT_VAL = PROJECT_ROOT / "src" / "val_vqa.json"
DEFAULT_SEGMENTS = PROJECT_ROOT / "src" / "surgvu_segments.json"
DEFAULT_BLACK_AUDIT = PROJECT_ROOT / "runs" / "dataset_audit" / "black_screen_audit.json"
DEFAULT_ENDPOINT_AUDIT = (
    PROJECT_ROOT
    / "runs"
    / "dataset_audit"
    / "black_screen_raw_endpoint_confirmation.json"
)
DEFAULT_TRAIN_OUTPUT = PROJECT_ROOT / "src" / "train_vqa_surgmotion_curated.json"
DEFAULT_VAL_OUTPUT = PROJECT_ROOT / "src" / "val_vqa_surgmotion_curated.json"
DEFAULT_TEST_OUTPUT = PROJECT_ROOT / "src" / "test_vqa_surgmotion_curated.json"
DEFAULT_ALL_OUTPUT = PROJECT_ROOT / "src" / "all_vqa_surgmotion_curated.json"
DEFAULT_DIAGNOSTIC_DIR = PROJECT_ROOT / "src" / "surgmotion_diagnostics"
DEFAULT_MANIFEST = PROJECT_ROOT / "runs" / "dataset_audit" / "curation_manifest.json"

GENERIC_ORGAN = "the tissue in the surgical field"
DROP_WHEN_FULLY_BLACK = {
    "tool_presence",
    "forceps_type",
    "tissue_cutting",
    "organ",
    "description",
}
FULLY_BLACK_WEIGHTS = {
    "procedure_type": 0.1,
    "suture_required": 0.1,
    "task_id": 0.2,
    "tool_purpose": 0.2,
}
SEMANTIC_ORGAN_QUESTIONS = (
    "What anatomical structure is this training exercise intended to represent?",
    "Which anatomical structure is simulated by this training task?",
    "What organ or structure does this exercise represent?",
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def segment_key(item):
    video = item.get("video", item.get("video_path"))
    return (video, float(item["t_start"]), float(item["t_end"]))

def case_number(item):
    match = re.search(r"case_?(\d+)", item.get("id", ""), flags=re.IGNORECASE)
    if match is None:
        match = re.search(r"case_?(\d+)", item.get("video", ""), flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Cannot determine case number for {item.get('id')}")
    return int(match.group(1))


def question_type(item):
    identifier = item["id"]
    if "_tool_presence_" in identifier or "_tp_bal_" in identifier:
        return "tool_presence"
    for name in (
        "forceps_type",
        "suture_required",
        "procedure_type",
        "tool_purpose",
        "tissue_cutting",
        "description",
        "task_id",
        "organ",
    ):
        if f"_{name}_" in identifier:
            return name
    raise ValueError(f"Cannot infer question type from {identifier}")


def stable_fraction(identifier):
    digest = hashlib.sha256(identifier.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def build_visual_status(black_audit, endpoint_audit):
    base_rows = black_audit["segments"]
    raw_confirmed = {
        segment_key(row)
        for row in endpoint_audit["segments"]
        if row["all_three_near_black"]
    }
    fully_black = set()
    partially_black = set()
    for row in base_rows:
        key = segment_key(row)
        if row["source"].startswith("cache") and row["all_near_black"]:
            fully_black.add(key)
        elif key in raw_confirmed:
            fully_black.add(key)
        elif row["any_near_black"]:
            partially_black.add(key)
    partially_black.difference_update(fully_black)
    return fully_black, partially_black


def annotate_original(item, fully_black, partially_black):
    row = dict(item)
    key = segment_key(row)
    qtype = question_type(row)
    if key in fully_black:
        visual_status = "fully_black"
    elif key in partially_black:
        visual_status = "partially_black"
    else:
        visual_status = "valid"

    first_answer = str(row["answers"][0]).strip().lower()
    if qtype == "organ" and first_answer == GENERIC_ORGAN:
        grounding_status = "generic_fallback"
    elif qtype == "organ":
        grounding_status = "semantic_task_label"
    elif qtype in DROP_WHEN_FULLY_BLACK:
        grounding_status = "visual_claim"
    else:
        grounding_status = "metadata_or_language"

    row.update(
        {
            "visual_status": visual_status,
            "grounding_status": grounding_status,
            "question_type": qtype,
        }
    )
    return row


def semantic_organ_answers(first_answer):
    label = str(first_answer).strip().rstrip(".?!")
    lower = label.lower()
    return [
        label,
        f"The training exercise represents the {lower}.",
        f"The simulated anatomical structure is the {lower}.",
        f"This training task is intended to model the {lower}.",
        f"The anatomical focus of this exercise is the {lower}.",
    ]


def curate_item(item, generic_organ_keep_fraction):
    row = dict(item)
    visual_status = row["visual_status"]
    grounding_status = row["grounding_status"]
    qtype = row["question_type"]
    sample_weight = 1.0
    actions = []

    if visual_status == "fully_black":
        if qtype in DROP_WHEN_FULLY_BLACK:
            return None, "drop_fully_black_visual_question"
        sample_weight = FULLY_BLACK_WEIGHTS.get(qtype, 0.1)
        actions.append("downweight_fully_black_metadata")
    elif visual_status == "partially_black":
        sample_weight = 0.5
        actions.append("downweight_partially_black")

    if grounding_status == "generic_fallback":
        if stable_fraction(row["id"]) >= generic_organ_keep_fraction:
            return None, "drop_generic_organ_fallback"
        sample_weight = min(sample_weight, 0.5)
        actions.append("downsample_generic_organ_fallback")
    elif grounding_status == "semantic_task_label":
        template_index = int(stable_fraction(row["id"]) * len(SEMANTIC_ORGAN_QUESTIONS))
        template_index = min(template_index, len(SEMANTIC_ORGAN_QUESTIONS) - 1)
        row["original_question"] = row["question"]
        row["question"] = SEMANTIC_ORGAN_QUESTIONS[template_index]
        row["answers"] = semantic_organ_answers(row["answers"][0])
        sample_weight = min(sample_weight, 0.5)
        actions.append("rewrite_semantic_organ")

    row["sample_weight"] = float(sample_weight)
    row["curation_action"] = "+".join(actions) if actions else "keep"
    return row, None


def curate_split(rows, fully_black, partially_black, generic_organ_keep_fraction):
    curated = []
    annotated = []
    dropped = Counter()
    for item in rows:
        marked = annotate_original(item, fully_black, partially_black)
        annotated.append(marked)
        output, reason = curate_item(marked, generic_organ_keep_fraction)
        if output is None:
            dropped[reason] += 1
        else:
            curated.append(output)
    return curated, annotated, dropped


def summarize(rows):
    return {
        "examples": len(rows),
        "visual_status": dict(Counter(row["visual_status"] for row in rows)),
        "grounding_status": dict(Counter(row["grounding_status"] for row in rows)),
        "question_type": dict(Counter(row["question_type"] for row in rows)),
        "curation_action": dict(Counter(row.get("curation_action", "diagnostic") for row in rows)),
        "sample_weight": dict(Counter(str(row.get("sample_weight", 1.0)) for row in rows)),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json", default=str(DEFAULT_TRAIN))
    parser.add_argument("--val-json", default=str(DEFAULT_VAL))
    parser.add_argument("--segments-json", default=str(DEFAULT_SEGMENTS))
    parser.add_argument("--black-audit", default=str(DEFAULT_BLACK_AUDIT))
    parser.add_argument("--endpoint-audit", default=str(DEFAULT_ENDPOINT_AUDIT))
    parser.add_argument("--train-output", default=str(DEFAULT_TRAIN_OUTPUT))
    parser.add_argument("--val-output", default=str(DEFAULT_VAL_OUTPUT))
    parser.add_argument("--test-output", default=str(DEFAULT_TEST_OUTPUT))
    parser.add_argument("--all-output", default=str(DEFAULT_ALL_OUTPUT))
    parser.add_argument("--test-case-start", type=int, default=147)
    parser.add_argument("--diagnostic-dir", default=str(DEFAULT_DIAGNOSTIC_DIR))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--generic-organ-keep-fraction", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 < args.generic_organ_keep_fraction <= 1:
        raise ValueError("--generic-organ-keep-fraction must be in (0, 1]")
    if args.test_case_start <= 0:
        raise ValueError("--test-case-start must be positive")

    train_rows = load_json(args.train_json)
    heldout_rows = load_json(args.val_json)
    val_rows = [row for row in heldout_rows if case_number(row) < args.test_case_start]
    test_rows = [row for row in heldout_rows if case_number(row) >= args.test_case_start]
    if not val_rows or not test_rows:
        raise RuntimeError("The requested validation/test case split produced an empty split")

    segments = load_json(args.segments_json)
    metadata_keys = {segment_key(row) for row in segments}
    qa_keys = {segment_key(row) for row in train_rows + val_rows + test_rows}
    missing_metadata = qa_keys.difference(metadata_keys)
    if missing_metadata:
        raise RuntimeError(f"{len(missing_metadata)} QA segments have no segment metadata")

    fully_black, partially_black = build_visual_status(
        load_json(args.black_audit), load_json(args.endpoint_audit)
    )
    train_curated, train_annotated, train_dropped = curate_split(
        train_rows, fully_black, partially_black, args.generic_organ_keep_fraction
    )
    val_curated, val_annotated, val_dropped = curate_split(
        val_rows, fully_black, partially_black, args.generic_organ_keep_fraction
    )

    test_curated, test_annotated, test_dropped = curate_split(
        test_rows, fully_black, partially_black, args.generic_organ_keep_fraction
    )
    save_json(train_curated, args.train_output)
    save_json(val_curated, args.val_output)
    save_json(test_curated, args.test_output)
    save_json(train_curated + val_curated + test_curated, args.all_output)

    diagnostic_dir = Path(args.diagnostic_dir)
    diagnostic_splits = {
        # Clean/model-facing slices use curated rows so rewritten simulation prompts,
        # drops, and weights exactly match validation. The black stress slice keeps
        # all original questions on purpose: it quantifies how unreliable those labels
        # are and must not be mixed into the primary metric.
        "val_clean.json": [
            row
            for row in val_curated
            if row["visual_status"] == "valid"
            and row["grounding_status"] != "generic_fallback"
        ],
        "val_black.json": [row for row in val_annotated if row["visual_status"] == "fully_black"],
        "val_partial_black.json": [
            row for row in val_annotated if row["visual_status"] == "partially_black"
        ],
        "val_simulated_organ.json": [
            row for row in val_curated if row["grounding_status"] == "semantic_task_label"
        ],
        "val_forceps.json": [
            row for row in val_curated if row["question_type"] == "forceps_type"
        ],
        "test_clean.json": [
            row
            for row in test_curated
            if row["visual_status"] == "valid"
            and row["grounding_status"] != "generic_fallback"
        ],
        "test_forceps.json": [
            row for row in test_curated if row["question_type"] == "forceps_type"
        ],
    }
    for filename, rows in diagnostic_splits.items():
        save_json(rows, diagnostic_dir / filename)

    manifest = {
        "inputs": {
            "train_json": str(args.train_json),
            "val_json": str(args.val_json),
            "black_audit": str(args.black_audit),
            "endpoint_audit": str(args.endpoint_audit),
        },
        "policy": {
            "generic_organ_keep_fraction": args.generic_organ_keep_fraction,
            "drop_when_fully_black": sorted(DROP_WHEN_FULLY_BLACK),
            "test_case_start": args.test_case_start,
            "fully_black_weights": FULLY_BLACK_WEIGHTS,
            "partially_black_weight": 0.5,
            "semantic_organ_weight": 0.5,
        },
        "outputs": {
            "train": str(args.train_output),
            "val": str(args.val_output),
            "all": str(args.all_output),
            "test": str(args.test_output),
        },
        "visual_segments": {
            "fully_black": len(fully_black),
            "partially_black": len(partially_black),
        },
        "train": {
            "original_examples": len(train_rows),
            "curated": summarize(train_curated),
            "dropped": dict(train_dropped),
        },
        "val": {
            "original_examples": len(val_rows),
            "curated": summarize(val_curated),
            "dropped": dict(val_dropped),
        },
        "test": {
            "original_examples": len(test_rows),
            "curated": summarize(test_curated),
            "dropped": dict(test_dropped),
        },
        "diagnostic_splits": {
            name: summarize(rows) for name, rows in diagnostic_splits.items()
        },
    }
    save_json(manifest, args.manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
