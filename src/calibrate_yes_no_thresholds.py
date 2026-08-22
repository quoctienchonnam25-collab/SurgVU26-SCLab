"""Sweep calibrated Yes/No decision thresholds (first-token logit margin, same
mechanism as predict_dual_encoder_vqa.py::tool_presence_answer_tensor) for every
binary question type on the true holdout, in one pass.

Motivation: the v3 checkpoint's --tool-presence-threshold=0.5 shows this margin-
based decision helps at least one Yes/No category; tissue_cutting and
suture_required are the other two binary question types in this corpus and have
never had a calibrated threshold picked for them specifically - this finds one
from data rather than guessing, and reports the free-generation baseline
alongside it so the gain (if any) is visible before wiring in routing.
"""
import argparse
import json
from pathlib import Path

from predict_dual_encoder_vqa import (
    answer_tensor,
    assert_gpu_idle,
    load_model,
    normalize_answer,
    tool_presence_answer_tensor,
)
from train_dual_encoder_vqa import DualEncoderQADataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def polarity(text):
    normalized = normalize_answer(text)
    if normalized.startswith("yes"):
        return True
    if normalized.startswith("no"):
        return False
    return None


def sweep(records):
    """records: list of (margin, ground_truth_bool). Returns best (threshold, accuracy)."""
    if not records:
        return None, None
    candidates = [round(t, 2) for t in [-3 + 0.1 * i for i in range(61)]]
    best_threshold, best_accuracy = 0.0, -1.0
    for threshold in candidates:
        correct = sum((margin >= threshold) == truth for margin, truth in records)
        accuracy = correct / len(records)
        if accuracy > best_accuracy:
            best_threshold, best_accuracy = threshold, accuracy
    return best_threshold, best_accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vqa-json", default=str(PROJECT_ROOT / "src" / "val_vqa_surgmotion_curated_v2.json"))
    parser.add_argument("--cache-root", default=str(PROJECT_ROOT / "frame_cache_16fr"))
    parser.add_argument(
        "--question-types", nargs="+", default=["tool_presence", "tissue_cutting", "suture_required"]
    )
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--limit-per-type", type=int, default=None)
    parser.add_argument("--output", default=str(PROJECT_ROOT / "runs" / "surgmotion_vqa" / "dual_encoder" / "yes_no_threshold_calibration.json"))
    args = parser.parse_args()

    assert_gpu_idle()
    model, tokenizer, processor, qwen_num_frames = load_model(args.checkpoint)

    items = json.load(open(args.vqa_json, encoding="utf-8"))
    dataset = DualEncoderQADataset(args.vqa_json, args.cache_root, is_train=False, qwen_num_frames=qwen_num_frames)

    by_type = {}
    for index, item in enumerate(items):
        qtype = item.get("question_type")
        if qtype not in args.question_types:
            continue
        truth = polarity(item["answers"][0])
        if truth is None:
            continue
        by_type.setdefault(qtype, []).append((index, item, truth))
    if args.limit_per_type:
        for qtype in by_type:
            by_type[qtype] = by_type[qtype][: args.limit_per_type]

    results = {}
    for qtype, type_items in by_type.items():
        records = []
        greedy_correct = 0
        for position, (index, item, truth) in enumerate(type_items):
            sample = dataset[index]
            qwen_frames = sample["qwen_frames"][:qwen_num_frames]
            aux_video = sample["aux_video"].unsqueeze(0)
            aux_valid = sample["aux_frame_valid_mask"].unsqueeze(0)

            # threshold=0.0 here is a no-op placeholder; only the returned margin is used,
            # the actual decision threshold is applied later by sweep().
            _, margin = tool_presence_answer_tensor(
                model, tokenizer, processor, qwen_frames, aux_video, aux_valid, item["question"], 0.0
            )
            records.append((margin, truth))

            greedy = answer_tensor(
                model, tokenizer, processor, qwen_frames, aux_video, aux_valid, item["question"], args.max_new_tokens
            )
            greedy_pred = polarity(greedy)
            if greedy_pred == truth:
                greedy_correct += 1

            if (position + 1) % 50 == 0:
                print(f"[{qtype}] {position + 1}/{len(type_items)}")

        best_threshold, best_accuracy = sweep(records)
        margin0_accuracy = sum((margin >= 0.0) == truth for margin, truth in records) / len(records)
        greedy_accuracy = greedy_correct / len(type_items)
        yes_rate = sum(truth for _, truth in records) / len(records)
        results[qtype] = {
            "n": len(records),
            "ground_truth_yes_rate": yes_rate,
            "greedy_free_generation_accuracy": greedy_accuracy,
            "margin_threshold_0.0_accuracy": margin0_accuracy,
            "best_threshold": best_threshold,
            "best_threshold_accuracy": best_accuracy,
        }
        print(f"=== {qtype} ===")
        print(json.dumps(results[qtype], indent=2))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(output_path, "w", encoding="utf-8"), indent=2)
    print(f"[calibration] wrote {output_path}")


if __name__ == "__main__":
    main()
