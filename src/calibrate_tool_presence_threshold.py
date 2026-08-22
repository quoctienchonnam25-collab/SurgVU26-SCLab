"""Validation-only conservative decoding calibration for tool-presence QA.

Scores the first assistant token for ``Yes`` versus ``No`` from a frozen
DualEncoderVQA checkpoint.  The threshold is selected only from the supplied
validation split to maximize tool-presence accuracy while meeting a requested
false-positive-rate target.  Other question types retain their existing
generated answers unchanged.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from predict_dual_encoder_vqa import (
    DEFAULT_BERTSCORE_MODEL,
    assert_gpu_idle,
    evaluate_rows,
    load_model,
)
from train_dual_encoder_vqa import DualEncoderCollator, DualEncoderQADataset


def polarity(text):
    return str(text).strip().lower().startswith("yes")


def token_ids_for_answer(collator, sample, answer):
    batch = collator(
        [
            {
                "qwen_frames": sample["qwen_frames"],
                "aux_video": sample["aux_video"],
                "aux_frame_valid_mask": sample["aux_frame_valid_mask"],
                "sample_weight": 1.0,
                "question": sample["question"],
                "answer": answer,
                "id": sample["id"],
            }
        ]
    )
    ids = batch["answer_ids"][0].tolist()
    if not ids:
        raise RuntimeError(f"No assistant token for answer {answer!r}")
    return ids, batch


def score_tool_presence(model, collator, dataset, items):
    probe = dataset[0]
    yes_ids, _ = token_ids_for_answer(collator, probe, "Yes")
    no_ids, _ = token_ids_for_answer(collator, probe, "No")
    yes_id, no_id = yes_ids[0], no_ids[0]
    if yes_id == no_id:
        raise RuntimeError("Yes and No resolve to the same first assistant token")

    scores = []
    for index, item in enumerate(items):
        if item.get("question_type") != "tool_presence":
            continue
        sample = dataset[index]
        _, batch = token_ids_for_answer(collator, sample, "Yes")
        model_inputs = {
            "prompt_ids": batch["prompt_ids"].to("cuda"),
            "pixel_values_videos": batch["pixel_values_videos"].to("cuda"),
            "video_grid_thw": batch["video_grid_thw"].to("cuda"),
            "aux_video": batch["aux_video"].to("cuda"),
            "aux_frame_valid_mask": batch["aux_frame_valid_mask"].to("cuda"),
            "second_per_grid_ts": batch.get("second_per_grid_ts"),
        }
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(**model_inputs)["logits"][0, -1].float()
        yes_logit = float(logits[yes_id])
        no_logit = float(logits[no_id])
        margin = yes_logit - no_logit
        scores.append(
            {
                "id": item["id"],
                "answer_yes": polarity(item["answers"][0]),
                "yes_token_id": yes_id,
                "no_token_id": no_id,
                "yes_logit": yes_logit,
                "no_logit": no_logit,
                "margin": margin,
            }
        )
        print(f"[score {len(scores)}] {item['id']} margin={margin:.5f}")
    return scores


def threshold_metrics(scores, threshold):
    predicted = [score["margin"] >= threshold for score in scores]
    truth = [score["answer_yes"] for score in scores]
    negatives = sum(not value for value in truth)
    false_positives = sum(pred and not actual for pred, actual in zip(predicted, truth))
    correct = sum(pred == actual for pred, actual in zip(predicted, truth))
    return {
        "threshold": threshold,
        "tool_presence_accuracy": correct / len(scores),
        "negative_examples": negatives,
        "false_positives": false_positives,
        "false_positive_rate": false_positives / negatives,
        "positive_predictions": sum(predicted),
    }


def select_threshold(scores, fpr_target):
    # Include an all-No option and score every actual decision boundary.
    candidates = [max(score["margin"] for score in scores) + 1.0]
    candidates.extend(sorted({score["margin"] for score in scores}, reverse=True))
    feasible = [threshold_metrics(scores, threshold) for threshold in candidates]
    feasible = [metric for metric in feasible if metric["false_positive_rate"] <= fpr_target]
    if not feasible:
        raise RuntimeError("No threshold satisfies the requested validation FPR target")
    # Accuracy first; then retain as many positive decisions as possible.
    return max(feasible, key=lambda metric: (metric["tool_presence_accuracy"], metric["positive_predictions"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vqa-json", required=True, help="validation split only")
    parser.add_argument("--base-predictions", required=True, help="generated predictions on that same split")
    parser.add_argument("--baseline-metrics", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--fixed-threshold",
        type=float,
        default=None,
        help="Use a threshold selected on another split; never tune on this split.",
    )
    parser.add_argument("--bertscore-model", default=str(DEFAULT_BERTSCORE_MODEL))
    args = parser.parse_args()

    assert_gpu_idle()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    items = json.load(open(args.vqa_json, encoding="utf-8"))
    base_rows = json.load(open(args.base_predictions, encoding="utf-8"))
    base_by_id = {row["id"]: row for row in base_rows}
    if {item["id"] for item in items} != set(base_by_id):
        raise RuntimeError("--base-predictions must contain exactly the validation IDs")
    baseline = json.load(open(args.baseline_metrics, encoding="utf-8"))
    baseline_fpr = baseline["semantic"]["tool_presence"]["false_positive_rate"]
    fpr_target = baseline_fpr - 0.01

    model, tokenizer, processor, qwen_num_frames = load_model(args.checkpoint)
    dataset = DualEncoderQADataset(args.vqa_json, args.cache_root, is_train=False, qwen_num_frames=qwen_num_frames)
    # Qwen's 8-frame video placeholder occupies roughly 800 tokens; keep the
    # verified 1200-token limit used by the DualEncoderVQA training pipeline.
    collator = DualEncoderCollator(tokenizer, processor, max_text_length=1200)
    scores = score_tool_presence(model, collator, dataset, items)
    if args.fixed_threshold is None:
        selected = select_threshold(scores, fpr_target)
        selection_method = "selected_on_this_split"
    else:
        selected = threshold_metrics(scores, args.fixed_threshold)
        selection_method = "fixed_from_another_split"
    score_by_id = {score["id"]: score for score in scores}
    threshold = selected["threshold"]
    rows = []
    for item in items:
        row = dict(base_by_id[item["id"]])
        if item.get("question_type") == "tool_presence":
            row["prediction"] = "Yes" if score_by_id[item["id"]]["margin"] >= threshold else "No"
        rows.append(row)

    json.dump(scores, open(output_dir / "tool_presence_scores.json", "w", encoding="utf-8"), indent=2)
    selection = {
        "split": str(Path(args.vqa_json).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "baseline_fpr": baseline_fpr,
        "fpr_target": fpr_target,
        "selection": selected,
    }
    json.dump(selection, open(output_dir / "threshold_selection.json", "w", encoding="utf-8"), indent=2)
    json.dump(rows, open(output_dir / "predictions.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(json.dumps(selection, indent=2))

    del model
    torch.cuda.empty_cache()
    evaluate_rows(rows, SimpleNamespace(bertscore_model=args.bertscore_model), output_dir)
    json.dump(rows, open(output_dir / "predictions.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
