"""Visual-ablation grounding audit for DualEncoderVQA checkpoints - isolates the
SurgMotion auxiliary branch's own causal contribution, on top of Qwen's own
(always-real, never-ablated) native video path.

Mirrors audit_surgmotion_visual_ablation.py's methodology (real vs forced-blank
on the same clip, stratified over visual_claim question types on visual_status
=="valid" segments) but only ablates the SurgMotion aux_frame_valid_mask, since
Qwen's own masked_scatter path has no equivalent "no visual" fallback to force -
its video is real in both conditions. This answers: does the SurgMotion branch
still add anything causally on top of Qwen's now-restored native vision, or did
the holdout gains (see runs/surgmotion_vqa/dual_encoder/val_holdout) come
entirely from Qwen's own vision tower?
"""
import argparse
import json
import random
from pathlib import Path

import torch

from evaluate import compute_bertscore_f1
from predict_dual_encoder_vqa import answer_tensor, assert_gpu_idle, load_model
from train_dual_encoder_vqa import DualEncoderQADataset, SURGMOTION_NUM_FRAMES

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def stratified_indices(items, question_types, per_type, seed):
    by_type = {}
    for index, item in enumerate(items):
        if item.get("visual_status") != "valid":
            continue
        qtype = item.get("question_type")
        if qtype not in question_types:
            continue
        by_type.setdefault(qtype, []).append(index)

    selected = []
    for qtype in question_types:
        candidates = sorted(by_type.get(qtype, []), key=lambda index: items[index]["id"])
        rng = random.Random(f"{seed}:{qtype}")
        rng.shuffle(candidates)
        selected.extend(candidates[:per_type])
    return sorted(selected)


def normalize(text):
    return " ".join(text.strip().lower().split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vqa-json", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--question-types", nargs="+", default=["tool_presence", "forceps_type", "tissue_cutting"]
    )
    parser.add_argument("--per-type", type=int, default=40)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--bertscore-model", default=str(PROJECT_ROOT / "checkpoints" / "base_models" / "roberta-large")
    )
    args = parser.parse_args()

    assert_gpu_idle()

    model, tokenizer, processor, qwen_num_frames = load_model(args.checkpoint)
    items = load_json(args.vqa_json)
    indices = stratified_indices(items, set(args.question_types), args.per_type, args.seed)
    print(f"[ablation] checkpoint={args.checkpoint} selected={len(indices)} items")

    dataset = DualEncoderQADataset(
        args.vqa_json, args.cache_root, is_train=False, qwen_num_frames=qwen_num_frames
    )

    rows = []
    for position, index in enumerate(indices):
        item = items[index]
        sample = dataset[index]
        qwen_frames = sample["qwen_frames"][:qwen_num_frames]
        aux_video = sample["aux_video"].unsqueeze(0)
        real_valid = sample["aux_frame_valid_mask"].unsqueeze(0)
        blank_valid = torch.zeros_like(real_valid)

        prediction_real = answer_tensor(
            model, tokenizer, processor, qwen_frames, aux_video, real_valid, item["question"], args.max_new_tokens
        )
        prediction_blank = answer_tensor(
            model, tokenizer, processor, qwen_frames, aux_video, blank_valid, item["question"], args.max_new_tokens
        )
        row = {
            "id": item["id"],
            "question_type": item["question_type"],
            "question": item["question"],
            "answers": item["answers"],
            "prediction_real": prediction_real,
            "prediction_blank_aux": prediction_blank,
            "changed": normalize(prediction_real) != normalize(prediction_blank),
        }
        rows.append(row)
        print(
            f"[{position + 1}/{len(indices)}] {item['id']} "
            f"real='{prediction_real}' blank_aux='{prediction_blank}' changed={row['changed']}"
        )

    real_preds = [row["prediction_real"] for row in rows]
    blank_preds = [row["prediction_blank_aux"] for row in rows]
    _, bert_real = compute_bertscore_f1(rows, real_preds, model_type=args.bertscore_model)
    _, bert_blank = compute_bertscore_f1(rows, blank_preds, model_type=args.bertscore_model)
    for row, real_score, blank_score in zip(rows, bert_real, bert_blank):
        row["bertscore_real"] = real_score
        row["bertscore_blank_aux"] = blank_score
        row["bertscore_delta"] = real_score - blank_score

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(output_dir / "ablation_predictions.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    def summarize(subset):
        if not subset:
            return None
        return {
            "examples": len(subset),
            "bertscore_real_mean": sum(row["bertscore_real"] for row in subset) / len(subset),
            "bertscore_blank_aux_mean": sum(row["bertscore_blank_aux"] for row in subset) / len(subset),
            "bertscore_delta_mean": sum(row["bertscore_delta"] for row in subset) / len(subset),
            "fraction_changed": sum(row["changed"] for row in subset) / len(subset),
        }

    metrics = {"overall": summarize(rows)}
    for qtype in args.question_types:
        metrics[qtype] = summarize([row for row in rows if row["question_type"] == qtype])
    json.dump(metrics, open(output_dir / "ablation_metrics.json", "w", encoding="utf-8"), indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
