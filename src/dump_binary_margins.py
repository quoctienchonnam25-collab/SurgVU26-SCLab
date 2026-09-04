"""Dump the first-token Yes/No logit margin for every binary holdout question.

calibrate_yes_no_thresholds.py reports only its own best threshold, so it cannot
answer "what is the accuracy at the threshold we actually ship?" without another
GPU pass. Saving the raw margins once makes every threshold question answerable
offline, forever - including the shipped values (tool_presence 0.2,
tissue_cutting 1.4, suture_required 0.3) and any future candidate.

Motivation (2026-09-03): BERTScore on binary questions is a pure linear function
of accuracy - wrong polarity scores 0.701, right scores 1.000, and 71.1% accuracy
predicts 0.9136 against a measured 0.9119. Binary questions are ~61% of the
holdout, so each +1pp of binary accuracy is worth roughly +0.0018 overall. That
makes the decision threshold one of the few remaining levers that needs no
retraining.
"""
import argparse
import json
from pathlib import Path

from predict_dual_encoder_vqa import (
    assert_gpu_idle,
    load_model,
    normalize_answer,
    tool_presence_answer_tensor,
)
from train_dual_encoder_vqa import DualEncoderQADataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BINARY_TYPES = ("tool_presence", "tissue_cutting", "suture_required")


def polarity(text):
    normalized = normalize_answer(text)
    if normalized.startswith("yes"):
        return True
    if normalized.startswith("no"):
        return False
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vqa-json", required=True)
    parser.add_argument("--cache-root", default=str(PROJECT_ROOT / "frame_cache_16fr"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    assert_gpu_idle()
    model, tokenizer, processor, qwen_num_frames = load_model(args.checkpoint)

    items = json.load(open(args.vqa_json, encoding="utf-8"))
    dataset = DualEncoderQADataset(
        args.vqa_json, args.cache_root, is_train=False, qwen_num_frames=qwen_num_frames
    )

    rows = []
    targets = [
        (index, item)
        for index, item in enumerate(items)
        if item.get("question_type") in BINARY_TYPES
        and polarity(item["answers"][0]) is not None
    ]
    for position, (index, item) in enumerate(targets):
        sample = dataset[index]
        _, margin = tool_presence_answer_tensor(
            model,
            tokenizer,
            processor,
            sample["qwen_frames"][:qwen_num_frames],
            sample["aux_video"].unsqueeze(0),
            sample["aux_frame_valid_mask"].unsqueeze(0),
            item["question"],
            0.0,  # placeholder: only the margin is used
        )
        rows.append({
            "id": item["id"],
            "question_type": item["question_type"],
            "margin": margin,
            "truth_yes": polarity(item["answers"][0]),
        })
        if (position + 1) % 100 == 0:
            print(f"[{position + 1}/{len(targets)}]", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(output, "w", encoding="utf-8"), indent=2)
    print(f"wrote {len(rows)} margins to {output}")


if __name__ == "__main__":
    main()
