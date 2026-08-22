"""Build DPO preference pairs from a visual-ablation audit
(audit_dual_encoder_visual_ablation.py's output).

Methodology: for a binary (Yes/No) question, a "shortcut" answer is one the
model still produces even when the SurgMotion aux branch sees no real video
(aux_frame_valid_mask all-zero, i.e. prediction_blank_aux) - it can only have
come from language prior / dataset bias, not visual grounding. Where that
blank-condition answer is WRONG (disagrees with ground truth) and the
real-video answer is CORRECT, we have direct evidence of what the shortcut
looks like and what it should be instead. DPO pair, both under the REAL video
input: chosen = ground-truth answer, rejected = the biased blank-condition
text. This trains the model to prefer the grounded answer over that specific
shortcut pattern even when the real video is present, rather than merely
hoping the aux branch's signal outweighs the prior at inference time.

Only "changed" rows can produce a pair (if blank and real predictions agree,
there is no shortcut/grounding contrast to learn from).
"""
import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def normalize_answer(text):
    return " ".join(str(text).strip().lower().split())


def polarity(text):
    normalized = normalize_answer(text)
    if normalized.startswith("yes"):
        return True
    if normalized.startswith("no"):
        return False
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-predictions", required=True, help="ablation_predictions.json from audit_dual_encoder_visual_ablation.py")
    parser.add_argument(
        "--question-types", nargs="+", default=["tool_presence", "tissue_cutting", "suture_required"]
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = json.load(open(args.ablation_predictions, encoding="utf-8"))
    pairs = []
    skipped_not_binary = 0
    skipped_unchanged = 0
    skipped_real_wrong = 0
    skipped_blank_correct = 0
    for row in rows:
        if row.get("question_type") not in args.question_types:
            continue
        if not row.get("changed"):
            skipped_unchanged += 1
            continue
        truth = polarity(row["answers"][0])
        real_pol = polarity(row["prediction_real"])
        blank_pol = polarity(row["prediction_blank_aux"])
        if truth is None or real_pol is None or blank_pol is None:
            skipped_not_binary += 1
            continue
        if real_pol != truth:
            skipped_real_wrong += 1
            continue
        if blank_pol == truth:
            skipped_blank_correct += 1
            continue
        pairs.append(
            {
                "id": row["id"],
                "question_type": row["question_type"],
                "question": row["question"],
                "chosen": row["answers"][0],
                "rejected": row["prediction_blank_aux"],
            }
        )

    print(
        f"[dpo-pairs] built {len(pairs)} pairs from {len(rows)} ablation rows "
        f"(skipped: not_binary={skipped_not_binary} unchanged={skipped_unchanged} "
        f"real_wrong={skipped_real_wrong} blank_already_correct={skipped_blank_correct})"
    )
    by_type = {}
    for pair in pairs:
        by_type[pair["question_type"]] = by_type.get(pair["question_type"], 0) + 1
    print(f"[dpo-pairs] per type: {by_type}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(pairs, open(output_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"[dpo-pairs] wrote {output_path}")


if __name__ == "__main__":
    main()
