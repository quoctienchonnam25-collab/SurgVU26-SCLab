"""Grand Challenge Category 2 entrypoint for the SurgMotion VQA submission image.

Thin wrapper around predict_surgmotion_vqa's load_model/run_grand_challenge, always
using stage_b_dev. This used to route organ/anatomical questions to quality_v1
instead, based on the true-holdout audit in runs/surgmotion_vqa/audit_dev_holdout
(BERTScore-F1 0.268 vs 0.497). That comparison turned out to be invalid: quality_v1's
audit run was scored against a different, non-curated val file than stage_b_dev's,
so the two organ subsets shared zero items by content. Re-running quality_v1 on the
actual curated organ subset stage_b_dev was scored on
(runs/surgmotion_vqa/audit_dev_holdout/quality_v1_on_organ_curated_v2) gives
quality_v1 0.110 - the worst of the three checkpoints on that question type, not the
best. stage_b_dev alone (0.268) beats it, so there is no case left for routing organ
questions away from the default checkpoint.

Also skips assert_gpu_idle() (a guard against contending processes on a shared dev
machine, meaningless inside an isolated single-case Grand Challenge container) and
skips the local-audit argparse surface (--vqa-json/--evaluate/--limit/etc.) that has
no role in the actual submission contract.
"""
import argparse

from predict_surgmotion_vqa import load_model, run_grand_challenge

DEFAULT_CHECKPOINT = "/app/checkpoints/surgmotion_v2_stage_b_dev"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="/input")
    parser.add_argument("--output-dir", default="/output")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument(
        "--qwen-model", default="/app/checkpoints/base_models/Qwen2.5-VL-3B-Instruct"
    )
    parser.add_argument(
        "--surgmotion-checkpoint",
        default="/app/repo/SurgMotion-main/ckpts/SurgMotion-vitl.pt",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model, tokenizer, num_frames = load_model(
        DEFAULT_CHECKPOINT, qwen_model=args.qwen_model, surgmotion_checkpoint=args.surgmotion_checkpoint
    )
    run_grand_challenge(args, model, tokenizer, num_frames)


if __name__ == "__main__":
    main()
