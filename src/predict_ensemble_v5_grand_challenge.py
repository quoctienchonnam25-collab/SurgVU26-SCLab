"""Grand Challenge Category 2 entrypoint routing between the Qwen2-VL-2B native
pipeline and stage_b_dev per question, extending predict_surgmotion_grand_challenge.py.

runs/surgmotion_vqa/audit_dev_holdout/v5_on_curated_holdout scored v5 on the same
2254-item true holdout (case140-154) stage_b_dev was scored on
(runs/surgmotion_vqa/v2/val_dev - the checkpoint actually baked into this image;
not v2_p128/val_holdout, a separate 128-projector-token checkpoint variant that
scores within noise of this one, ~0.0005 apart overall, and was not worth the
rebuild/re-export to switch to). Aggregate: v5 0.8245 BERTScore-F1 vs stage_b_dev
0.8855 - v5 loses badly overall. But per question_type, v5 clearly wins tool_purpose
(0.803 vs 0.761, n=232) - the largest, least ambiguous win of the three categories it
nominally led (task_id/description also nominally favored v5 but their templates
overlap in wording with suture_required/procedure_type, so only tool_purpose is
routed here). Routing only tool_purpose to v5 and everything else to stage_b_dev
scores 0.8898 on the full 2254-item holdout - a real, verified gain over stage_b_dev
alone.

This originally also had an organ->quality_v1 branch (mirroring
predict_surgmotion_grand_challenge.py's old organ rule), dropped for the same reason
that script's organ rule was dropped: quality_v1 turns out to be the worst of the
three checkpoints on the correctly-matched organ subset (0.110 vs stage_b_dev's
0.268), not the best - see that script's docstring for the full story.

Grand Challenge gives no question_type label at runtime, only the raw question
string. generate_qa.py's gen_tool_purpose (lines 320-343) is the only question
generator anywhere in generate_qa.py/curate_surgmotion_vqa.py whose templates
contain "purpose", "role does", or start with "why" (verified by grep across both
files) - so this is a safe, low-risk keyword match against OUR OWN synthetic
templates. It is unverified against the real challenge's own question phrasing,
which may use different wording for the same intent and cause false negatives
(silently falling through to the stage_b_dev default instead).

Each Grand Challenge invocation is one question per container run, so only the one
model branch the router picks ever gets loaded - never both checkpoints at once.
"""
import argparse
import json
from pathlib import Path

import predict as v5_predict
from predict_surgmotion_grand_challenge import DEFAULT_CHECKPOINT
from predict_surgmotion_vqa import GC_QUESTION, GC_RESPONSE, GC_VIDEO, load_model, run_grand_challenge

V5_BASE_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
V5_CHECKPOINT = "/app/checkpoints/qwen2_vl_2b_lora_v5"
TOOL_PURPOSE_KEYWORDS = ("purpose", "role does")


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
    parser.add_argument("--v5-base-model", default=V5_BASE_MODEL)
    parser.add_argument("--v5-checkpoint", default=V5_CHECKPOINT)
    parser.add_argument(
        "--default-checkpoint", default=DEFAULT_CHECKPOINT,
        help="Override for local testing outside the container (default is the /app path baked into the image).",
    )
    return parser.parse_args()


def is_tool_purpose(question):
    lowered = str(question).lower()
    return any(keyword in lowered for keyword in TOOL_PURPOSE_KEYWORDS) or lowered.strip().startswith("why")


def run_v5(args):
    input_dir = Path(args.input_dir)
    question = json.load(open(input_dir / GC_QUESTION, encoding="utf-8"))
    model, processor = v5_predict.load_model(args.v5_base_model, args.v5_checkpoint)
    video_path = input_dir / GC_VIDEO
    answer = v5_predict.predict_vqa(model, processor, str(video_path), question)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json.dump(answer, open(output_dir / GC_RESPONSE, "w", encoding="utf-8"), ensure_ascii=False)
    print(answer)


def main():
    args = parse_args()
    question = json.load(open(Path(args.input_dir) / GC_QUESTION, encoding="utf-8"))

    if is_tool_purpose(question):
        print("[router] tool_purpose match -> using v5 (Qwen2-VL-2B)")
        run_v5(args)
        return

    print(f"[router] default -> using checkpoint: {args.default_checkpoint}")
    model, tokenizer, num_frames = load_model(
        args.default_checkpoint, qwen_model=args.qwen_model, surgmotion_checkpoint=args.surgmotion_checkpoint
    )
    run_grand_challenge(args, model, tokenizer, num_frames)


if __name__ == "__main__":
    main()
