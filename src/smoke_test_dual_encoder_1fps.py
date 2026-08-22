"""Smoke-test DualEncoderVQA on public clips transcoded to challenge-like 1 FPS."""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import decord
import torch

from predict_dual_encoder_vqa import (
    SURGMOTION_NUM_FRAMES,
    answer_tensor,
    assert_gpu_idle,
    load_frames,
    load_model,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PUBLIC = PROJECT_ROOT / "SURGVU25_cat_2_sample_set_public"
DEFAULT_REPORT = (
    PROJECT_ROOT / "runs" / "surgmotion_vqa" / "dual_encoder_v3" / "smoke_1fps.json"
)
DEFAULT_CASES = ("case124", "case126", "case127")
GC_RESPONSE = "visual-context-response.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--public-root", default=str(DEFAULT_PUBLIC))
    parser.add_argument("--output-dir", default=str(DEFAULT_REPORT.parent / "smoke_1fps"))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--max-new-tokens", type=int, default=24)
    return parser.parse_args()


def transcode_one_fps(source, destination):
    environment = os.environ.copy()
    environment.pop("LD_LIBRARY_PATH", None)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vf",
            "fps=1",
            "-an",
            str(destination),
        ],
        check=True,
        env=environment,
    )


def main():
    args = parse_args()
    public_root = Path(args.public_root)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    report_rows = []

    assert_gpu_idle()
    model, tokenizer, processor, qwen_num_frames = load_model(args.checkpoint)
    if SURGMOTION_NUM_FRAMES % qwen_num_frames:
        raise RuntimeError("SurgMotion frame count is not divisible by Qwen frame count")

    with tempfile.TemporaryDirectory(prefix="surgvu_1fps_") as temporary:
        temporary_root = Path(temporary)
        for case_id in args.cases:
            source = public_root / case_id / f"{case_id}.mp4"
            question_path = source.with_name(f"{case_id}_question.json")
            if not source.is_file() or not question_path.is_file():
                raise FileNotFoundError(f"Missing public smoke case {case_id}")

            one_fps = temporary_root / f"{case_id}_1fps.mp4"
            transcode_one_fps(source, one_fps)
            reader = decord.VideoReader(str(one_fps), num_threads=1)
            fps = float(reader.get_avg_fps())
            frame_count = len(reader)
            if not 0.95 <= fps <= 1.05:
                raise RuntimeError(f"{case_id}: transcoded fps={fps}, expected 1")
            if not 25 <= frame_count <= 35:
                raise RuntimeError(
                    f"{case_id}: unexpected 1 FPS frame count {frame_count}"
                )

            pil_frames, aux_video, aux_valid = load_frames(
                one_fps, SURGMOTION_NUM_FRAMES
            )
            expected_video_shape = (1, 3, SURGMOTION_NUM_FRAMES, 256, 256)
            if tuple(aux_video.shape) != expected_video_shape:
                raise RuntimeError(
                    f"{case_id}: aux shape {tuple(aux_video.shape)} "
                    f"!= {expected_video_shape}"
                )
            if tuple(aux_valid.shape) != (1, SURGMOTION_NUM_FRAMES):
                raise RuntimeError(f"{case_id}: invalid frame mask shape")
            if not torch.isfinite(aux_video).all():
                raise RuntimeError(f"{case_id}: non-finite auxiliary tensor")

            stride = SURGMOTION_NUM_FRAMES // qwen_num_frames
            qwen_frames = pil_frames[::stride][:qwen_num_frames]
            if len(qwen_frames) != qwen_num_frames:
                raise RuntimeError(f"{case_id}: incorrect Qwen frame count")

            question = json.load(open(question_path, encoding="utf-8"))
            prediction = answer_tensor(
                model,
                tokenizer,
                processor,
                qwen_frames,
                aux_video,
                aux_valid,
                question,
                args.max_new_tokens,
            )
            if not isinstance(prediction, str) or not prediction.strip():
                raise RuntimeError(f"{case_id}: empty model response")

            case_output = output_root / case_id
            case_output.mkdir(parents=True, exist_ok=True)
            response_path = case_output / GC_RESPONSE
            with open(response_path, "w", encoding="utf-8") as handle:
                json.dump(prediction, handle, ensure_ascii=False)
            loaded_response = json.load(open(response_path, encoding="utf-8"))
            if not isinstance(loaded_response, str) or not loaded_response.strip():
                raise RuntimeError(f"{case_id}: invalid Grand Challenge response JSON")

            row = {
                "case": case_id,
                "source": str(source),
                "fps": fps,
                "frames": frame_count,
                "question": question,
                "prediction": prediction,
                "response": str(response_path),
                "aux_shape": list(aux_video.shape),
                "qwen_frames": len(qwen_frames),
                "valid_frames": int(aux_valid.sum().item()),
            }
            report_rows.append(row)
            print(json.dumps(row, ensure_ascii=False))

    report = {
        "status": "pass",
        "checkpoint": str(args.checkpoint),
        "cases": report_rows,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
