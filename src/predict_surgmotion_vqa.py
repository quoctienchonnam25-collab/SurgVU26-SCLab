"""Inference and local public-sample evaluation for SurgMotionVQA checkpoints."""

import argparse
import json
import os
import subprocess
from pathlib import Path

import cv2
import decord
import numpy as np
import torch
from transformers import AutoTokenizer

from evaluate import compute_bertscore_f1, evaluate_predictions
from surgmotion_vqa_prototype import (
    DEFAULT_QWEN_MODEL,
    DEFAULT_SURGMOTION_CHECKPOINT,
    build,
)
from train_surgmotion_vqa import (
    BLACK_MEAN_THRESHOLD,
    BLACK_P99_THRESHOLD,
    IMAGE_MEAN,
    IMAGE_STD,
    MODEL_FRAME_SIZE,
    SurgMotionQADataset,
    load_component_checkpoint,
    resolve_video_path,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PUBLIC_SAMPLE = PROJECT_ROOT / "SURGVU25_cat_2_sample_set_public"
DEFAULT_BERTSCORE_MODEL = PROJECT_ROOT / "checkpoints" / "base_models" / "roberta-large"
GC_VIDEO = "endoscopic-robotic-surgery-video.mp4"
GC_QUESTION = "visual-context-question.json"
GC_RESPONSE = "visual-context-response.json"


def assert_gpu_idle():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SurgMotionVQA inference")
    rows = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    conflicts = [row for row in rows if row.strip() and "nxnode" not in row]
    if conflicts:
        raise RuntimeError("GPU is busy: " + "; ".join(conflicts))


def frame_is_valid(frame):
    active = frame[: int(frame.shape[0] * 0.85)]
    gray = cv2.cvtColor(active, cv2.COLOR_RGB2GRAY)
    return not (
        float(gray.mean()) <= BLACK_MEAN_THRESHOLD
        and float(np.percentile(gray, 99)) <= BLACK_P99_THRESHOLD
    )


def load_video_tensor(video_path, num_frames, t_start=None, t_end=None):
    reader = decord.VideoReader(str(video_path), num_threads=1)
    if len(reader) <= 0:
        raise RuntimeError(f"Empty video: {video_path}")
    if t_start is None or t_end is None:
        start_frame, end_frame = 0, len(reader) - 1
    else:
        fps = float(reader.get_avg_fps())
        start_frame = int(float(t_start) * fps)
        end_frame = max(start_frame, int(float(t_end) * fps) - 1)
        start_frame = min(max(start_frame, 0), len(reader) - 1)
        end_frame = min(max(end_frame, start_frame), len(reader) - 1)
    indices = np.linspace(start_frame, end_frame, num_frames, dtype=np.int64)
    frames = reader.get_batch(indices).asnumpy()
    mask_y = int(0.85 * frames.shape[1])
    frames[:, mask_y:, :, :] = 0
    frames = np.stack(
        [cv2.resize(frame, (MODEL_FRAME_SIZE, MODEL_FRAME_SIZE)) for frame in frames]
    )
    valid = np.asarray([frame_is_valid(frame) for frame in frames], dtype=np.bool_)
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size:
        for index in np.flatnonzero(~valid):
            nearest = valid_indices[np.argmin(np.abs(valid_indices - index))]
            frames[index] = frames[nearest]
    array = frames.astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(3, 0, 1, 2).contiguous()
    tensor = (tensor - IMAGE_MEAN) / IMAGE_STD
    return tensor.unsqueeze(0), torch.from_numpy(valid).unsqueeze(0)


def load_model(checkpoint, qwen_model=None, surgmotion_checkpoint=None):
    checkpoint = Path(checkpoint)
    config_path = checkpoint / "surgmotion_vqa_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.load(open(config_path, encoding="utf-8"))
    num_frames = int(config["num_frames"])
    qwen_model = qwen_model or config.get("qwen_model") or str(DEFAULT_QWEN_MODEL)
    surgmotion_checkpoint = (
        surgmotion_checkpoint
        or config.get("surgmotion_checkpoint")
        or str(DEFAULT_SURGMOTION_CHECKPOINT)
    )
    model = build(
        surgmotion_checkpoint=surgmotion_checkpoint,
        qwen_model_id=qwen_model,
        num_frames=num_frames,
        # FP16, not build()'s bf16 default (that default stays bf16 for training,
        # which relies on it too - see train_surgmotion_vqa.py): the submission GPU
        # is a T4 (Turing, compute capability 7.5), which has no Tensor Core
        # acceleration for bf16 matmuls (that requires Ampere sm_80+). FP16 is fully
        # Tensor Core-accelerated on Turing and uses identical VRAM.
        dtype=torch.float16,
        local_files_only=True,
        train_surgmotion_blocks=0,
        freeze_qwen=False,
        projector_tokens=int(config.get("projector_tokens", 64)),
    )
    load_component_checkpoint(model, checkpoint)
    tokenizer_path = checkpoint / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path if tokenizer_path.is_dir() else qwen_model),
        local_files_only=True,
    )
    tokenizer.padding_side = "right"
    model.to("cuda")
    model.eval()
    return model, tokenizer, num_frames


def answer_tensor(model, tokenizer, video, valid, question, max_new_tokens):
    user_turn = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(
        user_turn, tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    prompt_length = encoded["input_ids"].shape[1]
    eos_ids = [tokenizer.eos_token_id]
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0:
        eos_ids.append(im_end)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model.generate(
            video.to("cuda", non_blocking=True),
            encoded["input_ids"].to("cuda"),
            encoded["attention_mask"].to("cuda"),
            frame_valid_mask=valid.to("cuda"),
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_ids,
        )
    answer_ids = output[0, prompt_length:].detach().cpu()
    return tokenizer.decode(answer_ids, skip_special_tokens=True).strip()


def answer_question(
    model,
    tokenizer,
    video_path,
    question,
    num_frames,
    max_new_tokens,
    t_start=None,
    t_end=None,
):
    video, valid = load_video_tensor(video_path, num_frames, t_start=t_start, t_end=t_end)
    return answer_tensor(model, tokenizer, video, valid, question, max_new_tokens)


def public_cases(input_dir):
    for video_path in sorted(Path(input_dir).glob("case*/case*.mp4")):
        case_id = video_path.stem
        question_path = video_path.with_name(f"{case_id}_question.json")
        answer_path = video_path.with_name(f"{case_id}.json")
        if question_path.is_file():
            yield case_id, video_path, question_path, answer_path


def evaluate_rows(rows, args, output_dir):
    if not rows or not all(row["answers"] for row in rows):
        raise RuntimeError("Evaluation requested but reference answers are missing")
    predictions = [row["prediction"] for row in rows]
    bertscore, per_example_bert = compute_bertscore_f1(
        rows, predictions, model_type=args.bertscore_model
    )
    bleu, per_example_bleu = evaluate_predictions(rows, predictions)
    for row, bert, bleu_row in zip(rows, per_example_bert, per_example_bleu):
        row["bertscore_f1"] = bert
        row["bleu4"] = bleu_row
    metrics = {"bertscore_f1": bertscore, "bleu4": float(bleu), "examples": len(rows)}
    json.dump(metrics, open(output_dir / "metrics.json", "w", encoding="utf-8"), indent=2)
    print(json.dumps(metrics, indent=2))


def run_vqa_json(args, model, tokenizer, num_frames):
    items = json.load(open(args.vqa_json, encoding="utf-8"))
    if args.limit is not None:
        items = items[: args.limit]
    cache_root = args.cache_root or str(PROJECT_ROOT / f"frame_cache_{num_frames}fr")
    dataset = SurgMotionQADataset(
        args.vqa_json,
        cache_root,
        is_train=False,
        num_frames=num_frames,
    )
    rows = []
    for index, item in enumerate(items):
        sample = dataset[index]
        prediction = answer_tensor(
            model,
            tokenizer,
            sample["video"].unsqueeze(0),
            sample["frame_valid_mask"].unsqueeze(0),
            item["question"],
            args.max_new_tokens,
        )
        row = dict(item)
        row["prediction"] = prediction
        rows.append(row)
        print(f"[{index + 1}/{len(items)}] {item['id']} -> {prediction}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(output_dir / "predictions.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    json.dump(
        [row["prediction"] for row in rows],
        open(output_dir / "predictions_list.json", "w", encoding="utf-8"),
        indent=2,
        ensure_ascii=False,
    )
    if args.evaluate:
        evaluate_rows(rows, args, output_dir)
        json.dump(rows, open(output_dir / "predictions.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)


def run_public(args, model, tokenizer, num_frames):
    rows = []
    for case_id, video_path, question_path, answer_path in public_cases(args.input_dir):
        question = json.load(open(question_path, encoding="utf-8"))
        prediction = answer_question(
            model, tokenizer, video_path, question, num_frames, args.max_new_tokens
        )
        references = json.load(open(answer_path, encoding="utf-8")) if answer_path.is_file() else []
        rows.append(
            {
                "id": case_id,
                "video": str(video_path),
                "question": question,
                "answers": references,
                "prediction": prediction,
            }
        )
        print(f"[{case_id}] {question} -> {prediction}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(output_dir / "predictions.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    if args.evaluate:
        evaluate_rows(rows, args, output_dir)
        json.dump(rows, open(output_dir / "predictions.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)


def run_grand_challenge(args, model, tokenizer, num_frames):
    input_dir = Path(args.input_dir)
    question = json.load(open(input_dir / GC_QUESTION, encoding="utf-8"))
    prediction = answer_question(
        model,
        tokenizer,
        input_dir / GC_VIDEO,
        question,
        num_frames,
        args.max_new_tokens,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json.dump(prediction, open(output_dir / GC_RESPONSE, "w", encoding="utf-8"), ensure_ascii=False)
    print(prediction)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-dir", default=str(DEFAULT_PUBLIC_SAMPLE))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "runs" / "surgmotion_public"))
    parser.add_argument("--qwen-model", default=None)
    parser.add_argument("--surgmotion-checkpoint", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--grand-challenge", action="store_true")
    parser.add_argument("--vqa-json", default=None)
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--bertscore-model", default=str(DEFAULT_BERTSCORE_MODEL))
    return parser.parse_args()


def main():
    args = parse_args()
    assert_gpu_idle()
    model, tokenizer, num_frames = load_model(
        args.checkpoint, args.qwen_model, args.surgmotion_checkpoint
    )
    if args.grand_challenge:
        run_grand_challenge(args, model, tokenizer, num_frames)
    elif args.vqa_json:
        run_vqa_json(args, model, tokenizer, num_frames)
    else:
        run_public(args, model, tokenizer, num_frames)


if __name__ == "__main__":
    main()
