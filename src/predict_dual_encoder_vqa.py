"""Inference and local public-sample evaluation for DualEncoderVQA checkpoints.
Mirrors predict_surgmotion_vqa.py's structure; the two differences are (1) a
checkpoint config file has architecture="DualEncoderVQA" and additional
qwen_num_frames/qwen_max_pixels keys, and (2) generation needs both Qwen's own
processor output and a SurgMotion clip tensor per call.
"""

import argparse
import json
import os
import subprocess
import re
from pathlib import Path

import cv2
import decord
import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoTokenizer

from dual_encoder_vqa_prototype import AUX_NUM_TOKENS, DEFAULT_QWEN_MODEL, build
from evaluate import compute_bertscore_f1, evaluate_predictions
from generate_qa import ALL_PRESENCE_TARGETS
from surgmotion_vqa_prototype import DEFAULT_SURGMOTION_CHECKPOINT
from train_dual_encoder_vqa import SURGMOTION_NUM_FRAMES, load_component_checkpoint
from train_surgmotion_vqa import (
    BLACK_MEAN_THRESHOLD,
    BLACK_P99_THRESHOLD,
    IMAGE_MEAN,
    IMAGE_STD,
    MODEL_FRAME_SIZE,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PUBLIC_SAMPLE = PROJECT_ROOT / "SURGVU25_cat_2_sample_set_public"
DEFAULT_BERTSCORE_MODEL = PROJECT_ROOT / "checkpoints" / "base_models" / "roberta-large"
GC_VIDEO = "endoscopic-robotic-surgery-video.mp4"
GC_QUESTION = "visual-context-question.json"
GC_RESPONSE = "visual-context-response.json"

TOOL_PRESENCE_WORDS = (
    "listed", "installed", "installation record", "tool record",
    "used", "being used", "present", "involved",
)
TOOL_PRESENCE_NAMES = tuple(
    sorted((name.lower() for name in ALL_PRESENCE_TARGETS), key=len, reverse=True)
)


def is_tool_presence_question(question):
    normalized = normalize_answer(question)
    return any(name in normalized for name in TOOL_PRESENCE_NAMES) and any(
        word in normalized for word in TOOL_PRESENCE_WORDS
    )


def is_tissue_cutting_question(question):
    """Matches all 3 generate_qa.py::gen_tissue_cutting templates - only 2 of the
    3 contain "tissue", so "cutting" alone is also accepted."""
    normalized = normalize_answer(question)
    return "cutting" in normalized or ("tissue" in normalized and "dissect" in normalized)


def is_suture_required_question(question):
    normalized = normalize_answer(question)
    return "suture" in normalized and "required" in normalized

def assert_gpu_idle():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DualEncoderVQA inference")
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
    ignored_pids = {pid.strip() for pid in os.environ.get("SURGVU_GPU_GUARD_IGNORE_PIDS", "").split(",") if pid.strip()}
    conflicts = [
        row for row in rows
        if row.strip()
        and row.split(",", 1)[0].strip() not in ignored_pids
        and "nxnode" not in row.lower()
    ]
    if conflicts:
        raise RuntimeError("GPU is busy: " + "; ".join(conflicts))


def frame_is_valid(frame):
    active = frame[: int(frame.shape[0] * 0.85)]
    gray = cv2.cvtColor(active, cv2.COLOR_RGB2GRAY)
    return not (
        float(gray.mean()) <= BLACK_MEAN_THRESHOLD
        and float(np.percentile(gray, 99)) <= BLACK_P99_THRESHOLD
    )


def load_frames(video_path, num_frames, t_start=None, t_end=None):
    """Returns (pil_frames, aux_video[1,C,T,H,W], aux_valid[1,T]) for one segment."""
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

    resampling = getattr(Image, "Resampling", Image)
    pil_frames = [Image.fromarray(frame).convert("RGB") for frame in frames]

    resized = np.stack(
        [cv2.resize(frame, (MODEL_FRAME_SIZE, MODEL_FRAME_SIZE)) for frame in frames]
    )
    valid = np.asarray([frame_is_valid(frame) for frame in resized], dtype=np.bool_)
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size:
        for index in np.flatnonzero(~valid):
            nearest = valid_indices[np.argmin(np.abs(valid_indices - index))]
            resized[index] = resized[nearest]
    array = resized.astype(np.float32) / 255.0
    aux_video = torch.from_numpy(array).permute(3, 0, 1, 2).contiguous()
    aux_video = (aux_video - IMAGE_MEAN) / IMAGE_STD
    return pil_frames, aux_video.unsqueeze(0), torch.from_numpy(valid).unsqueeze(0)


def load_model(checkpoint, qwen_model=None, surgmotion_checkpoint=None):
    checkpoint = Path(checkpoint)
    config_path = checkpoint / "dual_encoder_vqa_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.load(open(config_path, encoding="utf-8"))
    qwen_model = qwen_model or config.get("qwen_model") or str(DEFAULT_QWEN_MODEL)
    surgmotion_checkpoint = (
        surgmotion_checkpoint
        or config.get("surgmotion_checkpoint")
        or str(DEFAULT_SURGMOTION_CHECKPOINT)
    )
    qwen_num_frames = int(config.get("qwen_num_frames", 8))
    qwen_max_pixels = int(config.get("qwen_max_pixels", 112 * 112))
    aux_num_tokens = int(config.get("aux_num_tokens", AUX_NUM_TOKENS))
    lora_r = int(config.get("lora_r", 16))
    vision_lora_r = int(config.get("vision_lora_r", 0))
    use_fusion = bool(config.get("use_fusion", False))
    fusion_heads = int(config.get("fusion_heads", 8))

    model = build(
        surgmotion_checkpoint=surgmotion_checkpoint,
        qwen_model_id=qwen_model,
        aux_num_frames=SURGMOTION_NUM_FRAMES,
        aux_num_tokens=aux_num_tokens,
        lora_r=lora_r,
        vision_lora_r=vision_lora_r,
        use_fusion=use_fusion,
        fusion_heads=fusion_heads,
        # FP16, not BF16 (build()'s own default): the submission GPU is a T4
        # (Turing, sm75). Flash attention needs Ampere+ (sm80) and is never
        # available there regardless of dtype; PyTorch's memory-efficient SDPA
        # backend also rejects BF16 on Turing (no native BF16 compute hardware
        # pre-Ampere). Combined, that left repo/SurgMotion-main's
        # restricted-backend SDPA call (src/models/utils/modules.py's
        # _SDPA_BACKENDS = [FLASH, EFFICIENT]) with nothing to dispatch to on a
        # real graded run: "RuntimeError: No available kernel. Aborting
        # execution." FP16 is fully supported by memory-efficient attention on
        # Turing and uses identical VRAM.
        dtype=torch.float16,
        local_files_only=True,
        train_surgmotion_blocks=0,
    )
    load_component_checkpoint(model, checkpoint)

    tokenizer_path = checkpoint / "tokenizer"
    tokenizer_source = tokenizer_path if tokenizer_path.is_dir() else qwen_model
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_source), local_files_only=True)
    tokenizer.padding_side = "right"
    processor = AutoProcessor.from_pretrained(qwen_model, local_files_only=True)
    processor.image_processor.min_pixels = qwen_max_pixels
    processor.image_processor.max_pixels = qwen_max_pixels

    model.to("cuda")
    model.eval()
    return model, tokenizer, processor, qwen_num_frames


def answer_tensor(model, tokenizer, processor, pil_frames, aux_video, aux_valid, question, max_new_tokens):
    user_turn = [
        {
            "role": "user",
            "content": [{"type": "video", "video": pil_frames}, {"type": "text", "text": question}],
        }
    ]
    prompt_text = processor.apply_chat_template(user_turn, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(user_turn)
    prompt_inputs = processor(
        text=[prompt_text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    )

    eos_ids = [tokenizer.eos_token_id]
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0:
        eos_ids.append(im_end)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output_ids = model.generate(
            prompt_ids=prompt_inputs["input_ids"].to("cuda"),
            pixel_values_videos=prompt_inputs["pixel_values_videos"].to("cuda"),
            video_grid_thw=prompt_inputs["video_grid_thw"].to("cuda"),
            aux_video=aux_video.to("cuda"),
            aux_frame_valid_mask=aux_valid.to("cuda"),
            second_per_grid_ts=prompt_inputs.get("second_per_grid_ts"),
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_ids,
        )
    return tokenizer.decode(output_ids[0].detach().cpu(), skip_special_tokens=True).strip()

def tool_presence_answer_tensor(model, tokenizer, processor, pil_frames, aux_video, aux_valid, question, threshold):
    """Conservative Yes/No decision from the calibrated first-token margin."""
    user_turn = [
        {"role": "user", "content": [{"type": "video", "video": pil_frames}, {"type": "text", "text": question}]}
    ]
    prompt_text = processor.apply_chat_template(user_turn, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(user_turn)
    prompt_inputs = processor(
        text=[prompt_text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    )
    yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
    no_ids = tokenizer.encode("No", add_special_tokens=False)
    if not yes_ids or not no_ids:
        raise RuntimeError("Tokenizer cannot encode Yes/No for tool-presence calibration")
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(
            prompt_ids=prompt_inputs["input_ids"].to("cuda"),
            pixel_values_videos=prompt_inputs["pixel_values_videos"].to("cuda"),
            video_grid_thw=prompt_inputs["video_grid_thw"].to("cuda"),
            aux_video=aux_video.to("cuda"),
            aux_frame_valid_mask=aux_valid.to("cuda"),
            second_per_grid_ts=prompt_inputs.get("second_per_grid_ts"),
        )["logits"][0, -1].float()
    margin = float(logits[yes_ids[0]] - logits[no_ids[0]])
    return ("Yes" if margin >= threshold else "No"), margin



def answer_question(
    model, tokenizer, processor, video_path, question, qwen_num_frames, max_new_tokens,
    t_start=None, t_end=None, tool_presence_threshold=None,
    tissue_cutting_threshold=None, suture_required_threshold=None,
):
    pil_frames, aux_video, aux_valid = load_frames(
        video_path, SURGMOTION_NUM_FRAMES, t_start=t_start, t_end=t_end
    )
    stride = SURGMOTION_NUM_FRAMES // qwen_num_frames
    qwen_frames = pil_frames[::stride][:qwen_num_frames]

    # tool_presence_answer_tensor is question-agnostic (just a calibrated Yes/No
    # margin decision), so it's reused for all 3 binary question types here -
    # each with its own separately-calibrated threshold, since the ablation
    # evidence showed tissue_cutting's bias runs opposite to tool_presence's.
    threshold = None
    if tool_presence_threshold is not None and is_tool_presence_question(question):
        threshold = tool_presence_threshold
    elif tissue_cutting_threshold is not None and is_tissue_cutting_question(question):
        threshold = tissue_cutting_threshold
    elif suture_required_threshold is not None and is_suture_required_question(question):
        threshold = suture_required_threshold

    if threshold is not None:
        prediction, _ = tool_presence_answer_tensor(
            model, tokenizer, processor, qwen_frames, aux_video, aux_valid, question, threshold
        )
        return prediction
    return answer_tensor(model, tokenizer, processor, qwen_frames, aux_video, aux_valid, question, max_new_tokens)


def public_cases(input_dir):
    for video_path in sorted(Path(input_dir).glob("case*/case*.mp4")):
        case_id = video_path.stem
        question_path = video_path.with_name(f"{case_id}_question.json")
        answer_path = video_path.with_name(f"{case_id}.json")
        if question_path.is_file():
            yield case_id, video_path, question_path, answer_path

def normalize_answer(text):
    return re.sub(r"\s+", " ", str(text).strip().lower())


def answer_polarity(text):
    normalized = normalize_answer(text)
    if normalized.startswith("yes"):
        return True
    if normalized.startswith("no"):
        return False
    return None


FORCEPS_LABELS = ("cadiere forceps", "bipolar forceps", "prograsp forceps")


def forceps_set(text):
    normalized = normalize_answer(text)
    return {label for label in FORCEPS_LABELS if label in normalized}


def compute_semantic_metrics(rows):
    presence_rows = [row for row in rows if row.get("question_type") == "tool_presence"]
    presence_correct = sum(
        answer_polarity(row["prediction"]) == answer_polarity(row["answers"][0])
        for row in presence_rows
    )
    negative_rows = [
        row for row in presence_rows if answer_polarity(row["answers"][0]) is False
    ]
    false_positives = sum(
        answer_polarity(row["prediction"]) is True for row in negative_rows
    )

    forceps_rows = [row for row in rows if row.get("question_type") == "forceps_type"]
    exact = 0
    true_positive = false_positive = false_negative = 0
    for row in forceps_rows:
        expected = forceps_set(" ".join(row["answers"]))
        predicted = forceps_set(row["prediction"])
        exact += predicted == expected
        true_positive += len(predicted.intersection(expected))
        false_positive += len(predicted.difference(expected))
        false_negative += len(expected.difference(predicted))
    denominator = 2 * true_positive + false_positive + false_negative
    forceps_f1 = 2 * true_positive / denominator if denominator else 1.0

    return {
        "tool_presence": {
            "examples": len(presence_rows),
            "accuracy": presence_correct / len(presence_rows) if presence_rows else None,
            "negative_examples": len(negative_rows),
            "false_positives": false_positives,
            "false_positive_rate": (
                false_positives / len(negative_rows) if negative_rows else None
            ),
        },
        "forceps_type": {
            "examples": len(forceps_rows),
            "set_exact_match": exact / len(forceps_rows) if forceps_rows else None,
            "set_micro_f1": forceps_f1 if forceps_rows else None,
        },
    }


def evaluate_rows(rows, args, output_dir):
    if not rows or not all(row["answers"] for row in rows):
        raise RuntimeError("Evaluation requested but reference answers are missing")
    predictions = [row["prediction"] for row in rows]
    bertscore, per_example_bert = compute_bertscore_f1(rows, predictions, model_type=args.bertscore_model)
    bleu, per_example_bleu = evaluate_predictions(rows, predictions)
    for row, bert, bleu_row in zip(rows, per_example_bert, per_example_bleu):
        row["bertscore_f1"] = bert
        row["bleu4"] = bleu_row
    metrics = {
        "bertscore_f1": float(bertscore),
        "bleu4": float(bleu),
        "examples": len(rows),
        "semantic": compute_semantic_metrics(rows),
    }
    per_question_type = {}
    for qtype in sorted({row.get("question_type", "unknown") for row in rows}):
        indices = [
            index
            for index, row in enumerate(rows)
            if row.get("question_type", "unknown") == qtype
        ]
        per_question_type[qtype] = {
            "examples": len(indices),
            "bertscore_f1": float(np.mean([per_example_bert[index] for index in indices])),
            "bleu4": float(np.mean([per_example_bleu[index] for index in indices])),
        }
    metrics["per_question_type"] = per_question_type
    json.dump(metrics, open(output_dir / "metrics.json", "w", encoding="utf-8"), indent=2)
    print(json.dumps(metrics, indent=2))


def run_vqa_json(args, model, tokenizer, processor, qwen_num_frames):
    items = json.load(open(args.vqa_json, encoding="utf-8"))
    if args.limit is not None:
        items = items[: args.limit]

    from train_dual_encoder_vqa import DualEncoderQADataset

    dataset = DualEncoderQADataset(
        args.vqa_json, args.cache_root, is_train=False, qwen_num_frames=qwen_num_frames
    )

    rows = []
    for index, item in enumerate(items):
        sample = dataset[index]
        stride = SURGMOTION_NUM_FRAMES // qwen_num_frames
        qwen_frames = sample["qwen_frames"][:qwen_num_frames]
        prediction = answer_tensor(
            model,
            tokenizer,
            processor,
            qwen_frames,
            sample["aux_video"].unsqueeze(0),
            sample["aux_frame_valid_mask"].unsqueeze(0),
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
    if args.evaluate:
        evaluate_rows(rows, args, output_dir)
        json.dump(rows, open(output_dir / "predictions.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)


def run_public(args, model, tokenizer, processor, qwen_num_frames):
    rows = []
    for case_id, video_path, question_path, answer_path in public_cases(args.input_dir):
        question = json.load(open(question_path, encoding="utf-8"))
        prediction = answer_question(
            model, tokenizer, processor, video_path, question, qwen_num_frames, args.max_new_tokens,
            tool_presence_threshold=args.tool_presence_threshold,
            tissue_cutting_threshold=args.tissue_cutting_threshold,
            suture_required_threshold=args.suture_required_threshold,
        )
        references = json.load(open(answer_path, encoding="utf-8")) if answer_path.is_file() else []
        rows.append(
            {"id": case_id, "video": str(video_path), "question": question, "answers": references, "prediction": prediction}
        )
        print(f"[{case_id}] {question} -> {prediction}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(output_dir / "predictions.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    if args.evaluate:
        evaluate_rows(rows, args, output_dir)
        json.dump(rows, open(output_dir / "predictions.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)


def run_grand_challenge(args, model, tokenizer, processor, qwen_num_frames):
    input_dir = Path(args.input_dir)
    question = json.load(open(input_dir / GC_QUESTION, encoding="utf-8"))
    prediction = answer_question(
        model, tokenizer, processor, input_dir / GC_VIDEO, question, qwen_num_frames, args.max_new_tokens,
        tool_presence_threshold=args.tool_presence_threshold,
        tissue_cutting_threshold=args.tissue_cutting_threshold,
        suture_required_threshold=args.suture_required_threshold,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json.dump(prediction, open(output_dir / GC_RESPONSE, "w", encoding="utf-8"), ensure_ascii=False)
    print(prediction)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-dir", default=str(DEFAULT_PUBLIC_SAMPLE))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "runs" / "dual_encoder_vqa"))
    parser.add_argument("--qwen-model", default=None)
    parser.add_argument("--surgmotion-checkpoint", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--grand-challenge", action="store_true")
    parser.add_argument("--vqa-json", default=None)
    parser.add_argument("--cache-root", default=str(PROJECT_ROOT / "frame_cache_16fr"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--bertscore-model", default=str(DEFAULT_BERTSCORE_MODEL))
    parser.add_argument("--tool-presence-threshold", type=float, default=None)
    parser.add_argument("--tissue-cutting-threshold", type=float, default=None)
    parser.add_argument("--suture-required-threshold", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    assert_gpu_idle()
    model, tokenizer, processor, qwen_num_frames = load_model(args.checkpoint, args.qwen_model, args.surgmotion_checkpoint)
    if args.grand_challenge:
        run_grand_challenge(args, model, tokenizer, processor, qwen_num_frames)
    elif args.vqa_json:
        run_vqa_json(args, model, tokenizer, processor, qwen_num_frames)
    else:
        run_public(args, model, tokenizer, processor, qwen_num_frames)


if __name__ == "__main__":
    main()
