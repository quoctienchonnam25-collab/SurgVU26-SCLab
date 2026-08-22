"""Plain Qwen2.5-VL-3B + LoRA inference - no SurgMotion branch, no dual-encoder
complexity. Adapted from predict.py (which targeted the older Qwen2-VL-2B-Instruct)
after real prelim leaderboard evidence (2026-08-17, see project memory
surgvu_combo_prelim_regression) showed BOTH architecture complexity (vision LoRA +
higher text LoRA rank) and a local-only-validated data change regressed hard on the
real hidden test set, while the current #1 leaderboard entry ("QwenVL LoRA, 5-frame")
uses exactly this minimal shape. This mirrors that: a straight LoRA fine-tune of
Qwen's own native video understanding, nothing added on top.
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import glob
import torch
import numpy as np
from PIL import Image
import decord
import cv2
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_BASE_MODEL = os.path.join(PROJECT_ROOT, "checkpoints", "base_models", "Qwen2.5-VL-3B-Instruct")


def load_model(base_model_path=DEFAULT_BASE_MODEL, lora_path=None):
    print("Loading model and processor...")
    processor = AutoProcessor.from_pretrained(base_model_path, local_files_only=True)
    processor.image_processor.min_pixels = 112 * 112
    processor.image_processor.max_pixels = 224 * 224

    # FP16 (not BF16): the submission GPU is a T4 (Turing, sm75) - no Tensor Core
    # acceleration for bf16 matmuls there (needs Ampere sm80+). Same reasoning as
    # predict_dual_encoder_vqa.py::load_model().
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        dtype=torch.float16,
        local_files_only=True,
        device_map="auto",
    )
    if lora_path and os.path.exists(lora_path):
        print(f"Loading LoRA adapters from {lora_path}...")
        model = PeftModel.from_pretrained(model, lora_path)

    model.eval()
    return model, processor


def predict_vqa(model, processor, video_path, question, max_frames=5, frame_size=384, blind=False):
    """blind=True feeds black frames instead of the real video - a language-prior
    diagnostic: if a question scores nearly as well blind as with real frames, the
    model is answering from a learned text pattern rather than actually looking."""
    if blind:
        pil_frames = [Image.new("RGB", (frame_size, frame_size), (0, 0, 0)) for _ in range(max_frames)]
        return _generate_answer(model, processor, pil_frames, question)

    try:
        vr = decord.VideoReader(video_path)
        total_frames = len(vr)
        frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        frames_npy = vr.get_batch(frame_indices).asnumpy()

        H, W = frames_npy.shape[1], frames_npy.shape[2]
        mask_y = int(0.85 * H)
        frames_npy[:, mask_y:, :, :] = 0

        pil_frames = []
        for f in frames_npy:
            f_resized = cv2.resize(f, (frame_size, frame_size))
            pil_frames.append(Image.fromarray(f_resized))
    except Exception as e:
        print(f"Error loading {video_path}: {e}")
        pil_frames = [Image.new("RGB", (frame_size, frame_size), (0, 0, 0)) for _ in range(max_frames)]

    return _generate_answer(model, processor, pil_frames, question)


def predict_vqa_clip(model, processor, video_path, question, t_start, t_end, max_frames=5, frame_size=384):
    """Same as predict_vqa but for a specific [t_start, t_end] window rather than the
    whole video - used by evaluate-on-holdout, where each QA item is one 30s segment."""
    try:
        vr = decord.VideoReader(video_path)
        fps = vr.get_avg_fps()
        start_frame = int(float(t_start) * fps)
        end_frame = max(start_frame, int(float(t_end) * fps) - 1)
        start_frame = min(max(start_frame, 0), len(vr) - 1)
        end_frame = min(max(end_frame, start_frame), len(vr) - 1)
        frame_indices = np.linspace(start_frame, end_frame, max_frames, dtype=int)
        frames_npy = vr.get_batch(frame_indices).asnumpy()

        H, W = frames_npy.shape[1], frames_npy.shape[2]
        mask_y = int(0.85 * H)
        frames_npy[:, mask_y:, :, :] = 0

        pil_frames = []
        for f in frames_npy:
            f_resized = cv2.resize(f, (frame_size, frame_size))
            pil_frames.append(Image.fromarray(f_resized))
    except Exception as e:
        print(f"Error loading {video_path} [{t_start},{t_end}]: {e}")
        pil_frames = [Image.new("RGB", (frame_size, frame_size), (0, 0, 0)) for _ in range(max_frames)]

    return _generate_answer(model, processor, pil_frames, question)


def _generate_answer(model, processor, pil_frames, question):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": pil_frames},
                {"type": "text", "text": question},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    )
    inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.no_grad():
        # do_sample=False (greedy) explicitly: Qwen2.5-VL-3B-Instruct's own
        # generation_config.json ships do_sample=True with temperature=1e-6 as a
        # "greedy-like" trick, but dividing logits by a near-zero temperature
        # before softmax overflows to inf/nan under FP16 (confirmed via a real
        # crash during local eval: "probability tensor contains inf, nan or
        # element < 0" -> CUDA device-side assert, killing the whole process).
        # True greedy decoding sidesteps the temperature/softmax path entirely.
        generated_ids = model.generate(**inputs, max_new_tokens=50, do_sample=False)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return output_text.strip()


# Grand Challenge's actual Category 2 interface - identical contract to
# predict_dual_encoder_vqa.py / the old predict.py.
GC_VIDEO_SLUG = "endoscopic-robotic-surgery-video"
GC_QUESTION_SLUG = "visual-context-question"
GC_RESPONSE_SLUG = "visual-context-response"


def run_grand_challenge_case(input_dir, output_dir, model, processor, max_frames):
    video_path = os.path.join(input_dir, f"{GC_VIDEO_SLUG}.mp4")
    question_path = os.path.join(input_dir, f"{GC_QUESTION_SLUG}.json")

    with open(question_path, "r", encoding="utf-8") as f:
        question = json.load(f)

    print(f"Question: {question}")
    answer = predict_vqa(model, processor, video_path, question, max_frames=max_frames)
    print(f"Answer: {answer}")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{GC_RESPONSE_SLUG}.json"), "w", encoding="utf-8") as f:
        json.dump(answer, f, ensure_ascii=False)


def run_batch_directory(input_dir, output_dir, model, processor, max_frames):
    """Local dev convenience path (NOT the grand-challenge contract)."""
    video_files = glob.glob(os.path.join(input_dir, "**/*.mp4"), recursive=True)
    print(f"Found {len(video_files)} video files for prediction.")

    predictions = {}
    for video_path in video_files:
        video_dir = os.path.dirname(video_path)
        base_name = os.path.splitext(os.path.basename(video_path))[0]

        q_path = os.path.join(video_dir, f"{base_name}_question.json")
        if not os.path.exists(q_path):
            q_path = os.path.join(video_dir, "question.json")
        if not os.path.exists(q_path):
            print(f"Warning: Question file not found for {video_path}. Skipping.")
            continue

        with open(q_path, "r", encoding="utf-8") as f:
            question = json.load(f)

        print(f"Predicting for {base_name}: Q='{question}'")
        pred_ans = predict_vqa(model, processor, video_path, question, max_frames=max_frames)
        print(f"Prediction: '{pred_ans}'")
        predictions[base_name] = pred_ans

        with open(os.path.join(output_dir, f"{base_name}.json"), "w", encoding="utf-8") as f:
            json.dump(pred_ans, f, indent=4, ensure_ascii=False)

    with open(os.path.join(output_dir, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=4, ensure_ascii=False)
    print(f"Predictions saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="/input")
    parser.add_argument("--output_dir", type=str, default="/output")
    parser.add_argument("--model_path", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--max_frames", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model, processor = load_model(args.model_path, args.lora_path)

    gc_video_path = os.path.join(args.input_dir, f"{GC_VIDEO_SLUG}.mp4")
    if os.path.exists(gc_video_path):
        run_grand_challenge_case(args.input_dir, args.output_dir, model, processor, args.max_frames)
    else:
        run_batch_directory(args.input_dir, args.output_dir, model, processor, args.max_frames)


if __name__ == "__main__":
    main()
