import os

# Grand Challenge runs the evaluation container with --network none (confirmed via the
# official reference container's do_test_run.sh) - the model weights are already baked
# into the image at build time, so force offline mode rather than let huggingface_hub
# attempt (and hang/fail on) a revision-check network call at runtime.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import glob
import torch
import numpy as np
from PIL import Image
import decord
import cv2
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info
import argparse

def load_model(base_model_path="Qwen/Qwen2-VL-2B-Instruct", lora_path=None):
    print("Loading model and processor...")
    processor = AutoProcessor.from_pretrained(base_model_path)
    processor.image_processor.min_pixels = 112 * 112
    processor.image_processor.max_pixels = 224 * 224
    
    # For inference, BF16/FP16 is perfect and consumes very little VRAM (<4GB)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    if lora_path and os.path.exists(lora_path):
        print(f"Loading LoRA adapters from {lora_path}...")
        model = PeftModel.from_pretrained(model, lora_path)
        
    model.eval()
    return model, processor

def predict_vqa(model, processor, video_path, question, max_frames=16, frame_size=384, detector=None, blind=False):
    """blind=True feeds black frames instead of the real video - a language-prior
    diagnostic (inspired by SurgCheck's text-only ablation): if a question scores
    nearly as well blind as with real frames, the model is answering from a learned
    text pattern rather than actually looking at the clip."""
    if blind:
        pil_frames = [Image.new("RGB", (frame_size, frame_size), (0, 0, 0)) for _ in range(max_frames)]
        return _generate_answer(model, processor, pil_frames, question)

    if detector is not None:
        # Ground the prompt with what a dedicated detector actually sees in the clip -
        # trained specifically on the 14 tool classes, it catches things the VLM's own
        # learned tool-frequency prior gets wrong (see case122/132 on the public sample).
        # Sampled at a higher frame rate than the VQA model's own 8 frames, since a
        # miss traced to a tool only being visible for a couple of seconds at the edge
        # of frame - cheap to widen since detector inference is only a few ms/frame.
        try:
            from detect_tools import detect_tools_in_clip, format_detections_as_context
            vr_probe = decord.VideoReader(video_path)
            duration = len(vr_probe) / vr_probe.get_avg_fps()
            detections = detect_tools_in_clip(detector, video_path, 0, duration, n_frames=16)
            question = format_detections_as_context(detections) + " " + question
        except Exception as e:
            print(f"Detector context skipped for {video_path}: {e}")

    try:
        vr = decord.VideoReader(video_path)
        fps = vr.get_avg_fps()
        total_frames = len(vr)

        # Generate 30 frame indices uniformly distributed across the video
        frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        frames_npy = vr.get_batch(frame_indices).asnumpy()
        
        # Mask bottom 15% (UI overlay) to prevent cheating and ensure test robustness
        H, W = frames_npy.shape[1], frames_npy.shape[2]
        mask_y = int(0.85 * H)
        frames_npy[:, mask_y:, :, :] = 0
        
        pil_frames = []
        for f in frames_npy:
            f_resized = cv2.resize(f, (frame_size, frame_size))
            pil_frames.append(Image.fromarray(f_resized))
    except Exception as e:
        print(f"Error loading {video_path}: {e}")
        # Fallback to black frames
        pil_frames = [Image.new("RGB", (frame_size, frame_size), (0, 0, 0)) for _ in range(max_frames)]

    return _generate_answer(model, processor, pil_frames, question)


def _generate_answer(model, processor, pil_frames, question):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": pil_frames},
                {"type": "text", "text": question}
            ]
        }
    ]

    # Apply template and process vision info
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )

    # Move to GPU
    inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=50)

    # Trim prompt
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return output_text.strip()

# Grand Challenge's actual Category 2 interface (confirmed against the official
# reference container, github.com/isi-challenges/surgvu2025-category2-submission):
# the algorithm is invoked ONCE PER TEST CASE with these fixed socket filenames under
# /input, and must write exactly one fixed-name file under /output. This is NOT a
# "loop over a directory of cases" interface - grand-challenge mounts a fresh /input
# per case and calls the container fresh each time.
GC_VIDEO_SLUG = "endoscopic-robotic-surgery-video"
GC_QUESTION_SLUG = "visual-context-question"
GC_RESPONSE_SLUG = "visual-context-response"


def run_grand_challenge_case(input_dir, output_dir, model, processor, detector):
    """The real submission path: one video + one question in, one answer out."""
    video_path = os.path.join(input_dir, f"{GC_VIDEO_SLUG}.mp4")
    question_path = os.path.join(input_dir, f"{GC_QUESTION_SLUG}.json")

    with open(question_path, "r", encoding="utf-8") as f:
        question = json.load(f)

    print(f"Question: {question}")
    answer = predict_vqa(model, processor, video_path, question, detector=detector)
    print(f"Answer: {answer}")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{GC_RESPONSE_SLUG}.json"), "w", encoding="utf-8") as f:
        json.dump(answer, f, ensure_ascii=False)


def run_batch_directory(input_dir, output_dir, model, processor, detector):
    """Convenience path for local development only (NOT the grand-challenge contract):
    scans input_dir for however many case{N}.mp4 + case{N}_question.json pairs it
    finds (e.g. the public sample set) and predicts all of them in one process, so we
    don't have to re-load the model per case when comparing checkpoints locally."""
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

        with open(q_path, 'r', encoding='utf-8') as f:
            question = json.load(f)

        print(f"Predicting for {base_name}: Q='{question}'")
        pred_ans = predict_vqa(model, processor, video_path, question, detector=detector)
        print(f"Prediction: '{pred_ans}'")
        predictions[base_name] = pred_ans

        with open(os.path.join(output_dir, f"{base_name}.json"), 'w', encoding='utf-8') as f:
            json.dump(pred_ans, f, indent=4, ensure_ascii=False)

    with open(os.path.join(output_dir, "predictions.json"), 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=4, ensure_ascii=False)
    print(f"Predictions saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="/input", help="Input directory containing cases")
    parser.add_argument("--output_dir", type=str, default="/output", help="Output directory to save predictions")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="Path to base model")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA checkpoints")
    parser.add_argument("--detector_weights", type=str, default=None,
                         help="Path to a trained YOLO tool-detector .pt (see train_tool_detector.py). "
                              "Omit to run without detector-grounded context.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model, processor = load_model(args.model_path, args.lora_path)

    detector = None
    if args.detector_weights and os.path.exists(args.detector_weights):
        from detect_tools import load_detector
        print(f"Loading tool detector from {args.detector_weights}...")
        detector = load_detector(args.detector_weights)

    gc_video_path = os.path.join(args.input_dir, f"{GC_VIDEO_SLUG}.mp4")
    if os.path.exists(gc_video_path):
        run_grand_challenge_case(args.input_dir, args.output_dir, model, processor, detector)
    else:
        run_batch_directory(args.input_dir, args.output_dir, model, processor, detector)

if __name__ == "__main__":
    main()
