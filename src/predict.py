import os
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

def predict_vqa(model, processor, video_path, question, max_frames=16, frame_size=384, detector=None):
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

    # Load model
    model, processor = load_model(args.model_path, args.lora_path)

    detector = None
    if args.detector_weights and os.path.exists(args.detector_weights):
        from detect_tools import load_detector
        print(f"Loading tool detector from {args.detector_weights}...")
        detector = load_detector(args.detector_weights)
    
    # Find all cases in the input directory
    # Grand challenge input might have files directly in /input/ or inside case subdirectories
    # We will search for any .mp4 files in input_dir and its subdirectories
    video_files = glob.glob(os.path.join(args.input_dir, "**/*.mp4"), recursive=True)
    
    print(f"Found {len(video_files)} video files for prediction.")
    
    predictions = {}
    
    for video_path in video_files:
        video_dir = os.path.dirname(video_path)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Try to find corresponding question file
        # It could be named like [base_name]_question.json or question.json
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
        
        # Save individual prediction as expected by grand-challenge
        # e.g., output_dir/case122.json containing a single string
        out_json_path = os.path.join(args.output_dir, f"{base_name}.json")
        with open(out_json_path, 'w', encoding='utf-8') as f:
            json.dump(pred_ans, f, indent=4, ensure_ascii=False)
            
    # Also save a global summary predictions.json
    with open(os.path.join(args.output_dir, "predictions.json"), 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=4, ensure_ascii=False)
        
    print(f"Predictions saved to {args.output_dir}")

if __name__ == "__main__":
    main()
