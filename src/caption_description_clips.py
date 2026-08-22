"""Plan B: per-clip short captions for the `description` question type,
generated locally by base Qwen2.5-VL-3B-Instruct (no LoRA, no SurgMotion aux
branch - deliberately not the fine-tuned dual-encoder model, to avoid baking
its own biases/mode-collapse tendencies into the captions it's meant to
diversify).

Motivation: after the 2026-08-13 short-answer-length fix
(gen_description_question() in generate_qa.py), `description` answers are
still derived from only ~15-20 unique per-task `matched_description` texts
reused verbatim across every window that overlaps that task - genuinely
different clips within the same task get byte-identical target text. This
script generates a real per-window caption instead, to be blended in as
additional reference-answer diversity (not a wholesale replacement of the
human-authored text - see README/memory for that decision).

Run in --pilot mode first (small stratified sample, prints results for
manual review) before --scale (batch over every description-tagged window).
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import decord
import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QWEN_MODEL = PROJECT_ROOT / "checkpoints" / "base_models" / "Qwen2.5-VL-3B-Instruct"
DEFAULT_SEGMENTS = PROJECT_ROOT / "src" / "surgvu_segments_v2.json"
QWEN_NUM_FRAMES = 8
QWEN_MAX_PIXELS = 12544

DESCRIPTION_QUESTIONS = {
    "What is the surgeon doing in this step?",
    "Describe what is happening in this surgical clip.",
    "What activity is the surgeon performing?",
}

# Few-shot exemplars anchor style/length to the official public-sample
# register (short, factual, single sentence) - not the training corpus's own
# previous long-form answers, which is exactly what we're trying to move
# away from.
FEW_SHOT = [
    ("What is the surgeon doing in this step?", "The surgeon is dissecting tissue with monopolar scissors."),
    ("Describe what is happening in this surgical clip.", "The surgeon is placing a running suture with a needle driver."),
    ("What activity is the surgeon performing?", "The surgeon is cauterizing a vessel with bipolar forceps."),
]


def load_frames(video_path, num_frames, t_start, t_end):
    reader = decord.VideoReader(str(video_path), num_threads=1)
    fps = float(reader.get_avg_fps())
    start_frame = int(float(t_start) * fps)
    end_frame = max(start_frame, int(float(t_end) * fps) - 1)
    start_frame = min(max(start_frame, 0), len(reader) - 1)
    end_frame = min(max(end_frame, start_frame), len(reader) - 1)
    indices = np.linspace(start_frame, end_frame, num_frames, dtype=np.int64)
    frames = reader.get_batch(indices).asnumpy()
    mask_y = int(0.85 * frames.shape[1])
    frames[:, mask_y:, :, :] = 0
    return [Image.fromarray(frame).convert("RGB") for frame in frames]


def build_segment_lookup(segments_path):
    segments = json.load(open(segments_path, encoding="utf-8"))
    lookup = {}
    for seg in segments:
        key = (seg["video_path"], round(float(seg["t_start"]), 1), round(float(seg["t_end"]), 1))
        lookup[key] = seg
    return lookup


def caption_clip(model, processor, pil_frames, question, task_name):
    few_shot_text = "\n".join(f'Q: "{q}" -> A: "{a}"' for q, a in FEW_SHOT)
    instruction = (
        "You are captioning a 30-second clip from a surgical training video "
        f"(task: {task_name}). Answer in ONE short factual sentence describing "
        "the visible surgical activity - tools, tissue, and action only. Do not "
        "repeat the task name verbatim; describe what you actually see. Match "
        f"this style:\n{few_shot_text}\n\nQ: \"{question}\""
    )
    user_turn = [
        {"role": "user", "content": [{"type": "video", "video": pil_frames}, {"type": "text", "text": instruction}]}
    ]
    prompt_text = processor.apply_chat_template(user_turn, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(user_turn)
    inputs = processor(
        text=[prompt_text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to("cuda")
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        output_ids = model.generate(**inputs, max_new_tokens=48, do_sample=False)
    generated = output_ids[:, inputs["input_ids"].shape[1] :]
    text = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    # Keep only the first sentence, matching the short-answer register.
    for terminator in (".", "!", "?"):
        if terminator in text:
            text = text.split(terminator)[0] + terminator
            break
    return text


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vqa-json", required=True, help="e.g. src/train_vqa_v3.json or src/heldout_vqa_v3.json")
    parser.add_argument("--segments", default=str(DEFAULT_SEGMENTS))
    parser.add_argument("--qwen-model", default=str(DEFAULT_QWEN_MODEL))
    parser.add_argument("--output", required=True)
    parser.add_argument("--pilot-n", type=int, default=0, help="if >0, stratified sample this many by task instead of all")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    items = json.load(open(args.vqa_json, encoding="utf-8"))
    description_items = [item for item in items if item["question"] in DESCRIPTION_QUESTIONS]
    print(f"[caption] {len(description_items)} description QA items in {args.vqa_json}")

    lookup = build_segment_lookup(args.segments)

    random.seed(args.seed)
    if args.pilot_n:
        by_task = defaultdict(list)
        for item in description_items:
            key = (item["video"], round(float(item["t_start"]), 1), round(float(item["t_end"]), 1))
            seg = lookup.get(key)
            task = seg["task"] if seg else "UNKNOWN"
            by_task[task].append(item)
        tasks = sorted(by_task)
        selected = []
        idx = 0
        while len(selected) < args.pilot_n and any(by_task.values()):
            task = tasks[idx % len(tasks)]
            if by_task[task]:
                selected.append(by_task[task].pop(random.randrange(len(by_task[task]))))
            idx += 1
        description_items = selected
        print(f"[caption] pilot mode: sampled {len(description_items)} items across {len(tasks)} tasks")

    processor = AutoProcessor.from_pretrained(args.qwen_model, local_files_only=True)
    processor.image_processor.min_pixels = QWEN_MAX_PIXELS
    processor.image_processor.max_pixels = QWEN_MAX_PIXELS
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.qwen_model, dtype=torch.float16, local_files_only=True
    )
    model.to("cuda")
    model.eval()

    results = []
    for position, item in enumerate(description_items):
        key = (item["video"], round(float(item["t_start"]), 1), round(float(item["t_end"]), 1))
        seg = lookup.get(key)
        task_name = seg["task"] if seg else "unknown"
        original_desc = seg["description"] if seg else None
        pil_frames = load_frames(item["video"], QWEN_NUM_FRAMES, item["t_start"], item["t_end"])
        caption = caption_clip(model, processor, pil_frames, item["question"], task_name)
        results.append(
            {
                "id": item["id"],
                "video": item["video"],
                "t_start": item["t_start"],
                "t_end": item["t_end"],
                "task": task_name,
                "original_short_answer": item["answers"][0],
                "original_matched_description": original_desc,
                "generated_caption": caption,
            }
        )
        print(f"[{position + 1}/{len(description_items)}] task={task_name}")
        print(f"  original: {item['answers'][0]!r}")
        print(f"  generated: {caption!r}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(output_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"[caption] wrote {output_path}")


if __name__ == "__main__":
    main()
