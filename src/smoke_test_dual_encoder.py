"""Structural + trainability smoke test for DualEncoderVQA, mirroring the
validation methodology surgmotion_vqa_prototype.py's own docstring describes
(a handful of real examples, several optimizer steps, check loss goes down).
Not part of the training pipeline - a one-off diagnostic to run before
committing to train_dual_encoder_vqa.py and real GPU hours.
"""
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dual_encoder_vqa_prototype import DEFAULT_QWEN_MODEL, build
from train_surgmotion_vqa import (
    IMAGE_MEAN,
    IMAGE_STD,
    MODEL_FRAME_SIZE,
    NUM_FRAMES,
    cache_dir_for,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_ROOT = PROJECT_ROOT / "frame_cache_16fr"
TRAIN_JSON = PROJECT_ROOT / "src" / "train_vqa_surgmotion_curated_v2.json"


def load_pil_frames(item, num_frames=NUM_FRAMES):
    folder = cache_dir_for(item, CACHE_ROOT)
    frames = []
    for index in range(num_frames):
        with Image.open(folder / f"frame_{index}.jpg") as image:
            frames.append(image.convert("RGB").copy())
    return frames


def load_surgmotion_tensor(pil_frames):
    import numpy as np

    arrays = []
    for frame in pil_frames:
        resized = frame.resize((MODEL_FRAME_SIZE, MODEL_FRAME_SIZE), Image.BILINEAR)
        arrays.append(np.asarray(resized, dtype=np.uint8))
    array = np.stack(arrays, axis=0).astype("float32") / 255.0
    tensor = torch.from_numpy(array).permute(3, 0, 1, 2).contiguous()
    tensor = (tensor - IMAGE_MEAN) / IMAGE_STD
    valid = torch.ones(len(pil_frames), dtype=torch.bool)
    return tensor.unsqueeze(0), valid.unsqueeze(0)


def encode_conversation(tokenizer, processor, pil_frames, question, answer):
    user_turn = [{"role": "user", "content": [{"type": "video", "video": pil_frames}, {"type": "text", "text": question}]}]
    full_turns = user_turn + [{"role": "assistant", "content": answer}]

    prompt_text = processor.apply_chat_template(user_turn, tokenize=False, add_generation_prompt=True)
    full_text = processor.apply_chat_template(full_turns, tokenize=False, add_generation_prompt=False)

    # Run BOTH through the processor with the identical video content, so the
    # <video> placeholder expands into the same number of video_token_id repeats
    # (based on grid_thw) in both - a plain tokenizer() call on full_text would
    # instead see the raw, unexpanded placeholder and never match as a prefix.
    image_inputs, video_inputs = process_vision_info(user_turn)
    prompt_inputs = processor(
        text=[prompt_text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    )
    full_inputs = processor(
        text=[full_text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    )

    prompt_ids_list = prompt_inputs["input_ids"][0].tolist()
    full_ids_list = full_inputs["input_ids"][0].tolist()
    if full_ids_list[: len(prompt_ids_list)] != prompt_ids_list:
        raise RuntimeError("chat template prompt is not a prefix of the full conversation")
    answer_ids_list = full_ids_list[len(prompt_ids_list):]

    answer_ids = torch.tensor([answer_ids_list], dtype=torch.long)
    labels = answer_ids.clone()
    return prompt_inputs, answer_ids, labels


def main():
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_QWEN_MODEL, local_files_only=True)
    processor = AutoProcessor.from_pretrained(DEFAULT_QWEN_MODEL, local_files_only=True)
    # Unconstrained, Qwen's own processor uses a much higher native resolution/token
    # budget than v5/v6/v7 ever ran with - those pipelines cap this explicitly
    # (src/train.py, src/predict.py) for exactly this VRAM reason. Match that here;
    # Qwen's vision tower is only meant to supply general scene context in this
    # architecture, not maximum resolution.
    processor.image_processor.min_pixels = 112 * 112
    processor.image_processor.max_pixels = 112 * 112

    data = json.load(open(TRAIN_JSON, encoding="utf-8"))
    examples = []
    for item in data:
        if item.get("question_type") in ("tool_presence", "forceps_type") and item.get("visual_status") == "valid":
            examples.append(item)
        if len(examples) >= 4:
            break

    print(f"[smoke] using {len(examples)} real examples")
    model = build(aux_num_frames=NUM_FRAMES, train_surgmotion_blocks=1, lora_r=16)
    model.to("cuda")
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable)
    print(f"[smoke] trainable params: {trainable_count:,}")
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)

    batches = []
    for item in examples:
        pil_frames = load_pil_frames(item)
        aux_video, aux_valid = load_surgmotion_tensor(pil_frames)
        answer = item["answers"][0]
        # Qwen's own vision tower only needs to supply coarse general scene context
        # here (SurgMotion covers the surgical-motion-specific signal) - give it far
        # fewer, lower-resolution frames than SurgMotion's own 16, matching how
        # small an auxiliary role it plays in this architecture.
        qwen_frames = pil_frames[::2]
        prompt_inputs, answer_ids, labels = encode_conversation(tokenizer, processor, qwen_frames, item["question"], answer)
        batches.append((prompt_inputs, answer_ids, labels, aux_video, aux_valid, item))

    for index, (prompt_inputs, *_rest) in enumerate(batches):
        print(
            f"[smoke] example {index}: video_grid_thw={prompt_inputs['video_grid_thw'].tolist()} "
            f"prompt_len={prompt_inputs['input_ids'].shape[1]}"
        )

    for step in range(40):
        item_index = step % len(batches)
        prompt_inputs, answer_ids, labels, aux_video, aux_valid, item = batches[item_index]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(
                prompt_ids=prompt_inputs["input_ids"].to("cuda"),
                pixel_values_videos=prompt_inputs["pixel_values_videos"].to("cuda"),
                video_grid_thw=prompt_inputs["video_grid_thw"].to("cuda"),
                aux_video=aux_video.to("cuda"),
                aux_frame_valid_mask=aux_valid.to("cuda"),
                answer_ids=answer_ids.to("cuda"),
                labels=labels.to("cuda"),
            )
        loss = out["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_value = loss.item()
        del out, loss
        torch.cuda.empty_cache()
        print(
            f"[smoke] step={step} item={item['id']} loss={loss_value:.4f} "
            f"peak_mem={torch.cuda.max_memory_allocated()/1024**3:.2f}GiB"
        )

    print("[smoke] SUCCESS - forward/backward ran cleanly for 40 steps")


if __name__ == "__main__":
    main()
