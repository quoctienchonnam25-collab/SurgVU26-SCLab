"""Plain Qwen2.5-VL-3B + LoRA fine-tune - no SurgMotion branch, no dual-encoder
complexity, no vision-tower adaptation. See predict_plain_qwen25vl.py's module
docstring for why: real prelim leaderboard evidence (2026-08-17) showed this
project's added architecture complexity and a locally-validated data change both
regressed hard on the real hidden test set, while the current #1 leaderboard entry
uses exactly this minimal shape ("QwenVL LoRA, 5-frame").

Reuses the existing frame_cache_16fr cache (real SurgVU segments, 384x384 JPEGs,
16 frames/segment - see train_surgmotion_vqa.py::segment_hash/cache_dir_for) and
subsamples --max-frames of the 16, instead of decoding video directly per step.
This is safe under multiprocessing (pure JPEG reads) unlike the original train.py's
decord fallback path, and reuses the exact frame content every other checkpoint
this project has trained on rather than a fresh, possibly-differently-sampled
decode.
"""
import os
import random
import json
import torch
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, PeftModel
from qwen_vl_utils import process_vision_info
import argparse

from predict_plain_qwen25vl import predict_vqa_clip, DEFAULT_BASE_MODEL
from evaluate import evaluate_predictions
from train_surgmotion_vqa import cache_dir_for

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SURGMOTION_CACHE_FRAMES = 16


class SurgicalVQADataset(Dataset):
    def __init__(self, json_file, cache_root, max_frames=5, is_train=True):
        with open(json_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.cache_root = Path(cache_root)
        self.max_frames = max_frames
        self.is_train = is_train
        self._stride = SURGMOTION_CACHE_FRAMES // max_frames if max_frames <= SURGMOTION_CACHE_FRAMES else 1

    def __len__(self):
        return len(self.data)

    def _load_cached_frames(self, item):
        folder = cache_dir_for(item, self.cache_root)
        indices = list(range(0, SURGMOTION_CACHE_FRAMES, self._stride))[: self.max_frames]
        frames = []
        for index in indices:
            frame_path = folder / f"frame_{index}.jpg"
            with Image.open(frame_path) as image:
                frames.append(image.convert("RGB").copy())
        return frames

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["question"]

        if self.is_train:
            # generate_qa.py puts the canonical/shortest reference-matching form at
            # index 0 - bias sampling toward it (BERTScore-F1 rewards close lexical
            # match far more than semantically-equivalent paraphrases), same as the
            # original train.py.
            answers = item["answers"]
            if len(answers) > 1:
                weights = [0.55] + [0.45 / (len(answers) - 1)] * (len(answers) - 1)
                answer = random.choices(answers, weights=weights, k=1)[0]
            else:
                answer = answers[0]
        else:
            answer = item["answers"][0]

        pil_frames = self._load_cached_frames(item)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": pil_frames},
                    {"type": "text", "text": question},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]
        return messages


class Qwen2VLCollate:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        texts = []
        for messages in batch:
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)

        images_list, videos_list = [], []
        for messages in batch:
            images, videos = process_vision_info(messages)
            images_list.append(images)
            videos_list.append(videos)

        has_images = any(x is not None for x in images_list)
        has_videos = any(x is not None for x in videos_list)

        inputs = self.processor(
            text=texts,
            images=images_list if has_images else None,
            videos=videos_list if has_videos else None,
            padding=True,
            return_tensors="pt",
        )

        labels = inputs["input_ids"].clone()
        attention_mask = inputs["attention_mask"]
        seq_len = labels.shape[1]
        padding_side = getattr(self.processor.tokenizer, "padding_side", "right")

        for i, messages in enumerate(batch):
            real_len = int(attention_mask[i].sum().item())
            offset = (seq_len - real_len) if padding_side == "left" else 0

            user_text = self.processor.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)
            full_text = texts[i]

            if full_text.startswith(user_text):
                assistant_suffix = full_text[len(user_text):]
                suffix_ids = self.processor.tokenizer(assistant_suffix, add_special_tokens=False)["input_ids"]
                user_len = max(real_len - len(suffix_ids), 0)
            else:
                user_inputs = self.processor(
                    text=[user_text],
                    images=images_list[i] if images_list[i] is not None else None,
                    videos=videos_list[i] if videos_list[i] is not None else None,
                    return_tensors="pt",
                )
                user_len = user_inputs["input_ids"].shape[1]

            labels[i, offset:offset + user_len] = -100

        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        return inputs


class BleuEvalCallback(TrainerCallback):
    def __init__(self, val_json, cache_root, processor, n_samples=40, max_frames=5, seed=0):
        with open(val_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        rng = random.Random(seed)
        self.samples = rng.sample(data, min(n_samples, len(data)))
        self.cache_root = cache_root
        self.processor = processor
        self.max_frames = max_frames

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None or not self.samples or not state.is_world_process_zero:
            return
        was_training = model.training
        model.eval()
        preds = []
        for item in self.samples:
            try:
                pred = predict_vqa_clip(
                    model, self.processor, item["video"], item["question"],
                    item["t_start"], item["t_end"], max_frames=self.max_frames,
                )
            except Exception as e:
                print(f"[BLEU-eval] skipping sample due to error: {e}")
                pred = ""
            preds.append(pred)
        mean_bleu, _ = evaluate_predictions(self.samples, preds)
        print(f"[BLEU-eval] step {state.global_step}: mean max BLEU-4 = {mean_bleu:.4f} (n={len(self.samples)})")
        if was_training:
            model.train()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", type=str, required=True)
    parser.add_argument("--val_json", type=str, required=True)
    parser.add_argument("--cache_root", type=str, default=str(PROJECT_ROOT / "frame_cache_16fr"))
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--max_frames", type=int, default=5)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--grad_acc", type=int, default=4)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--bleu_eval_samples", type=int, default=0)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--init_lora_from", type=str, default=None)
    parser.add_argument("--eval_strategy", choices=("no", "steps", "epoch"), default="no")
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading model and processor...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cuda")

    processor = AutoProcessor.from_pretrained(args.base_model, local_files_only=True)
    processor.image_processor.min_pixels = 112 * 112
    processor.image_processor.max_pixels = 224 * 224

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    if args.init_lora_from:
        print(f"Warm-starting LoRA weights from {args.init_lora_from}...")
        model = PeftModel.from_pretrained(model, args.init_lora_from, is_trainable=True)
    else:
        model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("Preparing datasets...")
    train_dataset = SurgicalVQADataset(args.train_json, args.cache_root, max_frames=args.max_frames, is_train=True)
    val_dataset = SurgicalVQADataset(args.val_json, args.cache_root, max_frames=args.max_frames, is_train=False)
    collate_fn = Qwen2VLCollate(processor)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=args.logging_steps,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps if args.eval_strategy == "steps" else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_persistent_workers=args.dataloader_num_workers > 0,
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if args.eval_strategy != "no" else None,
        data_collator=collate_fn,
    )

    if args.bleu_eval_samples > 0:
        trainer.add_callback(
            BleuEvalCallback(args.val_json, args.cache_root, processor, n_samples=args.bleu_eval_samples, max_frames=args.max_frames)
        )

    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    print(f"Saving fine-tuned LoRA model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
