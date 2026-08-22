import os
import random
import json
import torch
import numpy as np
from collections import OrderedDict
from torch.utils.data import Dataset
from PIL import Image
import decord
import cv2
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from qwen_vl_utils import process_vision_info
import argparse

from predict import predict_vqa
from evaluate import evaluate_predictions

class SurgicalVQADataset(Dataset):
    # Many segments share the same source video (30s window / 15s stride over
    # multi-hour case videos), so re-opening the file per __getitem__ call is
    # the dominant training cost. Cache readers per worker process (this dict
    # starts empty and is only populated inside __getitem__, so it is safe to
    # pickle to DataLoader worker processes on spawn-based platforms).
    _VR_CACHE_SIZE = 2

    def __init__(self, json_file, processor, max_frames=16, is_train=True, frame_size=384):
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.processor = processor
        self.max_frames = max_frames
        self.is_train = is_train
        self.frame_size = frame_size
        self._vr_cache = OrderedDict()

    def __len__(self):
        return len(self.data)

    def _get_reader(self, video_path):
        vr = self._vr_cache.get(video_path)
        if vr is not None:
            self._vr_cache.move_to_end(video_path)
            return vr
        vr = decord.VideoReader(video_path)
        self._vr_cache[video_path] = vr
        if len(self._vr_cache) > self._VR_CACHE_SIZE:
            self._vr_cache.popitem(last=False)
        return vr

    def _load_cached_frames(self, frames_dir):
        """Load frames pre-extracted by extract_frames.py (run it once beforehand -
        it's what makes training fast/stable: no per-sample decord decode, no crashes
        with multiprocessing workers). Returns None if the cache entry isn't there yet,
        so __getitem__ can fall back to decoding straight from the source video."""
        if not frames_dir:
            return None
        try:
            frame_paths = [os.path.join(frames_dir, f"frame_{i}.jpg") for i in range(self.max_frames)]
            return [Image.open(p).convert("RGB") for p in frame_paths]
        except Exception:
            return None

    def _decode_frames(self, video_path, t_start, t_end):
        try:
            vr = self._get_reader(video_path)
            fps = vr.get_avg_fps()
            start_frame = int(t_start * fps)
            end_frame = int(t_end * fps)

            frame_indices = np.linspace(start_frame, end_frame - 1, self.max_frames, dtype=int)
            frame_indices = np.clip(frame_indices, 0, len(vr) - 1)

            frames_npy = vr.get_batch(frame_indices).asnumpy()

            # Mask bottom 15% (UI overlay) and resize for speed
            H, W = frames_npy.shape[1], frames_npy.shape[2]
            mask_y = int(0.85 * H)
            frames_npy[:, mask_y:, :, :] = 0

            pil_frames = []
            for f in frames_npy:
                f_resized = cv2.resize(f, (self.frame_size, self.frame_size))
                pil_frames.append(Image.fromarray(f_resized))
            return pil_frames
        except Exception as e:
            # Fallback to black frames
            return [Image.new("RGB", (self.frame_size, self.frame_size), (0, 0, 0)) for _ in range(self.max_frames)]

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']

        if self.is_train:
            # generate_qa.py always puts the shortest/canonical form (bare "Yes"/"No"
            # or the closest match to the reference phrasing) at index 0. BERTScore-F1
            # (the official metric) rewards close lexical match to reference wording far
            # more than semantically-equivalent paraphrases - e.g. "No, X was not used"
            # scores 1.0 against the references while "No, X is not listed" scores 0.71
            # despite both being correct - so bias training toward that canonical form
            # instead of sampling all 5 answer variants uniformly.
            answers = item['answers']
            if len(answers) > 1:
                weights = [0.55] + [0.45 / (len(answers) - 1)] * (len(answers) - 1)
                answer = random.choices(answers, weights=weights, k=1)[0]
            else:
                answer = answers[0]
        else:
            answer = item['answers'][0]

        pil_frames = self._load_cached_frames(item.get('frames_dir'))
        if pil_frames is None:
            pil_frames = self._decode_frames(item['video'], item['t_start'], item['t_end'])

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": pil_frames},
                    {"type": "text", "text": question}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer}
                ]
            }
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
            
        images_list = []
        videos_list = []
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
            return_tensors="pt"
        )
        
        labels = inputs["input_ids"].clone()
        attention_mask = inputs["attention_mask"]
        seq_len = labels.shape[1]
        padding_side = getattr(self.processor.tokenizer, "padding_side", "right")

        # Mask the user prompt (question + vision tokens) in labels, leaving only the
        # assistant answer as supervision. The naive way is to re-run the full processor
        # (vision + text) on the user-only turn per sample, but that repeats the expensive
        # vision preprocessing a second time for every sample. Since the assistant turn has
        # no vision content, we can instead measure the *answer* span with a cheap text-only
        # tokenization and mask everything before it - this needs vision processed only once.
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
                # Fallback for the rare case the chat template isn't a simple turn-append
                # (e.g. a system-prompt suffix depends on later turns). Correct but slow.
                user_inputs = self.processor(
                    text=[user_text],
                    images=images_list[i] if images_list[i] is not None else None,
                    videos=videos_list[i] if videos_list[i] is not None else None,
                    return_tensors="pt"
                )
                user_len = user_inputs["input_ids"].shape[1]

            labels[i, offset:offset + user_len] = -100

        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        return inputs


class BleuEvalCallback(TrainerCallback):
    """Tracks the actual competition metric (max-BLEU-4 over 5 references) on a small,
    fixed val subset alongside the Trainer's own eval loss. Loss doesn't tell you whether
    generated answers are actually any good, so without this you'd only find out at the
    end of a (possibly hours-long) run whether the run was worth doing."""

    def __init__(self, val_json, processor, n_samples=40, max_frames=16, frame_size=384, seed=0):
        with open(val_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        rng = random.Random(seed)
        self.samples = rng.sample(data, min(n_samples, len(data)))
        self.processor = processor
        self.max_frames = max_frames
        self.frame_size = frame_size

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None or not self.samples or not state.is_world_process_zero:
            return
        was_training = model.training
        model.eval()
        preds = []
        for item in self.samples:
            try:
                pred = predict_vqa(
                    model, self.processor, item['video'], item['question'],
                    max_frames=self.max_frames, frame_size=self.frame_size
                )
            except Exception as e:
                print(f"[BLEU-eval] skipping sample due to error: {e}")
                pred = ""
            preds.append(pred)
        mean_bleu, _ = evaluate_predictions(self.samples, preds)
        print(f"[BLEU-eval] step {state.global_step}: mean max BLEU-4 = {mean_bleu:.4f} (n={len(self.samples)})")
        if was_training:
            model.train()


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", type=str, default=r"d:\SCLab\Surgical-Video-Understanding\SurgVU-challenge\src\train_vqa.json")
    parser.add_argument("--val_json", type=str, default=r"d:\SCLab\Surgical-Video-Understanding\SurgVU-challenge\src\val_vqa.json")
    parser.add_argument("--output_dir", type=str, default=r"d:\SCLab\Surgical-Video-Understanding\SurgVU-challenge\checkpoints\qwen2_vl_2b_lora")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=2,
                         help="HF Trainer defaults per_device_eval_batch_size to 8 regardless of "
                              "--batch_size. At higher --max_frames (e.g. 16) this pushes eval batches "
                              "to the edge of VRAM, and near-OOM allocator behavior makes eval catastrophically "
                              "slower (~30x observed at frames=16/batch=8) without actually crashing - "
                              "keep this low unless you've confirmed headroom for a given frame count.")
    parser.add_argument("--grad_acc", type=int, default=4)
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                         help="CPU workers loading pre-extracted frame JPEGs in parallel with the GPU step. "
                              "Only safe at >0 once src/extract_frames.py has been run: the decord fallback "
                              "path (used when a segment has no frame cache) crashes under multiprocessing "
                              "on this machine, so keep this at 0 if you skip frame pre-extraction.")
    parser.add_argument("--bleu_eval_samples", type=int, default=40,
                         help="How many val examples to generate+BLEU-score at each eval_steps checkpoint (0 disables)")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                         help="Path to a checkpoint-N directory to resume training from "
                              "(restores model/optimizer/scheduler state and step count). "
                              "Use only to continue the SAME run on the SAME dataset after "
                              "an interruption - for warm-starting from a different stage's "
                              "adapter on a new dataset, use --init_lora_from instead.")
    parser.add_argument("--init_lora_from", type=str, default=None,
                         help="Path to a previously-saved LoRA adapter dir (e.g. a Stage 0 "
                              "domain-pretrain output) to warm-start THIS run's weights from, "
                              "with a fresh optimizer/step count. For sequential curriculum "
                              "training (pretrain on one corpus, then fine-tune on another) "
                              "rather than resuming an interrupted run.")
    parser.add_argument("--eval_steps", type=int, default=50,
                         help="Full-val-set eval loss is recomputed every N steps - this dominates wall-clock "
                              "time (each pass over all val samples takes minutes), so raise it for long runs.")
    parser.add_argument("--save_steps", type=int, default=100)
    args = parser.parse_args()

    # Load Qwen2-VL-2B model with QLoRA configuration
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    print("Loading model and processor...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Configure image/video resolution to fit in 24GB VRAM
    # Limit max video resolution to 224x224 per frame
    processor.image_processor.min_pixels = 112 * 112
    processor.image_processor.max_pixels = 224 * 224
    
    model = prepare_model_for_kbit_training(model)
    
    # Lora Configuration
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    if args.init_lora_from:
        print(f"Warm-starting LoRA weights from {args.init_lora_from}...")
        model = PeftModel.from_pretrained(model, args.init_lora_from, is_trainable=True)
    else:
        model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Datasets
    print("Preparing datasets...")
    train_dataset = SurgicalVQADataset(args.train_json, processor, is_train=True)
    val_dataset = SurgicalVQADataset(args.val_json, processor, is_train=False)
    
    collate_fn = Qwen2VLCollate(processor)
    
    # Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_persistent_workers=args.dataloader_num_workers > 0,
        dataloader_pin_memory=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn
    )

    if args.bleu_eval_samples > 0:
        trainer.add_callback(BleuEvalCallback(
            args.val_json, processor, n_samples=args.bleu_eval_samples
        ))

    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    # Save the adapter model
    print(f"Saving fine-tuned LoRA model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("Training completed successfully!")

if __name__ == "__main__":
    train()
