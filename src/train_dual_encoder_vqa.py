"""Train DualEncoderVQA on SurgVU QA pairs.

Qwen2.5-VL's own native vision pipeline (frozen) supplies general scene
understanding; a frozen SurgMotion branch supplies a short auxiliary token
sequence with surgical-motion-specific signal. Only the LoRA-adapted decoder,
the auxiliary projector, and (optionally) the last few SurgMotion blocks are
trained. See src/dual_encoder_vqa_prototype.py for the architecture rationale
and src/smoke_test_dual_encoder.py for the structural validation this was
built on (40-step overfit test on 4 real examples: loss 4.37 -> 0.28, peak
15.55 GiB at qwen-num-frames=8/qwen-max-pixels=112x112).

Reuses SurgVU segment/cache handling (segment_hash, cache_dir_for,
cache_preflight, resolve_video_path, black-frame detection) from
train_surgmotion_vqa.py unchanged - only the model-facing dataset/collator and
checkpoint format differ.
"""

import argparse
import json
import subprocess
from pathlib import Path

import torch
from peft import set_peft_model_state_dict
from PIL import Image
from qwen_vl_utils import process_vision_info
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoTokenizer, Trainer, TrainingArguments

from dual_encoder_vqa_prototype import AUX_NUM_TOKENS, DEFAULT_QWEN_MODEL, build
from surgmotion_vqa_prototype import DEFAULT_SURGMOTION_CHECKPOINT
from train_surgmotion_vqa import (
    BLACK_MEAN_THRESHOLD,
    BLACK_P99_THRESHOLD,
    IMAGE_MEAN,
    IMAGE_STD,
    MODEL_FRAME_SIZE,
    cache_dir_for,
    cache_preflight,
    extract_missing_cache,
    load_json,
    unique_segments,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN_JSON = PROJECT_ROOT / "src" / "train_vqa_surgmotion_curated_v2.json"
DEFAULT_VAL_JSON = PROJECT_ROOT / "src" / "val_vqa_surgmotion_curated_v2.json"
DEFAULT_CACHE_ROOT = PROJECT_ROOT / "frame_cache_16fr"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "checkpoints" / "dual_encoder_vqa_dev"

SURGMOTION_NUM_FRAMES = 16


class DualEncoderQADataset(Dataset):
    """Loads one segment's cached frames once, then derives both video
    representations from it: a normalized SurgMotion clip tensor (all 16
    cached frames, 256px) and a subsampled list of PIL frames for Qwen's own
    processor (qwen_num_frames of the 16, resolution capped by the processor's
    own min/max_pixels at collate time)."""

    def __init__(
        self,
        json_path,
        cache_root,
        is_train,
        qwen_num_frames=8,
        black_mean_threshold=BLACK_MEAN_THRESHOLD,
        black_p99_threshold=BLACK_P99_THRESHOLD,
        augment=False,
    ):
        self.data = load_json(json_path)
        self.cache_root = Path(cache_root)
        self.is_train = is_train
        self.qwen_num_frames = qwen_num_frames
        self.black_mean_threshold = black_mean_threshold
        self.black_p99_threshold = black_p99_threshold
        self.augment = augment
        if SURGMOTION_NUM_FRAMES % qwen_num_frames:
            raise ValueError(
                f"--qwen-num-frames={qwen_num_frames} must evenly divide "
                f"the {SURGMOTION_NUM_FRAMES}-frame cache for uniform subsampling"
            )
        self._qwen_stride = SURGMOTION_NUM_FRAMES // qwen_num_frames

    def __len__(self):
        return len(self.data)

    def choose_answer(self, answers):
        if not self.is_train or len(answers) == 1:
            return answers[0]
        weights = [0.55] + [0.45 / (len(answers) - 1)] * (len(answers) - 1)
        import random

        return random.choices(answers, weights=weights, k=1)[0]

    def frame_is_valid(self, frame):
        import cv2
        import numpy as np

        active = frame[: int(frame.shape[0] * 0.85)]
        gray = cv2.cvtColor(active, cv2.COLOR_RGB2GRAY)
        return not (
            float(gray.mean()) <= self.black_mean_threshold
            and float(np.percentile(gray, 99)) <= self.black_p99_threshold
        )

    def augment_frames(self, raw_frames):
        """Light, temporally-consistent augmentation: one color-jitter draw applied
        identically to every frame (per-frame-independent jitter would look like
        flicker/noise to SurgMotion's motion-sensitive encoder, not a realistic
        lighting variation), plus a small circular roll of the 16-frame sequence
        (-1/0/+1) approximating the clip starting slightly earlier or later.
        Train-only; never applied to eval/val data."""
        if not self.is_train or not self.augment:
            return raw_frames

        import random

        from torchvision.transforms import functional as tv_functional

        shift = random.randint(-1, 1)
        if shift:
            raw_frames = raw_frames[-shift:] + raw_frames[:-shift]

        from torchvision.transforms import ColorJitter

        fn_idx, brightness, contrast, saturation, hue = ColorJitter.get_params(
            brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.05, 0.05)
        )

        def jitter(frame):
            for fn_id in fn_idx:
                if fn_id == 0 and brightness is not None:
                    frame = tv_functional.adjust_brightness(frame, brightness)
                elif fn_id == 1 and contrast is not None:
                    frame = tv_functional.adjust_contrast(frame, contrast)
                elif fn_id == 2 and saturation is not None:
                    frame = tv_functional.adjust_saturation(frame, saturation)
                elif fn_id == 3 and hue is not None:
                    frame = tv_functional.adjust_hue(frame, hue)
            return frame

        return [jitter(frame) for frame in raw_frames]

    def load_pil_frames(self, item):
        folder = cache_dir_for(item, self.cache_root)
        frames = []
        for index in range(SURGMOTION_NUM_FRAMES):
            frame_path = folder / f"frame_{index}.jpg"
            if not frame_path.is_file():
                raise FileNotFoundError(
                    f"Missing cached frame {frame_path}; run with --prepare-missing-cache"
                )
            with Image.open(frame_path) as image:
                frames.append(image.convert("RGB").copy())
        return frames

    def __getitem__(self, index):
        import numpy as np

        item = self.data[index]
        raw_frames = self.load_pil_frames(item)
        raw_frames = self.augment_frames(raw_frames)

        resampling = getattr(Image, "Resampling", Image)
        arrays = []
        frame_valid = []
        for frame in raw_frames:
            resized = frame.resize((MODEL_FRAME_SIZE, MODEL_FRAME_SIZE), resampling.BILINEAR)
            array = np.asarray(resized, dtype=np.uint8)
            arrays.append(array)
            frame_valid.append(self.frame_is_valid(array))

        frame_valid = np.asarray(frame_valid, dtype=np.bool_)
        stacked = np.stack(arrays, axis=0)
        valid_indices = np.flatnonzero(frame_valid)
        if valid_indices.size:
            for invalid_index in np.flatnonzero(~frame_valid):
                nearest = valid_indices[np.argmin(np.abs(valid_indices - invalid_index))]
                stacked[invalid_index] = stacked[nearest]

        array = stacked.astype(np.float32) / 255.0
        aux_video = torch.from_numpy(array).permute(3, 0, 1, 2).contiguous()
        aux_video = (aux_video - IMAGE_MEAN) / IMAGE_STD

        qwen_frames = raw_frames[:: self._qwen_stride][: self.qwen_num_frames]

        return {
            "qwen_frames": qwen_frames,
            "aux_video": aux_video,
            "aux_frame_valid_mask": torch.from_numpy(frame_valid),
            "sample_weight": float(item.get("sample_weight", 1.0)),
            "question": item["question"],
            "answer": self.choose_answer(item["answers"]),
            "id": item["id"],
        }


class DualEncoderCollator:
    """Batch size is always 1 (per-device) - see DualEncoderVQA's own docstring
    for why. Builds the prompt (video + question, add_generation_prompt=True)
    and the full conversation (+ answer) through Qwen's own processor with
    identical video content, so the <video> placeholder expands to the same
    token run in both and the prompt is guaranteed to be a real prefix of the
    full sequence - then splits off the answer tokens as the supervised span."""

    def __init__(self, tokenizer, processor, max_text_length=384):
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_text_length = max_text_length
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    def __call__(self, examples):
        if len(examples) != 1:
            raise ValueError("DualEncoderCollator only supports batch size 1")
        item = examples[0]
        user_turn = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": item["qwen_frames"]},
                    {"type": "text", "text": item["question"]},
                ],
            }
        ]
        full_turns = user_turn + [{"role": "assistant", "content": item["answer"]}]

        prompt_text = self.processor.apply_chat_template(
            user_turn, tokenize=False, add_generation_prompt=True
        )
        full_text = self.processor.apply_chat_template(
            full_turns, tokenize=False, add_generation_prompt=False
        )

        image_inputs, video_inputs = process_vision_info(user_turn)
        prompt_inputs = self.processor(
            text=[prompt_text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        )
        full_inputs = self.processor(
            text=[full_text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        )

        prompt_ids_list = prompt_inputs["input_ids"][0].tolist()
        full_ids_list = full_inputs["input_ids"][0].tolist()
        if full_ids_list[: len(prompt_ids_list)] != prompt_ids_list:
            raise RuntimeError("Qwen chat prompt is not a prefix of the supervised text")
        answer_ids_list = full_ids_list[len(prompt_ids_list):]
        if not answer_ids_list:
            raise RuntimeError("No assistant tokens remain after prefix split")
        if len(full_ids_list) > self.max_text_length:
            raise ValueError(
                f"Tokenized example has {len(full_ids_list)} tokens, exceeding "
                f"--max-text-length={self.max_text_length}"
            )

        answer_ids = torch.tensor([answer_ids_list], dtype=torch.long)
        labels = answer_ids.clone()

        return {
            "prompt_ids": prompt_inputs["input_ids"],
            "pixel_values_videos": prompt_inputs["pixel_values_videos"],
            "video_grid_thw": prompt_inputs["video_grid_thw"],
            "second_per_grid_ts": prompt_inputs.get("second_per_grid_ts"),
            "aux_video": item["aux_video"].unsqueeze(0),
            "aux_frame_valid_mask": item["aux_frame_valid_mask"].unsqueeze(0),
            "answer_ids": answer_ids,
            "labels": labels,
            "sample_weight": torch.tensor([item["sample_weight"]], dtype=torch.float32),
        }


def load_component_checkpoint(target, checkpoint_path):
    """Load aux_projector, Qwen LoRA and an optional fine-tuned SurgMotion delta."""
    checkpoint = Path(checkpoint_path)
    projector_path = checkpoint / "aux_projector.safetensors"
    adapter_path = checkpoint / "lora_adapter" / "adapter_model.safetensors"
    if not projector_path.is_file() or not adapter_path.is_file():
        raise FileNotFoundError(f"Incomplete DualEncoderVQA checkpoint: {checkpoint}")

    projector_result = target.aux_projector.load_state_dict(
        load_file(projector_path, device="cpu"), strict=False
    )
    allowed_missing = {"no_visual_prefix"}
    missing_projector = set(projector_result.missing_keys).difference(allowed_missing)
    if missing_projector or projector_result.unexpected_keys:
        raise RuntimeError(
            "Invalid aux_projector checkpoint: "
            f"missing={sorted(missing_projector)}, unexpected={projector_result.unexpected_keys}"
        )

    adapter_state = load_file(adapter_path, device="cpu")
    result = set_peft_model_state_dict(target.text_model, adapter_state)
    missing_adapters = [key for key in getattr(result, "missing_keys", []) if "lora_" in key]
    unexpected = list(getattr(result, "unexpected_keys", []))
    if missing_adapters or unexpected:
        raise RuntimeError(
            f"Invalid adapter checkpoint: missing={missing_adapters}, unexpected={unexpected}"
        )

    fusion_path = checkpoint / "fusion.safetensors"
    if fusion_path.is_file():
        if target.fusion is None:
            raise RuntimeError(
                f"Checkpoint {checkpoint} has a fusion.safetensors but the target "
                "model was built with use_fusion=False - pass --use-fusion."
            )
        fusion_result = target.fusion.load_state_dict(
            load_file(fusion_path, device="cpu"), strict=False
        )
        if fusion_result.missing_keys or fusion_result.unexpected_keys:
            raise RuntimeError(
                "Invalid fusion checkpoint: "
                f"missing={fusion_result.missing_keys}, unexpected={fusion_result.unexpected_keys}"
            )

    vision_adapter_path = checkpoint / "vision_lora_adapter" / "adapter_model.safetensors"
    if vision_adapter_path.is_file():
        if not hasattr(target.qwen_model.visual, "peft_config"):
            raise RuntimeError(
                f"Checkpoint {checkpoint} has a vision_lora_adapter but the target "
                "model was built with vision_lora_r=0 - pass a matching --vision-lora-r."
            )
        vision_adapter_state = load_file(vision_adapter_path, device="cpu")
        vision_result = set_peft_model_state_dict(target.qwen_model.visual, vision_adapter_state)
        missing_vision_adapters = [
            key for key in getattr(vision_result, "missing_keys", []) if "lora_" in key
        ]
        unexpected_vision = list(getattr(vision_result, "unexpected_keys", []))
        if missing_vision_adapters or unexpected_vision:
            raise RuntimeError(
                "Invalid vision adapter checkpoint: "
                f"missing={missing_vision_adapters}, unexpected={unexpected_vision}"
            )

    encoder_path = checkpoint / "surgmotion_delta.safetensors"
    if encoder_path.is_file():
        encoder_state = load_file(encoder_path, device="cpu")
        encoder_result = target.surgmotion_encoder.load_state_dict(encoder_state, strict=False)
        if encoder_result.unexpected_keys:
            raise RuntimeError(
                f"Invalid SurgMotion delta: unexpected={encoder_result.unexpected_keys}"
            )
        target.surgmotion_delta_keys.update(encoder_state)
        print(f"[checkpoint] loaded SurgMotion delta with {len(encoder_state)} tensors")
    else:
        print("[checkpoint] no SurgMotion delta; using released encoder weights")


class DualComponentCheckpointTrainer(Trainer):
    """Save compact aux_projector/LoRA/encoder-delta component checkpoints -
    mirrors train_surgmotion_vqa.py's ComponentCheckpointTrainer."""

    def _save(self, output_dir=None, state_dict=None):
        output_dir = Path(output_dir or self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        projector_state = {
            name: value.detach().cpu().contiguous()
            for name, value in self.model.aux_projector.state_dict().items()
        }
        save_file(projector_state, output_dir / "aux_projector.safetensors")
        if self.model.fusion is not None:
            fusion_state = {
                name: value.detach().cpu().contiguous()
                for name, value in self.model.fusion.state_dict().items()
            }
            save_file(fusion_state, output_dir / "fusion.safetensors")
        self.model.text_model.save_pretrained(output_dir / "lora_adapter", safe_serialization=True)
        if hasattr(self.model.qwen_model.visual, "peft_config"):
            self.model.qwen_model.visual.save_pretrained(
                output_dir / "vision_lora_adapter", safe_serialization=True
            )

        delta_keys = set(self.model.surgmotion_delta_keys)
        delta_keys.update(
            name
            for name, parameter in self.model.surgmotion_encoder.named_parameters()
            if parameter.requires_grad
        )
        if delta_keys:
            encoder_state = self.model.surgmotion_encoder.state_dict()
            save_file(
                {name: encoder_state[name].detach().cpu().contiguous() for name in sorted(delta_keys)},
                output_dir / "surgmotion_delta.safetensors",
            )
            self.model.surgmotion_delta_keys.update(delta_keys)

        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir / "tokenizer")
        torch.save(self.args, output_dir / "training_args.bin")

        metadata = {
            "architecture": "DualEncoderVQA",
            "aux_num_tokens": getattr(self.model.aux_projector, "num_output_tokens", AUX_NUM_TOKENS),
            "surgmotion_num_frames": SURGMOTION_NUM_FRAMES,
            "qwen_num_frames": getattr(self.model, "qwen_num_frames", 8),
            "qwen_max_pixels": getattr(self.model, "qwen_max_pixels", 112 * 112),
            "qwen_model": str(getattr(self.model, "qwen_model_id", DEFAULT_QWEN_MODEL)),
            "surgmotion_checkpoint": str(
                getattr(self.model, "surgmotion_checkpoint_id", DEFAULT_SURGMOTION_CHECKPOINT)
            ),
            "surgmotion_delta_tensors": len(delta_keys),
            "train_surgmotion_blocks": getattr(self.model, "train_surgmotion_blocks", 0),
            "qwen_frozen": getattr(self.model, "qwen_frozen", False),
            "lora_r": getattr(self.model, "lora_r", 16),
            "vision_lora_r": getattr(self.model, "vision_lora_r", 0),
            "use_fusion": getattr(self.model, "use_fusion", False),
            "fusion_heads": getattr(self.model, "fusion_heads", 8),
        }
        with open(output_dir / "dual_encoder_vqa_config.json", "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        load_component_checkpoint(model or self.model, resume_from_checkpoint)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json", default=str(DEFAULT_TRAIN_JSON))
    parser.add_argument("--val-json", default=str(DEFAULT_VAL_JSON))
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--qwen-model", default=str(DEFAULT_QWEN_MODEL))
    parser.add_argument("--surgmotion-checkpoint", default=str(DEFAULT_SURGMOTION_CHECKPOINT))
    parser.add_argument("--qwen-num-frames", type=int, default=8)
    parser.add_argument(
        "--qwen-max-pixels",
        type=int,
        default=112 * 112,
        help="Caps Qwen's own video token budget - see smoke_test_dual_encoder.py "
        "for why this matters (OOM at the default resolution even on a 24GiB GPU).",
    )
    parser.add_argument("--aux-num-tokens", type=int, default=AUX_NUM_TOKENS)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument(
        "--vision-lora-r",
        type=int,
        default=0,
        help="0 (default) keeps Qwen's own vision tower fully frozen, as originally "
        "designed. >0 wraps it in a small LoRA adapter (target_modules=qkv/gate_proj/"
        "up_proj/down_proj) instead, for a low-rank domain adaptation on top of its "
        "pretrained weights - see dual_encoder_vqa_prototype.py::build() docstring.",
    )
    parser.add_argument(
        "--use-fusion",
        action="store_true",
        help="Insert a trainable CrossModalFusion cross-attention block (aux tokens "
        "attend to Qwen's own visual tokens) before appending aux tokens to the "
        "prompt, instead of leaving the two visual streams to interact only "
        "through the LLM's own self-attention post-concatenation.",
    )
    parser.add_argument("--fusion-heads", type=int, default=8)
    parser.add_argument(
        "--train-surgmotion-blocks",
        type=int,
        default=0,
        help="Unfreeze the final N SurgMotion ViT blocks plus encoder norm (Stage A).",
    )
    parser.add_argument(
        "--freeze-qwen",
        action="store_true",
        help="Freeze Qwen LoRA while adapting the aux_projector/SurgMotion in Stage A.",
    )
    parser.add_argument("--initial-checkpoint", default=None)
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Train-only light augmentation: temporally-consistent color jitter "
        "(one draw applied identically to all 16 frames) plus a small circular "
        "frame-sequence roll (-1/0/+1). Never applied to eval data.",
    )
    parser.add_argument("--prepare-missing-cache", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--cache-workers", type=int, default=1)
    parser.add_argument("--cache-decode-chunk-size", type=int, default=64)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--grad-acc", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--max-text-length",
        type=int,
        default=1200,
        help="Unlike SurgMotionVQA, Qwen's own video placeholder tokens count "
        "toward this (~800 at qwen-num-frames=8/qwen-max-pixels=112x112), not "
        "just question+answer text.",
    )
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-strategy", choices=("no", "steps", "epoch"), default="steps")
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random_seed = args.seed
    import random

    import numpy as np

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    train_data = load_json(args.train_json)
    val_data = load_json(args.val_json)
    all_segments = unique_segments(train_data, val_data)
    complete, missing = cache_preflight(all_segments, args.cache_root, num_frames=SURGMOTION_NUM_FRAMES)
    print(
        f"[preflight] unique_segments={len(all_segments)} "
        f"complete_cache={complete} missing_cache={len(missing)}"
    )

    if missing and args.prepare_missing_cache:
        extract_missing_cache(
            all_segments,
            args.cache_root,
            num_frames=SURGMOTION_NUM_FRAMES,
            decode_chunk_size=args.cache_decode_chunk_size,
            cache_workers=args.cache_workers,
        )
        complete, missing = cache_preflight(all_segments, args.cache_root, num_frames=SURGMOTION_NUM_FRAMES)
        print(f"[preflight] after extraction: complete={complete} missing={len(missing)}")
    if args.preflight_only:
        if missing:
            print(f"[preflight] NOT READY: {len(missing)} segments still need cache extraction")
        else:
            print(f"[preflight] READY: every requested segment has {SURGMOTION_NUM_FRAMES} cached frames")
        return
    if missing:
        raise RuntimeError(
            f"{len(missing)} segments lack a complete {SURGMOTION_NUM_FRAMES}-frame cache. "
            "Run again with --prepare-missing-cache."
        )
    if args.cache_only:
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full DualEncoderVQA training")
    gpu_processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    conflicting = [line for line in gpu_processes if line.strip() and "nxnode" not in line]
    if conflicting:
        raise RuntimeError(
            "Refusing to train while another GPU compute process is active: " + "; ".join(conflicting)
        )

    tokenizer = AutoTokenizer.from_pretrained(args.qwen_model, local_files_only=True)
    tokenizer.padding_side = "right"
    processor = AutoProcessor.from_pretrained(args.qwen_model, local_files_only=True)
    processor.image_processor.min_pixels = args.qwen_max_pixels
    processor.image_processor.max_pixels = args.qwen_max_pixels

    train_dataset = DualEncoderQADataset(
        args.train_json, args.cache_root, is_train=True, qwen_num_frames=args.qwen_num_frames,
        augment=args.augment,
    )
    val_dataset = DualEncoderQADataset(
        args.val_json, args.cache_root, is_train=False, qwen_num_frames=args.qwen_num_frames
    )
    collator = DualEncoderCollator(tokenizer, processor, max_text_length=args.max_text_length)

    print(
        f"[model] loading Qwen2.5-VL (native vision, frozen) + SurgMotion aux "
        f"(train_last={args.train_surgmotion_blocks}, tokens={args.aux_num_tokens}, "
        f"freeze_qwen={args.freeze_qwen})"
    )
    model = build(
        surgmotion_checkpoint=args.surgmotion_checkpoint,
        qwen_model_id=args.qwen_model,
        aux_num_frames=SURGMOTION_NUM_FRAMES,
        aux_num_tokens=args.aux_num_tokens,
        lora_r=args.lora_r,
        # BF16 for training (matches TrainingArguments' bf16=True below) - explicit
        # rather than relying on build()'s own default, since predict_dual_encoder_vqa.py
        # needs FP16 instead for T4 inference compatibility and passes that explicitly too.
        dtype=torch.bfloat16,
        local_files_only=True,
        train_surgmotion_blocks=args.train_surgmotion_blocks,
        freeze_qwen=args.freeze_qwen,
        vision_lora_r=args.vision_lora_r,
        use_fusion=args.use_fusion,
        fusion_heads=args.fusion_heads,
    )
    model.qwen_model_id = args.qwen_model
    model.surgmotion_checkpoint_id = args.surgmotion_checkpoint
    model.train_surgmotion_blocks = args.train_surgmotion_blocks
    model.qwen_frozen = args.freeze_qwen
    model.qwen_num_frames = args.qwen_num_frames
    model.qwen_max_pixels = args.qwen_max_pixels
    model.lora_r = args.lora_r
    model.vision_lora_r = args.vision_lora_r
    model.use_fusion = args.use_fusion
    model.fusion_heads = args.fusion_heads
    if args.initial_checkpoint:
        load_component_checkpoint(model, args.initial_checkpoint)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[model] trainable={trainable:,} total={total:,} ({100 * trainable / total:.3f}%)")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,
        fp16=False,
        max_grad_norm=1.0,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps if args.eval_strategy == "steps" else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        prediction_loss_only=True,
        remove_unused_columns=False,
        dataloader_num_workers=args.num_workers,
        dataloader_persistent_workers=args.num_workers > 0,
        dataloader_pin_memory=True,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        optim="adamw_torch",
    )

    trainer = DualComponentCheckpointTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if args.eval_strategy != "no" else None,
        data_collator=collator,
        processing_class=tokenizer,
    )
    print("[train] starting")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    trainer.save_state()
    print(f"[train] completed; components saved to {args.output_dir}")


if __name__ == "__main__":
    main()
