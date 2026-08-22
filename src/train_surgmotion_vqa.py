"""Train SurgMotionVQA on SurgVU QA pairs.

This is intentionally separate from train.py: Qwen's vision processor/tower is not used.
The input video is normalized for SurgMotion and injected as a learned visual prefix.
"""

import argparse
import hashlib
import json
import os
import random
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PureWindowsPath

import cv2
import decord
import numpy as np
import torch
from PIL import Image
from peft import set_peft_model_state_dict
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, Trainer, TrainingArguments

from surgmotion_vqa_prototype import (
    DEFAULT_QWEN_MODEL,
    DEFAULT_SURGMOTION_CHECKPOINT,
    build,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN_JSON = PROJECT_ROOT / "src" / "train_vqa_surgmotion_curated.json"
DEFAULT_VAL_JSON = PROJECT_ROOT / "src" / "val_vqa_surgmotion_curated.json"
DEFAULT_CACHE_ROOT = PROJECT_ROOT / "frame_cache_16fr"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "checkpoints" / "surgmotion_qwen2_5_vl_3b_lora"

NUM_FRAMES = 16
CACHE_FRAME_SIZE = 384
MODEL_FRAME_SIZE = 256
BLACK_MEAN_THRESHOLD = 3.0
BLACK_P99_THRESHOLD = 10.0
IMAGE_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1, 1)
IMAGE_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1, 1)


def segment_hash(item):
    """Match extract_frames.py exactly, including the original Windows path string."""
    return hashlib.md5(
        f"{item['video']}|{item['t_start']}|{item['t_end']}".encode("utf-8")
    ).hexdigest()


def legacy_segment_hash(item):
    """Hash used by the original Windows-generated QA/cache artifacts."""
    raw = item["video"]
    native_parts = list(Path(raw).parts)
    lowered = [part.lower() for part in native_parts]
    if "surgvu24" not in lowered:
        return None
    index = lowered.index("surgvu24")
    legacy_root = PureWindowsPath(
        r"d:\SCLab\Surgical-Video-Understanding\SurgVU-challenge"
    )
    legacy_video = str(legacy_root.joinpath(*native_parts[index:]))
    return hashlib.md5(
        f"{legacy_video}|{item['t_start']}|{item['t_end']}".encode("utf-8")
    ).hexdigest()


def cache_dir_for(item, cache_root):
    cache_root = Path(cache_root)
    primary = cache_root / segment_hash(item)
    if primary.is_dir():
        return primary
    legacy_hash = legacy_segment_hash(item)
    if legacy_hash is not None:
        legacy = cache_root / legacy_hash
        if legacy.is_dir():
            return legacy
    return primary


def cache_is_complete(item, cache_root, num_frames=NUM_FRAMES):
    folder = cache_dir_for(item, cache_root)
    return all((folder / f"frame_{index}.jpg").is_file() for index in range(num_frames))


def resolve_video_path(raw_path, project_root=PROJECT_ROOT):
    """Resolve either a native path or the Windows paths stored in transferred JSON."""
    native = Path(raw_path)
    if native.is_file():
        return native

    parts = list(PureWindowsPath(raw_path).parts)
    lowered = [part.lower() for part in parts]
    if "surgvu24" in lowered:
        index = lowered.index("surgvu24")
        candidate = Path(project_root).joinpath(*parts[index:])
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(f"Cannot resolve source video: {raw_path}")


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or not data:
        raise ValueError(f"Expected a non-empty JSON list: {path}")
    required = {"video", "t_start", "t_end", "question", "answers"}
    for index, item in enumerate(data):
        missing = required.difference(item)
        if missing:
            raise KeyError(f"{path}[{index}] is missing keys: {sorted(missing)}")
        if not item["answers"]:
            raise ValueError(f"{path}[{index}] has no answers")
    return data


def unique_segments(*datasets):
    unique = {}
    for dataset in datasets:
        for item in dataset:
            unique.setdefault(segment_hash(item), item)
    return list(unique.values())


def cache_preflight(items, cache_root, num_frames=NUM_FRAMES):
    complete = 0
    missing = []
    for item in unique_segments(items):
        if cache_is_complete(item, cache_root, num_frames=num_frames):
            complete += 1
        else:
            missing.append(item)
    return complete, missing


def _extract_video_cache(
    video_path, video_items, cache_root, decode_chunk_size, num_frames
):
    """Decode all requested timestamps for one source video in ascending order."""
    reader = decord.VideoReader(str(video_path))
    fps = float(reader.get_avg_fps())
    total_frames = len(reader)
    if total_frames <= 0 or fps <= 0:
        raise RuntimeError(f"Invalid video metadata: {video_path}")

    video_items = sorted(video_items, key=lambda value: float(value["t_start"]))
    destinations = defaultdict(list)
    for item in video_items:
        output_dir = cache_dir_for(item, cache_root)
        output_dir.mkdir(parents=True, exist_ok=True)
        start_frame = int(float(item["t_start"]) * fps)
        end_frame = int(float(item["t_end"]) * fps)
        indices = np.linspace(
            start_frame, max(end_frame - 1, start_frame), num_frames, dtype=int
        )
        indices = np.clip(indices, 0, total_frames - 1)
        for output_index, source_index in enumerate(indices):
            destinations[int(source_index)].append((output_dir, output_index))

    ordered_indices = sorted(destinations)
    for chunk_start in range(0, len(ordered_indices), decode_chunk_size):
        chunk = ordered_indices[chunk_start : chunk_start + decode_chunk_size]
        frames = reader.get_batch(np.asarray(chunk, dtype=np.int64)).asnumpy()
        mask_y = int(0.85 * frames.shape[1])
        frames[:, mask_y:, :, :] = 0

        for source_index, frame in zip(chunk, frames):
            resized = cv2.resize(frame, (CACHE_FRAME_SIZE, CACHE_FRAME_SIZE))
            encoded = Image.fromarray(resized)
            for output_dir, output_index in destinations[source_index]:
                final_path = output_dir / f"frame_{output_index}.jpg"
                temporary_path = output_dir / f"frame_{output_index}.jpg.tmp"
                encoded.save(temporary_path, format="JPEG", quality=90)
                os.replace(temporary_path, final_path)

    for item in video_items:
        if not cache_is_complete(item, cache_root, num_frames=num_frames):
            raise RuntimeError(
                f"Cache write verification failed: {cache_dir_for(item, cache_root)}"
            )
    return len(video_items)


def extract_missing_cache(
    items, cache_root, num_frames=NUM_FRAMES, decode_chunk_size=64, cache_workers=1
):
    """Fill missing cache entries without backward-seeking overlapping windows.

    SurgVU segments use a 30s window with a 15s stride. Decoding one segment at a
    time makes the reader seek backward 15s for almost every sample. Instead, merge
    all requested frame indices per source video, decode them once in increasing
    order, and fan each decoded frame out to the segment JPEGs that reference it.
    Independent source videos are processed concurrently.
    """
    if cache_workers < 1:
        raise ValueError("cache_workers must be >= 1")
    _, missing = cache_preflight(items, cache_root, num_frames=num_frames)
    if not missing:
        print("[cache] all requested segments are complete")
        return

    grouped = defaultdict(list)
    for item in missing:
        video_path = resolve_video_path(item["video"])
        grouped[video_path].append(item)

    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    progress = tqdm(total=len(missing), desc=f"Completing {num_frames}-frame cache")

    jobs = [(video_path, grouped[video_path]) for video_path in sorted(grouped, key=str)]
    try:
        if cache_workers == 1:
            for video_path, video_items in jobs:
                completed = _extract_video_cache(
                    video_path, video_items, cache_root, decode_chunk_size, num_frames
                )
                progress.update(completed)
        else:
            with ThreadPoolExecutor(
                max_workers=min(cache_workers, len(jobs)),
                thread_name_prefix="surgmotion-cache",
            ) as executor:
                futures = {
                    executor.submit(
                        _extract_video_cache,
                        video_path,
                        video_items,
                        cache_root,
                        decode_chunk_size,
                        num_frames,
                    ): video_path
                    for video_path, video_items in jobs
                }
                for future in as_completed(futures):
                    try:
                        progress.update(future.result())
                    except Exception as error:
                        raise RuntimeError(
                            f"Cache extraction failed for {futures[future]}"
                        ) from error
    finally:
        progress.close()

class SurgMotionQADataset(Dataset):
    def __init__(
        self,
        json_path,
        cache_root,
        is_train,
        num_frames=NUM_FRAMES,
        black_mean_threshold=BLACK_MEAN_THRESHOLD,
        black_p99_threshold=BLACK_P99_THRESHOLD,
    ):
        self.data = load_json(json_path)
        self.cache_root = Path(cache_root)
        self.is_train = is_train
        self.num_frames = num_frames
        self.black_mean_threshold = black_mean_threshold
        self.black_p99_threshold = black_p99_threshold
        resampling = getattr(Image, "Resampling", Image)
        self.resize_filter = resampling.BILINEAR

    def __len__(self):
        return len(self.data)

    def choose_answer(self, answers):
        if not self.is_train or len(answers) == 1:
            return answers[0]
        weights = [0.55] + [0.45 / (len(answers) - 1)] * (len(answers) - 1)
        return random.choices(answers, weights=weights, k=1)[0]

    def frame_is_valid(self, frame):
        active = frame[: int(frame.shape[0] * 0.85)]
        gray = cv2.cvtColor(active, cv2.COLOR_RGB2GRAY)
        return not (
            float(gray.mean()) <= self.black_mean_threshold
            and float(np.percentile(gray, 99)) <= self.black_p99_threshold
        )

    def load_video(self, item):
        folder = cache_dir_for(item, self.cache_root)
        frames = []
        frame_valid = []
        for index in range(self.num_frames):
            frame_path = folder / f"frame_{index}.jpg"
            if not frame_path.is_file():
                raise FileNotFoundError(
                    f"Missing cached frame {frame_path}; run with --prepare-missing-cache"
                )
            with Image.open(frame_path) as image:
                rgb = image.convert("RGB").resize(
                    (MODEL_FRAME_SIZE, MODEL_FRAME_SIZE), self.resize_filter
                )
                frame = np.asarray(rgb, dtype=np.uint8)
                frames.append(frame)
                frame_valid.append(self.frame_is_valid(frame))

        frame_valid = np.asarray(frame_valid, dtype=np.bool_)
        array = np.stack(frames, axis=0)
        # SurgMotion expects a fixed-length clip. Replace invalid frames with the
        # nearest valid temporal neighbor while retaining the original validity mask
        # for masked pooling. Fully-black clips remain black and bypass the encoder.
        valid_indices = np.flatnonzero(frame_valid)
        if valid_indices.size:
            for invalid_index in np.flatnonzero(~frame_valid):
                nearest = valid_indices[np.argmin(np.abs(valid_indices - invalid_index))]
                array[invalid_index] = array[nearest]

        array = array.astype(np.float32) / 255.0  # [T, H, W, C]
        tensor = torch.from_numpy(array).permute(3, 0, 1, 2).contiguous()
        return (tensor - IMAGE_MEAN) / IMAGE_STD, torch.from_numpy(frame_valid)

    def __getitem__(self, index):
        item = self.data[index]
        video, frame_valid_mask = self.load_video(item)
        return {
            "video": video,
            "frame_valid_mask": frame_valid_mask,
            "valid_fraction": frame_valid_mask.float().mean(),
            "sample_weight": float(item.get("sample_weight", 1.0)),
            "question": item["question"],
            "answer": self.choose_answer(item["answers"]),
        }


class SurgMotionCollator:
    def __init__(self, tokenizer, max_text_length=256):
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    def encode_conversation(self, question, answer):
        user_turn = [{"role": "user", "content": question}]
        full_turns = user_turn + [{"role": "assistant", "content": answer}]
        prompt_text = self.tokenizer.apply_chat_template(
            user_turn, tokenize=False, add_generation_prompt=True
        )
        full_text = self.tokenizer.apply_chat_template(
            full_turns, tokenize=False, add_generation_prompt=False
        )
        prompt_ids = self.tokenizer(
            prompt_text, add_special_tokens=False
        )["input_ids"]
        full_ids = self.tokenizer(
            full_text, add_special_tokens=False
        )["input_ids"]

        if full_ids[: len(prompt_ids)] != prompt_ids:
            common = 0
            for prompt_id, full_id in zip(prompt_ids, full_ids):
                if prompt_id != full_id:
                    break
                common += 1
            if common == 0:
                raise RuntimeError("Qwen chat prompt is not a prefix of the supervised text")
            prompt_length = common
        else:
            prompt_length = len(prompt_ids)

        if len(full_ids) > self.max_text_length:
            raise ValueError(
                f"Tokenized example has {len(full_ids)} tokens, exceeding "
                f"--max-text-length={self.max_text_length}"
            )
        labels = list(full_ids)
        labels[:prompt_length] = [-100] * prompt_length
        if all(label == -100 for label in labels):
            raise RuntimeError("No assistant tokens remain after label masking")
        return full_ids, labels

    def __call__(self, examples):
        encoded = [
            self.encode_conversation(item["question"], item["answer"])
            for item in examples
        ]
        max_length = max(len(input_ids) for input_ids, _ in encoded)
        batch_size = len(examples)
        input_ids = torch.full(
            (batch_size, max_length), self.tokenizer.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
        labels = torch.full((batch_size, max_length), -100, dtype=torch.long)

        for row, (ids, row_labels) in enumerate(encoded):
            length = len(ids)
            input_ids[row, :length] = torch.tensor(ids, dtype=torch.long)
            attention_mask[row, :length] = 1
            labels[row, :length] = torch.tensor(row_labels, dtype=torch.long)

        return {
            "video": torch.stack([item["video"] for item in examples]),
            "frame_valid_mask": torch.stack(
                [item["frame_valid_mask"] for item in examples]
            ),
            "valid_fraction": torch.stack([item["valid_fraction"] for item in examples]),
            "sample_weight": torch.tensor(
                [item["sample_weight"] for item in examples], dtype=torch.float32
            ),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def load_component_checkpoint(target, checkpoint_path):
    """Load projector, Qwen LoRA and an optional fine-tuned SurgMotion delta."""
    checkpoint = Path(checkpoint_path)
    projector_path = checkpoint / "projector.safetensors"
    adapter_path = checkpoint / "lora_adapter" / "adapter_model.safetensors"
    if not projector_path.is_file() or not adapter_path.is_file():
        raise FileNotFoundError(f"Incomplete SurgMotion checkpoint: {checkpoint}")

    projector_result = target.projector.load_state_dict(
        load_file(projector_path, device="cpu"), strict=False
    )
    allowed_missing = {"no_visual_prefix"}
    missing_projector = set(projector_result.missing_keys).difference(allowed_missing)
    if missing_projector or projector_result.unexpected_keys:
        raise RuntimeError(
            "Invalid projector checkpoint: "
            f"missing={sorted(missing_projector)}, "
            f"unexpected={projector_result.unexpected_keys}"
        )
    if "no_visual_prefix" in projector_result.missing_keys:
        print("[checkpoint] legacy projector has no no_visual_prefix; using fresh initialization")

    adapter_state = load_file(adapter_path, device="cpu")
    result = set_peft_model_state_dict(target.text_model, adapter_state)
    missing_adapters = [
        key for key in getattr(result, "missing_keys", []) if "lora_" in key
    ]
    unexpected = list(getattr(result, "unexpected_keys", []))
    if missing_adapters or unexpected:
        raise RuntimeError(
            f"Invalid adapter checkpoint: missing={missing_adapters}, unexpected={unexpected}"
        )

    encoder_path = checkpoint / "surgmotion_delta.safetensors"
    if encoder_path.is_file():
        encoder_state = load_file(encoder_path, device="cpu")
        encoder_result = target.surgmotion_encoder.load_state_dict(
            encoder_state, strict=False
        )
        if encoder_result.unexpected_keys:
            raise RuntimeError(
                f"Invalid SurgMotion delta: unexpected={encoder_result.unexpected_keys}"
            )
        target.surgmotion_delta_keys.update(encoder_state)
        print(f"[checkpoint] loaded SurgMotion delta with {len(encoder_state)} tensors")
    else:
        print("[checkpoint] no SurgMotion delta; using released encoder weights")


class ComponentCheckpointTrainer(Trainer):
    """Save compact projector/LoRA/encoder-delta component checkpoints."""

    def _save(self, output_dir=None, state_dict=None):
        output_dir = Path(output_dir or self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        projector_state = {
            name: value.detach().cpu().contiguous()
            for name, value in self.model.projector.state_dict().items()
        }
        save_file(projector_state, output_dir / "projector.safetensors")
        self.model.text_model.save_pretrained(
            output_dir / "lora_adapter", safe_serialization=True
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
                {
                    name: encoder_state[name].detach().cpu().contiguous()
                    for name in sorted(delta_keys)
                },
                output_dir / "surgmotion_delta.safetensors",
            )
            self.model.surgmotion_delta_keys.update(delta_keys)

        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir / "tokenizer")
        torch.save(self.args, output_dir / "training_args.bin")
        metadata = {
            "architecture": "SurgMotionVQA",
            "num_frames": self.model.num_frames,
            "frame_size": MODEL_FRAME_SIZE,
            "visual_signal_gate": {
                "black_mean_threshold": BLACK_MEAN_THRESHOLD,
                "black_p99_threshold": BLACK_P99_THRESHOLD,
                "invalid_frame_policy": "nearest_valid_frame_with_masked_pooling",
                "fully_black_policy": "learned_no_visual_prefix",
            },
            "weighted_per_sample_loss": True,
            "qwen_model": str(getattr(self.model, "qwen_model_id", DEFAULT_QWEN_MODEL)),
            "surgmotion_checkpoint": str(
                getattr(self.model, "surgmotion_checkpoint_id", DEFAULT_SURGMOTION_CHECKPOINT)
            ),
            "surgmotion_delta_tensors": len(delta_keys),
            "train_surgmotion_blocks": getattr(
                self.model, "train_surgmotion_blocks", 0
            ),
            "qwen_frozen": getattr(self.model, "qwen_frozen", False),
            "projector_tokens": getattr(self.model, "projector_tokens", 64),
        }
        with open(output_dir / "surgmotion_vqa_config.json", "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        load_component_checkpoint(model or self.model, resume_from_checkpoint)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json", default=str(DEFAULT_TRAIN_JSON))
    parser.add_argument("--val-json", default=str(DEFAULT_VAL_JSON))
    parser.add_argument(
        "--cache-root",
        default=None,
        help="Defaults to frame_cache_<num_frames>fr under the project root.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--qwen-model", default=str(DEFAULT_QWEN_MODEL))
    parser.add_argument("--surgmotion-checkpoint", default=str(DEFAULT_SURGMOTION_CHECKPOINT))
    parser.add_argument("--num-frames", type=int, default=NUM_FRAMES)
    parser.add_argument(
        "--train-surgmotion-blocks",
        type=int,
        default=0,
        help="Unfreeze the final N ViT blocks plus encoder norm (Stage A).",
    )
    parser.add_argument(
        "--projector-tokens",
        type=int,
        default=64,
        help="Visual-prefix token budget the VisualProjector pools SurgMotion's "
        "token grid down to. Must divide num_frames*128 (2048 at 16fr, 8192 at "
        "64fr) for the fast exact-reshape pooling path.",
    )
    parser.add_argument(
        "--freeze-qwen",
        action="store_true",
        help="Freeze Qwen LoRA while adapting SurgMotion/projector in Stage A.",
    )
    parser.add_argument(
        "--initial-checkpoint",
        default=None,
        help="Load component weights but start a fresh optimizer/scheduler.",
    )
    parser.add_argument("--prepare-missing-cache", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    # The transferred corpus currently resides on one disk; one sequential reader is
    # faster and more stable there. More workers remain available for SSD/RAID hosts.
    parser.add_argument("--cache-workers", type=int, default=1)
    parser.add_argument("--cache-decode-chunk-size", type=int, default=64)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--grad-acc", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-text-length", type=int, default=256)
    parser.add_argument("--black-mean-threshold", type=float, default=BLACK_MEAN_THRESHOLD)
    parser.add_argument("--black-p99-threshold", type=float, default=BLACK_P99_THRESHOLD)
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
    if args.num_frames <= 0 or args.num_frames % 2:
        raise ValueError("--num-frames must be a positive multiple of tubelet_size=2")
    if not 0 <= args.train_surgmotion_blocks <= 24:
        raise ValueError("--train-surgmotion-blocks must be in [0, 24]")
    if args.cache_root is None:
        args.cache_root = str(PROJECT_ROOT / f"frame_cache_{args.num_frames}fr")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_data = load_json(args.train_json)
    val_data = load_json(args.val_json)
    all_segments = unique_segments(train_data, val_data)
    complete, missing = cache_preflight(
        all_segments, args.cache_root, num_frames=args.num_frames
    )
    print(
        f"[preflight] unique_segments={len(all_segments)} "
        f"complete_cache={complete} missing_cache={len(missing)}"
    )

    if missing and args.prepare_missing_cache:
        extract_missing_cache(
            all_segments,
            args.cache_root,
            num_frames=args.num_frames,
            decode_chunk_size=args.cache_decode_chunk_size,
            cache_workers=args.cache_workers,
        )
        complete, missing = cache_preflight(
        all_segments, args.cache_root, num_frames=args.num_frames
    )
        print(f"[preflight] after extraction: complete={complete} missing={len(missing)}")
    if args.preflight_only:
        if missing:
            print(
                f"[preflight] NOT READY: {len(missing)} segments still need cache extraction"
            )
        else:
            print(
                f"[preflight] READY: every requested segment has "
                f"{args.num_frames} cached frames"
            )
        return
    if missing:
        raise RuntimeError(
            f"{len(missing)} segments lack a complete {args.num_frames}-frame cache. "
            "Run again with --prepare-missing-cache."
        )
    if args.cache_only:
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full SurgMotionVQA training")
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
            "Refusing to train while another GPU compute process is active: "
            + "; ".join(conflicting)
        )

    tokenizer = AutoTokenizer.from_pretrained(args.qwen_model, local_files_only=True)
    tokenizer.padding_side = "right"
    train_dataset = SurgMotionQADataset(
        args.train_json,
        args.cache_root,
        is_train=True,
        num_frames=args.num_frames,
        black_mean_threshold=args.black_mean_threshold,
        black_p99_threshold=args.black_p99_threshold,
    )
    val_dataset = SurgMotionQADataset(
        args.val_json,
        args.cache_root,
        is_train=False,
        num_frames=args.num_frames,
        black_mean_threshold=args.black_mean_threshold,
        black_p99_threshold=args.black_p99_threshold,
    )
    collator = SurgMotionCollator(tokenizer, max_text_length=args.max_text_length)

    print(
        f"[model] loading SurgMotion({args.num_frames}fr, "
        f"train_last={args.train_surgmotion_blocks}) + Qwen2.5-VL "
        f"(frozen={args.freeze_qwen})"
    )
    model = build(
        surgmotion_checkpoint=args.surgmotion_checkpoint,
        qwen_model_id=args.qwen_model,
        num_frames=args.num_frames,
        local_files_only=True,
        train_surgmotion_blocks=args.train_surgmotion_blocks,
        freeze_qwen=args.freeze_qwen,
        projector_tokens=args.projector_tokens,
    )
    model.qwen_model_id = args.qwen_model
    model.surgmotion_checkpoint_id = args.surgmotion_checkpoint
    model.train_surgmotion_blocks = args.train_surgmotion_blocks
    model.qwen_frozen = args.freeze_qwen
    model.projector_tokens = args.projector_tokens
    if args.initial_checkpoint:
        load_component_checkpoint(model, args.initial_checkpoint)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"[model] trainable={trainable:,} total={total:,} ({100 * trainable / total:.3f}%)")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
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

    trainer = ComponentCheckpointTrainer(
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
