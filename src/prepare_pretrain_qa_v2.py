"""Domain-adaptive "Stage 0" QA pretraining corpus, v2 - rebuild of
prepare_pretrain_qa.py targeting the CURRENT DualEncoderVQA (Qwen2.5-VL-3B)
pipeline instead of the old SurgMotionVQA (Qwen2-VL-2B) one. See project
memory `surgvu_prostatdv2_domain_adapt_contingency` for why a rewrite was
needed: the old pretrain checkpoint's base model doesn't match, and the
local ProstaTDv2 copy was corrupted (0-byte files) until re-downloaded.

Source: ProstaTDv2/v1.1 (robotic prostatectomy, real per-frame
<instrument, action, target> triplet labels via labelme-style JSON) -
`CholecT50/ProstaTDv2/{ProstaTDv1.1_cls,ProstaTDv2_cls}/<case>/<id>.json(+.jpg)`.
SurgVU's own tools.csv has no action/target labels at all, so this is the
only source of genuine verb/target supervision available; kept as a
*separate* pretraining stage (not mixed into SurgVU ground truth) and used
only via `--initial-checkpoint` warm-start, never claiming ProstaTD
triplets are true for SurgVU.

Each labeled image becomes a fake "segment" (video="prostatd://<case>/<id>",
t_start=0.0, t_end=1.0 - unique per image, shared across every shape drawn
from that image) with MAX_FRAMES duplicated copies of the same frame,
written directly into the SAME cache-folder format `train_dual_encoder_vqa.py`
reads (`cache_root/md5(video|t_start|t_end)/frame_{0..N}.jpg`, matching
`train_surgmotion_vqa.py::segment_hash`/`cache_dir_for`), so no video
decoding step or `--prepare-missing-cache` pass is needed - the cache is
already "complete" the moment this script finishes.

IMPORTANT (see the contingency memory): answers are kept to ONE short
sentence, matching the SurgVU public-sample register the 2026-08-13
description-answer-length fix aligned to - NOT the longer multi-clause
style CholecT50/ProstaTD triplet descriptions could tempt toward.
"""
import argparse
import glob
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROSTATD_ROOT = PROJECT_ROOT / "CholecT50" / "ProstaTDv2"
PROSTATD_DIRS = [
    PROSTATD_ROOT / "ProstaTDv2_cls",
    PROSTATD_ROOT / "ProstaTDv1.1_cls",
]
DEFAULT_CACHE_ROOT = PROJECT_ROOT / "frame_cache_16fr"
DEFAULT_TRAIN_OUTPUT = PROJECT_ROOT / "src" / "train_vqa_pretrain_prostatd_v2.json"
DEFAULT_VAL_OUTPUT = PROJECT_ROOT / "src" / "val_vqa_pretrain_prostatd_v2.json"
NUM_FRAMES = 16
# Must match real SurgVU cached frames (frame_cache_16fr/*/frame_*.jpg are 384x384,
# see extract_frames.py) as closely as possible: a 2026-08-16 smoke test found that
# caching at a larger size (640 max-side) produced ~1600 Qwen video tokens per
# example vs real SurgVU's ~800-900 (confirmed 811 at exactly 384x384) - the
# processor's min/max_pixels constraint on the video path does not clamp this as
# tightly as expected, so matching SurgVU's own cached resolution is the reliable
# fix, not a larger --max-text-length (which just traded the crash for an OOM).
CACHE_IMAGE_SIDE = 384

# Held out entirely for an internal train/val split (not part of any SurgVU gate).
VAL_CASES = {"esadv4", "psiv21", "pwhv9"}

INSTRUMENT_PHRASE = {
    "scissors": "scissors",
    "forceps": "forceps",
    "aspirator": "an aspirator",
    "needle driver": "a needle driver",
    "grasper": "a grasper",
    "clip applier": "a clip applier",
}

# action -> (gerund, third_person_singular). "suck" is rephrased to the more
# clinical "aspirate" since its target is always "fluid".
ACTION_FORMS = {
    "retract": ("retracting", "retracts"),
    "dissect": ("dissecting", "dissects"),
    "suck": ("aspirating", "aspirates"),
    "cut": ("cutting", "cuts"),
    "grasp": ("grasping", "grasps"),
    "clip": ("clipping", "clips"),
    "coagulate": ("coagulating", "coagulates"),
}

INVALID_VALUES = {"null", "?", "", None}

QUESTION_TEMPLATES = [
    "What is the surgeon doing in this step?",
    "Describe what is happening in this surgical clip.",
    "What activity is the surgeon performing?",
]


def segment_hash(video, t_start, t_end):
    return hashlib.md5(f"{video}|{t_start}|{t_end}".encode("utf-8")).hexdigest()


def load_shapes():
    """Yields (case_id, raw_case_dir, image_path, instrument, action, target)
    for every valid, mappable annotation. case_id is a display-only label
    (root_tag + case_dir); raw_case_dir is kept separate for the VAL_CASES
    membership check since root_tag itself contains underscores (e.g.
    "ProstaTDv1.1_cls"), making any single-underscore split of the
    concatenated case_id ambiguous."""
    for root_dir in PROSTATD_DIRS:
        if not root_dir.is_dir():
            continue
        root_tag = root_dir.name
        for case_dir in sorted(os.listdir(root_dir)):
            case_path = root_dir / case_dir
            if not case_path.is_dir():
                continue
            case_id = f"{root_tag}_{case_dir}"
            for json_path in sorted(glob.glob(str(case_path / "*.json"))):
                img_path = os.path.splitext(json_path)[0] + ".jpg"
                if not os.path.isfile(img_path):
                    continue
                with open(json_path, "r", encoding="utf-8") as handle:
                    record = json.load(handle)
                for shape in record.get("shapes", []):
                    instrument = shape.get("label")
                    attrs = shape.get("attributes", {})
                    action = attrs.get("Action")
                    target = attrs.get("Target")
                    if instrument not in INSTRUMENT_PHRASE:
                        continue
                    if action in INVALID_VALUES or action not in ACTION_FORMS:
                        continue
                    if target in INVALID_VALUES:
                        continue
                    yield case_id, case_dir, img_path, instrument, action, target


def build_answers(instrument, action, target):
    phrase = INSTRUMENT_PHRASE[instrument]
    gerund, third_person = ACTION_FORMS[action]
    phrase_cap = phrase[0].upper() + phrase[1:]
    return [
        f"The surgeon is using {phrase} to {action} the {target}.",
        f"{phrase_cap} is used to {action} the {target}.",
        f"The surgeon {third_person} the {target} using {phrase}.",
        f"This step involves {gerund} the {target} with {phrase}.",
        f"Using {phrase}, the surgeon {third_person} the {target}.",
    ]


def write_frame_cache(image_path, cache_dir):
    if cache_dir.is_dir() and (cache_dir / f"frame_{NUM_FRAMES - 1}.jpg").is_file():
        return  # already written (e.g. a second shape on the same image)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        # Fixed square resize (not aspect-preserving) - matches real SurgVU cached
        # frames exactly (also a fixed 384x384 square, distorting the source
        # video's own ~16:9 aspect - this is consistent with that, not a new
        # distortion introduced here) and reproduces the verified-safe token count.
        image = image.resize((CACHE_IMAGE_SIDE, CACHE_IMAGE_SIDE))
        for index in range(NUM_FRAMES):
            image.save(cache_dir / f"frame_{index}.jpg", quality=90)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--train-output", default=str(DEFAULT_TRAIN_OUTPUT))
    parser.add_argument("--val-output", default=str(DEFAULT_VAL_OUTPUT))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    cache_root = Path(args.cache_root)

    train_rows, val_rows = [], []
    image_cache_written = set()
    counts_by_case = defaultdict(int)

    shapes = list(load_shapes())
    print(f"[prepare_pretrain_qa_v2] {len(shapes)} valid (instrument, action, target) shapes")

    for position, (case_id, raw_case, img_path, instrument, action, target) in enumerate(
        tqdm(shapes, desc="Building pretrain QA")
    ):
        video = f"prostatd://{case_id}/{os.path.basename(img_path)}"
        t_start, t_end = 0.0, 1.0
        cache_dir = cache_root / segment_hash(video, t_start, t_end)
        if img_path not in image_cache_written:
            write_frame_cache(img_path, cache_dir)
            image_cache_written.add(img_path)

        question = QUESTION_TEMPLATES[position % len(QUESTION_TEMPLATES)]
        answers = build_answers(instrument, action, target)
        entry = {
            "id": f"prostatd_{case_id}_{os.path.splitext(os.path.basename(img_path))[0]}_{position}",
            "video": video,
            "t_start": t_start,
            "t_end": t_end,
            "question": question,
            "answers": answers,
        }

        counts_by_case[case_id] += 1
        if raw_case in VAL_CASES:
            val_rows.append(entry)
        else:
            train_rows.append(entry)

    Path(args.train_output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(train_rows, open(args.train_output, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    json.dump(val_rows, open(args.val_output, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"[prepare_pretrain_qa_v2] train={len(train_rows)} val={len(val_rows)} unique_images={len(image_cache_written)}")
    print(f"[prepare_pretrain_qa_v2] wrote {args.train_output}, {args.val_output}")


if __name__ == "__main__":
    main()
