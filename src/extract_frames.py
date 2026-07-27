"""
Pre-extracts and caches the video frames used by train_vqa.json / val_vqa.json to disk,
so training reads small JPEGs instead of decoding multi-hour case videos via decord on
every __getitem__. This is a one-time, single-process pass (safe: decord + multiprocessing
DataLoader workers crash on this machine - see train.py) that also de-duplicates work,
since several QA pairs share the same (video, t_start, t_end) segment.
"""
import os
import json
import hashlib
import numpy as np
import cv2
import decord
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict

MAX_FRAMES = 16
FRAME_SIZE = 384
VR_CACHE_SIZE = 4


def segment_hash(video, t_start, t_end):
    return hashlib.md5(f"{video}|{t_start}|{t_end}".encode("utf-8")).hexdigest()


def extract_segment_frames(vr, t_start, t_end, max_frames=MAX_FRAMES, frame_size=FRAME_SIZE):
    fps = vr.get_avg_fps()
    start_frame = int(t_start * fps)
    end_frame = int(t_end * fps)

    frame_indices = np.linspace(start_frame, end_frame - 1, max_frames, dtype=int)
    frame_indices = np.clip(frame_indices, 0, len(vr) - 1)
    frames_npy = vr.get_batch(frame_indices).asnumpy()

    # Mask bottom 15% (UI overlay) - must match train.py / predict.py exactly
    H, W = frames_npy.shape[1], frames_npy.shape[2]
    mask_y = int(0.85 * H)
    frames_npy[:, mask_y:, :, :] = 0

    frames = []
    for f in frames_npy:
        f_resized = cv2.resize(f, (frame_size, frame_size))
        frames.append(Image.fromarray(f_resized))
    return frames


def main():
    base = r"d:\SCLab\Surgical-Video-Understanding\SurgVU-challenge"
    train_path = os.path.join(base, "src", "train_vqa.json")
    val_path = os.path.join(base, "src", "val_vqa.json")
    cache_dir = os.path.join(base, "frame_cache_16fr")

    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(val_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    segments = OrderedDict()
    for item in train_data + val_data:
        key = (item["video"], item["t_start"], item["t_end"])
        segments.setdefault(key, []).append(item)

    os.makedirs(cache_dir, exist_ok=True)

    vr_cache = OrderedDict()

    def get_reader(path):
        vr = vr_cache.get(path)
        if vr is not None:
            vr_cache.move_to_end(path)
            return vr
        vr = decord.VideoReader(path)
        vr_cache[path] = vr
        if len(vr_cache) > VR_CACHE_SIZE:
            vr_cache.popitem(last=False)
        return vr

    # Process grouped by video regardless of the QA JSON's own ordering, so the small
    # VideoReader LRU cache actually gets reuse instead of thrashing between different
    # multi-hour case videos on every item (this alone is a >10x speed difference).
    ordered_segments = sorted(segments.items(), key=lambda kv: (kv[0][0], kv[0][1]))

    failed = 0
    skipped = 0
    for (video, t_start, t_end), items in tqdm(ordered_segments, desc="Extracting frames"):
        seg_dir = os.path.join(cache_dir, segment_hash(video, t_start, t_end))
        frame_paths = [os.path.join(seg_dir, f"frame_{i}.jpg") for i in range(MAX_FRAMES)]

        if all(os.path.exists(p) for p in frame_paths):
            skipped += 1
        else:
            os.makedirs(seg_dir, exist_ok=True)
            try:
                vr = get_reader(video)
                frames = extract_segment_frames(vr, t_start, t_end)
            except Exception as e:
                print(f"Error extracting {video} [{t_start}-{t_end}]: {e}")
                failed += 1
                frames = [Image.new("RGB", (FRAME_SIZE, FRAME_SIZE), (0, 0, 0)) for _ in range(MAX_FRAMES)]
            for p, frame in zip(frame_paths, frames):
                frame.save(p, quality=90)

        for item in items:
            item["frames_dir"] = seg_dir

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)

    print(f"\n{len(segments)} unique segments ({skipped} already cached, {failed} failed -> black frames)")
    print(f"Frame cache: {cache_dir}")
    print(f"Updated {train_path} and {val_path} with 'frames_dir'")


if __name__ == "__main__":
    main()
