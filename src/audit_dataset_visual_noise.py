"""Audit black/near-black visual segments over all SurgVU QA windows."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path, PureWindowsPath

import cv2
import decord
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def segment_key(item):
    return (item["video"], float(item["t_start"]), float(item["t_end"]))


def resolve_video(raw_path):
    native = Path(raw_path)
    if native.is_file():
        return native
    parts = list(PureWindowsPath(raw_path).parts)
    lowered = [part.lower() for part in parts]
    index = lowered.index("surgvu24")
    candidate = PROJECT_ROOT.joinpath(*parts[index:])
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def cache_candidates(item, cache_root):
    raw = item["video"]
    values = [raw]
    native_parts = list(Path(raw).parts)
    lowered = [part.lower() for part in native_parts]
    if "surgvu24" in lowered:
        index = lowered.index("surgvu24")
        legacy_root = PureWindowsPath(
            r"d:\SCLab\Surgical-Video-Understanding\SurgVU-challenge"
        )
        values.append(str(legacy_root.joinpath(*native_parts[index:])))
    folders = []
    for video in values:
        digest = hashlib.md5(
            f"{video}|{item['t_start']}|{item['t_end']}".encode("utf-8")
        ).hexdigest()
        folder = Path(cache_root) / digest
        if folder not in folders:
            folders.append(folder)
    return folders


def frame_metrics(frame):
    active = frame[: int(frame.shape[0] * 0.85)]
    small = cv2.resize(active, (96, 96), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    return {
        "mean": float(gray.mean()),
        "std": float(gray.std()),
        "p99": float(np.percentile(gray, 99)),
        "max": int(active.max()),
        "nonzero": float(np.count_nonzero(gray > 3) / gray.size),
    }


def audit_raw_video(payload):
    """Decode requested timestamps from one video in an isolated worker."""
    video_path, entries, samples_per_segment, decode_chunk_size = payload
    reader = decord.VideoReader(str(video_path), num_threads=1)
    fps = float(reader.get_avg_fps())
    total = len(reader)
    destinations = defaultdict(list)
    output = {}
    for key, start, end in entries:
        output[key] = [None] * samples_per_segment
        start_frame = int(start * fps)
        end_frame = int(end * fps)
        indices = np.linspace(
            start_frame,
            max(end_frame - 1, start_frame),
            samples_per_segment,
            dtype=int,
        )
        indices = np.clip(indices, 0, total - 1)
        for position, source_index in enumerate(indices):
            destinations[int(source_index)].append((key, position))

    ordered = sorted(destinations)
    for offset in range(0, len(ordered), decode_chunk_size):
        chunk = ordered[offset : offset + decode_chunk_size]
        frames = reader.get_batch(np.asarray(chunk, dtype=np.int64)).asnumpy()
        for source_index, frame in zip(chunk, frames):
            metrics = frame_metrics(frame)
            for key, position in destinations[source_index]:
                output[key][position] = metrics
    return output


def audit_raw_frame_ffmpeg(payload):
    """Fast-seek one midpoint frame with NVDEC and return grayscale metrics."""
    key, video_path, start, end = payload
    timestamp = (start + end) / 2
    environment = os.environ.copy()
    environment.pop("LD_LIBRARY_PATH", None)
    command = [
        "/usr/bin/ffmpeg",
        "-hwaccel", "cuda",
        "-hide_banner", "-loglevel", "error",
        "-ss", str(timestamp),
        "-i", video_path,
        "-frames:v", "1",
        "-vf", "crop=iw:trunc(ih*0.85/2)*2:0:0,scale=96:96",
        "-pix_fmt", "gray",
        "-f", "rawvideo",
        "pipe:1",
    ]
    completed = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    gray = np.frombuffer(completed.stdout, dtype=np.uint8)
    if gray.size != 96 * 96:
        raise RuntimeError(
            f"Expected 9216 grayscale bytes, got {gray.size}: {video_path} @ {timestamp}"
        )
    return key, {
        "mean": float(gray.mean()),
        "std": float(gray.std()),
        "p99": float(np.percentile(gray, 99)),
        "max": int(gray.max()),
        "nonzero": float(np.count_nonzero(gray > 3) / gray.size),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json", default=str(PROJECT_ROOT / "src/train_vqa.json"))
    parser.add_argument("--val-json", default=str(PROJECT_ROOT / "src/val_vqa.json"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "runs/dataset_audit/black_screen_audit.json"))
    parser.add_argument("--cache-root", default=str(PROJECT_ROOT / "frame_cache_16fr"))
    parser.add_argument(
        "--fallback-cache-root",
        default=str(PROJECT_ROOT / "frame_cache"),
        help="Optional older 8-frame cache checked before decoding raw video.",
    )
    parser.add_argument("--samples-per-segment", type=int, default=5)
    parser.add_argument(
        "--raw-samples-per-segment",
        type=int,
        default=1,
        help="Samples for cache misses; one midpoint keeps a full-corpus audit tractable.",
    )
    parser.add_argument("--decode-chunk-size", type=int, default=16)
    parser.add_argument("--raw-workers", type=int, default=8)
    parser.add_argument(
        "--raw-backend",
        choices=("ffmpeg_cuda", "decord"),
        default="ffmpeg_cuda",
    )
    args = parser.parse_args()

    split_data = {"train": load_json(args.train_json), "val": load_json(args.val_json)}
    segments = {}
    qa_counts = defaultdict(lambda: defaultdict(int))
    for split, rows in split_data.items():
        for item in rows:
            key = segment_key(item)
            segments.setdefault(key, item)
            qa_counts[key][split] += 1

    cache_root = Path(args.cache_root)
    fallback_cache_root = Path(args.fallback_cache_root)
    grouped = defaultdict(list)
    results = {}
    progress = tqdm(total=len(segments), desc="Auditing visual signal")
    cache_positions = np.linspace(0, 15, args.samples_per_segment, dtype=int)
    cached_segments = 0

    for key, item in segments.items():
        case_match = re.search(r"case_\d+", item["video"])
        results[key] = {
            "case": case_match.group(0) if case_match else "unknown",
            "video": item["video"],
            "t_start": float(item["t_start"]),
            "t_end": float(item["t_end"]),
            "qa_train": qa_counts[key]["train"],
            "qa_val": qa_counts[key]["val"],
            "source": None,
            "frames": [None] * args.samples_per_segment,
        }
        folder = next(
            (
                candidate
                for candidate in cache_candidates(item, cache_root)
                if all((candidate / f"frame_{index}.jpg").is_file() for index in range(16))
            ),
            None,
        )
        fallback_folder = next(
            (
                candidate
                for candidate in cache_candidates(item, fallback_cache_root)
                if all((candidate / f"frame_{index}.jpg").is_file() for index in range(8))
            ),
            None,
        )
        if folder is not None:
            selected_folder = folder
            selected_positions = cache_positions
            source = "cache16"
        elif fallback_folder is not None:
            selected_folder = fallback_folder
            selected_positions = np.linspace(0, 7, args.samples_per_segment, dtype=int)
            source = "cache8"
        else:
            selected_folder = None

        if selected_folder is not None:
            for position, cache_index in enumerate(selected_positions):
                frame_path = selected_folder / f"frame_{cache_index}.jpg"
                frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                if frame is None:
                    raise RuntimeError(f"Unreadable cached frame: {frame_path}")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results[key]["frames"][position] = frame_metrics(frame)
            results[key]["source"] = source
            cached_segments += 1
            progress.update(1)
        else:
            grouped[resolve_video(item["video"])].append((key, item))

    print(f"[audit] cache_segments={cached_segments} raw_segments={len(segments)-cached_segments}")
    if args.raw_backend == "ffmpeg_cuda":
        if args.raw_samples_per_segment != 1:
            raise ValueError("ffmpeg_cuda currently requires --raw-samples-per-segment 1")
        payloads = [
            (
                key,
                str(video_path),
                float(item["t_start"]),
                float(item["t_end"]),
            )
            for video_path in sorted(grouped, key=str)
            for key, item in grouped[video_path]
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.raw_workers) as executor:
            futures = [executor.submit(audit_raw_frame_ffmpeg, payload) for payload in payloads]
            for future in concurrent.futures.as_completed(futures):
                key, metrics = future.result()
                results[key]["frames"] = [metrics]
                results[key]["source"] = "raw_ffmpeg_cuda"
                progress.update(1)
    else:
        payloads = []
        for video_path in sorted(grouped, key=str):
            entries = [
                (key, float(item["t_start"]), float(item["t_end"]))
                for key, item in grouped[video_path]
            ]
            payloads.append(
                (str(video_path), entries, args.raw_samples_per_segment, args.decode_chunk_size)
            )

        with concurrent.futures.ProcessPoolExecutor(max_workers=args.raw_workers) as executor:
            futures = [executor.submit(audit_raw_video, payload) for payload in payloads]
            for future in concurrent.futures.as_completed(futures):
                decoded = future.result()
                for key, frames in decoded.items():
                    results[key]["frames"] = frames
                    results[key]["source"] = "raw_decord"
                progress.update(len(decoded))
    progress.close()
    serializable = []
    for result in results.values():
        frames = result["frames"]
        if any(frame is None for frame in frames):
            raise RuntimeError(f"Incomplete audit row: {result}")
        exact_flags = [frame["max"] <= 1 for frame in frames]
        near_flags = [frame["mean"] <= 3 and frame["p99"] <= 10 for frame in frames]
        severe_flags = [frame["mean"] <= 8 and frame["p99"] <= 25 for frame in frames]
        result.update({
            "n_exact_black": sum(exact_flags),
            "n_near_black": sum(near_flags),
            "n_severely_dark": sum(severe_flags),
            "all_exact_black": all(exact_flags),
            "all_near_black": all(near_flags),
            "all_severely_dark": all(severe_flags),
            "any_exact_black": any(exact_flags),
            "any_near_black": any(near_flags),
        })
        serializable.append(result)

    summary = {
        "segments": len(serializable),
        "qa_examples": sum(x["qa_train"] + x["qa_val"] for x in serializable),
    }
    for field in (
        "all_exact_black", "all_near_black", "all_severely_dark",
        "any_exact_black", "any_near_black",
    ):
        selected = [row for row in serializable if row[field]]
        summary[field] = {
            "segments": len(selected),
            "qa_examples": sum(row["qa_train"] + row["qa_val"] for row in selected),
            "cases": len({row["case"] for row in selected}),
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "segments": serializable}, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"saved {output}")


if __name__ == "__main__":
    main()
