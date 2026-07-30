"""Quantifies how much of each 30s segment's sampled frames actually fall within the
time range of the task/description text it's labeled with.

preprocess.py assigns a segment the task with the MAXIMUM OVERLAP with its
[t_start, t_end] window - but "maximum overlap" can still be a small fraction of the
30s window (e.g. a task that only occupies the last 8s of the window still "wins" if
no other task overlaps more). Since extract_frames.py samples 16 frames uniformly
across the FULL window regardless of where the winning task's own time range falls,
a low overlap fraction means most of what the model actually sees does not depict
the task/description text it's being trained to associate with those frames - exactly
the "noise" suspected. This script recomputes the true overlap fraction per segment
directly from tasks.csv (surgvu_segments.json only kept the winning task's name/desc,
not this fraction) and reports its distribution, broken down by task type.
"""
import os
import glob
import json
import pandas as pd
from collections import defaultdict

SURGVU24_DIR = r"d:\SCLab\Surgical-Video-Understanding\SurgVU-challenge\surgvu24"
LABELS_DIR = r"d:\SCLab\Surgical-Video-Understanding\SurgVU-challenge\SURGVU25_train_labels"
SEGMENTS_JSON = r"d:\SCLab\Surgical-Video-Understanding\SurgVU-challenge\src\surgvu_segments.json"


def main():
    with open(SEGMENTS_JSON, encoding="utf-8") as f:
        segments = json.load(f)

    # Group segments by case so we only load each case's tasks.csv once
    by_case = defaultdict(list)
    for seg in segments:
        by_case[seg["case"]].append(seg)

    results = []
    for case, segs in by_case.items():
        tasks_path = os.path.join(LABELS_DIR, case, "tasks.csv")
        if not os.path.exists(tasks_path):
            continue
        tasks_df = pd.read_csv(tasks_path)

        for seg in segs:
            part_id = seg["part"]
            t_start, t_end = seg["t_start"], seg["t_end"]
            window = t_end - t_start

            best_overlap = 0.0
            best_task_start, best_task_end = None, None
            for _, row in tasks_df.iterrows():
                start_part = int(row.get("start_part", 1))
                stop_part = int(row.get("stop_part", 1))
                if not (start_part <= part_id <= stop_part):
                    continue
                row_start = float(row.get("start_time", 0.0))
                row_end = float(row.get("stop_time", 0.0))
                overlap = min(t_end, row_end) - max(t_start, row_start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_task_start, best_task_end = row_start, row_end

            if seg["task"] == "Other":
                continue  # "Other" has no single task interval to compare against

            frac = best_overlap / window if window > 0 else 0.0
            results.append({
                "case": case, "task": seg["task"], "t_start": t_start, "t_end": t_end,
                "overlap_frac": frac,
            })

    df = pd.DataFrame(results)
    print(f"Segments with a non-'Other' task label: {len(df)}")
    print()
    print("=== Overlap fraction distribution (fraction of the 30s window actually")
    print("    covered by the task interval it's labeled with) ===")
    bins = [0, 0.25, 0.5, 0.75, 1.0, 1.01]
    labels = ["0-25%", "25-50%", "50-75%", "75-100%", "100%+(clip)"]
    df["bucket"] = pd.cut(df["overlap_frac"], bins=bins, labels=labels, right=False)
    print(df["bucket"].value_counts().sort_index())
    print()
    print(f"Mean overlap fraction: {df['overlap_frac'].mean():.3f}")
    print(f"Median overlap fraction: {df['overlap_frac'].median():.3f}")
    print(f"Segments with <50% overlap (majority of frames NOT depicting the labeled task): "
          f"{(df['overlap_frac'] < 0.5).sum()} / {len(df)} "
          f"({(df['overlap_frac'] < 0.5).mean():.1%})")
    print()
    print("=== Breakdown by task ===")
    print(df.groupby("task")["overlap_frac"].agg(["mean", "median", "count"]).sort_values("mean"))


if __name__ == "__main__":
    main()
