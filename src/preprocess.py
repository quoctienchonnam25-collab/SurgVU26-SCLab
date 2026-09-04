import argparse
import glob
import json
import os
from pathlib import Path

import decord
import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO_ROOT = PROJECT_ROOT / "surgvu24"
DEFAULT_LABEL_ROOT = PROJECT_ROOT / "surgvu24_labels_updated_v2" / "labels"
DEFAULT_DESCRIPTION_ROOT = PROJECT_ROOT / "SURGVU25_train_labels"
DEFAULT_OUTPUT = PROJECT_ROOT / "src" / "surgvu_segments_v2.json"

def time_to_seconds(t_str):
    if pd.isna(t_str) or not isinstance(t_str, str):
        try:
            val = float(t_str)
            return val if not pd.isna(val) else 0.0
        except ValueError:
            return 0.0
    t_str = t_str.strip()
    if not t_str or t_str == 'nan':
        return 0.0
    try:
        parts = t_str.split(':')
        if len(parts) == 3:
            h, m, s = parts
            return float(h) * 3600 + float(m) * 60 + float(s)
        elif len(parts) == 2:
            m, s = parts
            return float(m) * 60 + float(s)
        else:
            return float(t_str)
    except Exception:
        return 0.0

def _normalise_join_value(value):
    if pd.isna(value):
        return None
    try:
        number = float(value)
        return int(number) if number.is_integer() else round(number, 6)
    except (TypeError, ValueError):
        return str(value).strip().lower()


def _task_join_key(row):
    return (
        _normalise_join_value(row.get("start_part")),
        _normalise_join_value(row.get("start_time")),
        _normalise_join_value(row.get("stop_part")),
        _normalise_join_value(row.get("stop_time")),
        _normalise_join_value(row.get("groundtruth_taskname")),
    )


def enrich_task_descriptions(tasks_df, description_case_dir):
    """Exact-join old descriptions onto authoritative updated-v2 task rows."""
    tasks_df = tasks_df.copy()
    tasks_df["matched_description"] = ""
    if not description_case_dir:
        return tasks_df
    old_path = os.path.join(description_case_dir, "tasks.csv")
    if not os.path.exists(old_path):
        return tasks_df

    old_tasks = pd.read_csv(old_path)
    descriptions = {}
    for _, row in old_tasks.iterrows():
        description = str(row.get("matched_description", "")).strip()
        if description and description.lower() != "nan":
            descriptions.setdefault(_task_join_key(row), description)
    tasks_df["matched_description"] = [
        descriptions.get(_task_join_key(row), "") for _, row in tasks_df.iterrows()
    ]
    return tasks_df


def load_case_labels(case_dir, case_id, description_case_dir=None):
    # Load tasks.csv
    tasks_path = os.path.join(case_dir, "tasks.csv")
    if os.path.exists(tasks_path):
        tasks_df = pd.read_csv(tasks_path)
        tasks_df = enrich_task_descriptions(tasks_df, description_case_dir)
    else:
        tasks_df = pd.DataFrame()

    # Load tools.csv
    tools_path = os.path.join(case_dir, "tools.csv")
    if os.path.exists(tools_path):
        tools_df = pd.read_csv(tools_path)
    else:
        tools_df = pd.DataFrame()

    return tasks_df, tools_df

def task_interval_in_part(row, part_id, duration):
    """Express a possibly cross-part task in the current video's local timeline."""
    start_part = int(float(row.get("start_part", 1)))
    stop_part = int(float(row.get("stop_part", 1)))
    if not start_part <= part_id <= stop_part:
        return None
    row_start = time_to_seconds(row.get("start_time", 0.0))
    row_end = time_to_seconds(row.get("stop_time", 0.0))
    local_start = row_start if part_id == start_part else 0.0
    local_end = row_end if part_id == stop_part else duration
    local_start = min(max(local_start, 0.0), duration)
    local_end = min(max(local_end, 0.0), duration)
    if local_end <= local_start:
        return None
    return local_start, local_end


def process_case(
    case_name,
    surgvu24_dir,
    labels_dir,
    description_labels_dir=None,
    window_size=30.0,
    step_size=15.0,
):
    case_video_dir = os.path.join(surgvu24_dir, case_name)
    case_label_dir = os.path.join(labels_dir, case_name)

    description_case_dir = (
        os.path.join(description_labels_dir, case_name)
        if description_labels_dir
        else None
    )
    tasks_df, tools_df = load_case_labels(
        case_label_dir, case_name, description_case_dir=description_case_dir
    )
    
    # Find video parts
    video_paths = glob.glob(os.path.join(case_video_dir, "*.mp4"))
    video_paths.sort()

    segments = []
    
    # Target tools
    target_tools = [
        "needle driver",
        "monopolar curved scissors",
        "force bipolar",
        "clip applier",
        "cadiere forceps",
        "bipolar forceps",
        "vessel sealer",
        "permanent cautery hook/spatula",
        "prograsp forceps",
        "stapler",
        "grasping retractor",
        "tip-up fenestrated grasper"
    ]

    for video_path in video_paths:
        # Extract part number from video filename
        # e.g., case_000_video_part_001.mp4 -> part 1
        video_filename = os.path.basename(video_path)
        try:
            part_str = video_filename.split("part_")[-1].replace(".mp4", "")
            part_id = int(part_str)
        except Exception:
            part_id = 1

        # Load video using cv2 to get fps and total frames
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            cap.release()
            duration = total_frames / fps if fps > 0 else 0.0
        except Exception as e:
            print(f"Error loading {video_path}: {e}")
            continue

        # Preserve the one valid video shorter than 30 seconds instead of silently
        # excluding it from full-data training.
        starts = []
        t = 0.0
        while t + window_size <= duration:
            starts.append(t)
            t += step_size
        if not starts and duration > 0:
            starts.append(0.0)

        for t in starts:
            t_start = t
            t_end = min(t + window_size, duration)
            
            # Determine active task
            # Filter tasks matching current part and overlapping with [t_start, t_end]
            active_task = "Other"
            task_desc = "Activity outside of the structured training."
            
            if not tasks_df.empty:
                # Find matching row in tasks_df
                # start_part/stop_part are floats in CSV
                overlapping_tasks = []
                for _, row in tasks_df.iterrows():
                    interval = task_interval_in_part(row, part_id, duration)
                    if interval is not None:
                        row_start, row_end = interval
                        overlap_start = max(t_start, row_start)
                        overlap_end = min(t_end, row_end)
                        if overlap_end > overlap_start:
                            overlap_dur = overlap_end - overlap_start
                            overlapping_tasks.append((overlap_dur, row.get('groundtruth_taskname', 'Other'), row.get('matched_description', '')))
                
                if overlapping_tasks:
                    # Select the task with the maximum overlap duration - but only if it
                    # actually covers a majority of the window. Without this threshold, a
                    # task that ends (or starts) a fraction of a second inside the window
                    # still "wins" whenever no other task overlaps at all, mislabeling the
                    # entire 30s window - and everything 16-frame-sampled from it - with a
                    # task/description that describes almost none of what's actually shown
                    # (found via analyze_description_alignment.py: some segments had as
                    # little as 0.02% real overlap). Below the threshold, fall back to the
                    # existing "Other" default instead of crediting a boundary sliver.
                    MIN_OVERLAP_FRACTION = 0.5
                    overlapping_tasks.sort(reverse=True, key=lambda x: x[0])
                    best_overlap_dur = overlapping_tasks[0][0]
                    segment_duration = t_end - t_start
                    if best_overlap_dur / segment_duration >= MIN_OVERLAP_FRACTION:
                        raw_task = overlapping_tasks[0][1]
                        active_task = (
                            "Other"
                            if pd.isna(raw_task) or not str(raw_task).strip()
                            else str(raw_task).strip()
                        )
                        task_desc = str(overlapping_tasks[0][2]).strip()
                        if not task_desc or task_desc.lower() == "nan":
                            task_desc = (
                                f"The training activity is {active_task}."
                                if active_task != "Other"
                                else "Activity outside of the structured training."
                            )

            # Determine active tools
            active_tools = set()
            active_commercial_tools = set()
            if not tools_df.empty:
                for _, row in tools_df.iterrows():
                    install_part = int(row.get('install_case_part', 1))
                    uninstall_part = int(row.get('uninstall_case_part', 1))

                    if install_part <= part_id <= uninstall_part:
                        # install_case_time/uninstall_case_time are timestamps within the
                        # install/uninstall part's own video, not the current part. For a
                        # part strictly between them the tool is active for the whole part;
                        # at the install/uninstall part itself, only the recorded edge applies.
                        tool_start = time_to_seconds(row.get('install_case_time', '00:00:00')) if part_id == install_part else 0.0
                        tool_end = time_to_seconds(row.get('uninstall_case_time', '00:00:00')) if part_id == uninstall_part else duration

                        # A recorded 00:00:00 (or otherwise missing) uninstall time is a data
                        # sentinel, not a real timestamp - treat it as "still installed" rather
                        # than silently dropping the tool from this segment.
                        #
                        # 2026-09-02: that sentinel does not actually occur in this dataset -
                        # zero rows have uninstall_case_time == 00:00:00. What DOES occur is
                        # 348 rows (within a single part) whose recorded uninstall time is a
                        # real timestamp that is simply EARLIER than the install time, i.e.
                        # corrupt. Extending those to `duration` marked the tool present from
                        # its install right through to the end of the video - a median of 19.4
                        # and a mean of 85.9 minutes of invented presence per row. Across the
                        # dataset that flipped 11,471 of 200,648 segments (5.7%), adding 16,175
                        # spurious positive tool labels, all of them in tool_presence, which is
                        # ~45% of the generated questions. Corrupt rows are now dropped; the
                        # sentinel branch is kept only for a genuinely zero/missing timestamp.
                        #
                        # Note this only bites when install_part == uninstall_part == part_id.
                        # For a row spanning parts the `else duration` above already applies,
                        # and its uninstall time being "earlier" is just the two timestamps
                        # living in different parts' timelines - those 424 rows are fine.
                        if tool_end <= tool_start:
                            if tool_end == 0.0:
                                tool_end = duration
                            else:
                                continue

                        # Check overlap
                        if min(t_end, tool_end) > max(t_start, tool_start):
                            gt_tool = str(row.get('groundtruth_toolname', '')).strip().lower()
                            # Clean up potential extra spaces
                            if gt_tool in target_tools:
                                active_tools.add(gt_tool)
                                # commercial_toolname carries the specific variant
                                # (e.g. "Large Needle Driver" vs "Mega Needle Driver")
                                # that groundtruth_toolname collapses away - the
                                # organizers confirmed test questions distinguish
                                # these specific names, so keep them alongside the
                                # generic category rather than discarding them.
                                commercial = str(row.get('commercial_toolname', '')).strip().lower()
                                if commercial and commercial != 'unknown instrument':
                                    active_commercial_tools.add(commercial)

            segments.append({
                "case": case_name,
                "video_path": os.path.abspath(video_path),
                "part": part_id,
                "t_start": t_start,
                "t_end": t_end,
                "task": active_task,
                "description": task_desc,
                "tools": list(active_tools),
                "commercial_tools": list(active_commercial_tools)
            })

    return segments

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", default=str(DEFAULT_VIDEO_ROOT))
    parser.add_argument("--labels-root", default=str(DEFAULT_LABEL_ROOT))
    parser.add_argument(
        "--description-labels-root",
        default=str(DEFAULT_DESCRIPTION_ROOT),
        help="Old enriched labels used only for exact matched_description joins.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--window-size", type=float, default=30.0)
    parser.add_argument("--step-size", type=float, default=15.0)
    return parser.parse_args()


def main():
    args = parse_args()
    surgvu24_dir = os.path.abspath(args.video_root)
    labels_dir = os.path.abspath(args.labels_root)
    description_labels_dir = (
        os.path.abspath(args.description_labels_root)
        if args.description_labels_root
        else None
    )
    output_json = os.path.abspath(args.output)

    if not os.path.isdir(surgvu24_dir):
        raise FileNotFoundError(surgvu24_dir)
    if not os.path.isdir(labels_dir):
        raise FileNotFoundError(labels_dir)

    # Get all case directories
    case_dirs = glob.glob(os.path.join(surgvu24_dir, "case_*"))
    case_dirs.sort()
    
    all_segments = []
    print("Processing cases...")
    for case_dir in tqdm(case_dirs):
        case_name = os.path.basename(case_dir)
        segments = process_case(
            case_name,
            surgvu24_dir,
            labels_dir,
            description_labels_dir=description_labels_dir,
            window_size=args.window_size,
            step_size=args.step_size,
        )
        all_segments.extend(segments)
        
    print(f"Total segments indexed: {len(all_segments)}")
    
    # Save to JSON
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_segments, f, indent=4, ensure_ascii=False)
    print(f"Saved segment database to {output_json}")

if __name__ == "__main__":
    main()
