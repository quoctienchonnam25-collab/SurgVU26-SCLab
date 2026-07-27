import os
import glob
import pandas as pd
import json
from tqdm import tqdm
import decord

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

def load_case_labels(case_dir, case_id):
    # Load tasks.csv
    tasks_path = os.path.join(case_dir, "tasks.csv")
    if os.path.exists(tasks_path):
        tasks_df = pd.read_csv(tasks_path)
    else:
        tasks_df = pd.DataFrame()

    # Load tools.csv
    tools_path = os.path.join(case_dir, "tools.csv")
    if os.path.exists(tools_path):
        tools_df = pd.read_csv(tools_path)
    else:
        tools_df = pd.DataFrame()

    return tasks_df, tools_df

def process_case(case_name, surgvu24_dir, labels_dir):
    case_video_dir = os.path.join(surgvu24_dir, case_name)
    case_label_dir = os.path.join(labels_dir, case_name)

    tasks_df, tools_df = load_case_labels(case_label_dir, case_name)
    
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

        # Segment into 30-second windows with 15-second overlap
        window_size = 30.0
        step_size = 15.0
        
        t = 0.0
        while t + window_size <= duration:
            t_start = t
            t_end = t + window_size
            
            # Determine active task
            # Filter tasks matching current part and overlapping with [t_start, t_end]
            active_task = "Other"
            task_desc = "Activity outside of the structured training."
            
            if not tasks_df.empty:
                # Find matching row in tasks_df
                # start_part/stop_part are floats in CSV
                overlapping_tasks = []
                for _, row in tasks_df.iterrows():
                    start_part = int(row.get('start_part', 1))
                    stop_part = int(row.get('stop_part', 1))
                    
                    # If this row belongs to the current part
                    if start_part <= part_id <= stop_part:
                        row_start = float(row.get('start_time', 0.0))
                        row_end = float(row.get('stop_time', 0.0))
                        
                        # Calculate overlap
                        overlap_start = max(t_start, row_start)
                        overlap_end = min(t_end, row_end)
                        if overlap_end > overlap_start:
                            overlap_dur = overlap_end - overlap_start
                            overlapping_tasks.append((overlap_dur, row.get('groundtruth_taskname', 'Other'), row.get('matched_description', '')))
                
                if overlapping_tasks:
                    # Select the task with the maximum overlap duration
                    overlapping_tasks.sort(reverse=True, key=lambda x: x[0])
                    active_task = overlapping_tasks[0][1]
                    task_desc = overlapping_tasks[0][2]

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
                        if tool_end <= tool_start:
                            tool_end = duration

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
            
            t += step_size

    return segments

def main():
    surgvu24_dir = r"d:\SCLab\Surgical-Video-Understanding\SurgVU-challenge\surgvu24"
    labels_dir = r"d:\SCLab\Surgical-Video-Understanding\SurgVU-challenge\SURGVU25_train_labels"
    output_json = r"d:\SCLab\Surgical-Video-Understanding\SurgVU-challenge\src\surgvu_segments.json"

    # Get all case directories
    case_dirs = glob.glob(os.path.join(surgvu24_dir, "case_*"))
    case_dirs.sort()
    
    all_segments = []
    print("Processing cases...")
    for case_dir in tqdm(case_dirs):
        case_name = os.path.basename(case_dir)
        segments = process_case(case_name, surgvu24_dir, labels_dir)
        all_segments.extend(segments)
        
    print(f"Total segments indexed: {len(all_segments)}")
    
    # Save to JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_segments, f, indent=4, ensure_ascii=False)
    print(f"Saved segment database to {output_json}")

if __name__ == "__main__":
    main()
