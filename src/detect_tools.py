"""
Inference helper around the fine-tuned YOLOv5 tool detector (train_tool_detector.py).
Used by predict.py (and prepare_grounding_context.py for training-time consistency) to
ground the VQA prompt with explicit detected-tool evidence, e.g. "Detected tools:
cadiere forceps (0.91), needle driver (0.78)." - meant to reduce the kind of
false-positive tool-presence answers seen on the real public sample (the VQA model
guessing "Yes" from a learned tool-frequency prior instead of what's actually visible
in the clip).

n_frames defaults to 16 (double the VQA model's own 8) because case126's failure
traced to the tool only being visible at the edge of frame for a couple of seconds -
8 uniformly-spaced samples over a 30s clip missed it entirely; this is cheap to widen
since detector inference is a few ms/frame, unlike retraining the VQA model itself.
"""
import numpy as np
import decord
from ultralytics import YOLO

DEFAULT_WEIGHTS = r"d:\SCLab\Surgical-Video-Understanding\SurgVU-challenge\checkpoints\tool_detector\yolov5s_surgvu\weights\best.pt"


def load_detector(weights_path=DEFAULT_WEIGHTS):
    return YOLO(weights_path)


def detect_tools_in_clip(model, video_path, t_start, t_end, n_frames=16, conf_threshold=0.35):
    """Samples a few frames from [t_start, t_end], runs detection on each, and returns
    the tools found anywhere in the clip as a list of (tool_name, max_confidence),
    sorted by confidence descending."""
    vr = decord.VideoReader(video_path)
    fps = vr.get_avg_fps()
    start_frame = int(t_start * fps)
    end_frame = int(t_end * fps)
    frame_indices = np.linspace(start_frame, max(end_frame - 1, start_frame), n_frames, dtype=int)
    frame_indices = np.clip(frame_indices, 0, len(vr) - 1)
    frames = vr.get_batch(frame_indices).asnumpy()

    best_conf = {}
    for frame in frames:
        results = model.predict(frame, conf=conf_threshold, verbose=False)[0]
        for box in results.boxes:
            cls_name = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            if conf > best_conf.get(cls_name, 0.0):
                best_conf[cls_name] = conf

    return sorted(best_conf.items(), key=lambda x: -x[1])


def format_detections_as_context(detections, max_tools=3):
    """Turns [(tool, conf), ...] into a short prompt-prependable context string."""
    if not detections:
        return "Detected tools: none with high confidence."
    top = detections[:max_tools]
    parts = [f"{name} ({conf:.2f})" for name, conf in top]
    return "Detected tools: " + ", ".join(parts) + "."
