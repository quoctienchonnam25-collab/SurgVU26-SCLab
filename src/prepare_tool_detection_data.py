"""
Builds a YOLO-format tool-detection dataset from cat1_test_set_public's COCO
annotations - the only source of real bounding-box labels in this project (SurgVU's
own 155 training cases only have per-video/part tool presence timing in tools.csv,
no boxes). 7 videos, ~5.2k hand-labeled frames, 14 tool classes.

Split 90/10 by individual frame (stratified-ish via random shuffle within each video)
rather than by whole video: different videos in this set each feature only a handful
of the 14 tool classes (a by-video holdout left 2 classes with literally zero training
examples, and 2 different classes with zero val examples - unusable for evaluating
those classes at all), so a per-frame split is needed for every class to be
represented in both splits despite there being only 7 source videos.
"""
import os
import json
import random
import cv2
from tqdm import tqdm

BASE = r"d:\SCLab\Surgical-Video-Understanding\SurgVU-challenge"
CAT1_DIR = os.path.join(BASE, "cat1_test_set_public")
OUT_DIR = os.path.join(BASE, "tool_detection_data")

VIDEO_IDS = [1, 2, 3, 4, 5, 6, 7]
VAL_FRACTION = 0.1


def main():
    images_dir = {s: os.path.join(OUT_DIR, "images", s) for s in ("train", "val")}
    labels_dir = {s: os.path.join(OUT_DIR, "labels", s) for s in ("train", "val")}
    for d in list(images_dir.values()) + list(labels_dir.values()):
        os.makedirs(d, exist_ok=True)

    cat_names = None
    total_images = 0
    split_counts = {"train": 0, "val": 0}
    random.seed(42)

    for vid in VIDEO_IDS:
        coco_path = os.path.join(CAT1_DIR, f"{vid}_fps1_coco.json")
        video_path = os.path.join(CAT1_DIR, f"{vid}_fps1.mp4")
        with open(coco_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        if cat_names is None:
            categories = sorted(coco["categories"], key=lambda c: c["id"])
            cat_names = [c["name"] for c in categories]

        anns_by_image = {}
        for ann in coco["annotations"]:
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

        cap = cv2.VideoCapture(video_path)
        for img in tqdm(coco["images"], desc=f"video {vid}"):
            split = "val" if random.random() < VAL_FRACTION else "train"
            frame_idx = img["id"]
            W, H = img["width"], img["height"]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue

            stem = f"vid{vid}_{frame_idx:06d}"
            cv2.imwrite(os.path.join(images_dir[split], f"{stem}.jpg"), frame)

            lines = []
            for ann in anns_by_image.get(img["id"], []):
                x, y, w, h = ann["bbox"]
                cx, cy = (x + w / 2) / W, (y + h / 2) / H
                nw, nh = w / W, h / H
                lines.append(f"{ann['category_id']} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            with open(os.path.join(labels_dir[split], f"{stem}.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

            total_images += 1
            split_counts[split] += 1
        cap.release()

    yaml_path = os.path.join(OUT_DIR, "surgvu_tools.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {OUT_DIR}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(cat_names)}\n")
        f.write("names:\n")
        for i, name in enumerate(cat_names):
            f.write(f"  {i}: {name}\n")

    print(f"Wrote {total_images} images ({split_counts}) to {OUT_DIR}")
    print(f"Dataset yaml: {yaml_path}")
    print(f"Classes ({len(cat_names)}): {cat_names}")


if __name__ == "__main__":
    main()
