"""
Builds a domain-adaptive "Stage 0" QA pretraining corpus from ProstaTDv2/v1.1 (robotic
prostatectomy, already extracted locally) - real per-frame <instrument, action, target>
triplet labels. SurgVU's own tools.csv has no action/target labels at all, so this is
the only source of genuine verb/target supervision available; using it as a *separate*
pretraining stage (not mixed into SurgVU ground truth) avoids passing off an
out-of-domain label as if it were true for SurgVU.

Each labeled image is turned into a "clip" of MAX_FRAMES duplicated copies of the same
frame, saved through the same frame_cache contract extract_frames.py uses, so train.py's
existing SurgicalVQADataset can consume this data unmodified.
"""
import os
import json
import glob
import random
import hashlib
from collections import defaultdict
import cv2
from PIL import Image
from tqdm import tqdm

BASE = r"d:\SCLab\Surgical-Video-Understanding"
PROSTATD_DIRS = [
    os.path.join(BASE, "ProstaTDv2", "ProstaTDv2_cls"),
    os.path.join(BASE, "ProstaTDv2", "ProstaTDv1.1_cls"),
]
CACHE_DIR = os.path.join(BASE, "SurgVU-challenge", "frame_cache_pretrain")
SRC_DIR = os.path.join(BASE, "SurgVU-challenge", "src")
MAX_FRAMES = 8
FRAME_SIZE = 384

INSTRUMENT_PURPOSE = {
    "scissors": "to cut and dissect tissue",
    "forceps": "to grasp and hold tissue or objects",
    "aspirator": "to suck fluid and smoke from the surgical field",
    "needle driver": "to hold and drive needles during suturing",
    "grasper": "to grasp and retract tissue",
    "clip applier": "to apply clips for ligation of vessels or ducts",
}


def load_records():
    """Returns {case_id: [ {image, shapes: [(instrument, action, target), ...]} ]}"""
    by_case = defaultdict(list)
    for root_dir in PROSTATD_DIRS:
        if not os.path.isdir(root_dir):
            continue
        root_tag = os.path.basename(root_dir)
        for case_dir in sorted(os.listdir(root_dir)):
            case_path = os.path.join(root_dir, case_dir)
            if not os.path.isdir(case_path):
                continue
            case_id = f"{root_tag}_{case_dir}"
            for json_path in glob.glob(os.path.join(case_path, "*.json")):
                img_path = os.path.splitext(json_path)[0] + ".jpg"
                if not os.path.exists(img_path):
                    continue
                with open(json_path, "r", encoding="utf-8") as f:
                    d = json.load(f)
                shapes = []
                for s in d.get("shapes", []):
                    label = (s.get("label") or "").strip().lower()
                    if not label:
                        continue
                    attrs = s.get("attributes", {})
                    action = (attrs.get("Action") or "").strip().lower()
                    target = (attrs.get("Target") or "").strip().lower()
                    shapes.append((label, action, target))
                if shapes:
                    by_case[case_id].append({"image": img_path, "shapes": shapes})
    return by_case


def cache_frames(image_path):
    """Duplicates a single labeled frame MAX_FRAMES times into the shared frame-cache
    format extract_frames.py/train.py already use (frame_0.jpg..frame_{n-1}.jpg)."""
    seg_hash = hashlib.md5(image_path.encode("utf-8")).hexdigest()
    seg_dir = os.path.join(CACHE_DIR, seg_hash)
    frame_paths = [os.path.join(seg_dir, f"frame_{i}.jpg") for i in range(MAX_FRAMES)]
    if all(os.path.exists(p) for p in frame_paths):
        return seg_dir

    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, (FRAME_SIZE, FRAME_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)

    os.makedirs(seg_dir, exist_ok=True)
    for p in frame_paths:
        pil_img.save(p, quality=90)
    return seg_dir


def gen_qa_for_record(rec, all_instruments):
    """Mirrors generate_qa.py's style: tool presence, action, target, purpose."""
    shapes = rec["shapes"]
    present = {label for label, _, _ in shapes}
    results = []

    # tool presence (balanced by choosing a present vs absent tool with equal odds)
    if random.random() < 0.5 and present:
        tool = random.choice(list(present))
        is_present = True
    else:
        absent = [t for t in all_instruments if t not in present]
        tool = random.choice(absent) if absent else random.choice(list(all_instruments))
        is_present = tool in present
    q = random.choice([
        f"Is a {tool} being used here?",
        f"Is there a {tool} present in this image?",
        f"Was a {tool} used during this step?",
    ])
    answers = (["Yes", f"Yes, a {tool} is being used.", f"Yes, a {tool} is present."]
               if is_present else
               ["No", f"No, a {tool} was not used.", f"No, there's no {tool}."])
    results.append({"question": q, "answers": answers, "tool": tool})

    # action + target, from a random annotated shape with non-null action/target
    informative = [s for s in shapes if s[1] and s[1] != "null" and s[2] and s[2] != "null"]
    if informative:
        instrument, action, target = random.choice(informative)
        q = random.choice([
            "What action is being performed?",
            "What is the surgeon doing in this step?",
        ])
        answers = [
            action.capitalize(),
            f"The surgeon is performing {action}.",
            f"{action.capitalize()} is being performed.",
        ]
        results.append({"question": q, "answers": answers})

        q = random.choice([
            "What anatomical structure is being operated on?",
            "What is the target of this action?",
        ])
        answers = [
            target.capitalize(),
            f"The target is the {target}.",
            f"The {target} is being operated on.",
        ]
        results.append({"question": q, "answers": answers})

        purpose = INSTRUMENT_PURPOSE.get(instrument, "to assist during the surgical procedure")
        q = f"What is the purpose of using {instrument} in this procedure?"
        answers = [
            f"{purpose[0].upper()}{purpose[1:]}.",
            f"The {instrument} is used {purpose}.",
        ]
        results.append({"question": q, "answers": answers})

    return results


def main():
    by_case = load_records()
    cases = sorted(by_case.keys())
    random.seed(42)
    random.shuffle(cases)
    n_val = max(1, int(len(cases) * 0.1))
    val_cases = set(cases[:n_val])
    train_cases = [c for c in cases if c not in val_cases]

    all_instruments = sorted({label for recs in by_case.values() for r in recs for label, _, _ in r["shapes"]})
    print(f"Cases: {len(cases)} (train={len(train_cases)}, val={len(val_cases)})")
    print(f"Instruments: {all_instruments}")

    def build(cases_list, max_per_case=200):
        vqa = []
        for case in tqdm(cases_list, desc="Building pretrain QA"):
            recs = by_case[case]
            sampled = random.sample(recs, min(max_per_case, len(recs)))
            for rec in sampled:
                frames_dir = cache_frames(rec["image"])
                if frames_dir is None:
                    continue
                for i, qa in enumerate(gen_qa_for_record(rec, all_instruments)):
                    entry = {
                        "id": f"{case}_{os.path.basename(rec['image'])}_{i}",
                        "frames_dir": frames_dir,
                        "question": qa["question"],
                        "answers": qa["answers"],
                    }
                    if "tool" in qa:
                        entry["tool"] = qa["tool"]
                    vqa.append(entry)
        return vqa

    train_vqa = build(train_cases)
    val_vqa = build(val_cases)
    print(f"Generated {len(train_vqa)} pretrain-train QA, {len(val_vqa)} pretrain-val QA")

    train_path = os.path.join(SRC_DIR, "pretrain_train_vqa.json")
    val_path = os.path.join(SRC_DIR, "pretrain_val_vqa.json")
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_vqa, f, indent=2, ensure_ascii=False)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_vqa, f, indent=2, ensure_ascii=False)
    print(f"Saved to {train_path} and {val_path}")


if __name__ == "__main__":
    main()
