"""Self-distillation (light) step 1: use the current best fine-tuned model (v5) as
its own teacher/critic. Run it on a stratified sample of the val set, judge each
answer against the ground truth with a type-appropriate heuristic (polarity match
for yes/no questions, key-content-word overlap for open-ended ones), and report
accuracy broken down by question type and by which tool was involved - so we can
see whether errors are random or a systematic bias worth fixing in generate_qa.py.
"""
import json
import re
import random
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from predict import load_model, _generate_answer
from train import SurgicalVQADataset

TYPE_PATTERN = re.compile(r"_t\d+_(.+)_\d+$")
YES_NO_TYPES = {"tool_presence", "tp_bal", "suture_required", "tissue_cutting"}
STOPWORDS = {
    "a", "an", "the", "is", "are", "this", "that", "these", "those", "being",
    "used", "use", "of", "in", "on", "to", "and", "or", "mentioned", "listed",
    "here", "there", "it's", "its", "based", "likely", "appears", "summary",
    "describing", "clip", "step", "part", "surgical", "for", "during",
}


def question_type(item_id):
    m = TYPE_PATTERN.search(item_id)
    return m.group(1) if m else "unknown"


def polarity(s):
    sl = s.strip().lower()
    if sl.startswith("yes"):
        return "yes"
    if sl.startswith("no"):
        return "no"
    return None


def content_words(s):
    return {w for w in re.findall(r"[a-z']+", s.lower()) if w not in STOPWORDS and len(w) > 2}


def judge(qtype, pred, answers):
    if qtype in YES_NO_TYPES:
        gt_pols = [polarity(a) for a in answers]
        gt_pols = [p for p in gt_pols if p]
        if not gt_pols:
            return None
        gt = max(set(gt_pols), key=gt_pols.count)
        pp = polarity(pred)
        if pp is None:
            return False
        return pp == gt
    else:
        canonical = answers[0]
        cw = content_words(canonical)
        if not cw:
            return None
        pred_words = set(re.findall(r"[a-z']+", pred.lower()))
        overlap = cw & pred_words
        return (len(overlap) / len(cw)) >= 0.6


def load_frames(item, max_frames=16):
    ds = SurgicalVQADataset.__new__(SurgicalVQADataset)
    ds.max_frames = max_frames
    frames = ds._load_cached_frames(item.get("frames_dir"))
    if frames is None:
        frames = ds._decode_frames(item["video"], item["t_start"], item["t_end"])
    return frames


def main():
    random.seed(42)
    per_type_cap = int(sys.argv[1]) if len(sys.argv) > 1 else 35

    with open("src/val_vqa.json", encoding="utf-8") as f:
        val = json.load(f)

    buckets = {}
    for item in val:
        t = question_type(item["id"])
        buckets.setdefault(t, []).append(item)

    sample = []
    for t, items in buckets.items():
        random.shuffle(items)
        sample.extend(items[:per_type_cap])
    random.shuffle(sample)
    print(f"Sampled {len(sample)} items across {len(buckets)} types: "
          f"{ {t: min(len(v), per_type_cap) for t, v in buckets.items()} }")

    model, processor = load_model("Qwen/Qwen2-VL-2B-Instruct", "checkpoints/qwen2_vl_2b_lora_v5")

    results = []
    for i, item in enumerate(sample):
        qtype = question_type(item["id"])
        try:
            pil_frames = load_frames(item)
            pred = _generate_answer(model, processor, pil_frames, item["question"])
        except Exception as e:
            print(f"[{i}] ERROR on {item['id']}: {e}")
            continue
        correct = judge(qtype, pred, item["answers"])
        results.append({
            "id": item["id"], "type": qtype, "question": item["question"],
            "pred": pred, "canonical": item["answers"][0], "correct": correct,
        })
        if (i + 1) % 20 == 0:
            print(f"...{i + 1}/{len(sample)} done")

    with open("checkpoints/error_analysis_v5.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    from collections import defaultdict
    stats = defaultdict(lambda: [0, 0, 0])  # correct, wrong, unjudged
    for r in results:
        s = stats[r["type"]]
        if r["correct"] is True:
            s[0] += 1
        elif r["correct"] is False:
            s[1] += 1
        else:
            s[2] += 1

    print("\n=== Accuracy by question type ===")
    for t, (c, w, u) in sorted(stats.items()):
        n = c + w
        acc = c / n if n else float("nan")
        print(f"{t:20s}  acc={acc:.3f}  (correct={c}, wrong={w}, unjudged={u})")

    print("\n=== Sample of WRONG answers (for manual pattern inspection) ===")
    wrong = [r for r in results if r["correct"] is False]
    for r in wrong[:40]:
        print(f"[{r['type']}] Q: {r['question']!r}\n  canonical: {r['canonical']!r}\n  pred:      {r['pred']!r}\n")

    print("Done. Full results saved to checkpoints/error_analysis_v5.json")


if __name__ == "__main__":
    main()
