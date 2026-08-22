"""Score the submitted v5 checkpoint (Qwen2-VL-2B) on the same curated true-holdout
set (case140-154) that runs/surgmotion_vqa/audit_dev_holdout scored quality_v1 and
stage_b_dev on, so all three models can be compared on identical questions for the
first time - see runs/surgmotion_vqa/audit_dev_holdout/REPORT.md for why no such
comparison existed before this script.

Reuses predict.py's load_model/_generate_answer (v5's own inference path) and
train.py's SurgicalVQADataset._load_cached_frames/_decode_frames (the only code that
correctly respects a VQA item's t_start/t_end window - predict.py's own predict_vqa
samples across the whole video file instead, which is wrong for this windowed
dataset), exactly as src/error_analysis.py already does on a 350-item sample of the
*other* val file. This script runs the full 2254-item curated holdout instead.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import compute_bertscore_f1, evaluate_predictions
from predict import _generate_answer, load_model
from train import SurgicalVQADataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VQA_JSON = PROJECT_ROOT / "src" / "val_vqa_surgmotion_curated_v2.json"
DEFAULT_BERTSCORE_MODEL = PROJECT_ROOT / "checkpoints" / "base_models" / "roberta-large"


def assert_gpu_idle():
    """Same guard predict_surgmotion_vqa.py/train_surgmotion_vqa.py enforce - never
    run inference against a GPU another job on this shared dev machine is using."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script")
    rows = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    conflicts = [row for row in rows if row.strip() and "nxnode" not in row]
    if conflicts:
        raise RuntimeError("GPU is busy: " + "; ".join(conflicts))


def load_frames(item, max_frames=16, frame_size=384):
    ds = SurgicalVQADataset.__new__(SurgicalVQADataset)
    ds.max_frames = max_frames
    ds.frame_size = frame_size
    frames = ds._load_cached_frames(item.get("frames_dir"))
    if frames is None:
        frames = ds._decode_frames(item["video"], item["t_start"], item["t_end"])
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vqa-json", default=str(DEFAULT_VQA_JSON))
    parser.add_argument("--checkpoint", default="checkpoints/qwen2_vl_2b_lora_v5")
    parser.add_argument("--base-model", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument(
        "--output-dir",
        default="runs/surgmotion_vqa/audit_dev_holdout/v5_on_curated_holdout",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--bertscore-model", default=str(DEFAULT_BERTSCORE_MODEL))
    args = parser.parse_args()

    assert_gpu_idle()

    items = json.load(open(args.vqa_json, encoding="utf-8"))
    if args.limit is not None:
        items = items[: args.limit]

    model, processor = load_model(args.base_model, args.checkpoint)

    rows = []
    for index, item in enumerate(items):
        pil_frames = load_frames(item)
        prediction = _generate_answer(model, processor, pil_frames, item["question"])
        row = dict(item)
        row["prediction"] = prediction
        rows.append(row)
        print(f"[{index + 1}/{len(items)}] {item['id']} -> {prediction}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(output_dir / "predictions.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    json.dump(
        [row["prediction"] for row in rows],
        open(output_dir / "predictions_list.json", "w", encoding="utf-8"),
        indent=2,
        ensure_ascii=False,
    )

    predictions = [row["prediction"] for row in rows]
    bertscore, per_example_bert = compute_bertscore_f1(rows, predictions, model_type=args.bertscore_model)
    bleu, per_example_bleu = evaluate_predictions(rows, predictions)
    for row, bert, bleu_row in zip(rows, per_example_bert, per_example_bleu):
        row["bertscore_f1"] = bert
        row["bleu4"] = bleu_row
    metrics = {"bertscore_f1": bertscore, "bleu4": float(bleu), "examples": len(rows)}
    json.dump(metrics, open(output_dir / "metrics.json", "w", encoding="utf-8"), indent=2)
    json.dump(rows, open(output_dir / "predictions.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
