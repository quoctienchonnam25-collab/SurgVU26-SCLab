"""Full BERTScore-F1 evaluation for the plain Qwen2.5-VL-3B + LoRA checkpoint
(no SurgMotion, 5-frame) on a held-out VQA json - same official metric used
throughout this project (see evaluate.py::compute_bertscore_f1).
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm

from answer_postprocess import postprocess_answer
from predict_plain_qwen25vl import load_model, predict_vqa_clip, DEFAULT_BASE_MODEL
from evaluate import compute_bertscore_f1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vqa-json", required=True)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--lora-path", required=True)
    parser.add_argument("--max-frames", type=int, default=5)
    parser.add_argument("--bertscore-model", default="roberta-large")
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def question_type_of(item_id):
    for name in (
        "forceps_type", "suture_required", "procedure_type", "tool_purpose",
        "tissue_cutting", "description", "task_id", "organ",
    ):
        if f"_{name}_" in item_id:
            return name
    if "_tool_presence_" in item_id or "_tp_bal_" in item_id:
        return "tool_presence"
    return "unknown"


def main():
    args = parse_args()
    items = json.load(open(args.vqa_json, encoding="utf-8"))
    if args.limit:
        items = items[: args.limit]

    model, processor = load_model(args.base_model, args.lora_path)

    predictions = []
    for item in tqdm(items, desc="predicting"):
        try:
            pred = predict_vqa_clip(
                model, processor, item["video"], item["question"],
                item["t_start"], item["t_end"], max_frames=args.max_frames,
            )
        except Exception as e:
            print(f"\n[eval] {item.get('id')} failed, using empty prediction: {e}")
            pred = ""
        pred = postprocess_answer(item["question"], pred)
        predictions.append(pred)

    mean_bert, scores = compute_bertscore_f1(items, predictions, model_type=args.bertscore_model)
    print(f"\nOverall BERTScore-F1 ({len(items)} examples): {mean_bert:.4f}")

    by_type = defaultdict(list)
    for item, score in zip(items, scores):
        by_type[question_type_of(item["id"])].append(score)
    print("\nPer question type:")
    for qt in sorted(by_type):
        vals = by_type[qt]
        print(f"  {qt:20s} n={len(vals):5d}  mean={sum(vals)/len(vals):.4f}")

    if args.output:
        out = {
            "bertscore_f1": mean_bert,
            "examples": len(items),
            "per_question_type": {
                qt: {"examples": len(vals), "bertscore_f1": sum(vals) / len(vals)}
                for qt, vals in by_type.items()
            },
            "predictions": predictions,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args.output, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
