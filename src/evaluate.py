import json
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import argparse

# As of 2026-07-23 the official metric switched from BLEU-4 to BERTScore-F1
# (roberta-large, rescaled with baseline) - see docs/Evaluation Criteria...
# -new-metrics.pdf. compute_bleu_for_pair/evaluate_predictions are kept only
# for historical comparison against runs scored before the change.
_bertscorer = None


def get_bertscorer(model_type="roberta-large"):
    """Cached singleton: BERTScorer loads a full roberta-large model, so reuse it
    across repeated calls (e.g. train.py's periodic eval callback) instead of
    reloading every time."""
    global _bertscorer
    if _bertscorer is None:
        from bert_score import BERTScorer
        _bertscorer = BERTScorer(lang="en", model_type=model_type, rescale_with_baseline=True)
    return _bertscorer


def compute_bertscore_f1(vqa_items, predictions, model_type="roberta-large"):
    """Official SurgVU26 Category 2 metric: for each question, BERTScore-F1 between
    the predicted answer and each of the 5 ground-truth answers, taking the max F1 -
    then the mean of that max over all questions. bert_score's scorer already takes
    the max F1 when given multiple references per candidate, so this is one batched
    call rather than 5 separate ones per item."""
    if not vqa_items:
        return 0.0, []
    scorer = get_bertscorer(model_type)
    references = [item["answers"] for item in vqa_items]
    _, _, F1 = scorer.score(predictions, references)
    scores = F1.tolist()
    return float(np.mean(scores)), scores


def compute_bleu_for_pair(references, candidate):
    """
    references: list of lists of tokens (e.g., [['no'], ['no', 'forceps', 'are', 'not', 'mentioned']])
    candidate: list of tokens (e.g., ['no'])
    """
    smoothing_function = SmoothingFunction().method1
    weights = (0.25, 0.25, 0.25, 0.25)

    scores = []
    for ref in references:
        # Compute BLEU-4 between candidate and this single reference
        score = sentence_bleu([ref], candidate, weights=weights, smoothing_function=smoothing_function)
        scores.append(score)

    return max(scores) if scores else 0.0

def evaluate_predictions(vqa_items, predictions):
    """
    Legacy BLEU-4 metric (pre-2026-07-23). Kept for historical comparison; use
    compute_bertscore_f1 for the current official score.
    vqa_items: list of dictionaries, each containing 'answers' (list of strings)
    predictions: list of strings (predicted answers)
    """
    bleu_scores = []
    for item, pred in zip(vqa_items, predictions):
        references = [ans.split() for ans in item['answers']]
        candidate = pred.split()
        score = compute_bleu_for_pair(references, candidate)
        bleu_scores.append(score)

    mean_bleu = np.mean(bleu_scores)
    return mean_bleu, bleu_scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_json", type=str, default=r"d:\SCLab\Surgical-Video-Understanding\SurgVU-challenge\src\val_vqa.json")
    parser.add_argument("--preds_json", type=str, default=None, help="Path to predictions JSON (list of strings)")
    args = parser.parse_args()
    
    with open(args.val_json, 'r', encoding='utf-8') as f:
        val_items = json.load(f)
        
    if args.preds_json is not None:
        with open(args.preds_json, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
    else:
        # Baseline check: Use the first answer as a dummy prediction to test the code
        print("No predictions file provided. Running self-evaluation baseline (predicting first reference answer)...")
        predictions = [item['answers'][0] for item in val_items]
        
    mean_bleu, _ = evaluate_predictions(val_items, predictions)
    mean_bert, _ = compute_bertscore_f1(val_items, predictions)
    print(f"Evaluated {len(val_items)} items.")
    print(f"Mean Max BERTScore-F1 (official metric): {mean_bert:.6f}")
    print(f"Mean Max BLEU-4 (legacy metric, for reference only): {mean_bleu:.6f}")

if __name__ == "__main__":
    main()
