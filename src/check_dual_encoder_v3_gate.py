"""Apply the agreed quality gate before the all-155 submission run."""

import argparse
import json
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--report", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    baseline = load_json(args.baseline)
    candidate = load_json(args.candidate)

    baseline_fpr = baseline["semantic"]["tool_presence"]["false_positive_rate"]
    candidate_fpr = candidate["semantic"]["tool_presence"]["false_positive_rate"]
    baseline_forceps = baseline["semantic"]["forceps_type"]["set_exact_match"]
    candidate_forceps = candidate["semantic"]["forceps_type"]["set_exact_match"]

    checks = {
        "bleu4_improved": candidate["bleu4"] > baseline["bleu4"],
        "bertscore_within_tolerance": (
            candidate["bertscore_f1"] >= baseline["bertscore_f1"] - 0.005
        ),
        "tool_presence_fpr_reduced_by_0_01": (
            baseline_fpr is not None
            and candidate_fpr is not None
            and candidate_fpr <= baseline_fpr - 0.01
        ),
        "forceps_exact_match_not_worse": (
            baseline_forceps is not None
            and candidate_forceps is not None
            and candidate_forceps >= baseline_forceps
        ),
    }
    passed = all(checks.values())
    report = {
        "status": "pass" if passed else "fail",
        "checks": checks,
        "baseline": baseline,
        "candidate": candidate,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
