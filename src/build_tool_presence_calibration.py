"""Build a train-only calibration set for conservative tool-presence answers."""

import argparse
import json
from collections import Counter
from pathlib import Path


def polarity(row):
    return str(row["answers"][0]).strip().lower().startswith("yes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="curated_v3 train JSON only")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--negative-weight", type=float, default=4.0)
    parser.add_argument("--forceps-weight", type=float, default=1.0)
    args = parser.parse_args()
    if args.negative_weight <= 1:
        raise ValueError("--negative-weight must be greater than 1")

    rows = json.loads(Path(args.input).read_text(encoding="utf-8"))
    calibration = []
    for row in rows:
        qtype = row.get("question_type")
        if qtype not in {"tool_presence", "forceps_type"}:
            continue
        item = dict(row)
        original_weight = float(item.get("sample_weight", 1.0))
        if qtype == "tool_presence" and not polarity(item):
            item["sample_weight"] = original_weight * args.negative_weight
            item["calibration_role"] = "tool_presence_negative"
        elif qtype == "tool_presence":
            item["sample_weight"] = original_weight
            item["calibration_role"] = "tool_presence_positive"
        else:
            item["sample_weight"] = original_weight * args.forceps_weight
            item["calibration_role"] = "forceps_anchor"
        calibration.append(item)

    if not calibration:
        raise RuntimeError("No calibration examples were selected")
    roles = Counter(row["calibration_role"] for row in calibration)
    if not roles["tool_presence_negative"] or not roles["tool_presence_positive"]:
        raise RuntimeError("Calibration set must contain both tool-presence polarities")
    if not roles["forceps_anchor"]:
        raise RuntimeError("Calibration set must retain forceps anchors")

    Path(args.output).write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    report = {
        "source": str(Path(args.input).resolve()),
        "output": str(Path(args.output).resolve()),
        "examples": len(calibration),
        "roles": dict(roles),
        "negative_weight": args.negative_weight,
        "forceps_weight": args.forceps_weight,
        "contains_only_train_cases": all(int(row["id"].split("_")[1]) < 140 for row in calibration),
    }
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
