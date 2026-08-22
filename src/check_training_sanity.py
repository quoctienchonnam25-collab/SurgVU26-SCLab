"""Check that early training loss is finite and trends downward."""

import argparse
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer-state", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--max-step", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    state = json.load(open(args.trainer_state, encoding="utf-8"))
    points = [
        (int(row["step"]), float(row["loss"]))
        for row in state.get("log_history", [])
        if "loss" in row and int(row.get("step", 0)) <= args.max_step
    ]
    finite = bool(points) and all(math.isfinite(loss) for _, loss in points)
    window = max(2, min(3, len(points) // 2))
    first_mean = (
        sum(loss for _, loss in points[:window]) / window if len(points) >= 2 * window else None
    )
    last_mean = (
        sum(loss for _, loss in points[-window:]) / window if len(points) >= 2 * window else None
    )
    decreased = (
        first_mean is not None and last_mean is not None and last_mean < first_mean
    )
    report = {
        "status": "pass" if finite and decreased else "fail",
        "points": points,
        "first_mean": first_mean,
        "last_mean": last_mean,
        "finite": finite,
        "decreased": decreased,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(report_path, "w", encoding="utf-8"), indent=2)
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
