#!/usr/bin/env bash
set -Eeuo pipefail

# Host-side launcher. The Docker image uses the same command with /input,/output.
if [[ $# -ne 2 ]]; then
    echo "Usage: $0 INPUT_CASE_DIR OUTPUT_DIR" >&2
    exit 64
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/repo/surgmotion_env/bin/python}"
exec "$PYTHON_BIN" -u "$PROJECT_ROOT/src/predict_dual_encoder_vqa.py" \
    --checkpoint "$PROJECT_ROOT/checkpoints/dual_encoder_v3_stage_b_all155" \
    --grand-challenge \
    --input-dir "$1" \
    --output-dir "$2" \
    --tool-presence-threshold 0.2 \
    --tissue-cutting-threshold 1.4 \
    --suture-required-threshold 0.3
