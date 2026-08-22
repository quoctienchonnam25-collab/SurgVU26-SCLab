#!/usr/bin/env bash
set -Eeuo pipefail

# Build from an explicit minimal context, so the repository's legacy root
# .dockerignore cannot silently remove the base weights. Isolated build for
# the plain Qwen2.5-VL-3B + LoRA candidate (no SurgMotion, no dual-encoder) -
# see src/predict_plain_qwen25vl.py's module docstring. Does not touch any
# previous release's build_image.sh/Dockerfile/inference_policy.json.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_TAG="${1:-surgvu-plain-qwen25vl:latest}"
BUILD_CONTEXT="$(mktemp -d "$PROJECT_ROOT/.plain_qwen25vl_submission_context.XXXXXX")"
cleanup() { rm -rf "$BUILD_CONTEXT"; }
trap cleanup EXIT

mkdir -p \
    "$BUILD_CONTEXT/submission" \
    "$BUILD_CONTEXT/src" \
    "$BUILD_CONTEXT/checkpoints/base_models" \
    "$BUILD_CONTEXT/checkpoints/plain_qwen25vl_lora_all155"

cp -al "$PROJECT_ROOT/submission/Dockerfile.plain" "$BUILD_CONTEXT/submission/Dockerfile"
cp -al "$PROJECT_ROOT/submission/requirements.txt" "$BUILD_CONTEXT/submission/"
cp -al "$PROJECT_ROOT/src/predict_plain_qwen25vl.py" "$BUILD_CONTEXT/src/"
cp -al "$PROJECT_ROOT/checkpoints/base_models/Qwen2.5-VL-3B-Instruct" \
    "$BUILD_CONTEXT/checkpoints/base_models/"

# Only the top-level adapter/tokenizer files - NOT the checkpoint-NNNN/ or
# runs/ subdirectories (Trainer optimizer state, TensorBoard logs - not
# needed for inference and would bloat the image).
for f in "$PROJECT_ROOT/checkpoints/plain_qwen25vl_lora_all155/"*; do
    base="$(basename "$f")"
    if [[ -f "$f" ]]; then
        cp -al "$f" "$BUILD_CONTEXT/checkpoints/plain_qwen25vl_lora_all155/$base"
    fi
done

docker build -f "$BUILD_CONTEXT/submission/Dockerfile" -t "$IMAGE_TAG" "$BUILD_CONTEXT"
