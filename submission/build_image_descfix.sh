#!/usr/bin/env bash
set -Eeuo pipefail

# Build from an explicit minimal context, so the repository's legacy root
# .dockerignore cannot silently remove the DualEncoderVQA base weights.
# Isolated variant of build_image.sh for checkpoints/dual_encoder_v3_stage_b_all155_descfix
# (2026-08-13 description-answer-length fix) - does not touch the original
# build_image.sh/Dockerfile/inference_policy.json still used for the currently
# deployed all155 checkpoint.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_TAG="${1:-surgvu-dual-encoder-v3:descfix-latest}"
BUILD_CONTEXT="$(mktemp -d "$PROJECT_ROOT/.dual_encoder_submission_context.XXXXXX")"
cleanup() { rm -rf "$BUILD_CONTEXT"; }
trap cleanup EXIT

mkdir -p \
    "$BUILD_CONTEXT/submission" \
    "$BUILD_CONTEXT/src" \
    "$BUILD_CONTEXT/repo/SurgMotion-main/ckpts" \
    "$BUILD_CONTEXT/checkpoints/base_models" \
    "$BUILD_CONTEXT/checkpoints"

cp -al "$PROJECT_ROOT/submission/Dockerfile.descfix" "$BUILD_CONTEXT/submission/Dockerfile"
cp -al "$PROJECT_ROOT/submission/requirements.txt" "$BUILD_CONTEXT/submission/"
cp -al "$PROJECT_ROOT/submission/inference_policy_descfix.json" "$BUILD_CONTEXT/submission/inference_policy_descfix.json"
cp -al "$PROJECT_ROOT/src/"*.py "$BUILD_CONTEXT/src/"
mkdir -p "$BUILD_CONTEXT/repo/SurgMotion-main"
cp -al "$PROJECT_ROOT/repo/SurgMotion-main/src" "$BUILD_CONTEXT/repo/SurgMotion-main/"
cp -al "$PROJECT_ROOT/repo/SurgMotion-main/ckpts/SurgMotion-vitl.pt" \
    "$BUILD_CONTEXT/repo/SurgMotion-main/ckpts/"
cp -al "$PROJECT_ROOT/checkpoints/base_models/Qwen2.5-VL-3B-Instruct" \
    "$BUILD_CONTEXT/checkpoints/base_models/"
mkdir -p "$BUILD_CONTEXT/checkpoints/dual_encoder_v3_stage_b_all155_descfix"
cp -al "$PROJECT_ROOT/checkpoints/dual_encoder_v3_stage_b_all155_descfix/"{aux_projector.safetensors,dual_encoder_vqa_config.json,surgmotion_delta.safetensors} \
    "$BUILD_CONTEXT/checkpoints/dual_encoder_v3_stage_b_all155_descfix/"
cp -al "$PROJECT_ROOT/checkpoints/dual_encoder_v3_stage_b_all155_descfix/lora_adapter" \
    "$PROJECT_ROOT/checkpoints/dual_encoder_v3_stage_b_all155_descfix/tokenizer" \
    "$BUILD_CONTEXT/checkpoints/dual_encoder_v3_stage_b_all155_descfix/"

docker build -f "$BUILD_CONTEXT/submission/Dockerfile" -t "$IMAGE_TAG" "$BUILD_CONTEXT"
