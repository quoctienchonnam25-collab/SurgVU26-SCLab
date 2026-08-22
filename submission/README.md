# Grand Challenge submission containers

All three final-phase candidate containers (see the top-level
[README](../README.md#final-phase-submission-candidates-2026-08-22)) are built and run
from this directory. Each expects the Grand Challenge input filenames
`endoscopic-robotic-surgery-video.mp4` and `visual-context-question.json`, and creates
exactly one output file, `visual-context-response.json`, whose JSON value is one
non-empty response string. Per-release build/verification details (checksums, docker
inspect output, smoke-test results) are in each `submission/release_*/README_UPLOAD.md`.

| Release | Build script | Dockerfile | Image tag (default) |
|---|---|---|---|
| `release_20260806_thresholdfix` | `build_image.sh` | `Dockerfile` | `surgvu-dual-encoder-v3:latest` |
| `release_20260816_combo` | `build_image_combo.sh` | `Dockerfile.combo` | `surgvu-dual-encoder-v3:combo-latest` |
| `release_20260821_plain` | `build_image_plain.sh` | `Dockerfile.plain` | `surgvu-plain-qwen25vl:latest` |

Each build script is fully isolated (its own Dockerfile, minimal build context, and
`inference_policy*.json` where applicable) - building one release never touches another
release's files.

## Host dry run (thresholdfix / combo checkpoint, no Docker)

```bash
./submission/run_submission.sh INPUT_CASE_DIR OUTPUT_DIR
```

Runs `src/predict_dual_encoder_vqa.py` directly against
`checkpoints/dual_encoder_v3_stage_b_all155` with the `thresholdfix` calibration
thresholds. There is no equivalent dry-run script for `combo` or `plain` yet - build and
`docker run` those, or call `src/predict_dual_encoder_vqa.py` /
`src/predict_plain_qwen25vl.py` directly with the appropriate `--checkpoint`.

## Container build and run

Run from the repository root. Each `build_image*.sh` creates a temporary, minimal build
context containing only the inference code and the required Qwen/SurgMotion/adapter
weights for that specific release, then builds and discards the context.

```bash
# thresholdfix (0.8290 confirmed real prelim score)
./submission/build_image.sh surgvu-dual-encoder-v3:latest

# combo (vision-tower LoRA r=8 + text LoRA r=32)
./submission/build_image_combo.sh surgvu-dual-encoder-v3:combo-latest

# plain (Qwen2.5-VL-3B + LoRA, no SurgMotion)
./submission/build_image_plain.sh surgvu-plain-qwen25vl:latest

mkdir -p "$PWD/OUTPUT_DIR"
chmod o+rwx "$PWD/OUTPUT_DIR"
docker run --rm --gpus all \
  -v "$PWD/INPUT_CASE_DIR:/input:ro" \
  -v "$PWD/OUTPUT_DIR:/output" \
  <IMAGE_TAG>
```

For a local machine where a verified NX desktop process is the only GPU entry and
its name is hidden inside the container, pass its host PID explicitly with
`-e SURGVU_GPU_GUARD_IGNORE_PIDS=<nx-pid>`. Do not use this override for an
unknown GPU process.

The image is offline at runtime (`HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`); no datasets, frame caches, optimizer state or
development checkpoints are included.
