# DualEncoderVQA v3 submission

This package runs `dual_encoder_v3_stage_b_all155` with the calibrated
tool-presence threshold in `inference_policy.json`. It expects the Grand
Challenge input filenames `endoscopic-robotic-surgery-video.mp4` and
`visual-context-question.json`, and creates exactly one output file:
`visual-context-response.json`, whose JSON value is one non-empty response string.

## Host dry run

```bash
./submission/run_submission.sh INPUT_CASE_DIR OUTPUT_DIR
```

## Container build and run

Run from the repository root. `build_image.sh` creates a temporary, minimal build context containing only inference code and the required Qwen, SurgMotion and adapter weights.

```bash
./submission/build_image.sh surgvu-dual-encoder-v3:latest
docker run --rm --gpus all \
  -v "$PWD/INPUT_CASE_DIR:/input:ro" \
  -v "$PWD/OUTPUT_DIR:/output" \
  surgvu-dual-encoder-v3:latest
```

For a local machine where a verified NX desktop process is the only GPU entry and
its name is hidden inside the container, pass its host PID explicitly with
`-e SURGVU_GPU_GUARD_IGNORE_PIDS=<nx-pid>`. Do not use this override for an
unknown GPU process.

The image is offline at runtime (`HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`); no datasets, frame caches, optimizer state or
development checkpoints are included.
