# Grand Challenge submission containers

Three containers were submitted to the SurgVU 2026 Category 2 final phase (3
submissions/team, best score counts). All three are built and run from this
directory, from the same `DualEncoderVQA` checkpoint and calibrated Yes/No
thresholds (`dual_encoder_v3_stage_b_all155`) via the same `build_image.sh` /
`Dockerfile` - they differ only in answer-formatting/routing and container-robustness
code, which evolved commit-by-commit across the three slots. Each expects the Grand
Challenge input filenames `endoscopic-robotic-surgery-video.mp4` and
`visual-context-question.json`, and creates exactly one output file,
`visual-context-response.json`, whose JSON value is one non-empty response string.

| Slot | Release | Final score | Status |
|---|---|---|---|
| 1 | [`release_20260806_thresholdfix`](release_20260806_thresholdfix/README_UPLOAD.md) | 0.5526 | Real-leaderboard-confirmed baseline |
| 2 | [`release_20260831_thresholdfix_qfix`](release_20260831_thresholdfix_qfix/README_UPLOAD.md) | **0.5972** | **Best - the team's official score. Use this one.** |
| 3 | [`release_20260902_final`](release_20260902_final/README_UPLOAD.md) | 0.5820 | Robustness bundle, regressed vs. slot 2 - see its README for the negative-result writeup |

Per-release build/verification details (checksums, `docker inspect` output,
smoke-test results) are in each release's own `README_UPLOAD.md`.

## Quick start - build the best model (slot 2, 0.5972)

`checkpoints/` and `repo/SurgMotion-main/ckpts/` are gitignored (large binaries), so
a fresh clone needs the weights fetched first:

```bash
# 1. Download the LoRA adapter, projector, and SurgMotion delta (~185 MB)
huggingface-cli download Partrick86/surgvu26-sclab-slot2-checkpoint \
  surgvu26_sclab_slot2_dual_encoder_checkpoint.tar.gz --local-dir /tmp

# 2. Verify against the published checksum
echo "19dc1ac00eb3f553118494dd282f57e686c8370008eb683a99d6dadf52d421b8  /tmp/surgvu26_sclab_slot2_dual_encoder_checkpoint.tar.gz" | sha256sum -c -

# 3. Extract into the checkpoint directory build_image.sh expects
mkdir -p checkpoints/dual_encoder_v3_stage_b_all155
tar -xzf /tmp/surgvu26_sclab_slot2_dual_encoder_checkpoint.tar.gz \
  -C checkpoints/dual_encoder_v3_stage_b_all155
```

The checkpoint's Hugging Face repo: <https://huggingface.co/Partrick86/surgvu26-sclab-slot2-checkpoint>.

The two base models this checkpoint's LoRA/delta sit on top of are downloaded from
their own public sources instead of being re-hosted here (see
[Acknowledgements](../README.md#acknowledgements) in the top-level README):
`Qwen/Qwen2.5-VL-3B-Instruct` (Hugging Face) and `SurgMotion-vitl.pt`
(`CAIR-HKISI/SurgMotion`).

```bash
# Run from the repository root, once checkpoints/ and repo/SurgMotion-main/ckpts/ are populated.
./submission/build_image.sh surgvu-dual-encoder-v3:latest

mkdir -p "$PWD/OUTPUT_DIR"
chmod o+rwx "$PWD/OUTPUT_DIR"
docker run --rm --gpus all \
  -v "$PWD/INPUT_CASE_DIR:/input:ro" \
  -v "$PWD/OUTPUT_DIR:/output" \
  surgvu-dual-encoder-v3:latest
```

For a local machine where a verified NX desktop process is the only GPU entry and
its name is hidden inside the container, pass its host PID explicitly with
`-e SURGVU_GPU_GUARD_IGNORE_PIDS=<nx-pid>`. Do not use this override for an
unknown GPU process.

The image is offline at runtime (`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`);
no datasets, frame caches, optimizer state or development checkpoints are included.

## Host dry run (no Docker)

```bash
./submission/run_submission.sh INPUT_CASE_DIR OUTPUT_DIR
```

Runs `src/predict_dual_encoder_vqa.py` directly against
`checkpoints/dual_encoder_v3_stage_b_all155` with the `thresholdfix` calibration
thresholds - the same model and thresholds as slots 1 and 2, without slot 3's
container-robustness/routing changes.

## Reproducing slots 1 or 3 instead of slot 2

All three slots build from this same `build_image.sh` / `Dockerfile`; only the
inference code differs between them, evolving commit-by-commit. Each release's
`README_UPLOAD.md` documents exactly what changed relative to the previous slot -
check out the corresponding point in git history before running `build_image.sh` if
you need to reproduce a lower-scoring slot's exact behavior instead of the
recommended slot 2.
