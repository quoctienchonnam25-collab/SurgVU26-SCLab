# SurgVU 2026 Category 2 — FP16 fix upload bundle (2026-08-06)

Fixes a real graded-run failure from the previous (`release_20260805_nonroot`)
upload: `RuntimeError: No available kernel. Aborting execution.` after 6 minutes.

## Root cause

The submission GPU is a T4 (Turing, compute capability 7.5). The SurgMotion
auxiliary branch's attention (`repo/SurgMotion-main/src/models/utils/modules.py`)
was restricted to `[FLASH_ATTENTION, EFFICIENT_ATTENTION]` (a deliberate fix for
an earlier CUDA-OOM issue - see that file's comments). On Turing:

- Flash attention requires Ampere+ (sm80) and is **never** available, any dtype.
- The container was still running inference in **BF16**, and PyTorch's
  memory-efficient SDPA backend rejects BF16 on pre-Ampere hardware (no native
  BF16 compute unit before Ampere).

Together, the restricted backend list had *nothing* left to dispatch to on that
GPU, so PyTorch raised instead of silently falling back to `math` (which is
exactly what the restriction was written to prevent, for good reason - `math`
is what OOM'd in the first place).

## Fix

1. `src/predict_dual_encoder_vqa.py` / `src/dual_encoder_vqa_prototype.py`:
   inference now runs in **FP16**, not BF16 (training is untouched, still BF16 -
   only the inference path changed). FP16 is fully supported by memory-efficient
   attention on Turing and uses identical VRAM to BF16.
2. `repo/SurgMotion-main/src/models/utils/modules.py`: added a `_sdpa()` helper
   that tries the restricted `[FLASH, EFFICIENT]` list first and only falls back
   to the unrestricted dispatcher (which can include `math`) if that raises -
   so an unanticipated hardware/dtype combination degrades to "possibly slower,
   possibly higher memory" instead of a hard crash.

Both changes are inference-side only; the checkpoint weights
(`checkpoints/dual_encoder_v3_stage_b_all155`) are unchanged from the previous
release - no retraining involved.

## Verified before upload

- Rebuilt via `submission/build_image.sh` (minimal hardlinked build context,
  unchanged from the previous release).
- `docker inspect`: `Config.User=user`, `linux/amd64`.
- Smoke-tested through the **real** Grand Challenge single-case contract
  (`docker run --gpus all --network none -v .../input:/input:ro -v
  .../output:/output ...`, not the local batch-directory convenience path) -
  exit 0, valid `visual-context-response.json`.
- Archive produced by a single, direct `docker save | gzip` - no manual
  repacking (the previous `release_20260805_nonroot` upload failed validation
  with "Could not find manifest.json" because a manual repack step had added a
  `./` prefix to every path in the archive; this bundle was checked for that
  specifically: `manifest.json` present at the bare top-level path, zero
  `./`-prefixed entries).
- `docker load` round-tripped from this exact archive file successfully.

## Upload

Upload `surgvu_dual_encoder_v3_fp16fix_20260806.tar.gz` through Grand Challenge:
`Algorithm` → `Containers` → `Upload a Container`.

Verify before transfer:

```bash
sha256sum -c SHA256SUMS
```

Image ID: `sha256:972eb75e9df9a05f81aa3798ae3923205b029abb1d493018903cd5b0e78ae6d8`
