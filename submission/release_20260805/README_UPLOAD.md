# SurgVU 2026 Category 2 upload bundle

Upload `surgvu_dual_encoder_v3_freeze_20260805.tar.gz` through Grand Challenge:
`Algorithm` → `Containers` → `Upload a Container`.

The image is Linux/amd64 and is frozen as
`surgvu-dual-encoder-v3:freeze-20260805`. It includes Qwen2.5-VL-3B,
SurgMotion, and the all-155 checkpoint; no separate model tarball is needed.

Verify the upload archive before transfer:

```bash
sha256sum -c SHA256SUMS
```

`submission_manifest_v3_final.json` records the image digest, policy and
public-container smoke test. `CHECKPOINT_SHA256SUMS` records the component
