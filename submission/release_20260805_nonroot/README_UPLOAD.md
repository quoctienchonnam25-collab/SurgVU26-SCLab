# SurgVU 2026 Category 2 — non-root upload bundle

Upload `surgvu_dual_encoder_v3_nonroot_20260805.tar.gz` through Grand Challenge:
`Algorithm` → `Containers` → `Upload a Container`.

This supersedes the earlier root-container archive. The loaded image is
Linux/amd64 and has `Config.User="user"`, satisfying the non-root container
validation. It was load-tested from this archive before compression; its Docker
image ID is `sha256:59865095b50df66ec3eb055640b5ddd5415f4381e82206d4f346e678601540ee`.

All Qwen2.5-VL-3B, SurgMotion and all-155 checkpoint weights are inside the
container. No separate model tarball is required.

Verify the upload archive before transfer:

```bash
sha256sum -c SHA256SUMS
```

`submission_manifest_v3_nonroot_final.json` records policy and smoke-test
provenance. `CHECKPOINT_SHA256SUMS` records checkpoint component hashes in the
