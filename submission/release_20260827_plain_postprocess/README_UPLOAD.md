# SurgVU 2026 Category 2 — plain Qwen2.5-VL + answer post-processing (2026-08-27)

Same architecture/checkpoint as `release_20260821_plain` (plain
`Qwen2.5-VL-3B-Instruct` + standard text-LoRA r=16, 5 frames, no SurgMotion
branch, trained on the original long-form `description` data - see that
release's README for the full rationale: an architecture reset motivated by
this project's own real prelim regression plus external leaderboard-leader
precedent). That release was built+verified but never tested on real
infrastructure (prelim budget was exhausted before it existed). This release
adds the same `src/answer_postprocess.py` layer used in
`release_20260827_thresholdfix_postprocess` - see that README for the full
design rationale, including the empirical pitfall (collapsing a
recognized-but-wrong answer to bare form destroyed BERTScore partial credit,
0.613→0.171 on the one affected true-holdout row) that shaped the
fallback-only design actually shipped here.

## What changed relative to release_20260821_plain

- `src/predict_plain_qwen25vl.py`: both `run_grand_challenge_case()` (the
  real submission path) and `run_batch_directory()` (local dev convenience)
  now call `postprocess_answer(question, answer)` after generation.
- `src/evaluate_plain_qwen25vl.py`: same call added, so any future local
  holdout comparison reflects what actually ships.
- `submission/Dockerfile.plain` / `submission/build_image_plain.sh`: now
  also copy `src/answer_postprocess.py` and `src/generate_qa.py` (the
  post-processing layer's only dependency) into the image - the prior
  Dockerfile copied only `predict_plain_qwen25vl.py` by name, not a `src/*`
  glob, so this had to be added explicitly.
- No changes to training data, LoRA weights, or the base checkpoint.

## Verified before packaging

- Rebuilt via `submission/build_image_plain.sh
  surgvu-plain-qwen25vl:20260827-postprocess`.
- `docker inspect`: `Config.User=user`, `linux/amd64`,
  `sha256:f4d852556c9c59cea746d3790272c050d096e7f874479ea6f3cbf347a573e79d`.
- Smoke-tested through the real Grand Challenge single-case contract
  (`docker run --gpus all --network none ...`, `-e
  SURGVU_GPU_GUARD_IGNORE_PIDS=6529`) - exit 0, valid
  `visual-context-response.json` for case127's organ question (raw model
  output already matched the post-processing fallback string, so no crash
  and no unexpected transformation - the model itself missed this specific
  organ, independent of this change).
- `docker save | gzip` archive, `docker load` round-tripped to the identical
  image ID.

## Known limitations (inherited from release_20260821_plain, unchanged here)

- No Yes/No threshold calibration - this architecture's raw `generate()`
  output for the 3 binary categories is used as-is.
- Still not tested on real prelim/final infrastructure before this build -
  `release_20260806_thresholdfix` is the only release in this project with a
  confirmed real leaderboard score (0.8290 prelim, 0.5526 final).
- Same post-processing limitations as
  `release_20260827_thresholdfix_postprocess`: does not touch `description`
  or `task_id`, and cannot fix content errors, only phrasing drift.

## Upload

`surgvu_plain_qwen25vl_postprocess_20260827.tar.gz` through Grand Challenge:
`Algorithm` → `Containers` → `Upload a Container`.

Verify before transfer:

```bash
sha256sum -c SHA256SUMS
```

**Recommended final-phase strategy**: with 2 of 3 final slots remaining
after `release_20260806_thresholdfix` (0.5526, confirmed), this release
combines both open hypotheses (architecture reset + post-processing) into
one bet for one slot. If slot budget allows testing only one variable at a
time, prefer `release_20260827_thresholdfix_postprocess` first (isolates
the post-processing layer's own effect against the known checkpoint), and
use this release for the remaining slot regardless of that result, since it
is the only way to test the architecture-reset hypothesis on real
infrastructure at all before the final phase closes (2026-09-06).
