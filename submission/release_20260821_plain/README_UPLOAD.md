# SurgVU 2026 Category 2 — plain Qwen2.5-VL-3B + LoRA bundle (2026-08-21)

A deliberate architecture reset: plain `Qwen2.5-VL-3B-Instruct` + a standard
LoRA fine-tune (r=16, text-side only), **5 frames per clip, no SurgMotion
auxiliary branch, no dual-encoder complexity, no vision-tower adaptation**.
Trained on the ORIGINAL (long-form) `description` answers - the data variant
used by `release_20260806_thresholdfix` (confirmed real prelim score 0.8290),
**not** the short-answer variant used by `descfix`/`combo` (both regressed to
0.7653 on real prelim).

## Why: three converging signals against architecture complexity

1. **Historical precedent already in this codebase** (`dual_encoder_vqa_prototype.py`'s
   own module docstring): `SurgMotionVQA` (full deletion of Qwen's native
   vision tower, replaced by a SurgMotion-only encoder) won on this
   project's own held-out split but *lost* on the real Grand Challenge
   hidden test set (0.7643 vs 0.7693 for a plain Qwen2-VL-2B LoRA
   fine-tune - "v7"). That finding is why `DualEncoderVQA` was designed to
   keep Qwen's vision tower frozen in the first place.
2. **This project's own 2026-08-16/17 real-world result**: `combo`
   (vision-tower LoRA r=8 + text LoRA r=32) and `descfix` (same
   architecture as the confirmed-good `thresholdfix`, data-only change)
   scored an *identical* 0.7653 on real prelim - proving the architecture
   change added zero measurable value on the real test set, and something
   about the short-description-answer data change was the actual (or at
   least a much larger) regression driver.
3. **External evidence**: the #1 leaderboard entry as of 2026-08-20 scores
   0.8523 using "QwenVL LoRA, 5-frame" - i.e. almost exactly this shape.

Given `check_dual_encoder_v3_gate.py`-style local dev-scale holdout metrics
have now been shown twice this project to NOT predict real leaderboard
direction, and the SurgVU Org Team has since stated on the official forum
(2026-08-20) that prelim rankings themselves are "likely not indicative" of
real performance on such a small test set, this release was **not**
tuned or gated against any local metric - it is built directly on the
strongest available priors (historical + external) rather than a
local-metric-optimized candidate. A full BERTScore-F1 evaluation was run
after packaging anyway (see below) as a coherence check, not as the basis
for the decision.

## Critical fix applied post-training (2026-08-21)

The first full BERTScore-F1 evaluation run crashed mid-way (293/1314
examples in) with a CUDA device-side assert: `probability tensor contains
either inf, nan or element < 0`. Root cause: the base model's own
`generation_config.json` ships `do_sample=true, temperature=0.000001` as a
"greedy-like" trick, but dividing logits by a near-zero temperature
overflows to `inf` under FP16 (this container's inference dtype, required
for the submission T4 GPU), which then produces `nan` in the softmax used
for sampling. Fixed in `src/predict_plain_qwen25vl.py::_generate_answer()`
by passing `do_sample=False` explicitly (true greedy decoding, bypasses
the temperature/softmax path entirely). **This bug would have risked
crashing the real Grand Challenge submission mid-run** - the container
below was rebuilt from scratch with the fix and is the first version safe
to upload; an earlier build (image id `136d56e1d0dc...`) predates the fix
and must not be used.

## Real local evaluation: BERTScore-F1 = 0.9562 (1314 examples, true holdout case147-154)

Run post-fix via `src/evaluate_plain_qwen25vl.py` against
`runs/dataset_audit/backup_pre_description_fix_20260812/test_vqa_surgmotion_curated_v3.json`.
Per-category breakdown:

| category | BERTScore-F1 |
|---|---|
| suture_required | 0.9998 |
| tool_presence | 0.9881 |
| tool_purpose | 0.9869 |
| procedure_type | 0.9893 |
| tissue_cutting | 0.9854 |
| forceps_type | 0.9299 |
| task_id | 0.7163 |
| organ | 0.6468 |
| description | 0.4611 |

As always, treat this as a coherence signal, not a leaderboard predictor -
this project has twice observed local BERTScore gains fail to translate to
real prelim results (see `surgvu_combo_prelim_regression`). `description`
and `organ` are the weakest categories, consistent with prior findings
that BERTScore is most informative on longer free-text answers.

## What was verified before packaging

- **Local sanity check** (not a leaderboard proxy - just "is this
  coherent"): 15 random examples from the true holdout (case147-154,
  `runs/dataset_audit/backup_pre_description_fix_20260812/test_vqa_surgmotion_curated_v3.json`)
  spot-checked via direct generation. 12/15 exact string matches to the
  reference answer; all 3 misses were `tool_presence` Yes/No calls (the
  hardest category throughout this project), never off-topic or
  malformed output.
- Training: 1 epoch, all 155 cases (25,064 examples), `train_loss` 1.74 ->
  0.18 over 6266 steps (~4h on a single 3090), healthy convergence
  throughout, no instability.
- Full BERTScore-F1 evaluation, 1314 true-holdout examples: 0.9562 overall
  (see table above) - post do_sample fix.
- Built via `submission/build_image_plain.sh
  surgvu-plain-qwen25vl:20260821` - fully isolated from every prior
  release's build script/Dockerfile/checkpoint. Rebuilt 2026-08-21 to
  incorporate the do_sample fix above.
- `docker inspect`: `Config.User=user`, `linux/amd64`,
  `sha256:46c54bd8899df6d2fd584ab215a505fd8dbde3846ec9813691fcc5759f0b1541`.
- Smoke-tested through the real Grand Challenge single-case contract
  (`docker run --gpus all --network none ...`,
  `-e SURGVU_GPU_GUARD_IGNORE_PIDS=6529` for this host's NX session) -
  exit 0, valid `visual-context-response.json`.
- `docker save | gzip` archive, `docker load` round-tripped to the
  identical image ID (`46c54bd8899d...`).

## Known limitations / what was deliberately NOT done

- No Yes/No threshold calibration (unlike every dual-encoder release) -
  this architecture's raw `generate()` output is used as-is, matching the
  "plain LoRA" simplicity this release is testing.
- Not tested on prelim before packaging - prelim budget is understood to be
  effectively exhausted (self-policed 10-try limit, not platform-enforced,
  see project memory `surgvu_final_phase_deadlines`) and, per the Org
  Team's own 2026-08-20 statement, prelim rank isn't a reliable quality
  signal even when available.

## Upload

`surgvu_plain_qwen25vl_20260821.tar.gz` through Grand Challenge:
`Algorithm` -> `Containers` -> `Upload a Container`.

Verify before transfer:

```bash
sha256sum -c SHA256SUMS
```

**Recommended final-phase strategy** (3 submissions total, best-of-3
scored, per the Org Team's 2026-08-20 forum confirmation): submit
`release_20260806_thresholdfix` (confirmed real 0.8290) as an insurance
floor, and this release as an informed bet - a failed bet costs nothing
beyond the slot itself since only the best score counts.
