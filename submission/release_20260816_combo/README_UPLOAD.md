# SurgVU 2026 Category 2 — architecture combo bundle (2026-08-16)

Builds on `release_20260813_descfix` (same data: short-style `description`
answers, same FP16/T4 fix, same threshold-calibration mechanism) with an
**architecture change**: `checkpoints/dual_encoder_v3_stage_b_all155` was
retrained with two changes to `DualEncoderVQA` stacked together —
Qwen2.5-VL's own vision tower is no longer fully frozen (it now carries a
low-rank LoRA adapter, `vision_lora_r=8`), and the text-side LoRA rank was
raised `16→32`.

## Why: 4 isolated experiments, then a combination

An earlier investigation found that restoring Qwen's own frozen vision
tower (vs. `SurgMotionVQA`'s full deletion of it) was worth +0.0245
BERTScore-F1 in a matched comparison — the single biggest lever found in
this project, bigger than any data-curation refinement tried. This
motivated four follow-up dev-scale experiments (`case000-139` train,
gated against `dual_encoder_v3_stage_b_dev_descfix` on `case140-154`
test, N=1314, via `check_dual_encoder_v3_gate.py`):

| # | Change | Gate | Overall BERTScore Δ | organ Δ | task_id Δ | forceps_type Δ | description Δ |
|---|---|---|---|---|---|---|---|
| 1 | Qwen resolution 112²→224² | FAIL 1/4 | +0.0003 | +0.048 | +0.018 | -0.0125 | -0.0127 |
| 2 | vision tower LoRA r=8 | **PASS 4/4** | +0.0063 | +0.184 | +0.048 | +0.0071 | -0.0029 |
| 3 | text LoRA rank 32 | FAIL 3/4 | +0.0038 | +0.091 | +0.063 | -0.0070 | -0.0081 |
| 4 | CrossModalFusion cross-attn layer | FAIL 3/4 | +0.0019 | -0.033 | +0.051 | +0.0063 | -0.0222 |
| **combo (2+3)** | **vision LoRA r=8 + text LoRA r=32** | **PASS 4/4** | **+0.0067** | +0.054 | +0.081 | +0.0131 | **+0.0481** |

Experiment 1 (more raw resolution, no added adaptation capacity) and
Experiment 4 (an explicit fusion layer between the two visual streams)
both failed to beat a small low-rank adaptation of the vision tower itself
(Experiment 2) or the text-side LoRA capacity (Experiment 3). This combo
run tested whether Experiments 2 and 3 - which act through different parts
of the model (vision-side domain adaptation vs. text-side fusion capacity)
- would stack. They did: the combo has the best overall BERTScore of all
five dev-scale runs, passes the gate outright (no override needed, unlike
the earlier `descfix` release), and is the **only** one where `description`
improved rather than regressed (+0.048) - every other architecture
experiment cost a little `description` quality in exchange for gains
elsewhere; this one didn't.

## Threshold changes

All 3 Yes/No thresholds re-swept on this checkpoint
(`src/calibrate_yes_no_thresholds.py`, 2026-08-16, full case140-154
holdout, 2407 examples) via the dev-lineage proxy checkpoint
(`dual_encoder_v3_stage_b_dev_combo` - the all155 checkpoint has no fair
holdout of its own). All 3 calibrated accuracies are the highest measured
for any checkpoint in this project so far:

| | old (descfix) | new (combo) | calibrated accuracy |
|---|---|---|---|
| tool_presence | 0.2 | **0.1** | 76.3% (best yet) |
| tissue_cutting | 1.0 | **-1.7** (large shift) | 90.5% (best yet) |
| suture_required | 0.3 | **0.6** | 94.5% (best yet) |

See `submission/inference_policy_combo.json` for full detail and
provenance.

## Verified before packaging

- Built via `submission/build_image_combo.sh
  surgvu-dual-encoder-v3:combo-20260816` - an isolated copy of
  `build_image.sh`/`Dockerfile`/`inference_policy.json` (all suffixed
  `_combo` / `.combo`) that does **not** modify the files backing any
  previously-deployed container. Includes the new `vision_lora_adapter/`
  checkpoint component (absent from every earlier release, since this is
  the first checkpoint with `vision_lora_r>0`).
- `docker inspect`: `Config.User=user`, `linux/amd64`,
  `sha256:84a358320729342e5dd7dca693dd7e545e895fb57a291ee9dd3cd9650118a430`.
- Smoke-tested through the **real** Grand Challenge single-case contract
  (`docker run --gpus all --network none -v .../input:/input:ro -v
  .../output:/output ...`, `-e SURGVU_GPU_GUARD_IGNORE_PIDS=6529` to
  bypass this host's NX desktop session in the GPU-idle guard) - exit 0,
  valid `visual-context-response.json` (`"Yes"` for case122's "Are there
  forceps being used here?" - ground truth "No"; the same tool_presence
  "video-present -> say yes" bias documented for every prior release, not
  a packaging defect - and confirmed the container correctly
  auto-constructs the model with `vision_lora_r=8`/`lora_r=32` purely from
  `dual_encoder_vqa_config.json`, no extra CLI flags needed for
  architecture).
- Archive produced by a single, direct `docker save | gzip` - no manual
  repacking. `docker load` round-tripped from this exact archive file
  successfully - identical image ID before/after.

## Upload

**Not yet uploaded.** Per the SurgVU Org Team's submission-limits memo,
final-phase submissions are capped at 3/team and strictly enforced (final
phase opens 2026-08-21). Recommend uploading to the **prelim** phase first
(no slot cost) to check whether the architecture change moves the real
leaderboard score - `release_20260813_descfix` (the data-only fix) has not
yet been confirmed against the real leaderboard either as of this writing,
so that upload should probably happen first or alongside this one to keep
the two changes' effects distinguishable.

Upload `surgvu_dual_encoder_v3_combo_20260816.tar.gz` through Grand
Challenge: `Algorithm` -> `Containers` -> `Upload a Container`.

Verify before transfer:

```bash
sha256sum -c SHA256SUMS
```
