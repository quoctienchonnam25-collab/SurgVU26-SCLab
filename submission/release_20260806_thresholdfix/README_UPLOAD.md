# SurgVU 2026 Category 2 — calibrated Yes/No threshold bundle (2026-08-06)

Builds on `release_20260806_fp16fix` (same checkpoint weights, same FP16/T4
fix) with one change: the calibrated first-token logit-margin decision for
binary (Yes/No) questions, previously wired only for `tool_presence`
(threshold 0.5, picked by an earlier session without a data-driven sweep),
is now (a) re-calibrated via an actual holdout sweep and (b) extended to the
other 2 binary question types in this corpus, `tissue_cutting` and
`suture_required`.

## What changed and why

Free-generation (no threshold) evaluation on the full 2254-example holdout
(`checkpoints/dual_encoder_stage_b_dev`, case140-154 - excluded from that
checkpoint's training, see `src/calibrate_yes_no_thresholds.py`) showed
`tool_presence` accuracy at only 73.6% with a 38% false-positive rate,
confirming real room for improvement in the decision rule versus the answer
text quality itself (overall BERTScore-F1 stayed 0.909, i.e. this defect was
invisible to the aggregate metric).

A threshold sweep (`src/calibrate_yes_no_thresholds.py`, margins from
-3.0 to 3.0 in steps of 0.1, `-3 + 0.1*i` for i in 0..60) found:

| question_type      | n   | best threshold | accuracy: greedy → calibrated |
|---------------------|-----|-----------------|-------------------------------|
| tool_presence        | 954 | 0.2             | 73.6% → 74.3%                 |
| suture_required       | 221 | 0.3             | 90.5% → 91.0%                 |
| tissue_cutting        | 178 | 1.4             | 85.4% → 89.3%                 |

`src/predict_dual_encoder_vqa.py::answer_question()` now routes all 3
question types through the same generic `tool_presence_answer_tensor()`
margin mechanism (question-agnostic despite its name), each gated by its own
keyword detector (`is_tool_presence_question`, `is_tissue_cutting_question`,
`is_suture_required_question`) and its own independently-calibrated
threshold - `tissue_cutting`'s bias runs in the opposite direction from
`tool_presence`'s, so a single shared threshold would not have been correct.

**Caveat**: these thresholds were calibrated on the dev checkpoint's fair
holdout, not on `dual_encoder_v3_stage_b_all155` itself - that checkpoint
trains on all 155 cases (including case140-154), so it has no fair holdout
of its own to calibrate against directly (see `src/calibrate_yes_no_thresholds.py`'s
docstring and `submission/inference_policy.json`'s
`threshold_calibration_note`). These values are the best available evidence,
transferred across checkpoints, not a value calibrated on the exact
submission weights.

## Also attempted, NOT included: DPO bias correction

A manual DPO fine-tune (`src/train_dpo_dual_encoder.py`, reference model via
PEFT's `disable_adapter()`) was tried on 37 preference pairs built from a
1292-example visual-ablation audit
(`src/build_dpo_pairs_from_ablation.py`). Result was a **regression**:
`tool_presence` accuracy fell 73.6%→70.9% and false-positive rate rose
38%→54%, because the 37-pair set was itself imbalanced (22 "prefer
Yes" vs 9 "prefer No" for tool_presence), so the fine-tune over-corrected
toward answering "Yes". That checkpoint
(`checkpoints/dual_encoder_stage_b_dev_dpo`) was discarded and is not part
of this release.

## Verified before packaging

- Rebuilt via `submission/build_image.sh
  surgvu-dual-encoder-v3:thresholdfix-20260806` (same minimal hardlinked
  build context as prior releases).
- `docker inspect`: `Config.User=user`, `linux/amd64`,
  `sha256:c6f7905006f48e11a2da92db3b5152083f167d562a89a6fb58c87e8c15a1ffe7`.
- Smoke-tested through the **real** Grand Challenge single-case contract
  (`docker run --gpus all --network none -v .../input:/input:ro -v
  .../output:/output ...`) - exit 0, valid `visual-context-response.json`
  (`"Yes"` for case122's "Are there forceps being used here?").
- Archive produced by a single, direct `docker save | gzip` - no manual
  repacking. Verified zero `./`-prefixed paths, bare top-level
  `manifest.json`.
- `docker load` round-tripped from this exact archive file successfully.

## Upload

**Not yet uploaded.** Per the SurgVU Org Team's submission-limits memo,
final-phase submissions are capped at 3/team and strictly enforced (excess
discarded, uncounted); as of 2026-08-06 zero final-phase submissions have
been used and the final phase doesn't open until 2026-08-21, so there is no
urgency - confirm with the team before consuming a slot on either phase.

If proceeding: upload `surgvu_dual_encoder_v3_thresholdfix_20260806.tar.gz`
through Grand Challenge: `Algorithm` → `Containers` → `Upload a Container`.

Verify before transfer:

```bash
sha256sum -c SHA256SUMS
```
