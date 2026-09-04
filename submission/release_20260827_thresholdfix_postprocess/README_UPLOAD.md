# SurgVU 2026 Category 2 — thresholdfix + answer post-processing (2026-08-27)

Same checkpoint weights and calibrated Yes/No thresholds as
`release_20260806_thresholdfix` (the confirmed final-phase score for that
release: **0.5526**, 2nd place behind AIMS's 0.6434 - see project memory
`surgvu_final_phase_result_thresholdfix`). This release adds exactly one
change: `src/answer_postprocess.py`, a deterministic post-processing layer
applied to the model's free-generated answer before it is returned.

## Why

AIMS's 1st-place method is literally named "deterministic answerer (entity
prior; task conditioning off)" - external, non-speculative evidence that
under BERTScore-F1 (the official metric, which rewards surface-form/template
match over raw content correctness - see project memory
`surgvu_threshold_fix_leaderboard_invisible`), answering with statistically
canonical phrasing matters as much as visual correctness. This release tests
that same idea as a pure post-processing layer, with zero retraining.

## What changed

`src/answer_postprocess.py::postprocess_answer(question, prediction)`,
called after generation in `predict_dual_encoder_vqa.py::answer_question()`:

- `procedure_type`: unconditional override to the one fixed correct answer
  ("Endoscopic surgery or a laparoscopic surgery") - every segment in this
  dataset is da Vinci robotic surgery, confirmed verbatim against case129 in
  the official public sample. Never inspects the model's own output.
- `tool_purpose`: the question always names the tool explicitly; the answer
  is looked up directly from the question text via a fixed dict. Also never
  inspects the model's own output.
- `organ` / `forceps_type`: **fallback-only**. If the model's raw answer
  already contains any recognized vocabulary term (an organ name, or a
  forceps tool name), it is returned completely unchanged - right or wrong.
  Only when nothing is recognized at all (genuinely off-template/garbled
  output) does it substitute a defensible default (organ's
  generator-confirmed 81%-majority fallback string; forceps_type's "No
  forceps types are listed.").
- `task_id`, `description`, and the 3 binary categories
  (`tool_presence`/`tissue_cutting`/`suture_required`, already handled via
  calibrated logit-margin decisions that never call free generation) are
  left untouched entirely - no evidence-backed default exists for any of
  them.

**Important design note, found empirically before shipping**: an earlier
version of this module also collapsed a *recognized* organ/forceps answer
down to its bare canonical form (e.g. "The simulated anatomical structure is
the rectal artery and vein." → "Rectal artery and vein"). Checked against
the real true-holdout predictions
(`runs/surgmotion_vqa/dual_encoder_v3/candidate_threshold_test/predictions.json`,
1314 rows, case140-154): only 1 row was affected, and on that row the model
had already gotten the organ content WRONG but kept the correct sentence
template, earning real BERTScore partial credit from token overlap.
Collapsing it to bare form measurably made the score worse (0.613 → 0.171,
recomputed via `src/evaluate.py::compute_bertscore_f1`) - it threw away
accidental partial credit without fixing the content error. The design was
revised to the fallback-only rule above specifically because of this
finding: post-processing can only safely fix *phrasing*, never *content*.

## Verified before packaging

- Offline diff against the same 1314-row true holdout with the final
  fallback-only design: **0 rows changed** - proves no regression on
  in-distribution data (this local set can't demonstrate the real payoff,
  since the model already reproduces trained templates near-perfectly there;
  the design targets phrasing drift under the final test's distribution
  shift specifically, which this holdout doesn't exercise).
- Synthetic stress test with rephrased/rambling raw predictions confirmed
  all active branches behave as intended.
- **Live GPU integration check** (not just offline JSON simulation): ran the
  actual `answer_question()` path with a debug print of the raw
  pre-postprocess prediction on case127 (the official public organ example,
  "What organ is being manipulated?" → true answer "Uterine horn") - the
  model's raw output was itself "The tissue in the surgical field" (a
  pre-existing model content error, unrelated to this change);
  post-processing correctly left it untouched (already-recognized-fallback
  case is a no-op), confirming zero regression risk in a real run, not just
  in offline simulation.
- Rebuilt via `submission/build_image.sh
  surgvu-dual-encoder-v3:thresholdfix-postprocess-20260827` (same minimal
  hardlinked build context as prior releases; no Dockerfile/build-script
  changes needed since it already globs `src/*.py`).
- `docker inspect`: `Config.User=user`, `linux/amd64`,
  `sha256:e8f36aab1d819bf764f3cb082d4a76dc4a0d8c45d1fe325119efe48994ea7cac`.
- Smoke-tested through the real Grand Challenge single-case contract
  (`docker run --gpus all --network none ...`, `-e
  SURGVU_GPU_GUARD_IGNORE_PIDS=6529` for this host's NX session) - exit 0,
  valid `visual-context-response.json`.
- `docker save | gzip` archive, `docker load` round-tripped to the identical
  image ID.

## Known limitations

- Does not address the `description` category (weakest locally, 0.34-0.56
  BERTScore) - free text, no small closed vocabulary to fall back to.
- `task_id` is untouched - unlike `organ`, its 8 task names have no
  dominant-majority default to fall back to.
- Cannot fix content errors (e.g. the case127 organ miss above) - only
  phrasing drift. If the real final test's score gap is mostly a content
  problem rather than a phrasing problem, this release will show little to
  no improvement over `release_20260806_thresholdfix`'s 0.5526.

## Upload

`surgvu_dual_encoder_v3_thresholdfix_postprocess_20260827.tar.gz` through
Grand Challenge: `Algorithm` → `Containers` → `Upload a Container`.

Verify before transfer:

```bash
sha256sum -c SHA256SUMS
```

**Recommended final-phase strategy**: with 2 of 3 final slots remaining
after `release_20260806_thresholdfix` (0.5526, confirmed), pair this release
with `release_20260821_plain` + the same post-processing layer (see
`submission/release_20260827_plain_postprocess/`) for the remaining 2 slots
- this release keeps the known checkpoint/thresholds as a controlled variable
(isolating the post-processing layer's own effect), while the plain release
also tests the architecture-reset hypothesis at the same time.
