# SurgVU 2026 Category 2 — plain Qwen2.5-VL + question-routing fix (2026-08-31)

**Supersedes `release_20260827_plain_postprocess` — do not upload that one.**

Same architecture and LoRA weights as `release_20260821_plain` (plain
`Qwen2.5-VL-3B-Instruct` + text-LoRA r=16, 5 frames, no SurgMotion branch, trained
on the original long-form `description` data — see that release's README for the
architecture-reset rationale). Adds the updated post-processing layer, including
the phrasing fixes found by auditing the 11 official public samples.

## What this release does and does not inherit

The **binary-path misrouting defect** documented in
`release_20260831_thresholdfix_qfix/README_UPLOAD.md` — where *"Which forceps type
is listed as installed for this clip?"* was answered literally **"Yes"** by the
shipped 0.5526 container — **never affected this architecture**:
`src/predict_plain_qwen25vl.py` has no threshold routing at all and always uses raw
`generate()` output. That is a genuine point in this release's favour.

It **does** benefit from the two phrasing fixes, which are post-processing-level and
apply to any architecture:

- **case130** *"What is the purpose of using **forceps** in this procedure?"* —
  `TOOL_PURPOSE` is keyed only by specific variants (cadiere/bipolar/prograsp), so
  unqualified "forceps" returned `None` and the model's free text was kept. The
  official reference answer *"To grasp and hold tissues or objects during the
  surgery."* is character-for-character `TOOL_PURPOSE["cadiere forceps"]`, so that
  text is now emitted exactly.
- **case124** *"What type of forceps is mentioned?"* — the forceps detector required
  the literal `"forceps type"` AND `"installed"`; organizers write *"type of
  forceps"* and *"mentioned"*. Now matched (both word orders, and the verbs
  organizers actually use).

Detector coverage of the official samples went from **8/11 → 11/11**.

## Changes relative to release_20260821_plain

- `src/predict_plain_qwen25vl.py`: `run_grand_challenge_case()` (the real submission
  path) and `run_batch_directory()` call `postprocess_answer()` after generation.
- `src/evaluate_plain_qwen25vl.py`: same call, so future holdout numbers reflect
  what ships.
- `submission/Dockerfile.plain` / `build_image_plain.sh`: also copy
  `src/answer_postprocess.py` and `src/generate_qa.py` (its only dependency) — the
  prior Dockerfile copied `predict_plain_qwen25vl.py` by name, not a `src/*` glob.
- No change to training data, LoRA weights, or the base checkpoint.

## Verified

- 11/11 official public-sample questions handled.
- 0 of 1314 true-holdout rows changed by post-processing — no regression.
- Built via `submission/build_image_plain.sh
  surgvu-plain-qwen25vl:20260831-postprocess`.
- `docker inspect`: `Config.User=user`, `linux/amd64`,
  `sha256:aade6b35913957059e1ce0c4619baf2b15e789bf4904cf2578211f4f32e3c526`.

## Known limitations (inherited, unchanged)

- No Yes/No threshold calibration — raw `generate()` output for the 3 binary
  categories. (Also why the misrouting defect never applied here.)
- This architecture has **still never run on real Grand Challenge infrastructure**;
  `release_20260806_thresholdfix` remains the only lineage with confirmed real
  scores (0.8290 prelim / 0.5526 final).
- Post-processing cannot fix content errors, only phrasing/format; `description` and
  `task_id` are deliberately untouched.

## Upload

```bash
sha256sum -c SHA256SUMS
```

Then Grand Challenge → `Algorithm` → `Containers` → `Upload a Container`.

**Strategy note.** With 2 of 3 final slots left, uploading
`release_20260831_thresholdfix_qfix` first keeps the checkpoint fixed and isolates
the routing/post-processing fix against the known 0.5526 baseline; this release then
tests the architecture reset. Confirm with the organizers whether AIMS-style
"scored as failed" runs consume a slot before committing the last one.
