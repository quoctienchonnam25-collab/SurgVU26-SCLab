# SurgVU 2026 Category 2 — plain Qwen2.5-VL + container robustness (2026-09-02)

**Supersedes `release_20260831_plain_qfix` and `release_20260827_plain_postprocess`.
Upload this one.**

Same architecture and LoRA weights as `release_20260821_plain` (plain
`Qwen2.5-VL-3B-Instruct` + text-LoRA r=16, 5 frames, no SurgMotion branch, trained on
the original long-form `description` data). Adds the same grading-path robustness fix
applied to the dual-encoder release, plus the 08-31 phrasing fixes.

See `../release_20260902_thresholdfix_robust/README_UPLOAD.md` for the full account of
how these defects were found (short version: the 08-31 build scored **0.8290 on
prelim, byte-identical to the unfixed container**, and `0.5526 / 0.8290 = 0.666586`
against `2/3 = 0.666667` suggested roughly a third of final cases were returning
nothing).

## What this architecture was and was not exposed to

- **Defect 1, the fatal GPU-idle guard: never applied here.** `assert_gpu_idle()`
  exists only in `predict_dual_encoder_vqa.py`; this pipeline never called it. That
  is a genuine structural advantage of this release — if a busy grading GPU is what
  cost cases on the final run, this architecture was already immune.
- **Defect 2, unguarded grading path: this release WAS exposed and is now fixed.**
  `predict_vqa()` already degraded to black frames when a video could not be read,
  but generation itself, the question-payload read and the output write were all
  unguarded, so any failure there left `/output` empty and scored the case zero.
  `run_grand_challenge_case()` is now structurally incapable of exiting without
  writing a response: everything is wrapped, an empty generation counts as a failure,
  and the fallback reuses `answer_postprocess.postprocess_answer(question, "")` for a
  per-category prior.

Verified with a deliberately corrupt mp4: this build logs the read failure and still
writes `"The tissue in the surgical field"`, exiting 0.

## Also included, from the 08-31 build

Detector coverage of the 11 official public samples went **8/11 → 11/11**, fixing the
organizers' real phrasings — *"What type of forceps is mentioned?"* (they write
"type of forceps" and "mentioned", not "forceps type" and "installed") and
*"What is the purpose of using **forceps**…"* where unqualified "forceps" had no
`TOOL_PURPOSE` entry, yet the official reference answer is character-for-character
`TOOL_PURPOSE["cadiere forceps"]`.

The binary-path misrouting fix from that build does not apply here: this pipeline has
no threshold routing and always uses raw `generate()` output.

## Verified

- Corrupt-video test: exits 0 with a valid response.
- Normal single-case Grand Challenge contract (`docker run --gpus all --network
  none`): exit 0, valid `visual-context-response.json`.
- 0 of 1314 true-holdout rows changed by post-processing.
- `docker inspect`: `Config.User=user`, `linux/amd64`,
  `sha256:f477812d079e37faf3fa0d4d34823f1ad57e5b0a66dec40b2c6383c3989b905e`.

## Honest limitations

- This architecture has **still never run on real Grand Challenge infrastructure**.
  `release_20260806_thresholdfix` remains the only lineage with confirmed real scores
  (0.8290 prelim / 0.5526 final).
- No Yes/No threshold calibration — raw `generate()` output for the binary
  categories.
- Post-processing fixes format, never content; `description` and `task_id` are
  deliberately untouched.

## Upload

```bash
sha256sum -c SHA256SUMS
```

Then Grand Challenge → `Algorithm` → `Containers` → `Upload a Container`.

## Strategy with 2 slots left (deadline 2026-09-06)

Only the best of the 3 final submissions counts, so there is no downside to using
both remaining slots. Suggested order: upload
`release_20260902_thresholdfix_robust` first — it holds the checkpoint identical to
the known 0.5526 baseline, so its score isolates the effect of the robustness and
routing fixes. Then this release, which additionally tests the architecture reset.

Worth confirming with the organizers first: whether a run annotated "scored as
failed" (as AIMS's 26 Aug run was) consumes one of the 3 slots.
