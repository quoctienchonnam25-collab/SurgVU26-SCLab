# SurgVU 2026 Category 2 — thresholdfix + container robustness (2026-09-02)

**Supersedes `release_20260831_thresholdfix_qfix` and
`release_20260827_thresholdfix_postprocess`. Upload this one.**

Same checkpoint weights and calibrated thresholds as `release_20260806_thresholdfix`
(confirmed **0.8290 prelim / 0.5526 final**). Adds two container-level defect fixes
found on 2026-09-02, on top of the question-routing and truncation fixes from 08-31.

## Why these were looked for

`release_20260831_thresholdfix_qfix` scored **0.8290 on prelim — byte-identical to
the unfixed container**, which is also exactly what the earlier threshold-calibration
release scored. Three different containers, the same number three times: the prelim
set is small enough that it exercises none of the changed paths, so an unchanged
prelim score is no evidence in either direction. (The organizers said as much: prelim
rank "will likely not mean anything".)

That prompted re-deriving where the final-phase loss actually goes:

```
0.5526 / 0.8290 = 0.666586        2/3 = 0.666667
```

A match to ~1e-4 — precisely the shape you would get if **one third of final cases
produced no output at all** while the rest scored at prelim quality. That is
circumstantial (the final set may simply be harder), but it motivated auditing what
can make a case return nothing. Two such defects were found, and both are real
regardless of whether they explain this particular number. Independent supporting
signal: the AIMS team's 26 Aug final run is annotated on the leaderboard as having
"scored as failed", so case/run failures demonstrably occur on this platform.

## Defect 1 — the container kills itself if the grading GPU is not idle

`main()` called `assert_gpu_idle()` *before loading the model*. That helper exists so
a local run does not collide with training on this workstation's single GPU. In the
submission container it is fatal: it shells out to `nvidia-smi` with `check=True`
(so a missing binary or any non-zero exit raises) and then raises
`RuntimeError("GPU is busy")` if the driver reports **any** other compute process —
normal on shared grading infrastructure and entirely outside our control. The process
dies before the model loads, nothing is written to `/output`, the case scores zero.

**Proven on the packaged image, not inferred from code.** Holding ~256 MB on the GPU
with a dummy process and running
`surgvu-dual-encoder-v3:thresholdfix-postprocess-20260831` on a valid video:

```
RuntimeError: GPU is busy: 6529, [Not Found], 293; 689600, [Not Found], 554
response file: <none>
```

Note `[Not Found]`: inside the container `nvidia-smi` cannot resolve host PIDs to
process names, so the helper's `"nxnode" not in row.lower()` filter **never
matches** — the guard is strictly more trigger-happy in the container than it is
locally. Every process on the grading GPU counts as a conflict. This build answers
normally under the identical condition.

Fix: `assert_gpu_idle()` now runs only when `--grand-challenge` is **not** set, so
local development keeps its protection and the submission path never calls it.

## Defect 2 — no exception handling anywhere on the grading path

`run_grand_challenge()` had none. An undecodable video, a transient CUDA error or an
unexpected question payload propagated out of `main()` and left `/output` empty.
Demonstrated with a deliberately corrupt mp4:

| image | exit | response file |
|---|---|---|
| `thresholdfix-postprocess-20260831` (previous package) | 1 | **none** |
| this build | 0 | `"The tissue in the surgical field"` |

Fix: the grading path is now structurally incapable of exiting without writing a
response. Everything is wrapped, an empty generation counts as a failure, and the
fallback reuses `answer_postprocess.postprocess_answer(question, "")` so each
category gets its defensible prior (organ → the majority fallback string,
procedure_type → the single correct answer, tool_purpose → the looked-up purpose,
forceps_type → "No forceps types are listed."). Binary questions fall back to "Yes",
an arbitrary coin flip (the official sample is 4 Yes / 3 No) justified only because
any sentence beats writing nothing. Unrecognised shapes get "The surgical field is
visible in this clip."

## Also included, from the 08-31 build

- **Open-ended questions were misrouted to the binary Yes/No path**, so *"Which
  forceps type is listed as installed for this clip?"* was literally answered
  `"Yes"` by the shipped 0.5526 container. BERTScore-F1 against case124's real
  reference set: `"Yes"` = **−0.0392**, the fixed path's `"Bipolar Forceps"` =
  **0.2402**, a correct `"Cadiere Forceps"` = 1.0000.
- **Detector coverage of the 11 official public samples went 8/11 → 11/11**, fixing
  the organizers' actual phrasings (*"type of forceps"* / *"mentioned"*; unqualified
  *"forceps"* in a purpose question; *"cut"* rather than *"cutting"*).
- **`max_new_tokens` 24 → 50** (matching the plain pipeline), ending mid-clause
  truncation: measured +0.0703 on description rows, truncation 10/21 → 0/21, four
  rows reaching an exact 1.000.

## Verified

- Busy-GPU test: answers normally where the previous package crashed.
- Corrupt-video test: exits 0 with a valid response where the previous package
  exited 1 with no file.
- Normal single-case Grand Challenge contract (`docker run --gpus all --network
  none`): exit 0, valid `visual-context-response.json`.
- 11/11 official public-sample questions route correctly; 0 of 1314 true-holdout
  rows changed by post-processing.
- `docker inspect`: `Config.User=user`, `linux/amd64`,
  `sha256:c36334c7a972d341ab16462f8ce641074a4273b527d14f13b7ab579a9051baa0`.

## Honest limitations

- It is **not proven** that either defect caused the 0.5526 result; the 2/3 ratio is
  suggestive, not conclusive.
- Neither defect is detectable by any local metric, holdout or prelim run — they only
  manifest on grading infrastructure. Do not expect prelim to confirm this build.
- The generic-`"forceps"` → cadiere-purpose mapping rests on a single official
  sample (case130), though it matches character-for-character.

## Upload

```bash
sha256sum -c SHA256SUMS
```

Then Grand Challenge → `Algorithm` → `Containers` → `Upload a Container`.
