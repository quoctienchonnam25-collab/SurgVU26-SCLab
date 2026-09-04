# SurgVU 2026 Category 2 — final submission candidate (2026-09-02)

**Upload this one. It supersedes every other `release_*` directory.**

Same checkpoint and thresholds as `release_20260806_thresholdfix`. Contains every
verified fix accumulated since, and nothing else — no retraining, no architecture
change, no data change.

## Provenance: what is already confirmed on the real leaderboard

| submission | checkpoint | final score |
|---|---|---|
| `release_20260806_thresholdfix` | X | 0.5526 |
| `release_20260831_thresholdfix_qfix` | **X (identical)** | **0.5972** |

The +0.0446 came entirely from format/bug fixes with the model held fixed. Note the
same qfix container scored **0.8290 on prelim — byte-identical to the unfixed
build**, so prelim registered a real +0.0446 gain as exactly zero. Do not expect
prelim to say anything about this release either.

## What this build adds on top of the 0.5972 container

**1. Container robustness (2026-09-02).** Two defects that make a case score zero:

- `main()` called `assert_gpu_idle()` *before loading the model*, a dev-only guard
  that raises `RuntimeError("GPU is busy")` if the driver reports any other compute
  process. Proven fatal: holding 256 MB on the GPU and running the previous package
  gives `GPU is busy: ...` and **no response file**. Inside a container `nvidia-smi`
  cannot resolve host PIDs, so its `"nxnode"` name filter never matches — every
  process on a grading GPU counts as a conflict. Now skipped when
  `--grand-challenge` is set.
- `run_grand_challenge()` had **no exception handling**, so any decode/CUDA failure
  left `/output` empty. Verified with a corrupt mp4: previous package exits 1 with no
  file; this one exits 0 with a sensible fallback. The path is now structurally
  incapable of exiting without writing a response.

**2. Two more routing defects, found by a 54-case paraphrase audit** (new,
`src/audit_question_paraphrases.py` — run it before any future packaging):

- `"Does this clip involve a grasping retractor?"` fell through to prose: the word
  list held `"involved"`, which does not match `"involve"`. Now stem-based
  (list/install/use/involve).
- `"Is suturing required here?"` fell through: `"suturing"` does not contain the
  substring `"suture"` (no 'e'). Now matches on the stem `"sutur"`.

Audit status: **54/54 paraphrases**, **11/11 official public samples**, **0
misroutings across all 1314 true-holdout rows**.

**3. A non-finite Yes/No margin no longer silently answers "No".** Probing 300
random binary holdout questions found **1 whose margin came back `nan`** (FP16
overflow in the logits - the same numeric fragility that forced `do_sample=False`
in the plain pipeline). Since `nan >= threshold` evaluates to False, that question
was answered "No" for no reason at all. `answer_question()` now checks
`math.isfinite(margin)` and falls back to ordinary generation, reaching Yes/No
through a different numeric path instead of defaulting to one polarity.

**4. forceps_type answers are framed as a sentence echoing the question.** The five
references per question are shaped `[bare answer, frame1 + answer, ... frame4 +
answer]` and the metric takes the max, which makes surface form worth real points.
Measured against the official case124 references: a *wrong* bare name scores
**0.240** while the same wrong name inside the question's own frame scores
**0.772**, and a *correct* answer scores 1.000 either way. So
`canonicalize_forceps_type()` now returns the bare name and `frame_answer()` wraps
it — e.g. "The forceps in question is Bipolar Forceps." (0.609) becomes "The type of
forceps mentioned is Bipolar Forceps." (0.772).

Effect, measured on both sets: holdout 1314 rows **0.9201 → 0.9261**, official 11
samples **0.8237 → 0.8386**. Only forceps_type changes (+0.0620 over its 128 holdout
rows); every other category is bit-identical.

⚠️ **Applied to forceps_type only, deliberately.** Framing organ and task_id the
same way gained **+0.0329 on the 11 official samples** but lost **0.237 (organ)** and
**0.271 (task_id)** on the 1314-row holdout, net −0.0104. A correct bare name matches
`reference[0]` *exactly* (1.000), so a frame only pays off when it also matches one
of `references[1..4]` — and those frames had been written from the 11 official
questions, so they fit those and little else. The real-data sample was the
*misleading* one here; only checking the larger set caught it. Binary questions are
excluded for the opposite reason: bare "Yes"/"No" scores 0.851 in expectation versus
0.788–0.820 framed, and hedging ("may be", "it is unclear") is worst at 0.542–0.616.

## Verified

- 11 official public samples end-to-end through the real Grand Challenge contract:
  **BERTScore-F1 0.8386**, every answer well-formed. (These cases are inside the
  train split, so treat the number as a form check, not a generalization estimate.)
- Busy-GPU test: answers normally where the previous package crashed.
- Corrupt-video test: exits 0 with a valid response.
- `docker inspect`: `Config.User=user`, `linux/amd64`,
  `sha256:676d3f17c34665fceda5ed27ca22bd41f38b119cf4eec5e685c50b1a40e69b96`.

## What was investigated and deliberately NOT done

- **Retraining with question-paraphrase augmentation.** Premise tested first and
  **refuted**: asking the model the same question in this project's template wording
  vs the organizers' wording returns identical answers on every pair tried. The
  model is already phrasing-robust; only the regex detectors were brittle. No
  retrain was run.
- **A deterministic "No" prior for tools outside the 12-tool vocabulary.** Probed
  the container: it already answers Yes/No in the correct *form* for binary
  questions no detector claims, and there is no evidence the test contains
  out-of-vocabulary tool questions. Adding the prior would have been a guess.
- **Multi-view frame voting on the Yes/No path.** Implemented, measured, removed. A
  60-question probe suggested +2.5pp binary accuracy (it recovered 3/3 decisions
  that flip between frame subsets); re-running on 300 questions collapsed that to
  **+0.33pp** (77.59% → 77.93%, ~+0.0006 BERTScore), with averaging correct on only
  12/18 flips against a 50% baseline at SE 11.8pp. The underlying noise is real -
  frame-subset choice flips 6.3% of decisions, and 43.8% of them sit within 0.5
  logit of the threshold - but two uniform views land ~1s apart in a 30s clip and
  are far too correlated to cancel it. Not worth doubling inference time.
- **Any threshold re-tuning or data/architecture change.** Six such attempts in this
  project's history produced zero confirmed gains and several regressions.

## Open finding, not acted on — read before the methodology report (due 2026-09-13)

The organizers' answers appear to be derived from a **per-clip text summary**, not
from the tools.csv installation record this project labels from. case122's official
answer is *"No forceps"* while our labels have forceps present in **557/557**
segments; case126's is *"Yes, large needle driver"* while ours have it in 12%. Their
answer wording (*"not mentioned"*, *"not listed"*, *"no indication"*) and case129's
question (*"What procedure is this **summary** describing?"*) both point at a text
source. If true, our largest category is supervised against a subtly different
question from the graded one — a ground-truth definition mismatch that no retraining
on current labels can fix. Based on 3 disagreements out of 11, so recorded rather
than acted on.

Also corrected: a wrong Yes/No costs **~0.299** BERTScore against official
references (0.701 vs 1.000), not ~0 as this project long assumed. Binary questions
are 7/11 of the official sample, so binary accuracy is the real remaining lever —
the old "BERTScore is blind to Yes/No" conclusion came from a threshold change that
moved accuracy only +0.7pp, which is genuinely invisible.

## Upload

```bash
sha256sum -c SHA256SUMS
```

Then Grand Challenge → `Algorithm` → `Containers` → `Upload a Container`.

This is the last of 3 final-phase slots (0.5526 and 0.5972 already used). Only the
best score counts, so 0.5972 remains the floor regardless of the outcome. Worth
confirming with the organizers beforehand whether a run annotated "scored as failed"
consumes a slot.
