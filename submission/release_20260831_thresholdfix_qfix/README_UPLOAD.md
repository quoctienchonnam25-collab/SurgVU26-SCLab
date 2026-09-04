# SurgVU 2026 Category 2 — thresholdfix + question-routing fix (2026-08-31)

**Supersedes `release_20260827_thresholdfix_postprocess` — do not upload that one.**

Same checkpoint weights and calibrated Yes/No thresholds as
`release_20260806_thresholdfix` (confirmed final-phase score **0.5526**, 6th
place). This release fixes a **real defect proven to exist in that shipped
container**, plus the post-processing layer from the 08-27 build.

## The defect (proven end to end, not inferred from code)

`src/predict_dual_encoder_vqa.py::is_tool_presence_question()` matched on "a tool
name appears" AND "a presence word appears", with **no check that the question was
actually Yes/No-shaped**. `generate_qa.py::gen_forceps_type()`'s own templates —
*"Which forceps type is listed as installed for this clip?"* — contain both
("forceps" + "listed"), so they were routed to the calibrated binary
logit-margin path and answered literally **"Yes"**, against reference answers like
*"Cadiere Forceps"*.

Running that exact question through the shipped image
`surgvu-dual-encoder-v3:thresholdfix-20260806` returns `"Yes"`. Measured with
`src/evaluate.py::compute_bertscore_f1` against case124's real 5-reference set:

| prediction | BERTScore-F1 |
|---|---|
| `"Yes"` (old, misrouted) | **−0.0392** |
| `"Bipolar Forceps"` (new path; model's brand guess is wrong) | **0.2402** |
| `"Cadiere Forceps"` (correct) | **1.0000** |

So the fix is worth **+0.28 per affected question even when the model gets the
brand wrong**, and +1.04 when it's right — and this checkpoint scores 0.93 on
forceps_type in-distribution, so it usually is right.

## How it was found: auditing the 11 official public samples

Every detector in this project was written against `generate_qa.py`'s **synthetic**
templates and had never been validated against the organizers' real phrasing. The
11 official public samples are the only organizer-authored QA data available.
Checking each question against each detector found **3 of 11 mishandled**:

- **case124** *"What type of forceps is mentioned?"* — detector required the literal
  `"forceps type"` AND `"installed"`; organizers write *"type of forceps"* and
  *"mentioned"*. Nothing matched at all.
- **case130** *"What is the purpose of using **forceps** in this procedure?"* — asks
  about unqualified "forceps"; `TOOL_PURPOSE` is keyed only by specific variants, so
  the lookup returned `None`. Its official reference answer *"To grasp and hold
  tissues or objects during the surgery."* is **character-for-character**
  `TOOL_PURPOSE["cadiere forceps"]` — the organizers' key for bare "forceps" IS the
  cadiere text.
- **case131** *"Is tissue being cut during this clip?"* — the detector looked for
  `"cutting"` or (`"tissue"` and `"dissect"`); this phrasing has *"cut"*, so it
  silently missed its calibrated threshold. This is also one of generate_qa.py's own
  3 templates, so ~1/3 of tissue_cutting questions were never thresholded.

**After the fixes: 11/11 handled**, and case130 now returns the official reference
answer exactly.

## Changes

- New `is_binary_question()`: a question opening with
  what/which/who/where/when/how/why/name/list/describe/identify can never reach any
  binary threshold path. All 3 binary detectors gated on it.
- `is_tool_presence_question()` additionally excludes anything
  `is_forceps_type_question()` claims — gen_forceps_type's *no-forceps* template is
  Yes/No-shaped (*"Are any forceps types listed…?"*) yet answers with a full
  sentence (*"No forceps types are listed."*), never a bare "No".
- `is_tissue_cutting_question()` now accepts `\bcut(ting)?\b`.
- `answer_postprocess.is_forceps_type_question()` widened to both word orders and
  the verbs organizers actually use (installed/listed/mentioned/used/recorded/present).
- `extract_tool_purpose_answer()` falls back to the cadiere purpose text for
  unqualified "forceps".
- Plus the 08-27 post-processing layer (procedure_type/tool_purpose deterministic
  override; organ/forceps_type fallback-only). See
  `release_20260827_thresholdfix_postprocess/README_UPLOAD.md` for that design and
  the empirical pitfall behind it (collapsing a recognized-but-wrong answer to bare
  form destroys BERTScore partial credit, 0.613→0.171).

## Second defect fixed: answers truncated mid-clause at 24 tokens

The dual-encoder path defaulted to `--max-new-tokens 24` (the plain path uses 50),
and the shipped ENTRYPOINT never overrode it. On the 1314-row true holdout, **14
predictions ended with no terminal punctuation, all of them 20-22 words** — cut off
mid-clause. Raised to 50, matching the sibling pipeline rather than being tuned
against any score.

Measured A/B on the 21 description rows (same checkpoint, same prompts, only the cap
changed), scored against the long-form references this checkpoint was trained on:

| max_new_tokens | mean BERTScore-F1 | truncated |
|---|---|---|
| 24 | 0.4078 | 10/21 |
| **50** | **0.4781** | **0/21** |

**+0.0703**; 9 rows improved, 1 marginally worse (−0.016), 11 unchanged. Four rows
reached an exact **1.000** that had scored 0.73/0.78/0.66 while truncated — the model
was already generating the reference verbatim and was being cut off.

⚠️ **A methodological warning recorded with this release.** The first run of this A/B
scored against `src/test_vqa_surgmotion_curated_v3.json` and reported **−0.0083**,
i.e. "the fix makes things worse". That file holds the *short-form* description
answers created by the 2026-08-13 descfix, while this release's checkpoint
(`dual_encoder_v3_stage_b_all155`) was trained on the *long-form* originals in
`runs/dataset_audit/backup_pre_description_fix_20260812/`. Against deliberately
truncated references, a completed sentence overshoots and looks like a regression.
Same model, same predictions, opposite sign. Always confirm which description
variant a checkpoint was trained on before scoring it.

**Not re-calibrated**: the 3 thresholds in `submission/inference_policy.json` were
swept using the un-guarded detectors, so their sweep populations included the
misrouted open-ended questions. Re-sweeping needs a full holdout generation run,
and per project memory BERTScore-F1 is near-blind to Yes/No polarity anyway — the
win here is emitting a tool NAME instead of "Yes" at all, not the threshold value.

## Verified

- 11/11 official public-sample questions now route correctly.
- 0 of 1314 true-holdout rows (case140-154) changed by post-processing — no
  regression on in-distribution data.
- Container smoke test across all 8 question shapes on real GPU
  (`docker run --gpus all --network none`), each returning the correct *form*:
  binary → "Yes"/"No"; organ → text; procedure_type → the fixed answer;
  tool_purpose(forceps) → the exact official reference string; *"Are any forceps
  types listed…"* → full sentence, not a bare "No"; *"Which forceps type…"* →
  **"Bipolar Forceps"** (was `"Yes"`).
- `docker inspect`: `Config.User=user`, `linux/amd64`,
  `sha256:dfd7d5949509ca3fff70d884c21f68ba8742e209963899d7db1b00cc961f6e8d`.

## Upload

```bash
sha256sum -c SHA256SUMS
```

Then Grand Challenge → `Algorithm` → `Containers` → `Upload a Container`.

**Recommended as the next final-phase submission.** It holds the checkpoint fixed
against the known-0.5526 baseline, so any score change isolates the routing/
post-processing fix. Note the AIMS team's 26 Aug run is annotated as having
"scored as failed" and was re-uploaded — worth confirming with the organizers
whether a failed run consumes one of the 3 slots before spending the last one.
