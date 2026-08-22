# SurgVU 2026 - Category 2 (Surgical Visual Question Answering)

Team: SCLab-Surg (Chonnam National University) - Grand Challenge username `TienNQ27`

## Final-phase submission candidates (2026-08-22)

Three release lines are packaged and verified for the Grand Challenge final testing
phase (3 submissions/team, best score counts). Each `submission/release_*/README_UPLOAD.md`
documents its own build, evaluation, and verification steps in full; short summary:

| Release | Architecture | Real prelim score | Status |
|---|---|---|---|
| [`release_20260806_thresholdfix`](submission/release_20260806_thresholdfix/README_UPLOAD.md) | `DualEncoderVQA` (SurgMotion + frozen Qwen2.5-VL vision tower), calibrated Yes/No thresholds | **0.8290 (confirmed)** | Insurance floor - known-good, real-leaderboard-confirmed |
| [`release_20260816_combo`](submission/release_20260816_combo/README_UPLOAD.md) | `DualEncoderVQA` + vision-tower LoRA (r=8) + text LoRA (r=32) | 0.7653 (regressed vs. thresholdfix; likely a data-fix confound, see release notes) | Architecture combo, best local holdout score of the dual-encoder line |
| [`release_20260821_plain`](submission/release_20260821_plain/README_UPLOAD.md) | Plain `Qwen2.5-VL-3B-Instruct` + LoRA (r=16, text-only), 5 frames, **no SurgMotion branch** | Not yet tested on real prelim | Deliberate architecture reset - lightweight informed bet, built on historical + external precedent (see release notes for why) |

`src/dual_encoder_vqa_prototype.py`, `src/predict_dual_encoder_vqa.py`, `src/train_dual_encoder_vqa.py`
implement the `DualEncoderVQA`/SurgMotion line; `src/train_plain_qwen25vl.py`,
`src/predict_plain_qwen25vl.py`, `src/evaluate_plain_qwen25vl.py` implement the plain
Qwen2.5-VL line. `submission/build_image*.sh` + `submission/Dockerfile*` build each
release's container.

### Reproducing / running a final-phase container

See [`submission/README.md`](submission/README.md) for the exact build and run
command for each of the three candidates above (matches the Grand Challenge
single-case docker contract: `/input` in, `/output` out, offline at runtime).

### Data/models used beyond what the challenge provided (final-phase candidates)

- **SurgMotion** (`CAIR-HKISI/SurgMotion`, ViT-Large variant, Apache-2.0) - a
  video-native surgical foundation model pretrained on SurgMotion-15M, used as the
  motion-aware visual encoder in `DualEncoderVQA` (`thresholdfix` and `combo`
  releases only). [Paper](https://arxiv.org/abs/2602.05638),
  [weights](https://huggingface.co/CAIR-HKISI/SurgMotion),
  [code](https://github.com/CAIR-HKISI/SurgMotion). Not fine-tuned itself in
  `thresholdfix`; `combo` adds a low-rank LoRA adapter (r=8) on top of it.
- `release_20260821_plain` uses no data or pretrained components beyond
  `Qwen/Qwen2.5-VL-3B-Instruct` and the challenge-provided SurgVU training data.
- ProstaTDv2 domain-adaptive pretraining was re-tried against the current
  `DualEncoderVQA` line (underperformed the combo checkpoint on the gate check) and
  **not adopted**. The YOLOv5 tool-detector (see
  [Earlier iteration history](#earlier-iteration-history-v1-v5-superseded-by-the-releases-above)
  below) was only ever part of the earlier v1-v5 line and was never carried into the
  current architecture.

## Methodology - final-phase candidates

### Shared data pipeline

All three candidates start from the same base data path: `preprocess.py` +
`generate_qa.py` (9 question types over 30s video segments, per-tool and
per-commercial-variant Yes/No balancing - see the
[v1-v5 line's Methodology section](#methodology) below for the full mechanics,
unchanged at this layer). `curate_surgmotion_vqa.py` then applies quality-aware curation on top - it
drops unanswerable supervision on fully-black clips (found via a visual-signal
audit) and rewrites metadata-derived organ questions into simulation-appropriate
language - and `validate_qa_v3.py` runs schema/grammar checks (id/field
completeness, forbidden grammar patterns like "a forceps", presence-question
wording) before any GPU time is spent. Frames are pre-cached once
(`frame_cache_16fr/`, 384x384 JPEGs, 16 frames/segment) and reused read-only by
every training script, avoiding repeated video decoding.

One data variant diverges: the `description` question type originally answered
with a ~150-200 char, up-to-2-sentence excerpt of raw metadata. Comparing against
the 11 official public-sample Q&A pairs (the only real ground truth ever seen)
showed every reference answer is a single short sentence, so
`gen_description_question()` was rewritten (2026-08-13) to cap output at one
sentence (~140 chars). **`thresholdfix` and `plain` still use the original
long-form `description` answers; `combo` is the first release trained on the
short-form rewrite** (see `combo` below for why that matters).

### `DualEncoderVQA` architecture (`thresholdfix`, `combo`)

`Qwen2.5-VL-3B-Instruct`'s own vision tower is kept **fully intact** - native
multimodal RoPE, per-example dynamic resolution, untouched processing path - and
handles general scene understanding exactly as designed. A separate, frozen
**SurgMotion** encoder (`CAIR-HKISI/SurgMotion`, ViT-Large, 16-frame clips) runs
alongside it and is projected (`VisualProjector`, mean-pooled to `AUX_NUM_TOKENS
= 32` tokens) into a short auxiliary token sequence, inserted between the user's
prompt and the assistant's answer - giving the LoRA-adapted decoder an extra
surgical-motion-specific signal without touching Qwen's own visual path at all.

This design is a direct response to an earlier, more aggressive prototype
(`SurgMotionVQA`, see [Earlier iteration history](#earlier-iteration-history-v1-v5-superseded-by-the-releases-above))
that **replaced** Qwen's vision tower with SurgMotion entirely: it won on this
project's own held-out split but *lost* to a plain Qwen2-VL-2B LoRA fine-tune
(v7) on the real Grand Challenge hidden test set (0.7643 vs 0.7693) - trading
away Qwen's broadly-pretrained visual grounding for a narrower, only-shallowly-
adapted surgical encoder did not pay off out-of-distribution. A targeted
real-vs-blank ablation (`audit_surgmotion_visual_ablation.py`) found the one
place SurgMotion showed a real, causally-verified benefit was `forceps_type`
discrimination - `DualEncoderVQA` is a bet that keeping Qwen's native path fully
intact while adding SurgMotion only as a small supplementary signal keeps that
narrow benefit without the broad-robustness cost. Training (`train_dual_encoder_vqa.py`)
updates only the auxiliary projector and the LoRA-adapted decoder; both the Qwen
vision tower and the SurgMotion encoder stay frozen in the base configuration.

Every Yes/No-style question type (`tool_presence`, `tissue_cutting`,
`suture_required`) is decided by a calibrated first-token logit-margin
comparison rather than free-form generation - a threshold sweep
(`calibrate_yes_no_thresholds.py`, margins -3.0 to 3.0 in steps of 0.1) found
free-generation `tool_presence` accuracy was only 73.6% with a 38%
false-positive rate on a real, never-trained-on holdout (case140-154), while
this was invisible to the aggregate BERTScore-F1 metric (stayed 0.909) - i.e. a
real decision-rule defect the headline metric couldn't see. A manual DPO
fine-tune was also tried to correct this bias directly and made it worse (37
preference pairs were themselves imbalanced, so the model over-corrected toward
"Yes"); that checkpoint was discarded, not part of any release. Inference also
required an FP16-not-BF16 fix (`fp16fix`, folded into every release from
2026-08-06 on): the submission GPU is a T4 (Turing), which supports neither flash
attention (needs Ampere+) nor memory-efficient SDPA in BF16 (no native BF16
compute unit pre-Ampere) - the SurgMotion branch's attention backend had nowhere
left to dispatch to and crashed 6 minutes into a real graded run. Training stays
in BF16; only the inference path switched to FP16.

#### `thresholdfix` - base `DualEncoderVQA`, calibrated thresholds (0.8290 confirmed)

The base configuration: Qwen's vision tower fully frozen, text-side LoRA at
rank 16, trained on the long-form `description` data. Calibrated decision
thresholds (swept against a fair dev-lineage holdout, since the `all155`
checkpoint trains on every case and has no holdout of its own):

| question_type | threshold | accuracy (greedy -> calibrated) |
|---|---|---|
| `tool_presence` | 0.2 | 73.6% -> 74.3% |
| `suture_required` | 0.3 | 90.5% -> 91.0% |
| `tissue_cutting` | 1.4 | 85.4% -> 89.3% |

This is the only candidate with a confirmed real leaderboard score (0.8290,
prelim phase) - it is carried into the final phase as the insurance floor.

#### `combo` - vision-tower LoRA (r=8) + text LoRA (r=32)

Four isolated architecture changes were tried on top of `thresholdfix`
(dev-scale, `case000-139` train / `case140-154` test, gated by
`check_dual_encoder_v3_gate.py`), then the two that passed the gate were
combined:

| # | Change | Gate | Overall BERTScore delta |
|---|---|---|---|
| 1 | Qwen input resolution 112²->224² | FAIL 1/4 | +0.0003 |
| 2 | vision-tower LoRA r=8 | **PASS 4/4** | +0.0063 |
| 3 | text LoRA rank 16->32 | FAIL 3/4 | +0.0038 |
| 4 | `CrossModalFusion` cross-attention layer between the two visual streams | FAIL 3/4 | +0.0019 |
| **combo (2+3)** | **vision LoRA r=8 + text LoRA r=32** | **PASS 4/4** | **+0.0067** |

Unfreezing a small low-rank adapter on Qwen's own vision tower (previously fully
frozen) turned out to be the single biggest lever found in this project on the
dev-scale holdout - bigger than any data-curation change tried. `combo` also
trains on the short-form `description` data (see Shared data pipeline above),
making it the first release to test both changes together.

**Real-world caveat**: on the actual Grand Challenge prelim leaderboard, `combo`
scored 0.7653 - identical to a data-only variant (`descfix`, same architecture as
`thresholdfix`, only the short-form `description` data changed) that also scored
0.7653. Since an architecture change and a data-only change landed on the exact
same real score, the architecture change most likely added no measurable value
on the real test set, and the short-form `description` data is the more likely
regression driver versus `thresholdfix`'s confirmed 0.8290 - though per the
SurgVU Org Team's 2026-08-20 forum statement, the prelim test set is small enough
that ranking differences "will likely not mean anything," so this is treated as
suggestive evidence, not a settled conclusion.

### `plain` - architecture reset, no SurgMotion (informed bet, untested on real prelim)

Plain `Qwen2.5-VL-3B-Instruct` + a standard LoRA fine-tune (r=16, text-side
only), 5 frames per clip, trained on the long-form `description` data (like
`thresholdfix`, not `combo`/`descfix`'s short-form rewrite) - no SurgMotion
branch, no dual-encoder complexity, no vision-tower adaptation of any kind. This
is a deliberate simplification, not an incremental change, motivated by three
converging signals against architecture complexity:

1. **Historical precedent already in this codebase**: the `SurgMotionVQA` full
   vision-tower swap lost to a plain Qwen2-VL-2B LoRA fine-tune on the real
   hidden test set (0.7643 vs 0.7693) - the same finding that motivated
   `DualEncoderVQA`'s more conservative design in the first place.
2. **This project's own 2026-08-16/17 real-world result**: `combo` and
   `descfix` scored an identical 0.7653 on real prelim (see `combo` above),
   suggesting the added architecture complexity bought nothing measurable.
3. **External evidence**: the #1 leaderboard entry as of 2026-08-20 scores
   0.8523 using "QwenVL LoRA, 5-frame" - close to this exact shape.

Given local dev-scale holdout metrics have now twice failed to predict real
leaderboard direction, this release was **not** tuned or gated against any
local metric - it is built directly on historical + external precedent rather
than a locally-optimized candidate. A full BERTScore-F1 evaluation was still run
afterward as a coherence check (0.9562 overall, 1314-example true holdout
case147-154):

| category | BERTScore-F1 |
|---|---|
| `suture_required` | 0.9998 |
| `tool_presence` | 0.9881 |
| `tool_purpose` | 0.9869 |
| `procedure_type` | 0.9893 |
| `tissue_cutting` | 0.9854 |
| `forceps_type` | 0.9299 |
| `task_id` | 0.7163 |
| `organ` | 0.6468 |
| `description` | 0.4611 |

**Critical fix applied post-training**: the base model's own
`generation_config.json` ships `do_sample=true, temperature=0.000001` as a
"greedy-like" trick, but dividing by a near-zero temperature overflows to `inf`
under FP16 (this container's inference dtype), producing `nan` in the sampling
softmax and crashing generation mid-run (hit at 293/1314 examples during
evaluation). Fixed by passing `do_sample=False` explicitly in
`predict_plain_qwen25vl.py` (true greedy decoding, bypassing the
temperature/softmax path entirely) - this would otherwise have risked crashing
a real graded submission.

**Known limitations**: unlike every `DualEncoderVQA` release, `plain` has no
Yes/No threshold calibration - raw `generate()` output is used as-is, matching
the "plain LoRA" simplicity this release is testing. It was also not tested on
prelim before packaging (prelim budget was already exhausted when it was built).

## Earlier iteration history (v1-v5, superseded by the releases above)

The results below are from an earlier architecture (plain `Qwen2-VL-2B-Instruct` QLoRA,
16 frames, optional YOLOv5 tool-detector grounding) kept for historical reference; it
predates the `DualEncoderVQA`/SurgMotion and plain-`Qwen2.5-VL` lines used in the actual
final-phase candidates above.

## Results

Local scores are on the public sample set (`SURGVU25_cat_2_sample_set_public`, 11 Q&A
pairs) with the official metric (BERTScore-F1, `roberta-large`, rescaled with baseline;
max score over the 5 reference answers per question, mean over questions). These 11
cases were also used repeatedly for manual inspection during development, so they read
optimistic - the Grand Challenge Preliminary-phase score (unseen data) is the trustworthy
number where available.

| Model | Local (11-case) | Official Prelim phase |
|---|---|---|
| v1 - baseline fine-tune | 0.8007 | - |
| v2 - + per-tool Yes/No rebalancing | 0.8367 | - |
| v3 - + ProstaTDv2 domain pretrain (8 frames) | 0.8194 | - |
| v3 + YOLOv5 detector-grounded prompt | 0.8429 | - |
| **v5 - 16 frames + `commercial_toolname` fix** | 0.8781 | **0.7606** |
| v6 - short-canonical-answer training bias | 0.8173 (regression) | not submitted |
| **v7 - self-distillation-driven data fixes (see below)** | 0.8444 | submitted, pending |

`v5` was the first submitted model. `v6` tried biasing training toward the shortest
canonical answer template (since BERTScore-F1 rewards terse, reference-like phrasing) but
regressed - terse answers score *worse* than full sentences when the model gets the key
entity wrong, since a full sentence's correct grammatical scaffolding still earns partial
credit while a bare wrong noun doesn't. `v7` reverted to full-sentence answers and instead
targeted concrete biases found via self-distillation error analysis (see
[Methodology](#methodology)). See [Methodology](#methodology) for what each step changed
and why.

## Repository layout

```
src/
  preprocess.py               SurgVU raw videos + tools.csv/tasks.csv -> segment index
  generate_qa.py              segment index -> template-generated QA pairs
  extract_frames.py           pre-decodes/caches video frames referenced by the QA pairs
  train.py                    QLoRA fine-tuning (also used for the domain-pretrain stage)
  predict.py                  inference entrypoint (matches the Docker ENTRYPOINT)
  evaluate.py                 BERTScore-F1 (official) and BLEU-4 (legacy) scoring
  prepare_pretrain_qa.py      ProstaTDv2 -> domain-pretrain QA corpus
  prepare_tool_detection_data.py   cat1_test_set_public COCO boxes -> YOLO dataset
  train_tool_detector.py      fine-tunes YOLOv5 on the tool-detection dataset
  detect_tools.py             detector inference helper, used by predict.py for grounding
  error_analysis.py           self-distillation: model-vs-ground-truth audit by question type
  analyze_description_alignment.py   quantifies task/description label-vs-frame misalignment
Dockerfile, requirements.txt, .dockerignore    submission container definition
```

## Methodology

**Pipeline**: `preprocess.py` segments each training case's videos into 30s windows
(15s stride) and attaches ground-truth tool/task labels from `tools.csv`/`tasks.csv`.
`generate_qa.py` turns each segment into several template-generated question/answer
pairs (tool presence, tool purpose, task/organ identification, etc.), balanced so that
each tool's Yes/No ratio is close to 50/50 (an early version skewed ~80% Yes for common
tools like cadiere forceps, teaching the model a frequency prior instead of an actual
visual check). `extract_frames.py` pre-decodes the 16 frames each QA pair needs into a
JPEG cache, since decoding directly from the multi-hour source videos on every training
step was the dominant cost (~65s/step) and unstable under multi-process data loading on
Windows. `train.py` then does QLoRA fine-tuning (4-bit, LoRA r=16) of
`Qwen/Qwen2-VL-2B-Instruct` on this data.

**Domain-adaptive pretraining (Stage 0)**: SurgVU's own labels have no verb/target
annotations (only tool-presence timing), so a domain-adaptive pretraining stage was
added using **ProstaTDv2** (robotic prostatectomy, real per-frame
`<instrument, action, target>` triplet labels - closer procedurally to SurgVU than
laparoscopic alternatives like CholecT50/45, which were also explored but not used in
the final model due to instrument/procedure domain mismatch). `prepare_pretrain_qa.py`
turns ProstaTDv2's labels into a QA corpus in the same format, `train.py` trains a LoRA
adapter on it (Stage 0), and the SurgVU fine-tune (Stage 1) warm-starts from that
adapter's weights via `--init_lora_from` (a fresh optimizer/step count, as opposed to
`--resume_from_checkpoint` which resumes the *same* run/dataset after an interruption).

**Commercial tool-name granularity fix**: `tools.csv`'s `groundtruth_toolname` collapses
several distinct commercial variants (e.g. "Large Needle Driver", "Mega Needle Driver",
"Large SutureCut Needle Driver" all become "needle driver"). The organizers confirmed
the public sample's test questions probe this specific level of detail (e.g. "was a *large* needle
driver used"). `preprocess.py`/`generate_qa.py` now also track the `commercial_toolname`
column and, about half the time, ask about a specific named variant, checking presence
against that specific name rather than the generic category.

**Tool detector (optional grounding)**: `prepare_tool_detection_data.py` builds a YOLO
dataset from `cat1_test_set_public`'s COCO bounding-box annotations (the only source of
real box labels in this project) and `train_tool_detector.py` fine-tunes YOLOv5s on it
(mAP50 = 0.99 on held-out frames). `detect_tools.py` + `predict.py --detector_weights`
optionally prepend a "Detected tools: ..." context line to the prompt at inference time.
This measurably helped the 8-frame v3 model (0.8194 -> 0.8429) but did not improve the
final 16-frame v5 model on the public sample, so the submitted container runs **without**
the detector.

**Self-distillation error analysis (v7)**: rather than guessing what to fix next,
`error_analysis.py` uses the current best model (v5) as its own critic - it samples the
val set stratified by question type, judges each answer against ground truth with a
type-appropriate heuristic (Yes/No polarity match, or key-content-word overlap for
open-ended answers), and reports accuracy per type. This surfaced two concrete, fixable
biases: (1) `tool_presence` false-positives specifically on rare/named commercial
variants (e.g. "was a maryland bipolar forceps used?" answered Yes because *some*
bipolar forceps is almost always present in frame - `generate_qa.py`'s balancing only
tracked the generic tool key, which averaged away this variant-specific skew); (2)
`task_id`/`description` mode-collapse toward the single most common label ("Other" /
"Activity outside of the structured training.", ~73-74% of examples) - the model could
get most of these right by frequency alone, without looking at the clip. `v7` fixes both:
`gen_tool_presence` now tracks a separate balance key per commercial variant (not just the
generic tool), and `downsample_dominant_answers` caps the dominant label's share per
question type (74% -> 50%) instead of leaving it dominant.

**Task-boundary label misalignment fix**: `preprocess.py` assigns each 30s window the
task with maximum time-overlap - but "maximum" can still be a sliver (e.g. 0.02% of the
window) when a window straddles two tasks, mislabeling frames that mostly show something
else entirely. `analyze_description_alignment.py` quantified this (4.6% of task-labeled
segments had <50% real overlap) and traced the worst cases to exact task-transition
boundaries. Fixed with a minimum-overlap-fraction threshold (50%) before crediting a task
label, falling back to "Other" otherwise.

**Known open issue - forceps-type confusion**: `error_analysis.py` also found the model
frequently names the wrong forceps type (Cadiere/Bipolar/Prograsp are visually similar
grasper-style tools). Manual frame inspection confirmed this is a genuine visual
discrimination limit, not a data or label bug, and confirmed that detector-grounded
prompting doesn't reliably fix it either (tested: 0.8702 vs 0.8781 without grounding, on
the 11-case sample) - noted here as a limitation rather than invested in further.

**Evaluation metric**: `evaluate.py` implements the official BERTScore-F1 metric
(`bert-score` package, `roberta-large`, `rescale_with_baseline=True`, max score over the
5 references) as well as the original BLEU-4 metric used before the 2026-07-23 change,
kept for historical comparison.

## Data used beyond what the challenge provided

- **ProstaTDv2** (public dataset, robotic prostatectomy tool/action/target
  annotations) - used only for the Stage 0 domain-adaptive pretraining corpus, not
  mixed into SurgVU ground truth.
- `cat1_test_set_public` (provided by the challenge, Category 1 public sample) - used
  as the only source of real bounding-box labels, to train the optional tool detector.

## Reproducing

```bash
pip install -r requirements.txt

# 1. Build the segment index + QA pairs from the SurgVU training data
python src/preprocess.py
python src/generate_qa.py

# 2. Pre-cache the frames each QA pair needs (one-time, avoids decoding video during training)
python src/extract_frames.py

# 3. Domain-adaptive pretrain on ProstaTDv2 (see prepare_pretrain_qa.py for corpus prep)
python src/prepare_pretrain_qa.py
python src/train.py --train_json src/pretrain_train_vqa.json --val_json src/pretrain_val_vqa.json \
    --output_dir checkpoints/pretrain_lora_prostatd --bleu_eval_samples 0

# 4. Fine-tune on SurgVU, warm-started from the pretrain adapter
python src/train.py --init_lora_from checkpoints/pretrain_lora_prostatd \
    --output_dir checkpoints/qwen2_vl_2b_lora_v5 --eval_batch_size 2

# 5. (optional) Self-distillation audit, then a second fine-tune pass warm-started
#    from v5 with the data fixes it surfaced (see Methodology)
python src/error_analysis.py
python src/train.py --init_lora_from checkpoints/qwen2_vl_2b_lora_v5 \
    --output_dir checkpoints/qwen2_vl_2b_lora_v7 --eval_batch_size 2

# 6. Build the submission image (expects checkpoints/qwen2_vl_2b_lora_v7/ to exist)
docker build -t surgvu-submission .
```

`src/train.py --help` documents the remaining flags (frame count, eval/save cadence,
LoRA hyperparameters). Random seeds are fixed in `generate_qa.py`/`prepare_pretrain_qa.py`
for reproducible train/val splits.

## Acknowledgements

This project builds on the following external models, datasets, and code, in
addition to the SurgVU data and challenge infrastructure provided by the organizers:

- **[SurgMotion](https://github.com/CAIR-HKISI/SurgMotion)** (CAIR-HKISI, Apache-2.0) -
  video-native surgical foundation model (ViT-Large variant), pretrained on
  SurgMotion-15M, used as the motion-aware visual encoder in the `DualEncoderVQA`
  final-phase candidates (`thresholdfix`, `combo`). [Paper](https://arxiv.org/abs/2602.05638) ·
  [weights](https://huggingface.co/CAIR-HKISI/SurgMotion).
- **[Qwen2.5-VL-3B-Instruct](https://github.com/QwenLM/Qwen2.5-VL)** (Alibaba Qwen team) -
  base vision-language model, LoRA fine-tuned, used in the `plain` final-phase
  candidate and as the frozen/LoRA-adapted vision-language backbone in the
  `DualEncoderVQA` line. [Paper](https://arxiv.org/abs/2409.12191).
- **[Qwen2-VL-2B-Instruct](https://github.com/QwenLM/Qwen2-VL)** (Alibaba Qwen team) -
  base vision-language model, QLoRA fine-tuned, used in the earlier v1-v7 iteration
  line (see [Earlier iteration history](#earlier-iteration-history-v1-v5-superseded-by-the-releases-above)).
- **[ProstaTDv2](https://arxiv.org/abs/2506.01130)** (robotic prostatectomy
  instrument/action/target triplet annotations) - used for domain-adaptive
  pretraining, in both the v1-v7 line and (re-tried, not adopted) the current
  `DualEncoderVQA` line.
- **[CholecT50](https://github.com/CAMMA-public/cholect50)** (CAMMA, CC BY-NC-SA 4.0) -
  laparoscopic instrument/action/target triplet dataset, evaluated as a domain-pretrain
  alternative to ProstaTDv2 but not used in any released model (instrument/procedure
  domain mismatch with SurgVU's robotic setting).
- **[YOLOv5](https://github.com/ultralytics/yolov5)** (Ultralytics, AGPL-3.0) - object
  detector fine-tuned as the optional tool-detector used for grounding in the
  earlier v3 iteration; not part of any final-phase candidate.
- **[Hugging Face `transformers`](https://github.com/huggingface/transformers)** and
  **[`peft`](https://github.com/huggingface/peft)** - model loading, LoRA/QLoRA
  fine-tuning.
- **[`bert-score`](https://github.com/Tiiiger/bert_score)** - the official BERTScore-F1
  evaluation metric implementation used throughout this project.
