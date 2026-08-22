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
