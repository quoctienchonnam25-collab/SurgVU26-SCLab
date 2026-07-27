# SurgVU 2026 - Category 2 (Surgical Visual Question Answering)

Team: SCLab-Surg (Chonnam National University) - Grand Challenge username `TienNQ27`

Fine-tuned Qwen2-VL-2B-Instruct (QLoRA) for surgical video VQA, with a domain-adaptive
pretraining stage and an optional YOLOv5 tool-detector used to ground the VQA prompt.

## Results on the public sample set (`SURGVU25_cat_2_sample_set_public`, 11 Q&A pairs)

Scored with the official metric (BERTScore-F1, `roberta-large`, rescaled with baseline;
max score over the 5 reference answers per question, mean over questions):

| Model | Mean BERTScore-F1 |
|---|---|
| v1 - baseline fine-tune | 0.8007 |
| v2 - + per-tool Yes/No rebalancing | 0.8367 |
| v3 - + ProstaTDv2 domain pretrain (8 frames) | 0.8194 |
| v3 + YOLOv5 detector-grounded prompt | 0.8429 |
| **v5 - 16 frames + `commercial_toolname` fix (submitted)** | **0.8781** |

`v5` is the submitted model. See [Methodology](#methodology) for what each step changed
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

# 5. Build the submission image (expects checkpoints/qwen2_vl_2b_lora_v5/ to exist)
docker build -t surgvu-submission .
```

`src/train.py --help` documents the remaining flags (frame count, eval/save cadence,
LoRA hyperparameters). Random seeds are fixed in `generate_qa.py`/`prepare_pretrain_qa.py`
for reproducible train/val splits.
