# SurgVU 2026 - Category 2 (Surgical Visual Question Answering)

Team: SCLab-Surg (Chonnam National University) - Grand Challenge username `TienNQ27`

## Final-phase results (2026-09-03)

| Slot | Release | Final score |
|---|---|---|
| 1 | [`release_20260806_thresholdfix`](submission/release_20260806_thresholdfix/README_UPLOAD.md) | 0.5526 |
| 2 | [`release_20260831_thresholdfix_qfix`](submission/release_20260831_thresholdfix_qfix/README_UPLOAD.md) | **0.5972** (best, official score) |
| 3 | [`release_20260902_final`](submission/release_20260902_final/README_UPLOAD.md) | 0.5820 |

All three slots use the identical `DualEncoderVQA` checkpoint and thresholds - only the
container's answer formatting/routing code changed between them. Slot 2's +0.0446 over
slot 1 came from fixing 3 of 11 official question types that were misrouted or truncated
(full writeup in its release doc, linked above); slot 3 bundled four further fixes into
one slot and lost 0.0152 despite every one measuring positive locally - the lesson being
that a scarce, non-repeatable slot should carry exactly one change at a time.

## Architecture: `DualEncoderVQA`

Qwen2.5-VL-3B-Instruct's vision tower stays fully frozen; a separate frozen
[SurgMotion](https://github.com/CAIR-HKISI/SurgMotion) encoder (16-frame clips) adds a
32-token auxiliary signal through a trained projector. Only the projector and a rank-16
LoRA adapter on the text decoder are trained. Yes/No questions use a calibrated
logit-margin threshold instead of free-form generation. Two other architectures were
evaluated but never submitted (a LoRA'd vision tower + rank-32 text LoRA combo, and a
plain Qwen2.5-VL with no SurgMotion branch) - see the release docs above for why.

## Running / reproducing (best model, slot 2)

1. Download the checkpoint (~185 MB) from Hugging Face -
   <https://huggingface.co/Partrick86/surgvu26-sclab-slot2-checkpoint>:
   ```bash
   huggingface-cli download Partrick86/surgvu26-sclab-slot2-checkpoint \
     surgvu26_sclab_slot2_dual_encoder_checkpoint.tar.gz --local-dir /tmp
   echo "19dc1ac00eb3f553118494dd282f57e686c8370008eb683a99d6dadf52d421b8  /tmp/surgvu26_sclab_slot2_dual_encoder_checkpoint.tar.gz" | sha256sum -c -
   mkdir -p checkpoints/dual_encoder_v3_stage_b_all155
   tar -xzf /tmp/surgvu26_sclab_slot2_dual_encoder_checkpoint.tar.gz -C checkpoints/dual_encoder_v3_stage_b_all155
   ```
2. Download the two base models it sits on top of: `Qwen/Qwen2.5-VL-3B-Instruct`
   (Hugging Face) and `SurgMotion-vitl.pt` (`CAIR-HKISI/SurgMotion`).
3. Build and run - see [`submission/README.md`](submission/README.md) for exact commands.

## Acknowledgements

Built on the following external models/datasets, in addition to the SurgVU data and
challenge infrastructure:

- **[SurgMotion](https://github.com/CAIR-HKISI/SurgMotion)** (CAIR-HKISI, Apache-2.0) -
  motion-aware visual encoder. [Paper](https://arxiv.org/abs/2602.05638) ·
  [weights](https://huggingface.co/CAIR-HKISI/SurgMotion).
- **[Qwen2.5-VL-3B-Instruct](https://github.com/QwenLM/Qwen2.5-VL)** (Alibaba Qwen team) -
  base vision-language model, LoRA fine-tuned. [Paper](https://arxiv.org/abs/2409.12191).
- **[ProstaTDv2](https://arxiv.org/abs/2506.01130)** - tried for domain-adaptive
  pretraining, not adopted (underperformed on gate check).
- **[Hugging Face `transformers`](https://github.com/huggingface/transformers)** and
  **[`peft`](https://github.com/huggingface/peft)** - model loading, LoRA fine-tuning.
- **[`bert-score`](https://github.com/Tiiiger/bert_score)** - official evaluation metric.
