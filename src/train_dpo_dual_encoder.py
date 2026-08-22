"""Manual DPO fine-tune for DualEncoderVQA's LoRA decoder, correcting the
tool_presence/tissue_cutting/suture_required shortcut bias found by
audit_dual_encoder_visual_ablation.py (see build_dpo_pairs_from_ablation.py
for how chosen/rejected pairs are derived).

Not TRL's DPOTrainer: DualEncoderVQA's forward() takes prompt_ids +
pixel_values_videos + video_grid_thw + aux_video + aux_frame_valid_mask, not
TRL's expected single-input-ids signature, and inputs_embeds are built by
splicing 3 segments (Qwen prompt, aux tokens, answer) with custom mRoPE
position ids - there is nowhere to hook that shape into DPOTrainer without
reimplementing this file's forward() around it anyway. Loss is standard DPO
(Rafailov et al.): reference logprobs come from the SAME model with LoRA
disabled via PEFT's disable_adapter() context, which needs no second copy of
the ~3B-parameter base model in memory - the frozen base weights the adapter
sits on top of already function as the reference model with zero extra
memory cost.

Only the LoRA decoder is trained here (aux_projector/surgmotion_encoder
frozen) - this phase targets a specific decoder-level answer-preference bug,
not the visual-grounding gains Stage A/B already produced, so those
components are left untouched to avoid regressing them.
"""
import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from qwen_vl_utils import process_vision_info
from safetensors.torch import save_file
from transformers import AutoProcessor, AutoTokenizer

from dual_encoder_vqa_prototype import AUX_NUM_TOKENS, DEFAULT_QWEN_MODEL, build
from surgmotion_vqa_prototype import DEFAULT_SURGMOTION_CHECKPOINT
from train_dual_encoder_vqa import SURGMOTION_NUM_FRAMES, DualEncoderQADataset, load_component_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def build_answer_ids(processor, tokenizer, qwen_frames, question, answer_text, max_text_length):
    """Mirrors DualEncoderCollator's prompt/full split - identical video content
    through the same processor() call so the <video> placeholder expansion
    matches and the prompt is guaranteed to be a real prefix."""
    user_turn = [
        {"role": "user", "content": [{"type": "video", "video": qwen_frames}, {"type": "text", "text": question}]}
    ]
    full_turns = user_turn + [{"role": "assistant", "content": answer_text}]
    prompt_text = processor.apply_chat_template(user_turn, tokenize=False, add_generation_prompt=True)
    full_text = processor.apply_chat_template(full_turns, tokenize=False, add_generation_prompt=False)
    image_inputs, video_inputs = process_vision_info(user_turn)
    prompt_inputs = processor(
        text=[prompt_text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    )
    full_inputs = processor(
        text=[full_text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    )
    prompt_ids_list = prompt_inputs["input_ids"][0].tolist()
    full_ids_list = full_inputs["input_ids"][0].tolist()
    if full_ids_list[: len(prompt_ids_list)] != prompt_ids_list:
        raise RuntimeError("Qwen chat prompt is not a prefix of the DPO-target text")
    answer_ids_list = full_ids_list[len(prompt_ids_list):]
    if not answer_ids_list:
        raise RuntimeError("No assistant tokens remain after prefix split")
    if len(full_ids_list) > max_text_length:
        raise ValueError(f"DPO example has {len(full_ids_list)} tokens, exceeding --max-text-length={max_text_length}")
    return prompt_inputs, torch.tensor([answer_ids_list], dtype=torch.long)


def answer_logp(model, prompt_inputs, aux_video, aux_valid, answer_ids, device):
    """Sum of per-token log-probs of answer_ids under teacher forcing (labels
    supervise only the answer span, exactly as DualEncoderVQA.forward() already
    does for its cross-entropy loss - reused here as log-prob instead)."""
    output = model(
        prompt_ids=prompt_inputs["input_ids"].to(device),
        pixel_values_videos=prompt_inputs["pixel_values_videos"].to(device),
        video_grid_thw=prompt_inputs["video_grid_thw"].to(device),
        aux_video=aux_video.to(device),
        aux_frame_valid_mask=aux_valid.to(device),
        second_per_grid_ts=prompt_inputs.get("second_per_grid_ts"),
        answer_ids=answer_ids.to(device),
    )
    logits = output["logits"]
    prefix_len = logits.shape[1] - answer_ids.shape[1]
    shift_logits = logits[:, prefix_len - 1 : -1, :].float()
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_logp = log_probs.gather(-1, answer_ids.to(device).unsqueeze(-1)).squeeze(-1)
    return token_logp.sum(dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-checkpoint", required=True, help="DualEncoderVQA checkpoint to warm-start from (e.g. checkpoints/dual_encoder_stage_b_dev)")
    parser.add_argument("--dpo-pairs", required=True, help="Output of build_dpo_pairs_from_ablation.py")
    parser.add_argument("--vqa-json", required=True, help="Source vqa json the DPO pairs' ids were drawn from (for video/id lookup)")
    parser.add_argument("--cache-root", default=str(PROJECT_ROOT / "frame_cache_16fr"))
    parser.add_argument("--qwen-model", default=str(DEFAULT_QWEN_MODEL))
    parser.add_argument("--surgmotion-checkpoint", default=str(DEFAULT_SURGMOTION_CHECKPOINT))
    parser.add_argument("--qwen-num-frames", type=int, default=8)
    parser.add_argument("--qwen-max-pixels", type=int, default=112 * 112)
    parser.add_argument("--aux-num-tokens", type=int, default=AUX_NUM_TOKENS)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature")
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--grad-acc", type=int, default=8)
    parser.add_argument("--max-text-length", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DPO training")
    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained(args.qwen_model, local_files_only=True)
    tokenizer.padding_side = "right"
    processor = AutoProcessor.from_pretrained(args.qwen_model, local_files_only=True)
    processor.image_processor.min_pixels = args.qwen_max_pixels
    processor.image_processor.max_pixels = args.qwen_max_pixels

    model = build(
        surgmotion_checkpoint=args.surgmotion_checkpoint,
        qwen_model_id=args.qwen_model,
        aux_num_frames=SURGMOTION_NUM_FRAMES,
        aux_num_tokens=args.aux_num_tokens,
        lora_r=args.lora_r,
        dtype=torch.bfloat16,
        local_files_only=True,
        train_surgmotion_blocks=0,
    )
    load_component_checkpoint(model, args.initial_checkpoint)
    model.to(device)

    # Only the LoRA decoder moves in this phase - freeze everything else so the
    # already-validated visual-grounding gains from Stage A/B cannot regress.
    model.aux_projector.requires_grad_(False)
    model.surgmotion_encoder.requires_grad_(False)
    model.qwen_model.visual.requires_grad_(False)
    lora_params = [p for p in model.text_model.parameters() if p.requires_grad]
    if not lora_params:
        raise RuntimeError("No trainable LoRA parameters found on text_model")
    print(f"[dpo] trainable LoRA parameters: {sum(p.numel() for p in lora_params):,}")

    optimizer = torch.optim.AdamW(lora_params, lr=args.learning_rate)

    pairs = load_json(args.dpo_pairs)
    if not pairs:
        raise RuntimeError(f"No DPO pairs in {args.dpo_pairs}")
    vqa_items = load_json(args.vqa_json)
    index_by_id = {item["id"]: index for index, item in enumerate(vqa_items)}
    dataset = DualEncoderQADataset(
        args.vqa_json, args.cache_root, is_train=False, qwen_num_frames=args.qwen_num_frames
    )

    missing = [pair["id"] for pair in pairs if pair["id"] not in index_by_id]
    if missing:
        raise RuntimeError(f"{len(missing)} DPO pair ids not found in --vqa-json, e.g. {missing[:5]}")

    steps_per_epoch = len(pairs)
    total_steps = int(steps_per_epoch * args.epochs)
    print(f"[dpo] {len(pairs)} pairs, {args.epochs} epoch(s) -> {total_steps} examples, "
          f"grad_acc={args.grad_acc} ({total_steps // args.grad_acc} optimizer steps)")

    model.train()
    optimizer.zero_grad()
    running_loss = 0.0
    running_margin = 0.0
    accum_count = 0
    example_count = 0

    order = list(range(len(pairs)))
    epoch_int = int(args.epochs)
    fractional = args.epochs - epoch_int
    schedule = []
    for _ in range(max(epoch_int, 1) if epoch_int > 0 else 0):
        shuffled = order[:]
        random.shuffle(shuffled)
        schedule.extend(shuffled)
    if fractional > 0:
        shuffled = order[:]
        random.shuffle(shuffled)
        schedule.extend(shuffled[: int(len(order) * fractional)])
    if epoch_int == 0 and fractional == 0:
        schedule = order[:]

    for step, pair_index in enumerate(schedule):
        pair = pairs[pair_index]
        index = index_by_id[pair["id"]]
        sample = dataset[index]
        qwen_frames = sample["qwen_frames"][: args.qwen_num_frames]
        aux_video = sample["aux_video"].unsqueeze(0)
        aux_valid = sample["aux_frame_valid_mask"].unsqueeze(0)

        chosen_prompt, chosen_ids = build_answer_ids(
            processor, tokenizer, qwen_frames, pair["question"], pair["chosen"], args.max_text_length
        )
        rejected_prompt, rejected_ids = build_answer_ids(
            processor, tokenizer, qwen_frames, pair["question"], pair["rejected"], args.max_text_length
        )

        policy_chosen_logp = answer_logp(model, chosen_prompt, aux_video, aux_valid, chosen_ids, device)
        policy_rejected_logp = answer_logp(model, rejected_prompt, aux_video, aux_valid, rejected_ids, device)

        with torch.no_grad(), model.text_model.disable_adapter():
            ref_chosen_logp = answer_logp(model, chosen_prompt, aux_video, aux_valid, chosen_ids, device)
            ref_rejected_logp = answer_logp(model, rejected_prompt, aux_video, aux_valid, rejected_ids, device)

        policy_margin = policy_chosen_logp - policy_rejected_logp
        ref_margin = ref_chosen_logp - ref_rejected_logp
        logits = args.beta * (policy_margin - ref_margin)
        loss = -F.logsigmoid(logits).mean() / args.grad_acc
        loss.backward()

        running_loss += loss.item() * args.grad_acc
        running_margin += logits.item()
        accum_count += 1
        example_count += 1

        if accum_count == args.grad_acc or step == len(schedule) - 1:
            torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            accum_count = 0

        if (step + 1) % args.log_every == 0:
            print(
                f"[dpo] step {step + 1}/{len(schedule)} "
                f"loss={running_loss / args.log_every:.4f} "
                f"implicit_reward_margin={running_margin / args.log_every:.4f}"
            )
            running_loss = 0.0
            running_margin = 0.0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.text_model.save_pretrained(output_dir / "lora_adapter", safe_serialization=True)
    projector_state = {
        name: value.detach().cpu().contiguous() for name, value in model.aux_projector.state_dict().items()
    }
    save_file(projector_state, output_dir / "aux_projector.safetensors")

    # surgmotion_encoder is frozen for this DPO phase, but --initial-checkpoint may
    # already carry a Stage-A delta (e.g. dual_encoder_stage_b_dev's unfrozen last
    # block) via load_component_checkpoint's surgmotion_delta_keys bookkeeping -
    # that must be carried forward here too, or reloading this checkpoint standalone
    # would silently fall back to the released (non-fine-tuned) SurgMotion weights.
    if model.surgmotion_delta_keys:
        encoder_state = model.surgmotion_encoder.state_dict()
        save_file(
            {name: encoder_state[name].detach().cpu().contiguous() for name in sorted(model.surgmotion_delta_keys)},
            output_dir / "surgmotion_delta.safetensors",
        )
        print(f"[dpo] carried forward SurgMotion delta ({len(model.surgmotion_delta_keys)} tensors) from --initial-checkpoint")

    tokenizer.save_pretrained(output_dir / "tokenizer")
    metadata = {
        "architecture": "DualEncoderVQA",
        "aux_num_tokens": args.aux_num_tokens,
        "surgmotion_num_frames": SURGMOTION_NUM_FRAMES,
        "qwen_num_frames": args.qwen_num_frames,
        "qwen_max_pixels": args.qwen_max_pixels,
        "qwen_model": str(args.qwen_model),
        "surgmotion_checkpoint": str(args.surgmotion_checkpoint),
        "dpo_source_checkpoint": str(args.initial_checkpoint),
        "dpo_pairs_file": str(args.dpo_pairs),
        "dpo_beta": args.beta,
        "dpo_examples_trained": example_count,
    }
    json.dump(metadata, open(output_dir / "dual_encoder_vqa_config.json", "w", encoding="utf-8"), indent=2)
    print(f"[dpo] wrote checkpoint to {output_dir}")


if __name__ == "__main__":
    main()
