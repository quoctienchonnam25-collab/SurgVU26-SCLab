"""Dual-encoder VQA prototype: Qwen2.5-VL-3B's OWN native vision tower (frozen,
full multimodal-RoPE pipeline intact, per-example dynamic resolution - exactly as
designed) handles general scene understanding. A small, frozen SurgMotion
(surgical-domain motion encoder) branch is added as a short auxiliary token
sequence, inserted between the user's prompt and the assistant's answer, giving
the LoRA-adapted decoder an extra surgical-motion-specific signal without
touching or replacing Qwen's own vision path at all.

Rationale: the real-world audit (runs/surgmotion_vqa/audit_dev_holdout, 2026-08)
found the wholesale vision-tower swap in surgmotion_vqa_prototype.py's
SurgMotionVQA trailed a plain Qwen2-VL-2B LoRA fine-tune (v7) on the actual
Grand Challenge hidden test set (0.7643 ensemble vs 0.7693 v7 alone), despite
winning on this project's own held-out split - i.e. the swap traded away
Qwen's broad, robustly-pretrained visual grounding for a narrowly surgical-only
encoder that is only shallowly adapted (Stage A unfreezes one block for a
quarter epoch), and that trade did not pay off out-of-distribution. The one
place SurgMotion showed real, causally-verified benefit (see
audit_surgmotion_visual_ablation.py's real-vs-blank ablation) was forceps_type
discrimination. This architecture is a bet that keeping Qwen's native path
fully intact and adding SurgMotion only as a small supplementary signal keeps
the broad robustness while still making that forceps-type-style signal
available to the decoder.

STATUS: prototype, structurally smoke-tested but not yet trained end to end -
see train_dual_encoder_vqa.py.
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from surgmotion_vqa_prototype import (
    DEFAULT_SURGMOTION_CHECKPOINT,
    VisualProjector,
    load_surgmotion_encoder,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_QWEN_MODEL = os.path.join(
    PROJECT_ROOT, "checkpoints", "base_models", "Qwen2.5-VL-3B-Instruct"
)
AUX_NUM_TOKENS = 32  # divides 2048 (SurgMotion's 16-frame token count) exactly, staying on the fast reshape+mean pooling path in VisualProjector


class CrossModalFusion(nn.Module):
    """Experiment: lets the SurgMotion aux tokens cross-attend to Qwen's own
    native visual tokens before being appended to the prompt sequence, instead
    of leaving the two streams to interact purely through the LLM's own
    (LoRA-adapted) self-attention post-concatenation. Residual + LayerNorm
    around a standard multi-head cross-attention block; aux tokens are the
    query, Qwen's visual tokens are key/value."""

    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, aux_embeds, video_embeds):
        # aux_embeds: [1, num_aux_tokens, hidden]; video_embeds: [num_video_tokens, hidden]
        video_embeds = video_embeds.unsqueeze(0).to(dtype=aux_embeds.dtype)
        attended, _ = self.cross_attn(aux_embeds, video_embeds, video_embeds)
        return self.norm(aux_embeds + attended)


class DualEncoderVQA(nn.Module):
    """Qwen2.5-VL's own frozen vision tower (native mRoPE, untouched) + a frozen
    SurgMotion branch projected to a handful of auxiliary tokens, inserted right
    before the assistant's answer. Only the auxiliary projector and the
    LoRA-adapted decoder are trainable.

    Batch size is always 1 per forward call - matches every other training
    script in this project (grad accumulation instead of batching) and avoids
    needing any padding logic across examples of different length, since the
    prompt/aux/answer segments are built and concatenated per-example.
    """

    def __init__(
        self, qwen_model, surgmotion_encoder, aux_projector, text_model, lm_head,
        tubelet_size=2, fusion=None,
    ):
        super().__init__()
        self.qwen_model = qwen_model  # Qwen2_5_VLModel: .visual (frozen), get_video_features,
        # get_placeholder_mask, get_rope_index, get_input_embeddings - all reused as-is.
        self.surgmotion_encoder = surgmotion_encoder
        self.aux_projector = aux_projector
        self.text_model = text_model  # LoRA-wrapped qwen_model.language_model
        self.lm_head = lm_head
        self.tubelet_size = tubelet_size
        # None (default) preserves the original design exactly: aux tokens are
        # appended to the prompt sequence untouched, and the LLM's own
        # self-attention (through LoRA) is solely responsible for relating them
        # to Qwen's native visual tokens. When set, CrossModalFusion lets the aux
        # tokens attend to Qwen's visual tokens first - see build()'s docstring.
        self.fusion = fusion
        self.surgmotion_delta_keys = {
            name for name, parameter in surgmotion_encoder.named_parameters()
            if parameter.requires_grad
        }

    def train(self, mode=True):
        super().train(mode)
        # Keep both frozen encoders out of train mode so dropout/stochastic depth
        # cannot perturb their cached features, mirroring SurgMotionVQA.
        self.surgmotion_encoder.eval()
        self.qwen_model.visual.eval()
        return self

    def _aux_token_valid_mask(self, frame_valid_mask, num_video_tokens):
        batch_size, num_frames = frame_valid_mask.shape
        if num_frames % self.tubelet_size:
            raise ValueError(
                f"num_frames={num_frames} is not divisible by tubelet_size={self.tubelet_size}"
            )
        tubelet_mask = frame_valid_mask.reshape(
            batch_size, num_frames // self.tubelet_size, self.tubelet_size
        ).any(dim=-1)
        if num_video_tokens % tubelet_mask.shape[1]:
            raise ValueError(
                f"num_video_tokens={num_video_tokens} cannot be split across "
                f"{tubelet_mask.shape[1]} temporal tubelets"
            )
        spatial_tokens = num_video_tokens // tubelet_mask.shape[1]
        return tubelet_mask.repeat_interleave(spatial_tokens, dim=1)

    def encode_auxiliary(self, aux_video, frame_valid_mask=None):
        batch_size = aux_video.shape[0]
        if frame_valid_mask is None:
            frame_valid_mask = torch.ones(
                batch_size, aux_video.shape[2], dtype=torch.bool, device=aux_video.device
            )
        else:
            frame_valid_mask = frame_valid_mask.to(device=aux_video.device, dtype=torch.bool)
        has_visual = frame_valid_mask.any(dim=1)
        aux_embeds = self.aux_projector.no_visual(batch_size)
        if not has_visual.any():
            return aux_embeds

        encoder_trainable = any(
            parameter.requires_grad for parameter in self.surgmotion_encoder.parameters()
        )
        with torch.set_grad_enabled(encoder_trainable and self.training):
            video_tokens = self.surgmotion_encoder(aux_video[has_visual])
        token_valid_mask = self._aux_token_valid_mask(
            frame_valid_mask[has_visual], video_tokens.shape[1]
        )
        projected = self.aux_projector(video_tokens, token_valid_mask=token_valid_mask)
        aux_embeds = aux_embeds.to(dtype=projected.dtype).clone()
        aux_embeds[has_visual] = projected
        return aux_embeds

    def _qwen_prompt_embeds_and_positions(
        self, prompt_ids, pixel_values_videos, video_grid_thw, second_per_grid_ts=None
    ):
        """Reuses Qwen2.5-VL's own get_video_features/get_placeholder_mask/get_rope_index
        exactly as Qwen2_5_VLModel.forward() does internally - the only difference is we
        stop short of calling the language model, so we can splice in the auxiliary
        tokens first."""
        inputs_embeds = self.qwen_model.get_input_embeddings()(prompt_ids)
        video_embeds = self.qwen_model.get_video_features(pixel_values_videos, video_grid_thw)
        video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        _, video_mask = self.qwen_model.get_placeholder_mask(
            prompt_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
        attention_mask = torch.ones_like(prompt_ids)
        position_ids, _ = self.qwen_model.get_rope_index(
            prompt_ids,
            image_grid_thw=None,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            attention_mask=attention_mask,
        )
        return inputs_embeds, position_ids, video_embeds

    @staticmethod
    def _append_positions(base_position_ids, count, device):
        """Continue Qwen's own mRoPE convention for trailing plain-text tokens:
        `torch.arange(text_len) + (previous_block_max + 1)`, broadcast identically
        across all 3 rope dimensions (see Qwen2_5_VLModel.get_rope_index's own
        docstring example - this is exactly the rule it uses for text following a
        vision block)."""
        start = base_position_ids.max() + 1
        offsets = torch.arange(count, device=device) + start
        return offsets.view(1, 1, -1).expand(3, base_position_ids.shape[1], -1)

    def _assemble_prompt_and_aux(
        self, prompt_ids, pixel_values_videos, video_grid_thw, aux_video,
        aux_frame_valid_mask, second_per_grid_ts,
    ):
        """Shared by forward() and generate() - builds the [prompt + aux] embeds/
        position_ids prefix (everything before the answer). Kept as one method so
        the two call sites cannot silently drift apart."""
        prompt_embeds, prompt_positions, video_embeds = self._qwen_prompt_embeds_and_positions(
            prompt_ids, pixel_values_videos, video_grid_thw, second_per_grid_ts
        )
        aux_embeds = self.encode_auxiliary(aux_video, aux_frame_valid_mask).to(dtype=prompt_embeds.dtype)
        if self.fusion is not None:
            aux_embeds = self.fusion(aux_embeds, video_embeds)
        aux_positions = self._append_positions(prompt_positions, aux_embeds.shape[1], prompt_embeds.device)
        inputs_embeds = torch.cat([prompt_embeds, aux_embeds], dim=1)
        position_ids = torch.cat([prompt_positions, aux_positions], dim=-1)
        return inputs_embeds, position_ids

    def forward(
        self,
        prompt_ids,
        pixel_values_videos,
        video_grid_thw,
        aux_video,
        aux_frame_valid_mask=None,
        second_per_grid_ts=None,
        answer_ids=None,
        labels=None,
        sample_weight=None,
    ):
        if prompt_ids.shape[0] != 1:
            raise ValueError("DualEncoderVQA only supports batch size 1 per forward call")

        inputs_embeds, position_ids = self._assemble_prompt_and_aux(
            prompt_ids, pixel_values_videos, video_grid_thw, aux_video,
            aux_frame_valid_mask, second_per_grid_ts,
        )

        if answer_ids is not None:
            answer_embeds = self.qwen_model.get_input_embeddings()(answer_ids).to(dtype=inputs_embeds.dtype)
            answer_positions = self._append_positions(position_ids, answer_embeds.shape[1], inputs_embeds.device)
            inputs_embeds = torch.cat([inputs_embeds, answer_embeds], dim=1)
            position_ids = torch.cat([position_ids, answer_positions], dim=-1)

        attention_mask = torch.ones(
            inputs_embeds.shape[0], inputs_embeds.shape[1], dtype=torch.long, device=inputs_embeds.device
        )

        hidden_states = self.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        ).last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            prefix_len = inputs_embeds.shape[1] - labels.shape[1]
            full_labels = torch.cat(
                [
                    torch.full((1, prefix_len), -100, dtype=labels.dtype, device=labels.device),
                    labels,
                ],
                dim=1,
            )
            shift_logits = logits[:, :-1, :]
            shift_labels = full_labels[:, 1:]
            token_loss = F.cross_entropy(
                shift_logits.float().transpose(1, 2),
                shift_labels,
                ignore_index=-100,
                reduction="none",
            )
            supervised = shift_labels.ne(-100)
            per_sample_loss = (token_loss * supervised).sum(dim=1) / supervised.sum(dim=1).clamp_min(1)
            weight = sample_weight if sample_weight is not None else torch.ones_like(per_sample_loss)
            loss = (per_sample_loss * weight).sum() / weight.sum().clamp_min(1e-6)

        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def generate(
        self,
        prompt_ids,
        pixel_values_videos,
        video_grid_thw,
        aux_video,
        aux_frame_valid_mask=None,
        second_per_grid_ts=None,
        max_new_tokens=32,
        eos_token_id=None,
    ):
        self.eval()
        inputs_embeds, position_ids = self._assemble_prompt_and_aux(
            prompt_ids, pixel_values_videos, video_grid_thw, aux_video,
            aux_frame_valid_mask, second_per_grid_ts,
        )

        eos_ids = set()
        if eos_token_id is not None:
            eos_ids = {eos_token_id} if isinstance(eos_token_id, int) else set(eos_token_id)

        generated = []
        for _ in range(max_new_tokens):
            attention_mask = torch.ones(
                inputs_embeds.shape[0], inputs_embeds.shape[1], dtype=torch.long, device=inputs_embeds.device
            )
            hidden = self.text_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            ).last_hidden_state
            next_token = self.lm_head(hidden[:, -1]).argmax(dim=-1)
            generated.append(next_token)
            if eos_ids and int(next_token.item()) in eos_ids:
                break
            next_embed = self.qwen_model.get_input_embeddings()(next_token[:, None]).to(dtype=inputs_embeds.dtype)
            next_position = self._append_positions(position_ids, 1, inputs_embeds.device)
            inputs_embeds = torch.cat([inputs_embeds, next_embed], dim=1)
            position_ids = torch.cat([position_ids, next_position], dim=-1)

        if not generated:
            return torch.zeros((1, 0), dtype=torch.long, device=prompt_ids.device)
        return torch.stack(generated, dim=1)


def build(
    surgmotion_checkpoint=DEFAULT_SURGMOTION_CHECKPOINT,
    qwen_model_id=DEFAULT_QWEN_MODEL,
    aux_num_frames=16,
    aux_num_tokens=AUX_NUM_TOKENS,
    lora_r=16,
    # Default stays BF16 (matches train_dual_encoder_vqa.py's TrainingArguments
    # bf16=True - the dev GPU this trains on supports it fine). Inference on the
    # actual submission GPU (a T4) needs FP16 instead - see predict_dual_encoder_vqa.py
    # ::load_model(), which passes dtype=torch.float16 explicitly rather than relying
    # on this default, for exactly the reason explained there. Callers should always
    # pass dtype explicitly rather than relying on this default.
    dtype=torch.bfloat16,
    local_files_only=True,
    train_surgmotion_blocks=0,
    freeze_qwen=False,
    vision_lora_r=0,
    use_fusion=False,
    fusion_heads=8,
):
    """Constructs the dual-encoder prototype. Unlike SurgMotionVQA.build(), Qwen's
    own vision tower (`model.visual`) is kept and frozen rather than deleted.

    use_fusion=False (default) preserves the original design: the 32 aux tokens
    are appended to the prompt sequence untouched, and the LLM's own
    self-attention (through LoRA) is solely responsible for relating them to
    Qwen's native visual tokens. use_fusion=True inserts a small trainable
    CrossModalFusion block (cross-attention, aux tokens as query, Qwen's visual
    tokens as key/value) before that append, testing whether giving the two
    visual streams an explicit interaction point - instead of leaving it all to
    a rank-16 LoRA adapter over the concatenated sequence - helps.

    vision_lora_r=0 (default) keeps that vision tower fully frozen, as originally
    designed - matched-comparison evidence (see train_dual_encoder_vqa.py's module
    docstring / project memory) shows this frozen, natively-pretrained vision path
    is itself the largest single driver of this architecture's gain over
    SurgMotionVQA's from-scratch domain encoder, so it is deliberately never
    fine-tuned by default. vision_lora_r>0 is an experiment testing whether a
    small, low-rank adaptation on top of that strong pretrained initialization
    (not a full unfreeze) can add surgical-domain adaptation without discarding
    it, gated behind an explicit opt-in for exactly that reason."""
    from transformers import Qwen2_5_VLForConditionalGeneration
    from peft import LoraConfig, get_peft_model

    if not os.path.isfile(surgmotion_checkpoint):
        raise FileNotFoundError(f"SurgMotion checkpoint not found: {surgmotion_checkpoint}")
    if local_files_only and not os.path.isdir(qwen_model_id):
        raise FileNotFoundError(f"Local Qwen2.5-VL model not found: {qwen_model_id}")

    encoder = load_surgmotion_encoder(
        surgmotion_checkpoint,
        num_frames=aux_num_frames,
        train_last_n_blocks=train_surgmotion_blocks,
    )

    full_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        qwen_model_id,
        dtype=dtype,
        local_files_only=local_files_only,
    )
    text_hidden_size = full_model.config.text_config.hidden_size
    aux_projector = VisualProjector(
        in_dim=encoder.embed_dim,
        out_dim=text_hidden_size,
        hidden_dim=text_hidden_size,
        num_output_tokens=aux_num_tokens,
    )
    fusion = CrossModalFusion(text_hidden_size, num_heads=fusion_heads).to(dtype=dtype) if use_fusion else None

    qwen_model = full_model.model  # Qwen2_5_VLModel - keeps .visual, unlike SurgMotionVQA.
    if vision_lora_r > 0:
        # Qwen2_5_VLVisionAttention uses self.qkv/self.proj (not q/k/v/o_proj);
        # Qwen2_5_VLMLP uses self.gate_proj/up_proj/down_proj - same leaf names as
        # the language model's MLP, but get_peft_model is scoped to qwen_model.visual
        # only here, so there is no cross-talk with the text LoRA config below.
        # "proj" is deliberately excluded from target_modules: PatchEmbed also has
        # a submodule named "proj" (an nn.Conv3d), which naive substring/leaf-name
        # matching would otherwise also try to wrap.
        vision_peft_config = LoraConfig(
            r=vision_lora_r,
            lora_alpha=vision_lora_r * 2,
            target_modules=["qkv", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.0,  # DualEncoderVQA.train() always forces qwen_model.visual.eval() (see class above), so dropout would be inactive anyway
            bias="none",
        )
        qwen_model.visual = get_peft_model(qwen_model.visual, vision_peft_config)
        qwen_model.visual.print_trainable_parameters()
    else:
        qwen_model.visual.requires_grad_(False)
        qwen_model.visual.eval()
    text_model = qwen_model.language_model
    lm_head = full_model.lm_head
    lm_head.requires_grad_(False)

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * 2,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    # Qwen2.5-VL's vision tower uses fused qkv/proj naming (verified against
    # transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py: Qwen2VLVisionAttention
    # uses self.qkv/self.proj, not q_proj/k_proj/v_proj/o_proj), so this LoRA config
    # only ever matches language_model attention/MLP layers - the vision tower is
    # never touched by it even without an explicit exclusion.
    text_model = get_peft_model(text_model, peft_config)
    if freeze_qwen:
        text_model.requires_grad_(False)
    text_model.print_trainable_parameters()

    return DualEncoderVQA(qwen_model, encoder, aux_projector, text_model, lm_head, fusion=fusion)
