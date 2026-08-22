"""
Research prototype: SurgMotion (frozen, surgical-domain video encoder) + a learned
projector + Qwen2.5-VL-3B's language-model decoder (LoRA-adapted), bypassing
Qwen2.5-VL's own general-purpose vision tower.

This was flagged during the review as high-effort / not recommended within the
submission's time budget - it is an exploratory prototype for the report's future-work
section, NOT the submitted model (see checkpoints/qwen2_vl_2b_lora_v5 for that).

Rationale: Qwen2.5-VL's vision tower is trained on general image/video data. SurgMotion
is pretrained on 15M frames / 3,658 hours of surgical video (SurgMotion-15M) with a
motion-focused V-JEPA2 latent-prediction objective, which in principle should produce
stronger surgical-domain visual features (tool/tissue motion) than adapting a
general-purpose encoder on the comparatively tiny amount of SurgVU video available here.
Architecturally this mirrors SurgViVQA (frozen video encoder + trainable bridge + LoRA
LLM) but swaps VideoMAE (general-domain) for SurgMotion (surgical-domain) and BLIP's
cross-attention bridge for a cheaper linear/MLP projector (closer to LLaVA's
visual-prefix approach).

=== Validation status (2026-07-30) ===
VERIFIED:
  - SurgMotion-vitl.pt loads exactly into the vendored ViT-L architecture (0 missing /
    unexpected keys). A dummy (1, 3, 16, 256, 256) clip produces (1, 2048, 1024).
  - Real GPU forward/backward and component checkpoint save work with the complete
    SurgMotion + Qwen2.5-VL-3B text model. A 40-step four-example overfit test reduced
    block loss from 2.32 to the 0.45-0.97 range.
  - Valid, partially-black, and fully-black clips are wired end-to-end. Fully-black
    clips bypass SurgMotion and use a learned no-visual prefix; partially-black clips
    use a frame/tubelet mask. Per-example curation weights are applied to token loss.

STILL EXPERIMENTAL:
  - Qwen2.5-VL normally uses multimodal RoPE positions derived from vision grid_thw.
    This external visual-prefix design uses ordinary 1D decoder positions. The smoke
    test proves structural trainability, not that this approximation preserves quality.
  - SurgMotion was pretrained at 64 frames; the current 16-frame cache is outside that
    temporal configuration. A feature probe and held-out metrics remain necessary
    before committing to a costly 64-frame run.
  - DPOTrainer and GRPOTrainer are deliberately not wired in. Preference optimization
    should only be considered after a stable SFT baseline and a grounded preference set.

Next steps to actually validate this direction (in priority order):
  1. Forward/backward pass on GPU with a real batch from train_vqa.json, check loss
     decreases at all over a few dozen steps (cheapest possible sanity check).
  2. Compare SurgMotion-extracted features to Qwen2-VL's own vision tower output on a
     handful of known-hard clips (e.g. case126) via simple linear probing (train a tiny
     classifier on frozen features to answer "is a large needle driver present" and see
     which encoder's features make that probe easier) - answers whether the encoder
     swap is worth a full LoRA fine-tune before committing days of GPU time.
  3. If (2) looks promising: re-extract frames at 64 frames/clip (SurgMotion's native
     config) and build a real training script analogous to train.py.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SURGMOTION_REPO = os.path.join(PROJECT_ROOT, "repo", "SurgMotion-main")
DEFAULT_SURGMOTION_CHECKPOINT = os.path.join(
    SURGMOTION_REPO, "ckpts", "SurgMotion-vitl.pt"
)
DEFAULT_QWEN_MODEL = os.path.join(
    PROJECT_ROOT, "checkpoints", "base_models", "Qwen2.5-VL-3B-Instruct"
)

if SURGMOTION_REPO not in sys.path:
    sys.path.insert(0, SURGMOTION_REPO)


def _clean_backbone_keys(state_dict):
    """SurgMotion's released checkpoints prefix every key with "module.backbone." from
    their distributed-training wrapper; strip it to match the plain VisionTransformer
    module's own key names."""
    out = {}
    for k, v in state_dict.items():
        out[k.replace("module.", "").replace("backbone.", "")] = v
    return out


def load_surgmotion_encoder(
    checkpoint_path,
    num_frames=16,
    img_size=256,
    patch_size=16,
    tubelet_size=2,
    train_last_n_blocks=0,
):
    """Loads the frozen SurgMotion ViT-Large video encoder from a local checkpoint
    (repo/SurgMotion-main/ckpts/SurgMotion-vitl.pt - 300M params, embed_dim=1024).
    Verified exact match (0 missing/unexpected keys) against this architecture config.
    """
    from src.models import vision_transformer as sm_vit  # SurgMotion repo (vendored via sys.path)

    encoder = sm_vit.vit_large(
        img_size=(img_size, img_size),
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        patch_size=patch_size,
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        uniform_power=False,
        use_rope=True,
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["encoder"]
    state_dict = _clean_backbone_keys(state_dict)
    missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[surgmotion] missing={len(missing)} unexpected={len(unexpected)} keys loading {checkpoint_path}")

    for parameter in encoder.parameters():
        parameter.requires_grad = False
    if not 0 <= train_last_n_blocks <= len(encoder.blocks):
        raise ValueError(
            f"train_last_n_blocks must be in [0, {len(encoder.blocks)}], "
            f"got {train_last_n_blocks}"
        )
    if train_last_n_blocks:
        for block in encoder.blocks[-train_last_n_blocks:]:
            block.requires_grad_(True)
        encoder.norm.requires_grad_(True)
    encoder.eval()
    return encoder


class VisualProjector(nn.Module):
    """Maps SurgMotion's per-token embeddings into Qwen2.5's language embedding space,
    and pools the (fixed) 2048-token grid down to a smaller budget so the visual
    prefix doesn't dominate the LLM's context length."""

    def __init__(self, in_dim=1024, out_dim=2048, num_output_tokens=64, hidden_dim=2048):
        super().__init__()
        self.num_output_tokens = num_output_tokens
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        # A full learned prefix (rather than a zero tensor) lets the decoder learn a
        # stable language-prior path when the endoscope feed is unavailable. Keeping
        # it inside the projector also includes it in the existing bridge checkpoint.
        self.no_visual_prefix = nn.Parameter(torch.empty(1, num_output_tokens, out_dim))
        nn.init.normal_(self.no_visual_prefix, mean=0.0, std=0.02)

    def no_visual(self, batch_size):
        return self.no_visual_prefix.expand(batch_size, -1, -1)

    def _pool_token_bins(self, values):
        """Mean-pool [B, C, N] into balanced, non-overlapping token bins.

        AdaptiveAvgPool1d backward can exceed the CUDA shared-memory launch
        limit for SurgMotion's 8,192-token 64-frame output. The common paths
        divide exactly (2,048/64 and 8,192/64), so reshape+mean is both
        equivalent and substantially simpler for CUDA. Keep a balanced split
        fallback for other valid token counts.
        """
        num_tokens = values.shape[-1]
        if num_tokens < self.num_output_tokens:
            raise ValueError(
                f"Cannot pool {num_tokens} tokens into "
                f"{self.num_output_tokens} non-empty bins"
            )
        if num_tokens % self.num_output_tokens == 0:
            tokens_per_bin = num_tokens // self.num_output_tokens
            return values.reshape(
                *values.shape[:-1],
                self.num_output_tokens,
                tokens_per_bin,
            ).mean(dim=-1)

        short_bin = num_tokens // self.num_output_tokens
        num_long_bins = num_tokens % self.num_output_tokens
        split_sizes = (
            [short_bin + 1] * num_long_bins
            + [short_bin] * (self.num_output_tokens - num_long_bins)
        )
        return torch.stack(
            [token_bin.mean(dim=-1) for token_bin in values.split(split_sizes, dim=-1)],
            dim=-1,
        )

    def forward(self, video_tokens, token_valid_mask=None):
        # video_tokens: [B, N, in_dim] -> [B, num_output_tokens, in_dim] -> [B, num_output_tokens, out_dim]
        transposed = video_tokens.transpose(1, 2)
        if token_valid_mask is None:
            pooled = self._pool_token_bins(transposed).transpose(1, 2)
            pooled_valid = torch.ones(
                video_tokens.shape[0],
                self.num_output_tokens,
                1,
                dtype=torch.bool,
                device=video_tokens.device,
            )
        else:
            if token_valid_mask.shape != video_tokens.shape[:2]:
                raise ValueError(
                    f"token_valid_mask={tuple(token_valid_mask.shape)} does not match "
                    f"video tokens={tuple(video_tokens.shape[:2])}"
                )
            mask = token_valid_mask.to(dtype=video_tokens.dtype).unsqueeze(1)
            pooled_mask = self._pool_token_bins(mask).transpose(1, 2)
            pooled = self._pool_token_bins(transposed * mask).transpose(1, 2)
            pooled = pooled / pooled_mask.clamp_min(1e-6)
            pooled_valid = pooled_mask > 0

        projected = self.mlp(pooled)
        no_visual = self.no_visual(video_tokens.shape[0]).to(dtype=projected.dtype)
        return torch.where(pooled_valid, projected, no_visual)


class SurgMotionVQA(nn.Module):
    """Frozen SurgMotion encoder -> projector -> Qwen2.5-VL text backbone (LoRA) ->
    Qwen2.5-VL's frozen lm_head. Only the projector and LoRA adapters are trainable."""

    def __init__(
        self,
        surgmotion_encoder,
        projector,
        text_model,
        lm_head,
        tubelet_size=2,
        num_frames=16,
    ):
        super().__init__()
        self.surgmotion_encoder = surgmotion_encoder
        self.projector = projector
        self.text_model = text_model  # Generic-LoRA-wrapped Qwen2_5_VLTextModel
        self.lm_head = lm_head        # Explicitly frozen in build()
        self.tubelet_size = tubelet_size
        self.num_frames = num_frames
        self.surgmotion_delta_keys = {
            name for name, parameter in surgmotion_encoder.named_parameters()
            if parameter.requires_grad
        }

    def train(self, mode=True):
        super().train(mode)
        # nn.Module.train() recurses into every child. Keep the frozen encoder in
        # inference mode so stochastic depth/dropout cannot perturb cached features.
        self.surgmotion_encoder.eval()
        return self

    def _token_valid_mask(self, frame_valid_mask, num_video_tokens):
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

    def encode_visual(self, video, frame_valid_mask=None):
        batch_size = video.shape[0]
        if frame_valid_mask is None:
            frame_valid_mask = torch.ones(
                batch_size, video.shape[2], dtype=torch.bool, device=video.device
            )
        else:
            frame_valid_mask = frame_valid_mask.to(device=video.device, dtype=torch.bool)
        has_visual = frame_valid_mask.any(dim=1)
        visual_embeds = self.projector.no_visual(batch_size)
        if not has_visual.any():
            return visual_embeds

        encoder_trainable = any(
            parameter.requires_grad for parameter in self.surgmotion_encoder.parameters()
        )
        with torch.set_grad_enabled(encoder_trainable and self.training):
            video_tokens = self.surgmotion_encoder(video[has_visual])
        token_valid_mask = self._token_valid_mask(
            frame_valid_mask[has_visual], video_tokens.shape[1]
        )
        projected = self.projector(video_tokens, token_valid_mask=token_valid_mask)
        visual_embeds = visual_embeds.to(dtype=projected.dtype).clone()
        visual_embeds[has_visual] = projected
        return visual_embeds

    @torch.no_grad()
    def generate(
        self,
        video,
        input_ids,
        attention_mask,
        frame_valid_mask=None,
        max_new_tokens=32,
        eos_token_id=None,
    ):
        """Greedy generation for the external visual-prefix architecture."""
        self.eval()
        visual_embeds = self.encode_visual(video, frame_valid_mask)
        generated = input_ids
        generated_mask = attention_mask
        eos_ids = set()
        if eos_token_id is not None:
            eos_ids = {eos_token_id} if isinstance(eos_token_id, int) else set(eos_token_id)

        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        for _ in range(max_new_tokens):
            text_embeds = self.text_model.get_input_embeddings()(generated)
            prefix = visual_embeds.to(dtype=text_embeds.dtype)
            inputs_embeds = torch.cat([prefix, text_embeds], dim=1)
            prefix_mask = torch.ones(
                generated.shape[0], prefix.shape[1],
                dtype=generated_mask.dtype, device=generated_mask.device,
            )
            full_mask = torch.cat([prefix_mask, generated_mask], dim=1)
            hidden = self.text_model(
                inputs_embeds=inputs_embeds,
                attention_mask=full_mask,
                use_cache=False,
            ).last_hidden_state
            next_token = self.lm_head(hidden[:, -1]).argmax(dim=-1)
            generated = torch.cat([generated, next_token[:, None]], dim=1)
            generated_mask = torch.cat(
                [generated_mask, torch.ones_like(next_token[:, None])], dim=1
            )
            if eos_ids:
                finished |= torch.tensor(
                    [int(token) in eos_ids for token in next_token],
                    device=finished.device,
                    dtype=torch.bool,
                )
                if finished.all():
                    break
        return generated

    def forward(
        self,
        video,
        input_ids,
        attention_mask,
        labels=None,
        frame_valid_mask=None,
        valid_fraction=None,
        sample_weight=None,
    ):
        del valid_fraction  # Kept in the batch/output contract for diagnostics.
        batch_size = video.shape[0]
        visual_embeds = self.encode_visual(video, frame_valid_mask)

        text_embeds = self.text_model.get_input_embeddings()(input_ids)  # [B, L, H]
        # The projector starts in FP32 while the downloaded LLM is normally BF16.
        # Match dtypes explicitly so torch.cat and the first decoder layer are valid
        # even when the caller is not inside an autocast context.
        visual_embeds = visual_embeds.to(dtype=text_embeds.dtype)
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)   # [B, V+L, H]

        v = visual_embeds.shape[1]
        vis_mask = torch.ones(batch_size, v, dtype=attention_mask.dtype, device=attention_mask.device)
        full_attention_mask = torch.cat([vis_mask, attention_mask], dim=1)

        hidden_states = self.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            use_cache=False,
        ).last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            vis_labels = torch.full((batch_size, v), -100, dtype=labels.dtype, device=labels.device)
            full_labels = torch.cat([vis_labels, labels], dim=1)
            shift_logits = logits[:, :-1, :]
            shift_labels = full_labels[:, 1:]
            token_loss = F.cross_entropy(
                shift_logits.float().transpose(1, 2),
                shift_labels,
                ignore_index=-100,
                reduction="none",
            )
            supervised = shift_labels.ne(-100)
            per_sample_loss = (token_loss * supervised).sum(dim=1) / supervised.sum(
                dim=1
            ).clamp_min(1)
            if sample_weight is None:
                sample_weight = torch.ones_like(per_sample_loss)
            else:
                sample_weight = sample_weight.to(
                    device=per_sample_loss.device, dtype=per_sample_loss.dtype
                )
            loss = (per_sample_loss * sample_weight).sum() / sample_weight.sum().clamp_min(
                1e-6
            )

        return {"loss": loss, "logits": logits}


def build(
    surgmotion_checkpoint=DEFAULT_SURGMOTION_CHECKPOINT,
    qwen_model_id=DEFAULT_QWEN_MODEL,
    num_frames=16,
    lora_r=16,
    dtype=torch.bfloat16,
    local_files_only=True,
    train_surgmotion_blocks=0,
    freeze_qwen=False,
    projector_tokens=64,
):
    """Constructs the full prototype without Qwen2.5-VL's vision tower.

    The default model path is the downloaded local Qwen2.5-VL-3B snapshot. This
    function intentionally does not choose a device or start training; the caller must
    wait for the active training run to finish before moving it to a GPU.
    """
    from transformers import Qwen2_5_VLForConditionalGeneration
    from peft import LoraConfig, get_peft_model

    if not os.path.isfile(surgmotion_checkpoint):
        raise FileNotFoundError(f"SurgMotion checkpoint not found: {surgmotion_checkpoint}")
    if local_files_only and not os.path.isdir(qwen_model_id):
        raise FileNotFoundError(f"Local Qwen2.5-VL model not found: {qwen_model_id}")

    encoder = load_surgmotion_encoder(
        surgmotion_checkpoint,
        num_frames=num_frames,
        train_last_n_blocks=train_surgmotion_blocks,
    )

    full_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        qwen_model_id,
        dtype=dtype,
        local_files_only=local_files_only,
    )
    text_hidden_size = full_model.config.text_config.hidden_size
    projector = VisualProjector(
        in_dim=encoder.embed_dim,
        out_dim=text_hidden_size,
        hidden_dim=text_hidden_size,
        num_output_tokens=projector_tokens,
    )

    text_model = full_model.model.language_model
    lm_head = full_model.lm_head
    lm_head.requires_grad_(False)
    del full_model.model.visual  # SurgMotion replaces this entirely.

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * 2,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    # Do not set task_type="CAUSAL_LM" here. We are wrapping the bare text backbone,
    # not Qwen2_5_VLForConditionalGeneration; the former intentionally has no
    # prepare_inputs_for_generation method expected by PeftModelForCausalLM.
    text_model = get_peft_model(text_model, peft_config)
    if freeze_qwen:
        text_model.requires_grad_(False)
    text_model.print_trainable_parameters()

    del full_model
    return SurgMotionVQA(
        encoder,
        projector,
        text_model,
        lm_head,
        tubelet_size=2,
        num_frames=num_frames,
    )


if __name__ == "__main__":
    # CPU-only structural smoke test of the SurgMotion encoder/projector half.
    enc = load_surgmotion_encoder(
        DEFAULT_SURGMOTION_CHECKPOINT,
        num_frames=16,
    )
    dummy = torch.randn(1, 3, 16, 256, 256)
    with torch.no_grad():
        tokens = enc(dummy)
    proj = VisualProjector(in_dim=enc.embed_dim, out_dim=2048)
    with torch.no_grad():
        visual_embeds = proj(tokens)
    print("SurgMotion tokens:", tokens.shape, "-> projected:", visual_embeds.shape)
