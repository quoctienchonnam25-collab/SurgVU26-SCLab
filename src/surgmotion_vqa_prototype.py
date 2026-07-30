"""
Research prototype: SurgMotion (frozen, surgical-domain video encoder) + a learned
projector + Qwen2-VL-2B's language-model decoder (LoRA-adapted), bypassing Qwen2-VL's
own general-purpose vision tower.

This was flagged during the review as high-effort / not recommended within the
submission's time budget - it is an exploratory prototype for the report's future-work
section, NOT the submitted model (see checkpoints/qwen2_vl_2b_lora_v5 for that).

Rationale: Qwen2-VL's vision tower is trained on general image/video data. SurgMotion
is pretrained on 15M frames / 3,658 hours of surgical video (SurgMotion-15M) with a
motion-focused V-JEPA2 latent-prediction objective, which in principle should produce
stronger surgical-domain visual features (tool/tissue motion) than adapting a
general-purpose encoder on the comparatively tiny amount of SurgVU video available here.
Architecturally this mirrors SurgViVQA (frozen video encoder + trainable bridge + LoRA
LLM) but swaps VideoMAE (general-domain) for SurgMotion (surgical-domain) and BLIP's
cross-attention bridge for a cheaper linear/MLP projector (closer to LLaVA's
visual-prefix approach).

=== Validation status (2026-07) ===
VERIFIED (CPU smoke test, see conversation log): loading SurgMotion-vitl.pt into the
vendored `vision_transformer.vit_large` architecture from repo/SurgMotion-main is exact
(0 missing/unexpected keys), and a forward pass on a dummy (1, 3, 16, 256, 256) clip
produces the expected (1, 2048, 1024) token grid (2048 = (16/tubelet_size=2) * (256/16)
* (256/16)).

NOT YET VERIFIED (needs GPU time, which was occupied training v6 when this was written):
  - The Qwen2VLTextModel wiring below (attribute paths were confirmed by reading
    transformers' modeling_qwen2_vl.py source, not by running it).
  - Qwen2-VL uses multimodal RoPE (M-RoPE): position_ids are normally computed from the
    vision grid_thw layout. Feeding a prefix of visual-prefix embeddings with plain 1D
    position_ids (as this prototype does) instead of Qwen2-VL's own M-RoPE positions is
    an approximation - it may still train reasonably (the LM's attention doesn't
    strictly require the *original* multimodal position scheme once the vision tower
    itself is bypassed), but this is a real open question, not a settled fact.
  - No actual training loop/dataset wiring yet - `build()` only constructs the model.
  - SurgMotion was pretrained at num_frames=64; this prototype defaults to num_frames=16
    to reuse the existing frame cache cheaply. Whether RoPE lets the encoder generalize
    from 64 to 16 frames without a quality drop is untested.

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

SURGMOTION_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "repo", "SurgMotion-main")
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


def load_surgmotion_encoder(checkpoint_path, num_frames=16, img_size=256, patch_size=16, tubelet_size=2):
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

    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()
    return encoder


class VisualProjector(nn.Module):
    """Maps SurgMotion's per-token embeddings into Qwen2's language embedding space,
    and pools the (fixed) 2048-token grid down to a smaller budget so the visual
    prefix doesn't dominate the LLM's context length."""

    def __init__(self, in_dim=1024, out_dim=1536, num_output_tokens=64, hidden_dim=2048):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(num_output_tokens)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, video_tokens):
        # video_tokens: [B, N, in_dim] -> [B, num_output_tokens, in_dim] -> [B, num_output_tokens, out_dim]
        pooled = self.pool(video_tokens.transpose(1, 2)).transpose(1, 2)
        return self.mlp(pooled)


class SurgMotionVQA(nn.Module):
    """Frozen SurgMotion encoder -> trainable projector -> Qwen2VLTextModel (LoRA) ->
    Qwen2-VL's own lm_head. Only the projector and LoRA adapters are trainable."""

    def __init__(self, surgmotion_encoder, projector, text_model, lm_head):
        super().__init__()
        self.surgmotion_encoder = surgmotion_encoder
        self.projector = projector
        self.text_model = text_model  # LoRA-wrapped Qwen2VLTextModel
        self.lm_head = lm_head        # kept frozen (or could be unfrozen if desired)

    def forward(self, video, input_ids, attention_mask, labels=None):
        with torch.no_grad():
            video_tokens = self.surgmotion_encoder(video)          # [B, 2048, 1024]
        visual_embeds = self.projector(video_tokens)                # [B, V, 1536]

        text_embeds = self.text_model.get_input_embeddings()(input_ids)  # [B, L, 1536]
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)   # [B, V+L, 1536]

        v = visual_embeds.shape[1]
        vis_mask = torch.ones(video_tokens.shape[0], v, dtype=attention_mask.dtype, device=attention_mask.device)
        full_attention_mask = torch.cat([vis_mask, attention_mask], dim=1)

        hidden_states = self.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
        ).last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            vis_labels = torch.full((video_tokens.shape[0], v), -100, dtype=labels.dtype, device=labels.device)
            full_labels = torch.cat([vis_labels, labels], dim=1)
            shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
            shift_labels = full_labels[:, 1:].reshape(-1)
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

        return logits, loss


def build(
    surgmotion_checkpoint=r"d:\SCLab\Surgical-Video-Understanding\SurgVU-challenge\repo\SurgMotion-main\ckpts\SurgMotion-vitl.pt",
    qwen_model_id="Qwen/Qwen2-VL-2B-Instruct",
    num_frames=16,
    lora_r=16,
):
    """Constructs the full prototype model. Mirrors train.py's LoRA setup so results
    are comparable. NOT yet wired into a training loop - see module docstring."""
    from transformers import Qwen2VLForConditionalGeneration
    from peft import LoraConfig, get_peft_model

    encoder = load_surgmotion_encoder(surgmotion_checkpoint, num_frames=num_frames)
    projector = VisualProjector(in_dim=encoder.embed_dim, out_dim=1536)

    full_model = Qwen2VLForConditionalGeneration.from_pretrained(qwen_model_id, torch_dtype=torch.bfloat16)
    text_model = full_model.model.language_model  # Qwen2VLTextModel - no vision tower
    lm_head = full_model.lm_head
    del full_model.model.visual  # SurgMotion replaces this entirely - free the memory

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * 2,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    text_model = get_peft_model(text_model, peft_config)
    text_model.print_trainable_parameters()

    return SurgMotionVQA(encoder, projector, text_model, lm_head)


if __name__ == "__main__":
    # CPU-only structural smoke test of the SurgMotion half (the part that's actually
    # verified). Building the full Qwen2 half here would require GPU/real download
    # time - run build() directly once GPU is free to validate the rest.
    enc = load_surgmotion_encoder(
        r"d:\SCLab\Surgical-Video-Understanding\SurgVU-challenge\repo\SurgMotion-main\ckpts\SurgMotion-vitl.pt",
        num_frames=16,
    )
    dummy = torch.randn(1, 3, 16, 256, 256)
    with torch.no_grad():
        tokens = enc(dummy)
    proj = VisualProjector(in_dim=enc.embed_dim, out_dim=1536)
    with torch.no_grad():
        visual_embeds = proj(tokens)
    print("SurgMotion tokens:", tokens.shape, "-> projected:", visual_embeds.shape)
