"""
GR-1 frozen encoder for RAID visual integration.

Provides two capabilities:
  1. encode_frames(images)   -> (B, feat_dim) state features from frozen MAE + embed_img
  2. predict_next_feat(...)  -> (B, feat_dim) predicted next-state features via GR-1 fwd_pred head

All weights stay frozen. Only the RAID decoder trains.

GR-1 config (logs/configs.json from bytedance/GR-1):
  embed_dim=384, n_layer=12, n_head=12, img_feat_dim=768, patch_feat_dim=768,
  lang_feat_dim=512, resampler_num_latents=9, seq_len=10
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

GR1_REPO = Path("/home/ubuntu/GR-1")
_gr1_str = str(GR1_REPO)

import clip  # noqa: E402

# --- Load GR-1 models without polluting the 'models' namespace ---
# 1. Save any existing 'models' entries (likely RAID's src/models.py)
_saved_model_modules = {k: v for k, v in sys.modules.items()
                        if k == "models" or k.startswith("models.")}
# 2. Remove them so GR-1 can claim the 'models' namespace
for _k in _saved_model_modules:
    del sys.modules[_k]
# 3. Add GR-1 to front of sys.path, remove src/ so GR-1's models/ wins
_gr1_added = _gr1_str not in sys.path
if _gr1_added:
    sys.path.insert(0, _gr1_str)
# Temporarily move 'src' behind GR-1 if it's first
_src_idx = next((i for i, p in enumerate(sys.path) if p.endswith("/src") or p == "src"), None)

# 4. Import GR-1 modules
import models.vision_transformer as vits   # GR-1's ViT
from models.gr1 import GR1                 # GR-1 model class

# 5. Save GR-1 model refs, clean up GR-1's 'models' namespace
for _k in list(sys.modules.keys()):
    if _k == "models" or _k.startswith("models."):
        del sys.modules[_k]
# 6. Restore RAID's saved modules
sys.modules.update(_saved_model_modules)
# 7. Remove GR-1 from sys.path
if _gr1_added and _gr1_str in sys.path:
    sys.path.remove(_gr1_str)

# ---------------------------------------------------------------------------
# Constants matching logs/configs.json
# ---------------------------------------------------------------------------
GR1_CFG = {
    "embed_dim": 384,
    "n_layer": 12,
    "n_head": 12,
    "activation_function": "relu",
    "dropout": 0.1,
    "n_positions": 1024,
    "resampler_depth": 3,
    "resampler_dim_head": 128,
    "resampler_heads": 4,
    "resampler_num_media_embeds": 1,
    "resampler_num_latents": 9,
    "seq_len": 10,
    "act_dim": 7,
    "state_dim": 7,
    "use_hand_rgb": False,   # disable hand cam - LIBERO only has one view
    "clip_backbone": "ViT-B/32",
    "img_feat_dim": 768,
    "patch_feat_dim": 768,
    "lang_feat_dim": 512,
    "without_norm_pix_loss": False,
}

FEAT_DIM = GR1_CFG["embed_dim"]  # 384 - dimension of s_t / s_next features
IMAGE_SIZE = 224
RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD  = (0.229, 0.224, 0.225)


def _build_preprocess() -> T.Compose:
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=T.InterpolationMode.BICUBIC),
        T.Normalize(RGB_MEAN, RGB_STD),
    ])


class GR1Encoder(nn.Module):
    """
    Frozen GR-1 wrapper.

    Usage:
        enc = GR1Encoder.from_checkpoints(mae_ckpt, gr1_ckpt, device)
        feat_t    = enc.encode_frames(img_t)          # (B, 384)
        feat_next = enc.encode_frames(img_next)       # (B, 384)  -- GT at train time
        feat_pred = enc.predict_next_feat(img_t, lang) # (B, 384) -- at rollout time
    """

    def __init__(self, model_mae: nn.Module, model_gr1: GR1, device: torch.device):
        super().__init__()
        self.device = device
        self.preprocess = _build_preprocess()

        # Store as non-parameter attributes (frozen)
        self.mae = model_mae.to(device).eval()
        self.gr1 = model_gr1.to(device).eval()

        # Freeze everything
        for p in self.mae.parameters():
            p.requires_grad_(False)
        for p in self.gr1.parameters():
            p.requires_grad_(False)

        self.feat_dim = FEAT_DIM
        self._tokenizer = clip.tokenize

    # ------------------------------------------------------------------
    @classmethod
    def from_checkpoints(
        cls,
        mae_ckpt: str | Path,
        gr1_ckpt: str | Path,
        device: str | torch.device = "cuda",
    ) -> "GR1Encoder":
        device = torch.device(device)

        # --- MAE ViT-B ---
        model_mae = vits.__dict__["vit_base"](patch_size=16, num_classes=0)
        ckpt = torch.load(mae_ckpt, map_location="cpu", weights_only=False)
        model_mae.load_state_dict(ckpt["model"], strict=False)
        model_mae.eval()

        # --- CLIP ---
        model_clip, _ = clip.load(GR1_CFG["clip_backbone"], device=device)

        # --- GR-1 ---
        resampler_params = {
            "depth": GR1_CFG["resampler_depth"],
            "dim_head": GR1_CFG["resampler_dim_head"],
            "heads": GR1_CFG["resampler_heads"],
            "num_latents": GR1_CFG["resampler_num_latents"],
            "num_media_embeds": GR1_CFG["resampler_num_media_embeds"],
        }
        model_gr1 = GR1(
            model_clip=model_clip,
            model_mae=model_mae,
            state_dim=GR1_CFG["state_dim"],
            act_dim=GR1_CFG["act_dim"],
            hidden_size=GR1_CFG["embed_dim"],
            sequence_length=GR1_CFG["seq_len"],
            training_target=["act_pred", "fwd_pred"],
            img_feat_dim=GR1_CFG["img_feat_dim"],
            lang_feat_dim=GR1_CFG["lang_feat_dim"],
            patch_feat_dim=GR1_CFG["patch_feat_dim"],
            resampler_params=resampler_params,
            without_norm_pixel_loss=GR1_CFG["without_norm_pix_loss"],
            use_hand_rgb=GR1_CFG["use_hand_rgb"],
            n_layer=GR1_CFG["n_layer"],
            n_head=GR1_CFG["n_head"],
            n_inner=4 * GR1_CFG["embed_dim"],
            activation_function=GR1_CFG["activation_function"],
            n_positions=GR1_CFG["n_positions"],
            resid_pdrop=GR1_CFG["dropout"],
            attn_pdrop=GR1_CFG["dropout"],
        )
        payload = torch.load(gr1_ckpt, map_location="cpu", weights_only=False)
        state_dict = payload.get("state_dict", payload)
        missing, unexpected = model_gr1.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[GR1Encoder] Missing keys ({len(missing)}): {missing[:5]}...")
        model_gr1.eval()

        return cls(model_mae, model_gr1, device)

    # ------------------------------------------------------------------
    def _preprocess_images(self, images: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Accept (B, H, W, 3) uint8 or (B, 3, H, W) float, return (B, 3, 224, 224) normalised."""
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.shape[-1] == 3:          # (B, H, W, 3) → (B, 3, H, W)
            images = images.permute(0, 3, 1, 2)
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        images = images.to(self.device)
        return self.preprocess(images)     # resize + normalise

    @torch.no_grad()
    def encode_frames(self, images: torch.Tensor | np.ndarray) -> torch.Tensor:
        """
        Encode images to 384-dim GR-1 latent features.

        Args:
            images: (B, H, W, 3) uint8 or (B, 3, H, W) float

        Returns:
            feats: (B, 384)
        """
        imgs = self._preprocess_images(images)          # (B, 3, 224, 224)
        B = imgs.shape[0]
        obs_emb, _ = self.mae(imgs)                     # (B, 768)
        obs_emb = obs_emb.view(B, 1, -1).float()       # (B, 1, 768)
        feat = self.gr1.embed_img(obs_emb).squeeze(1)   # (B, 384)
        return feat

    @torch.no_grad()
    def predict_next_feat(
        self,
        images: torch.Tensor | np.ndarray,
        language: list[str] | str | None = None,
        robot_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict next-frame features using GR-1's forward prediction head.

        Runs one-step GR-1 inference (seq_len=1 context) and returns the
        obs_query output from the transformer — a 384-dim vector that
        represents the predicted next observation in GR-1's latent space.

        Args:
            images:      (B, H, W, 3) or (B, 3, H, W)
            language:    list of B strings, or a single string (broadcast)
            robot_state: (B, 7) normalized robot state; zeros if None

        Returns:
            next_feat: (B, 384)
        """
        imgs = self._preprocess_images(images)   # (B, 3, 224, 224)
        B = imgs.shape[0]
        dev = self.device

        # --- Build single-step sequence inputs ---
        rgb_seq = imgs.unsqueeze(1)              # (B, 1, 3, 224, 224)

        if robot_state is None:
            arm_state = torch.zeros(B, 1, 6, device=dev)
            gripper_state = torch.zeros(B, 1, 2, device=dev)
            gripper_state[:, :, 0] = 1.0        # gripper open
        else:
            rs = robot_state.to(dev)
            if rs.ndim == 2:
                rs = rs.unsqueeze(1)             # (B, 1, 7)
            arm_state = rs[:, :, :6].float()
            gripper_raw = rs[:, :, 6:7]
            # one-hot encode gripper (0=open, 1=closed)
            g_int = gripper_raw.long().clamp(0, 1)
            gripper_state = torch.zeros(B, 1, 2, device=dev)
            gripper_state.scatter_(2, g_int, 1.0)

        state_data = {"arm": arm_state, "gripper": gripper_state}
        attention_mask = torch.ones(B, 1, dtype=torch.long, device=dev)

        # --- Language tokenization ---
        if language is None:
            language = ["robot manipulation"] * B
        elif isinstance(language, str):
            language = [language] * B
        tokenized = self.gr1.model_clip.__class__  # just for type hint
        tokenized = clip.tokenize(language).to(dev)

        # --- GR-1 forward (act_pred + fwd_pred) ---
        with torch.no_grad():
            prediction = self.gr1(
                rgb=rgb_seq,
                hand_rgb=torch.zeros_like(rgb_seq),   # unused (use_hand_rgb=False)
                state=state_data,
                language=tokenized,
                attention_mask=attention_mask,
            )

        # Extract predicted next obs from the obs_query output.
        # GR-1's obs_preds is (B, 1, n_patches, patch_size^2*3) pixel predictions.
        # Instead of decoding pixels, we use the patch-latent output directly from
        # the transformer: obs_query_token hidden state → embed back to feat space.
        # We take the MAE CLS embedding of the PREDICTED pixel patches as the
        # s_next representation by re-encoding the reconstructed image.
        obs_preds = prediction.get("obs_preds")   # (B, 1, 196, 768) or None

        if obs_preds is not None:
            # Reconstruct image from predicted patches and re-encode with MAE
            next_feat = self._recon_and_encode(obs_preds[:, 0], imgs)  # (B, 384)
        else:
            # Fallback: use current frame features (degenerate proxy)
            next_feat = self.encode_frames(images)

        return next_feat

    def _recon_and_encode(
        self,
        patch_preds: torch.Tensor,   # (B, 196, 768) normalised pixel patches
        orig_imgs: torch.Tensor,     # (B, 3, 224, 224) normalised — used for MAE norm stats
    ) -> torch.Tensor:
        """
        Reconstruct a 224×224 image from MAE pixel-patch predictions,
        then re-encode with MAE to get a 384-dim feature.
        """
        B = patch_preds.shape[0]
        p = 16
        h = w = IMAGE_SIZE // p   # 14

        # Un-normalise patches (MAE normalises per-patch during training)
        # patch_preds: (B, 196, 768=p*p*3)
        patches = patch_preds  # keep normalised for re-encoding (MAE handles this)

        # Reshape to (B, 3, 224, 224)
        patches = patches.view(B, h, w, p, p, 3)
        patches = patches.permute(0, 5, 1, 3, 2, 4)   # (B, 3, h, p, w, p)
        recon = patches.reshape(B, 3, IMAGE_SIZE, IMAGE_SIZE)

        # Re-encode reconstructed image with frozen MAE
        obs_emb, _ = self.mae(recon)                  # (B, 768)
        obs_emb = obs_emb.view(B, 1, -1).float()
        feat = self.gr1.embed_img(obs_emb).squeeze(1)  # (B, 384)
        return feat
