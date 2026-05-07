"""
V-JEPA 2 frozen encoder for RAID visual integration.

Loads facebook/vjepa2-vitl-fpc64-256 via HuggingFace AutoModel,
freezes all parameters, and provides encode_frames() which outputs
1024-dim latent features from patch-token pooling.

Implements a module-level singleton cache so the model is loaded
once and reused across calls.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_vjepa_model: Optional[nn.Module] = None
_vjepa_device: Optional[torch.device] = None


def _get_vjepa_model(device: torch.device) -> nn.Module:
    global _vjepa_model, _vjepa_device
    if _vjepa_model is not None and _vjepa_device == device:
        return _vjepa_model
    from transformers import AutoModel
    print("[vjepa_encoder] loading V-JEPA 2 ViT-L ...")
    _vjepa_model = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
    for p in _vjepa_model.parameters():
        p.requires_grad_(False)
    _vjepa_model = _vjepa_model.to(device).eval()
    _vjepa_device = device
    return _vjepa_model


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


def _normalize_manual(frames: torch.Tensor) -> torch.Tensor:
    """Manual normalization returning (B, 1, C, H, W)."""
    mean = torch.tensor(_IMAGENET_MEAN, device=frames.device,
                        dtype=frames.dtype).view(1, 3, 1, 1)
    std  = torch.tensor(_IMAGENET_STD,  device=frames.device,
                        dtype=frames.dtype).view(1, 3, 1, 1)
    normed = (frames - mean) / std  # (B, C, H, W)
    return normed.unsqueeze(1)      # (B, 1, C, H, W)


# ---------------------------------------------------------------------------
# VJEPAEncoder
# ---------------------------------------------------------------------------

class VJEPAEncoder:
    def __init__(self, device: str | torch.device = "cuda"):
        self.device = torch.device(device)
        self.model = _get_vjepa_model(self.device)
        self.feat_dim = 1024

    @torch.no_grad()
    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Encode frames to 1024-dim V-JEPA 2 latent features.

        Args:
            frames: (B, C, H, W) float32 in [0, 1]

        Returns:
            feats: (B, 1024)
        """
        if frames.ndim == 3:
            frames = frames.unsqueeze(0)

        B, C, H, W = frames.shape

        # Resize to 256x256
        frames_resized = TF.resize(
            frames, [256, 256],
            interpolation=TF.InterpolationMode.BICUBIC,
            antialias=True
        )  # (B, C, 256, 256)

        # Normalize and add T=1 dim -> (B, 1, C, 256, 256)
        video = _normalize_manual(frames_resized).to(self.device)

        # Forward pass
        out = self.model(video)
        hs = out.last_hidden_state  # (B, num_tokens, 1024)

        # Mean pool patch tokens (skip CLS at position 0)
        if hs.shape[1] > 1:
            hs = hs[:, 1:]
        feat = hs.mean(dim=1)  # (B, 1024)
        return feat


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    enc = VJEPAEncoder(device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"[vjepa_encoder] feat_dim = {enc.feat_dim}")
    dummy = torch.rand(4, 3, 128, 128, dtype=torch.float32)
    feats = enc.encode_frames(dummy)
    print(f"[vjepa_encoder] output shape: {feats.shape}")
    assert feats.shape == (4, 1024), f"Expected (4, 1024), got {feats.shape}"
    print("[vjepa_encoder] OK")