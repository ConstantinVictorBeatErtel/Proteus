"""Frozen CLIP visual encoder + tactile MLP."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
CLIP_EMBED_DIM = 512


class FrozenCLIPEncoder(nn.Module):
    """
    Loads openai/clip-vit-base-patch32, freezes all weights.
    preprocess + encode_image → 512-dim L2-normalized features (CLIP default).
    """

    def __init__(self, model_id: str = CLIP_MODEL_ID, device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def embed_pil_batch(self, images: list) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, dtype=torch.float32)
        out = self.model.get_image_features(pixel_values=pixel_values)
        # Transformers ≥5 returns BaseModelOutputWithPooling; pooler_output is projected embeddings.
        feats = out if isinstance(out, torch.Tensor) else out.pooler_output
        return feats.detach().cpu().float()


class TactileEncoder(nn.Module):
    """MLP tactile encoder (12×64 → 64)."""

    def __init__(self, grid_h: int = 12, grid_w: int = 64, out_dim: int = 64):
        super().__init__()
        in_dim = grid_h * grid_w
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


if __name__ == "__main__":
    print("[encoders] Smoke test FrozenCLIPEncoder + TactileEncoder")
    tac = torch.randn(4, 12, 64)
    enc = TactileEncoder()
    print("tactile out:", enc(tac).shape)

    rng = torch.Generator().manual_seed(42)
    arr = torch.randint(
        0, 256, (224, 224, 3), generator=rng, dtype=torch.uint8
    ).numpy()
    img = Image.fromarray(arr)
    clip_enc = FrozenCLIPEncoder(device=torch.device("cpu"))
    z = clip_enc.embed_pil_batch([img])
    print("CLIP out:", z.shape)
