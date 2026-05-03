"""Visuo-tactile BC policies with a shared causal Transformer backbone."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def _causal_bool_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    # True = masked out (cannot attend). Upper triangle excludes future tokens.
    return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)


class CausalBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        xz = self.ln1(x)
        t = x.shape[1]
        attn_mask = _causal_bool_mask(t, x.device)
        y, _ = self.attn(xz, xz, xz, attn_mask=attn_mask, need_weights=False)
        x = x + y

        xz2 = self.ln2(x)
        x = x + self.ff(xz2)
        return x


class CausalTransformer(nn.Module):
    """Stack of causal multi-head self-attention + FFN layers."""

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                CausalBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def _encode_tactile_sequences(
    tactile_encoder: nn.Module, tactile_raw: torch.Tensor
) -> torch.Tensor:
    """tactile_raw: (B, L, H, W) → (B, L, d_out)."""
    if tactile_raw.dim() != 4:
        raise ValueError(
            f"tactile_raw must be (B, L, H, W); got {tuple(tactile_raw.shape)}"
        )
    b, l, h, w = tactile_raw.shape
    x = tactile_raw.reshape(b * l, h, w)
    z = tactile_encoder(x)
    return z.reshape(b, l, -1)


class VisuoTactilePolicy(nn.Module):
    """512-d visual + 64-d tactile fused → causal Transformer → actions."""

    def __init__(
        self,
        d_visual: int = 512,
        d_tactile: int = 64,
        d_model: int = 512,
        n_layers: int = 3,
        n_heads: int = 8,
        dropout: float = 0.1,
        action_dim: int = 7,
        tactile_encoder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.tactile_encoder = tactile_encoder if tactile_encoder is not None else nn.Identity()
        self.fusion_proj = nn.Linear(d_visual + d_tactile, d_model)
        self.transformer = CausalTransformer(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.action_head = nn.Linear(d_model, action_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.fusion_proj.weight)
        if self.fusion_proj.bias is not None:
            nn.init.zeros_(self.fusion_proj.bias)
        nn.init.xavier_uniform_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)

    def forward(
        self, z_visual: torch.Tensor, z_tactile_raw: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if z_visual.dim() != 3:
            z_visual = z_visual.unsqueeze(1)
        b, t, _ = z_visual.shape
        if z_tactile_raw is None:
            raise ValueError("z_tactile_raw required for VisuoTactilePolicy")

        if z_tactile_raw.dim() == 3:
            if z_tactile_raw.shape[1:] != (12, 64):
                raise ValueError(
                    f"3D tactile must be (B, 12, 64); got {tuple(z_tactile_raw.shape)}"
                )
            z_tactile_raw = z_tactile_raw.unsqueeze(1).expand(b, t, 12, 64)
        if z_tactile_raw.dim() != 4:
            raise ValueError(
                f"tactile must be (B, T, 12, 64) after prep; got {tuple(z_tactile_raw.shape)}"
            )
        if z_tactile_raw.shape[0] != b or z_tactile_raw.shape[1] != t:
            raise ValueError(
                f"Tactile (B,T) mismatch visual: visual ({b},{t}), tactile {tuple(z_tactile_raw.shape[:2])}"
            )
        if z_tactile_raw.shape[2:] != (12, 64):
            raise ValueError(f"Tactile H×W must be 12×64; got {tuple(z_tactile_raw.shape[2:])}")

        tac = _encode_tactile_sequences(self.tactile_encoder, z_tactile_raw)

        z = torch.cat([z_visual, tac], dim=-1)
        h = self.fusion_proj(z)
        h = self.transformer(h)
        return self.action_head(h)


class VisionOnlyPolicy(nn.Module):
    """Vision-only causal Transformer."""

    def __init__(
        self,
        d_visual: int = 512,
        d_model: int = 512,
        n_layers: int = 3,
        n_heads: int = 8,
        dropout: float = 0.1,
        action_dim: int = 7,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(d_visual, d_model)
        self.transformer = CausalTransformer(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.action_head = nn.Linear(d_model, action_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
        nn.init.xavier_uniform_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)

    def forward(self, z_visual: torch.Tensor, _: Optional[torch.Tensor] = None) -> torch.Tensor:
        if z_visual.dim() != 3:
            z_visual = z_visual.unsqueeze(1)
        h = self.proj(z_visual)
        h = self.transformer(h)
        return self.action_head(h)


class TactileOnlyPolicy(nn.Module):
    """Tactile-only policy."""

    def __init__(
        self,
        d_tactile_enc: int = 64,
        d_model: int = 512,
        n_layers: int = 3,
        n_heads: int = 8,
        dropout: float = 0.1,
        action_dim: int = 7,
        tactile_encoder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.tactile_encoder = tactile_encoder if tactile_encoder is not None else nn.Identity()
        self.proj = nn.Linear(d_tactile_enc, d_model)
        self.transformer = CausalTransformer(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.action_head = nn.Linear(d_model, action_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
        nn.init.xavier_uniform_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)

    def forward(self, _: torch.Tensor, tactile_raw: torch.Tensor) -> torch.Tensor:
        if tactile_raw.dim() == 3:
            if tactile_raw.shape[1:] != (12, 64):
                raise ValueError(
                    f"3D tactile must be (B, 12, 64); got {tuple(tactile_raw.shape)}"
                )
            tactile_raw = tactile_raw.unsqueeze(1)
        if tactile_raw.dim() != 4 or tactile_raw.shape[2:] != (12, 64):
            raise ValueError(f"Expected tactile_raw (B, T, 12, 64); got {tuple(tactile_raw.shape)}")
        tac = _encode_tactile_sequences(self.tactile_encoder, tactile_raw)
        h = self.proj(tac)
        h = self.transformer(h)
        return self.action_head(h)


if __name__ == "__main__":
    from encoders import TactileEncoder  # pylint: disable=import-error

    b, t = 8, 5
    zv = torch.randn(b, t, 512)
    tac_enc = TactileEncoder()

    vt = VisuoTactilePolicy(tactile_encoder=tac_enc)
    zraw = torch.randn(b, t, 12, 64)
    print("[policy] VT out:", vt(zv, zraw).shape)

    vo = VisionOnlyPolicy()
    print("[policy] Vision-only out:", vo(zv, None).shape)

    to = TactileOnlyPolicy(tactile_encoder=tac_enc)
    print("[policy] Tactile-only out:", to(torch.zeros(1), zraw).shape)
