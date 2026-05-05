"""Inverse dynamics MLP heads."""

from __future__ import annotations

import torch
import torch.nn as nn


class DirectMLP(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        sd = int(state_dim)
        ad = int(action_dim)
        self.state_dim = sd
        self.action_dim = ad
        self.net = nn.Sequential(
            nn.Linear(sd * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, ad),
        )

    def forward(self, s_t: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s_t, s_next], dim=-1))


class RAIDDecoder(nn.Module):
    """Gated RAID + prior dropout + train-time Gaussian jitter on pooled prior."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        sd = int(state_dim)
        ad = int(action_dim)
        self.state_dim = sd
        self.action_dim = ad
        sx = sd * 2
        self.gate_lin = nn.Linear(sx, ad)
        self.direct = nn.Sequential(
            nn.Linear(sx, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, ad),
        )
        self.prior_drop = nn.Dropout(p=0.5)
        self._prior_noise_std = 0.1

    def forward(self, s_t: torch.Tensor, s_next: torch.Tensor, a_prior: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s_t, s_next], dim=-1)
        g = torch.sigmoid(self.gate_lin(x))
        d = self.direct(x)

        if self.training:
            ap = self.prior_drop(a_prior)
            ap = ap + torch.randn_like(ap, device=ap.device, dtype=ap.dtype) * self._prior_noise_std
        else:
            ap = a_prior
        return g * d + (1.0 - g) * ap


class RAIDDecoderCrossAttn(nn.Module):
    """
    Gated RAID decoder where mean-pooling of retrieved actions is replaced by
    cross-attention weighting.  Attention scores are computed between a
    d_model-dimensional query (from the transition) and d_model-dimensional keys
    (from retrieved actions); the weights are then applied directly in action space
    so the prior stays interpretable.  Inherits the gate + prior-dropout + prior-noise
    regularisation that proved effective in the autoresearch (RAIDDecoder iter-7).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 7,
        k: int = 3,
        d_model: int = 64,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        import math as _math
        self._sqrt_d = _math.sqrt(d_model)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.k = int(k)
        self.d_model = int(d_model)

        sx = self.obs_dim * 2
        # Lightweight projections for attention scoring only.
        self.q_proj = nn.Linear(sx, d_model)
        self.k_proj = nn.Linear(action_dim, d_model)

        # Gate + direct branch — same structure as winning RAIDDecoder.
        self.gate_lin = nn.Linear(sx, action_dim)
        self.direct = nn.Sequential(
            nn.Linear(sx, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),
        )
        # Prior regularisation (proven in autoresearch iter-6/7).
        self.prior_drop = nn.Dropout(p=0.5)
        self._prior_noise_std = 0.1

        # Learnable per-DOF std for GRPO stage.
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.0))

    def forward(
        self,
        s_t: torch.Tensor,
        s_next: torch.Tensor,
        retrieved_actions: torch.Tensor,
        kv_key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s_t, s_next: (B, obs_dim)
            retrieved_actions: (B, k, action_dim)
            kv_key_padding_mask: optional (B, k), True = ignore this position (padding).
        Returns:
            out: (B, action_dim)
            attn_weights: (B, 1, k)
        """
        import torch.nn.functional as F
        x = torch.cat([s_t, s_next], dim=-1)  # (B, sx)

        # Attention weights over retrieved actions (scored in d_model space).
        q = self.q_proj(x).unsqueeze(1)                              # (B, 1, d_model)
        k = self.k_proj(retrieved_actions)                           # (B, k, d_model)
        scores = (q @ k.transpose(-2, -1)) / self._sqrt_d            # (B, 1, k)
        if kv_key_padding_mask is not None:
            scores = scores.masked_fill(kv_key_padding_mask.unsqueeze(1), float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)                     # (B, 1, k)

        # Weighted retrieved action — stays in action space (interpretable prior).
        a_prior_attn = (attn_weights @ retrieved_actions).squeeze(1)  # (B, action_dim)

        # Gate + direct branch.
        g = torch.sigmoid(self.gate_lin(x))
        d = self.direct(x)

        # Prior regularisation (dropout + noise, training only).
        if self.training:
            ap = self.prior_drop(a_prior_attn)
            ap = ap + torch.randn_like(ap) * self._prior_noise_std
        else:
            ap = a_prior_attn

        out = g * d + (1.0 - g) * ap
        return out, attn_weights

    def get_std(self) -> torch.Tensor:
        return self.log_std.exp().clamp(1e-4, 1.0)


if __name__ == "__main__":
    B, D, A = 4, 19, 7
    dm = DirectMLP(D, A)
    rd = RAIDDecoder(D, A)
    st = torch.randn(B, D)
    sn = torch.randn(B, D)
    ap = torch.randn(B, A)
    print(dm(st, sn).shape, rd(st, sn, ap).shape)
