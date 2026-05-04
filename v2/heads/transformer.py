"""Causal Transformer IDM over a 4-frame observation window.

Sequence layout: ``(obs_{t-2}, obs_{t-1}, obs_t, obs_{t+1})``. The action
``a_t`` is read from the position of ``obs_{t+1}`` (the position that has
the full causal cone over the past observations and the future frame).
This matches the LAPA / GR00T-N1 latent-action recipe of conditioning on
exactly two frames around the action, while letting past context
disambiguate motion in image-feature regimes.

For low-dim or single-frame regimes, callers can pass duplicate frames
for the past tokens; the head still operates correctly.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class TransformerIDM(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 7,
        seq_len: int = 4,
        d_model: int = 384,
        n_layers: int = 4,
        n_heads: int = 6,
        dim_ff: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.seq_len = int(seq_len)
        self.d_model = int(d_model)

        self.input_proj = nn.Linear(self.obs_dim, self.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.seq_len, self.d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, self.action_dim)

        # Causal mask: token i may attend to tokens 0..i.
        mask = torch.full((self.seq_len, self.seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("_causal_mask", mask, persistent=False)

    def _build_sequence(self, obs_window: torch.Tensor) -> torch.Tensor:
        # obs_window: (B, seq_len, obs_dim)
        if obs_window.shape[1] != self.seq_len:
            raise ValueError(
                f"expected window length {self.seq_len}, got {obs_window.shape[1]}"
            )
        x = self.input_proj(obs_window) * math.sqrt(self.d_model)
        return x + self.pos_emb

    def forward(self, obs_window: torch.Tensor) -> torch.Tensor:
        """Predict ``a_t`` from a (B, seq_len, obs_dim) window."""
        x = self._build_sequence(obs_window)
        h = self.encoder(x, mask=self._causal_mask, is_causal=True)
        h = self.norm(h)
        return self.head(h[:, -1])  # readout from the o_{t+1} position

    def forward_pair(self, obs_t: torch.Tensor, obs_next: torch.Tensor) -> torch.Tensor:
        """Convenience: tile a 2-frame pair into a 4-frame window."""
        if self.seq_len != 4:
            raise RuntimeError("forward_pair only valid when seq_len == 4")
        window = torch.stack([obs_t, obs_t, obs_t, obs_next], dim=1)
        return self.forward(window)
