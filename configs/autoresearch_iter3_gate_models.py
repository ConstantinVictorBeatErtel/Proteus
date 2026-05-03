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
    """Iter 3: gated blend — trust parametric inverse vs pooled prior."""

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

    def forward(self, s_t: torch.Tensor, s_next: torch.Tensor, a_prior: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s_t, s_next], dim=-1)
        g = torch.sigmoid(self.gate_lin(x))
        d = self.direct(x)
        return g * d + (1.0 - g) * a_prior


if __name__ == "__main__":
    B, D, A = 4, 19, 7
    dm = DirectMLP(D, A)
    rd = RAIDDecoder(D, A)
    st = torch.randn(B, D)
    sn = torch.randn(B, D)
    ap = torch.randn(B, A)
    print(dm(st, sn).shape, rd(st, sn, ap).shape)
