"""Inverse-dynamics heads, generalized to feature-vector inputs.

Forked from ``src/models.py`` so the original stays bit-identical and the
autoresearch ``src/train.py --condition raid --n_demos 25`` baseline keeps
reproducing val_mse ~ 0.397.

Differences from the source:
  * ``DirectMLP`` and ``RAIDDecoder`` accept any input width via the
    ``obs_dim`` constructor argument. The autoresearch-validated gating,
    prior dropout, and prior noise behavior are preserved.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DirectMLP(nn.Module):
    """Two-layer MLP head: predict ``a_t`` from ``concat(obs_t, obs_next)``."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        od = int(obs_dim)
        ad = int(action_dim)
        self.obs_dim = od
        self.action_dim = ad
        self.net = nn.Sequential(
            nn.Linear(od * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, ad),
        )

    def forward(self, obs_t: torch.Tensor, obs_next: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs_t, obs_next], dim=-1))


class RAIDDecoder(nn.Module):
    """Gated retrieval-augmented inverse-dynamics head.

    Output is ``g * direct(obs_t, obs_next) + (1 - g) * prior``. During
    training the prior is dropout-masked (p=0.5) and Gaussian-jittered
    (sigma=0.1) so the decoder cannot blindly copy retrieval. These
    settings are the autoresearch-validated values from
    ``src/models.py`` iterations 6-7.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        prior_dropout: float = 0.5,
        prior_noise_std: float = 0.1,
    ) -> None:
        super().__init__()
        od = int(obs_dim)
        ad = int(action_dim)
        self.obs_dim = od
        self.action_dim = ad
        sx = od * 2
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
        self.prior_drop = nn.Dropout(p=prior_dropout)
        self._prior_noise_std = float(prior_noise_std)

    def forward(self, obs_t: torch.Tensor, obs_next: torch.Tensor, a_prior: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs_t, obs_next], dim=-1)
        g = torch.sigmoid(self.gate_lin(x))
        d = self.direct(x)

        if self.training:
            ap = self.prior_drop(a_prior)
            ap = ap + torch.randn_like(ap) * self._prior_noise_std
        else:
            ap = a_prior

        return g * d + (1.0 - g) * ap
