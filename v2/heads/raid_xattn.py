"""Cross-attention RAID decoder.

Drop-in replacement for the mean-pool RAID prior. Instead of averaging
the top-k retrieved actions and feeding the resulting vector through a
gate, we score each retrieved action against the current ``(obs_t,
obs_next)`` query via dot-product attention and weight the actions in
their original 7-D space. The gated direct branch and prior dropout +
noise schedule that won the legacy autoresearch (RAIDDecoder iter-7) are
preserved verbatim.

Why this can beat mean-pool:

* When the k=3 retrieved transitions disagree, mean-pool produces a
  blurry prior that the decoder must override on every prediction.
  Attention can put 90 %+ of the weight on a single neighbour when one
  is genuinely more relevant than the others, so the prior is sharp
  rather than averaged.
* The attention scores are computed in a small ``d_model``-dimensional
  space but the weighted sum stays in action space, so the prior
  remains interpretable and the gate can still trade direct vs prior
  per-DOF.

Ported (with minor naming changes) from ``src/models.RAIDDecoderCrossAttn``
on the classmate's main branch, where it produces a 6× lift over the
mean-pool baseline on LIBERO image features.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RAIDCrossAttnDecoder(nn.Module):
    """Cross-attention prior + gated direct branch."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 7,
        k: int = 3,
        d_model: int = 64,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        prior_dropout: float = 0.5,
        prior_noise_std: float = 0.1,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.k = int(k)
        self.d_model = int(d_model)
        self._sqrt_d = math.sqrt(d_model)

        sx = self.obs_dim * 2
        # Lightweight projections used for attention scoring only.
        self.q_proj = nn.Linear(sx, d_model)
        self.k_proj = nn.Linear(action_dim, d_model)

        # Gate + direct branch — same structure as the winning legacy
        # ``RAIDDecoder``.
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

        # Prior regularisation (autoresearch iter-6/7 verbatim).
        self.prior_drop = nn.Dropout(p=float(prior_dropout))
        self._prior_noise_std = float(prior_noise_std)

    def forward(
        self,
        obs_t: torch.Tensor,
        obs_next: torch.Tensor,
        retrieved_actions: torch.Tensor,
        retrieved_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict ``a_t`` from a transition + a (B, k, action_dim) bag of retrieved actions.

        Args:
            obs_t: (B, obs_dim)
            obs_next: (B, obs_dim)
            retrieved_actions: (B, k, action_dim) — typically the top-k
                cosine-retrieved neighbours from ``FeatureMemoryBank``.
            retrieved_mask: optional (B, k) boolean mask; ``True`` marks
                a valid retrieved row, ``False`` marks padding (when the
                bank had fewer than k candidates).
        """
        x = torch.cat([obs_t, obs_next], dim=-1)
        q = self.q_proj(x).unsqueeze(1)               # (B, 1, d_model)
        k = self.k_proj(retrieved_actions)             # (B, k, d_model)
        scores = (q @ k.transpose(-2, -1)) / self._sqrt_d  # (B, 1, k)
        if retrieved_mask is not None:
            invalid = ~retrieved_mask.bool().unsqueeze(1)  # (B, 1, k)
            scores = scores.masked_fill(invalid, float("-inf"))
        # If a row has zero valid neighbours (all -inf) the softmax would
        # NaN — fall back to a uniform attention so the prior is still
        # well-defined and equal to the mean of the (zero-valued) actions.
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)

        a_prior = (attn @ retrieved_actions).squeeze(1)  # (B, action_dim)

        gate = torch.sigmoid(self.gate_lin(x))
        direct = self.direct(x)

        if self.training:
            ap = self.prior_drop(a_prior)
            ap = ap + torch.randn_like(ap) * self._prior_noise_std
        else:
            ap = a_prior

        return gate * direct + (1.0 - gate) * ap
