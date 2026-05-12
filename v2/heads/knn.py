"""k-nearest-neighbor retrieval baseline (no-train).

Given a populated memory bank, predicts the pooled mean of the top-k
retrieved actions. Mirrors the kNN baseline in ``src/evaluate.py`` so
the v2 matrix can report it next to the trainable heads.

The module carries a single dummy 1-element parameter so that
``torch.optim.AdamW(model.parameters(), ...)`` does not raise on an
empty parameter list — training the kNN head is a no-op (val_mse is
constant across epochs because the prediction does not depend on the
parameter), which is exactly the desired behavior.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..legacy.memory import FeatureMemoryBank


class KNNRetrievalHead(nn.Module):
    def __init__(self, memory: FeatureMemoryBank, k: int = 3) -> None:
        super().__init__()
        self.memory = memory
        self.k = int(k)
        # Single inert parameter so that torch.optim.AdamW does not raise
        # on an empty parameter list. The forward output never depends on
        # this parameter, so its gradient is identically zero.
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, obs_t: torch.Tensor, obs_next: torch.Tensor) -> torch.Tensor:
        retr, mk = self.memory.retrieve_batch(obs_t, obs_next, k=self.k, tau_min=None, exclude_idx=None)
        denom = mk.sum(dim=1, keepdim=True).clamp(min=1).float()
        summed = (retr.to(obs_t.device) * mk.to(obs_t.device).unsqueeze(-1).float()).sum(dim=1)
        return summed / denom + 0.0 * self._dummy.sum()
