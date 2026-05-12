"""OpenVLA-style q01/q99 action normalization.

Each dataset gets its own ``ActionStats`` (q01, q99, mean, std) computed
over its training split only; targets are mapped to ``[-1, 1]`` via
``2 * (a - q01) / (q99 - q01) - 1`` and clipped. This is the recipe used
by OpenVLA and adopted by RDT-1B / GR00T-N1 for cross-dataset mixing.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class ActionStats:
    q01: list[float]
    q99: list[float]
    mean: list[float]
    std: list[float]

    @property
    def action_dim(self) -> int:
        return len(self.q01)

    def to_tensors(self) -> dict[str, torch.Tensor]:
        return {
            "q01": torch.tensor(self.q01, dtype=torch.float32),
            "q99": torch.tensor(self.q99, dtype=torch.float32),
            "mean": torch.tensor(self.mean, dtype=torch.float32),
            "std": torch.tensor(self.std, dtype=torch.float32),
        }


def compute_action_stats(actions: np.ndarray, eps_std: float = 1e-6) -> ActionStats:
    """Compute q01/q99/mean/std over a ``(N, action_dim)`` array."""
    if actions.ndim != 2:
        raise ValueError(f"expected 2D actions, got shape {actions.shape}")
    q01 = np.quantile(actions, 0.01, axis=0).astype(np.float64)
    q99 = np.quantile(actions, 0.99, axis=0).astype(np.float64)
    mean = actions.mean(axis=0).astype(np.float64)
    std = np.maximum(actions.std(axis=0).astype(np.float64), eps_std)
    return ActionStats(
        q01=q01.tolist(),
        q99=q99.tolist(),
        mean=mean.tolist(),
        std=std.tolist(),
    )


def normalize_action(
    a: torch.Tensor,
    stats: ActionStats,
    mode: str = "q01_q99",
    clip: bool = True,
) -> torch.Tensor:
    """Normalize a raw action tensor into ``[-1, 1]`` (or z-score)."""
    if mode == "q01_q99":
        q01 = torch.as_tensor(stats.q01, dtype=a.dtype, device=a.device)
        q99 = torch.as_tensor(stats.q99, dtype=a.dtype, device=a.device)
        denom = (q99 - q01).clamp(min=1e-6)
        x = 2.0 * (a - q01) / denom - 1.0
        if clip:
            x = x.clamp(-1.0, 1.0)
        return x
    if mode == "zscore":
        m = torch.as_tensor(stats.mean, dtype=a.dtype, device=a.device)
        s = torch.as_tensor(stats.std, dtype=a.dtype, device=a.device).clamp(min=1e-6)
        return (a - m) / s
    raise ValueError(f"unknown mode={mode!r}")


def denormalize_action(
    a: torch.Tensor,
    stats: ActionStats,
    mode: str = "q01_q99",
) -> torch.Tensor:
    if mode == "q01_q99":
        q01 = torch.as_tensor(stats.q01, dtype=a.dtype, device=a.device)
        q99 = torch.as_tensor(stats.q99, dtype=a.dtype, device=a.device)
        denom = (q99 - q01).clamp(min=1e-6)
        return ((a + 1.0) / 2.0) * denom + q01
    if mode == "zscore":
        m = torch.as_tensor(stats.mean, dtype=a.dtype, device=a.device)
        s = torch.as_tensor(stats.std, dtype=a.dtype, device=a.device).clamp(min=1e-6)
        return a * s + m
    raise ValueError(f"unknown mode={mode!r}")


def save_stats_registry(stats_by_dataset: dict[str, ActionStats], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: asdict(v) for k, v in stats_by_dataset.items()}
    p.write_text(json.dumps(payload, indent=2))
    return p


def load_stats_registry(path: str | Path) -> dict[str, ActionStats]:
    p = Path(path)
    if not p.is_file():
        return {}
    raw = json.loads(p.read_text())
    return {k: ActionStats(**v) for k, v in raw.items()}
