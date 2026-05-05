"""LIBERO adapter via ``openvla/modified_libero_rlds``.

The OpenVLA-aligned RLDS exposes 7-D EE-delta + gripper actions and
8-D state at 256x256, which lets us share the RoboMimic action head
without per-dataset projection layers. Each suite is loaded as a
single HF dataset; transitions are flattened across demos in
trajectory-order, matching the IDM ``(obs_t, obs_next)`` recipe.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .stats import ActionStats, compute_action_stats


LIBERO_REPO = "openvla/modified_libero_rlds"
LIBERO_SUITES = ("libero_spatial", "libero_object", "libero_goal")
LIBERO_CONFIGS = {
    "libero_spatial": "libero_spatial_no_noops",
    "libero_object": "libero_object_no_noops",
    "libero_goal": "libero_goal_no_noops",
}
LIBERO_IMAGE_KEYS = ("image", "agentview_rgb", "agentview_image")


@dataclass(frozen=True)
class LiberoSpec:
    suite: str  # libero_spatial / libero_object / libero_goal
    modality: str = "image"  # image only

    @property
    def name(self) -> str:
        return self.suite


def suite_config_name(suite: str) -> str:
    if suite in LIBERO_CONFIGS:
        return LIBERO_CONFIGS[suite]
    if suite in LIBERO_CONFIGS.values():
        return suite
    raise ValueError(f"unknown LIBERO suite {suite!r}")


def _load_hf_dataset(suite: str, cache_dir: Path):
    from datasets import load_dataset

    return load_dataset(LIBERO_REPO, name=suite_config_name(suite), cache_dir=str(cache_dir), split="train")


def _steps_for_row(row: Any) -> Any:
    steps = row.get("steps")
    if steps is None:
        raise KeyError(
            "LIBERO RLDS row is missing the required `steps` field. "
            "This loader is verified against the dataset features.json schema."
        )
    return steps


def _state_from_observation(obs: dict[str, Any]) -> np.ndarray:
    for key in ("state", "robot_state"):
        if key in obs and obs[key] is not None:
            return np.asarray(obs[key], dtype=np.float32).reshape(-1)
    raise KeyError("LIBERO observation is missing `state` / `robot_state`")


def _image_from_observation(obs: dict[str, Any]) -> np.ndarray:
    for key in LIBERO_IMAGE_KEYS:
        if key in obs and obs[key] is not None:
            return np.asarray(obs[key], dtype=np.uint8)
    raise KeyError(f"LIBERO observation is missing all image keys {LIBERO_IMAGE_KEYS}")


def _demo_key_for_row(row: Any, episode_idx: int) -> str:
    meta = row.get("episode_metadata")
    if isinstance(meta, dict):
        file_path = meta.get("file_path")
        if file_path:
            return str(file_path)
    return f"demo_{episode_idx:05d}"


def episode_layout(ds: Any, max_demos: int | None = None) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    n = len(ds) if max_demos is None else min(len(ds), int(max_demos))
    for i in range(n):
        row = ds[i]
        steps = _steps_for_row(row)
        out.append((_demo_key_for_row(row, i), len(steps)))
    return out


def iter_episode_frames(ds: Any, max_demos: int | None = None) -> Any:
    n = len(ds) if max_demos is None else min(len(ds), int(max_demos))
    for i in range(n):
        row = ds[i]
        for step in _steps_for_row(row):
            yield _image_from_observation(step["observation"])


def build_transitions_from_rlds(ds: Any, max_demos: int | None = None) -> list[dict[str, np.ndarray]]:
    """Convert RLDS rows into a list of demo dicts ``{s, a, image}``.

    Each row of ``ds`` corresponds to one full episode; we read the
    ``steps`` field and lift it into per-timestep numpy arrays.
    """
    out: list[dict[str, np.ndarray]] = []
    n = len(ds) if max_demos is None else min(len(ds), int(max_demos))
    for i in range(n):
        row = ds[i]
        steps = _steps_for_row(row)
        actions = []
        states = []
        images = []
        for step in steps:
            obs = step["observation"]
            actions.append(np.asarray(step["action"], dtype=np.float64))
            states.append(_state_from_observation(obs))
            images.append(_image_from_observation(obs))
        out.append(
            {
                "s": np.stack(states, axis=0).astype(np.float32),
                "a": np.stack(actions, axis=0).astype(np.float32),
                "image": np.stack(images, axis=0),
                "_demo_key": np.asarray(_demo_key_for_row(row, i)),
            }
        )
    return out


def build_transitions(demos: list[dict[str, np.ndarray]]) -> list[tuple[np.ndarray, ...]]:
    """``(s_t, s_next, a_t, is_contact, demo_key, t)`` flat list."""
    out: list[tuple[np.ndarray, ...]] = []
    for d in demos:
        s = d["s"]
        a = d["a"]
        dk = str(np.asarray(d["_demo_key"]))
        T = s.shape[0]
        for t in range(T - 1):
            g0 = float(a[t, 6]) if a.shape[1] > 6 else 0.0
            g1 = float(a[t + 1, 6]) if a.shape[1] > 6 else 0.0
            contact = abs(g1 - g0) > 0.1
            out.append((s[t].copy(), s[t + 1].copy(), a[t].copy(), bool(contact), dk, int(t)))
    return out


class LiberoTransitionDataset(Dataset):
    """LIBERO transitions sharing the common ``obs_t / obs_next / action`` schema."""

    def __init__(
        self,
        spec: LiberoSpec,
        demos: list[dict[str, np.ndarray]],
        transitions: list[tuple[np.ndarray, ...]],
        action_stats: ActionStats,
    ) -> None:
        self.spec = spec
        self.demos = demos
        self.transitions = transitions
        self.action_dim = 7
        self.stats = action_stats
        self._q01 = torch.as_tensor(action_stats.q01, dtype=torch.float32)
        self._q99 = torch.as_tensor(action_stats.q99, dtype=torch.float32)
        self._demo_index = {str(np.asarray(d["_demo_key"])): i for i, d in enumerate(demos)}
        self._obs_windows = self._build_obs_windows()

    def __len__(self) -> int:
        return len(self.transitions)

    def _norm_action(self, a: torch.Tensor) -> torch.Tensor:
        denom = (self._q99 - self._q01).clamp(min=1e-6)
        return (2.0 * (a - self._q01) / denom - 1.0).clamp(-1.0, 1.0)

    def _build_obs_windows(self) -> torch.Tensor:
        if not self.transitions:
            return torch.zeros(0, 4, 8, dtype=torch.float32)
        obs_t = [torch.as_tensor(tr[0], dtype=torch.float32) for tr in self.transitions]
        obs_next = [torch.as_tensor(tr[1], dtype=torch.float32) for tr in self.transitions]
        demo_rows: dict[str, list[int]] = {}
        for row_idx, tr in enumerate(self.transitions):
            demo_rows.setdefault(str(tr[4]), []).append(row_idx)
        windows = []
        for row_idx, tr in enumerate(self.transitions):
            demo_key = str(tr[4])
            t = int(tr[5])
            rows = demo_rows[demo_key]
            windows.append(
                torch.stack(
                    [
                        obs_t[rows[max(0, t - 2)]],
                        obs_t[rows[max(0, t - 1)]],
                        obs_t[row_idx],
                        obs_next[row_idx],
                    ],
                    dim=0,
                )
            )
        return torch.stack(windows, dim=0)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s_t_np, s_n_np, a_np, contact, demo_key, t = self.transitions[idx]
        s_t = torch.as_tensor(s_t_np, dtype=torch.float32)
        s_n = torch.as_tensor(s_n_np, dtype=torch.float32)
        a = torch.as_tensor(a_np, dtype=torch.float32)
        return {
            "obs_t": s_t,
            "obs_next": s_n,
            "s_t": s_t,
            "s_next": s_n,
            "obs_window": self._obs_windows[int(idx)],
            "action": self._norm_action(a),
            "action_raw": a,
            "is_contact": torch.tensor(bool(contact), dtype=torch.bool),
            "idx": torch.tensor(idx, dtype=torch.long),
            "demo_key": demo_key,
            "t": torch.tensor(int(t), dtype=torch.long),
            "dataset_id": torch.tensor(0, dtype=torch.long),
        }

    def fetch_frames(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        _, _, _, _, demo_key, t = self.transitions[idx]
        d = self.demos[self._demo_index[str(demo_key)]]
        if d.get("image") is None:
            raise RuntimeError("no images materialized; load with keep_image_in_memory")
        return d["image"][int(t)], d["image"][int(t) + 1]


def make_train_val(
    spec: LiberoSpec,
    n_demos: int,
    cache_dir: Path,
    train_frac: float = 0.8,
    action_stats: ActionStats | None = None,
    keep_image_in_memory: bool = True,
) -> tuple[LiberoTransitionDataset, LiberoTransitionDataset, ActionStats]:
    raw = _load_hf_dataset(spec.suite, cache_dir)
    demos = build_transitions_from_rlds(raw, max_demos=n_demos)
    if not keep_image_in_memory:
        for d in demos:
            d["image"] = None  # type: ignore[assignment]

    n_train = int(np.ceil(train_frac * len(demos)))
    train_demos = demos[:n_train]
    val_demos = demos[n_train:]
    train_triples = build_transitions(train_demos)
    val_triples = build_transitions(val_demos)

    if action_stats is None:
        train_actions = np.stack([tr[2] for tr in train_triples], axis=0)
        action_stats = compute_action_stats(train_actions)

    train_ds = LiberoTransitionDataset(spec, train_demos, train_triples, action_stats)
    val_ds = LiberoTransitionDataset(spec, val_demos, val_triples, action_stats)
    return train_ds, val_ds, action_stats
