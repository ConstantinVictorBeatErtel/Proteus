"""
RoboMimic Lift HDF5 loader, transition tuples, normalization, PyTorch Dataset.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


# Observation keys concatenated into a single low-dim state vector s_t.
OBS_KEYS = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]

# Project root when running scripts from ~/proteus/raid/
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def state_dim_from_keys() -> int:
    dims = {"object": 10, "robot0_eef_pos": 3, "robot0_eef_quat": 4, "robot0_gripper_qpos": 2}
    return sum(dims[k] for k in OBS_KEYS)


DEFAULT_HDF5 = PROJECT_ROOT / "data/lift/ph/low_dim_v141.hdf5"


def load_demos(hdf5_path: str | Path, n_demos: int) -> list[dict[str, np.ndarray]]:
    """
    Load the first ``n_demos`` demonstrations (sorted by demo index).

    Returns a list of dicts with keys:
      - ``s``: (T, state_dim)
      - ``a``: (T, action_dim)
    """
    hdf5_path = Path(hdf5_path)
    demos: list[dict[str, np.ndarray]] = []
    state_dim = state_dim_from_keys()

    with h5py.File(hdf5_path, "r") as f:
        data_grp = f["data"]
        all_keys = sorted(data_grp.keys(), key=lambda x: int(x.split("_")[1]))
        subset = all_keys[:n_demos]

        for dk in subset:
            g = data_grp[dk]
            obs = g["obs"]
            parts = []
            for key in OBS_KEYS:
                parts.append(np.asarray(obs[key], dtype=np.float64))
            s = np.concatenate(parts, axis=-1)
            assert s.ndim == 2 and s.shape[1] == state_dim

            a = np.asarray(g["actions"], dtype=np.float64)

            demos.append({"s": s, "a": a})

    return demos


def build_transitions(demos: list[dict[str, np.ndarray]]) -> list[tuple[np.ndarray, ...]]:
    """
    Build (s_t, s_next, a_t, is_contact) transitions.

    ``is_contact``: ``|a_{t+1}[6]-a_{t}[6]| > 0.1`` comparing consecutive gripper command values.

    Uses raw vectors; normalization is applied in ``TransitionDataset``.
    """
    triples: list[tuple[np.ndarray, ...]] = []

    for demo in demos:
        s = demo["s"]
        a = demo["a"]
        T = s.shape[0]
        for t in range(T - 1):
            s_t = s[t].copy()
            s_next = s[t + 1].copy()
            a_t = a[t].copy()
            g0 = float(a[t, 6])
            g1 = float(a[t + 1, 6])
            contact = abs(g1 - g0) > 0.1

            triples.append((s_t, s_next, a_t, contact))

    return triples


def split_demo_indices(n_demos: int, train_frac: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    """80/20 split by demo index: train gets the first ceil(train_frac*n) demos."""
    n_train = int(math.ceil(train_frac * n_demos))
    all_idx = np.arange(n_demos)
    train_demos = all_idx[:n_train]
    val_demos = all_idx[n_train:]
    return train_demos, val_demos


def compute_norm_stats(
    transitions: list[tuple[np.ndarray, ...]],
    eps_std: float = 1e-6,
) -> dict[str, np.ndarray]:
    """Mean/std over states (s_t && s_next) and actions from the training transition list."""

    sts = []
    sns = []
    acts = []

    for s_t, s_next, a_t, _ in transitions:
        sts.append(s_t)
        sns.append(s_next)
        acts.append(a_t)

    sts = np.stack(sts, axis=0)
    sns = np.stack(sns, axis=0)
    acts = np.stack(acts, axis=0)

    all_states = np.concatenate([sts, sns], axis=0)
    state_mean = all_states.mean(axis=0).astype(np.float32)
    state_std = np.maximum(all_states.std(axis=0).astype(np.float32), eps_std)

    action_mean = acts.mean(axis=0).astype(np.float32)
    action_std = np.maximum(acts.std(axis=0).astype(np.float32), eps_std)

    return {
        "state_mean": state_mean,
        "state_std": state_std,
        "action_mean": action_mean,
        "action_std": action_std,
    }


def transitions_for_split(
    demos: list[dict[str, np.ndarray]], demo_indices: np.ndarray | list[int]
) -> list[tuple[np.ndarray, ...]]:
    sub = [demos[int(i)] for i in demo_indices]
    return build_transitions(sub)


def norm_stats_path(n_demos: int) -> Path:
    """Separate stats per ``n_demos`` so multi-scale evaluation stays consistent."""
    return PROJECT_ROOT / "configs" / f"norm_stats_{n_demos}demos.pt"


def save_norm_stats(stats: dict[str, np.ndarray], n_demos: int) -> Path:
    path = norm_stats_path(n_demos)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, path)
    return path


def load_norm_stats(n_demos: int) -> dict[str, np.ndarray]:
    path = norm_stats_path(n_demos)
    if not path.is_file():
        raise FileNotFoundError(f"Missing norm stats: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


class TransitionDataset(Dataset):
    """
    Samples normalized (s_t, s_next, action, is_contact) plus dataset index ``idx``.
    """

    def __init__(
        self,
        transitions_raw: list[tuple[np.ndarray, ...]],
        norm_stats: dict[str, Any],
        train_mean_action: torch.Tensor | None = None,
    ):
        """
        Args:
          transitions_raw: list of raw (s_t, s_next, a_t, contact) tuples.
          norm_stats: dict with numpy ``state_*`` and ``action_*`` arrays.
          train_mean_action: optional normalized mean action tensor (shape [action_dim]).
                             If omitted, recomputed from this split (use training split only externally).
        """
        self.transitions = transitions_raw
        self.sm = torch.as_tensor(norm_stats["state_mean"], dtype=torch.float32)
        self.ss = torch.as_tensor(norm_stats["state_std"], dtype=torch.float32)
        self.am = torch.as_tensor(norm_stats["action_mean"], dtype=torch.float32)
        self.as_ = torch.as_tensor(norm_stats["action_std"], dtype=torch.float32)

        self.state_dim = int(self.sm.numel())
        self.action_dim = int(self.am.numel())

        if train_mean_action is not None:
            self.register_mean = train_mean_action.clone()
        else:
            aa = [(self._norm_action(torch.as_tensor(tr[2], dtype=torch.float32))) for tr in transitions_raw]
            self.register_mean = torch.stack(aa, dim=0).mean(dim=0)

        self.contact_flags = torch.tensor([bool(tr[3]) for tr in transitions_raw], dtype=torch.bool)

    def _norm_state(self, s: torch.Tensor) -> torch.Tensor:
        return (s - self.sm) / self.ss

    def _norm_action(self, a: torch.Tensor) -> torch.Tensor:
        return (a - self.am) / self.as_

    def denorm_action(self, a_norm: torch.Tensor) -> torch.Tensor:
        return a_norm * self.as_ + self.am

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, index: int) -> dict[str, Any]:
        s_t, s_next, a_t, _ = self.transitions[index]
        s_t = torch.as_tensor(s_t, dtype=torch.float32)
        s_next = torch.as_tensor(s_next, dtype=torch.float32)
        a_t = torch.as_tensor(a_t, dtype=torch.float32)

        return {
            "s_t": self._norm_state(s_t),
            "s_next": self._norm_state(s_next),
            "action": self._norm_action(a_t),
            "is_contact": self.contact_flags[index],
            "idx": torch.tensor(index, dtype=torch.long),
        }


def make_train_val(
    n_demos: int,
    hdf5_path: Path | str | None = None,
    train_frac: float = 0.8,
) -> tuple[
    TransitionDataset,
    TransitionDataset,
    dict[str, np.ndarray],
]:
    """Build normalized train / val TransitionDatasets plus norm stats."""

    path = DEFAULT_HDF5 if hdf5_path is None else Path(hdf5_path)
    demos = load_demos(path, n_demos)
    train_didx, val_didx = split_demo_indices(n_demos, train_frac=train_frac)

    train_triples = transitions_for_split(demos, train_didx)
    val_triples = transitions_for_split(demos, val_didx)

    stats = compute_norm_stats(train_triples)
    stats_path = save_norm_stats(stats, n_demos)
    print(f"[data] Saved normalization stats → {stats_path}")

    train_mean_norm = TransitionDataset(train_triples, stats).register_mean.detach()

    train_ds = TransitionDataset(train_triples, stats, train_mean_action=train_mean_norm)
    val_ds = TransitionDataset(val_triples, stats, train_mean_action=train_mean_norm)

    return train_ds, val_ds, stats


if __name__ == "__main__":
    print("[data.__main__] sanity check …")
    tr, va, st = make_train_val(200)
    ex = tr[0]
    print(f"Train len={len(tr)}  Val len={len(va)}")
    print(f"Example keys={list(ex.keys())}")
    print(f"s_t shape={tuple(ex['s_t'].shape)}  s_next={tuple(ex['s_next'].shape)}  action={tuple(ex['action'].shape)}")
    print(f"state_dim={tr.state_dim}  action_dim={tr.action_dim}")
    print("[data.__main__] OK")

