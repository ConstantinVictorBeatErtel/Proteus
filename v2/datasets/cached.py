"""Feature-cached dataset wrapper.

Reads cached CLS features written by ``v2.features`` and exposes the
common ``{obs_t, obs_next, action, ...}`` dict schema. The cached
features are indexed in trajectory order (one row per timestep,
flattened across demos in HDF5/RLDS demo-key order). The wrapper
walks the same demos to reconstruct ``(demo_key, t)`` for each
transition so visualization can still pull raw RGB frames from the
underlying HDF5 / RLDS source.

Implementation status: this module ships the **RoboMimic** branch
end-to-end and a **placeholder** LIBERO branch that raises with a
clear pointer when called. The LIBERO RLDS schema differs across
``openvla/modified_libero_rlds`` revisions, so the agent that runs on
Colab must verify the schema before this branch is enabled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from ..runtime.drive import data_root, features_root
from .robomimic import IMAGE_OBS_KEY, RoboMimicSpec, hdf5_path_for, read_frames
from .stats import ActionStats, compute_action_stats


@dataclass
class _Transition:
    feat_t_idx: int  # row in the cached feature tensor
    feat_next_idx: int
    action: np.ndarray
    is_contact: bool
    demo_key: str
    t: int


def _load_features(dataset_name: str, encoder: str) -> torch.Tensor:
    from safetensors.torch import load_file

    target = features_root() / f"{dataset_name}_{encoder}_cls.safetensors"
    if not target.is_file():
        raise FileNotFoundError(
            f"missing cached features: {target}\n"
            "Run ``python3 -m v2.features --encoders {0} --datasets {1}`` first.".format(
                encoder, dataset_name
            )
        )
    return load_file(str(target))["features"].float()


class FeatureCachedDataset(Dataset):
    """Common wrapper around a (parent_dataset_name, encoder) feature cache."""

    def __init__(
        self,
        dataset_name: str,
        features: torch.Tensor,
        transitions: list[_Transition],
        action_stats: ActionStats,
        action_norm_mode: str = "q01_q99",
        frame_resolver=None,
    ) -> None:
        self.dataset_name = dataset_name
        self.features = features  # CPU fp32, [N_total_frames, feature_dim]
        self.transitions = transitions
        self.action_stats = action_stats
        self.action_norm_mode = action_norm_mode
        self.feature_dim = int(features.shape[1])
        self.action_dim = 7
        self._frame_resolver = frame_resolver
        self._q01 = torch.as_tensor(action_stats.q01, dtype=torch.float32)
        self._q99 = torch.as_tensor(action_stats.q99, dtype=torch.float32)
        self._a_mean = torch.as_tensor(action_stats.mean, dtype=torch.float32)
        self._a_std = torch.as_tensor(action_stats.std, dtype=torch.float32).clamp(min=1e-6)

    def __len__(self) -> int:
        return len(self.transitions)

    def _norm_action(self, a: torch.Tensor) -> torch.Tensor:
        if self.action_norm_mode == "zscore":
            return (a - self._a_mean) / self._a_std
        denom = (self._q99 - self._q01).clamp(min=1e-6)
        return (2.0 * (a - self._q01) / denom - 1.0).clamp(-1.0, 1.0)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        tr = self.transitions[idx]
        f_t = self.features[tr.feat_t_idx]
        f_n = self.features[tr.feat_next_idx]
        a = torch.as_tensor(tr.action, dtype=torch.float32)
        return {
            "obs_t": f_t,
            "obs_next": f_n,
            "s_t": f_t,
            "s_next": f_n,
            "action": self._norm_action(a),
            "action_raw": a,
            "is_contact": torch.tensor(bool(tr.is_contact), dtype=torch.bool),
            "idx": torch.tensor(idx, dtype=torch.long),
            "demo_key": tr.demo_key,
            "t": torch.tensor(int(tr.t), dtype=torch.long),
            "dataset_id": torch.tensor(0, dtype=torch.long),
        }

    def fetch_frames(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        if self._frame_resolver is None:
            raise RuntimeError(f"no frame resolver registered for {self.dataset_name!r}")
        tr = self.transitions[idx]
        return self._frame_resolver(tr.demo_key, tr.t)


def _robomimic_image_feature_offsets(hdf5_path: Path) -> tuple[list[str], dict[str, tuple[int, int]]]:
    """Walk the HDF5 in demo-key order to map each demo to ``(offset, length)``.

    The cached feature tensor is built by ``v2.features`` in the same
    demo-key sort order, so demo ``demo_0`` sits at rows
    ``[0, T_0)``, ``demo_1`` at ``[T_0, T_0 + T_1)``, and so on.
    """
    keys: list[str] = []
    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    with h5py.File(hdf5_path, "r") as f:
        sorted_keys = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
        for k in sorted_keys:
            T = int(f["data"][k]["actions"].shape[0])
            offsets[k] = (cursor, T)
            keys.append(k)
            cursor += T
    return keys, offsets


def _split_keys(keys: list[str], n_demos: int, train_frac: float) -> tuple[list[str], list[str]]:
    subset = keys[: max(1, int(n_demos))]
    n_train = int(math.ceil(train_frac * len(subset)))
    return subset[:n_train], subset[n_train:]


def _make_robomimic_image_cached(
    dataset_name: str,
    encoder: str,
    n_demos: int,
    action_norm_mode: str,
    train_frac: float = 0.8,
) -> tuple[FeatureCachedDataset, FeatureCachedDataset, int]:
    rest = dataset_name[len("robomimic_") : -len("_image")]
    task, variant = rest.split("_")
    spec = RoboMimicSpec(task=task, variant=variant, modality="image")
    hdf5_path = hdf5_path_for(spec, data_root() / "robomimic")
    if not hdf5_path.is_file():
        raise FileNotFoundError(f"missing RoboMimic image HDF5: {hdf5_path}")

    feats = _load_features(dataset_name, encoder)
    keys, offsets = _robomimic_image_feature_offsets(hdf5_path)
    train_keys, val_keys = _split_keys(keys, n_demos, train_frac)

    def _build_transitions(demo_keys: list[str]) -> list[_Transition]:
        out: list[_Transition] = []
        with h5py.File(hdf5_path, "r") as f:
            for k in demo_keys:
                base, length = offsets[k]
                actions = np.asarray(f["data"][k]["actions"], dtype=np.float64)
                for t in range(length - 1):
                    g0 = float(actions[t, 6])
                    g1 = float(actions[t + 1, 6])
                    contact = abs(g1 - g0) > 0.1
                    out.append(_Transition(
                        feat_t_idx=base + t,
                        feat_next_idx=base + t + 1,
                        action=actions[t].copy(),
                        is_contact=bool(contact),
                        demo_key=k,
                        t=int(t),
                    ))
        return out

    train_tr = _build_transitions(train_keys)
    val_tr = _build_transitions(val_keys)

    if not train_tr:
        raise RuntimeError(f"empty train split for {dataset_name!r}")

    train_actions = np.stack([tr.action for tr in train_tr], axis=0)
    action_stats = compute_action_stats(train_actions)

    def _resolve(demo_key: str, t: int) -> tuple[np.ndarray, np.ndarray]:
        frames = read_frames(hdf5_path, demo_key, [int(t), int(t) + 1], cam=IMAGE_OBS_KEY)
        return frames[0], frames[1]

    train_ds = FeatureCachedDataset(
        dataset_name=dataset_name, features=feats, transitions=train_tr,
        action_stats=action_stats, action_norm_mode=action_norm_mode, frame_resolver=_resolve,
    )
    val_ds = FeatureCachedDataset(
        dataset_name=dataset_name, features=feats, transitions=val_tr,
        action_stats=action_stats, action_norm_mode=action_norm_mode, frame_resolver=_resolve,
    )
    return train_ds, val_ds, train_ds.feature_dim


def build_feature_cached_train_val(
    dataset_name: str,
    encoder: str,
    n_demos: int,
    action_norm_mode: str = "q01_q99",
    train_frac: float = 0.8,
) -> tuple[FeatureCachedDataset, FeatureCachedDataset, int]:
    """Dispatch a feature-cached dataset by name."""
    if dataset_name.startswith("robomimic_") and dataset_name.endswith("_image"):
        return _make_robomimic_image_cached(
            dataset_name, encoder, n_demos, action_norm_mode, train_frac=train_frac
        )
    if dataset_name.startswith("libero_"):
        raise NotImplementedError(
            f"LIBERO feature-cached dataset {dataset_name!r}: the "
            "openvla/modified_libero_rlds schema must be verified before this "
            "branch is enabled. Inspect a sample row with "
            "``datasets.load_dataset('openvla/modified_libero_rlds', name='libero_spatial', "
            "split='train')[0].keys()`` and update v2/datasets/cached.py."
        )
    raise ValueError(f"unknown image-feature dataset: {dataset_name!r}")
