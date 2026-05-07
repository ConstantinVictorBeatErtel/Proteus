"""Feature-cached dataset wrapper.

Reads cached CLS features written by ``v2.features`` and exposes the
common ``{obs_t, obs_next, action, ...}`` dict schema. The cached
features are indexed in trajectory order (one row per timestep,
flattened across demos in HDF5/RLDS demo-key order). The wrapper
walks the same demos to reconstruct ``(demo_key, t)`` for each
transition so visualization can still pull raw RGB frames from the
underlying HDF5 / RLDS source.

This module ships both the **RoboMimic** and **LIBERO** image-feature
branches end-to-end. Each cached safetensors file carries a small
metadata sidecar; loaders verify frame count and row-order checksum
before training so stale caches fail loudly instead of silently
misaligning transitions.
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

from ..runtime.drive import data_root
from . import libero as lb
from .cache_layout import feature_cache_path, load_feature_cache_metadata, layout_checksum
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

    @property
    def feat_base_idx(self) -> int:
        return self.feat_t_idx - self.t


def _load_features(dataset_name: str, encoder: str) -> torch.Tensor:
    from safetensors.torch import load_file

    target = feature_cache_path(dataset_name, encoder)
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

        # Pre-stack everything into tensors so __getitem__ is just an index.
        if transitions:
            t_idx = torch.tensor([tr.feat_t_idx for tr in transitions], dtype=torch.long)
            n_idx = torch.tensor([tr.feat_next_idx for tr in transitions], dtype=torch.long)
            w_idx = torch.tensor(
                [
                    [
                        tr.feat_base_idx + max(0, tr.t - 2),
                        tr.feat_base_idx + max(0, tr.t - 1),
                        tr.feat_t_idx,
                        tr.feat_next_idx,
                    ]
                    for tr in transitions
                ],
                dtype=torch.long,
            )
            self._t_idx = t_idx
            self._n_idx = n_idx
            self._w_idx = w_idx
            raw_actions = torch.from_numpy(np.stack([tr.action for tr in transitions], axis=0).astype(np.float32, copy=False))
            self._raw_actions = raw_actions
            if action_norm_mode == "zscore":
                self._actions_norm = (raw_actions - self._a_mean) / self._a_std
            else:
                denom = (self._q99 - self._q01).clamp(min=1e-6)
                self._actions_norm = (2.0 * (raw_actions - self._q01) / denom - 1.0).clamp(-1.0, 1.0)
            self._contacts = torch.tensor([bool(tr.is_contact) for tr in transitions], dtype=torch.bool)
            self._t_steps = torch.tensor([int(tr.t) for tr in transitions], dtype=torch.long)
            self._demo_keys = [tr.demo_key for tr in transitions]
        else:
            self._t_idx = torch.zeros(0, dtype=torch.long)
            self._n_idx = torch.zeros(0, dtype=torch.long)
            self._w_idx = torch.zeros(0, 4, dtype=torch.long)
            self._raw_actions = torch.zeros(0, 7, dtype=torch.float32)
            self._actions_norm = torch.zeros(0, 7, dtype=torch.float32)
            self._contacts = torch.zeros(0, dtype=torch.bool)
            self._t_steps = torch.zeros(0, dtype=torch.long)
            self._demo_keys = []

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        i = int(idx)
        f_t = self.features[self._t_idx[i]]
        f_n = self.features[self._n_idx[i]]
        return {
            "obs_t": f_t,
            "obs_next": f_n,
            "s_t": f_t,
            "s_next": f_n,
            "obs_window": self.features.index_select(0, self._w_idx[i]),
            "action": self._actions_norm[i],
            "action_raw": self._raw_actions[i],
            "is_contact": self._contacts[i],
            "idx": torch.tensor(i, dtype=torch.long),
            "demo_key": self._demo_keys[i],
            "t": self._t_steps[i],
            "dataset_id": torch.tensor(0, dtype=torch.long),
        }

    # ---- bulk accessors used by FeatureMemoryBank.populate_vectorized ----

    def stacked_obs_t(self, key: str = "obs_t") -> torch.Tensor:
        return self.features.index_select(0, self._t_idx)

    def stacked_obs_next(self, key: str = "obs_next") -> torch.Tensor:
        return self.features.index_select(0, self._n_idx)

    def stacked_actions(self, key: str = "action") -> torch.Tensor:
        return self._actions_norm

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


def _verify_feature_layout(
    dataset_name: str,
    encoder: str,
    features: torch.Tensor,
    layout_entries: list[tuple[str, int]],
) -> None:
    target = feature_cache_path(dataset_name, encoder)
    meta = load_feature_cache_metadata(target)
    if meta is None:
        raise RuntimeError(
            f"cached features at {target} are missing their metadata sidecar. "
            "Re-run `python3 -m v2.features` so row-order verification is available."
        )
    if meta.frame_count != int(features.shape[0]) or meta.feature_dim != int(features.shape[1]):
        raise RuntimeError(
            f"cached features at {target} have shape {tuple(features.shape)} but metadata says "
            f"frame_count={meta.frame_count}, feature_dim={meta.feature_dim}. Re-extract features."
        )
    expected_checksum = layout_checksum(layout_entries)
    if meta.layout_checksum != expected_checksum:
        raise RuntimeError(
            f"cached features at {target} do not match the current dataset layout "
            f"(metadata checksum {meta.layout_checksum}, expected {expected_checksum}). "
            "This usually means the features file was built against a different HDF5/RLDS order; "
            "re-run `python3 -m v2.features` before training."
        )


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
    stride: int = 1,
) -> tuple[FeatureCachedDataset, FeatureCachedDataset, int]:
    rest = dataset_name[len("robomimic_") : -len("_image")]
    task, variant = rest.split("_")
    spec = RoboMimicSpec(task=task, variant=variant, modality="image")
    hdf5_path = hdf5_path_for(spec, data_root() / "robomimic")
    if not hdf5_path.is_file():
        raise FileNotFoundError(f"missing RoboMimic image HDF5: {hdf5_path}")

    feats = _load_features(dataset_name, encoder)
    keys, offsets = _robomimic_image_feature_offsets(hdf5_path)
    _verify_feature_layout(dataset_name, encoder, feats, [(k, offsets[k][1]) for k in keys])
    train_keys, val_keys = _split_keys(keys, n_demos, train_frac)

    stride = max(1, int(stride))

    def _build_transitions(demo_keys: list[str]) -> list[_Transition]:
        out: list[_Transition] = []
        with h5py.File(hdf5_path, "r") as f:
            for k in demo_keys:
                base, length = offsets[k]
                if length <= stride:
                    continue
                actions = np.asarray(f["data"][k]["actions"], dtype=np.float64)
                # Stride > 1 makes the IDM problem (o_t, o_{t+stride}) more
                # informative for image-feature inputs, where adjacent
                # 50 ms frames have near-identical CLS tokens. The action
                # we predict is still ``a_t`` (the action that started the
                # window), matching the LAPA recipe.
                for t in range(length - stride):
                    g0 = float(actions[t, 6])
                    g1 = float(actions[t + stride, 6])
                    contact = abs(g1 - g0) > 0.1
                    out.append(_Transition(
                        feat_t_idx=base + t,
                        feat_next_idx=base + t + stride,
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


def _make_libero_image_cached(
    dataset_name: str,
    encoder: str,
    n_demos: int,
    action_norm_mode: str,
    train_frac: float = 0.8,
    stride: int = 1,
) -> tuple[FeatureCachedDataset, FeatureCachedDataset, int]:
    feats = _load_features(dataset_name, encoder)

    suite = dataset_name  # "libero_spatial" / "libero_object" / "libero_goal" / "libero_10" / ...
    # The feature cache was built over every episode in the suite (see
    # v2.features), so its layout_checksum and row layout are over the full
    # list. Verify against the full list and build offsets over it; only
    # then slice to ``n_demos`` for the train/val pool.
    all_episodes = lb.find_libero_episodes(suite, data_root() / "libero")
    if not all_episodes:
        raise RuntimeError(f"no episodes loaded for suite {suite!r}")

    full_layout_entries = lb.episode_layout(all_episodes)
    _verify_feature_layout(dataset_name, encoder, feats, full_layout_entries)

    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for ep in all_episodes:
        offsets[ep.composite_key] = (cursor, ep.length)
        cursor += ep.length

    episodes = all_episodes[: max(1, int(n_demos))]
    n_train = int(math.ceil(train_frac * len(episodes)))
    train_eps = episodes[:n_train]
    val_eps = episodes[n_train:]

    stride = max(1, int(stride))

    # Cache the action arrays per (file, demo_key) so transition building
    # touches each HDF5 once instead of opening it per timestep.
    action_cache: dict[tuple[str, str], np.ndarray] = {}

    def _actions_for(ep: lb.LiberoEpisode) -> np.ndarray:
        key = (str(ep.hdf5_path), ep.demo_key)
        if key not in action_cache:
            action_cache[key] = lb.read_actions(ep.hdf5_path, ep.demo_key)
        return action_cache[key]

    def _build_transitions(eps: list[lb.LiberoEpisode]) -> list[_Transition]:
        out: list[_Transition] = []
        for ep in eps:
            base, length = offsets[ep.composite_key]
            if length <= stride:
                continue
            actions = _actions_for(ep)
            for t in range(length - stride):
                # LIBERO's action[6] is the gripper command, same as RoboMimic.
                g0 = float(actions[t, 6]) if actions.shape[1] > 6 else 0.0
                g1 = float(actions[t + stride, 6]) if actions.shape[1] > 6 else 0.0
                contact = abs(g1 - g0) > 0.1
                out.append(
                    _Transition(
                        feat_t_idx=base + t,
                        feat_next_idx=base + t + stride,
                        action=actions[t].copy(),
                        is_contact=bool(contact),
                        demo_key=ep.composite_key,
                        t=int(t),
                    )
                )
        return out

    train_tr = _build_transitions(train_eps)
    val_tr = _build_transitions(val_eps)
    if not train_tr:
        raise RuntimeError(
            f"empty train split for {dataset_name!r} "
            f"(n_demos={n_demos}, stride={stride}, episodes={len(episodes)})"
        )

    train_actions = np.stack([tr.action for tr in train_tr], axis=0)
    action_stats = compute_action_stats(train_actions)

    # For visualization: composite_key -> hdf5 path + raw demo key.
    file_index: dict[str, tuple[Path, str]] = {
        ep.composite_key: (ep.hdf5_path, ep.demo_key) for ep in episodes
    }

    def _resolve(demo_key: str, t: int) -> tuple[np.ndarray, np.ndarray]:
        hp, raw_dk = file_index[demo_key]
        # frame at t and t+stride for image preview matching the cell's stride.
        frames = lb.read_frames(hp, raw_dk, [int(t), int(t) + stride])
        return frames[0], frames[1]

    train_ds = FeatureCachedDataset(
        dataset_name=dataset_name,
        features=feats,
        transitions=train_tr,
        action_stats=action_stats,
        action_norm_mode=action_norm_mode,
        frame_resolver=_resolve,
    )
    val_ds = FeatureCachedDataset(
        dataset_name=dataset_name,
        features=feats,
        transitions=val_tr,
        action_stats=action_stats,
        action_norm_mode=action_norm_mode,
        frame_resolver=_resolve,
    )
    return train_ds, val_ds, train_ds.feature_dim


def build_feature_cached_train_val(
    dataset_name: str,
    encoder: str,
    n_demos: int,
    action_norm_mode: str = "q01_q99",
    train_frac: float = 0.8,
    stride: int = 1,
) -> tuple[FeatureCachedDataset, FeatureCachedDataset, int]:
    """Dispatch a feature-cached dataset by name.

    ``stride`` controls the temporal gap between ``obs_t`` and ``obs_next``
    inside each demo. With ``stride=1`` (default) we get adjacent-frame
    IDM, which has a tiny visual delta at 20 Hz; ``stride=5`` (250 ms)
    or ``stride=10`` (500 ms) give the encoder a meaningfully larger
    visual change and tend to lift image-feature IDM val_mse noticeably.
    """
    if dataset_name.startswith("robomimic_") and dataset_name.endswith("_image"):
        return _make_robomimic_image_cached(
            dataset_name, encoder, n_demos, action_norm_mode,
            train_frac=train_frac, stride=stride,
        )
    if dataset_name.startswith("libero_"):
        return _make_libero_image_cached(
            dataset_name, encoder, n_demos, action_norm_mode,
            train_frac=train_frac, stride=stride,
        )
    raise ValueError(f"unknown image-feature dataset: {dataset_name!r}")
