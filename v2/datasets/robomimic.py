"""RoboMimic v0.1 dataset adapter, generalized over task / variant / modality.

Generalizes the legacy ``src/data.py`` Lift-only loader to all four
single-arm tasks (Lift, Can, Square, Transport) in both PH and MH variants
and both ``low_dim`` and ``image`` modalities. The legacy file is left
untouched so ``src/train.py`` keeps reproducing val_mse ~ 0.397.

Image modality: RoboMimic stores RGB at 84x84 in HDF5 under
``data/<demo>/obs/{agentview_image,robot0_eye_in_hand_image}``. We
preserve frames lazily so visualization code can render any
``(obs_t, obs_next)`` pair without holding all images in memory.
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

from .stats import ActionStats, compute_action_stats

# Low-dim observation key composition shared across tasks. Lift / Can /
# Square / Transport all expose the same base keys but the ``object`` key
# width varies per task; we look it up at load time rather than hard-coding.
LOWDIM_OBS_KEYS = ("object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")

# Image modality uses the front camera by convention; eye-in-hand can be
# added later as a second view.
IMAGE_OBS_KEY = "agentview_image"


@dataclass(frozen=True)
class RoboMimicSpec:
    task: str  # lift / can / square / transport
    variant: str  # ph / mh
    modality: str  # low_dim / image

    @property
    def name(self) -> str:
        return f"robomimic_{self.task}_{self.variant}_{self.modality}"


def hdf5_path_for(spec: RoboMimicSpec, data_root: Path) -> Path:
    """Resolve the HDF5 path for a spec, checking several known layouts.

    Search order:
      1. ``<data_root>/v1.5/<task>/<variant>/<modality>_v141.hdf5``
         (the layout used by ``huggingface_hub.snapshot_download`` on
         ``amandlek/robomimic``)
      2. ``<data_root>/<task>/<variant>/<modality>_v141.hdf5``
         (flat layout some mirrors use)
      3. ``<repo_root>/data/<task>/<variant>/<modality>_v141.hdf5``
         (the legacy local path used by ``src/data.py``)

    Returns the first path that exists; if none does, returns the
    primary candidate so callers can produce a clear FileNotFoundError.
    """
    fname = f"{spec.modality}_v141.hdf5"
    candidates = [
        data_root / "v1.5" / spec.task / spec.variant / fname,
        data_root / spec.task / spec.variant / fname,
    ]
    repo_root = Path(__file__).resolve().parents[2]
    candidates.append(repo_root / "data" / spec.task / spec.variant / fname)
    for c in candidates:
        if c.is_file():
            return c
    return candidates[0]


def _lowdim_state_dim(obs_grp: h5py.Group) -> int:
    return sum(int(obs_grp[k].shape[-1]) for k in LOWDIM_OBS_KEYS)


def _build_lowdim_state(obs_grp: h5py.Group) -> np.ndarray:
    parts = [np.asarray(obs_grp[k], dtype=np.float64) for k in LOWDIM_OBS_KEYS]
    return np.concatenate(parts, axis=-1)


def load_demos(
    spec: RoboMimicSpec,
    hdf5_path: Path,
    n_demos: int,
    keep_image_in_memory: bool = False,
) -> tuple[list[dict[str, np.ndarray]], int]:
    """Load the first ``n_demos`` demonstrations.

    Returns ``(demos, state_dim)`` where each demo dict has at least
    ``s`` (T, state_dim) and ``a`` (T, 7). When ``modality == "image"``
    and ``keep_image_in_memory`` is ``True`` the dict also carries
    ``image`` (T, H, W, 3) uint8.

    For image runs we typically *do not* hold the full pixel buffer in
    memory. The frames are read lazily through :func:`read_frames`
    instead.
    """
    demos: list[dict[str, np.ndarray]] = []
    state_dim = 0
    with h5py.File(hdf5_path, "r") as f:
        data_grp = f["data"]
        all_keys = sorted(data_grp.keys(), key=lambda x: int(x.split("_")[1]))
        subset = all_keys[:n_demos]

        for dk in subset:
            g = data_grp[dk]
            obs = g["obs"]
            if state_dim == 0:
                state_dim = _lowdim_state_dim(obs)
            s = _build_lowdim_state(obs)
            assert s.ndim == 2 and s.shape[1] == state_dim
            a = np.asarray(g["actions"], dtype=np.float64)

            entry: dict[str, np.ndarray] = {"s": s, "a": a, "_demo_key": np.asarray(dk)}
            if spec.modality == "image" and keep_image_in_memory:
                entry["image"] = np.asarray(obs[IMAGE_OBS_KEY])
            demos.append(entry)

    return demos, state_dim


def read_frames(
    hdf5_path: Path,
    demo_key: str,
    timesteps: list[int] | np.ndarray,
    cam: str = IMAGE_OBS_KEY,
) -> np.ndarray:
    """Return ``(len(timesteps), H, W, 3)`` uint8 frames for visualization."""
    ts = list(int(t) for t in timesteps)
    with h5py.File(hdf5_path, "r") as f:
        arr = np.asarray(f["data"][demo_key]["obs"][cam])
    return arr[ts]


def build_transitions(demos: list[dict[str, np.ndarray]]) -> list[tuple[np.ndarray, ...]]:
    """``(s_t, s_next, a_t, is_contact, demo_key, t)`` per timestep ``t``."""
    out: list[tuple[np.ndarray, ...]] = []
    for d in demos:
        s = d["s"]
        a = d["a"]
        dk = str(np.asarray(d["_demo_key"]))
        T = s.shape[0]
        for t in range(T - 1):
            g0 = float(a[t, 6])
            g1 = float(a[t + 1, 6])
            contact = abs(g1 - g0) > 0.1
            out.append((s[t].copy(), s[t + 1].copy(), a[t].copy(), bool(contact), dk, int(t)))
    return out


def split_demo_indices(n_demos: int, train_frac: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    n_train = int(math.ceil(train_frac * n_demos))
    idx = np.arange(n_demos)
    return idx[:n_train], idx[n_train:]


def transitions_for_split(
    demos: list[dict[str, np.ndarray]], demo_indices: np.ndarray | list[int]
) -> list[tuple[np.ndarray, ...]]:
    sub = [demos[int(i)] for i in demo_indices]
    return build_transitions(sub)


class RoboMimicTransitionDataset(Dataset):
    """Normalized ``(obs_t, obs_next, action, is_contact)`` transitions.

    For ``modality == "low_dim"`` the obs is the proprioceptive state.
    For ``modality == "image"`` callers should *not* use this class
    directly; use :class:`RoboMimicImageFeatureDataset` against
    pre-extracted CLS features.
    """

    def __init__(
        self,
        spec: RoboMimicSpec,
        hdf5_path: Path,
        transitions: list[tuple[np.ndarray, ...]],
        state_dim: int,
        action_stats: ActionStats,
        state_mean: np.ndarray | None = None,
        state_std: np.ndarray | None = None,
        normalize_state: bool = True,
        action_norm_mode: str = "zscore",
    ) -> None:
        self.spec = spec
        self.hdf5_path = hdf5_path
        self.transitions = transitions
        self.state_dim = int(state_dim)
        self.action_dim = 7
        self.stats = action_stats
        self.action_norm_mode = action_norm_mode

        if normalize_state:
            if state_mean is None or state_std is None:
                states = np.stack(
                    [tr[0] for tr in transitions] + [tr[1] for tr in transitions], axis=0
                )
                state_mean = states.mean(axis=0).astype(np.float32)
                state_std = np.maximum(states.std(axis=0).astype(np.float32), 1e-6)
            self.state_mean = torch.as_tensor(state_mean, dtype=torch.float32)
            self.state_std = torch.as_tensor(state_std, dtype=torch.float32)
        else:
            self.state_mean = torch.zeros(self.state_dim)
            self.state_std = torch.ones(self.state_dim)

        self._q01 = torch.as_tensor(action_stats.q01, dtype=torch.float32)
        self._q99 = torch.as_tensor(action_stats.q99, dtype=torch.float32)
        self._a_mean = torch.as_tensor(action_stats.mean, dtype=torch.float32)
        self._a_std = torch.as_tensor(action_stats.std, dtype=torch.float32).clamp(min=1e-6)

    def __len__(self) -> int:
        return len(self.transitions)

    def _norm_state(self, s: torch.Tensor) -> torch.Tensor:
        return (s - self.state_mean) / self.state_std

    def _norm_action(self, a: torch.Tensor) -> torch.Tensor:
        if self.action_norm_mode == "zscore":
            return (a - self._a_mean) / self._a_std
        denom = (self._q99 - self._q01).clamp(min=1e-6)
        x = 2.0 * (a - self._q01) / denom - 1.0
        return x.clamp(-1.0, 1.0)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s_t_np, s_n_np, a_np, contact, demo_key, t = self.transitions[idx]
        s_t = torch.as_tensor(s_t_np, dtype=torch.float32)
        s_n = torch.as_tensor(s_n_np, dtype=torch.float32)
        a = torch.as_tensor(a_np, dtype=torch.float32)
        return {
            "obs_t": self._norm_state(s_t),
            "obs_next": self._norm_state(s_n),
            "s_t": self._norm_state(s_t),
            "s_next": self._norm_state(s_n),
            "action": self._norm_action(a),
            "action_raw": a,
            "is_contact": torch.tensor(bool(contact), dtype=torch.bool),
            "idx": torch.tensor(idx, dtype=torch.long),
            "demo_key": demo_key,
            "t": torch.tensor(int(t), dtype=torch.long),
            "dataset_id": torch.tensor(0, dtype=torch.long),  # set by MixedIDMDataset
        }

    def fetch_frames(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(frame_t, frame_next)`` uint8 arrays for visualization.

        Only meaningful when ``self.spec.modality == "image"``.
        """
        if self.spec.modality != "image":
            raise RuntimeError("frames only available when modality == 'image'")
        _, _, _, _, demo_key, t = self.transitions[idx]
        frames = read_frames(self.hdf5_path, demo_key, [int(t), int(t) + 1])
        return frames[0], frames[1]


def make_train_val(
    spec: RoboMimicSpec,
    n_demos: int,
    data_root: Path,
    train_frac: float = 0.8,
    action_stats: ActionStats | None = None,
    action_norm_mode: str = "zscore",
) -> tuple[RoboMimicTransitionDataset, RoboMimicTransitionDataset, ActionStats, int]:
    """Build train and val datasets with shared per-dataset action stats.

    ``action_norm_mode``:

    * ``"zscore"`` — ``(a - mean) / std``. Default. Matches the legacy
      ``src/data.py`` recipe so phase A reproduces the autoresearch
      baseline (val_mse ~ 0.397) and so mixing across RoboMimic tasks
      keeps action ranges compatible.
    * ``"q01_q99"`` — OpenVLA-style mapping into ``[-1, 1]``. Use this
      when mixing image-feature cells across heterogeneous embodiments
      and you want the OpenVLA / RDT-1B / GR00T-N1 normalization recipe.
    """
    hdf5_path = hdf5_path_for(spec, data_root)
    if not hdf5_path.is_file():
        raise FileNotFoundError(
            f"missing RoboMimic file: {hdf5_path}\n"
            "Tried v2 layout, flat layout, and the legacy ``data/`` layout. "
            "Run ``python3 -m v2.runtime.data_download`` first, or copy the "
            "HDF5 into ``data/<task>/<variant>/<modality>_v141.hdf5``."
        )

    demos, state_dim = load_demos(spec, hdf5_path, n_demos, keep_image_in_memory=False)
    train_didx, val_didx = split_demo_indices(n_demos, train_frac=train_frac)

    train_triples = transitions_for_split(demos, train_didx)
    val_triples = transitions_for_split(demos, val_didx)

    if action_stats is None:
        train_actions = np.stack([tr[2] for tr in train_triples], axis=0)
        action_stats = compute_action_stats(train_actions)

    states = np.stack(
        [tr[0] for tr in train_triples] + [tr[1] for tr in train_triples], axis=0
    )
    state_mean = states.mean(axis=0).astype(np.float32)
    state_std = np.maximum(states.std(axis=0).astype(np.float32), 1e-6)

    train_ds = RoboMimicTransitionDataset(
        spec, hdf5_path, train_triples, state_dim, action_stats,
        state_mean=state_mean, state_std=state_std,
        action_norm_mode=action_norm_mode,
    )
    val_ds = RoboMimicTransitionDataset(
        spec, hdf5_path, val_triples, state_dim, action_stats,
        state_mean=state_mean, state_std=state_std,
        action_norm_mode=action_norm_mode,
    )
    return train_ds, val_ds, action_stats, state_dim
