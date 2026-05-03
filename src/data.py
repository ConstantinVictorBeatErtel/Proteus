"""
Touch in the Wild — VTBC dataset and normalization utilities.

Loads four task zarr archives, computes delta actions from consecutive EEF states,
episode-level train/val split (80%/20%), and tactile/action normalization from train split.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize as tv_resize


# imagecodecs: required for compressed zarr arrays -----------------------------
import imagecodecs.numcodecs  # noqa: F401

imagecodecs.numcodecs.register_codecs()

import zarr  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TASKS = [
    "fluid_transfer",
    "test_tube_collection",
    "pencil_insertion",
    "whiteboard_erasing",
]
ZARR_PATH_TEMPLATE = os.path.join(
    REPO_ROOT,
    "data",
    "touch_in_the_wild",
    "four_tasks",
    "{task}",
    "{task}.zarr.zip",
)


def _open_zarr_for_task(task: str) -> zarr.Group:
    path = ZARR_PATH_TEMPLATE.format(task=task)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing zarr for task '{task}': {path}")
    return zarr.open(path, mode="r")


def _episode_ranges(episode_ends: np.ndarray) -> List[Tuple[int, int]]:
    """Return inclusive (start, end) global timestep indices per episode."""
    ranges: List[Tuple[int, int]] = []
    prev = -1
    for raw in episode_ends:
        end = int(raw)
        start = prev + 1
        ranges.append((start, end))
        prev = end
    return ranges


def _split_episode_ids(n_episodes: int) -> Tuple[List[int], List[int]]:
    """First 80% train, last 20% val — by episode index, deterministic."""
    n_train = int(0.8 * n_episodes)
    if n_train <= 0 or n_train >= n_episodes:
        raise ValueError(
            f"Episode split degenerate: n_episodes={n_episodes}, n_train={n_train}"
        )
    train_ids = list(range(0, n_train))
    val_ids = list(range(n_train, n_episodes))
    return train_ids, val_ids


@dataclass(frozen=True)
class SampleIndex:
    task: str
    t: int  # global flat index within this task's zarr (predict transition t -> t+1)


def enumerate_split_indices(split: str) -> Dict[str, List[SampleIndex]]:
    if split not in ("train", "val"):
        raise ValueError("split must be 'train' or 'val'")

    out: Dict[str, List[SampleIndex]] = {t: [] for t in TASKS}
    for task in TASKS:
        root = _open_zarr_for_task(task)
        ends = np.array(root["meta"]["episode_ends"][:]).reshape(-1)
        ranges = _episode_ranges(ends)
        train_ids, val_ids = _split_episode_ids(len(ranges))
        episode_ids = train_ids if split == "train" else val_ids

        for ep_idx in episode_ids:
            start, end = ranges[ep_idx]
            if end - start + 1 < 2:
                continue
            for t in range(start, end):
                out[task].append(SampleIndex(task=task, t=int(t)))
    return out


def count_episodes_per_split() -> Tuple[Dict[str, int], Dict[str, int]]:
    train_counts: Dict[str, int] = {}
    val_counts: Dict[str, int] = {}
    for task in TASKS:
        root = _open_zarr_for_task(task)
        ends = np.array(root["meta"]["episode_ends"][:]).reshape(-1)
        n_eps = len(_episode_ranges(ends))
        n_train = int(0.8 * n_eps)
        train_counts[task] = n_train
        val_counts[task] = n_eps - n_train
    return train_counts, val_counts


def compute_norm_stats(norm_stats_path: str) -> Dict[str, torch.Tensor]:
    """Compute action mean/std and tactile min/max on TRAIN split only; save to disk."""
    os.makedirs(os.path.dirname(norm_stats_path), exist_ok=True)
    action_chunks: List[np.ndarray] = []
    tactile_chunks: List[np.ndarray] = []

    for task in TASKS:
        root = _open_zarr_for_task(task)
        ends = np.array(root["meta"]["episode_ends"][:]).reshape(-1)
        ranges = _episode_ranges(ends)
        train_ids, _ = _split_episode_ids(len(ranges))
        pos_ds = root["data"]["robot0_eef_pos"]
        rot_ds = root["data"]["robot0_eef_rot_axis_angle"]
        grip_ds = root["data"]["robot0_gripper_width"]
        tac_ds = root["data"]["camera0_tactile"]

        n_trans = 0
        for ep_idx in train_ids:
            s_e, e_e = ranges[ep_idx]
            if e_e <= s_e:
                continue
            pos = np.asarray(pos_ds[s_e : e_e + 1], dtype=np.float32)
            rot = np.asarray(rot_ds[s_e : e_e + 1], dtype=np.float32)
            grip = np.asarray(grip_ds[s_e : e_e + 1], dtype=np.float32).reshape(-1)

            pos = pos.reshape(pos.shape[0], -1)
            rot = rot.reshape(rot.shape[0], -1)
            dpos = pos[1:] - pos[:-1]
            drot = rot[1:] - rot[:-1]
            dgrip = (grip[1:] - grip[:-1]).reshape(-1, 1).astype(np.float32)
            a = np.concatenate([dpos, drot, dgrip], axis=1).astype(np.float32)
            if a.shape[1] != 7:
                raise ValueError(
                    f"{task} ep {ep_idx}: expected 7-dim delta action, got {a.shape[1]}"
                )
            action_chunks.append(a)

            tac = np.asarray(tac_ds[s_e:e_e], dtype=np.float32)
            if tac.shape[1:] != (12, 64):
                raise ValueError(
                    f"{task} ep {ep_idx}: tactile shape {tac.shape}, expected (T,12,64)"
                )
            tactile_chunks.append(tac)
            n_trans += a.shape[0]

        print(f"[data] norm stats: {task} train transitions={n_trans}")

    actions = np.concatenate(action_chunks, axis=0)
    tactile_all = np.concatenate(tactile_chunks, axis=0)

    a_mean = actions.mean(axis=0)
    a_std = actions.std(axis=0)
    a_std = np.where(a_std < 1e-6, 1.0, a_std)

    t_min = tactile_all.min(axis=0)
    t_max = tactile_all.max(axis=0)
    span = np.maximum(t_max - t_min, 1e-8)

    stats = {
        "action_mean": torch.from_numpy(a_mean),
        "action_std": torch.from_numpy(a_std),
        "tactile_min": torch.from_numpy(t_min),
        "tactile_span": torch.from_numpy(span),
    }
    torch.save(stats, norm_stats_path)
    return stats


def load_norm_stats(norm_stats_path: str) -> Dict[str, torch.Tensor]:
    return torch.load(norm_stats_path, map_location="cpu", weights_only=False)


def task_num_timesteps(task: str) -> int:
    """Length of the time dimension for synced modalities (matches zarr RGB rows)."""
    root = _open_zarr_for_task(task)
    n = int(root["data"]["camera0_rgb"].shape[0])
    return n


class VTBCWindowDataset(Dataset):
    """
    Contiguous within-episode windows of length `horizon` for causal Transformer BC.

    Each item is a sequence aligned to timesteps t..t+horizon-1 (same episode, train/val split).
    Also returns `(task_idx, start_t)` so CLIP caches (per-task, per timestep) can be sliced.

    Shapes / types:
      rgb: (H, 224, 224, 3) uint8
      tactile: (H, 12, 64) float32 (normalized like VTBCDataset)
      actions: (H, 7) float32 normalized
      task_idx: int in [0, len(TASKS))
      start_t: int — first timestep index in this task zarr
    """

    def __init__(
        self,
        split: str,
        norm_stats_path: str,
        horizon: int = 10,
        image_size: int = 224,
        seed: int = 42,
    ) -> None:
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")
        self.split = split
        self.horizon = horizon
        self.image_size = image_size

        if not os.path.isfile(norm_stats_path):
            print(f"[data] Computing normalization stats → {norm_stats_path}")
            compute_norm_stats(norm_stats_path)
        self.stats = load_norm_stats(norm_stats_path)

        self.windows: List[Tuple[int, str, int]] = []  # (task_idx, task, start timestep)
        for ti, task in enumerate(TASKS):
            root = _open_zarr_for_task(task)
            ends = np.array(root["meta"]["episode_ends"][:]).reshape(-1)
            ranges = _episode_ranges(ends)
            train_ids, val_ids = _split_episode_ids(len(ranges))
            ep_ids = train_ids if split == "train" else val_ids
            for ep_idx in ep_ids:
                s_e, e_e = ranges[ep_idx]
                max_t_act = e_e - 1
                max_start = max_t_act - (horizon - 1)
                if max_start < s_e:
                    continue
                for s in range(s_e, max_start + 1):
                    self.windows.append((ti, task, int(s)))

        if split == "train":
            rng = np.random.default_rng(seed)
            rng.shuffle(self.windows)

        self.roots = {task: _open_zarr_for_task(task) for task in TASKS}

    def __len__(self) -> int:
        return len(self.windows)

    def _normalize_action(self, a: torch.Tensor) -> torch.Tensor:
        mean = self.stats["action_mean"].to(dtype=a.dtype, device=a.device)
        std = self.stats["action_std"].to(dtype=a.dtype, device=a.device)
        return (a - mean) / std

    def _normalize_tactile(self, t: torch.Tensor) -> torch.Tensor:
        t_min = self.stats["tactile_min"].to(dtype=t.dtype, device=t.device)
        span = self.stats["tactile_span"].to(dtype=t.dtype, device=t.device)
        return torch.clamp((t - t_min) / span, 0.0, 1.0)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        task_idx, task, s = self.windows[idx]
        root = self.roots[task]

        rgbs = []
        tacs = []
        acts = []

        for k in range(self.horizon):
            t = s + k
            rgb = np.asarray(root["data"]["camera0_rgb"][t], dtype=np.uint8)
            tactile = np.asarray(root["data"]["camera0_tactile"][t], dtype=np.float32)

            pos0 = np.asarray(root["data"]["robot0_eef_pos"][t], dtype=np.float32).reshape(-1)
            pos1 = np.asarray(root["data"]["robot0_eef_pos"][t + 1], dtype=np.float32).reshape(
                -1
            )
            r0 = np.asarray(
                root["data"]["robot0_eef_rot_axis_angle"][t], dtype=np.float32
            ).reshape(-1)
            r1 = np.asarray(
                root["data"]["robot0_eef_rot_axis_angle"][t + 1], dtype=np.float32
            ).reshape(-1)
            g0 = float(
                np.asarray(root["data"]["robot0_gripper_width"][t], dtype=np.float32).reshape(-1)[0]
            )
            g1 = float(
                np.asarray(
                    root["data"]["robot0_gripper_width"][t + 1],
                    dtype=np.float32,
                ).reshape(-1)[0]
            )

            action = torch.from_numpy(
                np.concatenate(
                    [
                        pos1 - pos0,
                        r1 - r0,
                        np.array([g1 - g0], dtype=np.float32),
                    ],
                    axis=0,
                )
            ).float()

            rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
            rgb_t = tv_resize(rgb_t, [self.image_size, self.image_size])
            rgb_u8 = rgb_t.byte().permute(1, 2, 0).contiguous()

            tactile_t = torch.from_numpy(tactile).float()
            tacs.append(self._normalize_tactile(tactile_t))
            rgbs.append(rgb_u8)
            acts.append(self._normalize_action(action))

        rgb_seq = torch.stack(rgbs, dim=0)
        tac_seq = torch.stack(tacs, dim=0)
        act_seq = torch.stack(acts, dim=0)
        return (
            rgb_seq,
            tac_seq,
            act_seq,
            torch.tensor(task_idx, dtype=torch.long),
            torch.tensor(s, dtype=torch.long),
        )


class VTBCDataset(Dataset):
    """
    Yields (rgb_uint8_224, tactile_norm_12x64, action_norm_7).

    RGB is NOT normalized here — CLIP preprocessing happens in FrozenCLIPEncoder.
    """

    def __init__(
        self,
        split: str,
        norm_stats_path: str,
        image_size: int = 224,
    ) -> None:
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")

        self.split = split
        self.image_size = image_size

        path = norm_stats_path
        if not os.path.isfile(path):
            print(f"[data] Computing normalization stats → {path}")
            self.stats = compute_norm_stats(path)
        else:
            self.stats = load_norm_stats(path)

        self._indices_by_task = enumerate_split_indices(split)
        self._flat: List[SampleIndex] = []
        self._offsets: Dict[str, int] = {}
        cursor = 0
        for task in TASKS:
            self._offsets[task] = cursor
            lst = self._indices_by_task[task]
            self._flat.extend(lst)
            cursor += len(lst)

        self.roots = {task: _open_zarr_for_task(task) for task in TASKS}

    def __len__(self) -> int:
        return len(self._flat)

    def _normalize_action(self, a: torch.Tensor) -> torch.Tensor:
        mean = self.stats["action_mean"].to(dtype=a.dtype, device=a.device)
        std = self.stats["action_std"].to(dtype=a.dtype, device=a.device)
        return (a - mean) / std

    def _normalize_tactile(self, t: torch.Tensor) -> torch.Tensor:
        t_min = self.stats["tactile_min"].to(dtype=t.dtype, device=t.device)
        span = self.stats["tactile_span"].to(dtype=t.dtype, device=t.device)
        return torch.clamp((t - t_min) / span, 0.0, 1.0)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self._flat[idx]
        task = sample.task
        t = sample.t

        root = self.roots[task]
        rgb = np.asarray(root["data"]["camera0_rgb"][t], dtype=np.uint8)
        tactile = np.asarray(root["data"]["camera0_tactile"][t], dtype=np.float32)

        pos0 = np.asarray(root["data"]["robot0_eef_pos"][t], dtype=np.float32).reshape(-1)
        pos1 = np.asarray(root["data"]["robot0_eef_pos"][t + 1], dtype=np.float32).reshape(
            -1
        )
        r0 = np.asarray(
            root["data"]["robot0_eef_rot_axis_angle"][t], dtype=np.float32
        ).reshape(-1)
        r1 = np.asarray(
            root["data"]["robot0_eef_rot_axis_angle"][t + 1], dtype=np.float32
        ).reshape(-1)
        g0 = float(np.asarray(root["data"]["robot0_gripper_width"][t], dtype=np.float32).reshape(-1)[0])
        g1 = float(
            np.asarray(root["data"]["robot0_gripper_width"][t + 1], dtype=np.float32).reshape(-1)[0]
        )

        action = torch.from_numpy(
            np.concatenate(
                [
                    pos1 - pos0,
                    r1 - r0,
                    np.array([g1 - g0], dtype=np.float32),
                ],
                axis=0,
            )
        ).float()

        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()  # C,H,W
        rgb_t = tv_resize(rgb_t, [self.image_size, self.image_size])

        tactile_t = torch.from_numpy(tactile).float()
        tactile_n = self._normalize_tactile(tactile_t)
        action_n = self._normalize_action(action)

        # uint8 RGB for caching script / inspection
        rgb_u8 = rgb_t.byte().permute(1, 2, 0).contiguous()

        return rgb_u8, tactile_n, action_n


if __name__ == "__main__":
    norm_path = os.path.join(REPO_ROOT, "configs", "norm_stats.pt")
    print("[data] Tasks:", TASKS)
    train_eps, val_eps = count_episodes_per_split()
    print("[data] Train episodes per task:")
    for t in TASKS:
        print(f"  {t}: {train_eps[t]}")
    print("[data] Val episodes per task:")
    for t in TASKS:
        print(f"  {t}: {val_eps[t]}")

    ds_tr = VTBCDataset(split="train", norm_stats_path=norm_path)
    ds_va = VTBCDataset(split="val", norm_stats_path=norm_path)
    print(f"[data] Train transitions: {len(ds_tr)}, Val transitions: {len(ds_va)}")
    r, tac, act = ds_tr[0]
    print(f"[data] Sample RGB dtype/shape: {r.dtype}, {tuple(r.shape)}")
    print(f"[data] Sample tactile dtype/shape: {tac.dtype}, {tuple(tac.shape)}")
    print(f"[data] Sample action dtype/shape: {act.dtype}, {tuple(act.shape)}")
    assert tac.shape == (12, 64)
    assert act.shape == (7,)
