"""
LIBERO dataset loader for RAID visual integration.

Loads LIBERO HDF5 files and returns:
  - image_t:    (H, W, 3) uint8  current frame
  - image_next: (H, W, 3) uint8  next frame
  - action:     (7,)      float32 normalised action
  - language:   str       task language instruction

Each LIBERO task has one HDF5 file with 50 demonstrations.
LIBERO-Spatial has 10 tasks → 500 episodes.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Observation image key names used in LIBERO HDF5 files
# ---------------------------------------------------------------------------
# LIBERO stores images at:  data/demo_X/obs/agentview_rgb  (or agentview_image)
# We'll try both and use whichever exists.
IMG_KEY_CANDIDATES = ["agentview_rgb", "agentview_image", "image"]


def _find_img_key(obs_group: h5py.Group) -> str:
    for k in IMG_KEY_CANDIDATES:
        if k in obs_group:
            return k
    # fallback: first key that looks like an image (has 3 dims + last dim 3 or 4)
    for k in obs_group.keys():
        shape = obs_group[k].shape
        if len(shape) >= 3 and shape[-1] in (3, 4):
            return k
    raise KeyError(f"No image key found in obs group. Keys: {list(obs_group.keys())}")


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def compute_norm_stats(hdf5_paths: list[str], max_demos: int = 999999) -> dict:
    """Compute per-dim mean/std over actions across all demos."""
    all_actions = []
    count = 0
    for path in hdf5_paths:
        with h5py.File(path, "r") as f:
            demos = sorted(f["data"].keys(),
                           key=lambda x: int(x.replace("demo_", "")))
            for demo in demos:
                if count >= max_demos:
                    break
                actions = f[f"data/{demo}/actions"][:]  # (T, 7)
                all_actions.append(actions)
                count += 1
        if count >= max_demos:
            break
    all_actions = np.concatenate(all_actions, axis=0)  # (N, 7)
    action_mean = all_actions.mean(axis=0).astype(np.float32)
    action_std  = all_actions.std(axis=0).astype(np.float32)
    action_std  = np.where(action_std < 1e-6, 1.0, action_std)
    return {
        "action_mean": torch.tensor(action_mean),
        "action_std":  torch.tensor(action_std),
    }


def save_norm_stats(stats: dict, path: str | Path) -> None:
    torch.save(stats, path)


def load_norm_stats(path: str | Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LiberoTransitionDataset(Dataset):
    """
    Flat dataset of (image_t, image_next, action_norm, language) transitions.

    Each index corresponds to one (t, t+1) step from any demo.
    """

    def __init__(
        self,
        hdf5_paths: list[str],
        norm_stats: dict,
        n_demos: Optional[int] = None,
        image_size: int = 128,
        skip_last: bool = True,
    ):
        self.norm_stats = norm_stats
        self.image_size = image_size

        self.images_t:    list[np.ndarray] = []   # (H, W, 3) uint8
        self.images_next: list[np.ndarray] = []
        self.actions:     list[np.ndarray] = []   # (7,) float32 normalised
        self.languages:   list[str]        = []

        action_mean = norm_stats["action_mean"].numpy()
        action_std  = norm_stats["action_std"].numpy()

        total_demos = 0
        for path in hdf5_paths:
            with h5py.File(path, "r") as f:
                # Language instruction stored in dataset attributes
                problem_info = json.loads(f["data"].attrs.get("problem_info", "{}"))
                lang = problem_info.get("language_instruction", "robot manipulation")
                lang = lang.strip('"\'')

                demos = sorted(f["data"].keys(),
                               key=lambda x: int(x.replace("demo_", "")))
                for demo in demos:
                    if n_demos is not None and total_demos >= n_demos:
                        break
                    grp = f[f"data/{demo}"]
                    obs_grp = grp["obs"]
                    img_key = _find_img_key(obs_grp)

                    images  = obs_grp[img_key][:]          # (T, H, W, 3) uint8
                    actions = grp["actions"][:]             # (T, 7) float32

                    T = len(actions)
                    end = T - 1 if skip_last else T

                    for t in range(end):
                        img_t    = images[t]
                        img_next = images[t + 1] if t + 1 < len(images) else images[t]
                        act      = (actions[t] - action_mean) / (action_std + 1e-8)

                        self.images_t.append(img_t)
                        self.images_next.append(img_next)
                        self.actions.append(act.astype(np.float32))
                        self.languages.append(lang)

                    total_demos += 1

                if n_demos is not None and total_demos >= n_demos:
                    break

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, idx: int) -> dict:
        return {
            "image_t":    torch.from_numpy(self.images_t[idx]),     # (H, W, 3) uint8
            "image_next": torch.from_numpy(self.images_next[idx]),  # (H, W, 3) uint8
            "action":     torch.from_numpy(self.actions[idx]),      # (7,)
            "language":   self.languages[idx],
        }


# ---------------------------------------------------------------------------
# Cached feature dataset (used after pre-computing GR-1 features)
# ---------------------------------------------------------------------------

class CachedFeatureDataset(Dataset):
    """
    Dataset that serves pre-computed GR-1 features.

    Expected cache file format (torch .pt dict):
        {
          "feat_t":    (N, 384) float32
          "feat_next": (N, 384) float32
          "actions":   (N, 7)   float32  normalised
          "languages": list[str] length N
        }
    """

    def __init__(self, cache_path: str | Path, n_demos: Optional[int] = None):
        data = torch.load(cache_path, map_location="cpu", weights_only=False)
        feat_t    = data["feat_t"]
        feat_next = data["feat_next"]
        actions   = data["actions"]

        # Optionally limit to first n_demos worth of data.
        # We store demo boundaries in the cache if available.
        if n_demos is not None and "demo_lengths" in data:
            lengths = data["demo_lengths"][:n_demos]
            end_idx = int(sum(lengths))
            feat_t    = feat_t[:end_idx]
            feat_next = feat_next[:end_idx]
            actions   = actions[:end_idx]

        self.feat_t    = feat_t.float()
        self.feat_next = feat_next.float()
        self.actions   = actions.float()
        self.feat_dim  = feat_t.shape[1]

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.feat_t[idx], self.feat_next[idx], self.actions[idx]


# ---------------------------------------------------------------------------
# Utility: find all HDF5 files in a LIBERO dataset directory
# ---------------------------------------------------------------------------

def find_hdf5_files(dataset_dir: str | Path) -> list[str]:
    """Recursively find all .hdf5 files in a directory."""
    pattern = str(Path(dataset_dir) / "**" / "*.hdf5")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        # also try .h5
        pattern = str(Path(dataset_dir) / "**" / "*.h5")
        files = sorted(glob.glob(pattern, recursive=True))
    return files


def make_train_val_split(
    hdf5_paths: list[str],
    norm_stats: dict,
    n_demos: int,
    val_frac: float = 0.2,
) -> tuple[CachedFeatureDataset | LiberoTransitionDataset,
           CachedFeatureDataset | LiberoTransitionDataset]:
    """Split demos 80/20 train/val (used before caching)."""
    n_val  = max(1, int(n_demos * val_frac))
    n_train = n_demos - n_val

    # Val uses last n_val demos (approximate: just pick last files / demos)
    all_demos = []
    for path in hdf5_paths:
        with h5py.File(path, "r") as f:
            demo_keys = sorted(f["data"].keys(),
                               key=lambda x: int(x.replace("demo_", "")))
            all_demos.extend([(path, dk) for dk in demo_keys])

    all_demos = all_demos[:n_demos]
    train_demos = all_demos[:n_train]
    val_demos   = all_demos[n_train:]

    def _build(demo_list):
        ds = LiberoTransitionDataset.__new__(LiberoTransitionDataset)
        ds.norm_stats  = norm_stats
        ds.image_size  = 128
        ds.images_t    = []
        ds.images_next = []
        ds.actions     = []
        ds.languages   = []
        action_mean = norm_stats["action_mean"].numpy()
        action_std  = norm_stats["action_std"].numpy()

        for path, demo in demo_list:
            with h5py.File(path, "r") as f:
                problem_info = json.loads(f["data"].attrs.get("problem_info", "{}"))
                lang = problem_info.get("language_instruction", "robot manipulation").strip('"\'')
                grp     = f[f"data/{demo}"]
                obs_grp = grp["obs"]
                img_key = _find_img_key(obs_grp)
                images  = obs_grp[img_key][:]
                actions = grp["actions"][:]
                for t in range(len(actions) - 1):
                    act = (actions[t] - action_mean) / (action_std + 1e-8)
                    ds.images_t.append(images[t])
                    ds.images_next.append(images[t + 1])
                    ds.actions.append(act.astype(np.float32))
                    ds.languages.append(lang)
        return ds

    return _build(train_demos), _build(val_demos)


# ---------------------------------------------------------------------------
# Cached V-JEPA 2 feature dataset
# ---------------------------------------------------------------------------

class CachedVJEPADataset(torch.utils.data.Dataset):
    """
    Dataset serving pre-computed V-JEPA 2 features from cached .pt files.

    Each cache file (from cache_vjepa_features.py) contains:
        feat_t:      (N, 1024) float32
        feat_next:   (N, 1024) float32
        actions:     (N, 7)    float32 (raw, unnormalised)
        episode_ids: (N,)      int64

    The train/val split is done by episode_id to avoid leakage.
    """

    def __init__(
        self,
        feature_files: list[str | Path],
        norm_stats: dict | None = None,
        n_demos: int | None = None,
        split: str = "train",
        train_ratio: float = 0.8,
        seed: int = 42,
    ):
        self.split = split
        self.train_ratio = train_ratio

        # Load all cache files and concatenate.
        self.feature_files = [str(fp) for fp in feature_files]
        all_feat_t = []
        all_feat_next = []
        all_actions = []
        all_episode_ids = []

        for fp in feature_files:
            data = torch.load(fp, map_location="cpu", weights_only=False)
            ft  = data["feat_t"]
            fn  = data["feat_next"]
            act = data["actions"]
            eid = data.get("episode_ids",
                            torch.zeros(len(ft), dtype=torch.long))

            all_feat_t.append(ft)
            all_feat_next.append(fn)
            all_actions.append(act)
            all_episode_ids.append(eid)

        self.all_feat_t      = torch.cat(all_feat_t, dim=0)
        self.all_feat_next   = torch.cat(all_feat_next, dim=0)
        self.all_actions     = torch.cat(all_actions, dim=0)
        self.all_episode_ids = torch.cat(all_episode_ids, dim=0)

        # Subsample demos if requested.
        # episode_ids are LOCAL per task file (0..49 each).
        # After concatenation we must use GLOBAL unique episode indices.
        if n_demos is not None:
            global_feat_t = []
            global_feat_next = []
            global_actions = []
            global_episode_ids = []
            global_ep_counter = 0

            for fp in feature_files:
                data = torch.load(fp, map_location="cpu", weights_only=False)
                ft  = data["feat_t"]
                fn  = data["feat_next"]
                act = data["actions"]
                eid = data.get("episode_ids",
                                torch.zeros(len(ft), dtype=torch.long))
                local_eps = eid.unique().sort()[0]
                for local_ep in local_eps:
                    if global_ep_counter >= n_demos:
                        break
                    mask_ep = eid == local_ep
                    global_feat_t.append(ft[mask_ep])
                    global_feat_next.append(fn[mask_ep])
                    global_actions.append(act[mask_ep])
                    n_trans = mask_ep.sum().item()
                    global_episode_ids.append(
                        torch.full((n_trans,), global_ep_counter, dtype=torch.long))
                    global_ep_counter += 1
                if global_ep_counter >= n_demos:
                    break

            self.all_feat_t      = torch.cat(global_feat_t, dim=0)
            self.all_feat_next   = torch.cat(global_feat_next, dim=0)
            self.all_actions     = torch.cat(global_actions, dim=0)
            self.all_episode_ids = torch.cat(global_episode_ids, dim=0)

        self.feat_dim   = 1024
        self.action_dim = 7

        # Split by episode ID deterministically.
        g = torch.Generator()
        g.manual_seed(seed)
        unique_eps = self.all_episode_ids.unique()
        n_total = len(unique_eps)
        n_train = max(1, int(train_ratio * n_total))
        perm = torch.randperm(n_total, generator=g)
        train_eps = set(unique_eps[perm[:n_train]].tolist())

        if split == "train":
            mask = torch.tensor([int(e) in train_eps for e in self.all_episode_ids])
        else:
            mask = torch.tensor([int(e) not in train_eps for e in self.all_episode_ids])

        idx = torch.where(mask)[0]
        self.feat_t    = self.all_feat_t[idx]
        self.feat_next = self.all_feat_next[idx]
        self.actions   = self.all_actions[idx]

        # Normalise actions if stats provided.
        if norm_stats is not None:
            self.action_mean = norm_stats["action_mean"].float()
            self.action_std  = norm_stats["action_std"].float()
            self.actions = (self.actions - self.action_mean) / (
                self.action_std.clamp(min=1e-8))
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std  = torch.ones(self.action_dim)

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, index: int):
        return (self.feat_t[index].float(),
                self.feat_next[index].float(),
                self.actions[index].float())


def make_train_val_vjepa(
    feature_dir: str | Path,
    n_demos: int | None = None,
    seed: int = 42,
) -> tuple[CachedVJEPADataset, CachedVJEPADataset]:
    """Build train/val CachedVJEPADataset from a directory of .pt cache files."""
    feature_dir = Path(feature_dir)
    feature_files = sorted(feature_dir.glob("*.pt"))
    if not feature_files:
        raise FileNotFoundError(f"No .pt feature files in {feature_dir}")

    # Compute norm stats from train split only (via a preliminary pass).
    # Build a temporary full dataset so we can derive stats from its train indices.
    temp = CachedVJEPADataset(feature_files, norm_stats=None,
                               n_demos=n_demos, split="train", seed=seed)
    norm_stats = {
        "action_mean": temp.actions.mean(dim=0),
        "action_std":  temp.actions.std(dim=0).clamp(min=1e-8),
    }

    train_ds = CachedVJEPADataset(feature_files, norm_stats=norm_stats,
                                   n_demos=n_demos, split="train", seed=seed)
    val_ds   = CachedVJEPADataset(feature_files, norm_stats=norm_stats,
                                   n_demos=n_demos, split="val", seed=seed)
    return train_ds, val_ds
