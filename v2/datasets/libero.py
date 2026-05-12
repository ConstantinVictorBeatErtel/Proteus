"""LIBERO adapter — direct HDF5 reader.

Reads LIBERO HDF5 files staged on Drive at::

    <RAID_ARTIFACT_ROOT>/data/libero/<suite>/*.hdf5

Each suite (``libero_spatial``, ``libero_object``, ``libero_goal``,
``libero_10``, ``libero_90``) contains one HDF5 per task, each with the
canonical LIBERO schema::

    data/
      demo_0/
        obs/
          agentview_rgb     (T, H, W, 3) uint8
          ee_pos / ee_ori / gripper_states / joint_states / ...
        actions             (T, 7) float32
      demo_1/ ...
      ...
    @attrs:
      problem_info: JSON with language_instruction

We pool tasks within a suite by walking files in deterministic
``sorted(...)`` order, then demos in ``demo_<int>`` order, building one
flat episode list per suite. Cached features and transitions both use
that same ordering, and the ``cache_layout.layout_checksum`` mechanism
catches any drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .stats import ActionStats, compute_action_stats


LIBERO_SUITES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
    "libero_90",
)
LIBERO_IMAGE_KEYS = ("agentview_rgb", "agentview_image", "image")


@dataclass(frozen=True)
class LiberoSpec:
    suite: str
    modality: str = "image"

    @property
    def name(self) -> str:
        return self.suite


@dataclass(frozen=True)
class LiberoEpisode:
    """One demo within one HDF5 file."""

    suite: str
    hdf5_path: Path
    demo_key: str  # e.g. demo_3
    length: int   # number of timesteps == len(actions)

    @property
    def composite_key(self) -> str:
        # Unique across (file, demo). Used as the visualization demo_key
        # so the eval panels can always pull frames back from the right
        # HDF5 file.
        return f"{self.hdf5_path.name}::{self.demo_key}"


def suite_config_name(suite: str) -> str:
    """Identity for compatibility with older callers; LIBERO suites are
    referred to by their short names everywhere now."""
    if suite not in LIBERO_SUITES:
        raise ValueError(f"unknown LIBERO suite {suite!r}")
    return suite


def libero_suite_root(libero_root: Path, suite: str) -> Path:
    """Resolve ``<libero_root>/<suite>``.

    ``libero_root`` is the directory that *contains* the suite folders
    — typically ``<RAID_ARTIFACT_ROOT>/data/libero``. To stay tolerant
    of callers that historically passed ``<artifact_root>/data``
    (without the ``libero`` segment), we also accept that and append
    the ``libero`` segment ourselves. This makes the function
    symmetric: pass either the data root or the libero root, both
    work.
    """
    p = Path(libero_root)
    direct = p / suite
    if direct.is_dir():
        return direct
    nested = p / "libero" / suite
    if nested.is_dir():
        return nested
    # Neither exists yet — fall through to the canonical layout so the
    # caller gets a clean FileNotFoundError pointing at the expected
    # location.
    return direct


def find_image_key(obs_grp: h5py.Group) -> str:
    for key in LIBERO_IMAGE_KEYS:
        if key in obs_grp:
            return key
    # Heuristic: any (T, H, W, 3|4) dataset.
    for key in obs_grp.keys():
        shape = obs_grp[key].shape
        if len(shape) == 4 and shape[-1] in (3, 4):
            return key
    raise KeyError(
        f"no image-shaped dataset found under {obs_grp.name!r}; "
        f"available keys: {list(obs_grp.keys())}"
    )


def find_libero_episodes(
    suite: str,
    libero_root: Path,
    max_demos: int | None = None,
) -> list[LiberoEpisode]:
    """Walk ``<libero_root>/<suite>/*.hdf5`` and produce a flat,
    deterministically-ordered episode list.

    ``libero_root`` is normally ``<RAID_ARTIFACT_ROOT>/data/libero`` but
    callers that pass ``<RAID_ARTIFACT_ROOT>/data`` are also accepted —
    :func:`libero_suite_root` handles both layouts.

    Order: HDF5 files sorted by filename, then ``demo_<N>`` sorted by N.
    Returning the FIRST ``max_demos`` episodes pools tasks roughly evenly
    when ``max_demos`` is much larger than the number of tasks (≈10).
    """
    suite_root = libero_suite_root(libero_root, suite)
    if not suite_root.is_dir():
        raise FileNotFoundError(
            f"missing LIBERO suite directory: {suite_root}\n"
            "Stage the HDF5 files under <RAID_ARTIFACT_ROOT>/data/libero/<suite>/"
        )
    hdf5_paths = sorted(suite_root.glob("*.hdf5"))
    if not hdf5_paths:
        # Some LIBERO releases use .h5
        hdf5_paths = sorted(suite_root.glob("*.h5"))
    if not hdf5_paths:
        raise FileNotFoundError(f"no HDF5 files in {suite_root}")

    out: list[LiberoEpisode] = []
    for hp in hdf5_paths:
        with h5py.File(hp, "r") as f:
            data_grp = f["data"]
            demo_keys = sorted(
                data_grp.keys(),
                key=lambda k: int(str(k).replace("demo_", "")) if str(k).startswith("demo_") else 0,
            )
            for dk in demo_keys:
                actions = data_grp[dk]["actions"]
                out.append(
                    LiberoEpisode(
                        suite=suite,
                        hdf5_path=hp,
                        demo_key=dk,
                        length=int(actions.shape[0]),
                    )
                )
                if max_demos is not None and len(out) >= int(max_demos):
                    return out
    return out


def episode_layout(episodes: Iterable[LiberoEpisode]) -> list[tuple[str, int]]:
    """``[(composite_key, length), ...]`` — used by the feature cache for layout verification."""
    return [(ep.composite_key, ep.length) for ep in episodes]


def iter_episode_frames(
    episodes: Iterable[LiberoEpisode],
    image_key: str | None = None,
) -> Iterator[np.ndarray]:
    """Yield every frame of every episode in order, exactly once."""
    eps = list(episodes)
    # Group consecutive episodes by file so we open each HDF5 only once.
    by_file: dict[Path, list[LiberoEpisode]] = {}
    file_order: list[Path] = []
    for ep in eps:
        if ep.hdf5_path not in by_file:
            file_order.append(ep.hdf5_path)
            by_file[ep.hdf5_path] = []
        by_file[ep.hdf5_path].append(ep)

    for hp in file_order:
        with h5py.File(hp, "r") as f:
            data_grp = f["data"]
            for ep in by_file[hp]:
                obs_grp = data_grp[ep.demo_key]["obs"]
                key = image_key or find_image_key(obs_grp)
                dset = obs_grp[key]
                T = int(dset.shape[0])
                if T != ep.length:
                    raise RuntimeError(
                        f"episode length mismatch for {ep.composite_key}: "
                        f"image dset has {T} frames but actions has {ep.length}"
                    )
                for i in range(T):
                    yield np.asarray(dset[i], dtype=np.uint8)


def read_frames(
    hdf5_path: Path,
    demo_key: str,
    timesteps: list[int] | np.ndarray,
    image_key: str | None = None,
) -> np.ndarray:
    """Slice a few specific frames out of one episode for visualization."""
    ts = sorted({int(t) for t in timesteps})
    with h5py.File(hdf5_path, "r") as f:
        obs_grp = f["data"][demo_key]["obs"]
        key = image_key or find_image_key(obs_grp)
        arr = obs_grp[key][ts, ...]
    out_order = [ts.index(int(t)) for t in timesteps]
    return arr[out_order]


def read_actions(hdf5_path: Path, demo_key: str) -> np.ndarray:
    """Per-episode action tensor as ``(T, 7) float64``."""
    with h5py.File(hdf5_path, "r") as f:
        return np.asarray(f["data"][demo_key]["actions"], dtype=np.float64)


def verify_libero_root(data_root: Path) -> dict[str, int]:
    """Return ``{suite_name: episode_count}`` for whatever suites are staged.
    Used by :mod:`v2.runtime.data_download` to confirm Drive layout."""
    out: dict[str, int] = {}
    for suite in LIBERO_SUITES:
        try:
            eps = find_libero_episodes(suite, data_root)
        except FileNotFoundError:
            continue
        out[suite] = len(eps)
    return out
