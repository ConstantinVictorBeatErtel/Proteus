"""Idempotent dataset downloads to ``<artifact_root>/data/``.

All downloads check Drive presence first and skip if the expected files are
already there with a non-trivial size. Re-running on a fresh Colab session
re-mounts Drive and immediately becomes a no-op.

Datasets in scope (action-space-identical, 7-D OSC_POSE @ 20 Hz):

* RoboMimic v0.1 PH+MH HDF5s for Lift / Can / Square / Transport
  (low_dim variants for all four; image variants for Can and Square only).
* LIBERO via the OpenVLA-aligned RLDS at ``openvla/modified_libero_rlds``
  for the spatial / object / goal suites.

Tool Hang image is intentionally skipped (~58 GB). Transport image is also
skipped to stay under the ~22 GB Drive budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .drive import data_root


# RoboMimic v0.1 — published on the HF mirror ``amandlek/robomimic`` under
# ``v1.5/<task>/<variant>/<modality>_v141.hdf5``.
ROBOMIMIC_TASKS = ("lift", "can", "square", "transport")
ROBOMIMIC_VARIANTS = ("ph", "mh")
ROBOMIMIC_LOWDIM = "low_dim_v141.hdf5"
ROBOMIMIC_IMAGE = "image_v141.hdf5"
ROBOMIMIC_REPO = "amandlek/robomimic"

# Image variants only kept for Can (1.9 GB) and Square (5.3 GB).
ROBOMIMIC_IMAGE_TASKS = ("can", "square")

# LIBERO via OpenVLA-aligned RLDS.
LIBERO_REPO = "openvla/modified_libero_rlds"
LIBERO_SUITES = ("libero_spatial", "libero_object", "libero_goal")


@dataclass
class DownloadResult:
    repo_id: str
    local_dir: Path
    skipped: bool
    bytes_on_disk: int


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def robomimic_allow_patterns(include_image: bool = True) -> list[str]:
    pats: list[str] = []
    for task in ROBOMIMIC_TASKS:
        for variant in ROBOMIMIC_VARIANTS:
            pats.append(f"v1.5/{task}/{variant}/{ROBOMIMIC_LOWDIM}")
    if include_image:
        for task in ROBOMIMIC_IMAGE_TASKS:
            for variant in ROBOMIMIC_VARIANTS:
                pats.append(f"v1.5/{task}/{variant}/{ROBOMIMIC_IMAGE}")
    return pats


def download_robomimic(include_image: bool = True, min_bytes: int = 100_000_000) -> DownloadResult:
    """Snapshot-download RoboMimic v0.1 to ``<artifact_root>/data/robomimic``.

    Skips download if the directory already contains at least ``min_bytes``.
    """
    target = data_root() / "robomimic"
    target.mkdir(parents=True, exist_ok=True)
    existing = _dir_size(target)
    if existing >= min_bytes:
        return DownloadResult(ROBOMIMIC_REPO, target, skipped=True, bytes_on_disk=existing)

    from huggingface_hub import snapshot_download  # local import to keep module light

    snapshot_download(
        repo_id=ROBOMIMIC_REPO,
        repo_type="dataset",
        local_dir=str(target),
        allow_patterns=robomimic_allow_patterns(include_image=include_image),
        local_dir_use_symlinks=False,
    )
    return DownloadResult(ROBOMIMIC_REPO, target, skipped=False, bytes_on_disk=_dir_size(target))


def libero_cache_dir() -> Path:
    p = data_root() / "libero"
    p.mkdir(parents=True, exist_ok=True)
    return p


def download_libero(suites: Iterable[str] = LIBERO_SUITES) -> list[DownloadResult]:
    """Pull LIBERO suites from ``openvla/modified_libero_rlds`` into the HF cache.

    Uses ``datasets.load_dataset`` so the parquet shards live in
    ``<artifact_root>/data/libero/<suite>/...``.
    """
    from datasets import load_dataset  # local import

    cache = libero_cache_dir()
    out: list[DownloadResult] = []
    for suite in suites:
        suite_dir = cache / suite
        existing = _dir_size(suite_dir)
        if existing > 100_000_000:
            out.append(DownloadResult(LIBERO_REPO, suite_dir, skipped=True, bytes_on_disk=existing))
            continue
        load_dataset(LIBERO_REPO, name=suite, cache_dir=str(cache))
        out.append(DownloadResult(LIBERO_REPO, suite_dir, skipped=False, bytes_on_disk=_dir_size(suite_dir)))
    return out


def ensure_all_data(include_image: bool = True) -> None:
    rm = download_robomimic(include_image=include_image)
    print(
        f"[data_download] robomimic: {'SKIPPED' if rm.skipped else 'DOWNLOADED'} "
        f"-> {rm.local_dir} ({rm.bytes_on_disk / 1e9:.2f} GB)"
    )
    libs = download_libero()
    for r in libs:
        print(
            f"[data_download] libero {r.local_dir.name}: {'SKIPPED' if r.skipped else 'DOWNLOADED'} "
            f"-> {r.local_dir} ({r.bytes_on_disk / 1e9:.2f} GB)"
        )


if __name__ == "__main__":
    ensure_all_data()
