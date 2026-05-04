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

If a legacy local checkout already provides the HDF5 under
``<repo_root>/data/<task>/<variant>/<modality>_v141.hdf5`` (the layout
``src/data.py`` reads), the downloader detects it and skips the network
request entirely.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .drive import data_root


# RoboMimic v0.1 — the official ARISE-Initiative HF repo. The legacy
# RAID checkout ships with ``data/lift/ph/low_dim_v141.hdf5`` already on
# disk so most users skip download entirely.
ROBOMIMIC_TASKS = ("lift", "can", "square", "transport")
ROBOMIMIC_VARIANTS = ("ph", "mh")
ROBOMIMIC_LOWDIM = "low_dim_v141.hdf5"
ROBOMIMIC_IMAGE = "image_v141.hdf5"

# Candidate HF repos in priority order. The first that has the file
# wins. Resolved at runtime by trying each ``hf_hub_download`` call.
ROBOMIMIC_REPO_CANDIDATES: tuple[str, ...] = (
    "amandlek/robomimic",
    "amandlek/robomimic-paper",
    "ARISE-Initiative/robomimic",
)

# Image variants only kept for Can (1.9 GB) and Square (5.3 GB).
ROBOMIMIC_IMAGE_TASKS = ("can", "square")

# LIBERO via OpenVLA-aligned RLDS.
LIBERO_REPO = "openvla/modified_libero_rlds"
LIBERO_SUITES = ("libero_spatial", "libero_object", "libero_goal")


def repo_root() -> Path:
    """Path to the repository root (parent of the ``v2`` subtree)."""
    return Path(__file__).resolve().parents[2]


def legacy_robomimic_paths() -> dict[tuple[str, str, str], Path]:
    """Map ``(task, variant, modality)`` to a legacy local HDF5 path."""
    out: dict[tuple[str, str, str], Path] = {}
    base = repo_root() / "data"
    for task in ROBOMIMIC_TASKS:
        for variant in ROBOMIMIC_VARIANTS:
            for modality, fname in (("low_dim", ROBOMIMIC_LOWDIM), ("image", ROBOMIMIC_IMAGE)):
                p = base / task / variant / fname
                if p.is_file():
                    out[(task, variant, modality)] = p
    return out


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


def _link_legacy(target_root: Path) -> int:
    """Symlink any legacy ``<repo>/data/<task>/<variant>/<file>`` into the
    Drive layout so ``hdf5_path_for`` finds them in either tree.

    Returns the number of links established.
    """
    legacy = legacy_robomimic_paths()
    if not legacy:
        return 0
    n = 0
    for (task, variant, modality), src in legacy.items():
        dst = target_root / "v1.5" / task / variant / src.name
        if dst.exists() or dst.is_symlink():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            dst.symlink_to(src)
            n += 1
        except OSError:
            # symlinks unsupported (some Drive mounts on Windows) -> copy.
            import shutil

            shutil.copy2(src, dst)
            n += 1
    return n


def download_robomimic(include_image: bool = True, min_bytes: int = 100_000_000) -> DownloadResult:
    """Stage RoboMimic v0.1 under ``<artifact_root>/data/robomimic``.

    Resolution order:

    1. If the directory already holds at least ``min_bytes``, skip
       (idempotent).
    2. If a legacy local checkout has any RoboMimic HDF5s, link them
       into the Drive layout and report success.
    3. Otherwise try ``huggingface_hub.snapshot_download`` against each
       repo in :data:`ROBOMIMIC_REPO_CANDIDATES` until one yields the
       expected files. The last attempted repo id is recorded in the
       result so a misconfigured environment is easy to debug.
    """
    target = data_root() / "robomimic"
    target.mkdir(parents=True, exist_ok=True)
    existing = _dir_size(target)
    if existing >= min_bytes:
        return DownloadResult("(cache)", target, skipped=True, bytes_on_disk=existing)

    linked = _link_legacy(target)
    if linked > 0:
        bytes_now = _dir_size(target)
        print(f"[data_download] linked {linked} legacy RoboMimic file(s) into {target}")
        if bytes_now >= min_bytes:
            return DownloadResult("(legacy)", target, skipped=True, bytes_on_disk=bytes_now)

    from huggingface_hub import snapshot_download  # local import to keep module light
    from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

    last_repo = ROBOMIMIC_REPO_CANDIDATES[0]
    last_err: Exception | None = None
    for repo_id in ROBOMIMIC_REPO_CANDIDATES:
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(target),
                allow_patterns=robomimic_allow_patterns(include_image=include_image),
                local_dir_use_symlinks=False,
            )
            last_repo = repo_id
            last_err = None
            break
        except (RepositoryNotFoundError, HfHubHTTPError, FileNotFoundError, OSError) as exc:
            print(f"[data_download] {repo_id}: {exc.__class__.__name__}: {exc}; trying next candidate")
            last_err = exc
            continue
    if last_err is not None:
        raise RuntimeError(
            "RoboMimic download failed across all candidates "
            f"{ROBOMIMIC_REPO_CANDIDATES}; last error: {last_err!r}. "
            "If you have the HDF5s locally, place them under "
            "``<repo>/data/<task>/<variant>/<modality>_v141.hdf5`` and re-run."
        )

    return DownloadResult(last_repo, target, skipped=False, bytes_on_disk=_dir_size(target))


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
