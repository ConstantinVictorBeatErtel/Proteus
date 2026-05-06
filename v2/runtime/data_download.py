"""Idempotent RoboMimic + LIBERO data download to ``<artifact_root>/data/``.

RoboMimic is fetched directly from the Stanford CDN (the same URLs the
official ``robomimic`` package uses) so the layout matches what
``v2/datasets/robomimic.hdf5_path_for`` looks for on disk.

LIBERO via ``openvla/modified_libero_rlds`` is TFDS-format and the HF
``datasets`` library cannot read it directly; this module logs a warning
and continues so phase A / B / C are not blocked.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .drive import data_root
from ..datasets.libero import LIBERO_SUITES, suite_config_name


ROBOMIMIC_TASKS = ("lift", "can", "square", "transport")
ROBOMIMIC_VARIANTS = ("ph", "mh")
ROBOMIMIC_MODALITIES = ("low_dim", "image")
ROBOMIMIC_BASE_URL = "http://downloads.cs.stanford.edu/downloads/rt_benchmark"

# Phase C / D image cells use only Square; we never download Tool Hang
# image (~58 GB) or Transport image (large + not in matrix).
ROBOMIMIC_IMAGE_TASKS = ("can", "square")

LIBERO_REPO = "openvla/modified_libero_rlds"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def legacy_robomimic_paths() -> dict[tuple[str, str, str], Path]:
    out: dict[tuple[str, str, str], Path] = {}
    base = repo_root() / "data"
    for task in ROBOMIMIC_TASKS:
        for variant in ROBOMIMIC_VARIANTS:
            for modality in ROBOMIMIC_MODALITIES:
                fname = f"{modality}_v141.hdf5"
                p = base / task / variant / fname
                if p.is_file():
                    out[(task, variant, modality)] = p
    return out


@dataclass
class DownloadResult:
    source: str
    local_path: Path
    skipped: bool
    bytes_on_disk: int


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    pct = min(100.0, 100 * block_num * block_size / total_size)
    sys.stdout.write(f"\r  {pct:5.1f}%  ({block_num*block_size/1e6:.1f} / {total_size/1e6:.1f} MB)")
    sys.stdout.flush()


def robomimic_target(task: str, variant: str, modality: str) -> Path:
    return data_root() / "robomimic" / "v1.5" / task / variant / f"{modality}_v141.hdf5"


def fetch_robomimic_file(task: str, variant: str, modality: str = "low_dim") -> DownloadResult:
    src_name = f"{modality}.hdf5"
    dst = robomimic_target(task, variant, modality)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 100_000:
        return DownloadResult(f"{task}/{variant}/{modality}", dst, skipped=True, bytes_on_disk=dst.stat().st_size)

    legacy = legacy_robomimic_paths().get((task, variant, modality))
    if legacy is not None:
        try:
            dst.symlink_to(legacy)
            return DownloadResult(f"legacy:{legacy}", dst, skipped=False, bytes_on_disk=legacy.stat().st_size)
        except OSError:
            import shutil
            shutil.copy2(legacy, dst)
            return DownloadResult(f"legacy:{legacy}", dst, skipped=False, bytes_on_disk=dst.stat().st_size)

    url = f"{ROBOMIMIC_BASE_URL}/{task}/{variant}/{src_name}"
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    print(f"[data_download] fetching {url}")
    t0 = time.time()
    try:
        urllib.request.urlretrieve(url, tmp, reporthook=_progress)
        sys.stdout.write("\n")
    except (urllib.error.URLError, OSError) as exc:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"download failed: {url} -> {exc!r}") from exc
    tmp.replace(dst)
    print(f"[data_download] wrote {dst} ({dst.stat().st_size/1e6:.1f} MB in {time.time()-t0:.1f}s)")
    return DownloadResult(url, dst, skipped=False, bytes_on_disk=dst.stat().st_size)


def download_robomimic(
    tasks: Iterable[str] = ROBOMIMIC_TASKS,
    variants: Iterable[str] = ROBOMIMIC_VARIANTS,
    modalities: Iterable[str] = ("low_dim",),
) -> list[DownloadResult]:
    results: list[DownloadResult] = []
    for task in tasks:
        for variant in variants:
            for modality in modalities:
                if modality == "image" and task not in ROBOMIMIC_IMAGE_TASKS:
                    continue
                try:
                    r = fetch_robomimic_file(task, variant, modality)
                    results.append(r)
                    print(
                        f"[data_download] {'SKIP' if r.skipped else 'OK  '} "
                        f"{task}/{variant}/{modality} ({r.bytes_on_disk/1e6:.1f} MB)"
                    )
                except RuntimeError as exc:
                    print(f"[data_download] FAIL {task}/{variant}/{modality}: {exc}")
    return results


def libero_cache_dir() -> Path:
    p = data_root() / "libero"
    p.mkdir(parents=True, exist_ok=True)
    return p


def download_libero(suites: Iterable[str] = LIBERO_SUITES) -> list[DownloadResult]:
    cache = libero_cache_dir()
    results: list[DownloadResult] = []
    try:
        from datasets import load_dataset
    except ImportError:
        print("[data_download] datasets not installed; skipping LIBERO")
        return results

    for suite in suites:
        try:
            load_dataset(LIBERO_REPO, name=suite_config_name(suite), cache_dir=str(cache), split="train")
        except Exception as exc:
            print(
                f"[data_download] LIBERO {suite} skipped: {exc.__class__.__name__}: {exc}\n"
                "  -> openvla/modified_libero_rlds is TFDS-format. To enable LIBERO\n"
                "     phase E, switch to a parquet-format mirror or install\n"
                "     tensorflow-datasets and write a TFDS reader."
            )
    return results


def ensure_all_data(
    tasks: Iterable[str] = ROBOMIMIC_TASKS,
    variants: Iterable[str] = ROBOMIMIC_VARIANTS,
    modalities: Iterable[str] = ("low_dim",),
    include_libero: bool = False,
) -> None:
    download_robomimic(tasks=tasks, variants=variants, modalities=modalities)
    if include_libero:
        download_libero()
    else:
        print("[data_download] LIBERO skipped (pass --include-libero to fetch)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=list(ROBOMIMIC_TASKS), choices=list(ROBOMIMIC_TASKS))
    ap.add_argument("--variants", nargs="+", default=list(ROBOMIMIC_VARIANTS), choices=list(ROBOMIMIC_VARIANTS))
    ap.add_argument("--modalities", nargs="+", default=["low_dim"], choices=list(ROBOMIMIC_MODALITIES))
    ap.add_argument("--include-libero", action="store_true")
    args = ap.parse_args()
    ensure_all_data(
        tasks=args.tasks,
        variants=args.variants,
        modalities=args.modalities,
        include_libero=args.include_libero,
    )


if __name__ == "__main__":
    main()