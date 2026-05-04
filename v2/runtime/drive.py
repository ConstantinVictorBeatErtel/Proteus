"""Drive mount, artifact-root resolution, atomic checkpoints.

Resolution order for the artifact root:
  1. ``$RAID_ARTIFACT_ROOT`` if set
  2. ``/content/drive/MyDrive/raid_v2`` if running on Colab with Drive mounted
  3. ``./artifacts`` relative to the repo root

All checkpoint writes go through :func:`atomic_save`, which writes to a
``.tmp`` sibling and then ``os.replace``s it onto the target path so a crash
between writes never leaves a half-written file on Drive.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

_COLAB_DRIVE_DEFAULT = "/content/drive/MyDrive/raid_v2"


def _running_on_colab() -> bool:
    try:
        import google.colab  # type: ignore  # noqa: F401
        return True
    except Exception:
        return False


def mount_drive(force: bool = False) -> Path | None:
    """Mount Google Drive if running on Colab. No-op otherwise.

    Returns the mount point (``/content/drive``) on success, ``None`` if not
    on Colab or the mount failed.
    """
    if not _running_on_colab():
        return None
    try:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive", force_remount=force)
        return Path("/content/drive")
    except Exception as exc:  # noqa: BLE001
        print(f"[drive] mount failed: {exc}")
        return None


def artifact_root() -> Path:
    """Resolve the artifact root. Creates it if missing."""
    env = os.environ.get("RAID_ARTIFACT_ROOT")
    if env:
        root = Path(env).expanduser()
    elif Path(_COLAB_DRIVE_DEFAULT).parent.exists():
        root = Path(_COLAB_DRIVE_DEFAULT)
    else:
        root = Path(__file__).resolve().parents[2] / "artifacts"
    root.mkdir(parents=True, exist_ok=True)
    return root


def data_root() -> Path:
    p = artifact_root() / "data"
    p.mkdir(parents=True, exist_ok=True)
    return p


def features_root() -> Path:
    p = artifact_root() / "features"
    p.mkdir(parents=True, exist_ok=True)
    return p


def runs_root() -> Path:
    p = artifact_root() / "runs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def results_root() -> Path:
    p = artifact_root() / "results"
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_save(state: Any, path: str | Path) -> Path:
    """Torch-save ``state`` atomically. Returns the final path."""
    final = Path(path)
    final.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=final.stem + "_", suffix=".tmp", dir=final.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        torch.save(state, tmp)
        os.replace(tmp, final)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return final


def atomic_write_bytes(data: bytes, path: str | Path) -> Path:
    """Write ``data`` to ``path`` atomically."""
    final = Path(path)
    final.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=final.stem + "_", suffix=".tmp", dir=final.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp_name, final)
    finally:
        if Path(tmp_name).exists():
            try:
                Path(tmp_name).unlink()
            except OSError:
                pass
    return final


def keep_last_n_checkpoints(directory: str | Path, n: int = 3, pattern: str = "ckpt_*.pt") -> int:
    """Drop all but the ``n`` newest checkpoint files matching ``pattern``.

    Files named ``ckpt_best.pt`` or ``ckpt_last.pt`` are pinned (never deleted).
    Returns the number of files deleted.
    """
    d = Path(directory)
    if not d.is_dir():
        return 0
    pinned = {"ckpt_best.pt", "ckpt_last.pt"}
    files = [p for p in d.glob(pattern) if p.name not in pinned]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    to_delete = files[n:]
    for p in to_delete:
        try:
            p.unlink()
        except OSError:
            pass
    return len(to_delete)


@dataclass
class CheckpointDir:
    """Per-run checkpoint directory under ``runs/<run_id>/``."""

    run_id: str

    @property
    def path(self) -> Path:
        p = runs_root() / self.run_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def best(self) -> Path:
        return self.path / "ckpt_best.pt"

    def last(self) -> Path:
        return self.path / "ckpt_last.pt"

    def step(self, step: int) -> Path:
        return self.path / f"ckpt_{step:08d}.pt"

    def run_id_file(self) -> Path:
        return self.path / "run_id.txt"

    def metrics_file(self) -> Path:
        return self.path / "metrics.parquet"


def free_disk_bytes(path: str | Path = "/") -> int:
    """Bytes free on the device hosting ``path``."""
    return shutil.disk_usage(str(path)).free
