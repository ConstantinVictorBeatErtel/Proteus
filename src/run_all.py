#!/usr/bin/env python3
"""End-to-end: CLIP caching (skip if caches exist) → train 3 policies → evaluate."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SRC_DIR = REPO_ROOT / "src"
CACHE_DIR = REPO_ROOT / "data" / "clip_cache"


def _task_list() -> list[str]:
    sys.path.insert(0, str(SRC_DIR))
    import data as datamod

    return list(datamod.TASKS)


def clip_complete() -> bool:
    if not CACHE_DIR.is_dir():
        return False
    for task in _task_list():
        tr = CACHE_DIR / f"{task}_train.pt"
        va = CACHE_DIR / f"{task}_val.pt"
        if not tr.is_file() or not va.is_file():
            return False
    return True


def run_step(cmd: list[str], cwd: Path) -> None:
    print("\n" + "=" * 72)
    print("RUN:", " ".join(cmd))
    print("=" * 72 + "\n")
    t0 = time.time()
    subprocess.check_call(cmd, cwd=str(cwd))
    print(f"[run_all] step done in {time.time() - t0:.1f}s\n")


def main() -> None:
    py = sys.executable

    if not clip_complete():
        print("[run_all] CLIP caches missing/incomplete → running cache_clip.py …")
        run_step([py, str(SRC_DIR / "cache_clip.py")], REPO_ROOT)
    else:
        print("[run_all] CLIP caches already present — skipping caching.\n")

    train_script = SRC_DIR / "train.py"
    for cond in ("vision_only", "tactile_only", "visuo_tactile"):
        print(f"[run_all] Training {cond} …")
        run_step([py, str(train_script), "--condition", cond], REPO_ROOT)

    print("[run_all] Evaluation …")
    run_step([py, str(SRC_DIR / "evaluate.py")], REPO_ROOT)

    print("[run_all] Pipeline complete.")
    print("  Figures:  python notebooks/02_results.py")


if __name__ == "__main__":
    main()
