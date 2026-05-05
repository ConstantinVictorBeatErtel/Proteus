"""Run training for all RAID / direct scale conditions then evaluation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCALES = (25, 50, 100, 200)
CONDITIONS = ("direct_mlp", "raid", "raid_crossattn")


def main() -> None:
    py = sys.executable
    for n in SCALES:
        for cond in CONDITIONS:
            cmd = [py, str(PROJECT_ROOT / "src" / "train.py"), "--condition", cond, "--n_demos", str(n)]
            print(f"[run_all] START {' '.join(cmd)}", flush=True)
            subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
            print(f"[run_all] DONE  cond={cond} n={n}", flush=True)

    ev = [py, str(PROJECT_ROOT / "src" / "evaluate.py")]
    print(f"[run_all] START {' '.join(ev)}", flush=True)
    subprocess.run(ev, cwd=PROJECT_ROOT, check=True)
    print("[run_all] finished evaluate.py", flush=True)


if __name__ == "__main__":
    main()
