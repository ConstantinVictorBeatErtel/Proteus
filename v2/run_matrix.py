"""Idempotent matrix orchestrator.

Reads ``v2/configs/matrix.yaml``, expands each phase's row into a list of
:class:`v2.train.CellConfig` instances, and dispatches one cell at a time
on the available GPU. Each cell is keyed by a deterministic ``run_id``;
re-running with a config that has already produced a ``ckpt_best.pt`` is
a no-op (so a Colab disconnect is recoverable).

Use ``--dry-run`` to enumerate cells without launching them. Use
``--phase <PHASE>`` to limit execution to a subset.
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import yaml

from .runtime.drive import CheckpointDir, results_root
from .train import CellConfig, train_cell


REPO_ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = REPO_ROOT / "v2" / "configs" / "matrix.yaml"


def _expand_phase(phase: dict) -> list[CellConfig]:
    cells: list[CellConfig] = []
    pid = phase["id"]
    heads = phase["heads"]
    datasets = phase["datasets"]
    n_demos_list = phase.get("n_demos", [25])
    seeds = phase.get("seeds", [42])
    encoders = phase.get("encoders", [None])

    for head, dataset, n_demos, seed, encoder in product(heads, datasets, n_demos_list, seeds, encoders):
        cells.append(
            CellConfig(
                phase=pid,
                head=head,
                dataset=dataset,
                n_demos=int(n_demos),
                seed=int(seed),
                encoder=encoder,
                n_epochs=int(phase.get("epochs", 50)),
                batch_size=int(phase.get("batch_size", 256)),
                lr=float(phase.get("lr", 1e-3)),
                weight_decay=float(phase.get("weight_decay", 1e-4)),
            )
        )
    return cells


def expand_matrix(matrix_path: Path = MATRIX_PATH, only_phase: str | None = None) -> list[CellConfig]:
    raw = yaml.safe_load(matrix_path.read_text())
    cells: list[CellConfig] = []
    for phase in raw["phases"]:
        if only_phase is not None and phase["id"] != only_phase:
            continue
        cells.extend(_expand_phase(phase))
    return cells


def _completed(cfg: CellConfig) -> bool:
    return CheckpointDir(cfg.run_id).best().is_file()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default=str(MATRIX_PATH))
    ap.add_argument("--phase", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--project", default="raid_v2")
    args = ap.parse_args()

    cells = expand_matrix(Path(args.matrix), only_phase=args.phase)
    print(f"[run_matrix] expanded {len(cells)} cells (phase filter: {args.phase})")

    if args.dry_run:
        for c in cells:
            print(json.dumps(
                {
                    "run_id": c.run_id, "phase": c.phase, "head": c.head,
                    "dataset": c.dataset, "n_demos": c.n_demos, "seed": c.seed,
                    "encoder": c.encoder, "completed": _completed(c),
                }
            ))
        return

    started = time.time()
    for i, cfg in enumerate(cells, 1):
        if _completed(cfg):
            print(f"[run_matrix] [{i}/{len(cells)}] SKIP completed run_id={cfg.run_id}")
            continue
        print(f"[run_matrix] [{i}/{len(cells)}] START phase={cfg.phase} head={cfg.head} dataset={cfg.dataset} n_demos={cfg.n_demos} seed={cfg.seed} run_id={cfg.run_id}")
        out = train_cell(cfg, project=args.project)
        print(f"[run_matrix] DONE  best_val_mse={out['best_val_mse']:.6f} elapsed={time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
