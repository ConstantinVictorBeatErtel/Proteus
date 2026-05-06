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
    action_norm = phase.get("action_norm", "zscore")
    strides = phase.get("strides", [int(phase.get("stride", 1))])

    for head, dataset, n_demos, seed, encoder, stride in product(
        heads, datasets, n_demos_list, seeds, encoders, strides,
    ):
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
                action_norm_mode=str(action_norm),
                stride=int(stride),
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
    ap.add_argument("--no-eval", action="store_true",
                    help="Skip the per-cell evaluation pass (no prediction PNGs).")
    ap.add_argument("--n-panels", type=int, default=8)
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
                    "encoder": c.encoder,
                    "action_norm_mode": c.action_norm_mode, "completed": _completed(c),
                }
            ))
        return

    started = time.time()
    n_failed = 0
    for i, cfg in enumerate(cells, 1):
        if _completed(cfg):
            print(f"[run_matrix] [{i}/{len(cells)}] SKIP completed run_id={cfg.run_id}")
            if not args.no_eval:
                _safe_eval(cfg.run_id, args.n_panels)
            continue
        print(
            f"[run_matrix] [{i}/{len(cells)}] START phase={cfg.phase} head={cfg.head} "
            f"dataset={cfg.dataset} n_demos={cfg.n_demos} seed={cfg.seed} "
            f"action_norm={cfg.action_norm_mode} run_id={cfg.run_id}"
        )
        try:
            out = train_cell(cfg, project=args.project)
        except Exception as exc:  # noqa: BLE001 — never let one cell kill the matrix
            n_failed += 1
            print(f"[run_matrix] FAIL  run_id={cfg.run_id}: {exc!r}")
            continue
        print(f"[run_matrix] DONE  best_val_mse={out['best_val_mse']:.6f} elapsed={time.time() - started:.0f}s")
        if not args.no_eval:
            _safe_eval(cfg.run_id, args.n_panels)

    print(f"[run_matrix] FINISHED ok={len(cells) - n_failed}/{len(cells)} failed={n_failed}")


def _safe_eval(run_id: str, n_panels: int) -> None:
    """Run evaluation + prediction-panel render, never killing the matrix on failure.

    Skips the entire eval pass if the grid figure already exists for this
    ``run_id`` so re-runs of an already-evaluated matrix are cheap.
    """
    from .runtime.drive import results_root

    grid_path = results_root() / "figures" / "predictions" / run_id / "grid.png"
    if grid_path.is_file():
        return
    try:
        from .evaluate import evaluate_cell

        m = evaluate_cell(run_id, n_panels=n_panels, render_panels=True)
        print(
            f"[run_matrix] EVAL  run_id={run_id} val_mse={m['val_mse']:.6f} "
            f"contact_mse={m.get('contact_mse')} -> "
            f"figures/predictions/{run_id}/grid.png"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[run_matrix] EVAL FAILED for run_id={run_id}: {exc!r}")


if __name__ == "__main__":
    main()
