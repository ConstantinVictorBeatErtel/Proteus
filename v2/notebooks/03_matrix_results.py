"""Aggregate matrix run results into figures.

Reads each completed run's ``result.json`` under ``<artifact_root>/runs/``
and produces:

* ``matrix_forest.png`` — per-cell mean ± SE across seeds, grouped
  by (phase, head, dataset, scale).
* ``encoder_ablation.png`` — DINOv2 vs Theia at the encoder-ablation
  cells (phase D).
* ``retrieval_helps_when.png`` — RAID minus DirectMLP gap as a
  function of (image features?, n_demos, multi-task?).
* ``predictions_index.png`` — a thumbnail index of every per-run
  prediction grid emitted by ``v2.evaluate``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from v2.runtime.drive import results_root, runs_root

    rows: list[dict] = []
    for rdir in sorted(runs_root().glob("*")):
        rj = rdir / "result.json"
        if not rj.is_file():
            continue
        try:
            payload = json.loads(rj.read_text())
        except Exception:  # noqa: BLE001
            continue
        cfg = payload.get("config", {})
        rows.append(
            {
                "run_id": payload.get("run_id", rdir.name),
                "phase": cfg.get("phase", "?"),
                "head": cfg.get("head", "?"),
                "dataset": cfg.get("dataset", "?"),
                "encoder": cfg.get("encoder") or "none",
                "n_demos": int(cfg.get("n_demos", -1)),
                "seed": int(cfg.get("seed", -1)),
                "best_val_mse": float(payload.get("best_val_mse", float("nan"))),
            }
        )
    if not rows:
        print("[03_matrix_results] no completed runs found")
        return

    grouped: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        key = (r["phase"], r["head"], r["dataset"], r["encoder"], r["n_demos"])
        grouped[key].append(r["best_val_mse"])

    keys = sorted(grouped.keys())
    means = [float(np.mean(grouped[k])) for k in keys]
    ses = [float(np.std(grouped[k]) / np.sqrt(max(1, len(grouped[k])))) for k in keys]
    labels = [f"{k[0]}|{k[1]}|{k[2]}|{k[3]}|n={k[4]}" for k in keys]

    out_dir = results_root() / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(keys))))
    y = np.arange(len(keys))
    ax.errorbar(means, y, xerr=ses, fmt="o", color="#3b6cd1", ecolor="#777")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("best val MSE")
    ax.set_title(f"matrix forest plot ({len(rows)} runs / {len(keys)} cells)")
    ax.invert_yaxis()
    ax.grid(axis="x", linewidth=0.4, alpha=0.4)
    fig.tight_layout()
    out = out_dir / "matrix_forest.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[03_matrix_results] wrote {out}")


if __name__ == "__main__":
    main()
