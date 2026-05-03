"""
02_results.py — Plots from configs/results.json and training loss curves.

Run from repo root:
    python3 notebooks/02_results.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS = PROJECT_ROOT / "configs"
FIGURES_DIR = PROJECT_ROOT / "notebooks" / "figures"

CONDITIONS_DISPLAY = ["mean_baseline", "nearest_neighbor", "direct_mlp", "raid"]
LABELS = {
    "mean_baseline": "mean (zeros)",
    "nearest_neighbor": "kNN pooled",
    "direct_mlp": "direct MLP",
    "raid": "RAID",
}
COLORS = {
    "mean_baseline": "#888888",
    "nearest_neighbor": "#2ca02c",
    "direct_mlp": "#ff7f0e",
    "raid": "#1f77b4",
}


def main() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    results_path = CONFIGS / "results.json"
    if not results_path.is_file():
        raise FileNotFoundError(f"Missing {results_path}; run evaluate.py first.")
    payload = json.loads(results_path.read_text())

    scales = sorted(int(k) for cond in payload.values() for k in cond.keys())

    # --- Fig 1: overall MSE vs number of demos
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    for c in CONDITIONS_DISPLAY:
        ys = [float(payload[c][str(n)]["mse"]) for n in scales]
        ax1.plot(scales, ys, marker="o", linewidth=2, label=LABELS[c], color=COLORS[c])
    ax1.set_xlabel("# train demos")
    ax1.set_ylabel("validation MSE (normalized actions)")
    ax1.set_title("Overall inverse-dynamics error vs data scale")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(scales)
    fig1.tight_layout()
    p1 = FIGURES_DIR / "mse_scaling.png"
    fig1.savefig(p1, dpi=140)
    plt.close(fig1)
    print(f"Saved {p1}")

    # --- Fig 2: contact-phase MSE
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for c in CONDITIONS_DISPLAY:
        ys = []
        for n in scales:
            v = payload[c][str(n)]["contact_mse"]
            ys.append(float("nan") if v is None else float(v))
        ax2.plot(scales, ys, marker="s", linewidth=2, label=LABELS[c], color=COLORS[c])
    ax2.set_xlabel("# train demos")
    ax2.set_ylabel("MSE during contact-rich steps")
    ax2.set_title("Contact-heavy subset error vs data scale")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(scales)
    fig2.tight_layout()
    p2 = FIGURES_DIR / "contact_mse_scaling.png"
    fig2.savefig(p2, dpi=140)
    plt.close(fig2)
    print(f"Saved {p2}")

    # --- Fig 3: validation loss curves (one panel per condition)
    fig3, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    cmap = plt.cm.viridis
    norms = matplotlib.colors.Normalize(vmin=min(scales), vmax=max(scales))
    for ci, cond in enumerate(["direct_mlp", "raid"]):
        ax = axes[ci]
        for n in scales:
            path = CONFIGS / f"loss_curves_{cond}_{n}demos.json"
            if not path.is_file():
                continue
            curves = json.loads(path.read_text())
            val = curves["val_mse"]
            ep = range(1, len(val) + 1)
            ax.plot(
                ep,
                val,
                label=f"n={n}",
                color=cmap(norms(n)),
                linewidth=1.8,
            )
        ax.set_title(f"{cond} — validation MSE")
        ax.set_xlabel("epoch")
        if ci == 0:
            ax.set_ylabel("val MSE")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
    fig3.suptitle("Training curves across data scales", y=1.02)
    fig3.tight_layout()
    p3 = FIGURES_DIR / "val_loss_by_scale.png"
    fig3.savefig(p3, dpi=140, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved {p3}")

    # --- Fig 4: retrieval hit rate (shared for NN & RAID)
    hr = []
    for n in scales:
        v = payload["nearest_neighbor"][str(n)]["hit_rate"]
        hr.append(float("nan") if v is None else float(v))

    fig4, ax4 = plt.subplots(figsize=(6.5, 4))
    ax4.bar([str(s) for s in scales], hr, color="#9467bd", edgecolor="white")
    ax4.set_xlabel("# train demos")
    ax4.set_ylabel("fraction of batches with ≥1 retrieval hit")
    ax4.set_title("Memory bank retrieval hit rate")
    ax4.set_ylim(0.0, 1.05)
    ax4.grid(True, axis="y", alpha=0.3)
    fig4.tight_layout()
    p4 = FIGURES_DIR / "retrieval_hit_rate.png"
    fig4.savefig(p4, dpi=140)
    plt.close(fig4)
    print(f"Saved {p4}")

    print("Done.")


if __name__ == "__main__":
    main()
