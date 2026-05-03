#!/usr/bin/env python3
"""
Generate publication-style figures from training logs + configs/results.json.

Writes to notebooks/figures/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"
FIG_DIR = REPO_ROOT / "notebooks" / "figures"

CONDITIONS = ["vision_only", "tactile_only", "visuo_tactile"]


def dof_order(results: dict) -> list[str]:
    return list(results["conditions"][CONDITIONS[0]]["mse_per_dof"].keys())


def plot_fluid_contact_heatmap() -> None:
    """10 consecutive raw tactile grids around a detected contact onset."""
    import matplotlib.pyplot as plt
    import numpy as np

    SRC = REPO_ROOT / "src"
    sys.path.insert(0, str(SRC))

    import imagecodecs.numcodecs  # noqa: F401

    imagecodecs.numcodecs.register_codecs()
    import torch
    import zarr

    task = "fluid_transfer"
    zpath = (
        REPO_ROOT
        / "data"
        / "touch_in_the_wild"
        / "four_tasks"
        / task
        / f"{task}.zarr.zip"
    )
    contact_pt = CONFIG_DIR / "contact_threshold.pt"
    thresh = 0.05
    if contact_pt.is_file():
        blob = torch.load(contact_pt, map_location="cpu", weights_only=False)
        thresh = float(blob.get("threshold", thresh))

    root = zarr.open(str(zpath), mode="r")
    tactile = root["data"]["camera0_tactile"]
    n = tactile.shape[0]

    onset = None
    for i in range(max(0, n - 20)):
        g = np.asarray(tactile[i], dtype=np.float32)
        g2 = np.asarray(tactile[i + 1], dtype=np.float32)
        mx1, mx2 = float(g.max()), float(g2.max())
        if mx1 < thresh < mx2:
            onset = i
            break

    if onset is None:
        for i in range(n - 20):
            mx = float(np.asarray(tactile[i], dtype=np.float32).max())
            if mx > thresh:
                onset = max(0, i - 2)
                break

    if onset is None:
        onset = max(0, n // 2)

    block = []
    for k in range(10):
        idx = int(min(onset + k, n - 1))
        block.append(np.asarray(tactile[idx], dtype=np.float32))

    block_arr = np.stack(block, axis=0)
    vmin = float(block_arr.min())
    vmax = float(block_arr.max())
    if vmax <= vmin:
        vmax = vmin + 1e-6

    fig, axs = plt.subplots(2, 5, figsize=(13, 4.5))
    for k in range(10):
        r = k // 5
        c = k % 5
        ax = axs[r, c]
        im = ax.imshow(block[k], aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"t+{k}")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        "fluid_transfer — 10 tactile pressure grids during rising contact onset"
    )
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    cbar = fig.colorbar(im, ax=axs, orientation="horizontal", fraction=0.06, pad=0.12)
    cbar.set_label("Raw tactile reading")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "tactile_heatmap.png"
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"[02_results] saved {out}")


def main() -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    results_path = CONFIG_DIR / "results.json"
    if not results_path.is_file():
        raise FileNotFoundError(f"Missing {results_path}; run python src/evaluate.py first.")

    with open(results_path, "r", encoding="utf-8") as fp:
        results = json.load(fp)

    # --- Loss curves ---
    plt.figure(figsize=(8, 5))
    for cond in CONDITIONS:
        lp = CONFIG_DIR / f"{cond}_losses.json"
        if not lp.is_file():
            print(f"[02_results] missing {lp} (skipped in loss plot)")
            continue
        hist = json.loads(lp.read_text(encoding="utf-8"))
        ep = np.arange(1, len(hist["train"]) + 1)
        plt.plot(ep, hist["train"], linestyle="--", linewidth=1.6, label=f"{cond} train")
        plt.plot(ep, hist["val"], linewidth=1.6, label=f"{cond} val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (normalized actions)")
    plt.title("Training and validation curves (all conditions)")
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    lc_path = FIG_DIR / "loss_curves.png"
    plt.savefig(lc_path, dpi=160)
    plt.close()
    print(f"[02_results] saved {lc_path}")

    # --- Overall MSE ---
    vals = [results["conditions"][c]["mse_overall"] for c in CONDITIONS]
    plt.figure(figsize=(6, 4))
    x = np.arange(len(CONDITIONS))
    plt.bar(x, vals, color=["#4C72B0", "#55A868", "#C44E52"])
    plt.xticks(x, CONDITIONS, rotation=18, ha="right")
    plt.ylabel("Overall val MSE")
    plt.title("Validation action MSE — all conditions")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    pmc = FIG_DIR / "mse_comparison.png"
    plt.savefig(pmc, dpi=160)
    plt.close()
    print(f"[02_results] saved {pmc}")

    # --- Contact grouping ---
    w = 0.32
    plt.figure(figsize=(7.5, 4.3))
    x = np.arange(len(CONDITIONS))
    contacts = [results["conditions"][c]["mse_contact"] for c in CONDITIONS]
    frees = [results["conditions"][c]["mse_noncontact"] for c in CONDITIONS]
    plt.bar(x - w / 2, contacts, width=w, label="contact", color="#CC6136")
    plt.bar(x + w / 2, frees, width=w, label="non-contact", color="#4B8BBE")
    plt.xticks(x, CONDITIONS, rotation=14, ha="right")
    plt.ylabel("Validation MSE (normalized)")
    plt.title("Contact timestep vs non-contact (calibrated tactile threshold)")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    cvc = FIG_DIR / "contact_vs_noncontact.png"
    plt.savefig(cvc, dpi=160)
    plt.close()
    print(f"[02_results] saved {cvc}")

    # --- Per-DOF ---
    dk = dof_order(results)
    d_vo = np.array([results["conditions"]["vision_only"]["mse_per_dof"][k] for k in dk])
    d_vt = np.array(
        [results["conditions"]["visuo_tactile"]["mse_per_dof"][k] for k in dk]
    )
    w2 = 0.38
    plt.figure(figsize=(9.5, 4.3))
    x = np.arange(len(dk))
    plt.bar(x - w2 / 2, d_vo, width=w2, label="vision_only")
    plt.bar(x + w2 / 2, d_vt, width=w2, label="visuo_tactile")
    plt.xticks(x, dk, rotation=20, ha="right")
    plt.ylabel("Validation MSE per DOF")
    plt.title("Per-DOF breakdown: baseline vs multimodal fusion")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    dof_path = FIG_DIR / "per_dof_mse.png"
    plt.savefig(dof_path, dpi=160)
    plt.close()
    print(f"[02_results] saved {dof_path}")

    plot_fluid_contact_heatmap()


if __name__ == "__main__":
    main()
