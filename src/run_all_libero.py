"""
Full sweep: train and evaluate all conditions × all demo scales on LIBERO-Spatial.

Usage:
    python src/run_all_libero.py \
        --feature_dir data/libero_spatial/features \
        --device cuda

Demo scales: 25, 50, 100, 200 (matching original RAID sweep).
Conditions:  direct_visual, raid_visual
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


CONDITIONS  = ["direct_visual", "raid_visual"]
DEMO_SCALES = [25, 50, 100, 200]


def run_training(condition: str, n_demos: int, feature_dir: str, device: str,
                 epochs: int = 100) -> float:
    cmd = [
        sys.executable, "src/train_libero.py",
        "--condition",   condition,
        "--n_demos",     str(n_demos),
        "--feature_dir", feature_dir,
        "--epochs",      str(epochs),
        "--device",      device,
    ]
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"[WARNING] Training exited with code {result.returncode}")

    # Read best val MSE from saved checkpoint
    ckpt_path = Path("models") / f"{condition}_{n_demos}demos_libero_best.pt"
    if ckpt_path.exists():
        import torch
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # Re-evaluate to get val MSE
        return evaluate_checkpoint(condition, n_demos, feature_dir, device)
    return float("nan")


def evaluate_checkpoint(condition: str, n_demos: int, feature_dir: str,
                         device: str) -> float:
    """Load best checkpoint and evaluate on validation set."""
    import torch
    from torch.utils.data import DataLoader
    import sys
    sys.path.insert(0, "src")
    from models import DirectMLPVisual, RAIDDecoderVisual
    from train_libero import _load_cached_datasets, populate_memory_from_cache, run_epoch

    feature_dir_p = Path(feature_dir)
    _, val_ds, feat_dim = _load_cached_datasets(feature_dir_p, n_demos)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=4)

    ckpt_path = Path("models") / f"{condition}_{n_demos}demos_libero_best.pt"
    if not ckpt_path.exists():
        return float("nan")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if condition == "direct_visual":
        model = DirectMLPVisual(feat_dim=feat_dim).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        memory_bank = None
    else:
        model = RAIDDecoderVisual(feat_dim=feat_dim).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        memory_bank = populate_memory_from_cache(feature_dir_p, n_demos, device,
                                                  feat_dim=feat_dim)

    val_mse = run_epoch(model, val_loader, None, memory_bank, condition, device)
    return val_mse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feature_dir", default="data/libero_spatial/features")
    p.add_argument("--device",      default="cuda")
    p.add_argument("--epochs",      type=int, default=100)
    args = p.parse_args()

    results = {}

    for condition in CONDITIONS:
        results[condition] = {}
        for n_demos in DEMO_SCALES:
            val_mse = run_training(condition, n_demos, args.feature_dir,
                                   args.device, args.epochs)
            results[condition][str(n_demos)] = {"val_mse": val_mse}
            print(f"\n  {condition} @ {n_demos} demos → val_mse={val_mse:.4f}")

    # Print summary table
    print("\n\n=== LIBERO-Spatial Offline Results ===")
    header = f"{'Condition':<22} {'25d':>8} {'50d':>8} {'100d':>8} {'200d':>8}"
    print(header)
    print("-" * len(header))
    for cond in CONDITIONS:
        vals = [results[cond].get(str(n), {}).get("val_mse", float("nan"))
                for n in DEMO_SCALES]
        print(f"{cond:<22} {vals[0]:>8.4f} {vals[1]:>8.4f} {vals[2]:>8.4f} {vals[3]:>8.4f}")

    # Save
    out_path = Path("configs") / "results_libero.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")

    # Validation gate
    direct_25  = results.get("direct_visual", {}).get("25", {}).get("val_mse", float("nan"))
    raid_25    = results.get("raid_visual",   {}).get("25", {}).get("val_mse", float("nan"))
    print(f"\n=== Stage 1 Validation Gate ===")
    print(f"direct_visual @ 25 demos: {direct_25:.4f}")
    print(f"raid_visual   @ 25 demos: {raid_25:.4f}")
    if raid_25 <= direct_25:
        print("✓ PASSED — raid_visual ≤ direct_visual. Proceed to Stage 2 (GRPO).")
    else:
        print("✗ FAILED — raid_visual > direct_visual. Do NOT proceed to GRPO yet.")


if __name__ == "__main__":
    main()
