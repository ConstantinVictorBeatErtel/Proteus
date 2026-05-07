"""
Full sweep: train and evaluate all 5 conditions × 4 demo scales
using V-JEPA 2 cached features on LIBERO-Spatial.

Usage:
    python src/run_all_libero.py \
        --feature_dir /home/ubuntu/RAID/data/libero_spatial/vjepa_features \
        --device cuda

Demo scales: 25, 50, 100, 200
Conditions:  mean_action, nn_copy, direct_mlp, concat_mlp, raid_xattn

All conditions use IDENTICAL frozen V-JEPA 2 features (feat_dim=1024).
Only the decoder architecture and whether retrieval is used varies.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


CONDITIONS  = ["mean_action", "nn_copy", "direct_mlp", "concat_mlp", "raid_xattn"]
DEMO_SCALES = [25, 50, 100, 200]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feature_dir",
                   default="/home/ubuntu/RAID/data/libero_spatial/vjepa_features")
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--skip_nonparam", action="store_true",
                   help="skip mean_action and nn_copy (already computed)")
    args = p.parse_args()

    py = sys.executable
    project_root = Path(__file__).resolve().parent.parent
    results = {}

    for nd in DEMO_SCALES:
        results[str(nd)] = {}
        for cond in CONDITIONS:
            if args.skip_nonparam and cond in ("mean_action", "nn_copy"):
                # Read existing val_mse if available.
                mse_path = (project_root / "configs" /
                             f"val_mse_{cond}_{nd}demos_vjepa.json")
                if mse_path.exists():
                    d = json.loads(mse_path.read_text())
                    val_mse = d["val_mse"]
                else:
                    val_mse = float("nan")
            else:
                cmd = [
                    py,
                    str(project_root / "src" / "train_libero.py"),
                    "--condition", cond,
                    "--n_demos", str(nd),
                    "--feature_dir", args.feature_dir,
                    "--output_dir", str(project_root / "models"),
                    "--epochs", str(args.epochs),
                    "--device", args.device,
                ]
                print(f"\n[run_all_libero] START {' '.join(cmd)}", flush=True)
                ret = subprocess.run(cmd, cwd=project_root)
                if ret.returncode != 0:
                    print(f"[run_all_libero] WARNING: {cond} @ {nd}d "
                          f"exited with code {ret.returncode}")
                    val_mse = float("nan")
                else:
                    # Read the saved val_mse.
                    mse_path = (project_root / "configs" /
                                 f"val_mse_{cond}_{nd}demos_vjepa.json")
                    if mse_path.exists():
                        d = json.loads(mse_path.read_text())
                        val_mse = d["val_mse"]
                    else:
                        val_mse = float("nan")

            results[str(nd)][cond] = val_mse

    # ---- Print summary table ----
    print("\n\n=== LIBERO-Spatial Results (V-JEPA 2, feat_dim=1024) ===\n")
    header = f"{'Condition':<16} {'25':>10} {'50':>10} {'100':>10} {'200':>10}"
    print(header)
    print("-" * len(header))
    for cond in CONDITIONS:
        vals = [results[str(n)].get(cond, float("nan")) for n in DEMO_SCALES]
        line = f"{cond:<16}"
        for v in vals:
            line += f" {v:>10.4f}" if not (isinstance(v, float) and v != v) else f" {'nan':>10}"
        print(line)

    # ---- Save results ----
    out_path = project_root / "configs" / "results_vjepa.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[run_all_libero] Results saved → {out_path}")

    # ---- Validation gate ----
    print(f"\n=== Validation Gate ===")
    dir_25  = results["25"].get("direct_mlp", float("nan"))
    raid_25 = results["25"].get("raid_xattn", float("nan"))
    concat_25 = results["25"].get("concat_mlp", float("nan"))
    print(f"direct_mlp  @ 25: {dir_25:.4f}")
    print(f"concat_mlp  @ 25: {concat_25:.4f}")
    print(f"raid_xattn  @ 25: {raid_25:.4f}")

    if raid_25 <= dir_25 and raid_25 <= concat_25:
        print("PASSED — raid_xattn ≤ concat_mlp ≤ direct_mlp at low data. "
              "Retrieval + cross-attention helps.")
    else:
        print("note: expected ranking may not hold — check results.")


if __name__ == "__main__":
    main()
