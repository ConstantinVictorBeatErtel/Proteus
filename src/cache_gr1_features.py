"""
Pre-compute and cache GR-1 features for the LIBERO dataset.

Run once before training:
    python src/cache_gr1_features.py \
        --dataset_dir data/libero_spatial \
        --output_dir  data/libero_spatial/features \
        --gr1_ckpt    checkpoints/gr1/snapshot_ABCD.pt \
        --mae_ckpt    checkpoints/gr1/mae_pretrain_vit_base.pth \
        --device      cuda

Produces one .pt file per task:
    data/libero_spatial/features/<task_name>_features.pt

Each file contains:
    {
      "feat_t":       (N, 384) float32
      "feat_next":    (N, 384) float32
      "actions":      (N, 7)   float32  normalised
      "demo_lengths": list[int]
      "task_name":    str
    }
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent))

from data_libero import (
    find_hdf5_files,
    compute_norm_stats,
    save_norm_stats,
    LiberoTransitionDataset,
    _find_img_key,
)
from gr1_encoder import GR1Encoder


def cache_task(
    hdf5_path: str,
    encoder: GR1Encoder,
    norm_stats: dict,
    output_dir: Path,
    batch_size: int = 64,
) -> Path:
    task_name = Path(hdf5_path).stem
    out_path  = output_dir / f"{task_name}_features.pt"

    if out_path.exists():
        print(f"  [skip] {task_name} already cached at {out_path}")
        return out_path

    print(f"  Caching {task_name} ...")
    ds = LiberoTransitionDataset([hdf5_path], norm_stats)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4,
                        pin_memory=True)

    all_feat_t    = []
    all_feat_next = []
    all_actions   = []

    t0 = time.time()
    for batch in loader:
        img_t    = batch["image_t"].to(encoder.device)    # (B, H, W, 3) uint8
        img_next = batch["image_next"].to(encoder.device)
        actions  = batch["action"]

        feat_t    = encoder.encode_frames(img_t)
        feat_next = encoder.encode_frames(img_next)

        all_feat_t.append(feat_t.cpu())
        all_feat_next.append(feat_next.cpu())
        all_actions.append(actions.cpu())

    feat_t    = torch.cat(all_feat_t,    dim=0)
    feat_next = torch.cat(all_feat_next, dim=0)
    actions   = torch.cat(all_actions,   dim=0)

    # Compute demo lengths for n_demos slicing
    demo_lengths = []
    with h5py.File(hdf5_path, "r") as f:
        demo_keys = sorted(f["data"].keys(), key=lambda x: int(x.replace("demo_", "")))
        for dk in demo_keys:
            T = f[f"data/{dk}/actions"].shape[0]
            demo_lengths.append(T - 1)  # transitions

    torch.save({
        "feat_t":       feat_t,
        "feat_next":    feat_next,
        "actions":      actions,
        "demo_lengths": demo_lengths,
        "task_name":    task_name,
    }, out_path)

    elapsed = time.time() - t0
    print(f"    {len(ds)} transitions → {out_path} ({elapsed:.1f}s)")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="data/libero_spatial")
    parser.add_argument("--output_dir",  default="data/libero_spatial/features")
    parser.add_argument("--gr1_ckpt",    default="checkpoints/gr1/snapshot_ABCD.pt")
    parser.add_argument("--mae_ckpt",    default="checkpoints/gr1/mae_pretrain_vit_base.pth")
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--batch_size",  type=int, default=128)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Find HDF5 files ----
    hdf5_files = find_hdf5_files(args.dataset_dir)
    # exclude already-existing or non-task files
    hdf5_files = [f for f in hdf5_files if "features" not in f]
    print(f"Found {len(hdf5_files)} LIBERO task files")
    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 files found in {args.dataset_dir}")

    # ---- Norm stats ----
    norm_path = Path(args.dataset_dir) / "norm_stats.pt"
    if norm_path.exists():
        print(f"Loading norm stats from {norm_path}")
        from data_libero import load_norm_stats
        norm_stats = load_norm_stats(norm_path)
    else:
        print("Computing norm stats ...")
        norm_stats = compute_norm_stats(hdf5_files)
        save_norm_stats(norm_stats, norm_path)
        print(f"Saved norm stats to {norm_path}")

    # ---- Load encoder ----
    print(f"Loading GR-1 encoder from {args.gr1_ckpt} ...")
    encoder = GR1Encoder.from_checkpoints(args.mae_ckpt, args.gr1_ckpt, args.device)
    print(f"  feat_dim = {encoder.feat_dim}")

    # ---- Cache each task ----
    cached = []
    for path in hdf5_files:
        out = cache_task(path, encoder, norm_stats, output_dir, batch_size=args.batch_size)
        cached.append(str(out))

    # ---- Write manifest ----
    manifest = {"cached_files": cached, "feat_dim": encoder.feat_dim}
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nDone. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
