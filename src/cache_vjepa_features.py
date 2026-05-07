"""
Cache V-JEPA 2 features for all LIBERO-Spatial demonstrations.

Run once before training:
    python src/cache_vjepa_features.py \
        --dataset_dir /home/ubuntu/RAID/data/libero_spatial/libero_spatial \
        --output_dir /home/ubuntu/RAID/data/libero_spatial/vjepa_features \
        --device cuda \
        --batch_size 32

With --dry_run: process only first task, first 2 demos (for pipeline verification).

Produces one .pt file per task:
    vjepa_features/{task_stem}.pt

Each file contains:
    {
      "feat_t":       (N, 1024) float32
      "feat_next":    (N, 1024) float32
      "actions":      (N, 7)    float32  raw (unnormalised)
      "episode_ids":  (N,)      int64    which demo each transition belongs to
    }
"""
from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import h5py
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))


def find_hdf5_files(dataset_dir: str | Path) -> list[str]:
    pattern = str(Path(dataset_dir) / "*.hdf5")
    files = sorted(glob.glob(pattern))
    if not files:
        pattern = str(Path(dataset_dir) / "*.h5")
        files = sorted(glob.glob(pattern))
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir",
                     default="/home/ubuntu/RAID/data/libero_spatial/libero_spatial")
    ap.add_argument("--output_dir",
                     default="/home/ubuntu/RAID/data/libero_spatial/vjepa_features")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hdf5_files = find_hdf5_files(args.dataset_dir)
    print(f"Found {len(hdf5_files)} task files in {args.dataset_dir}")

    if args.dry_run:
        hdf5_files = hdf5_files[:1]
        print("[dry_run] processing only first task file, first 2 demos")

    # Lazy-load encoder (avoids importing unless needed).
    from vjepa_encoder import VJEPAEncoder

    print(f"Loading V-JEPA 2 encoder on {args.device} …")
    enc = VJEPAEncoder(device=args.device)
    print(f"  feat_dim = {enc.feat_dim}")

    total_transitions = 0
    t_start = time.time()

    for hdf5_path in hdf5_files:
        task_stem = Path(hdf5_path).stem
        out_path = output_dir / f"{task_stem}.pt"

        if out_path.exists():
            print(f"[skip] {out_path} already cached")
            continue

        print(f"\n[processing] {task_stem}")

        all_feat_t = []
        all_feat_next = []
        all_actions = []
        all_episode_ids = []
        n_demos = 0
        n_trans = 0
        t_task = time.time()

        with h5py.File(hdf5_path, "r") as f:
            demo_keys = sorted(f["data"].keys(),
                                key=lambda x: int(x.split("_")[1]))
            if args.dry_run:
                demo_keys = demo_keys[:2]

            for dk in demo_keys:
                demo_idx = int(dk.split("_")[1])
                grp = f["data"][dk]
                images = grp["obs/agentview_rgb"][:]  # (T, 128, 128, 3) uint8
                raw_actions = grp["actions"][:]        # (T, 7) float32
                T = images.shape[0]

                if T < 2:
                    continue

                # Convert to float32/255, permute to (T, 3, 128, 128).
                frames = torch.from_numpy(images).float() / 255.0
                frames = frames.permute(0, 3, 1, 2)  # (T, 3, 128, 128)

                # Encode in batches.
                features = []
                for start in range(0, T, args.batch_size):
                    end = min(start + args.batch_size, T)
                    batch = frames[start:end]
                    feats = enc.encode_frames(batch)
                    features.append(feats.cpu())
                features = torch.cat(features, dim=0)  # (T, 1024)

                # Build transitions: feat_t[:-1], feat_next[1:], actions[:-1].
                feat_t = features[:-1]         # (T-1, 1024)
                feat_next = features[1:]       # (T-1, 1024)
                actions = torch.from_numpy(raw_actions[:-1]).float()  # (T-1, 7)
                ep_ids = torch.full((T - 1,), demo_idx, dtype=torch.long)

                all_feat_t.append(feat_t)
                all_feat_next.append(feat_next)
                all_actions.append(actions)
                all_episode_ids.append(ep_ids)

                n_demos += 1
                n_trans += (T - 1)

        # Concatenate all demos in this task.
        feat_t = torch.cat(all_feat_t, dim=0)
        feat_next = torch.cat(all_feat_next, dim=0)
        actions = torch.cat(all_actions, dim=0)
        episode_ids = torch.cat(all_episode_ids, dim=0)

        torch.save({
            "feat_t":      feat_t,
            "feat_next":   feat_next,
            "actions":     actions,
            "episode_ids": episode_ids,
        }, out_path)

        elapsed_task = time.time() - t_task
        total_transitions += n_trans
        print(f"  demos={n_demos}  transitions={n_trans}  "
              f"shape={tuple(feat_t.shape)}  "
              f"time={elapsed_task:.1f}s  → {out_path.name}")

    elapsed_total = time.time() - t_start
    print(f"\n=== Done ===")
    print(f"  total transitions across {len(hdf5_files)} tasks: {total_transitions}")
    print(f"  total time: {elapsed_total:.1f}s ({elapsed_total/60:.1f}min)")
    print(f"  output dir: {output_dir}")


if __name__ == "__main__":
    main()
