"""
Training loop for visual RAID on LIBERO with GR-1 features.

Usage:
    python src/train_libero.py \
        --condition raid_visual \
        --n_demos   50 \
        --feature_dir data/libero_spatial/features

Conditions: direct_visual | raid_visual
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

import sys
sys.path.insert(0, str(Path(__file__).parent))

from models import DirectMLPVisual, RAIDDecoderVisual
from memory import RAIDMemoryBank


# ---------------------------------------------------------------------------
# Helper: build dataset from cached features
# ---------------------------------------------------------------------------

def _load_cached_datasets(feature_dir: Path, n_demos: int, val_frac: float = 0.2):
    """
    Load all per-task cached feature files and split 80/20 train/val by demo.
    Returns (train_dataset, val_dataset, feat_dim).
    """
    from data_libero import CachedFeatureDataset

    manifest_path = feature_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        cached_files = [Path(f) for f in manifest["cached_files"]]
        feat_dim = manifest["feat_dim"]
    else:
        cached_files = sorted(feature_dir.glob("*_features.pt"))
        feat_dim = None

    if not cached_files:
        raise FileNotFoundError(f"No cached feature files in {feature_dir}")

    # Distribute n_demos across tasks (equal split)
    n_tasks = len(cached_files)
    demos_per_task = max(1, n_demos // n_tasks)
    n_val_per_task = max(1, int(demos_per_task * val_frac))
    n_train_per_task = demos_per_task - n_val_per_task

    train_datasets = []
    val_datasets   = []

    for cf in cached_files:
        data = torch.load(cf, map_location="cpu", weights_only=False)
        if feat_dim is None:
            feat_dim = data["feat_t"].shape[1]

        demo_lengths = data.get("demo_lengths", None)

        def _slice(feat_t, feat_next, actions, start_demo, end_demo):
            if demo_lengths is not None:
                start_idx = sum(demo_lengths[:start_demo])
                end_idx   = sum(demo_lengths[:end_demo])
            else:
                # approximate: equal split
                N = len(actions)
                step = N // demos_per_task if demos_per_task else N
                start_idx = start_demo * step
                end_idx   = min(end_demo * step, N)
            ds = CachedFeatureDataset.__new__(CachedFeatureDataset)
            ds.feat_t    = feat_t[start_idx:end_idx].float()
            ds.feat_next = feat_next[start_idx:end_idx].float()
            ds.actions   = actions[start_idx:end_idx].float()
            ds.feat_dim  = feat_t.shape[1]
            return ds

        ft  = data["feat_t"]
        fn  = data["feat_next"]
        act = data["actions"]

        total_demos_in_task = len(demo_lengths) if demo_lengths else demos_per_task
        end_demo  = min(demos_per_task, total_demos_in_task)
        val_start = end_demo - n_val_per_task

        train_datasets.append(_slice(ft, fn, act, 0, val_start))
        val_datasets.append(_slice(ft, fn, act, val_start, end_demo))

    from torch.utils.data import ConcatDataset

    class _SimpleConcat(torch.utils.data.Dataset):
        def __init__(self, datasets):
            self._ds = datasets
            self._lengths = [len(d) for d in datasets]
            self._total = sum(self._lengths)
            self._offsets = []
            off = 0
            for l in self._lengths:
                self._offsets.append(off)
                off += l

        def __len__(self):
            return self._total

        def __getitem__(self, idx):
            for i, (off, l) in enumerate(zip(self._offsets, self._lengths)):
                if idx < off + l:
                    return self._ds[i][idx - off]
            raise IndexError(idx)

    return _SimpleConcat(train_datasets), _SimpleConcat(val_datasets), feat_dim


# ---------------------------------------------------------------------------
# Memory bank population from cached features
# ---------------------------------------------------------------------------

def populate_memory_from_cache(feature_dir: Path, n_demos: int, device: str,
                                feat_dim: int | None = None) -> RAIDMemoryBank:
    """Fill memory bank with (feat_t, feat_next, action) from training demos."""
    cached_files = sorted(feature_dir.glob("*_features.pt"))
    n_tasks = len(cached_files)
    demos_per_task = max(1, n_demos // n_tasks)

    # Infer feat_dim from first cache file if not provided
    if feat_dim is None:
        first = torch.load(cached_files[0], map_location="cpu", weights_only=False)
        feat_dim = first["feat_t"].shape[1]

    bank = RAIDMemoryBank(obs_dim=feat_dim, action_dim=7, max_entries=300_000, device=device)

    for cf in cached_files:
        data = torch.load(cf, map_location="cpu", weights_only=False)
        demo_lengths = data.get("demo_lengths", None)

        if demo_lengths is not None:
            end_idx = int(sum(demo_lengths[:demos_per_task]))
        else:
            N = len(data["actions"])
            end_idx = min(N, demos_per_task * (N // max(n_tasks, 1)))

        feat_t    = data["feat_t"][:end_idx].float()
        feat_next = data["feat_next"][:end_idx].float()
        actions   = data["actions"][:end_idx].float()

        for i in range(len(feat_t)):
            bank.add(feat_t[i], feat_next[i], actions[i])

    return bank


# ---------------------------------------------------------------------------
# Training / validation epoch
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    memory_bank: RAIDMemoryBank | None,
    condition: str,
    device: str,
    k: int = 3,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    n = 0

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            feat_t, feat_next, actions = batch
            feat_t   = feat_t.to(device)
            feat_next = feat_next.to(device)
            actions  = actions.to(device)
            B = feat_t.shape[0]

            if condition == "direct_visual":
                pred = model(feat_t, feat_next)
                loss = nn.functional.mse_loss(pred, actions)

            elif condition == "raid_visual":
                # Batched retrieval: (B, k, 7) + (B, k) validity mask
                retrieved, valid_mask = memory_bank.retrieve_batch(
                    feat_t, feat_next, k=k
                )
                kv_pad = ~valid_mask  # True = ignore in attention
                pred, _ = model(feat_t, feat_next, retrieved,
                                kv_key_padding_mask=kv_pad)
                loss = nn.functional.mse_loss(pred, actions)
            else:
                raise ValueError(condition)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item() * B
            n += B

    return total_loss / max(n, 1)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    device = args.device
    feature_dir = Path(args.feature_dir)

    print(f"\n=== Training {args.condition} | n_demos={args.n_demos} ===")

    # ---- Load data ----
    train_ds, val_ds, feat_dim = _load_cached_datasets(feature_dir, args.n_demos)
    print(f"  train={len(train_ds)} val={len(val_ds)} feat_dim={feat_dim}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    # ---- Model ----
    action_dim = 7
    if args.condition == "direct_visual":
        model = DirectMLPVisual(feat_dim=feat_dim, action_dim=action_dim).to(device)
        memory_bank = None
    elif args.condition == "raid_visual":
        model = RAIDDecoderVisual(feat_dim=feat_dim, action_dim=action_dim).to(device)
        memory_bank = populate_memory_from_cache(feature_dir, args.n_demos, device,
                                                  feat_dim=feat_dim)
        print(f"  Memory bank: {memory_bank.ptr} entries")
    else:
        raise ValueError(f"Unknown condition: {args.condition}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_mse = float("inf")
    best_epoch   = 0
    ckpt_dir     = Path("models")
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path    = ckpt_dir / f"{args.condition}_{args.n_demos}demos_libero_best.pt"

    loss_curve = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_mse = run_epoch(model, train_loader, optimizer, memory_bank,
                              args.condition, device)
        val_mse   = run_epoch(model, val_loader,   None,      memory_bank,
                              args.condition, device)
        scheduler.step()

        loss_curve.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse})

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch   = epoch
            torch.save({"model_state_dict": model.state_dict(),
                        "feat_dim": feat_dim,
                        "condition": args.condition,
                        "n_demos": args.n_demos}, ckpt_path)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}/{args.epochs} | "
                  f"train={train_mse:.4f} val={val_mse:.4f} "
                  f"best={best_val_mse:.4f}@{best_epoch} "
                  f"({time.time()-t0:.1f}s)")

    # Save loss curve
    curve_path = Path("configs") / f"loss_curves_{args.condition}_{args.n_demos}demos_libero.json"
    curve_path.parent.mkdir(exist_ok=True)
    curve_path.write_text(json.dumps(loss_curve, indent=2))

    print(f"\nBest val MSE: {best_val_mse:.4f} @ epoch {best_epoch}")
    print(f"Checkpoint: {ckpt_path}")
    return best_val_mse


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--condition",    default="raid_visual",
                   choices=["direct_visual", "raid_visual"])
    p.add_argument("--n_demos",      type=int, default=50)
    p.add_argument("--feature_dir",  default="data/libero_spatial/features")
    p.add_argument("--epochs",       type=int, default=100)
    p.add_argument("--batch_size",   type=int, default=256)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--device",       default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
