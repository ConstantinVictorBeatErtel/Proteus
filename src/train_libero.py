"""
Training loop for all RAID conditions on cached V-JEPA 2 features.

Usage:
    python src/train_libero.py \
        --condition raid_xattn \
        --n_demos 50 \
        --feature_dir /home/ubuntu/RAID/data/libero_spatial/vjepa_features \
        --output_dir /home/ubuntu/RAID/models \
        --epochs 100 \
        --batch_size 256 \
        --lr 1e-3 \
        --weight_decay 1e-4 \
        --device cuda \
        --k 5

Conditions:
    mean_action : predict train mean action (no model)
    nn_copy     : retrieve top-1 action from memory bank (no model)
    direct_mlp  : train DirectMLPVisual (no retrieval)
    concat_mlp  : train ConcatMLPVisual (retrieval + concatenation)
    raid_xattn  : train RAIDDecoderVisual (retrieval + cross-attention)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_libero import make_train_val_vjepa
from memory_libero import VJEPAMemoryBank
from models_libero import (FEAT_DIM, DirectMLPVisual, ConcatMLPVisual,
                            RAIDDecoderVisual)

SEED = 42


def seed_all() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition",
                    choices=["mean_action", "nn_copy", "direct_mlp",
                             "concat_mlp", "raid_xattn"],
                    required=True)
    ap.add_argument("--n_demos", type=int,
                    choices=[25, 50, 100, 200], required=True)
    ap.add_argument("--feature_dir",
                    default="/home/ubuntu/RAID/data/libero_spatial/vjepa_features")
    ap.add_argument("--output_dir",
                    default="/home/ubuntu/RAID/models")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--k", type=int, default=5,
                    help="number of retrieved neighbours")
    args = ap.parse_args()

    seed_all()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cond = args.condition
    nd  = args.n_demos
    k   = args.k

    print(f"[train_libero] device={dev}  condition={cond}  "
          f"n_demos={nd}  k={k}  epochs={args.epochs}")

    # ---- Load data ----
    train_ds, val_ds = make_train_val_vjepa(args.feature_dir,
                                             n_demos=nd, seed=SEED)
    print(f"[train_libero] train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, drop_last=False, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, drop_last=False, num_workers=0)

    # ---- Save normalisation stats ----
    norm_stats = {
        "action_mean": train_ds.action_mean,
        "action_std":  train_ds.action_std,
    }
    stats_path = (PROJECT_ROOT / "configs" /
                   f"norm_stats_{nd}demos_vjepa.pt")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(norm_stats, stats_path)
    print(f"[train_libero] saved norm stats → {stats_path}")

    # ---- Build memory bank (for retrieval conditions) ----
    memory_bank: VJEPAMemoryBank | None = None
    use_retrieval = cond in ("nn_copy", "concat_mlp", "raid_xattn")
    if use_retrieval:
        memory_bank = VJEPAMemoryBank(feat_dim=FEAT_DIM, max_size=200000,
                                       device=dev)
        memory_bank.build_from_dataset(train_ds)

    # ---- Non-parametric conditions (no model to train) ----
    if cond == "mean_action":
        train_actions = torch.stack([train_ds[i][2] for i in range(len(train_ds))])
        mean_act = train_actions.mean(dim=0)
        val_actions = torch.stack([val_ds[i][2] for i in range(len(val_ds))])
        val_mse = float(nn.functional.mse_loss(
            mean_act.unsqueeze(0).expand(len(val_ds), -1),
            val_actions))
        print(f"[train_libero] mean_action val_mse={val_mse:.6f}")
        save_val_mse(cond, nd, val_mse)
        return

    if cond == "nn_copy":
        val_mse = eval_nn_copy(val_ds, memory_bank, k, dev)
        print(f"[train_libero] nn_copy val_mse={val_mse:.6f}")
        save_val_mse(cond, nd, val_mse)
        return

    # ---- Parametric conditions ----
    if cond == "direct_mlp":
        model = DirectMLPVisual(feat_dim=FEAT_DIM, action_dim=7,
                                hidden_dim=512, dropout=0.1).to(dev)
    elif cond == "concat_mlp":
        model = ConcatMLPVisual(feat_dim=FEAT_DIM, action_dim=7,
                                hidden_dim=512, dropout=0.1).to(dev)
    elif cond == "raid_xattn":
        model = RAIDDecoderVisual(feat_dim=FEAT_DIM, action_dim=7,
                                  hidden_dim=512, num_heads=8,
                                  dropout=0.1).to(dev)
    else:
        raise ValueError(f"Unknown condition: {cond}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    best_val = float("inf")
    best_epoch = -1
    train_curve: list[float] = []
    val_curve: list[float]   = []

    for epoch in range(1, args.epochs + 1):
        tr_mse = run_epoch(model, train_loader, optimizer, memory_bank,
                           cond, dev, k, train=True)
        va_mse = run_epoch(model, val_loader, None, memory_bank,
                           cond, dev, k, train=False)
        scheduler.step()

        train_curve.append(tr_mse)
        val_curve.append(va_mse)

        if va_mse < best_val:
            best_val = va_mse
            best_epoch = epoch
            ckpt_path = (Path(args.output_dir) /
                          f"{cond}_{nd}demos_vjepa_best.pt")
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_mse": va_mse,
                "condition": cond,
                "n_demos": nd,
            }, ckpt_path)

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"[train_libero] epoch {epoch:03d}/{args.epochs}  "
                  f"train={tr_mse:.6f}  val={va_mse:.6f}  "
                  f"best={best_val:.6f}@{best_epoch}")

    # Loss curves
    curves = {"train_mse": train_curve, "val_mse": val_curve,
              "condition": cond, "n_demos": nd}
    curves_path = (PROJECT_ROOT / "configs" /
                    f"loss_curves_{cond}_{nd}demos_vjepa.json")
    curves_path.parent.mkdir(parents=True, exist_ok=True)
    curves_path.write_text(json.dumps(curves, indent=2))
    print(f"[train_libero] loss curves → {curves_path}")
    print(f"[train_libero] best val_mse={best_val:.6f} at epoch {best_epoch}")

    # Save val MSE
    save_val_mse(cond, nd, best_val)


# -------------------------------------------------------------------
# Training / evaluation epochs
# -------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    memory_bank: VJEPAMemoryBank | None,
    condition: str,
    device: torch.device,
    k: int,
    train: bool,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)

    total_err = 0.0
    total_n   = 0

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            ft, fn, act = batch
            ft   = ft.to(device)
            fn   = fn.to(device)
            act  = act.to(device)
            B = ft.shape[0]

            if condition == "direct_mlp":
                pred = model(ft, fn)
            else:
                # Retrieval conditions
                ret_a, ret_s, ret_v = memory_bank.retrieve_batch(ft, fn, k=k)
                if condition == "concat_mlp":
                    pred = model(ft, fn, ret_a)
                elif condition == "raid_xattn":
                    pred = model(ft, fn, ret_a)
                else:
                    raise ValueError(condition)

            loss = nn.functional.mse_loss(pred, act)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_err += float(loss.item()) * B
            total_n   += B

    return total_err / max(1, total_n)


# -------------------------------------------------------------------
# Non-parametric evaluation
# -------------------------------------------------------------------

def eval_nn_copy(
    val_ds,
    memory_bank: VJEPAMemoryBank,
    k: int,
    device: torch.device,
) -> float:
    """nn_copy: for each validation sample, retrieve top-1 neighbour's action."""
    total_err = 0.0
    n = 0
    for i in range(len(val_ds)):
        ft, fn, act = val_ds[i]
        ft  = ft.unsqueeze(0).to(device)
        fn  = fn.unsqueeze(0).to(device)
        act = act.unsqueeze(0).to(device)
        ret_a, _, _ = memory_bank.retrieve_batch(ft, fn, k=k)
        pred = ret_a[:, 0, :]  # (1, 7) — top-1 action
        total_err += float(nn.functional.mse_loss(pred, act).item())
        n += 1
    return total_err / max(n, 1)


def save_val_mse(condition: str, n_demos: int, val_mse: float) -> None:
    mse_path = (PROJECT_ROOT / "configs" /
                 f"val_mse_{condition}_{n_demos}demos_vjepa.json")
    mse_path.parent.mkdir(parents=True, exist_ok=True)
    mse_path.write_text(json.dumps({
        "condition": condition,
        "n_demos": n_demos,
        "val_mse": val_mse,
    }, indent=2))


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------

if __name__ == "__main__":
    main()
