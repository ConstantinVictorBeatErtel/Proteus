"""Train RAID or direct inverse-dynamics MLP."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data import make_train_val, norm_stats_path
from memory import RAIDMemoryBank
from models import DirectMLP, RAIDDecoder, RAIDDecoderCrossAttn

SEED = 42


def seed_all() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def pooled_retrieved(actions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
    summed = (actions * mask.unsqueeze(-1).float()).sum(dim=1)
    return summed / denom.float()


def run_epoch_direct(
    model: DirectMLP,
    loader: DataLoader,
    dev: torch.device,
    optim: torch.optim.Optimizer | None,
    train: bool,
) -> float:
    if train:
        model.train()
    else:
        model.eval()

    sse = 0.0
    n_elem = 0

    for batch in loader:
        s_t = batch["s_t"].to(dev)
        s_n = batch["s_next"].to(dev)
        y = batch["action"].to(dev)

        with torch.set_grad_enabled(train):
            pred = model(s_t, s_n)
            err_sum = torch.nn.functional.mse_loss(pred, y, reduction="sum")
            if train and optim is not None:
                optim.zero_grad(set_to_none=True)
                (err_sum / pred.numel()).backward()
                optim.step()

        sse += float(err_sum.detach().cpu().item())
        n_elem += int(pred.numel())

    return sse / max(1, n_elem)


def run_epoch_raid(
    decoder: RAIDDecoder,
    mem: RAIDMemoryBank,
    loader: DataLoader,
    dev: torch.device,
    optim: torch.optim.Optimizer | None,
    train: bool,
) -> float:
    if train:
        decoder.train()
    else:
        decoder.eval()

    sse = 0.0
    n_elem = 0

    for batch in loader:
        s_t = batch["s_t"].to(dev)
        s_n = batch["s_next"].to(dev)
        y = batch["action"].to(dev)
        idx = batch["idx"]

        with torch.set_grad_enabled(train):
            retr, mk = mem.retrieve_batch(
                s_t,
                s_n,
                k=3,
                tau_min=None,
                exclude_idx=(idx.to(mem.device)) if train else None,
            )
            prior = pooled_retrieved(retr.to(dev), mk.to(dev))
            pred = decoder(s_t, s_n, prior)

            err_sum = torch.nn.functional.mse_loss(pred, y, reduction="sum")
            if train and optim is not None:
                optim.zero_grad(set_to_none=True)
                (err_sum / pred.numel()).backward()
                optim.step()

        sse += float(err_sum.detach().cpu().item())
        n_elem += int(pred.numel())

    return sse / max(1, n_elem)


def run_epoch_raid_crossattn(
    decoder: RAIDDecoderCrossAttn,
    mem: RAIDMemoryBank,
    loader: DataLoader,
    dev: torch.device,
    optim: torch.optim.Optimizer | None,
    train: bool,
) -> float:
    if train:
        decoder.train()
    else:
        decoder.eval()

    sse = 0.0
    n_elem = 0

    for batch in loader:
        s_t = batch["s_t"].to(dev)
        s_n = batch["s_next"].to(dev)
        y = batch["action"].to(dev)
        idx = batch["idx"]

        with torch.set_grad_enabled(train):
            retr, mk = mem.retrieve_batch(
                s_t,
                s_n,
                k=3,
                tau_min=None,
                exclude_idx=(idx.to(mem.device)) if train else None,
            )
            retr_b = retr.to(dev)
            mk_b = mk.to(dev)
            kv_pad = ~mk_b  # True = ignore invalid retrieved slot

            pred, _ = decoder(s_t, s_n, retr_b, kv_key_padding_mask=kv_pad)

            err_sum = torch.nn.functional.mse_loss(pred, y, reduction="sum")
            if train and optim is not None:
                optim.zero_grad(set_to_none=True)
                (err_sum / pred.numel()).backward()
                optim.step()

        sse += float(err_sum.detach().cpu().item())
        n_elem += int(pred.numel())

    return sse / max(1, n_elem)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["direct_mlp", "raid", "raid_crossattn"], required=True)
    ap.add_argument("--n_demos", type=int, choices=[25, 50, 100, 200], required=True)
    args = ap.parse_args()

    seed_all()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={dev} condition={args.condition} n_demos={args.n_demos}")

    train_ds, val_ds, _stats = make_train_val(args.n_demos)
    stats_path = norm_stats_path(args.n_demos)

    mem: RAIDMemoryBank | None = None
    if args.condition in ("raid", "raid_crossattn"):
        mem = RAIDMemoryBank(
            obs_dim=train_ds.state_dim,
            action_dim=train_ds.action_dim,
            max_entries=50_000,
            device=dev,
        )
        mem.populate_from_dataset(train_ds, desc="Fill bank (train-only)")

    if args.condition == "direct_mlp":
        model = DirectMLP(train_ds.state_dim, train_ds.action_dim, hidden_dim=256, dropout=0.1).to(dev)
        optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    elif args.condition == "raid_crossattn":
        model = RAIDDecoderCrossAttn(
            obs_dim=train_ds.state_dim,
            action_dim=train_ds.action_dim,
            k=3,
            d_model=128,
            nhead=4,
        ).to(dev)
        optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        assert mem is not None
    else:
        model = RAIDDecoder(train_ds.state_dim, train_ds.action_dim, hidden_dim=256, dropout=0.1).to(dev)
        optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        assert mem is not None

    tr_loader = DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=False, num_workers=0)
    va_loader = DataLoader(val_ds, batch_size=256, shuffle=False, drop_last=False, num_workers=0)

    best_val = float("inf")
    best_epoch = -1

    train_curve: list[float] = []
    val_curve: list[float] = []

    for epoch in range(1, 51):
        if args.condition == "direct_mlp":
            tr_mse = run_epoch_direct(model, tr_loader, dev, optim, train=True)
            va_mse = run_epoch_direct(model, va_loader, dev, optim=None, train=False)
        elif args.condition == "raid_crossattn":
            tr_mse = run_epoch_raid_crossattn(model, mem, tr_loader, dev, optim, train=True)
            va_mse = run_epoch_raid_crossattn(model, mem, va_loader, dev, optim=None, train=False)
        else:
            tr_mse = run_epoch_raid(model, mem, tr_loader, dev, optim, train=True)
            va_mse = run_epoch_raid(model, mem, va_loader, dev, optim=None, train=False)

        train_curve.append(tr_mse)
        val_curve.append(va_mse)

        if va_mse < best_val:
            best_val = va_mse
            best_epoch = epoch
            ckpt_path = PROJECT_ROOT / "models" / f"{args.condition}_{args.n_demos}demos_best.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_mse": va_mse,
                    "condition": args.condition,
                    "n_demos": args.n_demos,
                    "norm_stats_path": str(stats_path.relative_to(PROJECT_ROOT)),
                },
                ckpt_path,
            )

        if epoch == 1 or epoch % 5 == 0 or epoch == 50:
            print(f"[train] epoch {epoch:02d}/50  train_mse={tr_mse:.6f}  val_mse={va_mse:.6f}  best={best_val:.6f}@{best_epoch}")

    curves_path = PROJECT_ROOT / "configs" / f"loss_curves_{args.condition}_{args.n_demos}demos.json"
    curves_path.parent.mkdir(parents=True, exist_ok=True)
    curves_path.write_text(json.dumps({"train_mse": train_curve, "val_mse": val_curve}, indent=2))
    print(f"[train] saved loss curves → {curves_path}")
    print(f"[train] best checkpoint val_mse={best_val:.6f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
