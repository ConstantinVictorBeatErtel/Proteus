#!/usr/bin/env python3
"""
Training loop for VT-BC: vision_only, tactile_only, visuo_tactile policies.

CLIP embeddings MUST be cached (python src/cache_clip.py) — training fails loudly if missing.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, Literal, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

Condition = Literal["vision_only", "tactile_only", "visuo_tactile"]

NUM_WORKERS = int(os.environ.get("VTBC_NUM_WORKERS", "4"))

CLIP_CACHE_DIR = os.path.join(REPO_ROOT, "data", "clip_cache")
CONFIG_DIR = os.path.join(REPO_ROOT, "configs")
MODEL_DIR = os.path.join(REPO_ROOT, "models")
NORM_PATH = os.path.join(CONFIG_DIR, "norm_stats.pt")

SEED = 42

BATCH_SIZE = 64
EPOCHS = 50
LR = 3e-4
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
HORIZON = 10


def seed_everything(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def clip_cache_missing_message(split: str) -> str:
    return (
        f"CLIP embeddings for split '{split}' are missing under {CLIP_CACHE_DIR}.\n"
        "Run once:  python src/cache_clip.py\n"
        f"Expect files named: {{task}}_{split}.pt for each task."
    )


def load_clip_embeddings_per_task(split: str) -> Dict[str, torch.Tensor]:
    """Load full-timestep CLIP embeddings for every task."""
    import data as datamod

    out: Dict[str, torch.Tensor] = {}
    for task in datamod.TASKS:
        path = Path(CLIP_CACHE_DIR) / f"{task}_{split}.pt"
        if not path.is_file():
            raise FileNotFoundError(clip_cache_missing_message(split))
        blob = torch.load(path, map_location="cpu", weights_only=False)
        emb = blob["embeddings"]
        n_cached = emb.shape[0]
        n_real = datamod.task_num_timesteps(task)
        if n_cached != n_real:
            raise ValueError(
                f"CLIP cache length mismatch for {task}: cache={n_cached}, zarr={n_real}"
            )
        out[task] = emb.float()
        print(f"[train] Loaded CLIP cache {path.name} ({n_cached} × {emb.shape[1]})")
    return out


def stack_clip_sequences(
    emb_by_task: Dict[str, torch.Tensor],
    tasks: Sequence[str],
    starts: torch.Tensor,
    horizon: int,
) -> torch.Tensor:
    """tasks: iterable length B of task names; starts: LongTensor shape (B,). Output (B, H, 512)."""
    b = len(tasks)
    device = starts.device
    out = torch.empty(b, horizon, 512, device=device, dtype=torch.float32)
    for i in range(b):
        tnm = tasks[i]
        s = int(starts[i].item())
        out[i] = emb_by_task[tnm][s : s + horizon].to(device=device, dtype=torch.float32)
    return out


def make_policy(
    condition: Condition,
):
    import encoders as encmods
    from policy import TactileOnlyPolicy, VisionOnlyPolicy, VisuoTactilePolicy

    tac_enc = encmods.TactileEncoder(grid_h=12, grid_w=64, out_dim=64)

    if condition == "vision_only":
        pol = VisionOnlyPolicy(
            d_visual=512,
            d_model=512,
            n_layers=3,
            n_heads=8,
            dropout=0.1,
            action_dim=7,
        )
    elif condition == "tactile_only":
        pol = TactileOnlyPolicy(
            d_tactile_enc=64,
            d_model=512,
            n_layers=3,
            n_heads=8,
            dropout=0.1,
            action_dim=7,
            tactile_encoder=tac_enc,
        )
    else:
        pol = VisuoTactilePolicy(
            d_visual=512,
            d_tactile=64,
            d_model=512,
            n_layers=3,
            n_heads=8,
            dropout=0.1,
            action_dim=7,
            tactile_encoder=tac_enc,
        )
    return pol


def run_epoch_train(
    model,
    loader: DataLoader,
    emb_train: Dict[str, torch.Tensor],
    condition: Condition,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_sum = 0.0
    n_elems = 0
    tasks_list = tuple(__import__("data").TASKS)

    for batch in loader:
        _, tac_seq, action_seq, task_idx, start_t = batch
        tac_seq = tac_seq.to(device, dtype=torch.float32)
        action_seq = action_seq.to(device, dtype=torch.float32)

        bt = tac_seq.shape[0]
        task_names = [tasks_list[int(task_idx[j])] for j in range(bt)]

        optimizer.zero_grad(set_to_none=True)

        pred = None
        if condition == "vision_only":
            z_visual = stack_clip_sequences(
                emb_train, task_names, start_t.to(device=device), tac_seq.shape[1]
            )
            pred = model(z_visual.to(device), None)
        elif condition == "tactile_only":
            z_dummy = torch.zeros(bt, tac_seq.shape[1], 512, device=device)
            pred = model(z_dummy, tac_seq.to(device))
        else:
            z_visual = stack_clip_sequences(
                emb_train, task_names, start_t.to(device=device), tac_seq.shape[1]
            )
            pred = model(z_visual.to(device), tac_seq.to(device))

        loss_sum = F.mse_loss(pred, action_seq, reduction="sum")
        loss_sum.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()

        total_sum += loss_sum.detach().cpu().item()
        n_elems += pred.numel()
    return total_sum / max(n_elems, 1)


@torch.no_grad()
def run_epoch_eval(
    model,
    loader: DataLoader,
    emb_val: Dict[str, torch.Tensor],
    condition: Condition,
    device: torch.device,
) -> float:
    model.eval()
    total_sum = 0.0
    n_elems = 0
    tasks_list = tuple(__import__("data").TASKS)

    for batch in loader:
        _, tac_seq, action_seq, task_idx, start_t = batch
        tac_seq = tac_seq.to(device, dtype=torch.float32)
        action_seq = action_seq.to(device, dtype=torch.float32)
        bt = tac_seq.shape[0]
        task_names = [tasks_list[int(task_idx[j])] for j in range(bt)]

        if condition == "vision_only":
            z_visual = stack_clip_sequences(
                emb_val, task_names, start_t.to(device=device), tac_seq.shape[1]
            )
            pred = model(z_visual.to(device), None)
        elif condition == "tactile_only":
            z_dummy = torch.zeros(bt, tac_seq.shape[1], 512, device=device)
            pred = model(z_dummy, tac_seq.to(device))
        else:
            z_visual = stack_clip_sequences(
                emb_val, task_names, start_t.to(device=device), tac_seq.shape[1]
            )
            pred = model(z_visual.to(device), tac_seq.to(device))

        loss_sum = F.mse_loss(pred, action_seq, reduction="sum")
        total_sum += loss_sum.cpu().item()
        n_elems += pred.numel()

    return total_sum / max(n_elems, 1)


def ensure_tactile_caches_for_run() -> None:
    import data as dm

    force = os.environ.get("VTBC_FORCE_TACTILE_CACHE", "").lower() in ("1", "true", "yes")
    print("[train] Ensuring tactile npy caches (see data/tactile_cache/) …")
    dm.ensure_all_tactile_caches(force=force)


def train_one_condition(condition: Condition) -> None:
    import data as datamod

    ensure_tactile_caches_for_run()
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n========== Train condition: {condition} on {device} ==========")

    print("[train] Verifying CLIP caches exist …")
    emb_train = load_clip_embeddings_per_task("train")
    emb_val = load_clip_embeddings_per_task("val")

    train_ds = datamod.VTBCWindowDataset(
        split="train",
        norm_stats_path=NORM_PATH,
        horizon=HORIZON,
        seed=SEED,
    )
    val_ds = datamod.VTBCWindowDataset(
        split="val",
        norm_stats_path=NORM_PATH,
        horizon=HORIZON,
        seed=SEED,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    model = make_policy(condition).to(device)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )

    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    best_val = float("inf")
    best_path = Path(MODEL_DIR) / f"{condition}_best.pt"
    loss_log = {"train": [], "val": []}

    for epoch in range(1, EPOCHS + 1):
        train_mse = run_epoch_train(
            model, train_loader, emb_train, condition, optimizer, device
        )
        val_mse = run_epoch_eval(model, val_loader, emb_val, condition, device)
        scheduler.step()

        loss_log["train"].append(train_mse)
        loss_log["val"].append(val_mse)

        is_best = val_mse < best_val
        if is_best:
            best_val = val_mse
            torch.save(
                {
                    "condition": condition,
                    "epoch": epoch,
                    "val_mse": val_mse,
                    "model_state": model.state_dict(),
                },
                best_path,
            )

        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            extra = "  **[new best]**" if is_best else ""
            print(
                f"  Epoch {epoch:03d}/{EPOCHS} — "
                f"train MSE={train_mse:.6f}  val MSE={val_mse:.6f}{extra}"
            )

    log_path = Path(CONFIG_DIR) / f"{condition}_losses.json"
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(loss_log, f, indent=2)
    print(f"[train] Saved loss history → {log_path}")
    print(f"[train] Saved best checkpoint ({best_val:.6f}) → {best_path}")


def train_all_three() -> None:
    train_one_condition("vision_only")
    train_one_condition("tactile_only")
    train_one_condition("visuo_tactile")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition",
        choices=["vision_only", "tactile_only", "visuo_tactile", "all"],
        default="all",
    )
    args = parser.parse_args()

    if args.condition == "all":
        train_all_three()
    else:
        train_one_condition(args.condition)  # type: ignore[arg-type]
