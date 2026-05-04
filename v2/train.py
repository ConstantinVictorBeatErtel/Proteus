"""Config-driven IDM trainer with W&B-resumable Drive checkpoints.

This entry point is intentionally distinct from ``src/train.py``; the
legacy CLI continues to reproduce the autoresearch baseline at
val_mse ~ 0.397.

Run a single cell directly:

    python3 -m v2.train \
        --head direct_mlp --dataset robomimic_lift_ph_low_dim \
        --n_demos 25 --seed 42

Or via the matrix orchestrator (preferred):

    python3 -m v2.run_matrix --phase A
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .legacy.memory import FeatureMemoryBank
from .legacy.models import DirectMLP, RAIDDecoder
from .heads import DiffusionPolicyIDM, TransformerIDM
from .runtime.drive import CheckpointDir, atomic_save, runs_root, results_root
from .runtime.wandb_resume import deterministic_run_id, init_run


CKPT_INTERVAL_SEC = 300  # 5 min Drive write cadence


@dataclasses.dataclass
class CellConfig:
    phase: str
    head: str
    dataset: str  # canonical adapter name, e.g. "robomimic_lift_ph_low_dim"
    n_demos: int
    seed: int
    encoder: str | None = None  # only set for image-feature cells
    n_epochs: int = 50
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4

    @property
    def run_id(self) -> str:
        return deterministic_run_id(
            self.phase, self.head, self.dataset, self.encoder or "none",
            self.n_demos, self.seed,
        )


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset(dataset_name: str, n_demos: int, seed: int, encoder: str | None) -> tuple[Dataset, Dataset, int, int]:
    """Resolve the dataset name to ``(train_ds, val_ds, obs_dim, action_dim)``.

    Image-feature cells are dispatched only when ``encoder`` is set; in
    that case the obs is the precomputed CLS feature loaded from
    ``features/<dataset>_<encoder>_cls.safetensors``. We do not eagerly
    materialize image-feature datasets in this base implementation
    because we may run on a fresh Colab session where features are
    cached but the heavy HF/HDF5 deps might not be on path; the matrix
    runner handles those branches.
    """
    if dataset_name.startswith("robomimic_") and "_low_dim" in dataset_name:
        from .runtime.drive import data_root
        from .datasets import robomimic as rm

        _, task, variant, _ = dataset_name.split("_", 3)
        spec = rm.RoboMimicSpec(task=task, variant=variant, modality="low_dim")
        train_ds, val_ds, _stats, state_dim = rm.make_train_val(
            spec=spec, n_demos=n_demos, data_root=data_root() / "robomimic",
        )
        return train_ds, val_ds, state_dim, 7
    raise NotImplementedError(
        f"build_dataset for {dataset_name!r}: implement image-feature "
        "and LIBERO branches in the matrix runner"
    )


def _pooled_retrieved(actions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
    summed = (actions * mask.unsqueeze(-1).float()).sum(dim=1)
    return summed / denom.float()


def build_head(cfg: CellConfig, obs_dim: int, action_dim: int) -> torch.nn.Module:
    if cfg.head == "direct_mlp":
        return DirectMLP(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=256, dropout=0.1)
    if cfg.head == "raid":
        return RAIDDecoder(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=256, dropout=0.1)
    if cfg.head == "transformer":
        return TransformerIDM(obs_dim=obs_dim, action_dim=action_dim, seq_len=4)
    if cfg.head == "diffusion":
        return DiffusionPolicyIDM(obs_dim=obs_dim, action_dim=action_dim, n_obs_steps=2)
    raise ValueError(f"unknown head {cfg.head!r}")


def _train_step(
    head: str,
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    mem: FeatureMemoryBank | None,
    device: torch.device,
    train: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    obs_t = batch["obs_t"].to(device)
    obs_n = batch["obs_next"].to(device)
    y = batch["action"].to(device)
    if head == "direct_mlp":
        pred = model(obs_t, obs_n)
    elif head == "raid":
        assert mem is not None
        retr, mk = mem.retrieve_batch(
            obs_t, obs_n, k=3, tau_min=None,
            exclude_idx=batch["idx"].to(mem.device) if train else None,
        )
        prior = _pooled_retrieved(retr.to(device), mk.to(device))
        pred = model(obs_t, obs_n, prior)
    elif head == "transformer":
        pred = model.forward_pair(obs_t, obs_n)
    elif head == "diffusion":
        if train:
            loss = model.loss(obs_t, obs_n, y)
            return loss, loss.detach()
        pred = model.sample(obs_t, obs_n)
    else:
        raise ValueError(head)
    loss = torch.nn.functional.mse_loss(pred, y)
    return loss, pred.detach()


def train_cell(cfg: CellConfig, project: str = "raid_v2") -> dict[str, Any]:
    seed_all(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, val_ds, obs_dim, action_dim = build_dataset(cfg.dataset, cfg.n_demos, cfg.seed, cfg.encoder)

    model = build_head(cfg, obs_dim, action_dim).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    mem: FeatureMemoryBank | None = None
    if cfg.head == "raid":
        mem = FeatureMemoryBank(obs_dim=obs_dim, action_dim=action_dim, max_entries=200_000, device=device)
        mem.populate_from_dataset(train_ds, desc="Fill bank", obs_t_key="obs_t", obs_next_key="obs_next")

    tr_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False, num_workers=0)
    va_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=0)

    ck = CheckpointDir(cfg.run_id)
    best_val = math.inf
    best_epoch = -1
    last_ckpt_at = time.time()

    history: list[dict[str, float]] = []

    with init_run(project=project, run_id=cfg.run_id, config=dataclasses.asdict(cfg)) as run:
        for epoch in range(1, cfg.n_epochs + 1):
            # train
            model.train()
            tr_sse = 0.0
            tr_n = 0
            for batch in tr_loader:
                loss, _pred = _train_step(cfg.head, model, batch, mem, device, train=True)
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
                tr_sse += float(loss.detach().cpu().item()) * batch["action"].numel()
                tr_n += int(batch["action"].numel())
            tr_mse = tr_sse / max(1, tr_n)

            # val
            model.eval()
            va_sse = 0.0
            va_n = 0
            with torch.no_grad():
                for batch in va_loader:
                    loss, _pred = _train_step(cfg.head, model, batch, mem, device, train=False)
                    va_sse += float(loss.detach().cpu().item()) * batch["action"].numel()
                    va_n += int(batch["action"].numel())
            va_mse = va_sse / max(1, va_n)

            history.append({"epoch": epoch, "train_mse": tr_mse, "val_mse": va_mse})
            run.log({"epoch": epoch, "train_mse": tr_mse, "val_mse": va_mse}, step=epoch)

            if va_mse < best_val:
                best_val = va_mse
                best_epoch = epoch
                atomic_save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": dataclasses.asdict(cfg),
                        "epoch": epoch,
                        "val_mse": va_mse,
                    },
                    ck.best(),
                )

            now = time.time()
            if now - last_ckpt_at > CKPT_INTERVAL_SEC or epoch == cfg.n_epochs:
                atomic_save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": dataclasses.asdict(cfg),
                        "epoch": epoch,
                        "val_mse": va_mse,
                    },
                    ck.last(),
                )
                last_ckpt_at = now

        result = {
            "run_id": cfg.run_id,
            "best_val_mse": best_val,
            "best_epoch": best_epoch,
            "final_val_mse": history[-1]["val_mse"] if history else math.nan,
            "config": dataclasses.asdict(cfg),
        }
        # Persist a small JSON next to the run for cheap aggregation later.
        (ck.path / "result.json").write_text(json.dumps(result, indent=2))
        return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="ad-hoc")
    ap.add_argument("--head", required=True, choices=["direct_mlp", "raid", "transformer", "diffusion"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n_demos", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--encoder", default=None)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--project", default="raid_v2")
    args = ap.parse_args()

    cfg = CellConfig(
        phase=args.phase, head=args.head, dataset=args.dataset,
        n_demos=args.n_demos, seed=args.seed, encoder=args.encoder,
        n_epochs=args.epochs,
    )
    out = train_cell(cfg, project=args.project)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
