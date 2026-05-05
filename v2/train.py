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
from .heads import DiffusionPolicyIDM, KNNRetrievalHead, TransformerIDM
from .runtime.drive import CheckpointDir, atomic_save, runs_root, results_root
from .runtime.wandb_resume import deterministic_run_id, init_run


CKPT_INTERVAL_SEC = 300  # 5 min Drive write cadence

# Heads that have no learnable behavior; the training loop short-circuits
# but still produces a checkpoint and val_mse for matrix aggregation.
NO_TRAIN_HEADS = {"knn"}


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
    action_norm_mode: str = "zscore"

    @property
    def run_id(self) -> str:
        return deterministic_run_id(
            self.phase, self.head, self.dataset, self.encoder or "none",
            self.n_demos, self.seed, self.action_norm_mode,
        )


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


_MIXED_LOWDIM_FULL = (
    ("lift", "ph"), ("lift", "mh"),
    ("can", "ph"), ("can", "mh"),
    ("square", "ph"), ("square", "mh"),
)


def _build_robomimic_lowdim(task: str, variant: str, n_demos: int, action_norm_mode: str):
    from .runtime.drive import data_root
    from .datasets import robomimic as rm

    spec = rm.RoboMimicSpec(task=task, variant=variant, modality="low_dim")
    return rm.make_train_val(
        spec=spec, n_demos=n_demos, data_root=data_root() / "robomimic",
        action_norm_mode=action_norm_mode,
    )


def build_dataset(
    dataset_name: str,
    n_demos: int,
    seed: int,
    encoder: str | None,
    action_norm_mode: str = "zscore",
) -> tuple[Dataset, Dataset, int, int]:
    """Resolve the dataset name to ``(train_ds, val_ds, obs_dim, action_dim)``.

    Recognized dataset names:

    * ``robomimic_<task>_<variant>_low_dim`` — single-task RoboMimic,
      proprioceptive state.
    * ``mixed_robomimic_lowdim_full`` / ``mixed_robomimic_lowdim_subset25``
      — concatenation across the 7-D-action RoboMimic tasks
      (PH+MH); narrower state vectors are right-padded to the widest
      task so cross-task low-dim cells run cleanly. ``transport`` is
      excluded here because its low-dim action width is not 7-D. The
      ``subset25`` variant trims each child to 25% of the requested
      demo count.
    * ``robomimic_<task>_<variant>_image`` — RoboMimic image-feature
      cells; expects cached CLS features at
      ``<artifact_root>/features/<dataset>_<encoder>_cls.safetensors``.
    * ``libero_{spatial,object,goal}`` — LIBERO suites, image-feature
      via the same cache as above.

    Image-feature cells require ``encoder`` to be set; the cached
    features are loaded into a :class:`v2.datasets.cached.FeatureCachedDataset`
    that exposes the same dict schema as the low-dim adapters.
    """
    # Single-task RoboMimic low-dim.
    if dataset_name.startswith("robomimic_") and dataset_name.endswith("_low_dim"):
        rest = dataset_name[len("robomimic_") : -len("_low_dim")]
        task, variant = rest.split("_")
        train_ds, val_ds, _stats, state_dim = _build_robomimic_lowdim(
            task, variant, n_demos, action_norm_mode
        )
        return train_ds, val_ds, state_dim, 7

    # Mixed-task RoboMimic low-dim.
    if dataset_name.startswith("mixed_robomimic_lowdim_"):
        from .datasets.mixed import MixedIDMDataset, PaddedObservationDataset

        suffix = dataset_name[len("mixed_robomimic_lowdim_") :]
        scale = 1.0 if suffix in {"full", "100pct"} else 0.25 if suffix in {"subset25", "25pct"} else None
        if scale is None:
            raise ValueError(f"unknown mixed-lowdim variant: {suffix!r}")
        per_task_n = max(1, int(round(n_demos * scale)))
        train_members: list[tuple[str, Dataset]] = []
        val_members: list[tuple[str, Dataset]] = []
        max_state_dim = -1
        pending: list[tuple[str, Dataset, Dataset, int]] = []
        for task, variant in _MIXED_LOWDIM_FULL:
            try:
                tr, va, _stats, sd = _build_robomimic_lowdim(task, variant, per_task_n, action_norm_mode)
            except FileNotFoundError as exc:
                print(f"[build_dataset] skipping {task}/{variant}: {exc}")
                continue
            name = f"robomimic_{task}_{variant}_low_dim"
            pending.append((name, tr, va, sd))
            max_state_dim = max(max_state_dim, sd)
        for name, tr, va, sd in pending:
            if sd != max_state_dim:
                print(
                    f"[build_dataset] padding {name} from obs_dim={sd} to mixed obs_dim={max_state_dim}"
                )
                tr = PaddedObservationDataset(tr, target_obs_dim=max_state_dim)
                va = PaddedObservationDataset(va, target_obs_dim=max_state_dim)
            train_members.append((name, tr))
            val_members.append((name, va))
        if not train_members:
            raise FileNotFoundError(
                "No RoboMimic tasks could be loaded for the mixed dataset; "
                "run ``python3 -m v2.runtime.data_download`` first."
            )
        return MixedIDMDataset(train_members), MixedIDMDataset(val_members), max_state_dim, 7

    # Image-feature cells (RoboMimic image / LIBERO) — load cached features.
    if encoder is not None:
        from .datasets.cached import build_feature_cached_train_val

        train_ds, val_ds, obs_dim = build_feature_cached_train_val(
            dataset_name=dataset_name, encoder=encoder, n_demos=n_demos,
            action_norm_mode=action_norm_mode,
        )
        return train_ds, val_ds, obs_dim, 7

    raise NotImplementedError(
        f"build_dataset: don't know how to load {dataset_name!r}. "
        "Image-feature datasets require ``encoder`` to be set; "
        "supported low-dim names are ``robomimic_<task>_<variant>_low_dim`` "
        "and ``mixed_robomimic_lowdim_{full,subset25}``."
    )


def _pooled_retrieved(actions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
    summed = (actions * mask.unsqueeze(-1).float()).sum(dim=1)
    return summed / denom.float()


def build_head(
    cfg: CellConfig,
    obs_dim: int,
    action_dim: int,
    memory: FeatureMemoryBank | None = None,
) -> torch.nn.Module:
    if cfg.head == "direct_mlp":
        return DirectMLP(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=256, dropout=0.1)
    if cfg.head == "raid":
        return RAIDDecoder(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=256, dropout=0.1)
    if cfg.head == "transformer":
        return TransformerIDM(obs_dim=obs_dim, action_dim=action_dim, seq_len=4)
    if cfg.head == "diffusion":
        clip = (-1.0, 1.0) if cfg.action_norm_mode == "q01_q99" else None
        return DiffusionPolicyIDM(
            obs_dim=obs_dim, action_dim=action_dim, n_obs_steps=2,
            clip_action_range=clip,
        )
    if cfg.head == "knn":
        if memory is None:
            raise ValueError("kNN head requires a populated FeatureMemoryBank")
        return KNNRetrievalHead(memory=memory, k=3)
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
        obs_window = batch.get("obs_window")
        pred = model(obs_window.to(device)) if obs_window is not None else model.forward_pair(obs_t, obs_n)
    elif head == "diffusion":
        if train:
            loss = model.loss(obs_t, obs_n, y)
            return loss, loss.detach()
        pred = model.sample(obs_t, obs_n)
    elif head == "knn":
        pred = model(obs_t, obs_n)
    else:
        raise ValueError(head)
    loss = torch.nn.functional.mse_loss(pred, y)
    return loss, pred.detach()


def train_cell(cfg: CellConfig, project: str = "raid_v2") -> dict[str, Any]:
    seed_all(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, val_ds, obs_dim, action_dim = build_dataset(
        cfg.dataset, cfg.n_demos, cfg.seed, cfg.encoder, action_norm_mode=cfg.action_norm_mode,
    )

    mem: FeatureMemoryBank | None = None
    if cfg.head in {"raid", "knn"}:
        mem = FeatureMemoryBank(
            obs_dim=obs_dim, action_dim=action_dim,
            max_entries=max(1024, len(train_ds) + 64),
            device=device,
        )
        mem.populate_from_dataset(train_ds, desc="Fill bank", obs_t_key="obs_t", obs_next_key="obs_next")

    model = build_head(cfg, obs_dim, action_dim, memory=mem).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    tr_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False, num_workers=0)
    va_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=0)

    ck = CheckpointDir(cfg.run_id)
    best_val = math.inf
    best_epoch = -1
    last_ckpt_at = time.time()

    history: list[dict[str, float]] = []

    no_train = cfg.head in NO_TRAIN_HEADS
    effective_epochs = 1 if no_train else cfg.n_epochs

    with init_run(project=project, run_id=cfg.run_id, config=dataclasses.asdict(cfg)) as run:
        for epoch in range(1, effective_epochs + 1):
            # train
            tr_mse = float("nan")
            if not no_train:
                model.train()
                tr_sse_t = torch.zeros((), device=device)
                tr_n = 0
                for batch in tr_loader:
                    loss, _pred = _train_step(cfg.head, model, batch, mem, device, train=True)
                    optim.zero_grad(set_to_none=True)
                    loss.backward()
                    optim.step()
                    tr_sse_t = tr_sse_t + loss.detach() * batch["action"].numel()
                    tr_n += int(batch["action"].numel())
                tr_mse = float(tr_sse_t.item()) / max(1, tr_n)

            # val
            model.eval()
            va_sse_t = torch.zeros((), device=device)
            va_n = 0
            with torch.no_grad():
                for batch in va_loader:
                    loss, _pred = _train_step(cfg.head, model, batch, mem, device, train=False)
                    va_sse_t = va_sse_t + loss.detach() * batch["action"].numel()
                    va_n += int(batch["action"].numel())
            va_mse = float(va_sse_t.item()) / max(1, va_n)

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
            if now - last_ckpt_at > CKPT_INTERVAL_SEC or epoch == effective_epochs:
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
    ap.add_argument("--head", required=True, choices=["direct_mlp", "raid", "transformer", "diffusion", "knn"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n_demos", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--encoder", default=None)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--action_norm", default="zscore", choices=["zscore", "q01_q99"])
    ap.add_argument("--project", default="raid_v2")
    args = ap.parse_args()

    cfg = CellConfig(
        phase=args.phase, head=args.head, dataset=args.dataset,
        n_demos=args.n_demos, seed=args.seed, encoder=args.encoder,
        n_epochs=args.epochs, action_norm_mode=args.action_norm,
    )
    out = train_cell(cfg, project=args.project)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
