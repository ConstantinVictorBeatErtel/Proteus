"""Evaluation that produces both metrics and prediction-vs-GT panels.

For each completed cell we load ``ckpt_best.pt``, run the model over the
val split, and emit:

  * ``<artifact_root>/results/figures/predictions/<run_id>/<i>.png`` —
    one panel per sampled transition showing obs_t / obs_{t+1} / action
    bars (predicted vs ground truth). Up to ``n_panels`` panels are
    written; defaults to 12 evenly-spaced indices.
  * ``<artifact_root>/results/figures/predictions/<run_id>/grid.png`` —
    a single grid figure for at-a-glance review.
  * Aggregate MSE / contact-MSE / per-DOF MSE appended to
    ``<artifact_root>/results/matrix.parquet``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .legacy.memory import FeatureMemoryBank
from .legacy.models import DirectMLP, RAIDDecoder
from .heads import DiffusionPolicyIDM, TransformerIDM
from .runtime.drive import CheckpointDir, results_root
from .train import CellConfig, build_dataset, build_head
from .visualize import render_grid, render_transition


def _load_cell_config(run_id: str) -> CellConfig:
    ck = CheckpointDir(run_id)
    if not ck.best().is_file():
        raise FileNotFoundError(f"missing best checkpoint for run {run_id}")
    blob = torch.load(ck.best(), map_location="cpu", weights_only=False)
    return CellConfig(**blob["config"])


def _predict(head: str, model: torch.nn.Module, batch: dict[str, torch.Tensor], mem: FeatureMemoryBank | None, device: torch.device) -> torch.Tensor:
    obs_t = batch["obs_t"].to(device)
    obs_n = batch["obs_next"].to(device)
    if head == "direct_mlp":
        return model(obs_t, obs_n)
    if head == "raid":
        assert mem is not None
        from .train import _pooled_retrieved

        retr, mk = mem.retrieve_batch(obs_t, obs_n, k=3, tau_min=None, exclude_idx=None)
        prior = _pooled_retrieved(retr.to(device), mk.to(device))
        return model(obs_t, obs_n, prior)
    if head == "transformer":
        return model.forward_pair(obs_t, obs_n)
    if head == "diffusion":
        return model.sample(obs_t, obs_n)
    if head == "knn":
        return model(obs_t, obs_n)
    raise ValueError(head)


def evaluate_cell(run_id: str, n_panels: int = 12, render_panels: bool = True) -> dict[str, Any]:
    cfg = _load_cell_config(run_id)
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
        mem.populate_from_dataset(train_ds, desc="Eval bank", obs_t_key="obs_t", obs_next_key="obs_next")

    model = build_head(cfg, obs_dim, action_dim, memory=mem).to(device)
    blob = torch.load(CheckpointDir(run_id).best(), map_location=device, weights_only=False)
    model.load_state_dict(blob["model_state_dict"], strict=False)
    model.eval()

    va_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    sse_t = torch.zeros((), device=device, dtype=torch.float64)
    n_elem = 0
    contact_sse_t = torch.zeros((), device=device, dtype=torch.float64)
    contact_n = 0
    per_dof_sse_t = torch.zeros(action_dim, device=device, dtype=torch.float64)
    per_dof_n = 0

    panel_indices: set[int] = set()
    if render_panels and len(val_ds) > 0:
        n_panels = min(n_panels, len(val_ds))
        panel_indices = set(int(i) for i in np.linspace(0, len(val_ds) - 1, n_panels, dtype=int).tolist())
    captured: dict[int, dict[str, Any]] = {}

    with torch.no_grad():
        offset = 0
        for batch in va_loader:
            pred = _predict(cfg.head, model, batch, mem, device)
            y = batch["action"].to(device)
            err = pred - y
            err_sq = err * err
            sse_t = sse_t + err_sq.sum().double()
            n_elem += int(err.numel())
            iso = batch["is_contact"].to(device)
            if iso.any():
                e_c_sq = err_sq[iso]
                contact_sse_t = contact_sse_t + e_c_sq.sum().double()
                contact_n += int(e_c_sq.numel())
            per_dof_sse_t = per_dof_sse_t + err_sq.sum(dim=0).double()
            per_dof_n += int(err.shape[0])

            B = int(batch["action"].shape[0])
            wanted_local = [i for i in range(B) if (offset + i) in panel_indices]
            if wanted_local:
                pred_cpu = pred[wanted_local].detach().cpu().numpy()
                gt_cpu = batch["action"][wanted_local].detach().cpu().numpy()
                obs_t_cpu = batch["obs_t"][wanted_local].detach().cpu().numpy()
                obs_n_cpu = batch["obs_next"][wanted_local].detach().cpu().numpy()
                t_cpu = batch["t"][wanted_local].detach().cpu().numpy() if "t" in batch else None
                contact_cpu = batch["is_contact"][wanted_local].detach().cpu().numpy()
                demo_keys_batch = batch.get("demo_key")
                for j, i_in_batch in enumerate(wanted_local):
                    global_idx = offset + i_in_batch
                    captured[global_idx] = {
                        "pred": pred_cpu[j],
                        "gt": gt_cpu[j],
                        "obs_t": obs_t_cpu[j],
                        "obs_next": obs_n_cpu[j],
                        "demo_key": (demo_keys_batch[i_in_batch] if isinstance(demo_keys_batch, list) else "?"),
                        "t": int(t_cpu[j]) if t_cpu is not None else -1,
                        "is_contact": bool(contact_cpu[j]),
                    }
            offset += B

    val_mse = float(sse_t.item()) / max(1, n_elem)
    contact_mse = float(contact_sse_t.item()) / max(1, contact_n) if contact_n > 0 else None
    per_dof_mse = (per_dof_sse_t.detach().cpu() / max(1, per_dof_n)).tolist()

    metrics = {
        "run_id": run_id,
        "config": asdict(cfg),
        "val_mse": val_mse,
        "contact_mse": contact_mse,
        "per_dof_mse": per_dof_mse,
        "best_epoch": int(blob.get("epoch", -1)),
    }

    if render_panels and captured:
        out_dir = results_root() / "figures" / "predictions" / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        panels: list[dict] = []
        for k in sorted(captured):
            cap = captured[k]
            frame_t = None
            frame_next = None
            try:
                # If the underlying dataset supports raw frames, fetch them.
                ft, fn = val_ds.fetch_frames(k)  # type: ignore[attr-defined]
                frame_t = ft
                frame_next = fn
            except Exception:  # noqa: BLE001
                pass
            render_transition(
                save_path=out_dir / f"{k:05d}.png",
                frame_t=frame_t,
                frame_next=frame_next,
                action_gt=cap["gt"],
                action_pred=cap["pred"],
                title=f"{cfg.dataset} {cfg.head} demo={cap['demo_key']} t={cap['t']}",
                state_t=cap["obs_t"],
                state_next=cap["obs_next"],
            )
            panels.append({
                "frame_t": frame_t,
                "frame_next": frame_next,
                "action_gt": cap["gt"],
                "action_pred": cap["pred"],
                "title": f"idx={k}{' (contact)' if cap['is_contact'] else ''}",
            })
        render_grid(out_dir / "grid.png", panels, title=f"{cfg.dataset} / {cfg.head} / seed {cfg.seed}")

    out_path = CheckpointDir(run_id).path / "metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--n-panels", type=int, default=12)
    ap.add_argument("--no-panels", action="store_true")
    args = ap.parse_args()
    out = evaluate_cell(args.run_id, n_panels=args.n_panels, render_panels=not args.no_panels)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
