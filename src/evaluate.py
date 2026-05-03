#!/usr/bin/env python3
"""Load checkpoints, calibrated contact threshold from train tactile, aggregate val MSE metrics."""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

CONDITIONS = ("vision_only", "tactile_only", "visuo_tactile")
DOF_LABELS = ["eef_x", "eef_y", "eef_z", "rot_x", "rot_y", "rot_z", "gripper"]

CONFIG_DIR = os.path.join(REPO_ROOT, "configs")
MODEL_DIR = os.path.join(REPO_ROOT, "models")
NORM_PATH = os.path.join(CONFIG_DIR, "norm_stats.pt")
RESULTS_JSON = os.path.join(CONFIG_DIR, "results.json")
CONTACT_PT = os.path.join(CONFIG_DIR, "contact_threshold.pt")

HORIZON = 10


def tactile_max_activation(tac: torch.Tensor) -> torch.Tensor:
    """(B, H, 12, 64) → (B, H) spatial max per timestep."""
    return tac.amax(dim=(2, 3))


def _stack_clip(
    emb_by_task: Dict[str, torch.Tensor],
    tasks: Sequence[str],
    starts: torch.Tensor,
    horizon: int,
    device: torch.device,
) -> torch.Tensor:
    b = len(tasks)
    out = torch.empty(b, horizon, 512, device=device, dtype=torch.float32)
    for i in range(b):
        tnm = tasks[i]
        s = int(starts[i].item())
        out[i] = emb_by_task[tnm][s : s + horizon].to(device=device, dtype=torch.float32)
    return out


@torch.no_grad()
def accumulate_val_stats(
    model: torch.nn.Module,
    loader,
    emb_val: Dict[str, torch.Tensor],
    condition: str,
    thresh: float,
    device: torch.device,
    tasks_tuple: Sequence[str],
) -> Dict[str, Any]:
    model.eval()

    sse_all = 0.0
    n_elem_all = 0

    sse_c = 0.0
    n_elem_c = 0
    sse_f = 0.0
    n_elem_f = 0

    sse_dim = [0.0] * 7
    n_per_dof = 0  # scalar count per dof (same for all dof)
    # per-dimension over all pooled timesteps
    for batch in loader:
        _, tac_seq, action_seq, task_idx, start_t = batch
        tac_seq = tac_seq.to(device=device, dtype=torch.float32)
        action_seq = action_seq.to(device=device, dtype=torch.float32)

        bt, h, _, _ = tac_seq.shape
        task_names = [tasks_tuple[int(task_idx[j])] for j in range(bt)]

        if condition == "vision_only":
            z_vis = _stack_clip(
                emb_val, task_names, start_t.to(device=device), h, device
            )
            pred = model(z_vis, None)
        elif condition == "tactile_only":
            z_dummy = torch.zeros(bt, h, 512, device=device)
            pred = model(z_dummy, tac_seq)
        else:
            z_vis = _stack_clip(
                emb_val, task_names, start_t.to(device=device), h, device
            )
            pred = model(z_vis, tac_seq)

        err = pred - action_seq  # (B, H, 7)
        e2 = err * err

        n_per_dof += bt * h
        sse_all += float(e2.sum().item())
        n_elem_all += e2.numel()

        mx = tactile_max_activation(tac_seq).reshape(-1)
        ef = e2.reshape(-1, 7)
        m_contact = mx > thresh
        sse_c += float(ef[m_contact].sum().item())
        n_elem_c += int(ef[m_contact].numel())
        m_free = ~m_contact
        sse_f += float(ef[m_free].sum().item())
        n_elem_f += int(ef[m_free].numel())

        for d in range(7):
            sse_dim[d] += float(e2[:, :, d].sum().item())

    def mse(sum_sq: float, n_el: int) -> float:
        return float(sum_sq / max(n_el, 1))

    return {
        "mse_overall": mse(sse_all, n_elem_all),
        "mse_contact": mse(sse_c, n_elem_c),
        "mse_noncontact": mse(sse_f, n_elem_f),
        "mse_per_dof": {
            DOF_LABELS[d]: float(sse_dim[d] / max(n_per_dof, 1)) for d in range(7)
        },
        "counts": {
            "total_elements": n_elem_all,
            "contact_elements": n_elem_c,
            "free_elements": n_elem_f,
        },
    }


def calibrate_contact_threshold(pct_nonzero: float = 10.0) -> Tuple[float, Dict[str, Any]]:
    """Train-set calibration: pct_nonzero percentile of timestep max tactile (excluding ~0 readings)."""
    import data as datamod

    if not os.path.isfile(NORM_PATH):
        raise FileNotFoundError(
            f"Missing {NORM_PATH}; run training once or `python src/data.py` to build norm stats."
        )
    stats = torch.load(NORM_PATH, map_location="cpu", weights_only=False)
    t_min = stats["tactile_min"].numpy().astype(np.float64)
    t_span = stats["tactile_span"].numpy().astype(np.float64)
    t_span = np.maximum(t_span, 1e-12)

    vals: List[float] = []
    for task in datamod.TASKS:
        root = datamod._open_zarr_for_task(task)
        ends = np.array(root["meta"]["episode_ends"][:]).reshape(-1)
        ranges = datamod._episode_ranges(ends)
        train_ids, _ = datamod._split_episode_ids(len(ranges))
        tactile_ds = root["data"]["camera0_tactile"]

        n_added = 0
        for ep_idx in train_ids:
            s_e, e_e = ranges[ep_idx]
            if e_e <= s_e:
                continue
            # Tactile at timestep t pairs with transition t→t+1 for t in [s_e, e_e)
            block = np.asarray(tactile_ds[s_e:e_e], dtype=np.float64)
            if block.ndim != 3 or block.shape[1:] != (12, 64):
                raise ValueError(
                    f"Unexpected tactile shape for {task} ep {ep_idx}: {block.shape}"
                )
            norm = np.clip((block - t_min) / t_span, 0.0, 1.0)
            mx = norm.reshape(norm.shape[0], -1).max(axis=1).astype(np.float64)
            vals.extend(mx.tolist())
            n_added += int(mx.shape[0])

        print(
            f"[evaluate] contact calibration scan: {task} "
            f"(train episodes={len(train_ids)}, timesteps={n_added})"
        )

    arr = np.array(vals, dtype=np.float64)
    pos = arr[arr > 1e-6]
    if pos.size > 10:
        th = float(np.percentile(pos, pct_nonzero))
    else:
        th = float(np.percentile(arr, pct_nonzero)) if arr.size else 0.05

    meta = {
        "percentile_nonzero_used": pct_nonzero,
        "num_timesteps_seen": len(vals),
        "num_nonzero_max": int((arr > 1e-6).sum()),
        "threshold": th,
        "note": ("10th pct of timestep max tactile (excluding near-zero flats) "
                 "unless too few nonzero; then pct on all."),
    }
    torch.save(meta, CONTACT_PT)
    print(f"[evaluate] Saved contact calibration → {CONTACT_PT}")
    print(f"[evaluate] contact threshold = {th:.6g} ({meta})")
    return th, meta


def main() -> None:
    import data as datamod
    from torch.utils.data import DataLoader

    import train as tr

    torch.manual_seed(42)
    np.random.seed(42)

    os.makedirs(CONFIG_DIR, exist_ok=True)

    thresh, meta = calibrate_contact_threshold(10.0)

    emb_val = tr.load_clip_embeddings_per_task("val")

    val_ds = datamod.VTBCWindowDataset(
        split="val",
        norm_stats_path=NORM_PATH,
        horizon=HORIZON,
        seed=42,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=64,
        shuffle=False,
        drop_last=False,
        num_workers=tr.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tasks_list = tuple(datamod.TASKS)

    out_payload: Dict[str, Any] = {
        "contact_calibration": meta,
        "dof_labels": DOF_LABELS,
        "normalized_actions": True,
        "conditions": {},
    }

    print("\n" + "=" * 72)
    print(f"{'condition':<16} {'overall':>12} {'contact':>12} {'non-contact':>12}")
    print("=" * 72)

    for cond in CONDITIONS:
        ckpt_path = os.path.join(MODEL_DIR, f"{cond}_best.pt")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"Missing checkpoint {ckpt_path}. Train first: python src/train.py --condition {cond}"
            )

        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = tr.make_policy(cond)  # type: ignore[arg-type]
        model.load_state_dict(ck["model_state"])
        model.to(device)

        stats = accumulate_val_stats(
            model,
            val_loader,
            emb_val,
            cond,
            thresh,
            device,
            tasks_list,
        )
        out_payload["conditions"][cond] = stats

        print(
            f"{cond:<16} {stats['mse_overall']:12.6f} "
            f"{stats['mse_contact']:12.6f} {stats['mse_noncontact']:12.6f}"
        )

    print("=" * 72 + "\n")

    with open(RESULTS_JSON, "w", encoding="utf-8") as fp:
        json.dump(out_payload, fp, indent=2)
    print(f"[evaluate] Wrote aggregated results → {RESULTS_JSON}")


if __name__ == "__main__":
    main()
