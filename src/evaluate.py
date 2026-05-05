"""Evaluate RAID and baselines on the Lift validation split."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data import make_train_val
from memory import RAIDMemoryBank
from models import DirectMLP, RAIDDecoder, RAIDDecoderCrossAttn

SEED = 42
SCALES = [25, 50, 100, 200]
K_RETR = 3


def seed_all() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def dev() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pooled_mean(actions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    den = mask.sum(dim=1, keepdim=True).clamp(min=1)
    num = (actions * mask.unsqueeze(-1).float()).sum(dim=1)
    return num / den.float()


def ratio_mse(sumsq: float, n: int) -> float | None:
    return None if n <= 0 else float(sumsq / float(n))


@torch.no_grad()
def eval_scale(n_demos: int, device: torch.device, tau_min: float | None) -> dict[str, dict[str, Any]]:
    train_ds, val_ds, _ = make_train_val(n_demos)
    val_ld = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

    bank = RAIDMemoryBank(
        obs_dim=train_ds.state_dim,
        action_dim=train_ds.action_dim,
        max_entries=50000,
        device=device,
    )
    bank.populate_from_dataset(train_ds, desc=f"Eval bank n={n_demos}")

    dpath = PROJECT_ROOT / "models" / f"direct_mlp_{n_demos}demos_best.pt"
    rpath = PROJECT_ROOT / "models" / f"raid_{n_demos}demos_best.pt"
    capp = PROJECT_ROOT / "models" / f"raid_crossattn_{n_demos}demos_best.pt"
    if not dpath.is_file():
        raise FileNotFoundError(str(dpath))
    if not rpath.is_file():
        raise FileNotFoundError(str(rpath))
    if not capp.is_file():
        raise FileNotFoundError(str(capp))

    dm = DirectMLP(train_ds.state_dim, train_ds.action_dim, hidden_dim=256, dropout=0.1).to(device)
    rd = RAIDDecoder(train_ds.state_dim, train_ds.action_dim, hidden_dim=256, dropout=0.1).to(device)
    ca = RAIDDecoderCrossAttn(
        obs_dim=train_ds.state_dim,
        action_dim=train_ds.action_dim,
        k=K_RETR,
    ).to(device)
    dm.load_state_dict(torch.load(dpath, map_location=device, weights_only=False)["model_state_dict"])
    rd.load_state_dict(torch.load(rpath, map_location=device, weights_only=False)["model_state_dict"])
    ca.load_state_dict(torch.load(capp, map_location=device, weights_only=False)["model_state_dict"])
    dm.eval()
    rd.eval()
    ca.eval()

    A = train_ds.action_dim
    names = ("mean_baseline", "nearest_neighbor", "direct_mlp", "raid", "raid_crossattn")
    acc: dict[str, dict[str, float | int | torch.Tensor]] = {}
    for nm in names:
        acc[nm] = {
            "sse": 0.0,
            "ne": 0,
            "dof": torch.zeros(A, dtype=torch.float64),
            "ns": 0,
            "sc": 0.0,
            "nc": 0,
            "sn": 0.0,
            "nn": 0,
        }

    qhits = 0
    qtot = 0
    for batch in val_ld:
        st = batch["s_t"].to(device)
        sn = batch["s_next"].to(device)
        y = batch["action"].to(device)
        iso = batch["is_contact"].to(device)

        retr, mk = bank.retrieve_batch(st, sn, k=K_RETR, tau_min=tau_min, exclude_idx=None)
        prior = pooled_mean(retr.to(device), mk.to(device))

        qhits += int(mk.any(dim=1).sum().item())
        qtot += int(mk.shape[0])

        kv_pad = ~mk.to(device)

        pred_ca, _ = ca(st, sn, retr.to(device), kv_key_padding_mask=kv_pad)

        preds = {
            "mean_baseline": torch.zeros_like(y),
            "nearest_neighbor": prior,
            "direct_mlp": dm(st, sn),
            "raid": rd(st, sn, prior),
            "raid_crossattn": pred_ca,
        }

        for nm, pr in preds.items():
            d = pr - y
            acc[nm]["sse"] += float((d * d).sum().item())
            acc[nm]["ne"] += int(d.numel())
            acc[nm]["dof"] += torch.sum(d * d, dim=0).detach().cpu().to(torch.float64)
            acc[nm]["ns"] += int(d.shape[0])

            if bool(iso.any().item()):
                dc = d[iso]
                acc[nm]["sc"] += float((dc * dc).sum().item())
                acc[nm]["nc"] += int(dc.numel())
            neg = ~iso
            if bool(neg.any().item()):
                dn = d[neg]
                acc[nm]["sn"] += float((dn * dn).sum().item())
                acc[nm]["nn"] += int(dn.numel())

    hr = float(qhits) / float(max(1, qtot))

    out: dict[str, dict[str, Any]] = {}
    for nm, a in acc.items():
        dof = (a["dof"] / float(max(1, int(a["ns"])))).tolist()
        out[nm] = {
            "mse": float(a["sse"]) / float(max(1, int(a["ne"]))),
            "contact_mse": ratio_mse(float(a["sc"]), int(a["nc"])),
            "noncontact_mse": ratio_mse(float(a["sn"]), int(a["nn"])),
            "per_dof_mse": dof,
            "hit_rate": (
                hr
                if nm in ("nearest_neighbor", "raid", "raid_crossattn")
                else None
            ),
        }
    return out


def fmt(x: float | None) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "nan"
    return f"{x:.6f}"


def print_table(all_rows: dict[str, dict[str, Any]]) -> None:
    print("\n" + "=" * 128)
    hdr = (
        f"{'n':>4} | {'mean':>10} | {'kNN':>10} | {'mlp':>10} | {'raid':>10} | {'xattn':>10} | "
        f"{'c_mb':>10} | {'c_knn':>10} | {'c_mlp':>10} | {'c_rd':>10} | {'c_xa':>10} | {'hit':>6}"
    )
    print(hdr)
    print("-" * 128)
    for nd in SCALES:
        sk = str(nd)
        hr = float(all_rows["nearest_neighbor"][sk]["hit_rate"] or 0.0)
        print(
            f"{nd:>4} | {fmt(all_rows['mean_baseline'][sk]['mse']):>10} | "
            f"{fmt(all_rows['nearest_neighbor'][sk]['mse']):>10} | "
            f"{fmt(all_rows['direct_mlp'][sk]['mse']):>10} | "
            f"{fmt(all_rows['raid'][sk]['mse']):>10} | "
            f"{fmt(all_rows['raid_crossattn'][sk]['mse']):>10} | "
            f"{fmt(all_rows['mean_baseline'][sk]['contact_mse']):>10} | "
            f"{fmt(all_rows['nearest_neighbor'][sk]['contact_mse']):>10} | "
            f"{fmt(all_rows['direct_mlp'][sk]['contact_mse']):>10} | "
            f"{fmt(all_rows['raid'][sk]['contact_mse']):>10} | "
            f"{fmt(all_rows['raid_crossattn'][sk]['contact_mse']):>10} | {hr:>6.3f}"
        )
    print("=" * 128 + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau-min", dest="tau_min", type=float, default=None)
    args = ap.parse_args()

    seed_all()
    device = dev()
    all_rows: dict[str, dict[str, Any]] = {
        k: {}
        for k in [
            "mean_baseline",
            "nearest_neighbor",
            "direct_mlp",
            "raid",
            "raid_crossattn",
        ]
    }

    for nd in SCALES:
        piece = eval_scale(nd, device, args.tau_min)
        for cond, met in piece.items():
            all_rows[cond][str(nd)] = met
        print(f"[eval] n_demos={nd} done")

    outp = PROJECT_ROOT / "configs" / "results.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(all_rows, indent=2))
    print(f"[eval] wrote {outp}")
    print_table(all_rows)


if __name__ == "__main__":
    main()

