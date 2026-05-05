"""
Run GRPO fine-tuning on top of BC-pretrained RAIDDecoderCrossAttn.

Usage:
    python src/run_grpo.py --n_demos 200 --n_updates 500 --G 8

Requires:
    - models/raid_crossattn_{n_demos}demos_best.pt (from Stage 1 BC training)
    - configs/norm_stats_{n_demos}demos.pt
    - data/lift/ph/low_dim_v141.hdf5
    - robosuite installed: pip install robosuite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data import load_norm_stats, make_train_val  # noqa: E402
from grpo import grpo_train  # noqa: E402
from memory import RAIDMemoryBank  # noqa: E402
from models import RAIDDecoderCrossAttn  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_demos", type=int, default=200)
    parser.add_argument("--n_updates", type=int, default=500)
    parser.add_argument("--G", type=int, default=8)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available() and args.device == "cuda":
        print("[run_grpo] CUDA unavailable; using CPU.")

    norm_stats = load_norm_stats(args.n_demos)
    obs_dim = int(norm_stats["state_mean"].shape[0])

    ckpt_path = PROJECT_ROOT / "models" / f"raid_crossattn_{args.n_demos}demos_best.pt"
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload

    policy = RAIDDecoderCrossAttn(obs_dim=obs_dim).to(device)
    policy.load_state_dict(sd)
    print(f"Loaded BC policy from {ckpt_path}. obs_dim={obs_dim}")

    train_ds, _, _ = make_train_val(args.n_demos)

    memory_bank = RAIDMemoryBank(
        obs_dim=train_ds.state_dim,
        action_dim=train_ds.action_dim,
        max_entries=50_000,
        device=device,
    )
    memory_bank.populate_from_dataset(train_ds, desc="GRPO bank (train-only)")
    print(f"[run_grpo] Memory bank populated: ptr={memory_bank.ptr}")

    log = grpo_train(
        policy=policy,
        memory_bank=memory_bank,
        norm_stats=norm_stats,
        n_updates=args.n_updates,
        G=args.G,
        beta=args.beta,
        device=device,
        checkpoint_dir="models",
        project_root=PROJECT_ROOT,
    )
    sr_vals = []
    for e in log:
        v = e.get("success_rate")
        if isinstance(v, (int, float)):
            sr_vals.append(float(v))
    sr_max = max(sr_vals, default=0.0)
    print(f"\nDone. Best SR in logged updates: {sr_max:.3f}")


if __name__ == "__main__":
    main()
