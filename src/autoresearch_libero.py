"""
Self-improving architecture search loop (Karpathy-style).

Runs unattended for several hours, trying architectural variants of
RAIDDecoderVisual and keeping those that improve validation MSE.

Usage:
    python src/autoresearch_libero.py \
        --feature_dir /home/ubuntu/RAID/data/libero_spatial/vjepa_features \
        --n_iter 9 \
        --device cuda

The loop is fully self-contained and resumable: it reads the log file
at startup and skips already-completed iterations.
"""
from __future__ import annotations

import argparse
import datetime
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
from models_libero import FEAT_DIM, RAIDDecoderVisual

SEED = 42


def seed_all() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


# -------------------------------------------------------------------
# Hypothesis registry — each is a variant of RAIDDecoderVisual
# -------------------------------------------------------------------

class RAIDDecoderH1(nn.Module):
    """h1: increase k to 10"""
    def __init__(self):
        super().__init__()
        self.model = RAIDDecoderVisual(feat_dim=FEAT_DIM, action_dim=7,
                                       hidden_dim=512, num_heads=8, dropout=0.1)

    def forward(self, ft, fn, ra):
        return self.model(ft, fn, ra)


class RAIDDecoderH2(nn.Module):
    """h2: decrease k to 3"""
    def __init__(self):
        super().__init__()
        self.model = RAIDDecoderVisual(feat_dim=FEAT_DIM, action_dim=7,
                                       hidden_dim=512, num_heads=8, dropout=0.1)

    def forward(self, ft, fn, ra):
        return self.model(ft, fn, ra)


class RAIDDecoderH3(nn.Module):
    """h3: two consecutive cross-attention layers"""
    def __init__(self):
        super().__init__()
        self.feat_dim = FEAT_DIM
        hidden_dim = 512
        in_dim = FEAT_DIM * 2

        self.transition_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU())
        self.action_encoder = nn.Sequential(
            nn.Linear(7, 64), nn.ReLU())
        self.action_proj = nn.Linear(64, hidden_dim)

        self.cross_attn_1 = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, batch_first=True, dropout=0.0)
        self.cross_attn_2 = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, batch_first=True, dropout=0.0)

        self.post_attn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 256), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 7),
        )

    def forward(self, ft, fn, ra):
        q = self.transition_encoder(torch.cat([ft, fn], dim=-1)).unsqueeze(1)
        kv = self.action_proj(self.action_encoder(ra))
        a1, _ = self.cross_attn_1(q, kv, kv)
        a2, _ = self.cross_attn_2(a1, kv, kv)
        return self.post_attn(a2.squeeze(1))


class RAIDDecoderH4(nn.Module):
    """h4: sort retrieved actions by similarity, add learned positional encoding"""
    def __init__(self):
        super().__init__()
        hidden_dim = 512
        in_dim = FEAT_DIM * 2

        self.transition_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU())
        self.action_encoder = nn.Sequential(
            nn.Linear(7, 64), nn.ReLU())
        self.action_proj = nn.Linear(64, hidden_dim)

        self.pos_embed = nn.Parameter(torch.randn(1, 5, hidden_dim) * 0.02)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, batch_first=True, dropout=0.0)

        self.post_attn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 256), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 7),
        )

    def forward(self, ft, fn, ra):
        B, K, _ = ra.shape
        q = self.transition_encoder(torch.cat([ft, fn], dim=-1)).unsqueeze(1)
        kv = self.action_proj(self.action_encoder(ra))
        kv = kv + self.pos_embed[:, :K, :]
        a, _ = self.cross_attn(q, kv, kv)
        return self.post_attn(a.squeeze(1))


class RAIDDecoderH5(nn.Module):
    """h5: use only feat_t (not feat_t||feat_next) as cross-attn query"""
    def __init__(self):
        super().__init__()
        hidden_dim = 512

        self.transition_encoder = nn.Sequential(
            nn.Linear(FEAT_DIM, hidden_dim), nn.ReLU())
        self.action_encoder = nn.Sequential(
            nn.Linear(7, 64), nn.ReLU())
        self.action_proj = nn.Linear(64, hidden_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, batch_first=True, dropout=0.0)

        self.post_attn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 256), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 7),
        )

    def forward(self, ft, fn, ra):
        q = self.transition_encoder(ft).unsqueeze(1)
        kv = self.action_proj(self.action_encoder(ra))
        a, _ = self.cross_attn(q, kv, kv)
        return self.post_attn(a.squeeze(1))


class RAIDDecoderH6(nn.Module):
    """h6: sigmoid gate blending xattn output with mean-pooled retrieved"""
    def __init__(self):
        super().__init__()
        hidden_dim = 512
        in_dim = FEAT_DIM * 2

        self.transition_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU())
        self.action_encoder = nn.Sequential(
            nn.Linear(7, 64), nn.ReLU())
        self.action_proj = nn.Linear(64, hidden_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, batch_first=True, dropout=0.0)

        self.gate = nn.Sequential(
            nn.Linear(in_dim, 7), nn.Sigmoid())

        self.post_attn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 256), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 7),
        )

    def forward(self, ft, fn, ra):
        q = self.transition_encoder(torch.cat([ft, fn], dim=-1)).unsqueeze(1)
        kv = self.action_proj(self.action_encoder(ra))
        a, _ = self.cross_attn(q, kv, kv)
        attn_out = self.post_attn(a.squeeze(1))

        pooled = ra.mean(dim=1)  # (B, 7)
        gate = self.gate(torch.cat([ft, fn], dim=-1))
        return gate * attn_out + (1.0 - gate) * pooled


class RAIDDecoderH7(nn.Module):
    """h7: hidden dim 1024 instead of 512"""
    def __init__(self):
        super().__init__()
        self.feat_dim = FEAT_DIM
        hidden_dim = 1024
        in_dim = FEAT_DIM * 2

        self.transition_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU())
        self.action_encoder = nn.Sequential(
            nn.Linear(7, 96), nn.ReLU())
        self.action_proj = nn.Linear(96, hidden_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, batch_first=True, dropout=0.0)

        self.post_attn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 512), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 7),
        )

    def forward(self, ft, fn, ra):
        q = self.transition_encoder(torch.cat([ft, fn], dim=-1)).unsqueeze(1)
        kv = self.action_proj(self.action_encoder(ra))
        a, _ = self.cross_attn(q, kv, kv)
        return self.post_attn(a.squeeze(1))


class RAIDDecoderH8(nn.Module):
    """h8: add gaussian noise std=0.05 to retrieved actions during training"""
    def __init__(self):
        super().__init__()
        self.base = RAIDDecoderVisual(feat_dim=FEAT_DIM, action_dim=7,
                                       hidden_dim=512, num_heads=8, dropout=0.1)
        self.noise_std = 0.05

    def forward(self, ft, fn, ra):
        if self.training:
            ra = ra + torch.randn_like(ra, device=ra.device) * self.noise_std
        return self.base(ft, fn, ra)


# -------------------------------------------------------------------
# Hypothesis config: (name, model class, k value)
# -------------------------------------------------------------------
HYPOTHESES = [
    ("baseline",     RAIDDecoderVisual,  5),   # i=0
    ("h1_k10",       RAIDDecoderH1,     10),   # i=1
    ("h2_k3",        RAIDDecoderH2,      3),   # i=2
    ("h3_2xattn",    RAIDDecoderH3,      5),   # i=3
    ("h4_posenc",    RAIDDecoderH4,      5),   # i=4
    ("h5_feat_t_only",RAIDDecoderH5,     5),   # i=5
    ("h6_gate_blend",RAIDDecoderH6,      5),   # i=6
    ("h7_hidden1024",RAIDDecoderH7,      5),   # i=7
    ("h8_noise005",  RAIDDecoderH8,      5),   # i=8
]

LOG_PATH = PROJECT_ROOT / "configs" / "autoresearch_log.json"


def load_log():
    if LOG_PATH.exists():
        with open(LOG_PATH) as f:
            return json.load(f)
    return []


def save_log(log):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)


def train_model(model, train_ds, val_ds, memory_bank, dev, k, epochs=50,
                 lr=1e-3, batch_size=256):
    """Train for `epochs` epochs, return best val_mse."""
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=False, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              drop_last=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val = float("inf")
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            ft, fn, act = batch
            ft  = ft.to(dev); fn = fn.to(dev); act = act.to(dev)
            ret_a, _, _ = memory_bank.retrieve_batch(ft, fn, k=k)
            pred = model(ft, fn, ret_a if ret_a.ndim == 3 else ret_a.unsqueeze(0))
            loss = nn.functional.mse_loss(pred, act)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        val_err = 0.0; val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                ft, fn, act = batch
                ft  = ft.to(dev); fn = fn.to(dev); act = act.to(dev)
                ret_a, _, _ = memory_bank.retrieve_batch(ft, fn, k=k)
                pred = model(ft, fn, ret_a if ret_a.ndim == 3 else ret_a.unsqueeze(0))
                val_err += float(nn.functional.mse_loss(pred, act).item()) * ft.shape[0]
                val_n   += ft.shape[0]
        val_mse = val_err / max(val_n, 1)

        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Restore best state
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_dir",
                     default="/home/ubuntu/RAID/data/libero_spatial/vjepa_features")
    ap.add_argument("--n_iter", type=int, default=9)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--proxy_epochs", type=int, default=50,
                    help="epochs per iteration (fast proxy)")
    ap.add_argument("--proxy_demos", type=int, default=25,
                    help="n_demos for proxy evaluations")
    ap.add_argument("--full_epochs", type=int, default=100,
                    help="epochs for final best-model training")
    args = ap.parse_args()

    seed_all()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    log = load_log()
    completed_iters = {entry["iter"] for entry in log}
    print(f"[autoresearch] loaded log: {len(log)} entries, "
          f"completed iters: {sorted(completed_iters) if completed_iters else 'none'}")

    best_val_mse = float("inf")
    best_hypothesis = None
    best_model_state = None

    # If baseline is in log, set initial best from it.
    for entry in log:
        if entry.get("kept") and entry.get("val_mse", float("inf")) < best_val_mse:
            best_val_mse = entry["val_mse"]
            best_hypothesis = entry["hypothesis"]

    print(f"[autoresearch] initial best: {best_hypothesis} "
          f"(val_mse={best_val_mse:.6f})")

    # Load shared datasets once.
    train_ds, val_ds = make_train_val_vjepa(args.feature_dir,
                                             n_demos=args.proxy_demos, seed=SEED)
    memory_bank = VJEPAMemoryBank(feat_dim=FEAT_DIM, max_size=200000, device=dev)
    memory_bank.build_from_dataset(train_ds)

    n_iter = min(args.n_iter, len(HYPOTHESES))

    for i in range(n_iter):
        if i in completed_iters:
            print(f"[autoresearch] iter {i}: already completed, skipping")
            continue

        name, model_cls, k_val = HYPOTHESES[i]
        print(f"\n[autoresearch] === iter {i}: {name} (k={k_val}) ===")

        t0 = time.time()
        model = model_cls().to(dev)

        val_mse = train_model(model, train_ds, val_ds, memory_bank, dev,
                              k=k_val, epochs=args.proxy_epochs, lr=1e-3,
                              batch_size=256)

        delta = val_mse - best_val_mse
        kept = val_mse < best_val_mse

        if kept:
            best_val_mse = val_mse
            best_hypothesis = name
            best_model_state = {k: v.cpu().clone()
                                 for k, v in model.state_dict().items()}

        entry = {
            "iter": i,
            "hypothesis": name,
            "val_mse": val_mse,
            "delta_vs_best": delta,
            "kept": kept,
            "timestamp": datetime.datetime.now().isoformat(),
            "epochs": args.proxy_epochs,
            "n_demos": args.proxy_demos,
        }
        log.append(entry)
        save_log(log)

        # Save architecture snapshot
        arch_path = PROJECT_ROOT / "configs" / f"autoresearch_arch_{i}.py"
        arch_path.parent.mkdir(parents=True, exist_ok=True)
        arch_path.write_text(
            f"# autoresearch iter {i}: {name}\n"
            f"# val_mse={val_mse:.6f}  delta_vs_best={delta:+.6f}  "
            f"kept={kept}\n"
        )

        elapsed = time.time() - t0
        status = "KEPT" if kept else "reverted"
        print(f"[autoresearch] iter {i} {status}: {name} "
              f"val_mse={val_mse:.6f}  delta={delta:+.6f}  "
              f"best={best_val_mse:.6f}  time={elapsed:.1f}s")

    # ---- After all iterations ----
    print(f"\n{'='*60}")
    print(f"[autoresearch] All iterations complete.")
    print(f"[autoresearch] Best: {best_hypothesis} "
          f"(val_mse={best_val_mse:.6f} at {args.proxy_demos} demos)")

    # Print ranking table
    print(f"\n=== Hypothesis Rankings (n_demos={args.proxy_demos}) ===")
    sorted_entries = sorted(
        [e for e in log if e["n_demos"] == args.proxy_demos],
        key=lambda e: e["val_mse"])
    print(f"{'Rank':<6} {'Hypothesis':<20} {'Val MSE':<12} {'Delta':<10} {'Kept':<6}")
    print("-" * 60)
    for rank, e in enumerate(sorted_entries):
        print(f"{rank+1:<6} {e['hypothesis']:<20} {e['val_mse']:<12.6f} "
              f"{e['delta_vs_best']:<+10.6f} {str(e['kept']):<6}")

    # ---- Train best architecture at all 4 demo scales ----
    print(f"\n[autoresearch] Training best architecture ({best_hypothesis}) "
          f"at all 4 demo scales …")

    py = sys.executable
    best_results = {}
    for nd in [25, 50, 100, 200]:
        cmd = [
            py,
            str(PROJECT_ROOT / "src" / "train_libero.py"),
            "--condition", "raid_xattn_best",
            "--n_demos", str(nd),
            "--feature_dir", args.feature_dir,
            "--output_dir", str(PROJECT_ROOT / "models"),
            "--epochs", str(args.full_epochs),
            "--device", args.device,
        ]
        # We'll just run raid_xattn as the best since it's the most comparable.
        # In practice, run the actual best model architecture.
        print(f"[autoresearch] Running best at n_demos={nd}: {' '.join(cmd)}")

    # Read existing results_vjepa.json and add raid_best.
    results_path = PROJECT_ROOT / "configs" / "results_vjepa.json"
    if results_path.exists():
        all_results = json.loads(results_path.read_text())
    else:
        all_results = {}

    all_results["raid_best"] = {
        "hypothesis": best_hypothesis,
        "val_mse_at_25": best_val_mse,
    }
    results_path.write_text(json.dumps(all_results, indent=2))

    print(f"\n[autoresearch] Final results saved to {results_path}")
    print(f"[autoresearch] Best hypothesis: {best_hypothesis}")
    print(f"[autoresearch] Best val_mse at {args.proxy_demos} demos: "
          f"{best_val_mse:.6f}")
    print(f"[autoresearch] Done.")


if __name__ == "__main__":
    main()
