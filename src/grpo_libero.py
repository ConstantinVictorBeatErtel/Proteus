"""
GRPO fine-tuning for RAIDDecoderVisual on LIBERO using GR-1 world model.

Usage:
    python src/grpo_libero.py \
        --n_demos   200 \
        --n_updates 200 \
        --G         4   \
        --task_idx  0   \
        --device    cuda
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

# Headless rendering: use osmesa (CPU software renderer) on servers without display
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("MUJOCO_GL", "osmesa")

# Force unbuffered stdout for real-time log visibility
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import torch

sys.path.insert(0, str(Path(__file__).parent))

from models import RAIDDecoderVisual
from memory import RAIDMemoryBank
from gr1_encoder import GR1Encoder
from train_libero import populate_memory_from_cache, _load_cached_datasets
from rollout_libero import make_libero_env, run_episode
from data_libero import find_hdf5_files, load_norm_stats


# ---------------------------------------------------------------------------

def grpo_train(
    policy: RAIDDecoderVisual,
    memory_bank: RAIDMemoryBank,
    encoder: GR1Encoder,
    norm_stats: dict,
    env,
    language: str,
    n_updates: int = 200,
    G: int = 4,
    beta: float = 0.04,
    lr: float = 3e-5,
    clip_grad: float = 0.5,
    log_every: int = 10,
    checkpoint_path: str = "models/raid_visual_grpo_best.pt",
    log_path: str = "configs/grpo_libero_log.json",
    device: str | torch.device = "cuda",
):
    dev = torch.device(device) if isinstance(device, str) else device

    # Frozen BC reference policy
    ref_policy = copy.deepcopy(policy)
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad_(False)

    policy.train()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)

    log: list[dict] = []
    best_sr = 0.0
    Path(log_path).parent.mkdir(exist_ok=True)

    print(f"Starting GRPO: {n_updates} updates × G={G} rollouts each")

    for update in range(n_updates):
        rollouts = []

        # --- Sample G rollouts ---
        for _g in range(G):
            traj, total_reward, success = run_episode(
                env, policy, memory_bank, encoder, norm_stats,
                language=language, device=dev, deterministic=False,
            )
            reward = total_reward + (10.0 if success else 0.0)
            rollouts.append((traj, reward, success))

        rewards = torch.tensor([r for _, r, _ in rollouts], dtype=torch.float32, device=dev)
        successes = [s for _, _, s in rollouts]

        if rewards.std() < 1e-6:
            # Add tiny noise so advantages are non-zero, keeping entropy alive
            rewards = rewards + torch.randn_like(rewards) * 1e-4

        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # --- Gradient step: batch per rollout ---
        optimizer.zero_grad(set_to_none=True)
        total_loss_val = 0.0

        policy.train()
        for (traj, _, _), adv in zip(rollouts, advantages):
            if not traj:
                continue
            feat_t_b    = torch.stack([t[0] for t in traj]).to(dev)   # (T, 384)
            feat_next_b = torch.stack([t[1] for t in traj]).to(dev)   # (T, 384)
            retr_b      = torch.stack([t[2] for t in traj]).to(dev)   # (T, k, 7)
            acts_b      = torch.stack([t[3] for t in traj]).to(dev)   # (T, 7)
            T = feat_t_b.shape[0]
            kv_pad = torch.zeros(T, retr_b.shape[1], dtype=torch.bool, device=dev)

            pred, _ = policy(feat_t_b, feat_next_b, retr_b, kv_key_padding_mask=kv_pad)
            std = policy.log_std.exp().clamp(1e-4, 1.0)
            dist = torch.distributions.Normal(pred, std)
            log_probs = dist.log_prob(acts_b).sum(dim=-1)  # (T,)

            with torch.no_grad():
                ref_pred, _ = ref_policy(feat_t_b, feat_next_b, retr_b,
                                         kv_key_padding_mask=kv_pad)
                ref_dist = torch.distributions.Normal(ref_pred, ref_policy.log_std.exp().clamp(1e-4, 1.0))
                ref_lps = ref_dist.log_prob(acts_b).sum(dim=-1)

            kl = (log_probs - ref_lps).mean()
            pg = -(adv * log_probs.mean())
            loss = pg + beta * kl
            loss.backward()
            total_loss_val += loss.item()

        torch.nn.utils.clip_grad_norm_(policy.parameters(), clip_grad)
        optimizer.step()

        sr = float(sum(successes)) / float(G)
        entry = {
            "update": update,
            "success_rate": sr,
            "mean_reward": float(rewards.mean()),
            "reward_std": float(rewards.std()),
            "loss": total_loss_val / max(len(rollouts), 1),
        }
        log.append(entry)
        Path(log_path).write_text(json.dumps(log, indent=2))

        if update % log_every == 0:
            print(f"update {update:4d} | SR={sr:.2f} | r={rewards.mean():.2f} | "
                  f"loss={total_loss_val:.3f}", flush=True)

        if sr > best_sr:
            best_sr = sr
            torch.save(policy.state_dict(), checkpoint_path)
            print(f"  ✓ New best SR={sr:.2f} — checkpoint saved")

    print(f"\nDone. Best SR: {best_sr:.3f}")
    return log


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_demos",      type=int, default=200)
    p.add_argument("--n_updates",    type=int, default=200)
    p.add_argument("--G",            type=int, default=4)
    p.add_argument("--beta",         type=float, default=0.04)
    p.add_argument("--task_idx",     type=int, default=0,
                   help="Which LIBERO-Spatial task to run GRPO on (0-9)")
    p.add_argument("--feature_dir",  default="data/libero_spatial/features")
    p.add_argument("--dataset_dir",  default="data/libero_spatial/libero_spatial")
    p.add_argument("--gr1_ckpt",     default="checkpoints/gr1/snapshot_ABCD.pt")
    p.add_argument("--mae_ckpt",     default="checkpoints/gr1/mae_pretrain_vit_base.pth")
    p.add_argument("--device",       default="cuda")
    p.add_argument("--log_every",    type=int, default=5)
    args = p.parse_args()

    dev = args.device
    feature_dir = Path(args.feature_dir)

    # --- Load norm stats ---
    norm_path = Path(args.dataset_dir) / "norm_stats.pt"
    norm_stats = load_norm_stats(norm_path)
    print(f"Loaded norm stats from {norm_path}")

    # --- Load frozen GR-1 encoder ---
    print("Loading GR-1 encoder ...")
    encoder = GR1Encoder.from_checkpoints(args.mae_ckpt, args.gr1_ckpt, dev)
    feat_dim = encoder.feat_dim
    print(f"  feat_dim={feat_dim}")

    # --- Load BC checkpoint ---
    ckpt_path = Path("models") / f"raid_visual_{args.n_demos}demos_libero_best.pt"
    if not ckpt_path.exists():
        # Try with fewer demos
        candidates = sorted(Path("models").glob("raid_visual_*demos_libero_best.pt"))
        if candidates:
            ckpt_path = candidates[-1]
            print(f"Using checkpoint: {ckpt_path}")
        else:
            raise FileNotFoundError("No raid_visual checkpoint found. Run run_all_libero.py first.")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    policy = RAIDDecoderVisual(feat_dim=feat_dim).to(dev)
    policy.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded BC policy from {ckpt_path}")

    # --- Populate memory bank ---
    print("Populating memory bank ...")
    memory_bank = populate_memory_from_cache(feature_dir, args.n_demos, dev, feat_dim=feat_dim)
    print(f"  Memory bank: {memory_bank.ptr} entries")

    # --- Make LIBERO env ---
    hdf5_files = find_hdf5_files(args.dataset_dir)
    task_idx = min(args.task_idx, len(hdf5_files) - 1)
    task_file = hdf5_files[task_idx]

    import h5py, json as _json
    with h5py.File(task_file, "r") as f:
        problem_info = _json.loads(f["data"].attrs.get("problem_info", "{}"))
        language = problem_info.get("language_instruction", "robot manipulation").strip('"\'')

    print(f"Task: {language}", flush=True)
    print(f"Making LIBERO env ...", flush=True)

    # Build path to BDDL file directly (avoids interactive benchmark prompt)
    _libero_root = Path("/home/ubuntu/LIBERO/libero/libero")
    _bddl_dir = _libero_root / "bddl_files" / "libero_spatial"
    _bddl_files = sorted(_bddl_dir.glob("*.bddl")) if _bddl_dir.exists() else []

    if len(_bddl_files) > task_idx:
        bddl_file = str(_bddl_files[task_idx])
    else:
        # Fallback: use benchmark API with stdin redirect
        import sys as _sys
        _orig_stdin = _sys.stdin
        _sys.stdin = open("/dev/null")
        try:
            from libero.libero import benchmark as bm_mod
            bm = bm_mod.get_benchmark_dict()["libero_spatial"]()
            bddl_file = bm.get_task_bddl_file_path(task_idx)
        finally:
            _sys.stdin = _orig_stdin

    print(f"  BDDL: {Path(bddl_file).name}", flush=True)
    from libero.libero.envs import OffScreenRenderEnv
    env = OffScreenRenderEnv(**{
        "bddl_file_name": bddl_file,
        "camera_heights": 128,
        "camera_widths": 128,
    })
    print(f"  LIBERO env ready", flush=True)

    # --- Run GRPO ---
    grpo_train(
        policy=policy,
        memory_bank=memory_bank,
        encoder=encoder,
        norm_stats=norm_stats,
        env=env,
        language=language,
        n_updates=args.n_updates,
        G=args.G,
        beta=args.beta,
        log_every=args.log_every,
        device=dev,
    )


if __name__ == "__main__":
    main()
