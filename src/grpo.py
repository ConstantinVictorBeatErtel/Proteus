"""GRPO-style fine-tuning for RAIDDecoderCrossAttn (BC-pretrained)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from rollout import make_env, run_episode


def grpo_train(
    policy,  # RAIDDecoderCrossAttn (BC pretrained)
    memory_bank,
    norm_stats: dict,
    n_updates: int = 500,
    G: int = 8,
    beta: float = 0.04,
    lr: float = 3e-5,
    clip_grad: float = 0.5,
    log_every: int = 25,
    checkpoint_dir: str | Path = "models",
    project_root: Path | None = None,
    device: str | torch.device = "cuda",
):
    """
    GRPO fine-tuning of RAIDDecoderCrossAttn policy.

    For each update:
      1. Sample G rollouts from current policy
      2. Compute episode rewards (shaped + success bonus)
      3. Normalize rewards within group → advantages
      4. Policy gradient loss + KL penalty to frozen BC reference
      5. Update policy weights

    Memory bank stays frozen. Only policy weights update.
    """
    dev = torch.device(device) if isinstance(device, str) else device
    ckpt_root = Path(checkpoint_dir) if project_root is None else Path(project_root) / checkpoint_dir
    ckpt_root.mkdir(parents=True, exist_ok=True)
    configs_dir = Path("configs") if project_root is None else Path(project_root) / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    ref_policy = copy.deepcopy(policy)
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad_(False)

    policy.train()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)

    env = make_env()
    log: list[dict] = []
    best_sr = 0.0
    log_path = configs_dir / "grpo_log.json"

    def _persist_log() -> None:
        log_path.write_text(json.dumps(log, indent=2))

    log_path.write_text(json.dumps([], indent=2))

    try:
        for update in range(n_updates):
            rollouts: list[tuple[list, float, bool]] = []

            for _g in range(G):
                traj, total_reward, success = run_episode(
                    env,
                    policy,
                    memory_bank,
                    norm_stats,
                    device=dev,
                    deterministic=False,
                )
                reward = total_reward + (10.0 if success else 0.0)
                rollouts.append((traj, reward, success))

            rewards = torch.tensor([r for _, r, _ in rollouts], dtype=torch.float32, device=dev)
            successes = [s for _, _, s in rollouts]

            sr_pre = float(sum(successes)) / float(G)

            if rewards.std() < 1e-6:
                log.append(
                    {
                        "update": int(update),
                        "success_rate": sr_pre,
                        "mean_reward": float(rewards.mean().item()),
                        "reward_std": float(rewards.std().item()),
                        "skipped_gradient": True,
                        "skip_reason": "near_constant_group_rewards",
                    }
                )
                _persist_log()
                continue

            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

            # Batch gradient computation: one forward pass per rollout (not per step).
            total_loss_val = 0.0
            n_terms = 0
            optimizer.zero_grad(set_to_none=True)
            policy.train()

            for (traj, _, _), adv in zip(rollouts, advantages):
                if not traj:
                    continue
                # Stack entire rollout into batches for a single forward pass.
                # s_next_b uses the same s_t proxy as the rollout (on-policy consistency).
                s_t_b    = torch.stack([t[0] for t in traj])          # (T, obs)
                s_next_b = s_t_b                                       # proxy: same as rollout
                retr_b   = torch.stack([t[2] for t in traj])          # (T, k, 7)
                acts_b   = torch.stack([t[3] for t in traj])          # (T, 7)
                T = s_t_b.shape[0]
                kv_pad = torch.zeros(T, retr_b.shape[1], dtype=torch.bool, device=dev)

                pred, _ = policy(s_t_b, s_next_b, retr_b, kv_key_padding_mask=kv_pad)
                std = policy.get_std()
                dist = torch.distributions.Normal(pred, std)
                log_probs = dist.log_prob(acts_b).sum(dim=-1)  # (T,)

                with torch.no_grad():
                    ref_pred, _ = ref_policy(s_t_b, s_next_b, retr_b, kv_key_padding_mask=kv_pad)
                    ref_dist = torch.distributions.Normal(ref_pred, ref_policy.get_std())
                    ref_lps = ref_dist.log_prob(acts_b).sum(dim=-1)   # (T,)

                kl = (log_probs - ref_lps).mean()
                pg = -(adv * log_probs.mean())
                rollout_loss = pg + beta * kl
                rollout_loss.backward()
                total_loss_val += rollout_loss.item()
                n_terms += 1

            if n_terms == 0:
                log.append(
                    {
                        "update": int(update),
                        "success_rate": sr_pre,
                        "mean_reward": float(rewards.mean().item()),
                        "reward_std": float(rewards.std().item()),
                        "skipped_gradient": True,
                        "skip_reason": "no_loss_terms",
                    }
                )
                _persist_log()
                continue

            torch.nn.utils.clip_grad_norm_(policy.parameters(), clip_grad)
            optimizer.step()
            total_loss_val /= max(n_terms, 1)

            sr = float(sum(successes)) / float(G)
            entry = {
                "update": int(update),
                "success_rate": sr,
                "mean_reward": float(rewards.mean().item()),
                "reward_std": float(rewards.std().item()),
                "loss_mean": total_loss_val,
            }
            log.append(entry)
            _persist_log()

            if update % log_every == 0:
                print(
                    f"update {update:4d} | SR: {sr:.2f} | mean_r: {rewards.mean():.2f} | "
                    f"loss: {total_loss_val:.4f}",
                    flush=True,
                )

            if sr > best_sr:
                best_sr = sr
                torch.save(policy.state_dict(), ckpt_root / "raid_crossattn_grpo_best.pt")

    finally:
        _persist_log()
        env.close()
    return log
