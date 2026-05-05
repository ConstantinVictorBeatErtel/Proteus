"""
Robosuite episode rollouts for GRPO fine-tuning.

Inference note (paper): At rollout time ``s_next`` is unknown.
The decoder query uses ``s_next_proxy = s_t`` so the cross-attention query matches
training structure while only ``s_t`` drives retrieval (``retrieve_single``).
"""

from __future__ import annotations

import numpy as np
import torch

from data import OBS_KEYS

try:
    import robosuite as suite
except ImportError:  # pragma: no cover
    suite = None


def make_env():
    if suite is None:
        raise ImportError("robosuite is required. Install with: pip install robosuite")
    return suite.make(
        "Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=True,
        control_freq=20,
        horizon=200,
    )


def _object_vec(obs_dict: dict, target_dim: int = 10) -> np.ndarray:
    if "object" in obs_dict:
        v = np.asarray(obs_dict["object"], dtype=np.float32).ravel()
    else:
        v = np.asarray(obs_dict["object-state"], dtype=np.float32).ravel()
    if v.size >= target_dim:
        return v[:target_dim].copy()
    out = np.zeros(target_dim, dtype=np.float32)
    out[: v.size] = v
    return out


def extract_obs(obs_dict: dict, norm_stats: dict, device: torch.device | str = "cuda") -> torch.Tensor:
    """
    Concatenate robosuite obs dict into a normalized low-dim vector matching
    ``data.OBS_KEYS`` order (RoboMimic Lift low-dim layout).
    """
    dev = torch.device(device) if isinstance(device, str) else device
    parts: list[np.ndarray] = []
    for key in OBS_KEYS:
        if key == "object":
            parts.append(_object_vec(obs_dict, 10))
        else:
            parts.append(np.asarray(obs_dict[key], dtype=np.float32).ravel())

    s = np.concatenate(parts).astype(np.float32)
    sm = torch.as_tensor(norm_stats["state_mean"], dtype=torch.float32, device=dev)
    ss = torch.as_tensor(norm_stats["state_std"], dtype=torch.float32, device=dev)
    t = torch.as_tensor(s, device=dev, dtype=torch.float32)
    return (t - sm) / (ss + 1e-8)


def denormalize_action(a_norm: torch.Tensor, norm_stats: dict, device: torch.device | str = "cuda") -> torch.Tensor:
    dev = torch.device(device) if isinstance(device, str) else device
    mean = torch.as_tensor(norm_stats["action_mean"], dtype=torch.float32, device=dev)
    std = torch.as_tensor(norm_stats["action_std"], dtype=torch.float32, device=dev)
    return a_norm * std + mean


def _step_env(env, action_np: np.ndarray):
    out = env.step(action_np)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
        if not isinstance(info, dict):
            info = {}
        return obs, float(reward), done, info
    obs, reward, done, info = out  # type: ignore[misc]
    return obs, float(reward), bool(done), dict(info)


def run_episode(
    env,
    policy: torch.nn.Module,
    memory_bank,
    norm_stats: dict,
    device: torch.device | str = "cuda",
    deterministic: bool = False,
):
    """
    Run one episode. Returns:
      trajectory: list of (s_t, s_next, retrieved_actions, action_taken, log_prob)
      total_reward: sum of shaped rewards
      success: bool
    """
    dev = torch.device(device) if isinstance(device, str) else device

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    trajectory: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    total_reward = 0.0
    prev_s = extract_obs(obs, norm_stats, device=dev)

    info: dict = {}
    for _step in range(env.horizon):
        s_t = prev_s

        with torch.no_grad():
            retrieved = memory_bank.retrieve_single(s_t, k=3)
            retrieved_batch = retrieved.unsqueeze(0)

        # s_next unknown at inference: use s_t as proxy (see module docstring).
        s_next_proxy = s_t.unsqueeze(0)
        s_t_batch = s_t.unsqueeze(0)

        if deterministic:
            with torch.no_grad():
                pred, _ = policy(s_t_batch, s_next_proxy, retrieved_batch)
                action_norm = pred.squeeze(0)
                log_prob = torch.zeros(1, device=dev, dtype=torch.float32)
        else:
            pred, _ = policy(s_t_batch, s_next_proxy, retrieved_batch)
            std = policy.get_std()
            dist = torch.distributions.Normal(pred.squeeze(0), std)
            action_norm = dist.sample()
            log_prob = dist.log_prob(action_norm).sum()

        action = denormalize_action(action_norm, norm_stats, device=dev)
        action_np = action.detach().cpu().numpy().reshape(-1).clip(-1.0, 1.0)

        obs, reward, done, info = _step_env(env, action_np)
        total_reward += reward

        next_s = extract_obs(obs, norm_stats, device=dev)
        trajectory.append(
            (
                s_t,
                next_s,
                retrieved_batch.squeeze(0),
                action_norm.detach(),
                log_prob,
            )
        )

        prev_s = next_s
        if done:
            break

    success = bool(info.get("success", False))
    return trajectory, total_reward, success
