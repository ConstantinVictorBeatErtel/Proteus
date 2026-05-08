"""
LIBERO episode rollouts for GRPO fine-tuning with GR-1 world model.

Key difference from rollout.py (RoboMimic):
  - Uses LIBERO's sim environment
  - GR-1 predicts next-frame features (predict_next_feat) = no (s_t, s_t) proxy hack
  - Memory bank keyed on GR-1 visual features, not low-dim state

Speed note: osmesa (CPU) rendering takes ~337ms/step on this A10 instance.
Set max_steps=30 for GRPO (30 × 337ms × G=4 ≈ 40s/update).
With EGL (GPU rendering) this would be ~5ms/step — upgrade when available.

Inference flow per step:
    frame_t  →  GR-1.encode_frames  →  feat_t          (384-dim)
    frame_t  →  GR-1.predict_next_feat  →  feat_next   (384-dim, world model)
    (feat_t, feat_next, top-k retrieved actions)  →  RAIDDecoderVisual  →  action_norm
    action_norm  →  denormalise  →  LIBERO sim step
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from gr1_encoder import GR1Encoder
from memory import RAIDMemoryBank


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

def _get_frame(obs) -> np.ndarray:
    """Extract (H, W, 3) uint8 RGB from LIBERO obs dict."""
    if isinstance(obs, dict):
        for key in ("agentview_image", "agentview_rgb", "image"):
            if key in obs:
                img = np.asarray(obs[key], dtype=np.uint8)
                if img.ndim == 3:
                    return img
    raise KeyError(f"No image key in obs. Keys: {list(obs.keys()) if isinstance(obs, dict) else type(obs)}")


def _shaped_reward(obs: dict, prev_obs: dict | None = None) -> float:
    """
    Dense shaped reward for LIBERO pick-and-place using proprioceptive observations.

    Components:
      - reach:   negative distance from EE to nearest object (always active)
      - lift:    +1 if object is lifted above table
    """
    reward = 0.0
    # Distance from EE to object (lower = better → negative reward)
    for key in ("akita_black_bowl_1_to_robot0_eef_pos",
                "bowl_to_robot0_eef_pos",):
        if key in obs:
            dist = float(np.linalg.norm(obs[key]))
            reward += -0.5 * dist   # shaped reach reward
            break

    # Lift reward: bowl z > some threshold (table ~= 0.8m in LIBERO)
    for key in ("akita_black_bowl_1_pos", "bowl_pos"):
        if key in obs:
            bowl_z = float(np.asarray(obs[key])[2])
            if bowl_z > 0.9:   # lifted off table
                reward += 0.5
            break

    return reward


def _get_robot_state(obs) -> np.ndarray:
    """Extract 7-dim robot state from LIBERO obs dict."""
    if isinstance(obs, dict):
        parts = []
        for k in ("ee_pos", "ee_states"):
            if k in obs:
                parts.append(np.asarray(obs[k], dtype=np.float32).ravel())
                break
        for k in ("gripper_states",):
            if k in obs:
                gv = np.asarray(obs[k], dtype=np.float32).ravel()
                parts.append(gv[:1])
                break
        if parts:
            state = np.concatenate(parts)
            if len(state) < 7:
                state = np.pad(state, (0, 7 - len(state)))
            return state[:7]
    return np.zeros(7, dtype=np.float32)


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_libero_env(task_idx: int = 0):
    """Create a LIBERO-Spatial OffScreenRenderEnv for the given task index."""
    from libero.libero.envs import OffScreenRenderEnv
    bddl_dir = Path("/home/ubuntu/LIBERO/libero/libero/bddl_files/libero_spatial")
    bddl_files = sorted(bddl_dir.glob("*.bddl"))
    if not bddl_files:
        raise FileNotFoundError(f"No BDDL files in {bddl_dir}")
    bddl = str(bddl_files[min(task_idx, len(bddl_files) - 1)])
    return OffScreenRenderEnv(**{
        "bddl_file_name": bddl,
        "camera_heights": 128,
        "camera_widths":  128,
    }), Path(bddl).stem


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env,
    policy,
    memory_bank: RAIDMemoryBank,
    encoder: GR1Encoder,
    norm_stats: dict,
    language: str = "robot manipulation",
    device: str | torch.device = "cuda",
    deterministic: bool = False,
    k: int = 3,
    max_steps: int = 30,
    capture_frames: bool = False,
) -> tuple[list, float, bool] | tuple[list, float]:
    """
    Run one episode using LIBERO sim.

    max_steps=30 by default to limit osmesa rendering overhead (~337ms/step).
    Increase if using EGL/GPU rendering.

    Returns:
        trajectory: list of (feat_t, feat_next_pred, retrieved, action_taken)
        total_reward: float
        success: bool
    """
    dev = torch.device(device) if isinstance(device, str) else device

    action_mean = torch.as_tensor(norm_stats["action_mean"], dtype=torch.float32, device=dev)
    action_std  = torch.as_tensor(norm_stats["action_std"],  dtype=torch.float32, device=dev)

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    trajectory = []
    frames = []
    total_reward = 0.0
    success = False

    frame_t     = _get_frame(obs)
    robot_state = _get_robot_state(obs)

    env_horizon = getattr(env, "horizon", 500)
    n_steps = min(max_steps, env_horizon)

    for _step in range(n_steps):
        with torch.no_grad():
            img_t  = torch.from_numpy(frame_t).unsqueeze(0)          # (1, H, W, 3)
            feat_t = encoder.encode_frames(img_t)                     # (1, 384)
            rs_t   = torch.from_numpy(robot_state).unsqueeze(0)      # (1, 7)
            feat_next_pred = encoder.predict_next_feat(
                img_t, language=[language], robot_state=rs_t
            )   # (1, 384)

            retrieved, valid_mask = memory_bank.retrieve_batch(feat_t, feat_next_pred, k=k)
            kv_pad = ~valid_mask  # (1, k)

        if deterministic:
            with torch.no_grad():
                pred, _ = policy(feat_t, feat_next_pred, retrieved, kv_key_padding_mask=kv_pad)
                action_norm = pred.squeeze(0)
        else:
            with torch.no_grad():
                pred, _ = policy(feat_t, feat_next_pred, retrieved, kv_key_padding_mask=kv_pad)
                std = policy.log_std.exp().clamp(1e-4, 1.0)
                dist = torch.distributions.Normal(pred.squeeze(0), std)
                action_norm = dist.sample()

        action    = action_norm * action_std + action_mean
        action_np = action.detach().cpu().numpy().clip(-1.0, 1.0)

        result = env.step(action_np)
        obs, env_reward, done, info = result[:4]
        if capture_frames:
            frames.append(_get_frame(obs))

        # Use dense shaped reward + env reward + success bonus at episode end
        shaped = _shaped_reward(obs)
        reward = shaped + float(env_reward)
        total_reward += reward

        if hasattr(env, "check_success"):
            try:
                success_now = bool(env.check_success())
            except Exception:
                success_now = bool(info.get("success", False))
        else:
            success_now = bool(info.get("success", done and reward > 0))
        if success_now:
            success = True

        trajectory.append((
            feat_t.squeeze(0).cpu(),
            feat_next_pred.squeeze(0).cpu(),
            retrieved.squeeze(0).cpu(),
            action_norm.detach().cpu(),
        ))

        frame_t     = _get_frame(obs)
        robot_state = _get_robot_state(obs)

        if done or success:
            break

    if capture_frames:
        return frames, total_reward
    return trajectory, total_reward, success
