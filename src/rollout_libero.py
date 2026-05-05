"""
LIBERO episode rollouts for GRPO fine-tuning with GR-1 world model.

Key difference from rollout.py (RoboMimic version):
  - Uses LIBERO's BulletSim environment
  - GR-1 predicts next-frame features (predict_next_feat) → no (s_t, s_t) proxy hack
  - Memory bank keyed on GR-1 visual features, not low-dim state

Inference flow per step:
    frame_t  →  GR-1 encode  →  feat_t
    frame_t  →  GR-1 predict_next_feat  →  feat_next_pred
    (feat_t, feat_next_pred, retrieved_actions)  →  RAIDDecoderVisual  →  action_norm
    action_norm  →  denormalise  →  sim step
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
# Environment factory
# ---------------------------------------------------------------------------

def make_libero_env(task_name: str, bddl_file: str | None = None):
    """
    Create a LIBERO environment for the given task.

    task_name: e.g. 'pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate'
    bddl_file: optional explicit .bddl path; if None, resolved from libero benchmark
    """
    try:
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError:
        raise ImportError("Install libero: pip install libero")

    if bddl_file is None:
        bm = benchmark.get_benchmark_dict()["libero_spatial"]()
        # Find matching task
        for i in range(bm.n_tasks):
            task = bm.get_task(i)
            if task_name.lower().replace(" ", "_") in task.name.lower().replace(" ", "_"):
                bddl_file = bm.get_task_bddl_file_path(i)
                break
        if bddl_file is None:
            # Fall back to first task
            bddl_file = bm.get_task_bddl_file_path(0)

    env = OffScreenRenderEnv(**{
        "bddl_file_name": bddl_file,
        "camera_heights": 128,
        "camera_widths": 128,
    })
    return env


def _get_frame(obs) -> np.ndarray:
    """Extract (128, 128, 3) uint8 RGB from LIBERO obs dict."""
    if isinstance(obs, dict):
        for key in ("agentview_image", "agentview_rgb", "image"):
            if key in obs:
                img = np.asarray(obs[key], dtype=np.uint8)
                if img.ndim == 3:
                    return img
    raise KeyError(f"Cannot find image in obs keys: {list(obs.keys()) if isinstance(obs, dict) else type(obs)}")


def _get_robot_state(obs) -> np.ndarray:
    """Extract 7-dim robot state [ee_pos(3), ee_ori(3), gripper(1)] from LIBERO obs."""
    if isinstance(obs, dict):
        parts = []
        for k in ("ee_pos", "ee_ori", "ee_states"):
            if k in obs:
                parts.append(np.asarray(obs[k], dtype=np.float32).ravel())
                break
        for k in ("gripper_states",):
            if k in obs:
                gv = np.asarray(obs[k], dtype=np.float32).ravel()
                parts.append(gv[:1])  # just take first dim
                break
        if parts:
            state = np.concatenate(parts)
            if len(state) < 7:
                state = np.pad(state, (0, 7 - len(state)))
            return state[:7]
    return np.zeros(7, dtype=np.float32)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env,
    policy,                  # RAIDDecoderVisual
    memory_bank: RAIDMemoryBank,
    encoder: GR1Encoder,
    norm_stats: dict,
    language: str = "robot manipulation",
    device: str | torch.device = "cuda",
    deterministic: bool = False,
    k: int = 3,
    max_steps: int = 100,    # cap for GRPO speed; GR-1 predict_next ~500ms/step
) -> tuple[list, float, bool]:
    """
    Run one episode using LIBERO sim.

    Returns:
        trajectory: list of (feat_t, feat_next_gt, retrieved, action_taken)
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
    total_reward = 0.0
    success = False
    info = {}

    frame_t = _get_frame(obs)
    robot_state_t = _get_robot_state(obs)

    env_horizon = env.horizon if hasattr(env, "horizon") else 500
    for _step in range(min(max_steps, env_horizon)):
        # --- Encode current frame ---
        with torch.no_grad():
            img_tensor = torch.from_numpy(frame_t).unsqueeze(0)   # (1, H, W, 3)
            feat_t = encoder.encode_frames(img_tensor)             # (1, 384)

            # --- GR-1 predicts next-frame features (world model) ---
            rs_tensor = torch.from_numpy(robot_state_t).unsqueeze(0)  # (1, 7)
            feat_next_pred = encoder.predict_next_feat(
                img_tensor, language=[language], robot_state=rs_tensor
            )   # (1, 384)

            # --- Retrieve from memory bank ---
            retrieved, valid_mask = memory_bank.retrieve_batch(
                feat_t, feat_next_pred, k=k
            )   # (1, k, 7), (1, k)
            kv_pad = ~valid_mask  # (1, k)

        # --- Policy forward pass ---
        if deterministic:
            with torch.no_grad():
                pred, _ = policy(feat_t, feat_next_pred, retrieved,
                                 kv_key_padding_mask=kv_pad)
                action_norm = pred.squeeze(0)
        else:
            with torch.no_grad():
                pred, _ = policy(feat_t, feat_next_pred, retrieved,
                                 kv_key_padding_mask=kv_pad)
                std = policy.log_std.exp().clamp(1e-4, 1.0)
                dist = torch.distributions.Normal(pred.squeeze(0), std)
                action_norm = dist.sample()

        # --- Denormalise and step ---
        action = action_norm * action_std + action_mean
        action_np = action.detach().cpu().numpy().clip(-1.0, 1.0)

        result = env.step(action_np)
        if len(result) == 4:
            obs, reward, done, info = result
        else:
            obs, reward, done, trunc, info = result
            done = done or trunc

        total_reward += float(reward)

        # Check success via env.check_success() if available, else use done+reward
        if hasattr(env, "check_success"):
            try:
                success_now = bool(env.check_success())
            except Exception:
                success_now = bool(info.get("success", False))
        else:
            success_now = bool(info.get("success", done and reward > 0))
        if success_now:
            success = True

        # Store transition (feat_next_pred is GR-1's prediction, consistent with policy input)
        trajectory.append((
            feat_t.squeeze(0).cpu(),          # (384,)
            feat_next_pred.squeeze(0).cpu(),  # (384,) predicted — consistent with policy input
            retrieved.squeeze(0).cpu(),       # (k, 7)
            action_norm.detach().cpu(),       # (7,)
        ))

        frame_t = _get_frame(obs)
        robot_state_t = _get_robot_state(obs)

        if done or success:
            break

    return trajectory, total_reward, success
