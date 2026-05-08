import os, sys, argparse
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import imageio
from PIL import Image, ImageDraw, ImageFont
from src.rollout_libero import run_episode
from src.models import DirectMLPVisual, RAIDDecoderVisual

def label_frame(frame, text):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 160, 22], fill=(0, 0, 0))
    draw.text((4, 4), text, fill=(255, 255, 255))
    return np.array(img)

def load_policy(condition, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    if condition == "raid_visual":
        model = RAIDDecoderVisual(feat_dim=384).to(device)
    else:
        model = DirectMLPVisual(feat_dim=384).to(device)
    model.load_state_dict(state)
    model.eval()
    return model

def run_condition(condition, ckpt_path, task_idx, n_episodes, device):
    policy = load_policy(condition, ckpt_path, device)
    all_frames, all_rewards = [], []
    for ep in range(n_episodes):
        frames, reward = run_episode(policy, task_idx=task_idx,
                                     capture_frames=True, device=device)
        frames = [label_frame(f, condition) for f in frames]
        all_frames.append(frames)
        all_rewards.append(reward)
        print(f"  {condition} ep{ep}: {len(frames)} frames, reward={reward:.3f}")
    print(f"  {condition} mean reward: {np.mean(all_rewards):.3f}")
    return all_frames

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_idx", type=int, default=1)
    parser.add_argument("--n_demos", type=int, default=200)
    parser.add_argument("--n_episodes", type=int, default=3)
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    conditions = {
        "raid_visual":   f"checkpoints/models/raid_visual_demos{args.n_demos}_libero_best.pt",
        "direct_visual": f"checkpoints/models/direct_visual_demos{args.n_demos}_libero_best.pt",
    }

    all_episode_frames = {}
    for condition, ckpt in conditions.items():
        print(f"\nRunning {condition}...")
        all_episode_frames[condition] = run_condition(
            condition, ckpt, args.task_idx, args.n_episodes, device)

    for condition, episodes in all_episode_frames.items():
        best = max(episodes, key=len)
        path = f"{args.output_dir}/{condition}_demos{args.n_demos}.mp4"
        imageio.mimsave(path, best, fps=10)
        print(f"Saved {path}")

    left  = max(all_episode_frames["raid_visual"],   key=len)
    right = max(all_episode_frames["direct_visual"], key=len)
    n = min(len(left), len(right))
    combined = [np.concatenate([left[i], right[i]], axis=1) for i in range(n)]
    out = f"{args.output_dir}/raid_vs_direct_demos{args.n_demos}.mp4"
    imageio.mimsave(out, combined, fps=10)
    print(f"\nSaved side-by-side: {out}")

if __name__ == "__main__":
    main()
