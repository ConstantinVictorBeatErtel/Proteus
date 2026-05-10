import os, sys, argparse, json
os.environ.setdefault("MUJOCO_GL", "egl")

_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _repo)
sys.path.insert(0, os.path.join(_repo, "src"))

import numpy as np
import torch
import imageio
from pathlib import Path
from PIL import Image, ImageDraw

from rollout_libero import make_libero_env, run_episode
from models import DirectMLPVisual, RAIDDecoderVisual
from gr1_encoder import GR1Encoder
from train_libero import populate_memory_from_cache
from data_libero import find_hdf5_files, load_norm_stats
import h5py


def label_frame(frame, text):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 160, 22], fill=(0, 0, 0))
    draw.text((4, 4), text, fill=(255, 255, 255))
    return np.array(img)


def load_policy(condition, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    if condition == "raid_visual":
        model = RAIDDecoderVisual(feat_dim=384).to(device)
    else:
        model = DirectMLPVisual(feat_dim=384).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def run_condition(condition, ckpt_path, env, language, memory_bank, encoder,
                  norm_stats, n_episodes, device, max_steps=200):
    policy = load_policy(condition, ckpt_path, device)
    all_frames, all_rewards = [], []
    for ep in range(n_episodes):
        frames, total_reward = run_episode(
            env, policy, memory_bank, encoder, norm_stats,
            language=language, device=device,
            capture_frames=True, max_steps=max_steps,
        )
        frames = [label_frame(f, condition) for f in frames]
        all_frames.append(frames)
        all_rewards.append(total_reward)
        print(f"  {condition} ep{ep}: {len(frames)} frames, reward={total_reward:.3f}")
    print(f"  {condition} mean reward: {np.mean(all_rewards):.3f}")
    return all_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_idx",   type=int, default=1)
    parser.add_argument("--n_demos",    type=int, default=200)
    parser.add_argument("--n_episodes", type=int, default=3)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--max_steps",  type=int, default=200)
    parser.add_argument("--feature_dir",  default="data/libero_spatial/features")
    parser.add_argument("--dataset_dir",  default="data/libero_spatial/libero_spatial/libero_spatial")
    parser.add_argument("--gr1_ckpt",    default="checkpoints/gr1/snapshot_ABCD.pt")
    parser.add_argument("--mae_ckpt",    default="checkpoints/gr1/mae_pretrain_vit_base.pth")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load norm stats
    norm_path = Path(args.dataset_dir) / "norm_stats.pt"
    norm_stats = load_norm_stats(norm_path)
    print(f"Loaded norm stats from {norm_path}")

    # Load GR-1 encoder
    print("Loading GR-1 encoder ...")
    encoder = GR1Encoder.from_checkpoints(args.mae_ckpt, args.gr1_ckpt, device)
    feat_dim = encoder.feat_dim
    print(f"  feat_dim={feat_dim}")

    # Populate memory bank
    print("Populating memory bank ...")
    memory_bank = populate_memory_from_cache(
        Path(args.feature_dir), args.n_demos, device, feat_dim=feat_dim)
    print(f"  Memory bank: {memory_bank.ptr} entries")

    # Get language instruction from HDF5
    hdf5_files = find_hdf5_files(args.dataset_dir)
    task_file = hdf5_files[min(args.task_idx, len(hdf5_files)-1)]
    with h5py.File(task_file, "r") as f:
        problem_info = json.loads(f["data"].attrs.get("problem_info", "{}"))
        language = problem_info.get("language_instruction", "robot manipulation").strip('"\'')
    print(f"Task: {language}")

    # Make LIBERO env
    print("Creating LIBERO env ...")
    env, task_name = make_libero_env(args.task_idx)
    print(f"  Task: {task_name}")

    conditions = {
        "raid_visual":   f"models/raid_visual_{args.n_demos}demos_libero_best.pt",
        "direct_visual": f"models/direct_visual_{args.n_demos}demos_libero_best.pt",
    }

    all_episode_frames = {}
    for condition, ckpt in conditions.items():
        print(f"\nRunning {condition}...")
        all_episode_frames[condition] = run_condition(
            condition, ckpt, env, language, memory_bank, encoder,
            norm_stats, args.n_episodes, device, max_steps=args.max_steps)

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
