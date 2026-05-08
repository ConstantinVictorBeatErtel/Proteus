import argparse
import json
import os
import random
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

_repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "src"))

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from data_libero import _find_img_key, find_hdf5_files
from gr1_encoder import GR1Encoder, IMAGE_SIZE, RGB_MEAN, RGB_STD
from models import DirectMLPVisual, RAIDDecoderVisual
from train_libero import populate_memory_from_cache


@dataclass
class TransitionRef:
    feature_path: Path
    hdf5_path: Path
    task_name: str
    language: str
    demo_idx: int
    step_idx: int
    cache_idx: int


def resolve_repo_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else (_repo / path)


def load_policy(condition: str, ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = payload.get("model_state_dict", payload)
    feat_dim = int(payload.get("feat_dim", 384)) if isinstance(payload, dict) else 384
    if condition == "raid_visual":
        model = RAIDDecoderVisual(feat_dim=feat_dim).to(device)
    elif condition == "direct_visual":
        model = DirectMLPVisual(feat_dim=feat_dim).to(device)
    else:
        raise ValueError(f"Unknown condition: {condition}")
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def feature_files(feature_dir: Path) -> list[Path]:
    manifest = feature_dir / "manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text())
        files = [Path(p) for p in data["cached_files"]]
        files = [p if p.is_absolute() else (_repo / p) for p in files]
    else:
        files = sorted(feature_dir.glob("*_features.pt"))
    files = [p for p in files if p.exists()]
    if not files:
        raise FileNotFoundError(f"No cached feature files found in {feature_dir}")
    return files


def hdf5_by_stem(dataset_dir: Path) -> dict[str, Path]:
    files = [Path(p) for p in find_hdf5_files(dataset_dir)]
    if not files:
        raise FileNotFoundError(f"No HDF5 files found in {dataset_dir}")
    return {p.stem: p for p in files}


def language_for_file(hdf5_path: Path) -> str:
    with h5py.File(hdf5_path, "r") as f:
        info = json.loads(f["data"].attrs.get("problem_info", "{}"))
    return info.get("language_instruction", "robot manipulation").strip('"\'')


def build_transition_index(
    feature_dir: Path,
    dataset_dir: Path,
    n_demos: int,
) -> list[TransitionRef]:
    hdf5_lookup = hdf5_by_stem(dataset_dir)
    refs: list[TransitionRef] = []
    files = feature_files(feature_dir)
    demos_per_task = max(1, n_demos // len(files))

    for fp in files:
        cache = torch.load(fp, map_location="cpu", weights_only=False)
        task_name = str(cache.get("task_name", fp.name.removesuffix("_features.pt")))
        hdf5_path = hdf5_lookup.get(task_name)
        if hdf5_path is None:
            raise FileNotFoundError(f"Could not match cached task {task_name!r} to an HDF5 file")

        demo_lengths = [int(x) for x in cache["demo_lengths"]]
        language = language_for_file(hdf5_path)
        offset = 0
        for demo_idx, length in enumerate(demo_lengths[:demos_per_task]):
            for step_idx in range(length):
                refs.append(
                    TransitionRef(
                        feature_path=fp,
                        hdf5_path=hdf5_path,
                        task_name=task_name,
                        language=language,
                        demo_idx=demo_idx,
                        step_idx=step_idx,
                        cache_idx=offset + step_idx,
                    )
                )
            offset += length
    if not refs:
        raise RuntimeError("No transitions found after applying n_demos limit")
    return refs


def load_frame(ref: TransitionRef) -> np.ndarray:
    with h5py.File(ref.hdf5_path, "r") as f:
        obs = f[f"data/demo_{ref.demo_idx}/obs"]
        key = _find_img_key(obs)
        frame = obs[key][ref.step_idx]
    frame = np.asarray(frame)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    return frame.astype(np.uint8)


def patches_to_image(
    patch_preds: torch.Tensor,
    current_preprocessed: torch.Tensor,
) -> np.ndarray:
    """Unpatch GR-1 normalized patch predictions into a displayable RGB image."""
    patch_size = 16
    grid = IMAGE_SIZE // patch_size

    patches = patch_preds.detach().float().cpu()
    if patches.ndim == 3:
        patches = patches[0]
    patches = patches.reshape(grid, grid, patch_size, patch_size, 3)

    cur = current_preprocessed.detach().float().cpu()
    if cur.ndim == 4:
        cur = cur[0]
    cur_patches = cur.reshape(3, grid, patch_size, grid, patch_size)
    cur_patches = cur_patches.permute(1, 3, 2, 4, 0).reshape(grid, grid, patch_size * patch_size * 3)

    mean = cur_patches.mean(dim=-1, keepdim=True).numpy()
    std = cur_patches.var(dim=-1, unbiased=True, keepdim=True).sqrt().clamp_min(1e-6).numpy()
    patches = patches.numpy() * std.reshape(grid, grid, 1, 1, 1) + mean.reshape(grid, grid, 1, 1, 1)

    img = patches.transpose(0, 2, 1, 3, 4).reshape(IMAGE_SIZE, IMAGE_SIZE, 3)
    img = img * np.array(RGB_STD, dtype=np.float32) + np.array(RGB_MEAN, dtype=np.float32)

    if not np.isfinite(img).all() or img.max() - img.min() < 1e-6:
        img = np.nan_to_num(img, nan=0.5, posinf=1.0, neginf=0.0)
    lo, hi = np.percentile(img, [1, 99])
    if hi > lo and (img.min() < -0.2 or img.max() > 1.2):
        img = (img - lo) / (hi - lo)

    img = np.clip(img, 0.0, 1.0)
    img = (img * 255.0).round().astype(np.uint8)
    return np.array(Image.fromarray(img).resize((128, 128), Image.BICUBIC))


@torch.no_grad()
def predict_next_frame(encoder: GR1Encoder, frame: np.ndarray, language: str) -> np.ndarray:
    imgs = encoder._preprocess_images(frame[None])  # (1, 3, 224, 224)
    device = encoder.device
    rgb_seq = imgs.unsqueeze(1)
    state_data = {
        "arm": torch.zeros(1, 1, 6, device=device),
        "gripper": torch.tensor([[[1.0, 0.0]]], device=device),
    }
    attention_mask = torch.ones(1, 1, dtype=torch.long, device=device)

    import clip

    tokenized = clip.tokenize([language]).to(device)
    pred = encoder.gr1(
        rgb=rgb_seq,
        hand_rgb=torch.zeros_like(rgb_seq),
        state=state_data,
        language=tokenized,
        attention_mask=attention_mask,
    )
    obs_preds = pred["obs_preds"]
    if obs_preds is None:
        raise RuntimeError("GR-1 did not return obs_preds")
    return patches_to_image(obs_preds[0, 0], imgs[0])


def current_frame_128(frame: np.ndarray) -> np.ndarray:
    if frame.shape[:2] == (128, 128):
        return frame
    return np.array(Image.fromarray(frame).resize((128, 128), Image.BICUBIC))


def cache_batch(samples: list[TransitionRef]) -> dict[Path, dict]:
    out: dict[Path, dict] = {}
    for fp in sorted({s.feature_path for s in samples}):
        out[fp] = torch.load(fp, map_location="cpu", weights_only=False)
    return out


def actions_for_samples(
    samples: list[TransitionRef],
    cache: dict[Path, dict],
    raid_model: torch.nn.Module,
    direct_model: torch.nn.Module,
    memory_bank,
    device: torch.device,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feat_t = torch.stack([cache[s.feature_path]["feat_t"][s.cache_idx].float() for s in samples]).to(device)
    feat_next = torch.stack([cache[s.feature_path]["feat_next"][s.cache_idx].float() for s in samples]).to(device)
    gt = torch.stack([cache[s.feature_path]["actions"][s.cache_idx].float() for s in samples]).to(device)

    retrieved, valid_mask = memory_bank.retrieve_batch(feat_t, feat_next, k=k)
    raid_pred, _ = raid_model(feat_t, feat_next, retrieved, kv_key_padding_mask=~valid_mask)
    direct_pred, _ = direct_model(feat_t, feat_next)
    return (
        raid_pred.detach().cpu().numpy(),
        direct_pred.detach().cpu().numpy(),
        gt.detach().cpu().numpy(),
    )


def draw_action_bars(ax, values: np.ndarray, title: str, ylim: float) -> None:
    labels = ["x", "y", "z", "roll", "pitch", "yaw", "grip"]
    colors = ["#2f7fbd" if v >= 0 else "#d95f59" for v in values]
    ax.bar(np.arange(7), values, color=colors, width=0.72)
    ax.axhline(0, color="#2c2c2c", linewidth=0.8)
    ax.set_ylim(-ylim, ylim)
    ax.set_xticks(np.arange(7), labels, fontsize=6)
    ax.tick_params(axis="y", labelsize=7, length=2)
    ax.set_title(title, fontsize=9, pad=3)
    ax.grid(axis="y", alpha=0.2, linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#d0d0d0")


def render_grid(
    samples: list[TransitionRef],
    current_frames: list[np.ndarray],
    pred_frames: list[np.ndarray],
    raid_actions: np.ndarray,
    direct_actions: np.ndarray,
    gt_actions: np.ndarray,
    output_path: Path,
) -> None:
    rows = len(samples)
    fig = plt.figure(figsize=(15.5, max(2.55 * rows, 4.2)), dpi=180)
    gs = fig.add_gridspec(
        rows,
        6,
        width_ratios=[1.52, 1.0, 1.0, 1.24, 1.24, 1.24],
        wspace=0.36,
        hspace=0.58,
    )

    all_actions = np.concatenate([raid_actions, direct_actions, gt_actions], axis=0)
    ylim = float(max(1.0, np.nanmax(np.abs(all_actions)) * 1.15))
    headers = ["sample", "current", "GR-1 predicted next", "RAID action", "Direct action", "GT action"]

    for r, sample in enumerate(samples):
        for c in range(6):
            ax = fig.add_subplot(gs[r, c])
            if r == 0:
                ax.set_title(headers[c], fontsize=10, fontweight="bold", pad=8)
            if c == 0:
                ax.axis("off")
                wrapped = textwrap.fill(sample.language, width=34)
                ax.text(
                    0.0,
                    0.68,
                    wrapped,
                    ha="left",
                    va="top",
                    fontsize=8.5,
                    color="#262626",
                    linespacing=1.22,
                    transform=ax.transAxes,
                )
                ax.text(
                    0.0,
                    0.18,
                    f"demo {sample.demo_idx}\nstep {sample.step_idx}",
                    ha="left",
                    va="top",
                    fontsize=8.5,
                    color="#555555",
                    linespacing=1.28,
                    transform=ax.transAxes,
                )
            elif c == 1:
                ax.imshow(current_frames[r])
                ax.axis("off")
            elif c == 2:
                ax.imshow(pred_frames[r])
                ax.axis("off")
            elif c == 3:
                draw_action_bars(ax, raid_actions[r], "raid_visual", ylim)
            elif c == 4:
                draw_action_bars(ax, direct_actions[r], "direct_visual", ylim)
            else:
                draw_action_bars(ax, gt_actions[r], "ground truth", ylim)

    fig.suptitle("RAID transition samples from cached LIBERO features", fontsize=14, y=0.992)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_demos", type=int, default=200)
    parser.add_argument("--n_rows", type=int, default=4)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--dataset_dir", default="data/libero_spatial/libero_spatial/libero_spatial")
    parser.add_argument("--feature_dir", default="data/libero_spatial/features")
    parser.add_argument("--gr1_ckpt", default="checkpoints/gr1/snapshot_ABCD.pt")
    parser.add_argument("--mae_ckpt", default="checkpoints/gr1/mae_pretrain_vit_base.pth")
    parser.add_argument("--raid_ckpt", default=None)
    parser.add_argument("--direct_ckpt", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dataset_dir = resolve_repo_path(args.dataset_dir)
    feature_dir = resolve_repo_path(args.feature_dir)
    output_dir = resolve_repo_path(args.output_dir)
    raid_ckpt = resolve_repo_path(args.raid_ckpt or f"models/raid_visual_{args.n_demos}demos_libero_best.pt")
    direct_ckpt = resolve_repo_path(args.direct_ckpt or f"models/direct_visual_{args.n_demos}demos_libero_best.pt")

    print(f"Using device: {device}")
    print("Indexing cached transitions ...")
    refs = build_transition_index(feature_dir, dataset_dir, args.n_demos)
    samples = random.sample(refs, k=min(args.n_rows, len(refs)))
    print(f"Sampled {len(samples)} transitions from {len(refs)} available")

    print("Loading policies and memory bank ...")
    raid_model = load_policy("raid_visual", raid_ckpt, device)
    direct_model = load_policy("direct_visual", direct_ckpt, device)
    memory_bank = populate_memory_from_cache(feature_dir, args.n_demos, str(device), feat_dim=384)
    print(f"Memory bank entries: {memory_bank.ptr}")

    print("Loading GR-1 for pixel predictions ...")
    encoder = GR1Encoder.from_checkpoints(resolve_repo_path(args.mae_ckpt), resolve_repo_path(args.gr1_ckpt), device)

    cache = cache_batch(samples)
    raid_actions, direct_actions, gt_actions = actions_for_samples(
        samples, cache, raid_model, direct_model, memory_bank, device, args.k
    )

    current_frames = []
    pred_frames = []
    for i, sample in enumerate(samples, start=1):
        frame = load_frame(sample)
        current_frames.append(current_frame_128(frame))
        pred_frames.append(predict_next_frame(encoder, frame, sample.language))
        print(f"  [{i}/{len(samples)}] {sample.task_name} demo={sample.demo_idx} step={sample.step_idx}")

    out_path = output_dir / f"transition_grid_{args.n_demos}demos.png"
    render_grid(samples, current_frames, pred_frames, raid_actions, direct_actions, gt_actions, out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
