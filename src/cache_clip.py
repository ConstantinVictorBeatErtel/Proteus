#!/usr/bin/env python3
"""
Pre-compute frozen CLIP (ViT-B/32) image embeddings for every timestep in each task zarr.

Saves tensors to data/clip_cache/{task}_{train|val}.pt (full-timeline duplicate files).

Run once before training:  python src/cache_clip.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def register_zarr_codecs() -> None:
    import imagecodecs.numcodecs  # noqa: F401

    imagecodecs.numcodecs.register_codecs()


def main() -> None:
    register_zarr_codecs()

    torch.manual_seed(42)
    np.random.seed(42)

    import data as datamod

    REPO_LOCAL = Path(REPO_ROOT)
    CACHE_DIR = REPO_LOCAL / "data" / "clip_cache"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    from encoders import FrozenCLIPEncoder

    clip = FrozenCLIPEncoder(model_id="openai/clip-vit-base-patch32")
    bs = 32
    total_start = time.time()

    for task in datamod.TASKS:
        root = datamod._open_zarr_for_task(task)
        n_frames = datamod.task_num_timesteps(task)
        rgb_ds = root["data"]["camera0_rgb"]
        out = torch.zeros(n_frames, 512, dtype=torch.float32)

        t0 = time.time()
        n_batches = (n_frames + bs - 1) // bs
        for batch_i, start in enumerate(range(0, n_frames, bs)):
            stop = min(n_frames, start + bs)
            pil_list = []
            for i in range(start, stop):
                arr = np.asarray(rgb_ds[i], dtype=np.uint8)
                pil_list.append(Image.fromarray(arr))

            feats = clip.embed_pil_batch(pil_list)
            out[start:stop] = feats.cpu().float()

            if batch_i % max(1, n_batches // 25) == 0:
                print(
                    f"[cache_clip] {task}: encoded {stop}/{n_frames} "
                    f"({stop / max(time.time()-t0,1e-6):.1f} frames/s)"
                )

        elapsed = time.time() - t0
        payload = {
            "task": task,
            "num_timesteps": n_frames,
            "embeddings": out,
            "model_id": "openai/clip-vit-base-patch32",
            "embedding_dim": 512,
        }
        for split in ("train", "val"):
            path = CACHE_DIR / f"{task}_{split}.pt"
            torch.save(payload, path)
            print(f"[cache_clip] wrote {path}")
        print(f"[cache_clip] task {task} done in {elapsed:.1f}s")

    total = time.time() - total_start
    print(f"[cache_clip] ALL DONE in {total:.1f}s total")


if __name__ == "__main__":
    main()
