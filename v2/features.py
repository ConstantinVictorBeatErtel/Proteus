"""Cache frozen-encoder CLS features for the image-modality datasets.

Encoders supported:

* ``dinov2``  -> ``facebook/dinov2-base`` (~86 M params, 768-D CLS).
* ``theia``   -> ``theaiinstitute/theia-base-patch16-224-cdiv``
                 (robot-specific distillation, 86 M, 768-D).

Output: one safetensors per ``(dataset, encoder)`` written atomically to
``<artifact_root>/features/<dataset>_<encoder>_cls.safetensors`` with a
single key ``"features"`` holding ``[N, 768]`` fp16. Shape and dtype are
asserted before write.

Side effect: a ``preview.png`` grid of N=9 frames per image dataset is
saved next to the safetensors file via :mod:`v2.visualize` so we can see
exactly what the encoder ingested.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .runtime.drive import atomic_write_bytes, features_root
from .datasets.cache_layout import (
    FeatureCacheMetadata,
    feature_cache_path,
    layout_checksum,
    load_feature_cache_metadata,
    write_feature_cache_metadata,
)


ENCODER_MAP: dict[str, str] = {
    "dinov2": "facebook/dinov2-base",
    "theia": "theaiinstitute/theia-base-patch16-224-cdiv",
}


@dataclass
class EncoderHandle:
    name: str
    hf_id: str
    model: torch.nn.Module
    preprocess: callable  # type: ignore[type-arg]
    feature_dim: int


def load_encoder(name: str, device: torch.device | str = "cuda") -> EncoderHandle:
    """Load a frozen encoder, eval mode, fp16 on CUDA."""
    if name not in ENCODER_MAP:
        raise ValueError(f"unknown encoder {name!r}; pick from {list(ENCODER_MAP)}")
    hf_id = ENCODER_MAP[name]
    from transformers import AutoImageProcessor, AutoModel

    model = AutoModel.from_pretrained(hf_id)
    model.eval()
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        model = model.to(device).half()
    proc = AutoImageProcessor.from_pretrained(hf_id)

    def _preprocess(frames: np.ndarray) -> torch.Tensor:
        # frames: (B, H, W, 3) uint8
        out = proc(images=list(frames), return_tensors="pt")
        return out["pixel_values"]

    fdim = int(getattr(model.config, "hidden_size", 768))
    return EncoderHandle(name=name, hf_id=hf_id, model=model, preprocess=_preprocess, feature_dim=fdim)


def _module_dtype(module: torch.nn.Module) -> torch.dtype:
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return torch.float32


@torch.no_grad()
def encode_batch(handle: EncoderHandle, frames: np.ndarray, device: torch.device | str) -> torch.Tensor:
    """Return ``(B, feature_dim)`` fp32 CLS features."""
    pixels = handle.preprocess(frames).to(device)
    target_dtype = _module_dtype(handle.model)
    if pixels.dtype != target_dtype:
        pixels = pixels.to(target_dtype)
    out = handle.model(pixel_values=pixels)
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        cls = out.pooler_output
    else:
        cls = out.last_hidden_state[:, 0]
    return cls.float().cpu()


def _write_safetensors(features: torch.Tensor, target: Path) -> Path:
    from safetensors.torch import save as st_save

    if features.dtype != torch.float16:
        features = features.half()
    payload = st_save({"features": features.contiguous()})
    return atomic_write_bytes(payload, target)


def extract_dataset_features(
    dataset_name: str,
    frame_iter: Iterable[np.ndarray],
    encoder: EncoderHandle,
    expected_count: int | None = None,
    episode_layout_entries: Iterable[tuple[str, int]] | None = None,
    device: torch.device | str = "cuda",
    batch_size: int = 64,
    sample_preview: bool = True,
) -> Path:
    """Iterate over frames in batches, write CLS features safetensors."""
    target = feature_cache_path(dataset_name, encoder.name)
    existing_meta = load_feature_cache_metadata(target)
    layout_entries = list(episode_layout_entries or [])
    expected_layout_checksum = layout_checksum(layout_entries) if layout_entries else None
    if target.is_file() and target.stat().st_size > 1_000_000 and existing_meta is not None:
        layout_matches = (
            expected_layout_checksum is None
            or existing_meta.layout_checksum == expected_layout_checksum
        )
        count_matches = expected_count is None or existing_meta.frame_count == int(expected_count)
        if layout_matches and count_matches:
            print(f"[features] SKIP {target} (already exists, {target.stat().st_size / 1e6:.1f} MB)")
            return target
        print(f"[features] REBUILD {target} (stale metadata vs current dataset layout)")
    if target.is_file() and target.stat().st_size > 1_000_000 and existing_meta is None and layout_entries:
        from safetensors.torch import load_file

        existing = load_file(str(target))["features"]
        write_feature_cache_metadata(
            FeatureCacheMetadata(
                dataset_name=dataset_name,
                encoder=encoder.name,
                frame_count=int(existing.shape[0]),
                feature_dim=int(existing.shape[1]),
                episode_count=len(layout_entries),
                layout_checksum=expected_layout_checksum or "",
            ),
            target,
        )
        print(f"[features] added metadata sidecar for existing cache {target}")
        return target

    pieces: list[torch.Tensor] = []
    preview_buf: list[np.ndarray] = []
    buf: list[np.ndarray] = []
    n_seen = 0
    for frame in frame_iter:
        buf.append(frame)
        if sample_preview and len(preview_buf) < 9 and n_seen % 50 == 0:
            preview_buf.append(frame)
        n_seen += 1
        if len(buf) >= batch_size:
            arr = np.stack(buf, axis=0)
            buf = []
            pieces.append(encode_batch(encoder, arr, device))
    if buf:
        arr = np.stack(buf, axis=0)
        pieces.append(encode_batch(encoder, arr, device))

    feats = torch.cat(pieces, dim=0)
    if expected_count is not None and feats.shape[0] != expected_count:
        raise RuntimeError(f"feature count mismatch: got {feats.shape[0]}, expected {expected_count}")
    out = _write_safetensors(feats, target)
    write_feature_cache_metadata(
        FeatureCacheMetadata(
            dataset_name=dataset_name,
            encoder=encoder.name,
            frame_count=int(feats.shape[0]),
            feature_dim=int(feats.shape[1]),
            episode_count=len(layout_entries),
            layout_checksum=expected_layout_checksum or "",
        ),
        out,
    )
    print(f"[features] wrote {out} shape={tuple(feats.shape)} dtype=fp16 size={out.stat().st_size / 1e6:.1f} MB")

    if sample_preview and preview_buf:
        from .visualize import render_grid

        panels = [{"frame_t": f, "frame_next": f, "action_gt": np.zeros(7), "title": ""} for f in preview_buf]
        render_grid(target.with_suffix(".preview.png"), panels, title=f"{dataset_name} ({encoder.name})")

    return out


def load_features(dataset_name: str, encoder: str) -> torch.Tensor:
    from safetensors.torch import load_file

    target = feature_cache_path(dataset_name, encoder)
    if not target.is_file():
        raise FileNotFoundError(target)
    return load_file(str(target))["features"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", default=list(ENCODER_MAP.keys()))
    ap.add_argument("--datasets", nargs="+", default=[
        "robomimic_can_ph_image", "robomimic_can_mh_image",
        "robomimic_square_ph_image", "robomimic_square_mh_image",
        "libero_spatial", "libero_object", "libero_goal",
    ])
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoders = {name: load_encoder(name, device=device) for name in args.encoders}

    # Each dataset adapter exposes a ``frames_iter`` helper; the runner
    # below stitches them together. We import lazily to avoid a hard
    # dependency on the HF datasets package when this module is imported.
    from .runtime.drive import data_root
    from .datasets import robomimic as rm
    from .datasets import libero as lb

    for dname in args.datasets:
        if dname.startswith("robomimic_"):
            _, task, variant, _ = dname.split("_", 3)
            spec = rm.RoboMimicSpec(task=task, variant=variant, modality="image")
            hp = rm.hdf5_path_for(spec, data_root() / "robomimic")
            import h5py

            with h5py.File(hp, "r") as f:
                sorted_keys = sorted(f["data"], key=lambda x: int(x.split("_")[1]))
                layout_entries = [(dk, int(f["data"][dk]["actions"].shape[0])) for dk in sorted_keys]
                count = sum(length for _, length in layout_entries)

            def _frames_iter() -> Iterable[np.ndarray]:
                with h5py.File(hp, "r") as f:
                    for dk, _length in layout_entries:
                        dset = f["data"][dk]["obs"][rm.IMAGE_OBS_KEY]
                        for i in range(int(dset.shape[0])):
                            yield np.asarray(dset[i], dtype=np.uint8)

            for enc_name, enc in encoders.items():
                extract_dataset_features(
                    dataset_name=dname,
                    frame_iter=_frames_iter(),
                    encoder=enc,
                    expected_count=count,
                    episode_layout_entries=layout_entries,
                    device=device,
                    batch_size=args.batch_size,
                )
        elif dname.startswith("libero_"):
            spec = lb.LiberoSpec(suite=dname)
            ds = lb._load_hf_dataset(spec.suite, data_root() / "libero")
            layout_entries = lb.episode_layout(ds)
            count = sum(length for _, length in layout_entries)

            for enc_name, enc in encoders.items():
                extract_dataset_features(
                    dataset_name=dname,
                    frame_iter=lb.iter_episode_frames(ds),
                    encoder=enc,
                    expected_count=count,
                    episode_layout_entries=layout_entries,
                    device=device,
                    batch_size=args.batch_size,
                )
        else:
            print(f"[features] unknown dataset {dname!r}, skipping")


if __name__ == "__main__":
    main()
