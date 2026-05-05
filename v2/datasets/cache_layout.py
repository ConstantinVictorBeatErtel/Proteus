"""Helpers for feature-cache paths and layout metadata verification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from ..runtime.drive import atomic_write_bytes, features_root


@dataclass(frozen=True)
class FeatureCacheMetadata:
    dataset_name: str
    encoder: str
    frame_count: int
    feature_dim: int
    episode_count: int
    layout_checksum: str


def feature_cache_path(dataset_name: str, encoder: str) -> Path:
    return features_root() / f"{dataset_name}_{encoder}_cls.safetensors"


def metadata_path_for(features_path: str | Path) -> Path:
    return Path(features_path).with_suffix(".metadata.json")


def layout_checksum(entries: Iterable[tuple[str, int]]) -> str:
    digest = hashlib.sha1()
    for demo_key, length in entries:
        digest.update(f"{demo_key}\t{int(length)}\n".encode("utf-8"))
    return digest.hexdigest()


def write_feature_cache_metadata(metadata: FeatureCacheMetadata, features_path: str | Path) -> Path:
    target = metadata_path_for(features_path)
    payload = json.dumps(asdict(metadata), indent=2, sort_keys=True).encode("utf-8")
    return atomic_write_bytes(payload, target)


def load_feature_cache_metadata(features_path: str | Path) -> FeatureCacheMetadata | None:
    target = metadata_path_for(features_path)
    if not target.is_file():
        return None
    raw = json.loads(target.read_text())
    return FeatureCacheMetadata(**raw)
