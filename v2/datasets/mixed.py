"""Concatenated multi-dataset wrapper with per-dataset normalization.

Each child dataset already produces normalized actions in ``[-1, 1]``
under its own q01/q99 stats. The ``MixedIDMDataset`` simply concatenates
their indices and rewrites ``dataset_id`` so a downstream head can
condition on it (e.g. via a learned 8-D embedding).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


@dataclass
class _Member:
    name: str
    ds: Dataset
    dataset_id: int
    length: int


def _infer_obs_dim(ds: Dataset) -> int:
    for attr in ("obs_dim", "state_dim", "feature_dim"):
        value = getattr(ds, attr, None)
        if value is not None:
            return int(value)
    if len(ds) == 0:  # type: ignore[arg-type]
        raise ValueError("cannot infer obs dim from an empty dataset")
    ex = ds[0]  # type: ignore[index]
    for key in ("obs_t", "s_t"):
        if key in ex:
            return int(ex[key].shape[-1])
    raise ValueError(f"cannot infer obs dim for dataset type {type(ds).__name__}")


class PaddedObservationDataset(Dataset):
    """Right-pad observation vectors so mixed-task low-dim cells can run."""

    def __init__(self, base: Dataset, target_obs_dim: int) -> None:
        self.base = base
        self.source_obs_dim = _infer_obs_dim(base)
        self.target_obs_dim = int(target_obs_dim)
        self.obs_dim = self.target_obs_dim
        self.action_dim = int(getattr(base, "action_dim", 7))
        self._pad_right = self.target_obs_dim - self.source_obs_dim
        if self._pad_right < 0:
            raise ValueError(
                f"target_obs_dim={self.target_obs_dim} is smaller than source_obs_dim={self.source_obs_dim}"
            )
        self.obs_mask = torch.cat(
            [
                torch.ones(self.source_obs_dim, dtype=torch.bool),
                torch.zeros(self._pad_right, dtype=torch.bool),
            ]
        )

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def _pad_obs(self, x: torch.Tensor) -> torch.Tensor:
        if self._pad_right == 0:
            return x
        return F.pad(x, (0, self._pad_right))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = dict(self.base[int(idx)])  # type: ignore[index]
        for key in ("obs_t", "obs_next", "s_t", "s_next"):
            if key in ex:
                ex[key] = self._pad_obs(ex[key])
        if "obs_window" in ex:
            ex["obs_window"] = F.pad(ex["obs_window"], (0, self._pad_right))
        ex["obs_mask"] = self.obs_mask.clone()
        return ex

    def _stack_with_padding(self, fn_name: str, key: str) -> torch.Tensor:
        getter = getattr(self.base, fn_name, None)
        if getter is None:
            rows = [self[i][key] for i in range(len(self))]  # type: ignore[index]
            return torch.stack(rows, dim=0)
        return self._pad_obs(getter(key))

    def stacked_obs_t(self, key: str = "obs_t") -> torch.Tensor:
        return self._stack_with_padding("stacked_obs_t", key)

    def stacked_obs_next(self, key: str = "obs_next") -> torch.Tensor:
        return self._stack_with_padding("stacked_obs_next", key)

    def stacked_actions(self, key: str = "action") -> torch.Tensor:
        getter = getattr(self.base, "stacked_actions", None)
        if getter is None:
            rows = [self.base[i][key] for i in range(len(self.base))]  # type: ignore[index]
            return torch.stack(rows, dim=0)
        return getter(key)

    def stacked_obs_window(self, key: str = "obs_window") -> torch.Tensor:
        getter = getattr(self.base, "stacked_obs_window", None)
        if getter is None:
            rows = [self[i][key] for i in range(len(self))]  # type: ignore[index]
            return torch.stack(rows, dim=0)
        return F.pad(getter(key), (0, self._pad_right))

    def fetch_frames(self, idx: int) -> tuple[Any, Any]:
        fetcher = getattr(self.base, "fetch_frames")
        return fetcher(int(idx))


class MixedIDMDataset(Dataset):
    """Round-robin concatenation of action-normalized child datasets."""

    def __init__(self, members: Sequence[tuple[str, Dataset]]) -> None:
        self._members: list[_Member] = []
        offset = 0
        self._ranges: list[tuple[int, int, int]] = []  # (lo, hi, dataset_id)
        for did, (name, ds) in enumerate(members):
            length = len(ds)  # type: ignore[arg-type]
            self._members.append(_Member(name=name, ds=ds, dataset_id=did, length=length))
            self._ranges.append((offset, offset + length, did))
            offset += length
        self._total = offset

    def __len__(self) -> int:
        return self._total

    def member_for(self, global_idx: int) -> tuple[_Member, int]:
        for lo, hi, did in self._ranges:
            if lo <= global_idx < hi:
                return self._members[did], global_idx - lo
        raise IndexError(global_idx)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        gidx = int(idx)
        member, local = self.member_for(gidx)
        ex = member.ds[local]  # type: ignore[index]
        ex = dict(ex)
        # Override ``idx`` with the global mixed-dataset index. The
        # FeatureMemoryBank's ``populate_vectorized`` stacks members
        # contiguously, so a child's local index is *not* the bank
        # row — the global idx is. Without this override, RAID's
        # ``exclude_idx`` during training would mask the wrong bank row.
        ex["idx"] = torch.tensor(gidx, dtype=torch.long)
        ex["dataset_id"] = torch.tensor(member.dataset_id, dtype=torch.long)
        ex["dataset_name"] = member.name
        return ex

    @property
    def member_names(self) -> list[str]:
        return [m.name for m in self._members]

    # ---- bulk accessors for FeatureMemoryBank.populate_vectorized ----

    def _concat_member_stack(self, fn_name: str, key: str) -> torch.Tensor:
        parts = []
        for m in self._members:
            getter = getattr(m.ds, fn_name, None)
            if getter is None:
                # Fallback: fetch each row via __getitem__ (slow, but
                # only triggered if a member adapter doesn't pre-stack).
                rows = [m.ds[i][key] for i in range(m.length)]  # type: ignore[index]
                parts.append(torch.stack(rows, dim=0))
            else:
                parts.append(getter(key))
        return torch.cat(parts, dim=0)

    def stacked_obs_t(self, key: str = "obs_t") -> torch.Tensor:
        return self._concat_member_stack("stacked_obs_t", key)

    def stacked_obs_next(self, key: str = "obs_next") -> torch.Tensor:
        return self._concat_member_stack("stacked_obs_next", key)

    def stacked_actions(self, key: str = "action") -> torch.Tensor:
        return self._concat_member_stack("stacked_actions", key)

    def stacked_obs_window(self, key: str = "obs_window") -> torch.Tensor:
        return self._concat_member_stack("stacked_obs_window", key)
