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
from torch.utils.data import Dataset


@dataclass
class _Member:
    name: str
    ds: Dataset
    dataset_id: int
    length: int


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
