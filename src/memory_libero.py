"""
RAID Memory Bank for V-JEPA 2 features.

Stores (feat_t, feat_next, action) transitions with cosine-similarity
retrieval over the concatenated query key (feat_t || feat_next, 2048-dim).

Used by retrieval-augmented conditions (concat_mlp, raid_xattn) and
non-parametric baselines (nn_copy).
"""
from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


class VJEPAMemoryBank:
    def __init__(self, feat_dim: int = 1024, max_size: int = 200000,
                  device: str | torch.device = "cpu"):
        self.feat_dim = int(feat_dim)
        self.max_size = int(max_size)
        self.device = torch.device(device)

        kdim = feat_dim * 2  # concat(feat_t, feat_next) = 2048
        self.keys = torch.zeros(max_size, kdim, device=self.device,
                                 dtype=torch.float32)
        self.actions = torch.zeros(max_size, 7, device=self.device,
                                    dtype=torch.float32)
        self._ptr = 0

    def clear(self) -> None:
        self._ptr = 0

    @property
    def ptr(self) -> int:
        return self._ptr

    def add(self, feat_t: torch.Tensor, feat_next: torch.Tensor,
            action: torch.Tensor) -> None:
        if self._ptr >= self.max_size:
            raise RuntimeError(f"VJEPAMemoryBank overflow "
                                f"(max_size={self.max_size})")
        st = feat_t.detach().reshape(-1).to(self.device, dtype=torch.float32)
        sn = feat_next.detach().reshape(-1).to(self.device, dtype=torch.float32)
        aa = action.detach().reshape(-1).to(self.device, dtype=torch.float32)

        key = torch.cat([st, sn], dim=-1)
        key = F.normalize(key, dim=-1, eps=1e-8)

        i = self._ptr
        self.keys[i] = key
        self.actions[i] = aa
        self._ptr += 1

    def add_batch(self, feat_t: torch.Tensor, feat_next: torch.Tensor,
                  actions: torch.Tensor) -> None:
        for i in range(len(feat_t)):
            self.add(feat_t[i], feat_next[i], actions[i])

    def build_from_dataset(self, dataset: Any) -> None:
        self.clear()
        for i in range(len(dataset)):
            ft, fn, act = dataset[i]
            self.add(ft, fn, act)
        print(f"[memory] VJEPAMemoryBank populated: "
              f"ptr={self.ptr}/{self.max_size}")

    def retrieve(
        self,
        query_feat_t: torch.Tensor,
        query_feat_next: torch.Tensor,
        k: int = 5,
        exclude_idx: int | None = None,
    ):
        """
        Retrieve top-k actions by cosine similarity.

        Args:
            query_feat_t:  (feat_dim,) or (1, feat_dim)
            query_feat_next:(feat_dim,) or (1, feat_dim)
            k: number of neighbours
            exclude_idx: if provided, mask out this bank index

        Returns:
            retrieved_actions: (k, 7)
            similarities:      (k,)
        """
        n = self.ptr
        if n == 0:
            return (torch.zeros(k, 7, device=self.device, dtype=torch.float32),
                    torch.zeros(k, device=self.device, dtype=torch.float32))

        qt = query_feat_t.detach().reshape(-1).to(self.device, dtype=torch.float32)
        qn = query_feat_next.detach().reshape(-1).to(self.device, dtype=torch.float32)
        query = F.normalize(torch.cat([qt, qn], dim=-1), dim=-1, eps=1e-8)

        sims = (self.keys[:n] * query.unsqueeze(0)).sum(dim=-1)

        if exclude_idx is not None:
            r = int(exclude_idx)
            if 0 <= r < n:
                sims[r] = float("-inf")

        kk = min(k, n)
        top = torch.topk(sims, k=kk, largest=True)
        actions_ret = self.actions[top.indices]
        # Pad if fewer than k.
        if kk < k:
            pad_a = torch.zeros(k - kk, 7, device=self.device, dtype=torch.float32)
            pad_s = torch.full((k - kk,), float("-inf"), device=self.device,
                                dtype=torch.float32)
            actions_ret = torch.cat([actions_ret, pad_a], dim=0)
            sims_ret    = torch.cat([top.values, pad_s], dim=0)
        else:
            sims_ret = top.values
        return actions_ret, sims_ret

    def retrieve_batch(
        self,
        feat_t: torch.Tensor,
        feat_next: torch.Tensor,
        k: int = 5,
        exclude_indices: torch.Tensor | None = None,
    ):
        """
        Batched retrieval.

        Args:
            feat_t:       (B, feat_dim)
            feat_next:    (B, feat_dim)
            k:            number of neighbours
            exclude_indices: (B,) bank indices to exclude per query

        Returns:
            retrieved_actions: (B, k, 7)
            similarities:      (B, k)
            valid_mask:        (B, k) bool — True = valid neighbour
        """
        n = self.ptr
        B = feat_t.shape[0]
        out_dev = feat_t.device

        if n == 0:
            return (torch.zeros(B, k, 7, device=out_dev, dtype=torch.float32),
                    torch.zeros(B, k, device=out_dev, dtype=torch.float32),
                    torch.zeros(B, k, dtype=torch.bool, device=out_dev))

        qc = torch.cat([feat_t, feat_next], dim=-1).float().to(self.device)
        qk = F.normalize(qc, dim=-1, eps=1e-8)
        scores = qk @ self.keys[:n].T  # (B, n)

        if exclude_indices is not None:
            ee = exclude_indices.long().reshape(-1).to(self.device)
            rr = torch.arange(B, device=self.device)
            scores[rr, ee] = float("-inf")

        kk = min(k, n)
        top = scores.topk(kk, dim=-1)

        gathered = self.actions[top.indices]  # (B, kk, 7)
        valid = torch.isfinite(top.values) & top.values.gt(float("-inf"))

        # Pad to full k.
        ret_actions = torch.zeros(B, k, 7, device=self.device, dtype=torch.float32)
        ret_sims    = torch.zeros(B, k, device=self.device, dtype=torch.float32)
        ret_valid   = torch.zeros(B, k, dtype=torch.bool, device=self.device)

        ret_actions[:, :kk] = gathered
        ret_sims[:, :kk] = top.values
        ret_valid[:, :kk] = valid

        return (ret_actions.to(out_dev),
                ret_sims.to(out_dev),
                ret_valid.to(out_dev))


if __name__ == "__main__":
    B, D = 8, 1024
    bank = VJEPAMemoryBank(feat_dim=D, max_size=1000, device="cpu")

    # Fill with random data.
    ft  = torch.randn(200, D)
    fn  = torch.randn(200, D)
    act = torch.randn(200, 7)
    for i in range(200):
        bank.add(ft[i], fn[i], act[i])

    print(f"ptr = {bank.ptr}")

    # Single query.
    ra, sim = bank.retrieve(ft[0], fn[0], k=5)
    print(f"Single retrieve: actions={ra.shape} sims={sim.shape}")

    # Batch query with self-exclusion.
    BQ = 4
    ret_a, ret_s, ret_v = bank.retrieve_batch(
        ft[:BQ], fn[:BQ], k=5,
        exclude_indices=torch.arange(BQ, dtype=torch.long))
    print(f"Batch retrieve: actions={ret_a.shape} sims={ret_s.shape} "
          f"valid={ret_v.sum(dim=1)}")
    print("[memory_libero] OK")
