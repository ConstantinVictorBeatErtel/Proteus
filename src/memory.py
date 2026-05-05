"""RAID Memory Bank — cosine retrieval over stored normalized transitions."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


class RAIDMemoryBank:
    """Store transitions with retrieval key ``F.normalize(concat(s_t, s_next))``."""

    __slots__ = ("obs_dim", "action_dim", "max_entries", "s_t", "s_next", "actions", "keys", "_ptr")

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 7,
        max_entries: int = 50_000,
        device: torch.device | str | None = None,
    ):
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.max_entries = int(max_entries)
        dev = torch.device(device) if device is not None else torch.device("cpu")

        self.s_t = torch.zeros(self.max_entries, self.obs_dim, device=dev, dtype=torch.float32)
        self.s_next = torch.zeros(self.max_entries, self.obs_dim, device=dev, dtype=torch.float32)
        self.actions = torch.zeros(self.max_entries, self.action_dim, device=dev, dtype=torch.float32)
        self.keys = torch.zeros(self.max_entries, self.obs_dim * 2, device=dev, dtype=torch.float32)
        self._ptr = 0

    def clear(self) -> None:
        self._ptr = 0

    @property
    def ptr(self) -> int:
        return self._ptr

    @property
    def device(self) -> torch.device:
        return self.keys.device

    def add(self, s_t: torch.Tensor, s_next: torch.Tensor, action: torch.Tensor) -> None:
        if self._ptr >= self.max_entries:
            raise RuntimeError(f"RAIDMemoryBank overflow (max_entries={self.max_entries})")

        st = s_t.detach().reshape(-1).to(self.device, dtype=torch.float32)
        sn = s_next.detach().reshape(-1).to(self.device, dtype=torch.float32)
        aa = action.detach().reshape(-1).to(self.device, dtype=torch.float32)

        nk = F.normalize(torch.cat([st, sn], dim=-1), dim=-1, eps=1e-8)

        i = self._ptr
        self.s_t[i], self.s_next[i], self.actions[i] = st, sn, aa
        self.keys[i] = nk
        self._ptr += 1

    def populate_from_dataset(self, ds: Any, desc: str | None = None) -> None:
        self.clear()
        try:
            from tqdm import tqdm

            itr = tqdm(range(len(ds)), desc=desc or "Populate memory bank")
        except Exception:
            itr = range(len(ds))

        for i in itr:
            ex = ds[i]
            self.add(ex["s_t"], ex["s_next"], ex["action"])

        print(f"[memory] RAIDMemoryBank populated: ptr={self.ptr} / max={self.max_entries}")

    @staticmethod
    def _query_key(s_t: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([s_t.reshape(-1), s_next.reshape(-1)], dim=-1).float()
        return F.normalize(cat, dim=-1, eps=1e-8)

    def retrieve(
        self,
        s_t: torch.Tensor,
        s_next: torch.Tensor,
        k: int = 3,
        tau_min: float | None = None,
        exclude_bank_row: int | torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        n = self.ptr
        if n == 0:
            return None

        q = self._query_key(s_t, s_next).to(self.device, dtype=torch.float32)
        sims = (self.keys[:n] * q.unsqueeze(0)).sum(dim=-1)

        if exclude_bank_row is not None:
            r = int(exclude_bank_row.item()) if isinstance(exclude_bank_row, torch.Tensor) else int(exclude_bank_row)
            if 0 <= r < n:
                m = torch.ones_like(sims, dtype=torch.bool)
                m[r] = False
                sims = sims.masked_fill(~m, float("-inf"))

        if tau_min is not None:
            sims = sims.masked_fill(~(sims > tau_min), float("-inf"))

        usable = torch.isfinite(sims) & sims.gt(float("-inf"))
        if not usable.any():
            return None

        kk = min(k, int(usable.sum().item()))
        idx = torch.topk(sims, k=kk, largest=True).indices
        return self.actions[idx]

    def retrieve_batch(
        self,
        s_t_b: torch.Tensor,
        s_next_b: torch.Tensor,
        k: int = 3,
        tau_min: float | None = None,
        exclude_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return padded actions `(B,K,A)` plus `(B,K)` validity mask."""

        dv = self.device
        out_dev = s_t_b.device
        B = int(s_t_b.shape[0])
        kk_req = max(1, int(k))
        n = self.ptr

        if n == 0:
            return (
                torch.zeros(B, kk_req, self.action_dim, device=out_dev, dtype=torch.float32),
                torch.zeros(B, kk_req, dtype=torch.bool, device=out_dev),
            )

        qc = torch.cat([s_t_b, s_next_b], dim=-1).float().to(dv)
        qk = F.normalize(qc, dim=-1, eps=1e-8)
        scores = qk @ self.keys[:n].transpose(0, 1)

        if exclude_idx is not None:
            ee = exclude_idx.long().reshape(-1)
            rr = torch.arange(B, device=scores.device)
            scores[rr, ee] = float("-inf")

        if tau_min is not None:
            scores = scores.masked_fill(scores <= tau_min, float("-inf"))

        kk_eff = min(kk_req, n)
        topv, topi = scores.topk(kk_eff, dim=-1)

        gathered = self.actions[topi]
        pad_a = torch.zeros(B, kk_req, self.action_dim, device=dv, dtype=torch.float32)
        pad_m = torch.zeros(B, kk_req, dtype=torch.bool, device=dv)
        pad_a[:, :kk_eff] = gathered
        pad_m[:, :kk_eff] = torch.isfinite(topv) & topv.gt(float("-inf"))

        return pad_a.to(out_dev), pad_m.to(out_dev)

    def retrieve_single(self, s_t: torch.Tensor, k: int = 3) -> torch.Tensor:
        """Retrieval keyed on ``s_t`` alone (no transition pair) for rollout / inference."""
        n = self.ptr
        kk_req = max(1, int(k))
        out_dev = s_t.device
        if n == 0:
            return torch.zeros(kk_req, self.action_dim, device=out_dev, dtype=torch.float32)

        dv = self.device
        st = s_t.detach().reshape(-1).to(dv, dtype=torch.float32)
        q = F.normalize(st.unsqueeze(0), dim=-1, eps=1e-8)
        s_t_keys = F.normalize(self.s_t[:n], dim=-1, eps=1e-8)
        sims = (q @ s_t_keys.T).squeeze(0)

        kk_eff = min(kk_req, n)
        topi = sims.topk(kk_eff, largest=True).indices
        gathered = self.actions[topi].to(out_dev, dtype=torch.float32)

        if kk_eff < kk_req:
            pad = torch.zeros(kk_req - kk_eff, self.action_dim, device=out_dev, dtype=torch.float32)
            gathered = torch.cat([gathered, pad], dim=0)
        return gathered

    def hit_rate(self, ds: Any, k: int = 3, tau_min: float | None = None) -> float:
        """Share of samples with at least one valid retrieved neighbour."""

        if len(ds) == 0:
            return float("nan")

        good = 0
        total = len(ds)
        for i in range(total):
            ex = ds[i]
            ret = self.retrieve(ex["s_t"], ex["s_next"], k=k, tau_min=tau_min)
            good += ret is not None and ret.shape[0] > 0
        return float(good) / float(total)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, "/home/ubuntu/proteus/raid/src")
    from data import make_train_val

    print("[memory.__main__] Building train subset …")
    tr, _, _ = make_train_val(200)
    mem = RAIDMemoryBank(obs_dim=tr.state_dim, action_dim=tr.action_dim, device="cpu")
    mem.populate_from_dataset(tr)
    ex = tr[0]
    rr = mem.retrieve(ex["s_t"], ex["s_next"], k=3)
    print("[memory.__main__] retrieve demo0 shape=", None if rr is None else tuple(rr.shape))
    print("[memory.__main__] hit_rate=", mem.hit_rate(tr, k=3, tau_min=None))
