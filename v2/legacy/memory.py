"""Cosine-retrieval memory bank, generalized over arbitrary feature widths.

Forked from ``src/memory.py`` so the original stays bit-identical and the
autoresearch ``src/train.py`` baseline keeps reproducing.

Differences from the source:
  * ``obs_dim`` is no longer tied to the low-dim state width — pass any
    feature width (e.g. 768 for DINOv2 CLS).
  * ``key_fn`` selects how the retrieval key is built from
    ``(obs_t, obs_next)``: ``"concat"`` (default, matches the source) or
    ``"mean"`` (averaged then unit-normalized, useful when the feature
    width is large enough that the full concat is wasteful).
"""

from __future__ import annotations

from typing import Any, Callable, Literal

import torch
import torch.nn.functional as F


def _key_concat(obs_t: torch.Tensor, obs_next: torch.Tensor) -> torch.Tensor:
    cat = torch.cat([obs_t.reshape(-1), obs_next.reshape(-1)], dim=-1).float()
    return F.normalize(cat, dim=-1, eps=1e-8)


def _key_concat_batch(obs_t: torch.Tensor, obs_next: torch.Tensor) -> torch.Tensor:
    cat = torch.cat([obs_t, obs_next], dim=-1).float()
    return F.normalize(cat, dim=-1, eps=1e-8)


def _key_mean(obs_t: torch.Tensor, obs_next: torch.Tensor) -> torch.Tensor:
    avg = 0.5 * (obs_t.reshape(-1).float() + obs_next.reshape(-1).float())
    return F.normalize(avg, dim=-1, eps=1e-8)


def _key_mean_batch(obs_t: torch.Tensor, obs_next: torch.Tensor) -> torch.Tensor:
    avg = 0.5 * (obs_t.float() + obs_next.float())
    return F.normalize(avg, dim=-1, eps=1e-8)


KeyFn = Literal["concat", "mean"]


class FeatureMemoryBank:
    """Cosine retrieval over stored ``(obs_t, obs_next, action)`` triples."""

    __slots__ = (
        "obs_dim",
        "action_dim",
        "max_entries",
        "obs_t",
        "obs_next",
        "actions",
        "keys",
        "_ptr",
        "_key_fn",
        "_key_fn_batch",
        "_key_dim",
    )

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 7,
        max_entries: int = 50_000,
        device: torch.device | str | None = None,
        key_fn: KeyFn = "concat",
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.max_entries = int(max_entries)
        dev = torch.device(device) if device is not None else torch.device("cpu")

        if key_fn == "concat":
            self._key_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = _key_concat
            self._key_fn_batch: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = _key_concat_batch
            self._key_dim = self.obs_dim * 2
        elif key_fn == "mean":
            self._key_fn = _key_mean
            self._key_fn_batch = _key_mean_batch
            self._key_dim = self.obs_dim
        else:
            raise ValueError(f"unknown key_fn={key_fn!r}")

        self.obs_t = torch.zeros(self.max_entries, self.obs_dim, device=dev, dtype=torch.float32)
        self.obs_next = torch.zeros(self.max_entries, self.obs_dim, device=dev, dtype=torch.float32)
        self.actions = torch.zeros(self.max_entries, self.action_dim, device=dev, dtype=torch.float32)
        self.keys = torch.zeros(self.max_entries, self._key_dim, device=dev, dtype=torch.float32)
        self._ptr = 0

    def clear(self) -> None:
        self._ptr = 0

    @property
    def ptr(self) -> int:
        return self._ptr

    @property
    def device(self) -> torch.device:
        return self.keys.device

    def add(self, obs_t: torch.Tensor, obs_next: torch.Tensor, action: torch.Tensor) -> None:
        if self._ptr >= self.max_entries:
            raise RuntimeError(f"FeatureMemoryBank overflow (max_entries={self.max_entries})")

        ot = obs_t.detach().reshape(-1).to(self.device, dtype=torch.float32)
        on = obs_next.detach().reshape(-1).to(self.device, dtype=torch.float32)
        aa = action.detach().reshape(-1).to(self.device, dtype=torch.float32)

        i = self._ptr
        self.obs_t[i] = ot
        self.obs_next[i] = on
        self.actions[i] = aa
        self.keys[i] = self._key_fn(ot, on)
        self._ptr += 1

    def populate_vectorized(
        self,
        obs_t: torch.Tensor,
        obs_next: torch.Tensor,
        actions: torch.Tensor,
    ) -> None:
        """Bulk-populate the bank from pre-stacked tensors.

        ``obs_t`` and ``obs_next`` should be ``(N, obs_dim)``; ``actions``
        is ``(N, action_dim)``. Single host->device copy plus a single
        batched normalization for the keys, instead of N Python calls.
        """
        n = int(obs_t.shape[0])
        if n > self.max_entries:
            raise RuntimeError(
                f"FeatureMemoryBank overflow: {n} entries > max_entries={self.max_entries}. "
                "Reconstruct the bank with a larger ``max_entries``."
            )
        ot = obs_t.detach().to(self.device, dtype=torch.float32)
        on = obs_next.detach().to(self.device, dtype=torch.float32)
        aa = actions.detach().to(self.device, dtype=torch.float32)
        self.obs_t[:n] = ot
        self.obs_next[:n] = on
        self.actions[:n] = aa
        self.keys[:n] = self._key_fn_batch(ot, on)
        self._ptr = n

    def populate_from_dataset(
        self,
        ds: Any,
        desc: str | None = None,
        obs_t_key: str = "s_t",
        obs_next_key: str = "s_next",
        action_key: str = "action",
    ) -> None:
        """Populate from a Dataset.

        Fast path: if ``ds`` exposes ``stacked_obs_t``, ``stacked_obs_next``,
        and ``stacked_actions`` methods (returning pre-stacked tensors),
        we populate in one vectorized batch — typically two orders of
        magnitude faster than the per-row Python loop. The slow path is
        kept for compatibility with adapters that don't pre-tensor.
        """
        self.clear()
        if (
            hasattr(ds, "stacked_obs_t")
            and hasattr(ds, "stacked_obs_next")
            and hasattr(ds, "stacked_actions")
        ):
            self.populate_vectorized(
                ds.stacked_obs_t(obs_t_key),
                ds.stacked_obs_next(obs_next_key),
                ds.stacked_actions(action_key),
            )
            print(f"[memory] FeatureMemoryBank populated (vectorized): ptr={self.ptr} / max={self.max_entries}")
            return

        try:
            from tqdm import tqdm

            itr = tqdm(range(len(ds)), desc=desc or "Populate memory bank")
        except Exception:  # noqa: BLE001
            itr = range(len(ds))

        for i in itr:
            ex = ds[i]
            self.add(ex[obs_t_key], ex[obs_next_key], ex[action_key])

        print(f"[memory] FeatureMemoryBank populated (slow path): ptr={self.ptr} / max={self.max_entries}")

    def retrieve(
        self,
        obs_t: torch.Tensor,
        obs_next: torch.Tensor,
        k: int = 3,
        tau_min: float | None = None,
        exclude_bank_row: int | torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        n = self.ptr
        if n == 0:
            return None

        q = self._key_fn(obs_t, obs_next).to(self.device, dtype=torch.float32)
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
        obs_t_b: torch.Tensor,
        obs_next_b: torch.Tensor,
        k: int = 3,
        tau_min: float | None = None,
        exclude_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(actions[B,K,A], mask[B,K])`` over the top-k retrieved rows."""
        dv = self.device
        out_dev = obs_t_b.device
        B = int(obs_t_b.shape[0])
        kk_req = max(1, int(k))
        n = self.ptr

        if n == 0:
            return (
                torch.zeros(B, kk_req, self.action_dim, device=out_dev, dtype=torch.float32),
                torch.zeros(B, kk_req, dtype=torch.bool, device=out_dev),
            )

        qk = self._key_fn_batch(obs_t_b, obs_next_b).to(dv)
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

    def hit_rate(self, ds: Any, k: int = 3, tau_min: float | None = None) -> float:
        if len(ds) == 0:
            return float("nan")
        good = 0
        total = len(ds)
        for i in range(total):
            ex = ds[i]
            ret = self.retrieve(ex["s_t"], ex["s_next"], k=k, tau_min=tau_min)
            good += ret is not None and ret.shape[0] > 0
        return float(good) / float(total)
