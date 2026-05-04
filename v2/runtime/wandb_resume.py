"""Thin W&B wrapper with run-id-keyed resume across Colab disconnects.

The ``run_id`` is persisted to ``<artifact_root>/runs/<run_id>/run_id.txt`` on
first launch. On a re-launch with the same deterministic id, ``wandb.init``
resumes the prior run with full history continuity.

If ``WANDB_DISABLED=1`` or wandb is not installed, the wrapper degrades to a
no-op ``DummyRun`` that mirrors the subset of the API we use.
"""

from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from typing import Any, Iterator

from .drive import CheckpointDir


def deterministic_run_id(*parts: Any) -> str:
    """SHA1 of the joined parts. Stable across machines so resume works."""
    s = "::".join(str(p) for p in parts)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


class DummyRun:
    """Stand-in when wandb is disabled or unavailable."""

    def __init__(self, run_id: str) -> None:
        self.id = run_id
        self.name = run_id
        self.step = 0

    def log(self, data: dict[str, Any], step: int | None = None) -> None:
        if step is not None:
            self.step = max(self.step, step)

    def finish(self, exit_code: int = 0) -> None:  # noqa: ARG002
        pass

    def watch(self, *_args: Any, **_kw: Any) -> None:
        pass

    def summary_update(self, mapping: dict[str, Any]) -> None:
        pass


def _wandb_disabled() -> bool:
    if os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}:
        return True
    try:
        import wandb  # type: ignore  # noqa: F401
        return False
    except Exception:
        return True


def _wandb_can_log_online() -> bool:
    """True iff wandb has a credential it can use to talk to the API."""
    if os.environ.get("WANDB_API_KEY"):
        return True
    if os.environ.get("WANDB_MODE", "").lower() in {"offline", "disabled"}:
        return False
    # Last resort: ask the SDK whether it can find a netrc credential.
    try:
        from wandb.sdk.lib.apikey import api_key  # type: ignore

        return bool(api_key(None))
    except Exception:
        return False


@contextmanager
def init_run(
    project: str,
    run_id: str,
    config: dict[str, Any] | None = None,
    name: str | None = None,
    tags: list[str] | None = None,
) -> Iterator[Any]:
    """Open a W&B run keyed on ``run_id`` with ``resume="allow"``.

    Persists ``run_id`` to ``<artifact_root>/runs/<run_id>/run_id.txt`` so a
    later session with the same deterministic id reattaches.

    If ``WANDB_DISABLED`` is set or ``wandb`` is not installed, returns a
    no-op :class:`DummyRun`. If ``wandb`` is available but no API key is
    found, runs in ``WANDB_MODE=offline`` so training never blocks on an
    interactive login prompt.
    """
    ck = CheckpointDir(run_id)
    rid_file = ck.run_id_file()
    if not rid_file.exists():
        rid_file.write_text(run_id)

    if _wandb_disabled():
        run: Any = DummyRun(run_id)
        try:
            yield run
        finally:
            run.finish()
        return

    import wandb  # type: ignore

    if not _wandb_can_log_online() and not os.environ.get("WANDB_MODE"):
        os.environ["WANDB_MODE"] = "offline"
        print("[wandb_resume] no WANDB_API_KEY found; running in offline mode")

    try:
        run = wandb.init(
            project=project,
            id=run_id,
            resume="allow",
            config=config or {},
            name=name or run_id,
            tags=tags or [],
            dir=str(ck.path),
        )
    except Exception as exc:  # noqa: BLE001 — never let logging kill training
        print(f"[wandb_resume] wandb.init failed ({exc!r}); falling back to DummyRun")
        run = DummyRun(run_id)
    try:
        yield run
    finally:
        try:
            run.finish()
        except Exception:  # noqa: BLE001
            pass
