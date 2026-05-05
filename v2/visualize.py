"""Render what the IDM model is actually doing.

Each rendered panel shows, for one transition:

  * the observation frame at time ``t``           (left subplot)
  * the observation frame at time ``t+1``         (middle subplot)
  * the model's predicted 7-D action vs the
    ground-truth action that produced the
    transition, as a side-by-side bar chart       (right subplot)

The 7 action dimensions are labelled ``(dx, dy, dz, drx, dry, drz, grip)``
which is the OSC_POSE layout shared by RoboMimic and the OpenVLA-aligned
LIBERO RLDS. When ``action_pred`` is omitted the panel still renders the
ground truth alone, useful for dataset-level previews from the feature
extraction pass.

For low-dim datasets where there is no RGB frame available we fall back to
a 3-D scatter of the end-effector position with an arrow for the
ground-truth XYZ delta, so the viewer can still tell what the model is
trying to explain.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

ACTION_LABELS = ("dx", "dy", "dz", "drx", "dry", "drz", "grip")


def _ensure_matplotlib():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # noqa: F401

    return matplotlib


def _action_bar_panel(ax, action_gt: np.ndarray, action_pred: np.ndarray | None) -> None:
    n = len(ACTION_LABELS)
    x = np.arange(n)
    width = 0.4 if action_pred is not None else 0.6
    ax.bar(x - (width / 2 if action_pred is not None else 0), action_gt, width=width, label="ground truth", color="#3b6cd1")
    if action_pred is not None:
        ax.bar(x + width / 2, action_pred, width=width, label="prediction", color="#d97043")
    ax.axhline(0.0, color="#777", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(ACTION_LABELS, fontsize=8)
    # Adaptive ylim so both q01_q99 ([-1, 1]) and z-score (often > 1) plots
    # render cleanly without clipping the tallest bar.
    stack = [action_gt]
    if action_pred is not None:
        stack.append(np.asarray(action_pred))
    finite_max = float(np.max(np.abs(np.concatenate(stack)))) if stack else 1.0
    ymax = max(1.1, finite_max * 1.15)
    ax.set_ylim(-ymax, ymax)
    ax.set_ylabel("normalized action")
    ax.legend(fontsize=8, loc="upper right")
    if action_pred is not None:
        rmse = float(np.sqrt(np.mean((action_pred - action_gt) ** 2)))
        ax.set_title(f"action: pred vs GT (RMSE {rmse:.3f})", fontsize=9)
    else:
        ax.set_title("action: ground truth", fontsize=9)


def _state_panel(ax, state_t: np.ndarray, state_next: np.ndarray, action_gt: np.ndarray) -> None:
    """Fallback panel when no image frames are available."""
    if state_t.shape[0] >= 9 and state_next.shape[0] >= 9:
        # RoboMimic low-dim layout is ``object + eef_pos + eef_quat + gripper``;
        # the object block width varies by task, but the last 9 dims are stable.
        eef_t = state_t[-9:-6]
        eef_n = state_next[-9:-6]
    else:
        eef_t = state_t[:3]
        eef_n = state_next[:3]
    ax.set_aspect("equal")
    ax.scatter([eef_t[0]], [eef_t[1]], color="#3b6cd1", s=60, label="t")
    ax.scatter([eef_n[0]], [eef_n[1]], color="#d97043", s=60, label="t+1")
    ax.annotate(
        "",
        xy=(eef_n[0], eef_n[1]),
        xytext=(eef_t[0], eef_t[1]),
        arrowprops=dict(arrowstyle="->", color="#444", lw=1.2),
    )
    ax.set_xlabel("eef_x")
    ax.set_ylabel("eef_y")
    ax.set_title(f"state delta (XY); a_xyz = ({action_gt[0]:+.2f}, {action_gt[1]:+.2f}, {action_gt[2]:+.2f})", fontsize=9)
    ax.legend(fontsize=8)


def render_transition(
    save_path: str | Path,
    frame_t: np.ndarray | None,
    frame_next: np.ndarray | None,
    action_gt: np.ndarray,
    action_pred: np.ndarray | None = None,
    title: str | None = None,
    state_t: np.ndarray | None = None,
    state_next: np.ndarray | None = None,
) -> Path:
    """Save one (frame_t, frame_next, action) panel as a PNG."""
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    has_images = frame_t is not None and frame_next is not None
    cols = 3 if has_images else 2
    fig_w = 4 * cols
    fig, axes = plt.subplots(1, cols, figsize=(fig_w, 3.2))
    if cols == 2:
        axes = list(axes)
    if has_images:
        axes[0].imshow(frame_t)
        axes[0].set_title("obs_t", fontsize=9)
        axes[0].axis("off")
        axes[1].imshow(frame_next)
        axes[1].set_title("obs_{t+1}", fontsize=9)
        axes[1].axis("off")
        _action_bar_panel(axes[2], np.asarray(action_gt, dtype=np.float32),
                          None if action_pred is None else np.asarray(action_pred, dtype=np.float32))
    else:
        if state_t is not None and state_next is not None:
            _state_panel(axes[0], np.asarray(state_t), np.asarray(state_next), np.asarray(action_gt))
        else:
            axes[0].axis("off")
            axes[0].text(0.5, 0.5, "no observation\navailable", ha="center", va="center")
        _action_bar_panel(axes[1], np.asarray(action_gt, dtype=np.float32),
                          None if action_pred is None else np.asarray(action_pred, dtype=np.float32))

    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return save_path


def render_grid(
    save_path: str | Path,
    panels: Sequence[dict],
    title: str | None = None,
    cols: int = 3,
) -> Path:
    """Render up to N transitions as a grid of panels."""
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    n = len(panels)
    rows = max(1, (n + cols - 1) // cols)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[None, :]
    elif cols == 1:
        axes = axes[:, None]

    for i, panel in enumerate(panels):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        frame_t = panel.get("frame_t")
        frame_next = panel.get("frame_next")
        action_gt = np.asarray(panel.get("action_gt"))
        action_pred = panel.get("action_pred")
        if frame_t is not None and frame_next is not None:
            ax.axis("off")
            h = frame_t.shape[0]
            stitched = np.concatenate([frame_t, frame_next], axis=1)
            ax.imshow(stitched)
            ax.axhline(h - 0.5, color="white", linewidth=0)
        else:
            # No RGB frames available -> render the action bars in the
            # grid cell so the panel is still informative.
            _action_bar_panel(
                ax, action_gt,
                None if action_pred is None else np.asarray(action_pred),
            )
        rmse = ""
        if action_pred is not None:
            rmse = f" RMSE={float(np.sqrt(np.mean((np.asarray(action_pred) - action_gt) ** 2))):.3f}"
        ax.set_title(panel.get("title", "") + rmse, fontsize=8)

    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r, c].axis("off")

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return save_path


def sample_dataset_preview(
    dataset,
    save_path: str | Path,
    n_samples: int = 9,
    seed: int = 0,
) -> Path | None:
    """Sample N transitions from any adapter that exposes ``fetch_frames``.

    Returns the saved path, or ``None`` if frames are not available.
    """
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    panels: list[dict] = []
    for i in indices:
        try:
            f_t, f_n = dataset.fetch_frames(int(i))
        except Exception:  # noqa: BLE001
            return None
        ex = dataset[int(i)]
        action_gt = ex["action_raw"].cpu().numpy() if "action_raw" in ex else ex["action"].cpu().numpy()
        panels.append(
            {
                "frame_t": f_t,
                "frame_next": f_n,
                "action_gt": action_gt,
                "title": f"idx={int(i)} demo={ex.get('demo_key','?')} t={int(ex['t'].item())}",
            }
        )
    return render_grid(save_path, panels, title=f"{type(dataset).__name__} preview")
