"""
01_eda.py — Exploratory Data Analysis for the RoboMimic Lift (low-dim) dataset.

Run from the repo root:
    python notebooks/01_eda.py
"""

import os
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HDF5_PATH = "data/lift/ph/low_dim_v141.hdf5"
FIGURES_DIR = "notebooks/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load file and print full key structure
# ---------------------------------------------------------------------------
print("=" * 60)
print("KEY STRUCTURE")
print("=" * 60)

def _print_tree(name: str, obj) -> None:
    depth = name.count("/")
    indent = "  " * depth
    short = name.split("/")[-1]
    if isinstance(obj, h5py.Dataset):
        print(f"{indent}{short}  [shape={obj.shape}, dtype={obj.dtype}]")
    else:
        print(f"{indent}{short}/")

with h5py.File(HDF5_PATH, "r") as f:
    f.visititems(_print_tree)

# ---------------------------------------------------------------------------
# 2. Summary statistics
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

with h5py.File(HDF5_PATH, "r") as f:
    data_grp = f["data"]
    demo_keys = sorted(data_grp.keys())          # e.g. ['demo_0', 'demo_1', ...]
    n_demos = len(demo_keys)

    ep_lengths = np.array([data_grp[d]["actions"].shape[0] for d in demo_keys])

    # Action dimensionality from demo_0
    action_dim = data_grp[demo_keys[0]]["actions"].shape[1]

    # Observation dimensionality = sum of all obs sub-arrays at t=0
    obs_grp = data_grp[demo_keys[0]]["obs"]
    obs_keys = sorted(obs_grp.keys())
    obs_dim = sum(obs_grp[k].shape[1] for k in obs_keys)

    print(f"Number of demos      : {n_demos}")
    print(f"Episode lengths      : min={ep_lengths.min()}  "
          f"max={ep_lengths.max()}  mean={ep_lengths.mean():.2f}  "
          f"std={ep_lengths.std():.2f}")
    print(f"Action dimensionality: {action_dim}")
    print(f"Obs dimensionality   : {obs_dim}  "
          f"(across {len(obs_keys)} modalities)")
    print(f"  Modalities: {obs_keys}")

    # -----------------------------------------------------------------------
    # 3. Collect all actions for distribution plots + normalization stats
    # -----------------------------------------------------------------------
    all_actions = np.concatenate(
        [data_grp[d]["actions"][:] for d in demo_keys], axis=0
    )  # (N_total_steps, action_dim)

    # -----------------------------------------------------------------------
    # 5. Normalization stats (printed before figures so they appear together)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ACTION NORMALIZATION STATS  (mean / std per DOF)")
    print("=" * 60)
    action_labels = [
        "joint_0", "joint_1", "joint_2", "joint_3",
        "joint_4", "joint_5", "gripper",
    ]
    action_mean = all_actions.mean(axis=0)
    action_std  = all_actions.std(axis=0)
    col_w = max(len(l) for l in action_labels)
    print(f"{'DOF':<{col_w}}   {'mean':>10}   {'std':>10}")
    print("-" * (col_w + 26))
    for i, label in enumerate(action_labels):
        print(f"{label:<{col_w}}   {action_mean[i]:>10.6f}   {action_std[i]:>10.6f}")

    # -----------------------------------------------------------------------
    # 6. Sample observation from demo_0, timestep 0
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SAMPLE OBSERVATION  (demo_0, t=0)")
    print("=" * 60)
    for k in obs_keys:
        vec = obs_grp[k][0]
        fmt = np.array2string(vec, precision=4, suppress_small=True, max_line_width=120)
        print(f"  {k:<30s} shape={vec.shape}  {fmt}")

# ---------------------------------------------------------------------------
# 3. Action distribution plots
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PLOTTING")
print("=" * 60)

fig, axes = plt.subplots(
    nrows=action_dim, ncols=1,
    figsize=(8, 2.5 * action_dim),
    tight_layout=True,
)
fig.suptitle("Action Distributions — RoboMimic Lift (low-dim)", fontsize=13, y=1.002)

for i, ax in enumerate(axes):
    col = all_actions[:, i]
    ax.hist(col, bins=60, color="steelblue", edgecolor="none", alpha=0.85)
    ax.axvline(action_mean[i], color="crimson", linewidth=1.5, label=f"mean={action_mean[i]:.3f}")
    ax.set_xlabel(f"{action_labels[i]}  (DOF {i})")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)

out_path = os.path.join(FIGURES_DIR, "action_distributions.png")
fig.savefig(out_path, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")

# ---------------------------------------------------------------------------
# 4. Episode length distribution
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(ep_lengths, bins=range(ep_lengths.min(), ep_lengths.max() + 2),
        color="darkorange", edgecolor="white", alpha=0.9)
ax.axvline(ep_lengths.mean(), color="navy", linewidth=2,
           label=f"mean={ep_lengths.mean():.1f}")
ax.set_xlabel("Episode length (timesteps)")
ax.set_ylabel("Number of demos")
ax.set_title("Episode Length Distribution — RoboMimic Lift (low-dim)")
ax.legend()
fig.tight_layout()

out_path = os.path.join(FIGURES_DIR, "episode_lengths.png")
fig.savefig(out_path, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")

print("\nDone.")
