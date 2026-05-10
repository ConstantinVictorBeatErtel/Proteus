# RAID + GR-1 + LIBERO: Project Status

**Last updated:** 2026-05-10 (final Lambda GRPO artifacts preserved)  
**Continuity file** — read this if picking up this project in a new context window.

---

## Latest Update: Final Lambda Run

The older sections below are retained as historical diagnosis. The final Lambda
session resolved the biggest infrastructure blocker by running LIBERO with EGL,
preserved the source changes in the repo, and saved the final GRPO artifact
bundle under `raid_grpo_final/`.

Current source of truth:

| Item | Status |
|------|--------|
| Offline BC sweep | Re-run successfully on cached GR-1 features |
| Best BC condition | `raid_visual` at every demo scale |
| Final GRPO full run | 382 updates logged |
| Final GRPO polish run | 195 updates logged |
| Best observed SR | **0.25** |
| Best observed mean reward | **1.226** at polish update 158 |
| Recommended checkpoint | `raid_grpo_final/raid_visual_grpo_polish_best_20260509_005631.pt` |

Fresh BC validation MSE:

| Condition | 25 demos | 50 demos | 100 demos | 200 demos |
|-----------|----------|----------|-----------|-----------|
| `direct_visual` | 0.852 | 0.639 | 0.580 | 0.554 |
| `raid_visual` | **0.131** | **0.158** | **0.171** | **0.174** |

Interpretation: RAID's retrieval-augmented visual decoder is still clearly
better than direct imitation offline. GRPO is no longer purely a shaped-reward
reach improvement: it found intermittent successful rollouts, but the policy is
not robustly solved yet. See `GRPO_FINAL_RUN.md` for the artifact inventory and
the final run narrative.

---

## What Was Built

RAID (Retrieval-Augmented Inverse Dynamics) extended with:
- **GR-1** (ByteDance) as a frozen visual world model / encoder
- **LIBERO-Spatial** as the training + rollout dataset
- **RAIDDecoderVisual** = cross-attention decoder on 384-dim GR-1 features
- **GRPO** fine-tuning loop (Group Relative Policy Optimization)

### Inference flow (at GRPO rollout time):
```
frame_t  →  GR-1.encode_frames()       →  feat_t         (384-dim)
frame_t  →  GR-1.predict_next_feat()   →  feat_next      (384-dim, world-model predicted)
(feat_t, feat_next, top-k retrieved actions)  →  RAIDDecoderVisual  →  action_norm
action_norm  →  denormalise  →  LIBERO sim step
```

---

## Results Summary

### Stage 1: Offline BC — ✅ PASSED (massively)

| Condition | 25 demos | 50 demos | 100 demos | 200 demos |
|-----------|----------|----------|-----------|-----------|
| `direct_visual` | 0.842 | 0.637 | 0.570 | 0.552 |
| `raid_visual` | **0.132** | **0.154** | **0.169** | **0.171** |

`raid_visual` is **6× better** than `direct_visual` at 25 demos. GR-1 features + cross-attention retrieval works very well offline.

### Stage 2: GRPO Online — ⚠️ PARTIAL (reward improving, SR = 0)

| Metric | Value |
|--------|-------|
| Updates completed | 86 / 100 |
| Starting mean reward | -3.881 |
| Best mean reward | -3.153 (update 59) |
| Final mean reward | -3.331 |
| Improvement | +0.73 (18.8%) |
| Success rate (any update) | **0.00** |

**The policy learned to move the EE closer to the bowl** (shaped reach reward improved 18.8%), but **never completed the task** (SR = 0 throughout).

---

## Diagnosis: What Is Not Working and Why

### Problem 1: Episode horizon too short (root cause of SR=0)

The LIBERO pick-and-place task requires ~80–150 steps to complete (reach → grasp → lift → place). We were limited to **max_steps=30** due to osmesa (CPU) rendering overhead of **337ms per step**.

```
Time budget:  30 steps × 337ms × G=4 rollouts = ~40s/update  (acceptable)
Needed steps: ~80-150 steps to complete pick-and-place
Gap:          policy never has enough steps to succeed → SR always 0
```

With GPU-accelerated rendering (EGL), each step takes **~5ms** instead of 337ms, so:
```
With EGL:  150 steps × 5ms × G=4 rollouts = ~3s/update (vs 40s now)
           100 updates in ~5 minutes instead of ~85 minutes
```

### Problem 2: GRPO signal too weak at 30 steps

With only shaped reach reward and no success events, GRPO is optimising a proxy signal (EE-to-bowl distance) that may not correlate well with task completion. True GRPO works best when **some rollouts succeed and some fail** — the contrast drives meaningful policy improvement.

### Problem 3: GR-1 world-model drift at inference

The BC training used **ground-truth** `(feat_t, feat_next)` pairs from real frames. At GRPO rollout time, `feat_next` comes from **GR-1's prediction**, which may differ from what the real frame encodes. This training/inference distribution mismatch limits how well the BC-trained policy transfers to rollouts.

This was the same `s_next` proxy problem from the original low-dim RAID, partially solved but not fully eliminated.

### Problem 4: Osmesa rendering is a hard constraint on this A10

No fix possible without either:
- GPU EGL rendering (needs `libEGL` linked to NVIDIA driver properly)
- A different rendering stack (e.g., mujoco native renderer)
- A headless GPU with proper driver setup

---

## What a Stronger GPU Instance Solves

A fresh Lambda Labs instance (A100 or H100) with proper GPU rendering setup would unlock:

| Item | Current (A10 + osmesa) | New (A100 + EGL) |
|------|----------------------|-----------------|
| Env step time | 337ms | ~5ms |
| Steps per episode | 30 | 150+ |
| Time per update | ~40s | ~3s |
| 100 updates | ~85 min | ~5 min |
| Can reach SR > 0? | No (too few steps) | Yes |
| GRPO signal quality | Weak (reach only) | Strong (actual success/failure) |

**The core algorithm is validated.** The bottleneck is purely infrastructure (rendering).

---

## Recommended Next Steps (priority order)

### If spinning up a new GPU instance:

1. **Setup EGL rendering** (do this first, before anything else):
   ```bash
   sudo apt-get install -y libegl1-mesa-dev libgles2-mesa-dev
   # Test: MUJOCO_GL=egl python3 -c "import mujoco; print('EGL OK')"
   ```

2. **Install everything** (same as this session but with EGL):
   ```bash
   git clone git@github.com:ConstantinVictorBeatErtel/RAID.git
   cd RAID
   pip install robosuite==1.4.0 bddl cloudpickle gym easydict
   pip install -e /path/to/LIBERO
   pip install clip timm
   # Copy checkpoints: checkpoints/gr1/snapshot_ABCD.pt + mae_pretrain_vit_base.pth
   # Copy dataset: data/libero_spatial/ (or re-download)
   ```

3. **Re-cache GR-1 features** (fast with GPU):
   ```bash
   python3 src/cache_gr1_features.py --dataset_dir data/libero_spatial/libero_spatial --device cuda
   ```

4. **Re-run BC sweep** (validates Stage 1 again on new machine):
   ```bash
   python3 src/run_all_libero.py --feature_dir data/libero_spatial/features --device cuda
   ```

5. **Run GRPO with full horizon**:
   ```bash
   MUJOCO_GL=egl python3 src/grpo_libero.py \
       --n_demos 200 --n_updates 500 --G 8 \
       --task_idx 1 --device cuda --log_every 5
   # Change max_steps=30 → max_steps=150 in src/rollout_libero.py line 113
   ```

### If staying on current instance:

Option A — Use robosuite Lift task (no image, faster env):
- Revert to low-dim RAID + `s_next_proxy = s_t` (already proved SR=0 with that too)
- Not recommended — fundamental proxy problem not solved

Option B — Latent rollout (no rendering at all):
- Render initial frame once per episode
- Use GR-1 `predict_next_feat` in a chain for all subsequent steps (pure imagination)
- Step env with `use_camera_obs=False` (41ms/step)
- Expected speed: ~8s/episode at 150 steps, ~32s/update with G=4
- Risk: GR-1 predictions drift from real state over long horizons

---

## Repository Layout (key files)

| File | Purpose |
|------|---------|
| `src/data_libero.py` | LIBERO HDF5 loader; `LiberoTransitionDataset`, `CachedFeatureDataset` |
| `src/gr1_encoder.py` | Frozen GR-1 wrapper: `encode_frames()` + `predict_next_feat()` |
| `src/cache_gr1_features.py` | Pre-caches GR-1 features for all 10 LIBERO-Spatial tasks |
| `src/models.py` | Added `DirectMLPVisual` + `RAIDDecoderVisual` (feat_dim=384) |
| `src/train_libero.py` | BC training loop for visual RAID on cached features |
| `src/run_all_libero.py` | Full offline sweep: 2 conditions × 4 demo scales |
| `src/rollout_libero.py` | LIBERO episode rollout with GR-1 next-frame prediction |
| `src/grpo_libero.py` | GRPO training loop for LIBERO + GR-1 |
| `STATUS.md` | This file |

---

## Data & Checkpoints (on Lambda Labs instance at time of writing)

| Path | Contents |
|------|---------|
| `data/libero_spatial/libero_spatial/` | 10 LIBERO-Spatial HDF5 task files (50 demos each) |
| `data/libero_spatial/libero_spatial/norm_stats.pt` | Action mean/std |
| `data/libero_spatial/features/` | Pre-cached GR-1 features (~62k transitions) |
| `checkpoints/gr1/mae_pretrain_vit_base.pth` | MAE ViT-Base pretrained weights |
| `checkpoints/gr1/snapshot_ABCD.pt` | GR-1 ABCD checkpoint |
| `models/raid_visual_*demos_libero_best.pt` | BC-trained RAIDDecoderVisual checkpoints |
| `configs/grpo_libero_log.json` | GRPO training log (86 updates) |

---

## Known Fixes Already Applied

| Issue | Fix |
|-------|-----|
| GR-1 `embed_timestep.view()` crashes for seq_len=1 | Patched `/home/ubuntu/GR1/models/gr1.py`: `weight` → `weight[:sequence_length]` |
| LIBERO requires robosuite 1.4.0 (had 1.5.2) | `pip install robosuite==1.4.0` |
| LIBERO missing deps | `pip install bddl cloudpickle gym easydict` |
| No EGL on A10 → use osmesa | `PYOPENGL_PLATFORM=osmesa MUJOCO_GL=osmesa` |
| GR-1 `models/` package shadows RAID `src/models.py` | `gr1_encoder.py` saves/restores `sys.modules['models']` around GR-1 import |
| LIBERO benchmark prompt blocks stdin | Direct BDDL file path (bypasses `get_benchmark_dict()`) |
| `RAIDMemoryBank.add()` wrong args in `train_libero.py` | Fixed to `bank.add(feat_t[i], feat_next[i], actions[i])` |
| LIBERO sparse reward → GRPO always skips | Added `_shaped_reward()` (reach + lift distance) |

---

## Research Context

**Why this architecture?**  
RAID's IDM `(s_t, s_next) → action` requires `s_next` at inference. GR-1 predicts future visual state from the current frame, enabling closed-loop rollouts without a ground-truth next state.

**Why GRPO?**  
BC minimises imitation error. GRPO fine-tunes on actual task rewards, enabling the policy to exceed BC performance on the real task.

**Why not DreamZero (NVIDIA WAM)?**  
23B parameters in BF16 = ~46GB VRAM. A10 has 23GB. Incompatible. Ideal upgrade target.

**Core result:**  
The GR-1 + RAID visual approach works dramatically better than low-dim RAID (0.132 vs 0.842 MSE at 25 demos). The GRPO pipeline is architecturally sound — it just needs GPU rendering to run episodes long enough for success events to occur.
