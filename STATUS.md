# RAID + GR-1 + LIBERO: Project Status

**Last updated:** 2026-05-05  
**Continuity file** — read this if picking up this project in a new context window.

---

## Architecture Summary

RAID (Retrieval-Augmented Inverse Dynamics) extended with:
- **GR-1** (ByteDance) as a frozen visual world model / encoder
- **LIBERO-Spatial** as the training + rollout dataset (replaces RoboMimic Lift)
- **RAIDDecoderVisual** = cross-attention decoder operating on 384-dim GR-1 features
- **GRPO** fine-tuning loop (Group Relative Policy Optimization) on top of BC-pretrained policy

### Inference flow (at GRPO rollout time):
```
frame_t  →  GR-1.encode_frames()  →  feat_t         (384-dim)
frame_t  →  GR-1.predict_next_feat()  →  feat_next  (384-dim, world-model predicted)
(feat_t, feat_next, top-k retrieved actions)  →  RAIDDecoderVisual  →  action_norm
action_norm  →  denormalise  →  LIBERO sim step
```

This eliminates the original `s_next` proxy problem: GR-1 predicts the next visual state
without requiring a forward dynamics model of the low-dim state.

---

## Repository Layout (key new files)

| File | Purpose |
|------|---------|
| `src/data_libero.py` | LIBERO HDF5 loader; `LiberoTransitionDataset`, `CachedFeatureDataset` |
| `src/gr1_encoder.py` | Frozen GR-1 wrapper: `encode_frames()` + `predict_next_feat()` |
| `src/cache_gr1_features.py` | Pre-caches GR-1 features for all 10 LIBERO-Spatial tasks |
| `src/models.py` | Added `DirectMLPVisual` + `RAIDDecoderVisual` (feat_dim=384) |
| `src/train_libero.py` | BC training loop for visual RAID on cached features |
| `src/run_all_libero.py` | Full offline sweep: 2 conditions × 4 demo scales |
| `src/rollout_libero.py` | LIBERO episode rollout using GR-1 next-frame prediction |
| `src/grpo_libero.py` | GRPO training loop for LIBERO + GR-1 |

---

## Data & Checkpoints (on Lambda Labs instance)

| Path | Contents |
|------|---------|
| `data/libero_spatial/libero_spatial/` | 10 LIBERO-Spatial HDF5 task files (50 demos each) |
| `data/libero_spatial/libero_spatial/norm_stats.pt` | Action mean/std for normalisation |
| `data/libero_spatial/features/` | Pre-cached GR-1 features (10 × ~5000 transitions) |
| `data/libero_spatial/features/manifest.json` | List of cached .pt files + feat_dim=384 |
| `checkpoints/gr1/mae_pretrain_vit_base.pth` | MAE ViT-Base pretrained weights |
| `checkpoints/gr1/snapshot_ABCD.pt` | GR-1 ABCD checkpoint (ByteDance) |
| `models/raid_visual_*demos_libero_best.pt` | BC-trained RAIDDecoderVisual checkpoints |
| `models/direct_visual_*demos_libero_best.pt` | BC-trained DirectMLPVisual checkpoints |

---

## Results

### Stage 1: Offline BC Sweep — LIBERO-Spatial (✓ PASSED)

| Condition | 25 demos | 50 demos | 100 demos | 200 demos |
|-----------|----------|----------|-----------|-----------|
| `direct_visual` | 0.8420 | 0.6372 | 0.5700 | 0.5523 |
| `raid_visual` | **0.1320** | **0.1543** | **0.1687** | **0.1714** |

`raid_visual` at 25 demos = 0.132 MSE vs `direct_visual` 0.842 MSE → **6× improvement**.  
Stage 1 gate: `raid_visual ≤ direct_visual` ✓ **PASSED by a wide margin.**

**Why RAID wins so strongly here:** GR-1's 384-dim features are semantically rich enough
that retrieved demonstrations from visually similar states are highly informative.
The cross-attention weighting further focuses on the most relevant retrieved action.

### Previous Stage 1 (low-dim RoboMimic, for reference)

| Condition | 25 demos |
|-----------|----------|
| `direct_mlp` | 0.336 |
| `raid` | 0.397 |
| `raid_crossattn` | 0.403 |

The visual + GR-1 path massively outperforms the old low-dim approach.

---

## Stage 2: GRPO Status — IN PROGRESS

### What works:
- ✅ GR-1 `encode_frames()` produces 384-dim features from 128×128 frames
- ✅ GR-1 `predict_next_feat()` produces different features from current frame (world model active)
- ✅ LIBERO env creates and resets correctly (osmesa headless rendering)
- ✅ Import namespace conflict between GR-1's `models/` package and RAID's `src/models.py` — FIXED in `gr1_encoder.py`
- ✅ `rollout_libero.py` structure complete with `run_episode()`
- ✅ `grpo_libero.py` GRPO loop complete

### Current bottleneck:
- Each episode step takes ~500ms (GR-1 `predict_next_feat` is expensive — full GPT-2 forward pass)
- At 200-500 steps/episode × G=4 rollouts → 400s+ per update
- **Fix needed:** limit `max_steps` per episode to ~50-100 for GRPO training speed

### Next actions (for new model picking this up):
1. **Limit episode length:** Add `max_steps=100` arg to `run_episode()` in `rollout_libero.py`
2. **Start GRPO run:** `PYOPENGL_PLATFORM=osmesa MUJOCO_GL=osmesa python3 src/grpo_libero.py --n_demos 200 --n_updates 100 --G 4 --task_idx 1 --device cuda`
3. **Monitor:** `tail -f configs/grpo_libero_log.json`
4. **Success gate:** SR > 0.0 at any update proves the pipeline works end-to-end

---

## Known Issues & Fixes Applied

| Issue | Fix |
|-------|-----|
| GR-1's `models/` shadows RAID's `src/models.py` | `gr1_encoder.py` saves/restores `sys.modules['models']` around GR-1 import |
| GR-1's `embed_timestep.view()` fails for seq_len < max_seq_len | Patched `/home/ubuntu/GR1/models/gr1.py` line 226: `self.embed_timestep.weight` → `self.embed_timestep.weight[:sequence_length]` |
| LIBERO requires robosuite 1.4.0 but 1.5.2 installed | Downgraded to `pip install robosuite==1.4.0` |
| LIBERO missing deps: `bddl`, `cloudpickle`, `gym`, `easydict` | `pip install bddl cloudpickle gym easydict` |
| No EGL display for headless rendering | Use `PYOPENGL_PLATFORM=osmesa MUJOCO_GL=osmesa` |
| `RAIDMemoryBank.add()` called with wrong args in `train_libero.py` | Fixed to `bank.add(feat_t[i], feat_next[i], actions[i])` |
| `retrieve()` called with wrong signature in `train_libero.py` | Fixed to use `retrieve_batch()` for efficiency |
| GR-1 `wpe.weight` listed as missing key | Expected — GR-1 uses its own `embed_timestep`, not GPT-2's `wpe` |

---

## Environment Setup

```bash
# On Lambda Labs A10 instance (23GB VRAM)
# OS: Ubuntu 22.04, Python 3.10, CUDA

# Key installed packages:
# torch 2.x, robosuite==1.4.0, libero (from /home/ubuntu/LIBERO)
# bddl, cloudpickle, gym, easydict, clip, timm

# LIBERO data at:  /home/ubuntu/RAID/data/libero_spatial/
# GR-1 repo at:    /home/ubuntu/GR1/
# GR-1 checkpoints: /home/ubuntu/RAID/checkpoints/gr1/

# Run GRPO:
cd /home/ubuntu/RAID
echo "N" | PYOPENGL_PLATFORM=osmesa MUJOCO_GL=osmesa python3 src/grpo_libero.py \
    --n_demos 200 --n_updates 100 --G 4 --task_idx 1 --device cuda

# Monitor:
tail -f configs/grpo_libero_log.json
```

---

## Research Context

**Why GR-1 + RAID?**  
RAID's inverse dynamics model `(s_t, s_next) → action` is powerful but requires `s_next`
at inference time, which is unavailable. GR-1 is a GPT-style visual world model that predicts
future video frames from current observation history. By using GR-1's predicted next frame
as `s_next`, RAID can operate in a closed-loop rollout.

**Why not DreamZero (NVIDIA WAM)?**  
DreamZero is 23B parameters in BF16 (~46GB). The A10 GPU has 23GB VRAM. Incompatible.
DreamZero would be the ideal upgrade if a larger GPU instance is available later.

**GRPO motivation:**  
BC training minimises imitation error but doesn't optimise task success. GRPO fine-tunes
the policy decoder weights using online rollout rewards, keeping the memory bank frozen.
This is analogous to RLHF for language models: BC is supervised pre-training, GRPO is RL fine-tuning.
