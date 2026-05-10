# RAID — Retrieval-Augmented Inverse Dynamics with GR-1 Visual Encoder

**RAID** (Retrieval-Augmented Inverse Dynamics) is an offline behaviour cloning framework for robot manipulation that uses **GR-1** (ByteDance) as a frozen visual world-model encoder and a learned cross-attention decoder conditioned on retrieved training actions.

Evaluated on **LIBERO-Spatial** (10 pick-and-place tasks, 50 demonstrations each).

---

## Motivation: Ken Goldberg's action inference gap

In contact-rich manipulation, the mapping from RGB frames to correct motor commands is inherently ambiguous: contact state, friction, and task intent are latent in images that a neural encoder alone cannot fully resolve.

**RAID** addresses this with a retrieval-augmented architecture: given current and predicted next visual features, retrieve the *k* most similar training transitions and feed their executed actions as a prior into a cross-attention decoder. The decoder learns to *correct* this prior rather than predict actions from scratch, directly addressing the distribution mismatch that cripples direct imitation from limited data.

---

## Architecture

```
frame_t  ──►  GR-1.encode_frames()       ──►  feat_t    (384-dim, frozen MAE + embed_img)
frame_t  ──►  GR-1.predict_next_feat()   ──►  feat_next (384-dim, GR-1 predicted next-frame feature)

Memory bank: retrieve top-k actions by cosine similarity on (feat_t, feat_next)

(feat_t, feat_next, [a_1, …, a_k])  ──►  RAIDDecoderVisual (cross-attention)  ──►  action_norm
action_norm  ──►  denormalise  ──►  7-DOF end-effector command
```

**All GR-1 weights stay frozen.** Only the RAID decoder trains.

### Experimental conditions

| Condition | Model | Retrieval |
|-----------|-------|-----------|
| `direct_visual` | 3-layer MLP on (feat_t, feat_next) | No |
| `raid_visual` | Cross-attention decoder | Yes (k=3 in the final visual runs) |

---

## Key Results

### Stage 1 — Offline Behaviour Cloning (validation MSE ↓)

| Demo scale | `direct_visual` | `raid_visual` | Improvement |
|------------|----------------|---------------|-------------|
| 25 demos | 0.852 | **0.131** | **6.5×** |
| 50 demos | 0.639 | **0.158** | 4.0× |
| 100 demos | 0.580 | **0.171** | 3.4× |
| 200 demos | 0.554 | **0.174** | 3.2× |

Retrieval-augmented cross-attention provides a **6× MSE reduction** at the lowest data regime (25 demos), demonstrating strong sample efficiency from retrieval as a prior.

### Stage 2 — GRPO Online Fine-Tuning

| Metric | Value |
|--------|-------|
| Full run | 382 updates logged |
| Polish run | 195 updates logged |
| Best success rate | **0.25** |
| Best mean reward | **1.226** (polish update 158) |
| Final checkpoint | `raid_grpo_final/raid_visual_grpo_polish_best_20260509_005631.pt` |

The final GRPO run produced intermittent task completions (best SR = 25% over
4 rollout samples) and moved mean reward from strongly negative shaped rewards
to positive reward spikes. It is not a solved policy yet: success is sparse and
unstable, but the online stage is now demonstrably capable of finding successful
rollouts under EGL.

---

## Repository Layout

| Path | Role |
|------|------|
| `src/gr1_encoder.py` | Frozen GR-1 encoder — `encode_frames()` and `predict_next_feat()` |
| `src/data_libero.py` | LIBERO HDF5 loader; normalisation; train/val split by demo |
| `src/memory.py` | `RAIDMemoryBank` — cosine-similarity top-k retrieval over `(feat_t, feat_next)` |
| `src/memory_libero.py` | Earlier LIBERO/V-JEPA memory-bank implementation retained for reference |
| `src/models.py` | Active `DirectMLPVisual` and `RAIDDecoderVisual` implementations |
| `src/models_libero.py` | Earlier visual model definitions retained for reference/autoresearch |
| `src/train_libero.py` | Training loop for `direct_visual` and `raid_visual` |
| `src/run_all_libero.py` | Sweep driver: all conditions × all demo scales |
| `src/rollout_libero.py` | LIBERO environment rollout with GR-1 + RAID inference |
| `src/grpo_libero.py` | GRPO online fine-tuning loop |
| `src/cache_gr1_features.py` | Pre-compute and cache GR-1 features for the dataset |
| `configs/` | Per-scale norm stats, loss curves, sweep results |
| `scripts/compare_video.py` | Generate side-by-side rollout videos for RAID vs direct visual policies |
| `scripts/visualize_transitions.py` | Render static transition grids with frames, GR-1 predictions, and action bars |
| `raid_grpo_final/` | Final GRPO logs, plots, scripts, and best checkpoints from the Lambda run |
| `GRPO_FINAL_RUN.md` | Narrative record of the final Lambda run and preserved artifacts |
| `STATUS.md` | Full session-by-session diagnosis and continuity notes |

---

## Running the Experiment

**Step 1 — Cache GR-1 features (run once):**
```bash
python src/cache_gr1_features.py \
    --dataset_dir data/libero_spatial/libero_spatial \
    --output_dir  data/libero_spatial/features \
    --device cuda
```

**Step 2 — Full BC sweep (4 scales × 2 conditions):**
```bash
python src/run_all_libero.py \
    --feature_dir data/libero_spatial/features \
    --device cuda
```

**Step 3 — GRPO online fine-tuning:**
```bash
python src/grpo_libero.py \
    --n_demos 200 \
    --feature_dir data/libero_spatial/features \
    --dataset_dir data/libero_spatial/libero_spatial/libero_spatial \
    --device cuda
```

The preserved final Lambda script can also be run from the artifact bundle:

```bash
MUJOCO_GL=egl python raid_grpo_final/grpo_libero_remote_final.py \
    --n_demos 200 \
    --n_updates 300 \
    --G 4 \
    --checkpoint_path models/raid_visual_grpo_best.pt \
    --log_path configs/grpo_libero_log.json
```

See `GRPO_FINAL_RUN.md` for the final run summary and artifact inventory.

---

## Hyperparameters

| Item | Value |
|------|-------|
| GR-1 feature dim | 384 |
| Retrieval k | 3 |
| Cross-attention projection dim | 64 |
| Optimizer | AdamW, lr=1e-3, wd=1e-4 |
| Epochs | 100 |
| Batch size | 256 |
| Demo scales | 25, 50, 100, 200 |
| Train/val split | 80/20 by demonstration |
| Random seed | 42 |

---

## Dependencies

- Python 3.10+
- PyTorch (CUDA)
- `h5py`, `numpy`, `tqdm`
- GR-1 repo (`~/GR-1`) from [bytedance/GR-1](https://github.com/bytedance/GR-1)
- LIBERO (`libero` package) for simulation rollouts

---

## Low-Dimensional Baseline

The original RAID experiment on **RoboMimic Lift** with low-dimensional proprioceptive state is preserved in `src/data.py`, `src/models.py`, `src/train.py`, `src/evaluate.py`, and `src/run_all.py`. See `configs/results.json` for those results.
