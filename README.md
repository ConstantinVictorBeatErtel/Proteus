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
frame_t  ──►  GR-1.predict_next_feat()   ──►  feat_next (384-dim, GR-1 forward-prediction head)

Memory bank: retrieve top-k actions by cosine similarity on feat_t

(feat_t, feat_next, [a_1, …, a_k])  ──►  RAIDDecoderVisual (cross-attention)  ──►  action_norm
action_norm  ──►  denormalise  ──►  7-DOF end-effector command
```

**All GR-1 weights stay frozen.** Only the RAID decoder trains.

### Experimental conditions

| Condition | Model | Retrieval |
|-----------|-------|-----------|
| `direct_visual` | 3-layer MLP on (feat_t, feat_next) | No |
| `raid_visual` | Cross-attention decoder | Yes (k=5) |

---

## Key Results

### Stage 1 — Offline Behaviour Cloning (validation MSE ↓)

| Demo scale | `direct_visual` | `raid_visual` | Improvement |
|------------|----------------|---------------|-------------|
| 25 demos | 0.842 | **0.132** | **6.4×** |
| 50 demos | 0.637 | **0.154** | 4.1× |
| 100 demos | 0.570 | **0.169** | 3.4× |
| 200 demos | 0.552 | **0.171** | 3.2× |

Retrieval-augmented cross-attention provides a **6× MSE reduction** at the lowest data regime (25 demos), demonstrating strong sample efficiency from retrieval as a prior.

### Stage 2 — GRPO Online Fine-Tuning

| Metric | Value |
|--------|-------|
| Updates completed | 86 / 100 |
| Starting mean reward | −3.881 |
| Best mean reward | −3.153 (update 59) |
| Improvement | +18.8% |
| Success rate | 0.00 |

The policy learned to move the end-effector closer to the target (shaped reach reward improved 18.8%), but **never completed the task** (SR = 0). Root cause: osmesa CPU rendering limits episodes to 30 steps due to ~337 ms/step overhead; pick-and-place requires ~80–150 steps to complete.

---

## Repository Layout

| Path | Role |
|------|------|
| `src/gr1_encoder.py` | Frozen GR-1 encoder — `encode_frames()` and `predict_next_feat()` |
| `src/data_libero.py` | LIBERO HDF5 loader; normalisation; train/val split by demo |
| `src/memory_libero.py` | `RAIDMemoryBankLibero` — cosine-similarity top-k retrieval |
| `src/models_libero.py` | `DirectMLPVisual` and `RAIDDecoderVisual` (cross-attention) |
| `src/train_libero.py` | Training loop for `direct_visual` and `raid_visual` |
| `src/run_all_libero.py` | Sweep driver: all conditions × all demo scales |
| `src/rollout_libero.py` | LIBERO environment rollout with GR-1 + RAID inference |
| `src/grpo_libero.py` | GRPO online fine-tuning loop |
| `src/cache_gr1_features.py` | Pre-compute and cache GR-1 features for the dataset |
| `configs/` | Per-scale norm stats, loss curves, sweep results |
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
    --feature_dir data/libero_spatial/features \
    --model_path  models/raid_visual_50demos_best.pt \
    --device cuda
```

---

## Hyperparameters

| Item | Value |
|------|-------|
| GR-1 feature dim | 384 |
| Retrieval k | 5 |
| Cross-attention heads | 4 |
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
