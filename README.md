# RAID — Retrieval-Augmented Inverse Dynamics (Lift baseline)

This repository implements an inverse-dynamics experiment on RoboMimic **Lift** (low-dimensional observations) comparing a **direct** multi-layer perceptron with a **RAID-style** decoder that conditions on retrieved training actions given the current transition \((s_t, s_{t+1})\).

## Motivation: action inference under limited data

In contact-rich manipulation, the mapping from observable state to feasible torques/commands is subtle: small kinematic cues co-occur with different contact modes, friction, and task intent that are never fully observable in demos alone. Robotics researchers—including **Ken Goldberg** and collaborators in work on robotics data scarcity, manipulation reliability, and the limits of imitation from video or simulation—stress that scaling data is necessary but often **not sufficient**: there is effectively a bottleneck in **inferring physically appropriate actions** from partial trajectories (“what control actually produced \((s_t \rightarrow s_{t+1})\) under contacts I cannot see cleanly?”).

**RAID** (retrieval-augmented inverse dynamics) addresses that bottleneck with a pragmatic hybrid: reuse **nearest-neighbour actions** from memory as a coarse prior \(\hat{a}\), then learn a neural **residual decoder** \(\Delta(s_t,s_{t+1},\hat{a})\) so the final prediction is anchored in real executed controls from similar past transitions rather than interpolated only from scalar losses on a finite training set.

## Experimental conditions

| Condition | Prediction | Role |
|-----------|--------------|------|
| **Mean baseline** | Zero vector in normalized action space | Trivial baseline (normalized targets are centred but not trivial at test time relative to variability). |
| **Nearest neighbour (kNN)** | Mean of top-\(k\) retrieved training actions weighted by retrieval mask | Retrieval-only inverse dynamics; exposes how far memory alone explains the validation split. |
| **Direct MLP** | \(f_\theta(s_t, s_{t+1})\) | Pure parametric inverse model (no retrieval). |
| **RAID** | \( \hat{a}_{\mathrm{prior}} + g_\phi(s_t, s_{t+1}, \hat{a}_{\mathrm{prior}}) \) — here implemented as concatenation-conditioned MLP on \((s_t,s_{t+1},\hat{a})\) | Learn to correct pooled retrieval outputs. |

Evaluations additionally report **contact** vs **non-contact** subsets (contact inferred from successive gripper command changes) and **per-DOF** MSE plus **hit rate** for retrieval (fraction of minibatches where at least one neighbour matches within the retrieval mask).

## Data scaling design

Train/validation splits are **by demonstration** (\(80\%\) train / \(20\%\) val — see `make_train_val` in `src/data.py`). The same architecture and hyperparameters are trained with **four dataset sizes**:

| Demonstrations \(N\) | Purpose |
|----------------------|---------|
| 25 | Small-data regime |
| 50 | |
| 100 | |
| 200 | Larger-data scaling point |

Normalization statistics are saved per scale as `configs/norm_stats_{N}demos.pt` so each experiment uses statistics from its own train split only.

## How to run the full experiment

From the repository root (this directory):

```bash
python src/run_all.py
```

This sequentially trains **`direct_mlp`** and **`raid`** for \(N \in \{25,50,100,200\}\), then runs `src/evaluate.py` across all scales. Use `python3` if `python` is not on your PATH.

**One-off training** (writes `configs/loss_curves_{condition}_{N}demos.json` and `models/{condition}_{N}demos_best.pt`):

```bash
python src/train.py --condition direct_mlp --n_demos 100
python src/train.py --condition raid           --n_demos 100
```

**Evaluation only** (requires matching checkpoints):

```bash
python src/evaluate.py
# optional: python src/evaluate.py --tau-min 0.0
```

Results are merged into **`configs/results.json`** and a summary table is printed to the terminal.

## Repository layout — what each core file does

| Path | Role |
|------|------|
| `src/data.py` | HDF5 loader; builds \((s_t,s_{t+1},a_t,\texttt{contact})\); normalizes states/actions; saves per-scale stats; `TransitionDataset` + `make_train_val`. |
| `src/memory.py` | `RAIDMemoryBank`: store train transitions/features, batch \(k\)-nearest retrieval, optional \(\tau_{\min}\) gating. |
| `src/models.py` | `DirectMLP` and `RAIDDecoder` (two-layer MLP with LayerNorm/ReLU/dropout as used in training). |
| `src/train.py` | fifty-epoch optimization; best-val checkpointing; RAID uses train-only bank and **excludes self** at retrieval indices during training. |
| `src/evaluate.py` | Loads best checkpoints per scale; builds train-only bank per scale; aggregates MSE breakdowns + hit rate; writes `configs/results.json`. |
| `src/run_all.py` | Sweep driver: all train jobs then evaluate. |
| `notebooks/01_eda.py` | Dataset inspection and static EDA plots. |
| `notebooks/02_results.py` | Paper-style plots from JSON outputs (see below). |
| `configs/v6.yaml` | Human-readable reference hyperparameters/paths (**not** read by train/eval CLI). |
| `configs/README.md` | Artifact naming patterns for configs/outputs. |
| `data/lift/ph/low_dim_v141.hdf5` | RoboMimic Lift low-dim expert demonstrations (if present in checkout). |

## Reproducing all figures

1. Produce metrics and curves (training + evaluation):

   ```bash
   python src/run_all.py
   ```

2. Render figures (**requires prior `configs/results.json` and loss JSONs from training**):

   ```bash
   python notebooks/02_results.py
   ```

Artifacts are written to **`notebooks/figures/`**:

| Generated file | Content |
|----------------|---------|
| `mse_scaling.png` | Overall validation MSE vs \(N\) for all four evaluation conditions |
| `contact_mse_scaling.png` | Same for contact-heavy timesteps |
| `val_loss_by_scale.png` | Validation loss curves: `direct_mlp` vs `raid`, coloured by \(N\) |
| `retrieval_hit_rate.png` | Bar chart of retrieval hit fraction vs \(N\) |

`notebooks/01_eda.py` optionally generates `notebooks/figures/action_distributions.png` and `episode_lengths.png` from the HDF5 file.

## Hyperparameters (reference)

These match the scripted defaults (`src/train.py` / `configs/v6.yaml`). All values are illustrative of this baseline; tweak in code/YAML references.

| Item | Value |
|------|--------|
| Optimizer | AdamW, \(\mathrm{lr}=10^{-3}\), weight decay \(10^{-4}\) |
| Epochs | 50 |
| Batch size | 256 |
| Hidden width / layers | 256, two hidden layers (+ LayerNorm/ReLU/dropout 0.1) |
| RAID \(k\) retrieval | 3 |
| Memory capacity | \(5\times10^4\) entries |
| Random seed | 42 |

## Dependencies

Approximate runtime stack:

- Python 3.10+
- PyTorch
- NumPy
- `h5py`
- `tqdm`
- Matplotlib (`notebooks/` scripts)

CUDA is used automatically when available (`train.py`, `evaluate.py`).

---

## Visual RAID with V-JEPA 2

This extension replaces low-dimensional proprioception with **frozen V-JEPA 2 ViT-L**
visual features (1024-dim) on the **LIBERO-Spatial** benchmark — 10 manipulation tasks
with 50 demonstrations each. All conditions share the same cached encoder features,
enabling a **fair ablation** of decoder architectures and retrieval mechanisms.

### Scientific framing

DreamZero (NVIDIA, arXiv 2602.15922) validates that world models decompose into
(1) video prediction and (2) inverse dynamics model (IDM). RAID is a retrieval-augmented
IDM that replaces expensive CEM search with a single decoder forward pass. V-JEPA 2
predicts in latent embedding space (true JEPA-style), unlike pixel-space autoregressive
models such as GR-1.

### Conditions (all use identical frozen 1024-dim features)

| Condition | Model | Retrieval | Fusion |
|-----------|-------|-----------|--------|
| `mean_action` | None | No | Train mean |
| `nn_copy` | None | Yes (top-1) | Copy neighbour |
| `direct_mlp` | 3-layer MLP | No | Concat(feat_t, feat_next) |
| `concat_mlp` | 3-layer MLP | Yes (k=5) | Concat + mean-pool |
| `raid_xattn` | Cross-attention decoder | Yes (k=5) | Learned attention |

### Execution order

**Step 1 — Test encoder:**
```bash
python src/vjepa_encoder.py
```
Expected output: `torch.Size([4, 1024])`

**Step 2 — Dry run to verify pipeline:**
```bash
python src/cache_vjepa_features.py \
    --dataset_dir /home/ubuntu/RAID/data/libero_spatial/libero_spatial \
    --output_dir /home/ubuntu/RAID/data/libero_spatial/vjepa_features \
    --device cuda --dry_run
```

**Step 3 — Cache features (run once, ~30-45 min):**
```bash
python src/cache_vjepa_features.py \
    --dataset_dir /home/ubuntu/RAID/data/libero_spatial/libero_spatial \
    --output_dir /home/ubuntu/RAID/data/libero_spatial/vjepa_features \
    --device cuda
```

**Step 4 — Fair comparison sweep (~2-4 hours):**
```bash
python src/run_all_libero.py \
    --feature_dir /home/ubuntu/RAID/data/libero_spatial/vjepa_features \
    --device cuda
```

**Step 5 — Autoresearch loop (~4-8 hours, run in tmux):**
```bash
tmux new -s autoreach
python src/autoresearch_libero.py \
    --feature_dir /home/ubuntu/RAID/data/libero_spatial/vjepa_features \
    --n_iter 9 --device cuda
```

### Artifacts produced

| Path | Content |
|------|---------|
| `data/libero_spatial/vjepa_features/*.pt` | Cached V-JEPA 2 features (one per task) |
| `configs/norm_stats_{N}demos_vjepa.pt` | Per-scale action normalisation stats |
| `configs/results_vjepa.json` | Full sweep results (5 conditions × 4 scales) |
| `configs/autoresearch_log.json` | All autoresearch iterations with val_mse |
| `models/{condition}_{N}demos_vjepa_best.pt` | Best checkpoint per condition/scale |

### Critical fairness constraint

ALL five conditions use IDENTICAL frozen V-JEPA 2 features as input (`feat_dim=1024`).
The only thing that varies between conditions is the decoder architecture and whether
retrieval is used. No condition trains or fine-tunes the encoder.

### New source files

| File | Role |
|------|------|
| `src/vjepa_encoder.py` | Frozen V-JEPA 2 ViT-L encoder wrapper |
| `src/cache_vjepa_features.py` | Feature caching script |
| `src/data_libero.py` | LIBERO data loaders + `CachedVJEPADataset` |
| `src/memory_libero.py` | `VJEPAMemoryBank` for cosine retrieval |
| `src/models_libero.py` | `DirectMLPVisual`, `ConcatMLPVisual`, `RAIDDecoderVisual` |
| `src/train_libero.py` | Training loop for all 5 conditions |
| `src/run_all_libero.py` | Sweep driver |
| `src/autoresearch_libero.py` | Self-improving architecture search |
| `program.md` | Full protocol documentation |
