# Proteus v6 — Visuo-Tactile Behavior Cloning (VT-BC)

Behavior cloning with a **frozen CLIP ViT-B/32** visual backbone (512-d embeddings cached on disk), a **learned tactile MLP** encoder (768 → 256 → 64), and a small **causal Transformer** (3 layers, 8 heads, dropout 0.1) predicting normalized **Δ end-effector** actions (7-DOF: XYZ, axis-angle XYZ, gripper delta).

Dataset: Hugging Face / local **Touch in the Wild — four_tasks** (`fluid_transfer`, `test_tube_collection`, `pencil_insertion`, `whiteboard_erasing`), each `{task}/{task}.zarr.zip`.

**No** robosuite, robomimic, or MuJoCo imports.

---

## Experimental conditions

| Condition        | Inputs used                          |
|----------------|---------------------------------------|
| `vision_only`  | Frozen CLIP embeddings only           |
| `tactile_only` | Normalized tactile grid only           |
| `visuo_tactile`| CLIP embeddings + tactile (primary)    |

Training optimizes single **MSE** on normalized actions (computed from consecutive `robot0_eef_*` samples).  

**Splits:** first **80% of episodes per task → train**, last **20% → validation** (no episode leakage).

---

## Contact-timestep evaluation

Normalized tactile grids `(12 × 64)` are min–max scaled from the **training** split.  
A timestep is **contact** if its spatial **max activation** exceeds a **threshold** calibrated on train data:  

**10th percentile** of timestep max-values among readings with max &gt; `1e-6` (fallback: percentile over all reads if almost no nonzero). Threshold is saved to `configs/contact_threshold.pt`.  

Evaluation reports **overall**, **contact**, and **non-contact** validation MSE (same normalized action space).

---

## Project layout (`/home/ubuntu/proteus/vtbc`)

| Path | Role |
|------|------|
| `src/data.py` | Zarr loaders, Δ-action derivation, normalization (`configs/norm_stats.pt`), `VTBCDataset`, `VTBCWindowDataset` (horizon **10**) |
| `src/encoders.py` | `FrozenCLIPEncoder`, `TactileEncoder` |
| `src/policy.py` | `VisuoTactilePolicy`, `VisionOnlyPolicy`, `TactileOnlyPolicy`, causal Transformer |
| `src/cache_clip.py` | One-time CLIP embedding cache per task timestep → `data/clip_cache/{task}_{train,val}.pt` |
| `src/train.py` | AdamW (`lr=3e-4`, `wd=0.1`), cosine schedule, **50** epochs, batch **64**, grad clip **1.0**; checkpoints `models/{condition}_best.pt`, logs `configs/{condition}_losses.json` |
| `src/evaluate.py` | Loads best checkpoints + calibrates threshold; writes `configs/results.json` |
| `src/run_all.py` | Cache (if missing) → train three conditions → evaluate |
| `notebooks/02_results.py` | Figures under `notebooks/figures/` |
| `requirements.txt` | Python dependencies |

Caches and checkpoints stay under `data/clip_cache/`, `models/`, `configs/` (omit from git if desired).

---

## Environment

```bash
cd /home/ubuntu/proteus/vtbc
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If your system Torch was built against NumPy 1.x, use `numpy>=1.24,<2` (already constrained in `requirements.txt`) inside the venv to avoid ABI warnings.

Workers: optional `export VTBC_NUM_WORKERS=0` if DataLoader multiprocessing fails on your filesystem.

Random seed **42** is fixed in scripts.

---

## Full pipeline — exact commands (in order)

```bash
cd /home/ubuntu/proteus/vtbc
source .venv/bin/activate   # optional

pip install -r requirements.txt

# 1) One-time CLIP cache (skipped by run_all if files already exist)
python src/cache_clip.py

# 2) Train all three ablations (~50 epochs each)
python src/train.py --condition vision_only
python src/train.py --condition tactile_only
python src/train.py --condition visuo_tactile

# 3) Evaluate + calibrated contact breakdown
python src/evaluate.py

# 4) Figures from logs + configs/results.json
python notebooks/02_results.py
```

**One-shot orchestration (recommended):**

```bash
cd /home/ubuntu/proteus/vtbc
python src/run_all.py
python notebooks/02_results.py
```

---

## Reproducing report figures (`notebooks/figures/`)

Requires completed training + evaluation.

| Figure | Produced by |
|--------|--------------|
| `loss_curves.png` | Loads `configs/*_losses.json` |
| `mse_comparison.png` | Loads `configs/results.json` (`mse_overall`) |
| `contact_vs_noncontact.png` | Loads `mse_contact`, `mse_noncontact` |
| `per_dof_mse.png` | Compares **vision_only** vs **visuo_tactile** per DOF |
| `tactile_heatmap.png` | Reads raw `fluid_transfer` zarr; uses `configs/contact_threshold.pt` if present |

---

## Sanity: inspect data only

```bash
python src/data.py
python src/policy.py
python src/encoders.py
```
