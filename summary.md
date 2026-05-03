# RAID project — complete session summary

Consolidated narrative of how the **Proteus RAID** (Retrieval-Augmented Inverse Dynamics) codebase on RoboMimic **Lift** (low-dim) was built, run, documented, versioned, and improved. For autoresearch-only detail, see [`AUTORESEARCH_SUMMARY.md`](AUTORESEARCH_SUMMARY.md); for per-iteration logs, see [`configs/autoresearch_log.md`](configs/autoresearch_log.md).

---

## 1. What this project is

We implemented a **controlled comparison** of inverse-dynamics models:

- **Direct MLP:** predicts normalized action from \((s_t, s_{next})\) only.
- **RAID:** uses a **memory bank** of train-only transitions to build a **pooled prior** over \(k=3\) retrieved actions, then a **decoder** combines the transition with that prior.
- **Evaluation baselines:** zero action (mean baseline in normalized space), and **kNN pooled** prediction (no learned decoder).

Training uses **MSE on normalized actions**, **50 epochs**, **AdamW**, **batch 256**, seed **42**. Splits are **80/20 by demonstration** at scales **25 / 50 / 100 / 200** demos.

---

## 2. What we built first (initial codebase)

These pieces were implemented or completed so the pipeline is end-to-end reproducible:

| Area | Role |
|------|------|
| [`src/data.py`](src/data.py) | HDF5 loading, transitions, normalization, `norm_stats_{N}demos.pt`, `make_train_val`, `TransitionDataset`. |
| [`src/memory.py`](src/memory.py) | `RAIDMemoryBank`, batch retrieval, mask handling, populate from dataset. |
| [`src/models.py`](src/models.py) | `DirectMLP`, `RAIDDecoder` (evolved further during autoresearch). |
| [`src/train.py`](src/train.py) | CLI `--condition direct_mlp\|raid`, `--n_demos`, checkpoint best val, JSON loss curves, RAID excludes self on train retrieval. |
| [`src/evaluate.py`](src/evaluate.py) | Was a stub; replaced with **full eval**: four conditions on val loader, train-only bank per scale, metrics (MSE / contact / non-contact / per-DOF), retrieval hit rate, [`configs/results.json`](configs/results.json), printed table. |
| [`src/run_all.py`](src/run_all.py) | Sequential `train` for all `{direct_mlp, raid} × scales`, then `evaluate.py`, `cwd=repo root`, `sys.executable`. |
| [`notebooks/02_results.py`](notebooks/02_results.py) | Reads `results.json` + loss JSONs → **`mse_scaling.png`**, **`contact_mse_scaling.png`**, **`val_loss_by_scale.png`**, **`retrieval_hit_rate.png`**. |
| [`notebooks/01_eda.py`](notebooks/01_eda.py) | Existing EDA for the HDF5 (distributions, episode lengths). |
| [`configs/v6.yaml`](configs/v6.yaml) | Reference hyperparameters (not parsed at runtime by training). |
| [`configs/README.md`](configs/README.md) | Short pointer to artifact naming. |
| [`README.md`](README.md) | User-facing overview, conditions, scaling, how to run, file map, figure reproduction (updated over time conceptually before autoresearch changed the RAID block). |

**Fixes along the way:** `evaluate.py` was non-functional placeholders; it was rewritten. Tooling friction (truncated writes / bad patches) required careful restores; **`src/memory.py`** was verified complete on disk. **Git:** first commit needed **local `user.name` / `user.email`**; **HTTPS push** lacked credentials → **SSH remote** (`git@github.com:…`) and **`ssh -T git@github.com`** confirmed, then **`git push`**.

**Hygiene:** [`.gitignore`](.gitignore) for bytecode, venvs, IDE crumbs; **`README.md`** + **`.gitignore`** committed separately earlier in the timeline.

---

## 3. First full experiment results (baseline RAID)

After wiring everything, **`python3 src/run_all.py`** and **`python3 notebooks/02_results.py`** produced:

**Observed qualitative issue:** the original **concat** `RAIDDecoder` tended to **underperform the direct MLP** at most scales (the decoder effectively leaned on a noisy pooled prior). Example **validation MSE** (normalized actions) from an earlier table in project notes:

| Condition | 25 | 50 | 100 | 200 |
|-----------|-----|-----|------|------|
| Direct MLP | ~0.336 | ~0.358 | ~0.296 | ~0.183 |
| RAID (old) | ~0.444 | ~0.512 | ~0.536 | ~0.424 |

That motivated the **`program.md`** autoresearch directive and the structured search over **`RAIDDecoder`** only.

---

## 4. Autoresearch and architecture improvements

We added **`program.md`** describing an agent loop (hypotheses, train @25 demos, keep/revert, log, eventual full sweep).

**Eight iterations** (see [`configs/autoresearch_log.md`](configs/autoresearch_log.md)) tested: residual heads, detached prior concat, **learned gating**, separate encoders, scaled prior / transition-only trunk, prior dropout, prior noise, and a wider trunk. **Snapshot files** under `configs/autoresearch_*.py` preserved intermediate architectures when reverting.

**Outcome:**

- Best **training** RAID val MSE @25 demos: **≈0.397** (**0.396789**).
- **Large gain vs old RAID** (~0.44 → ~0.40).
- Still **below direct MLP** on the same metric (~**0.336** @25).

**Accepted design:** **sigmoid gate** blending a **transition-only inverse branch** with the **pooled prior**, plus **dropout on the prior path** and **Gaussian noise on the prior during training** (see current [`src/models.py`](src/models.py)).

We re-ran **`python3 src/run_all.py`**, refreshed figures, and pushed **`autoresearch: best RAID val_mse=0.397 after 8 iterations`**.

Summary doc spun out as [`AUTORESEARCH_SUMMARY.md`](AUTORESEARCH_SUMMARY.md).

---

## 5. Current evaluation snapshot (after autoresearch re-sweep)

From [`configs/results.json`](configs/results.json) (figures in `notebooks/figures/`):

| Scale | Direct MLP MSE | RAID MSE |
|-------|----------------|-----------|
| 25 | ~0.336 | ~0.397 |
| 50 | ~0.358 | ~0.398 |
| 100 | ~0.296 | ~(see JSON) |
| 200 | ~0.183 | ~(see JSON) |

Raid **contact-phase** error can be competitive on some scales; overall MSE still favors the direct baseline in this setup.

---

## 6. How to reproduce everything

```bash
cd /path/to/raid

# Full training + evaluation (all conditions × scales); ~GPU minutes on a small box
python3 src/run_all.py

# Figures
python3 notebooks/02_results.py

# RAID-only quick iterate (hypothesis testing)
python3 src/train.py --condition raid --n_demos 25
```

Artifacts: **`models/`**, **`configs/loss_curves_*.json`**, **`configs/norm_stats_*demos.pt`**, **`configs/results.json`**, **`notebooks/figures/`**.

---

## 7. Git / GitHub timeline (high level)

1. **`git init`**, first commit **`RAID v6 initial build and results`** (code + data + checkpoints + configs as tracked at the time).
2. Push issues: **identity** then **SSH** + force push to **`ConstantinVictorBeatErtel/Proteus`**.
3. Later commits: **README + .gitignore**, **figures/results** (already present in some cases), **`program.md`**, **autoresearch commit**, **`AUTORESEARCH_SUMMARY.md`**, and this **`summary.md`**.

---

## 8. Files that are “paper trail” for this work

| File | Purpose |
|------|---------|
| [`README.md`](README.md) | Science + usage (project overview). |
| [`summary.md`](summary.md) | This file — end-to-end session chronicle. |
| [`AUTORESEARCH_SUMMARY.md`](AUTORESEARCH_SUMMARY.md) | Autoresearch digest. |
| [`program.md`](program.md) | Original autoresearch instruction spec. |
| [`configs/autoresearch_log.md`](configs/autoresearch_log.md) | Iteration-by-iteration results. |
| [`configs/autoresearch_*_models.py`](configs/) | Frozen architecture snapshots. |

---

## 9. Open points / honest limitations

- **RAID still loses to direct MLP** on aggregate val MSE under this recipe; further gains may need **training protocol** (e.g. loss weighting, schedule), **memory / retrieval**, or **stronger inductive bias** — outside the “edit `RAIDDecoder` only” autoresearch box.
- **[`README.md`](README.md)** still describes RAID in a slightly more generic “residual” story; the **implemented** decoder is the **gated + regularized prior** variant after autoresearch.
- Large binaries (**HDF5**, **`.pt`**) are in the repo history; consider **Git LFS** or **download scripts** for future clones.

This document is meant as a **single entry point** for “what happened in this RAID project,” from first code to last push.
