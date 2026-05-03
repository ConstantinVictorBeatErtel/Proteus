# RAID autoresearch — what we did

This document summarizes the **autonomous architecture search** performed on the RAID (Retrieval-Augmented Inverse Dynamics) decoder for RoboMimic **Lift** (low-dim), following [`program.md`](program.md) and the eight-iteration plan in the repo history.

## Goal

- Improve RAID so that **training validation MSE** at **`n_demos=25`** beats the **direct MLP** (~**0.336**) and the old RAID baseline (~**0.444**).
- Metric each iteration:  
  `python3 src/train.py --condition raid --n_demos 25`  
  then read **`[train] best checkpoint val_mse=...`**.

## Outcome

| Model (25 demos, train val checkpoint) | Approx. best val MSE |
|----------------------------------------|----------------------|
| Direct MLP (unchanged baseline)        | **~0.336**           |
| RAID (original concat decoder)         | **~0.444**           |
| **RAID after autoresearch (final)**    | **~0.397** (0.396789) |

RAID was **materially improved** (0.44 → 0.40) but **did not surpass** the direct MLP at the autoresearch metric.

## Winning architecture

The accepted `RAIDDecoder` in [`src/models.py`](src/models.py):

1. **Learned gate** per action dimension from the transition `[s_t, s_{t+1}]`: blend parametric inverse vs pooled retrieval prior (`g * direct(trans) + (1-g) * prior`).
2. **Dropout on the prior path** (`p=0.5`) during training so the model cannot blindly copy retrieval.
3. **Gaussian noise** on the prior during training (`σ=0.1`) for additional regularisation.

Iterations that **hurt** validation (pure residual-only head, disjoint encoders, scalar-scaled prior + transition-only trunk, blindly wider trunk) were **reverted**.

## Eight iterations (abbreviated)

| # | Idea | Trial val\_mse @25 | Decision |
|---|------|-------------------|-----------|
| 1 | Residual `a_prior + δ` | 0.622 | Reverted |
| 2 | Detached prior in concat MLP | 0.444 | Kept tie; later superseded |
| 3 | Gated blend | **0.431** | Kept |
| 4 | Separate transition / prior encoders | 0.483 | Reverted |
| 5 | `scale * prior + MLP(trans)` | 0.446 | Reverted |
| 6 | Prior dropout in gated RAID | **0.399** | Kept |
| 7 | + prior noise at train time | **0.397** | Kept (best) |
| 8 | 2× wider `direct` trunk | 0.410 | Reverted |

Detailed per-iteration notes: [`configs/autoresearch_log.md`](configs/autoresearch_log.md).

## Artifacts and hygiene

- **Iteration logs:** [`configs/autoresearch_log.md`](configs/autoresearch_log.md)
- **Architecture snapshots (for debugging / revert):**  
  `configs/autoresearch_iter3_gate_models.py`, `configs/autoresearch_iter6_kept_models.py`, `configs/autoresearch_iter7_kept_models.py`
- **Checkpoints:** `models/raid_{25,50,100,200}demos_best.pt` (RAID refreshed after full sweep).
- **Metrics table:** [`configs/results.json`](configs/results.json)
- **Figures:** regenerated with `python3 notebooks/02_results.py` under `notebooks/figures/`
- **Full retrain + eval sweep:** `python3 src/run_all.py`

## Constraints we followed

- Edits limited to **`src/models.py`** (and temporary snapshot files under `configs/`); **`src/train.py`** was not changed for interface.
- Data pipeline, memory bank, and **`src/evaluate.py`** unchanged.
- Each training run completed fully; failures were logged and architectures reverted from saved snapshots where applicable.

## Git

Changes were committed and pushed to **`main`** with message:

`autoresearch: best RAID val_mse=0.397 after 8 iterations`

This summary file is a high-level narrative only; the authoritative step-by-step log remains [`configs/autoresearch_log.md`](configs/autoresearch_log.md).
