# RAID experiment configs

- **`v6.yaml`** — hyperparameters & paths for the Lift low-dim pipeline (`train.py` / `evaluate.py`). CLI flags remain the source of truth for runnable scripts; YAML documents the nominal settings used in this milestone.

Artifacts produced by scripts:

| File pattern | Produced by |
|-------------|--------------|
| `norm_stats_{n}demos.pt` | Dataset helper when training |
| `loss_curves_{condition}_{n}demos.json` | `src/train.py` |
| `{condition}_{n}demos_best.pt` under `models/` | `src/train.py` |
| **`results.json`** | `src/evaluate.py` (all conditions & scales in one nested JSON) |

Run the full sweep:

```bash
cd raid
python3 src/run_all.py
python3 notebooks/02_results.py
```

Figures are written to `notebooks/figures/` (see notebook script for filenames).
