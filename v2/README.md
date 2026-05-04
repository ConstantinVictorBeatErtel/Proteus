# v2: multi-dataset image-feature inverse-dynamics expansion

Additive subtree on top of the legacy `src/` package. Nothing under
`src/`, `configs/`, or `notebooks/` is modified; the legacy autoresearch
baseline (`python3 src/train.py --condition raid --n_demos 25` →
`val_mse ≈ 0.397`) keeps reproducing exactly.

## What's here

```
v2/
  runtime/              # drive mount, atomic checkpoints, W&B-resume,
                        # idempotent dataset downloaders
  datasets/             # RoboMimic, LIBERO, mixed-dataset adapters
                        # + per-dataset q01/q99 normalization stats
  legacy/               # forks of src/memory.py + src/models.py with
                        # feature-vector generalizations (originals untouched)
  heads/                # TransformerIDM and DiffusionPolicyIDM
  visualize.py          # render obs_t / obs_{t+1} + predicted vs GT action panels
  features.py           # cache DINOv2 / Theia CLS features as safetensors on Drive
  train.py              # config-driven trainer with W&B resume="allow"
  evaluate.py           # eval that emits prediction PNGs
  run_matrix.py         # idempotent matrix orchestrator
  configs/matrix.yaml   # experimental matrix (phases A-F)
  tests/test_smoke.py   # heads + visualize smoke tests
  notebooks/            # aggregation and figure scripts
```

## Setup (Colab)

```python
!pip install -q -r v2/requirements.txt
import os
os.environ.setdefault("RAID_ARTIFACT_ROOT", "/content/drive/MyDrive/raid_v2")

from v2.runtime.drive import mount_drive, artifact_root
mount_drive()
print("artifacts at:", artifact_root())

from v2.runtime.data_download import ensure_all_data
ensure_all_data()
```

Subsequent sessions skip the download because each adapter checks Drive
presence first.

## Cache encoder features once

```bash
python3 -m v2.features --encoders dinov2 theia
```

Writes one safetensors per (dataset, encoder) under
`<artifact_root>/features/`. Re-running is a no-op.

## Run the experimental matrix

```bash
# Enumerate without launching:
python3 -m v2.run_matrix --phase A --dry-run

# Run a phase end-to-end (resumable on disconnect):
python3 -m v2.run_matrix --phase A
python3 -m v2.run_matrix --phase C
```

Each cell is keyed by a deterministic SHA1 of
`(phase, head, dataset, encoder, n_demos, seed)`; checkpoints land in
`<artifact_root>/runs/<run_id>/` with rolling-last + best.

## Visualize what the model is doing

After training a cell:

```bash
python3 -m v2.evaluate --run-id <RUN_ID> --n-panels 12
```

Writes per-transition panels to
`<artifact_root>/results/figures/predictions/<run_id>/{i}.png` plus a
single grid figure. Each panel shows the obs_t and obs_{t+1} frames (RGB
when available) side-by-side with a bar chart comparing the model's
predicted 7-D action to the ground-truth action that produced the
transition.

For a quick dataset-level preview without training, the feature
extraction pass writes a 9-frame `*.preview.png` next to each
safetensors file showing what the encoder ingested.

## Smoke tests

```bash
pip install pytest
pytest v2/tests/
```

The four head smoke tests + the visualize panel test all run on CPU in
under a minute and gate the implementation against trivially-broken
heads.
