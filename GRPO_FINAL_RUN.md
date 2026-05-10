# Final GRPO Lambda Run

This note records the final remote RAID + GR-1 + LIBERO work performed on the
Lambda A10 instance and the artifacts preserved in `raid_grpo_final/`.

## What Changed

- Brought the visual RAID pipeline to an end-to-end runnable state on LIBERO
  with EGL rendering.
- Patched GR-1 compatibility issues for the installed `transformers` version:
  `config.n_ctx` -> `config.n_positions`, removed the obsolete
  `tokenizer_class` docstring argument, and sliced timestep embeddings for
  single-step inference.
- Patched RAID visual inference so `DirectMLPVisual` and `RAIDDecoderVisual`
  share the same rollout-facing return shape.
- Rewrote `scripts/compare_video.py` to instantiate the real encoder, memory
  bank, LIBERO environment, and policy path rather than relying on a simplified
  API.
- Added `scripts/visualize_transitions.py`, which renders static transition
  grids from cached features. Each row shows metadata, the current HDF5 frame,
  GR-1's predicted next frame reconstructed from `obs_preds`, and action bars
  for RAID, direct visual, and ground truth.
- Ran GRPO in two phases: a longer full run and a shorter polish run initialized
  from the best full-run checkpoint.

## Offline BC Results

The fresh sweep matched the previous visual RAID result pattern:

| Demo scale | `raid_visual` MSE | `direct_visual` MSE |
|------------|-------------------|---------------------|
| 25 demos | 0.131 | 0.852 |
| 50 demos | 0.158 | 0.639 |
| 100 demos | 0.171 | 0.580 |
| 200 demos | 0.174 | 0.554 |

The retrieval-augmented visual decoder remains much stronger than the direct
visual baseline at every data scale.

## Final GRPO Results

Two GRPO logs were preserved:

| Run | Updates logged | Best SR | Best mean reward | Best checkpoint |
|-----|----------------|---------|------------------|-----------------|
| Full run | 382 | 0.25 | 0.925 at update 294 | `raid_visual_grpo_initial_best.pt` |
| Polish run | 195 | 0.25 | 1.226 at update 158 | `raid_visual_grpo_polish_best_20260509_005631.pt` |

The final policy is not robustly solved, but it crossed from purely shaped
reward improvement into intermittent successful rollouts. The best observed
success rate is 25% with group size 4, and the best polish checkpoint is the
recommended checkpoint to inspect or continue from.

## Preserved Artifact Bundle

`raid_grpo_final/` contains:

| File | Purpose |
|------|---------|
| `grpo_final_summary.json` | Compact summary of the final polish run |
| `grpo_full_stopped_20260508_230003.json` | Full-run update log |
| `grpo_full_stopped_20260508_230003.log` | Full-run console log |
| `grpo_full_stopped_run_visual.png` | Full-run reward/success visualization |
| `grpo_polish_stopped_20260509_005631.json` | Polish-run update log |
| `grpo_polish_stopped_20260509_005631.log` | Polish-run console log |
| `grpo_polish_final_visual.png` | Polish-run reward/success visualization |
| `raid_visual_grpo_initial_best.pt` | Best checkpoint from the full run |
| `raid_visual_grpo_polish_best_20260509_005631.pt` | Best checkpoint from the polish run |
| `grpo_libero_remote_final.py` | Final remote GRPO runner script |
| `rollout_libero_remote_final.py` | Final remote rollout helper snapshot |

Generated transition grids and rollout videos were downloaded from the instance
but intentionally kept out of git, except for the GRPO summary plots above.

## Reproduction Notes

The remote run expected these heavyweight assets on disk:

- LIBERO-Spatial HDF5 data under `data/libero_spatial/libero_spatial/libero_spatial/`
- cached GR-1 features under `data/libero_spatial/features/`
- GR-1 checkpoints under `checkpoints/gr1/`
- BC checkpoints under `models/`
- the external GR-1 checkout with the compatibility patch in
  `third_party_patches/gr1_lambda_compat.patch`

The key command shape was:

```bash
cd ~/RAID
MUJOCO_GL=egl python3 raid_grpo_final/grpo_libero_remote_final.py
```

Use the polish checkpoint for follow-up evaluation or continued online training:

```text
raid_grpo_final/raid_visual_grpo_polish_best_20260509_005631.pt
```
