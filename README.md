# RAID: Retrieval-Augmented Inverse Dynamics for Robotic Manipulation

RAID studies the action-inference step for robot world models: given a current visual state and a dreamed next visual state, what 7-DOF motor command should the robot execute?

We repurpose GR-1 by dropping its language/action-output role, freezing its visual encoder, and using its 384-dimensional class-token feature as the current state `f_t`. GR-1's one-step prediction head supplies the dreamed next state `f_hat_{t+1}`. RAID then decodes `(f_t, f_hat_{t+1})` into a normalized robot action by combining a direct MLP trunk with a cross-attention prior over retrieved demonstrator actions.

[Project page](https://constantinvictorbeatertel.github.io/RAID/) | [Final report](paper/RAID_Report_vf.pdf)

## Headline Result

On LIBERO-Spatial, RAID is strongest in the low-data setting: with only 25 demonstrations, it reaches `0.131` normalized validation MSE versus `0.852` for the direct visual MLP, a roughly `6.5x` improvement.

| Demonstrations | Direct visual MLP | RAID visual | Improvement |
| --- | ---: | ---: | ---: |
| 25 | 0.852 | **0.131** | **6.5x** |
| 50 | 0.637 | **0.154** | 4.1x |
| 100 | 0.570 | **0.169** | 3.4x |
| 200 | 0.552 | **0.171** | 3.2x |

The gap narrows as demonstrations increase, which suggests RAID is primarily a sample-efficiency mechanism rather than a guaranteed asymptotic improvement. Retrieval helps most when the parametric decoder has too little data to learn reliable action mappings; at larger data scales, the direct model benefits more from coverage while RAID can inherit bias from imperfect nearest-neighbor matches.

## Method

RAID stores demonstrated transitions in a memory bank:

```text
M = {(concat(f_i, f_{i+1}), a_i)}
```

At inference time, GR-1 and RAID compute:

```text
f_t            = Enc_GR1(s_t)
f_hat_{t+1}   = g_GR1(f_t)
R_k            = Ret(M, concat(f_t, f_hat_{t+1}))
a_hat_t        = d_phi(f_t, f_hat_{t+1}, R_k)
```

The RAID head has three parts:

| Component | Role |
| --- | --- |
| Direct trunk | Two-hidden-layer MLP over `concat(f_t, f_hat_{t+1})` |
| Cross-attention prior | Retrieves top-`k=3` similar transitions and builds an action prior from their actions |
| Per-dimension gate | Blends the direct estimate and retrieval prior separately for each action dimension |

The predicted action is normalized 7-DOF control: `(dx, dy, dz, dtheta_x, dtheta_y, dtheta_z, grip)`.

## GRPO Probe

Starting from the `N=200` behavior-cloned RAID checkpoint, GRPO improved closed-loop shaped reward but did not produce a stable solved policy. The run logged 195 updates, reached best mean reward `1.226` at update 158, and achieved a peak group success rate of `25%` on some updates. This shows the RAID prior can be refined through simulator interaction, but sparse manipulation success remains unreliable without more stable online training.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `src/gr1_encoder.py` | Frozen GR-1 feature encoder and one-step feature prediction wrapper |
| `src/data_libero.py` | LIBERO HDF5 loading, action normalization, and demo-level splits |
| `src/memory_libero.py` | Dense cosine-similarity memory bank retrieval |
| `src/models_libero.py` | Direct visual MLP and RAID visual decoder |
| `src/train_libero.py` | Training loop for `direct_visual` and `raid_visual` |
| `src/run_all_libero.py` | LIBERO sweep driver across demo scales |
| `src/rollout_libero.py` | Closed-loop LIBERO rollout evaluation |
| `src/grpo_libero.py` | GRPO online fine-tuning probe |
| `configs/results_libero.json` | Main GR-1 + RAID validation results |
| `configs/loss_curves_*_libero.json` | Per-epoch train/validation curves for the appendix |
| `paper/RAID_Report_vf.pdf` | Final project report |

Earlier RoboMimic and V-JEPA/DINO/SigLIP exploration code is preserved in `src/data.py`, `src/models.py`, `src/train.py`, `src/run_all.py`, and related `configs/` files.

## Reproducing

Cache GR-1 features once:

```bash
python src/cache_gr1_features.py \
  --dataset_dir data/libero_spatial/libero_spatial \
  --output_dir data/libero_spatial/features \
  --device cuda
```

Run the LIBERO behavior-cloning sweep:

```bash
python src/run_all_libero.py \
  --feature_dir data/libero_spatial/features \
  --device cuda
```

Run the GRPO fine-tuning probe:

```bash
python src/grpo_libero.py \
  --feature_dir data/libero_spatial/features \
  --model_path models/raid_visual_200demos_libero_best.pt \
  --device cuda
```

## Setup Notes

- Python 3.10+
- PyTorch with CUDA recommended
- `h5py`, `numpy`, `tqdm`
- LIBERO simulator package for rollout and GRPO experiments
- Public GR-1 checkpoint and MAE ViT-base weights from [bytedance/GR-1](https://github.com/bytedance/GR-1)

Large datasets, generated feature caches, and newly trained checkpoints are intentionally ignored by git; regenerate them with the scripts above.
