# RAID Visual — Retrieval-Augmented Inverse Dynamics with V-JEPA 2

## What RAID is and why retrieval helps

RAID (Retrieval-Augmented Inverse Dynamics) learns a mapping from state transitions
(s_t, s_{t+1}) to actions a_t by querying a memory bank of past transitions. Instead
of learning a pure parametric mapping (which overfits at low data), RAID anchors each
prediction in real executed actions from similar past transitions.

The scientific framing: DreamZero (NVIDIA, arXiv 2602.15922) validates that world
models decompose into (1) video prediction and (2) inverse dynamics model (IDM).
RAID is a retrieval-augmented IDM that replaces expensive CEM search with a single
decoder forward pass. V-JEPA 2 is a true JEPA-style world model — it predicts in
latent embedding space, not pixels — unlike GR-1 which is pixel-space autoregressive.

## The 5 conditions and why they form a fair ablation

All five conditions use IDENTICAL frozen V-JEPA 2 ViT-L features (feat_dim=1024).
The only thing that varies is the decoder architecture and whether retrieval is used.

1. **mean_action**: Predict the mean of all training actions. Trivial baseline —
   measures how much variance there is in the action space.

2. **nn_copy**: For each validation transition, retrieve the single most similar
   train transition and copy its action. Measures how far raw retrieval goes
   without any learned model.

3. **direct_mlp**: A 3-layer MLP on concat(feat_t, feat_next). The pure parametric
   inverse dynamics model — strong at high data, weak at low data.

4. **concat_mlp**: Same MLP architecture but with concat(feat_t, feat_next,
   pooled_retrieved_actions) as input. Retrieval-augmented, but with a simple
   fusion (mean-pool). This is the FAIR comparison point — any advantage of
   cross-attention in condition 5 must come from the attention mechanism, not
   from retrieval or features.

5. **raid_xattn**: Full RAID with multi-head cross-attention. The transition
   features form a query; retrieved actions form keys/values; attention-learned
   weights produce the output. This is the proposed architecture.

**Expected result**: raid_xattn < concat_mlp <= direct_mlp at low data.
At high data, all three learned conditions converge as data dominates retrieval.

## Autoresearch hypotheses

Karpathy-style self-improving agent loop. Each iteration changes ONE thing:

| Iter | Name           | Change                                     |
|------|----------------|--------------------------------------------|
| 0    | baseline       | RAIDDecoderVisual as specified (k=5, h=512, heads=8) |
| 1    | h1_k10         | Increase k to 10                           |
| 2    | h2_k3          | Decrease k to 3                            |
| 3    | h3_2xattn      | Two cross-attention layers                 |
| 4    | h4_posenc      | Sort by similarity + learned position enc. |
| 5    | h5_feat_t_only | Query on feat_t only (not feat_t||feat_next) |
| 6    | h6_gate_blend  | Sigmoid gate blending xattn with mean-pool |
| 7    | h7_hidden1024  | Hidden dim 1024 instead of 512             |
| 8    | h8_noise005    | Gaussian noise std=0.05 on retrieved actions |

## How to run everything end to end

### Step 1 — Test encoder
```bash
python src/vjepa_encoder.py
```

### Step 2 — Dry run to verify pipeline
```bash
python src/cache_vjepa_features.py \
    --dataset_dir /home/ubuntu/RAID/data/libero_spatial/libero_spatial \
    --output_dir /home/ubuntu/RAID/data/libero_spatial/vjepa_features \
    --device cuda --dry_run
```

### Step 3 — Cache features (run once, ~30-45 min)
```bash
python src/cache_vjepa_features.py \
    --dataset_dir /home/ubuntu/RAID/data/libero_spatial/libero_spatial \
    --output_dir /home/ubuntu/RAID/data/libero_spatial/vjepa_features \
    --device cuda
```

### Step 4 — Fair comparison sweep (~2-4 hours)
```bash
python src/run_all_libero.py \
    --feature_dir /home/ubuntu/RAID/data/libero_spatial/vjepa_features \
    --device cuda
```

### Step 5 — Autoresearch (~4-8 hours, run in tmux)
```bash
tmux new -s autoreach
python src/autoresearch_libero.py \
    --feature_dir /home/ubuntu/RAID/data/libero_spatial/vjepa_features \
    --n_iter 9 --device cuda
```

## Expected results and interpretation

### Ablation experiment
At N=25 demos (low-data regime), we expect:
- mean_action: highest MSE (no learning, no retrieval)
- nn_copy: lower than mean_action (retrieval-only, no learning)
- direct_mlp: moderate (parametric-only, starts to overfit)
- concat_mlp: lower than direct_mlp (retrieval helps via feature augmentation)
- raid_xattn: lowest MSE (retrieval + learned attention weighting)

At N=200 demos, all three learned conditions should converge.

### Autoresearch
The loop should identify at least 1-2 improvements over the baseline.
Common findings:
- Small k (3) often better than large k (10) — fewer but more relevant neighbours
- Noise regularisation on retrieved actions helps generalisation
- Simple architectures (1-layer cross-attn, moderate hidden dim) outperform
  complex ones at low data

### Critical fairness constraint
ALL conditions use IDENTICAL frozen V-JEPA 2 features as input.
The only thing that varies is the decoder architecture and whether
retrieval is used. This is non-negotiable.
