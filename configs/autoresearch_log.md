# RAID autoresearch log

Baseline (paper table): RAID `val_mse≈0.444` vs direct MLP `0.336` at n_demos=25. Goal: `<0.336`. Metric: `[train] best checkpoint val_mse=...`.

---
## Iteration 0 (baseline)
- Established starting snapshot before hypothesis trials.
- best_val_so_far reference: **0.444**

## Iteration 1
- Hypothesis: Residual prediction (`a_prior + net(concat)`)
- Change: `forward` returns `a_prior + self.net(torch.cat(...))`
- val_mse: 0.621970
- vs baseline: worse
- Decision: reverted
- Notes: Residual head destabilized validation (best at late epoch ~0.62); reverted architecture and re-synced checkpoint with baseline concat weights.

## Iteration 2
- Hypothesis: Detached prior (iter1 reverted; concat + `a_prior.detach()` in `forward`)
- Change: `ctx = a_prior.detach(); net(cat(s_t,s_next,ctx))`
- val_mse: 0.444306
- vs baseline: same
- Decision: kept

## Iteration 3
- Hypothesis: Learned gate blend (parametric inverse vs pooled prior)
- Change: `g=sigmoid(Linear(trans)); return g*direct(trans)+(1-g)*a_prior`
- val_mse: 0.431137
- vs baseline: improved
- Decision: kept

## Iteration 4
- Hypothesis: Separate encoders for transition vs pooled prior
- Change: 128-d projections + fused 256-d trunk
- val_mse: 0.483271
- vs baseline: worse
- Decision: reverted (restored iter-3 gated `RAIDDecoder`)
- Notes: best at epoch 1; separate projection lost gating inductive bias.

## Iteration 5
- Hypothesis: Prior residual with scalar learned scale (`scale*a_prior + net(trans)`)
- Change: scalar `sigmoid(Linear(trans))`; MLP ignores prior in concat
- val_mse: 0.446020
- vs baseline: worse (vs iter-3 best 0.431137)
- Decision: reverted (restored iter-3 gated decoder)
- Notes: Slightly beats raw 0.444 but inferior to gated blend.

## Iteration 6
- Hypothesis: Prior dropout inside gated RAID (`Dropout` on pooled prior pathway)
- Change: `(1-g)*prior_drop(a_prior)` during training only
- val_mse: 0.398695
- vs baseline (0.431137): improved
- Decision: kept
- Notes: First clear win over gated-only baseline; pushes prior branch to diversify.

## Iteration 7
- Hypothesis: Additive Gaussian jitter on pooled prior during training
- Change: `(prior_drop(ap) + 0.1*N(0,I))` in train mode
- val_mse: 0.396789
- vs iter-6 best: improved
- Decision: kept

## Iteration 8
- Hypothesis: Wider direct pathway (`hidden_dim*2` inside `Sequential`)
- val_mse: 0.409794
- vs iter-7 best (0.396789): worse
- Decision: reverted (narrower gated+drop+noise config)

---
## Summary
- Best RAID `val_mse` **0.396789** @25 demos — beats historic 0.444 and iteration-3 gated 0.431, **does not beat** direct MLP 0.336.
- Winning stack: gated blend + pooled-prior dropout + train-time Gaussian prior noise.
