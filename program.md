markdown# RAID Autoresearch Program

## What you are doing
You are an AI research agent autonomously improving the RAID (Retrieval-Augmented Inverse Dynamics) decoder for robot manipulation. You modify `src/models.py`, run a training experiment, check if the result improved, keep or discard the change, and repeat.

## The problem
RAID is supposed to outperform a direct MLP inverse dynamics model in low-data regimes by using a memory bank of demonstrated transitions as a prior. It currently fails:

| Condition | 25 demos | 50 demos | 100 demos | 200 demos |
|---|---|---|---|---|
| Direct MLP (baseline to beat) | 0.336 | 0.358 | 0.296 | 0.183 |
| RAID (current, broken) | 0.444 | 0.512 | 0.536 | 0.424 |
| Nearest neighbor | 0.617 | 0.744 | 0.717 | 0.567 |

RAID is worse than direct MLP at every scale. The root cause: the decoder takes the path of least resistance and copies the noisy retrieved prior (a_prior) rather than learning from the state transition (s_t, s_next). The prior is the mean of k=3 not-quite-right retrieved actions, so copying it hurts.

## The metric
Run: `python3 src/train.py --condition raid --n_demos 25`
Read val_mse from the final printed line: `[train] best checkpoint val_mse=X.XXXXXX`
**Target: val_mse < 0.336 (beat direct MLP at 25 demos)**
Secondary target: val_mse < 0.444 (beat current RAID)

## The file you edit
**`src/models.py`** — specifically the `RAIDDecoder` class. You may also edit the `forward()` call signature if needed, but `src/train.py` calls it as `decoder(s_t, s_n, prior)` so keep that interface or update train.py consistently.

Current RAIDDecoder:
```python
def forward(self, s_t, s_next, a_prior):
    return self.net(torch.cat([s_t, s_next, a_prior], dim=-1))
```
Input: concat(s_t [19-dim], s_next [19-dim], a_prior [7-dim]) = 45-dim
Output: a_hat [7-dim]

## Hypotheses to try (in order)
1. **Residual prediction**: return `a_prior + self.net(concat(s_t, s_next, a_prior))`. Decoder learns only the correction delta. If prior is bad, delta → 0 and output falls back to prior.
2. **Learned gate**: `g = sigmoid(Linear(state_dim*2, action_dim)(concat(s_t, s_next)))`. Output: `g * mlp(s_t, s_next) + (1-g) * a_prior`. Gate learns when to trust retrieval vs parametric.
3. **Detached prior**: `a_prior_detached = a_prior.detach()`. Stop gradients from flowing through the prior. Forces decoder to treat prior as fixed context, not a learnable shortcut.
4. **Separate encoders**: encode (s_t, s_next) and a_prior through separate linear projections before concatenating. Gives the model separate representations for transition and prior.
5. **Prior as residual with learned scale**: `scale = sigmoid(Linear(state_dim*2, 1)(concat(s_t, s_next)))`. Output: `scale * a_prior + self.net(concat(s_t, s_next))`. Learned scalar weight on the prior.
6. **Free hypothesis**: propose your own architectural change based on what you've observed.

## Loop instructions
1. Read the current best val_mse from `configs/autoresearch_log.md` (create if missing, start with baseline 0.444)
2. Pick the next hypothesis to try (or propose your own if you have a better idea)
3. Edit `src/models.py` to implement the change
4. Run: `python3 src/train.py --condition raid --n_demos 25`
5. Read the val_mse from output
6. If improved: keep the change, update `configs/autoresearch_log.md`, run full eval at all scales: `python3 src/run_all.py` (only RAID conditions), regenerate figures
7. If worse: revert `src/models.py` to the previous version, log the result as a failure
8. Repeat for at least 8 iterations or until val_mse < 0.336
9. After completing all iterations: run `python3 src/run_all.py` for all conditions, regenerate all figures, push to GitHub: `git add -A && git commit -m "autoresearch: best RAID val_mse=X.XXX after N iterations" && git push origin main`

## Log format
Append to `configs/autoresearch_log.md` after each iteration:
```
## Iteration N
- Hypothesis: [name]
- Change: [one-line description]
- val_mse: X.XXXXX
- vs baseline: [improved/worse]
- Decision: [kept/reverted]
- Notes: [any observations]
```

## Constraints
- Only edit `src/models.py` and `src/train.py` (if needed for interface changes)
- Do not change the data pipeline, memory bank, or evaluation code
- Do not change n_demos=25 for the iteration metric — this is the hardest regime
- Each experiment must complete fully before starting the next
- If training crashes, log it as a failure and revert
