"""Smoke tests for the IDM heads.

Each head must drive synthetic-IDM regression loss below 1.0 within 200
optimization steps. The synthetic task: action equals a fixed linear map
of ``concat(obs_t, obs_next)`` plus a small noise floor, so a competent
head should learn it trivially.
"""

from __future__ import annotations

import pytest
import torch

from v2.heads import DiffusionPolicyIDM, KNNRetrievalHead, TransformerIDM
from v2.legacy.memory import FeatureMemoryBank
from v2.legacy.models import DirectMLP, RAIDDecoder


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _synth(n: int, obs_dim: int, action_dim: int = 7):
    obs_t = torch.randn(n, obs_dim)
    obs_next = obs_t + 0.3 * torch.randn(n, obs_dim)
    W = torch.randn(2 * obs_dim, action_dim) * 0.3
    a = torch.cat([obs_t, obs_next], dim=-1) @ W + 0.05 * torch.randn(n, action_dim)
    return obs_t, obs_next, a


def test_direct_mlp_smoke():
    obs_t, obs_next, a = _synth(1024, obs_dim=19)
    model = DirectMLP(obs_dim=19, action_dim=7)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(200):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(obs_t, obs_next), a)
        loss.backward()
        opt.step()
    assert loss.item() < 1.0


def test_raid_decoder_smoke():
    obs_t, obs_next, a = _synth(1024, obs_dim=19)
    prior = a + 0.5 * torch.randn_like(a)
    model = RAIDDecoder(obs_dim=19, action_dim=7)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(200):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(obs_t, obs_next, prior), a)
        loss.backward()
        opt.step()
    assert loss.item() < 1.0


def test_transformer_idm_smoke():
    obs_t, obs_next, a = _synth(1024, obs_dim=19)
    model = TransformerIDM(obs_dim=19, action_dim=7, seq_len=4, d_model=64, n_layers=2, n_heads=4, dim_ff=128)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(200):
        opt.zero_grad()
        pred = model.forward_pair(obs_t, obs_next)
        loss = torch.nn.functional.mse_loss(pred, a)
        loss.backward()
        opt.step()
    assert loss.item() < 1.0


def test_diffusion_idm_loss_decreases():
    obs_t, obs_next, a = _synth(512, obs_dim=19)
    model = DiffusionPolicyIDM(obs_dim=19, action_dim=7, hidden_dim=128, n_train_timesteps=50, n_inference_steps=8)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    initial_losses = []
    final_losses = []
    for step in range(400):
        opt.zero_grad()
        loss = model.loss(obs_t, obs_next, a)
        loss.backward()
        opt.step()
        if step < 10:
            initial_losses.append(loss.item())
        if step >= 380:
            final_losses.append(loss.item())
    assert sum(final_losses) / len(final_losses) < sum(initial_losses) / len(initial_losses)


def test_knn_retrieval_head_smoke():
    """kNN should produce non-trivial retrieval predictions and be a no-op under training."""
    obs_dim = 19
    n = 256
    obs_t, obs_next, a = _synth(n, obs_dim=obs_dim)
    mem = FeatureMemoryBank(obs_dim=obs_dim, action_dim=7, max_entries=n + 1, device="cpu")
    for i in range(n):
        mem.add(obs_t[i], obs_next[i], a[i])

    head = KNNRetrievalHead(memory=mem, k=3)
    pred = head(obs_t[:32], obs_next[:32])
    assert pred.shape == (32, 7)
    # The retrieved actions are pooled from the closest 3 transitions; for
    # the very first query the closest match is itself, so the prediction
    # should be close to the ground truth.
    init_loss = torch.nn.functional.mse_loss(pred, a[:32]).item()
    assert init_loss < 1.0

    # The single dummy parameter should produce a finite gradient via the
    # ``+ 0.0 * self._dummy.sum()`` term, but the gradient is identically
    # zero so AdamW.step() is a true no-op.
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    pred = head(obs_t, obs_next)
    loss = torch.nn.functional.mse_loss(pred, a)
    loss.backward()
    grads_zero = all(
        (p.grad is None) or torch.equal(p.grad, torch.zeros_like(p.grad))
        for p in head.parameters()
    )
    assert grads_zero
    opt.step()


def test_visualize_action_panel_only(tmp_path):
    """Visualize: render an action-only panel without RGB frames."""
    import numpy as np

    from v2 import visualize

    out = visualize.render_transition(
        save_path=tmp_path / "panel.png",
        frame_t=None,
        frame_next=None,
        action_gt=np.array([0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 1.0]),
        action_pred=np.array([0.12, -0.18, 0.28, 0.05, 0.0, 0.01, 0.95]),
        title="smoke",
        state_t=np.zeros(19),
        state_next=np.zeros(19),
    )
    assert out.is_file()
    assert out.stat().st_size > 1000  # non-trivial PNG
