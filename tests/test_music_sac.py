# tests/test_music_sac.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import pytest
from music_sac import *


def test_normalizer_stats():
    norm = Normalizer(size=3)
    data = np.random.randn(2000, 3).astype(np.float32) * 2.0 + 5.0  # mean≈5, std≈2
    norm.update(data)
    norm.recompute_stats()
    np.testing.assert_allclose(norm.mean, 5.0, atol=0.15)
    np.testing.assert_allclose(norm.std,  2.0, atol=0.15)

def test_normalizer_normalize_numpy():
    norm = Normalizer(size=3)
    data = np.random.randn(2000, 3).astype(np.float32) * 2.0 + 5.0
    norm.update(data); norm.recompute_stats()
    x = np.array([[5.0, 5.0, 5.0]], dtype=np.float32)
    out = norm.normalize(x)
    np.testing.assert_allclose(out, 0.0, atol=0.15)

def test_normalizer_normalize_tensor():
    norm = Normalizer(size=3)
    data = np.random.randn(2000, 3).astype(np.float32) * 2.0 + 5.0
    norm.update(data); norm.recompute_stats()
    x = torch.tensor([[5.0, 5.0, 5.0]])
    out = norm.normalize(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 3)
    torch.testing.assert_close(out, torch.zeros(1, 3), atol=0.15, rtol=0)

def test_normalizer_clip():
    norm = Normalizer(size=1, clip_range=5.0)
    data = np.array([[0.0]] * 100, dtype=np.float32)  # mean=0, std≈eps
    norm.update(data); norm.recompute_stats()
    x = np.array([[1e9]], dtype=np.float32)
    out = norm.normalize(x)
    assert out[0, 0] <= 5.0


def test_mine_output_shape():
    mine = MINENet(hidden=64)
    o_tau = torch.randn(16, 2, OBS_DIM)
    neg_loss = mine(o_tau)
    assert neg_loss.shape == (16, 1), f"expected (16,1), got {neg_loss.shape}"

def test_mine_neg_loss_on_correlated_data():
    """After training, MINE estimate (= -neg_loss) should be > 0 for correlated data."""
    torch.manual_seed(0)
    mine = MINENet(hidden=64)
    opt = torch.optim.Adam(mine.parameters(), lr=1e-3)

    for _ in range(300):
        base = torch.randn(64, 2, 3)
        o_tau = torch.zeros(64, 2, OBS_DIM)
        o_tau[:, :, GRIP_POS_SLICE] = base                          # x = base
        o_tau[:, :, OBJ_POS_SLICE]  = base + torch.randn(64, 2, 3) * 0.1  # y ≈ x
        loss = mine(o_tau).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        neg_loss_val = mine(o_tau).mean().item()
    assert neg_loss_val < 0, f"Expected neg_loss < 0 (MI > 0), got {neg_loss_val}"

def test_mine_higher_mi_for_correlated_vs_independent():
    """MINE estimate should be higher for correlated pairs than independent pairs."""
    torch.manual_seed(1)
    mine = MINENet(hidden=64)
    opt = torch.optim.Adam(mine.parameters(), lr=1e-3)

    for _ in range(300):
        base = torch.randn(64, 2, 3)
        o_tau = torch.zeros(64, 2, OBS_DIM)
        o_tau[:, :, GRIP_POS_SLICE] = base
        o_tau[:, :, OBJ_POS_SLICE]  = base + torch.randn(64, 2, 3) * 0.1
        mine(o_tau).mean().backward()
        opt.step(); opt.zero_grad()

    with torch.no_grad():
        # Correlated
        base = torch.randn(64, 2, 3)
        corr = torch.zeros(64, 2, OBS_DIM)
        corr[:, :, GRIP_POS_SLICE] = base
        corr[:, :, OBJ_POS_SLICE]  = base + torch.randn(64, 2, 3) * 0.1
        mi_corr = -mine(corr).mean().item()

        # Independent
        indep = torch.zeros(64, 2, OBS_DIM)
        indep[:, :, GRIP_POS_SLICE] = torch.randn(64, 2, 3)
        indep[:, :, OBJ_POS_SLICE]  = torch.randn(64, 2, 3)
        mi_indep = -mine(indep).mean().item()

    assert mi_corr > mi_indep, f"Expected MI(corr)={mi_corr:.3f} > MI(indep)={mi_indep:.3f}"
