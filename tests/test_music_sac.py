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
