# music_sac.py
"""MUSIC-u: Mutual Information State Intrinsic Control (ICLR 2021)
Single-file CleanRL-style PyTorch implementation.
Reference: https://openreview.net/forum?id=OthEq8I5v1
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Constants ──────────────────────────────────────────────────────────
LOG_STD_MIN = -5
LOG_STD_MAX = 2
OBS_DIM = 25       # FetchPickAndPlace observation dim
GOAL_DIM = 3       # desired/achieved goal dim
ACT_DIM = 4        # action dim
MAX_U = 1.0        # action magnitude bound

# Obs split indices (grip_pos=0:3, object_pos=3:6) — see baselines/her/util.py
GRIP_POS_SLICE = slice(0, 3)
OBJ_POS_SLICE  = slice(3, 6)


class Normalizer:
    """Running mean/std normalizer. Thread-unsafe (single process)."""

    def __init__(self, size: int, eps: float = 1e-2, clip_range: float = 5.0):
        self.size = size
        self.eps = eps
        self.clip_range = clip_range
        self._sum   = np.zeros(size, np.float32)
        self._sumsq = np.zeros(size, np.float32)
        self._count = np.array([0], np.float32)
        self.mean = np.zeros(size, np.float32)
        self.std  = np.ones(size, np.float32)

    def update(self, v: np.ndarray) -> None:
        v = v.reshape(-1, self.size)
        self._sum   += v.sum(axis=0)
        self._sumsq += (v ** 2).sum(axis=0)
        self._count[0] += v.shape[0]

    def recompute_stats(self) -> None:
        self.mean = self._sum / self._count
        self.std  = np.sqrt(np.maximum(
            self.eps ** 2,
            self._sumsq / self._count - (self._sum / self._count) ** 2,
        ))

    def normalize(self, v):
        if isinstance(v, np.ndarray):
            return np.clip((v - self.mean) / self.std, -self.clip_range, self.clip_range)
        mean = torch.as_tensor(self.mean, dtype=torch.float32).to(v.device)
        std  = torch.as_tensor(self.std,  dtype=torch.float32).to(v.device)
        return torch.clamp((v - mean) / std, -self.clip_range, self.clip_range)
