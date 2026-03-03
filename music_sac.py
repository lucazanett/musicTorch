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


class MINENet(nn.Module):
    """MINE estimator for MI(grip_pos; object_pos) across consecutive timesteps.
    Input: o_tau of shape (B, 2, OBS_DIM)
    Output: neg_loss of shape (B, 1)  — minimize to maximize MI lower bound.
    Reference: baselines/her/discriminator.py:Discriminator (state_mi scope)
    """

    def __init__(self, x_dim: int = 3, y_dim: int = 3, hidden: int = 128):
        super().__init__()
        self.Wx  = nn.Linear(x_dim, hidden // 2)
        self.Wy  = nn.Linear(y_dim, hidden // 2)
        self.out = nn.Linear(hidden // 2, 1)

    def _T(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute T(x, y). x, y: (..., dim) — any leading dims."""
        return torch.tanh(self.out(F.relu(self.Wx(x) + self.Wy(y))))

    def forward(self, o_tau: torch.Tensor) -> torch.Tensor:
        # o_tau: (B, 2, OBS_DIM)
        x = o_tau[:, :, GRIP_POS_SLICE]   # (B, 2, 3)
        y = o_tau[:, :, OBJ_POS_SLICE]    # (B, 2, 3)

        # Shuffle y along the batch axis to create marginal (x, ỹ) samples.
        # Note: the TF1 reference shuffles the T axis, but with T=2 that is a
        # near-useless coin flip. We shuffle the B axis instead (per design doc),
        # which always produces genuine independent marginal samples.
        y_t    = y.permute(1, 0, 2)                          # (2, B, 3)
        y_shuf = y_t[:, torch.randperm(y_t.shape[1]), :]     # shuffle B dim
        y_shuffle = y_shuf.permute(1, 0, 2)                  # (B, 2, 3)

        x_conc = torch.cat([x, x],         dim=1)            # (B, 4, 3)
        y_conc = torch.cat([y, y_shuffle], dim=1)            # (B, 4, 3)

        output = self._T(x_conc, y_conc)                     # (B, 4, 1)
        T_xy   = output[:, :2, :]                            # (B, 2, 1) — joint
        T_x_y  = output[:, 2:, :]                            # (B, 2, 1) — marginal

        mean_exp = torch.mean(torch.exp(T_x_y), dim=1)      # (B, 1)
        mine_est = torch.mean(T_xy, dim=1) - torch.log(mean_exp + 1e-8)
        return -mine_est                                      # neg_loss: minimize this


class Actor(nn.Module):
    """SAC Gaussian policy with tanh squashing.
    Input:  normalized obs (B, obs_dim) + normalized goal (B, goal_dim)
    Output: action in [-max_u, max_u]^act_dim, log_prob (B,1)
    Reference: baselines/her/actor_critic.py:mlp_gaussian_policy + apply_squashing_func
    """

    def __init__(self, obs_dim: int, goal_dim: int, act_dim: int,
                 hidden: int = 256, max_u: float = MAX_U):
        super().__init__()
        self.max_u = max_u
        self.net = nn.Sequential(
            nn.Linear(obs_dim + goal_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),              nn.ReLU(),
        )
        self.mu_layer      = nn.Linear(hidden, act_dim)
        self.log_std_layer = nn.Linear(hidden, act_dim)

    def _mu_log_std(self, o_norm: torch.Tensor, g_norm: torch.Tensor):
        h = self.net(torch.cat([o_norm, g_norm], dim=-1))
        mu = self.mu_layer(h)
        log_std = torch.tanh(self.log_std_layer(h))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1.0)
        return mu, log_std

    def sample(self, o_norm: torch.Tensor, g_norm: torch.Tensor):
        mu, log_std = self._mu_log_std(o_norm, g_norm)
        std = log_std.exp()
        x_t = mu + torch.randn_like(mu) * std           # reparameterization
        y_t = torch.tanh(x_t)
        action = y_t * self.max_u
        # log prob with squashing correction
        log_prob = (
            -0.5 * ((x_t - mu) / (std + 1e-8)).pow(2)
            - log_std
            - 0.5 * np.log(2 * np.pi)
        ).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1.0 - y_t.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        return action, log_prob

    def get_action(self, o_norm: torch.Tensor, g_norm: torch.Tensor) -> np.ndarray:
        """Deterministic action for rollouts/eval (no noise)."""
        with torch.no_grad():
            mu, _ = self._mu_log_std(o_norm, g_norm)
        return (torch.tanh(mu) * self.max_u).cpu().numpy()
