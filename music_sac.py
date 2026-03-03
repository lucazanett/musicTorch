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


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class TwinQ(nn.Module):
    """Twin Q-networks for SAC critic.
    Input:  normalized obs + normalized goal + action/max_u
    Output: Q1, Q2 each of shape (B, 1)
    Reference: baselines/her/actor_critic.py:ActorCritic (Q scope, twin via reuse=False)
    """

    def __init__(self, obs_dim: int, goal_dim: int, act_dim: int,
                 hidden: int = 256, max_u: float = MAX_U):
        super().__init__()
        self.max_u = max_u
        in_dim = obs_dim + goal_dim + act_dim
        self.q1 = _mlp(in_dim, hidden, 1)
        self.q2 = _mlp(in_dim, hidden, 1)

    def forward(self, o_norm: torch.Tensor, g_norm: torch.Tensor,
                a: torch.Tensor):
        x = torch.cat([o_norm, g_norm, a / self.max_u], dim=-1)
        return self.q1(x), self.q2(x)


def soft_update(target: nn.Module, source: nn.Module, tau: float = 0.05) -> None:
    """Polyak averaging: target <- tau*source + (1-tau)*target."""
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


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


class EpisodeBuffer:
    """Episode-based replay buffer with HER 'future' goal relabeling.
    Stores complete episodes; samples individual transitions.
    Reference: baselines/her/replay_buffer.py + her.py:make_sample_her_transitions
    """

    def __init__(self, obs_dim: int, goal_dim: int, act_dim: int,
                 T: int, buffer_size: int = 1_000_000, replay_k: int = 4):
        self.T           = T
        self.future_p    = 1.0 - 1.0 / (1 + replay_k)
        self.max_episodes = buffer_size // T
        self.o   = np.zeros((self.max_episodes, T + 1, obs_dim),  np.float32)
        self.ag  = np.zeros((self.max_episodes, T + 1, goal_dim), np.float32)
        self.g   = np.zeros((self.max_episodes, T,     goal_dim), np.float32)
        self.u   = np.zeros((self.max_episodes, T,     act_dim),  np.float32)
        self._ptr  = 0
        self.size  = 0

    def store_episode(self, ep: dict) -> None:
        idx = self._ptr % self.max_episodes
        self.o[idx]  = ep['o']    # (T+1, obs_dim)
        self.ag[idx] = ep['ag']
        self.g[idx]  = ep['g']
        self.u[idx]  = ep['u']
        self._ptr += 1
        self.size = min(self.size + 1, self.max_episodes)

    def sample(self, batch_size: int, reward_fn) -> dict:
        ep_idx = np.random.randint(0, self.size, batch_size)
        t      = np.random.randint(0, self.T,    batch_size)

        g = self.g[ep_idx, t].copy()  # must copy: mutated in-place by HER below

        # HER: replace some goals with future achieved goals (future_p fraction)
        her_mask   = np.random.uniform(size=batch_size) < self.future_p
        her_idxs   = np.where(her_mask)[0]
        # Compute future_t only for HER transitions to make intent clear
        future_off = np.random.randint(1, self.T + 1, size=len(her_idxs))
        future_t   = np.minimum(t[her_idxs] + future_off, self.T)
        g[her_idxs] = self.ag[ep_idx[her_idxs], future_t]

        o    = self.o [ep_idx, t    ]
        o_2  = self.o [ep_idx, t + 1]
        ag   = self.ag[ep_idx, t    ]
        ag_2 = self.ag[ep_idx, t + 1]
        u    = self.u [ep_idx, t    ]
        r    = reward_fn(ag_2, g)

        return dict(o=o, o_2=o_2, ag=ag, ag_2=ag_2, g=g, u=u, r=r)

    def sample_mi_pairs(self, batch_size: int) -> np.ndarray:
        """Return (batch_size, 2, obs_dim) consecutive observation pairs for MINE."""
        ep_idx = np.random.randint(0, self.size, batch_size)
        t      = np.random.randint(0, self.T,    batch_size)
        o_curr = self.o[ep_idx, t    ]   # (B, obs_dim)
        o_next = self.o[ep_idx, t + 1]   # (B, obs_dim)
        return np.stack([o_curr, o_next], axis=1)   # (B, 2, obs_dim)
