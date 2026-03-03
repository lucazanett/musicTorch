# MUSIC-u PyTorch Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement MUSIC-u (Mutual Information State Intrinsic Control, unsupervised) as a single CleanRL-style PyTorch file `music_sac.py`.

**Architecture:** SAC + HER + MINE intrinsic reward. No skill discriminator, no task reward during training. MINE estimates MI(grip_pos, object_pos) from consecutive observation pairs and provides an intrinsic reward that drives the robot to interact with the object.

**Tech Stack:** Python 3.10+, PyTorch 2.x, NumPy, Gymnasium, gymnasium-robotics (FetchPickAndPlace-v3 or similar).

---

## Reference

- Design doc: `docs/plans/2026-03-03-music-u-pytorch-design.md`
- Original TF1 code: `baselines/her/`
  - `actor_critic.py` — SAC actor and critic networks
  - `discriminator.py` — MINENet implementation
  - `normalizer.py` — running mean/std
  - `her.py` — HER transition sampling
  - `ddpg.py` — main agent: training loop, losses
  - `util.py` — `split_observation_np` for obs splitting

---

## Setup Note

Before starting, install dependencies:
```bash
pip install torch numpy gymnasium gymnasium-robotics pytest
```

MuJoCo is required for Fetch environments. If not available, Tasks 1–8 can be completed without it — the environment is only needed in Tasks 9–10.

---

## Task 1: Project Skeleton

**Files:**
- Create: `music_sac.py`
- Create: `tests/test_music_sac.py`

**Step 1: Create the skeleton file**

```python
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
```

**Step 2: Create test file skeleton**

```python
# tests/test_music_sac.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import pytest
from music_sac import *
```

**Step 3: Verify imports work**

```bash
python -c "import music_sac; print('OK')"
pytest tests/test_music_sac.py --collect-only
```
Expected: no import errors, 0 tests collected.

**Step 4: Commit**

```bash
git add music_sac.py tests/test_music_sac.py
git commit -m "feat: add music_sac.py skeleton and test file"
```

---

## Task 2: Normalizer

Tracks running mean/std for observations and goals. Pure numpy — no TF ops.

**Files:**
- Modify: `music_sac.py`
- Modify: `tests/test_music_sac.py`

**Step 1: Write the failing tests**

```python
# tests/test_music_sac.py

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
```

**Step 2: Run tests — verify they fail**

```bash
pytest tests/test_music_sac.py::test_normalizer_stats -v
```
Expected: `FAILED` with `ImportError` or `AttributeError: module 'music_sac' has no attribute 'Normalizer'`

**Step 3: Implement Normalizer**

```python
# music_sac.py — add after constants

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
```

**Step 4: Run tests — verify they pass**

```bash
pytest tests/test_music_sac.py -k "normalizer" -v
```
Expected: 4 tests PASSED.

**Step 5: Commit**

```bash
git add music_sac.py tests/test_music_sac.py
git commit -m "feat: add Normalizer with running mean/std"
```

---

## Task 3: MINENet

The core MUSIC component. Estimates MI(grip_pos; object_pos) using the MINE DV lower bound.

**Files:**
- Modify: `music_sac.py`
- Modify: `tests/test_music_sac.py`

**Step 1: Write the failing tests**

```python
# tests/test_music_sac.py

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
```

**Step 2: Run tests — verify they fail**

```bash
pytest tests/test_music_sac.py -k "mine" -v
```
Expected: FAILED with `AttributeError`

**Step 3: Implement MINENet**

Key: additive MLP — `T(x,y) = tanh(out(relu(Wx(x) + Wy(y))))`.
Shuffle along the time axis (dim=0 after transpose) to create marginal samples — matches original TF `tf.random_shuffle` on `(T, batch, dim)`.

```python
# music_sac.py

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

        # Shuffle y along time axis (mirrors tf.random_shuffle on (T, B, dim))
        y_t   = y.permute(1, 0, 2)                         # (2, B, 3)
        y_shuf = y_t[torch.randperm(y_t.shape[0])]         # shuffle T dim
        y_shuffle = y_shuf.permute(1, 0, 2)                 # (B, 2, 3)

        x_conc = torch.cat([x, x],         dim=1)           # (B, 4, 3)
        y_conc = torch.cat([y, y_shuffle], dim=1)           # (B, 4, 3)

        output = self._T(x_conc, y_conc)                    # (B, 4, 1)
        T_xy   = output[:, :2, :]                           # (B, 2, 1) — joint
        T_x_y  = output[:, 2:, :]                           # (B, 2, 1) — marginal

        mean_exp = torch.mean(torch.exp(T_x_y), dim=1)     # (B, 1)
        mine_est = torch.mean(T_xy, dim=1) - torch.log(mean_exp + 1e-8)
        return -mine_est                                     # neg_loss: minimize this
```

**Step 4: Run tests — verify they pass**

```bash
pytest tests/test_music_sac.py -k "mine" -v
```
Expected: 3 tests PASSED. (The training tests take ~5s each.)

**Step 5: Commit**

```bash
git add music_sac.py tests/test_music_sac.py
git commit -m "feat: add MINENet for mutual information estimation"
```

---

## Task 4: Actor

SAC Gaussian policy with tanh squashing.

**Files:**
- Modify: `music_sac.py`
- Modify: `tests/test_music_sac.py`

**Step 1: Write the failing tests**

```python
def test_actor_sample_shapes():
    actor = Actor(obs_dim=OBS_DIM, goal_dim=GOAL_DIM, act_dim=ACT_DIM)
    o = torch.randn(8, OBS_DIM)
    g = torch.randn(8, GOAL_DIM)
    action, log_prob = actor.sample(o, g)
    assert action.shape   == (8, ACT_DIM)
    assert log_prob.shape == (8, 1)

def test_actor_action_bounded():
    actor = Actor(obs_dim=OBS_DIM, goal_dim=GOAL_DIM, act_dim=ACT_DIM, max_u=MAX_U)
    o = torch.randn(64, OBS_DIM)
    g = torch.randn(64, GOAL_DIM)
    action, _ = actor.sample(o, g)
    assert torch.all(action >= -MAX_U - 1e-6)
    assert torch.all(action <=  MAX_U + 1e-6)

def test_actor_get_action_deterministic():
    torch.manual_seed(42)
    actor = Actor(obs_dim=OBS_DIM, goal_dim=GOAL_DIM, act_dim=ACT_DIM)
    actor.eval()
    o = torch.randn(1, OBS_DIM)
    g = torch.randn(1, GOAL_DIM)
    a1 = actor.get_action(o, g)
    a2 = actor.get_action(o, g)
    np.testing.assert_array_equal(a1, a2)

def test_actor_log_prob_finite():
    actor = Actor(obs_dim=OBS_DIM, goal_dim=GOAL_DIM, act_dim=ACT_DIM)
    o = torch.randn(32, OBS_DIM)
    g = torch.randn(32, GOAL_DIM)
    _, log_prob = actor.sample(o, g)
    assert torch.all(torch.isfinite(log_prob))
```

**Step 2: Run tests — verify they fail**

```bash
pytest tests/test_music_sac.py -k "actor" -v
```

**Step 3: Implement Actor**

```python
# music_sac.py

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
```

**Step 4: Run tests — verify they pass**

```bash
pytest tests/test_music_sac.py -k "actor" -v
```
Expected: 4 tests PASSED.

**Step 5: Commit**

```bash
git add music_sac.py tests/test_music_sac.py
git commit -m "feat: add SAC Actor with Gaussian policy and tanh squashing"
```

---

## Task 5: TwinQ (Critic)

Two independent Q-networks and a Polyak-averaged target copy.

**Files:**
- Modify: `music_sac.py`
- Modify: `tests/test_music_sac.py`

**Step 1: Write the failing tests**

```python
def test_twinq_output_shapes():
    critic = TwinQ(obs_dim=OBS_DIM, goal_dim=GOAL_DIM, act_dim=ACT_DIM)
    o = torch.randn(8, OBS_DIM)
    g = torch.randn(8, GOAL_DIM)
    a = torch.randn(8, ACT_DIM)
    q1, q2 = critic(o, g, a)
    assert q1.shape == (8, 1)
    assert q2.shape == (8, 1)

def test_twinq_q1_q2_differ():
    """The two Q heads should give different outputs (independent weights)."""
    torch.manual_seed(0)
    critic = TwinQ(obs_dim=OBS_DIM, goal_dim=GOAL_DIM, act_dim=ACT_DIM)
    o = torch.randn(4, OBS_DIM)
    g = torch.randn(4, GOAL_DIM)
    a = torch.randn(4, ACT_DIM)
    q1, q2 = critic(o, g, a)
    assert not torch.allclose(q1, q2)

def test_soft_update():
    critic = TwinQ(obs_dim=OBS_DIM, goal_dim=GOAL_DIM, act_dim=ACT_DIM)
    target = TwinQ(obs_dim=OBS_DIM, goal_dim=GOAL_DIM, act_dim=ACT_DIM)
    # Record initial target param
    p0 = list(target.parameters())[0].data.clone()
    p_src = list(critic.parameters())[0].data.clone()
    soft_update(target, critic, tau=0.05)
    p1 = list(target.parameters())[0].data.clone()
    expected = 0.05 * p_src + 0.95 * p0
    torch.testing.assert_close(p1, expected)
```

**Step 2: Run tests — verify they fail**

```bash
pytest tests/test_music_sac.py -k "twinq or soft_update" -v
```

**Step 3: Implement TwinQ and soft_update**

```python
# music_sac.py

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
    """Polyak averaging: target ← tau*source + (1-tau)*target."""
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)
```

**Step 4: Run tests — verify they pass**

```bash
pytest tests/test_music_sac.py -k "twinq or soft_update" -v
```
Expected: 3 tests PASSED.

**Step 5: Commit**

```bash
git add music_sac.py tests/test_music_sac.py
git commit -m "feat: add TwinQ critic and soft_update for Polyak averaging"
```

---

## Task 6: EpisodeBuffer

Stores full episodes and samples transitions with HER "future" goal relabeling.

**Files:**
- Modify: `music_sac.py`
- Modify: `tests/test_music_sac.py`

**Step 1: Write the failing tests**

```python
def _make_episode(T=5, obs_dim=OBS_DIM, goal_dim=GOAL_DIM, act_dim=ACT_DIM):
    return {
        'o':  np.random.randn(T + 1, obs_dim).astype(np.float32),
        'ag': np.random.randn(T + 1, goal_dim).astype(np.float32),
        'g':  np.random.randn(T,     goal_dim).astype(np.float32),
        'u':  np.random.randn(T,     act_dim).astype(np.float32),
    }

def _dummy_reward(ag, g):
    return (np.linalg.norm(ag - g, axis=-1) < 0.05).astype(np.float32) - 1.0

def test_buffer_store_and_size():
    buf = EpisodeBuffer(obs_dim=OBS_DIM, goal_dim=GOAL_DIM, act_dim=ACT_DIM,
                        T=5, buffer_size=50)
    assert buf.size == 0
    buf.store_episode(_make_episode(T=5))
    assert buf.size == 1
    for _ in range(20):
        buf.store_episode(_make_episode(T=5))
    assert buf.size == min(21, buf.max_episodes)

def test_buffer_sample_shapes():
    buf = EpisodeBuffer(obs_dim=OBS_DIM, goal_dim=GOAL_DIM, act_dim=ACT_DIM,
                        T=5, buffer_size=50)
    for _ in range(5):
        buf.store_episode(_make_episode(T=5))
    batch = buf.sample(32, _dummy_reward)
    assert batch['o'].shape   == (32, OBS_DIM)
    assert batch['o_2'].shape == (32, OBS_DIM)
    assert batch['g'].shape   == (32, GOAL_DIM)
    assert batch['u'].shape   == (32, ACT_DIM)
    assert batch['r'].shape   == (32,)

def test_buffer_her_relabeling():
    """High replay_k → most goals replaced → most rewards should be 0 (success)."""
    T = 10
    buf = EpisodeBuffer(obs_dim=3, goal_dim=3, act_dim=1, T=T,
                        buffer_size=200, replay_k=100)
    # Achieved goal is always [1,0,0]; desired goal is random
    ep = {
        'o':  np.zeros((T + 1, 3), np.float32),
        'ag': np.tile([1.0, 0.0, 0.0], (T + 1, 1)).astype(np.float32),
        'g':  np.random.randn(T, 3).astype(np.float32),
        'u':  np.zeros((T, 1), np.float32),
    }
    for _ in range(20):
        buf.store_episode(ep)
    batch = buf.sample(200, _dummy_reward)
    # With replay_k=100 nearly all goals replaced → ag≈g → reward≈0
    assert batch['r'].mean() > -0.3, f"Expected many successes, got mean_r={batch['r'].mean():.2f}"

def test_buffer_sample_mi_pairs():
    buf = EpisodeBuffer(obs_dim=OBS_DIM, goal_dim=GOAL_DIM, act_dim=ACT_DIM,
                        T=5, buffer_size=50)
    for _ in range(5):
        buf.store_episode(_make_episode(T=5))
    pairs = buf.sample_mi_pairs(32)
    assert pairs.shape == (32, 2, OBS_DIM)
```

**Step 2: Run tests — verify they fail**

```bash
pytest tests/test_music_sac.py -k "buffer" -v
```

**Step 3: Implement EpisodeBuffer**

```python
# music_sac.py

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

        g = self.g[ep_idx, t].copy()

        # HER: replace some goals with future achieved goals
        her_mask    = np.random.uniform(size=batch_size) < self.future_p
        her_idxs    = np.where(her_mask)[0]
        future_off  = np.random.randint(1, self.T + 1, size=batch_size)
        future_t    = np.minimum(t + future_off, self.T)
        g[her_idxs] = self.ag[ep_idx[her_idxs], future_t[her_idxs]]

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
```

**Step 4: Run tests — verify they pass**

```bash
pytest tests/test_music_sac.py -k "buffer" -v
```
Expected: 5 tests PASSED.

**Step 5: Commit**

```bash
git add music_sac.py tests/test_music_sac.py
git commit -m "feat: add EpisodeBuffer with HER future goal relabeling"
```

---

## Task 7: MI Reward Helper

Computes clipped MINE intrinsic reward for a batch of transitions.

**Files:**
- Modify: `music_sac.py`
- Modify: `tests/test_music_sac.py`

**Step 1: Write the failing test**

```python
def test_compute_mi_reward_shape_and_range():
    mine = MINENet()
    o   = np.random.randn(32, OBS_DIM).astype(np.float32)
    o_2 = np.random.randn(32, OBS_DIM).astype(np.float32)
    r = compute_mi_reward(mine, o, o_2, mi_r_scale=5000)
    assert r.shape == (32, 1)
    assert np.all(r >= 0.0)
    assert np.all(r <= 1.0)
```

**Step 2: Run test — verify it fails**

```bash
pytest tests/test_music_sac.py::test_compute_mi_reward_shape_and_range -v
```

**Step 3: Implement compute_mi_reward**

```python
# music_sac.py

def compute_mi_reward(mine: MINENet, o: np.ndarray, o_2: np.ndarray,
                      mi_r_scale: float = 5000.0) -> np.ndarray:
    """Compute clipped MINE intrinsic reward for a batch of transitions.
    Args:
        o, o_2: (B, obs_dim) consecutive observations
    Returns:
        r_i: (B, 1) float32, values in [0, 1]
    Reference: baselines/her/her.py:_sample_her_transitions (mi_r_scale * mi_trans)
               baselines/her/ddpg.py:_create_network (clip(mi_r_scale*m, 0, 1))
    """
    o_tau = np.stack([o, o_2], axis=1)                          # (B, 2, obs_dim)
    t     = torch.as_tensor(o_tau, dtype=torch.float32)
    with torch.no_grad():
        neg_loss = mine(t).cpu().numpy()                         # (B, 1)
    mi_est = -neg_loss                                           # MINE lower bound
    return np.clip(mi_r_scale * mi_est, 0.0, 1.0).astype(np.float32)
```

**Step 4: Run test — verify it passes**

```bash
pytest tests/test_music_sac.py::test_compute_mi_reward_shape_and_range -v
```
Expected: PASSED.

**Step 5: Commit**

```bash
git add music_sac.py tests/test_music_sac.py
git commit -m "feat: add compute_mi_reward helper"
```

---

## Task 8: SAC Update Functions

Critic update (Bellman), actor update, entropy temperature update.

**Files:**
- Modify: `music_sac.py`
- Modify: `tests/test_music_sac.py`

**Step 1: Write the failing tests**

```python
def test_update_critic_reduces_loss():
    """Critic loss should be finite and Q values should change after update."""
    torch.manual_seed(0)
    actor  = Actor(OBS_DIM, GOAL_DIM, ACT_DIM)
    critic = TwinQ(OBS_DIM, GOAL_DIM, ACT_DIM)
    target = TwinQ(OBS_DIM, GOAL_DIM, ACT_DIM)
    target.load_state_dict(critic.state_dict())
    opt_c = torch.optim.Adam(critic.parameters(), lr=1e-3)
    log_alpha = torch.tensor(0.0, requires_grad=True)

    norm_o = Normalizer(OBS_DIM); norm_g = Normalizer(GOAL_DIM)
    o   = torch.randn(32, OBS_DIM)
    o_2 = torch.randn(32, OBS_DIM)
    g   = torch.randn(32, GOAL_DIM)
    a   = torch.randn(32, ACT_DIM)
    r_i = torch.rand(32, 1)

    q1_before, _ = critic(norm_o.normalize(o), norm_g.normalize(g), a)
    loss = update_critic(critic, target, actor, opt_c, norm_o, norm_g,
                         o, o_2, g, a, r_i, log_alpha, gamma=0.98,
                         clip_return=50.0, action_l2=1.0)
    assert np.isfinite(loss)
    q1_after, _ = critic(norm_o.normalize(o), norm_g.normalize(g), a)
    assert not torch.allclose(q1_before, q1_after)

def test_update_actor_and_alpha():
    torch.manual_seed(0)
    actor  = Actor(OBS_DIM, GOAL_DIM, ACT_DIM)
    critic = TwinQ(OBS_DIM, GOAL_DIM, ACT_DIM)
    opt_a     = torch.optim.Adam(actor.parameters(),  lr=1e-3)
    log_alpha = torch.tensor(0.0, requires_grad=True)
    opt_alpha = torch.optim.Adam([log_alpha], lr=1e-3)
    norm_o = Normalizer(OBS_DIM); norm_g = Normalizer(GOAL_DIM)
    o = torch.randn(32, OBS_DIM); g = torch.randn(32, GOAL_DIM)

    alpha_before = log_alpha.item()
    update_actor(actor, critic, opt_a, norm_o, norm_g, o, g,
                 log_alpha, action_l2=1.0)
    update_alpha(log_alpha, opt_alpha, actor, norm_o, norm_g, o, g,
                 target_entropy=-ACT_DIM)
    assert log_alpha.item() != alpha_before
```

**Step 2: Run tests — verify they fail**

```bash
pytest tests/test_music_sac.py -k "update_critic or update_actor" -v
```

**Step 3: Implement update functions**

```python
# music_sac.py

def update_critic(
    critic: TwinQ, target: TwinQ, actor: Actor,
    opt: torch.optim.Optimizer,
    norm_o: Normalizer, norm_g: Normalizer,
    o: torch.Tensor, o_2: torch.Tensor, g: torch.Tensor,
    a: torch.Tensor, r_i: torch.Tensor, log_alpha: torch.Tensor,
    gamma: float = 0.98, clip_return: float = 50.0,
    action_l2: float = 1.0,
) -> float:
    """Bellman update for TwinQ critic.
    target_Q = r_i + gamma * (min_Q_target(s',a') - alpha * log_pi(a'))
    Reference: baselines/her/ddpg.py:_create_network (target_tf, Q_loss_tf)
    """
    alpha = log_alpha.detach().exp()
    with torch.no_grad():
        o2_norm = norm_o.normalize(o_2)
        g_norm  = norm_g.normalize(g)
        a_next, log_prob_next = actor.sample(o2_norm, g_norm)
        q1_t, q2_t = target(o2_norm, g_norm, a_next)
        q_target = r_i + gamma * (torch.min(q1_t, q2_t) - alpha * log_prob_next)
        q_target = torch.clamp(q_target, -clip_return, clip_return)

    o_norm = norm_o.normalize(o)
    g_norm = norm_g.normalize(g)
    q1, q2 = critic(o_norm, g_norm, a)
    loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
    opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()


def update_actor(
    actor: Actor, critic: TwinQ,
    opt: torch.optim.Optimizer,
    norm_o: Normalizer, norm_g: Normalizer,
    o: torch.Tensor, g: torch.Tensor,
    log_alpha: torch.Tensor, action_l2: float = 1.0,
) -> float:
    """SAC policy gradient update.
    L_pi = alpha * log_pi(a|s) - min_Q(s, a) + action_l2 * ||a/max_u||^2
    Reference: baselines/her/ddpg.py:_create_network (pi_loss_tf)
    """
    alpha  = log_alpha.detach().exp()
    o_norm = norm_o.normalize(o)
    g_norm = norm_g.normalize(g)
    a, log_prob = actor.sample(o_norm, g_norm)
    q1, q2 = critic(o_norm, g_norm, a)
    l2_reg = action_l2 * (a / actor.max_u).pow(2).mean()
    loss = (alpha * log_prob - torch.min(q1, q2) + l2_reg).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()


def update_alpha(
    log_alpha: torch.Tensor, opt: torch.optim.Optimizer,
    actor: Actor, norm_o: Normalizer, norm_g: Normalizer,
    o: torch.Tensor, g: torch.Tensor,
    target_entropy: float,
) -> float:
    """Entropy temperature update (learnable alpha).
    L_alpha = -alpha * (log_pi(a|s) + target_entropy)
    """
    o_norm = norm_o.normalize(o)
    g_norm = norm_g.normalize(g)
    with torch.no_grad():
        _, log_prob = actor.sample(o_norm, g_norm)
    loss = -(log_alpha * (log_prob + target_entropy)).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()
```

**Step 4: Run tests — verify they pass**

```bash
pytest tests/test_music_sac.py -k "update_critic or update_actor" -v
```
Expected: 2 tests PASSED.

**Step 5: Run all tests so far**

```bash
pytest tests/test_music_sac.py -v
```
Expected: all tests PASSED.

**Step 6: Commit**

```bash
git add music_sac.py tests/test_music_sac.py
git commit -m "feat: add SAC update functions for critic, actor, and alpha"
```

---

## Task 9: Rollout Collector

Collects one episode by running the current policy in the environment.

**Files:**
- Modify: `music_sac.py`

(No unit test possible without MuJoCo; tested by smoke test in Task 10.)

**Step 1: Add reward function and rollout collector**

```python
# music_sac.py

def compute_reward_np(achieved_goal: np.ndarray, desired_goal: np.ndarray,
                      threshold: float = 0.05) -> np.ndarray:
    """Sparse reward: 0 for success, -1 otherwise.
    Reference: gymnasium_robotics FetchPickAndPlace reward function.
    """
    d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
    return (d < threshold).astype(np.float32) - 1.0


def collect_episode(env, actor: Actor, norm_o: Normalizer, norm_g: Normalizer,
                    T: int, noise_eps: float = 0.2,
                    random_eps: float = 0.3) -> dict:
    """Run one episode and return it as a dict of arrays.
    Returns:
        o:  (T+1, obs_dim) observations
        ag: (T+1, goal_dim) achieved goals
        g:  (T, goal_dim)   desired goals
        u:  (T, act_dim)    actions
    Reference: baselines/her/rollout.py:RolloutWorker
    """
    obs_seq, ag_seq, g_seq, u_seq = [], [], [], []
    obs_dict, _ = env.reset()
    o   = obs_dict['observation']
    ag  = obs_dict['achieved_goal']
    g   = obs_dict['desired_goal']
    obs_seq.append(o.copy()); ag_seq.append(ag.copy())

    for _ in range(T):
        o_t = torch.as_tensor(
            norm_o.normalize(o[None]), dtype=torch.float32)
        g_t = torch.as_tensor(
            norm_g.normalize(g[None]), dtype=torch.float32)
        action = actor.get_action(o_t, g_t).flatten()

        # Epsilon-random exploration
        noise  = noise_eps * np.random.randn(*action.shape)
        action = np.clip(action + noise, -MAX_U, MAX_U)
        if np.random.uniform() < random_eps:
            action = np.random.uniform(-MAX_U, MAX_U, size=action.shape)

        obs_dict, _, _, _, _ = env.step(action)
        o   = obs_dict['observation']
        ag  = obs_dict['achieved_goal']
        g_  = obs_dict['desired_goal']   # goal stays fixed per episode

        obs_seq.append(o.copy()); ag_seq.append(ag.copy())
        g_seq.append(g.copy()); u_seq.append(action.copy())
        g = g_

    return {
        'o':  np.array(obs_seq,  np.float32),   # (T+1, obs_dim)
        'ag': np.array(ag_seq,   np.float32),
        'g':  np.array(g_seq,    np.float32),   # (T, goal_dim)
        'u':  np.array(u_seq,    np.float32),
    }
```

**Step 2: Commit**

```bash
git add music_sac.py
git commit -m "feat: add collect_episode and sparse reward function"
```

---

## Task 10: Training Loop and Smoke Test

Wire everything into a `train()` function and verify it runs for a few cycles.

**Files:**
- Modify: `music_sac.py`

**Step 1: Add train() function**

```python
# music_sac.py

import gymnasium as gym
import time

def train(
    env_name: str = 'FetchPickAndPlace-v3',
    n_epochs: int = 200,
    n_cycles: int = 50,
    n_batches: int = 40,
    rollout_batch: int = 2,
    T: int = 50,
    batch_size: int = 256,
    buffer_size: int = 1_000_000,
    replay_k: int = 4,
    hidden: int = 256,
    gamma: float = 0.98,
    polyak: float = 0.95,
    lr: float = 0.001,
    mi_r_scale: float = 5000.0,
    action_l2: float = 1.0,
    clip_return: float = 50.0,
    noise_eps: float = 0.2,
    random_eps: float = 0.3,
    seed: int = 0,
):
    np.random.seed(seed); torch.manual_seed(seed)

    env      = gym.make(env_name)
    eval_env = gym.make(env_name)

    norm_o = Normalizer(OBS_DIM)
    norm_g = Normalizer(GOAL_DIM)

    actor  = Actor( OBS_DIM, GOAL_DIM, ACT_DIM, hidden)
    critic = TwinQ( OBS_DIM, GOAL_DIM, ACT_DIM, hidden)
    target = TwinQ( OBS_DIM, GOAL_DIM, ACT_DIM, hidden)
    mine   = MINENet()
    target.load_state_dict(critic.state_dict())
    # Freeze target — updated only via soft_update
    for p in target.parameters():
        p.requires_grad_(False)

    log_alpha = torch.tensor(0.0, requires_grad=True)
    target_entropy = -float(ACT_DIM)

    opt_actor = torch.optim.Adam(actor.parameters(),  lr=lr)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=lr)
    opt_mine  = torch.optim.Adam(mine.parameters(),   lr=lr)
    opt_alpha = torch.optim.Adam([log_alpha],          lr=lr)

    buffer = EpisodeBuffer(OBS_DIM, GOAL_DIM, ACT_DIM, T, buffer_size, replay_k)

    for epoch in range(n_epochs):
        t_start = time.time()

        for cycle in range(n_cycles):
            # ── 1. Collect rollouts ──────────────────────────────────
            for _ in range(rollout_batch):
                ep = collect_episode(env, actor, norm_o, norm_g, T,
                                     noise_eps, random_eps)
                buffer.store_episode(ep)
                norm_o.update(ep['o'].reshape(-1, OBS_DIM))
                norm_g.update(ep['g'].reshape(-1, GOAL_DIM))
            norm_o.recompute_stats()
            norm_g.recompute_stats()

            if buffer.size < 1:
                continue

            # ── 2. Train MINE ────────────────────────────────────────
            for _ in range(n_batches):
                o_tau_np = buffer.sample_mi_pairs(batch_size)
                o_tau    = torch.as_tensor(o_tau_np, dtype=torch.float32)
                loss_mine = mine(o_tau).mean()
                opt_mine.zero_grad(); loss_mine.backward(); opt_mine.step()

            # ── 3. Train SAC ─────────────────────────────────────────
            for _ in range(n_batches):
                batch = buffer.sample(batch_size, compute_reward_np)
                r_i   = compute_mi_reward(mine, batch['o'], batch['o_2'],
                                          mi_r_scale)

                # numpy → tensor
                def t(x): return torch.as_tensor(x, dtype=torch.float32)
                o_t   = t(batch['o']);   o2_t  = t(batch['o_2'])
                g_t   = t(batch['g']);   a_t   = t(batch['u'])
                ri_t  = t(r_i)

                update_critic(critic, target, actor, opt_critic,
                              norm_o, norm_g, o_t, o2_t, g_t, a_t,
                              ri_t, log_alpha, gamma, clip_return, action_l2)
                update_actor( actor, critic, opt_actor,
                              norm_o, norm_g, o_t, g_t, log_alpha, action_l2)
                update_alpha( log_alpha, opt_alpha, actor,
                              norm_o, norm_g, o_t, g_t, target_entropy)
                soft_update(target, critic, tau=1.0 - polyak)

        # ── 4. Evaluate ──────────────────────────────────────────────
        success_rate = evaluate(eval_env, actor, norm_o, norm_g, T, n_episodes=10)
        elapsed = time.time() - t_start
        print(f"Epoch {epoch+1:3d}/{n_epochs}  "
              f"success={success_rate:.3f}  "
              f"alpha={log_alpha.exp().item():.4f}  "
              f"t={elapsed:.1f}s")

    env.close(); eval_env.close()


def evaluate(env, actor: Actor, norm_o: Normalizer, norm_g: Normalizer,
             T: int, n_episodes: int = 10) -> float:
    successes = 0
    for _ in range(n_episodes):
        obs_dict, _ = env.reset()
        o = obs_dict['observation']
        g = obs_dict['desired_goal']
        for _ in range(T):
            o_t = torch.as_tensor(
                norm_o.normalize(o[None]), dtype=torch.float32)
            g_t = torch.as_tensor(
                norm_g.normalize(g[None]), dtype=torch.float32)
            a = actor.get_action(o_t, g_t).flatten()
            obs_dict, _, terminated, truncated, info = env.step(a)
            o = obs_dict['observation']
            if terminated or truncated:
                break
        successes += float(info.get('is_success', 0.0))
    return successes / n_episodes


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',    default='FetchPickAndPlace-v3')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seed',   type=int, default=0)
    args = parser.parse_args()
    train(env_name=args.env, n_epochs=args.epochs, seed=args.seed)
```

**Step 2: Smoke test — run for 2 epochs**

```bash
python music_sac.py --epochs 2
```
Expected output (values will vary):
```
Epoch   1/2  success=0.000  alpha=...  t=...s
Epoch   2/2  success=0.000  alpha=...  t=...s
```
No exceptions. Success rate of 0 at epoch 2 is expected — the agent has barely trained.

**Step 3: Run all unit tests one final time**

```bash
pytest tests/test_music_sac.py -v
```
Expected: all tests PASSED.

**Step 4: Commit**

```bash
git add music_sac.py
git commit -m "feat: add training loop, evaluate, and CLI entrypoint"
```

---

## Final Run

To reproduce MUSIC-u from the paper on FetchPickAndPlace:

```bash
python music_sac.py --env FetchPickAndPlace-v3 --epochs 200 --seed 0
```

The agent trains without any task reward. Success rate should start increasing around epoch 50–100 as the MI reward drives the robot to interact with the object.

---

## Component Summary

| File | Purpose |
|---|---|
| `music_sac.py` | Complete single-file implementation |
| `tests/test_music_sac.py` | Unit tests for all components |
| `docs/plans/2026-03-03-music-u-pytorch-design.md` | Design reference |

| Class/Function | Lines (approx) | Tests |
|---|---|---|
| `Normalizer` | 25 | 4 |
| `MINENet` | 30 | 3 |
| `Actor` | 35 | 4 |
| `TwinQ` + `soft_update` | 25 | 3 |
| `EpisodeBuffer` | 45 | 5 |
| `compute_mi_reward` | 10 | 1 |
| `update_critic/actor/alpha` | 40 | 2 |
| `collect_episode` | 30 | — |
| `train` + `evaluate` | 80 | smoke |
