# tests/test_music_sac.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import pytest
from music_sac import *


def test_normalizer_stats():
    np.random.seed(42)
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


# ── EpisodeBuffer helpers ───────────────────────────────────────────────────

def _make_episode(T=5, obs_dim=OBS_DIM, goal_dim=GOAL_DIM, act_dim=ACT_DIM):
    return {
        'o':  np.random.randn(T + 1, obs_dim).astype(np.float32),
        'ag': np.random.randn(T + 1, goal_dim).astype(np.float32),
        'g':  np.random.randn(T,     goal_dim).astype(np.float32),
        'u':  np.random.randn(T,     act_dim).astype(np.float32),
    }

def _dummy_reward(ag, g):
    return (np.linalg.norm(ag - g, axis=-1) < 0.05).astype(np.float32) - 1.0


# ── EpisodeBuffer tests ─────────────────────────────────────────────────────

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
    # With replay_k=100 nearly all goals replaced → ag≈g≈[1,0,0] → reward≈0
    assert batch['r'].mean() > -0.3, f"Expected many successes, got mean_r={batch['r'].mean():.2f}"

def test_buffer_sample_mi_pairs():
    buf = EpisodeBuffer(obs_dim=OBS_DIM, goal_dim=GOAL_DIM, act_dim=ACT_DIM,
                        T=5, buffer_size=50)
    for _ in range(5):
        buf.store_episode(_make_episode(T=5))
    pairs = buf.sample_mi_pairs(32)
    assert pairs.shape == (32, 2, OBS_DIM)


def test_compute_mi_reward_shape_and_range():
    mine = MINENet()
    o   = np.random.randn(32, OBS_DIM).astype(np.float32)
    o_2 = np.random.randn(32, OBS_DIM).astype(np.float32)
    r = compute_mi_reward(mine, o, o_2, mi_r_scale=5000)
    assert r.shape == (32, 1)
    assert np.all(r >= 0.0)
    assert np.all(r <= 1.0)
