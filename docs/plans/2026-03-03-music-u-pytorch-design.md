# MUSIC-u PyTorch Design

**Date:** 2026-03-03
**Scope:** MUSIC-u only — mutual information intrinsic reward to accelerate learning in sparse reward environments. No skill discovery, no skill discriminator, no latent skill vector z.
**Structure:** Single-file CleanRL-style (`music_sac.py`), standalone PyTorch + Gymnasium, single process (no MPI).
**Reference:** [ICLR 2021 paper](https://openreview.net/forum?id=OthEq8I5v1), original TF1 repo in `baselines/her/`.

---

## Algorithm Summary

MUSIC-u = SAC + HER + MINE intrinsic reward.

The agent learns without any task reward. Instead, it receives an intrinsic reward proportional to the mutual information between the robot's gripper position and the object's position across consecutive timesteps. This drives the agent to interact with the object, which leads to discovering useful behaviors that can later be finetuned with sparse task rewards.

---

## Observation Structure

Environment: `FetchPickAndPlace-v1` (obs dim = 25).

Split used by MINE (env-specific, hardcoded for Fetch envs):
```
obs[0:3]   → grip_pos        (robot gripper XYZ)  — used as x in MINE
obs[3:6]   → object_pos      (object XYZ)          — used as y in MINE
obs[6:9]   → object_rel_pos
obs[9:11]  → gripper_state
obs[11:14] → object_rot
obs[14:17] → object_velp
obs[17:20] → object_velr
obs[20:23] → grip_velp
obs[23:25] → gripper_vel
```

The goal `g` has dim=3 (desired object XYZ).

---

## Components

### `Normalizer`
Pure numpy running mean/std. Updated from episode data after each cycle. Used to normalize actor and critic inputs at inference time (returns torch tensor).

```python
normalize(x: np.ndarray) -> torch.Tensor
    # returns clip((x - mean) / std, -5, 5)
update(v: np.ndarray)          # accumulate sum, sumsq, count
recompute_stats()              # update mean/std from accumulators
```

### `MINENet(nn.Module)`
Estimates a lower bound on MI(grip_pos; object_pos) using the MINE DV representation.

**Input:** `o_tau` of shape `(B, 2, 25)` — pairs of consecutive observations
**Internal split:**
```
x = o_tau[:, :, 0:3]   # grip_pos across 2 timesteps, shape (B, 2, 3)
y = o_tau[:, :, 3:6]   # object_pos across 2 timesteps, shape (B, 2, 3)
```

**Architecture (additive MLP, applied along time dim):**
```
Wx: Linear(3, hidden//2)   # applied to x
Wy: Linear(3, hidden//2)   # applied to y
T  = tanh(Linear(hidden//2, 1)(ReLU(Wx(x) + Wy(y))))   # shape (B, 2, 1)
```

**MINE estimate (DV lower bound):**
```
Shuffle y along batch dim to get y_shuffle
x_conc = cat([x, x], dim=1)           # (B, 4, 3)  — joint + marginal
y_conc = cat([y, y_shuffle], dim=1)   # (B, 4, 3)
output = T_net(x_conc, y_conc)        # (B, 4, 1)
T_xy   = output[:, :2, :]             # joint
T_x_y  = output[:, 2:, :]            # marginal

mine_est = mean(T_xy) - log(mean(exp(T_x_y)))   # per sample → (B, 1)
neg_loss = -mine_est                             # minimize this
```

**Forward returns:** `neg_loss` per sample, shape `(B, 1)`.

### `Actor(nn.Module)`
Gaussian policy with tanh squashing (SAC-style).

**Input:** `concat(normalize(o), normalize(g))` — shape `(B, 25+3=28)`
**Architecture:** `Linear(28, 256) → ReLU → Linear(256, 256) → ReLU → Linear(256, act_dim*2)`
**Output:** `mu` (4), `log_std` (4, clamped to [-5, 2])

**Sampling (reparameterization):**
```
std = exp(log_std)
x_t = mu + eps * std           # eps ~ N(0,1)
y_t = tanh(x_t)
action = y_t * max_u           # scaled to [-max_u, max_u]
log_prob = gaussian_logprob(x_t, mu, log_std)
         - sum(log(1 - y_t^2 + 1e-6), dim=1)   # squashing correction
```

`get_action(o, g)` returns deterministic `tanh(mu) * max_u` (for rollouts/eval).

### `TwinQ(nn.Module)`
Two independent Q-networks. A frozen `target_critic` is maintained with Polyak averaging.

**Input:** `concat(normalize(o), normalize(g), a/max_u)` — shape `(B, 25+3+4=32)`
**Architecture:** Two separate `Linear(32,256)→ReLU→Linear(256,256)→ReLU→Linear(256,1)` heads
**Returns:** `Q1, Q2` both shape `(B, 1)`

Target network update: `θ_target ← polyak * θ_target + (1 - polyak) * θ`

### `EpisodeBuffer`
Stores complete episodes (not individual transitions).

**Storage shapes** (buffer_size_episodes = buffer_size // T):
```
o:  (max_episodes, T+1, 25)
ag: (max_episodes, T+1, 3)
g:  (max_episodes, T,   3)
u:  (max_episodes, T,   4)
```

**`store_episode(episode)`** — circular overwrite.

**`sample(batch_size) -> dict`** — HER "future" strategy:
1. Sample random (episode_idx, t) pairs
2. For each, with probability `future_p = 1 - 1/(1+replay_k)`, replace goal with `ag[episode_idx, t']` where `t' ~ Uniform(t+1, T)`
3. Compute sparse reward: `r = float(||ag_2 - g|| < threshold)`
4. Return `{o, o_2, ag, ag_2, g, u, r}` as numpy arrays

**`sample_mi_pairs(batch_size) -> np.ndarray`** — returns `(batch_size, 2, 25)` pairs of consecutive obs for MINE training.

---

## Intrinsic Reward

For each transition in a SAC training batch, MI reward is computed as:
```python
o_tau = np.stack([batch['o'], batch['o_2']], axis=1)  # (B, 2, 25)
with torch.no_grad():
    neg_loss = mine_net(o_tau)       # (B, 1)
mi_est = -neg_loss                   # MINE estimate
r_i = np.clip(mi_r_scale * mi_est, 0, 1)   # mi_r_scale=5000
```

No extrinsic reward is used during unsupervised training (`r_scale=0`).

---

## Loss Functions

### MINE Loss
```python
loss_mine = neg_loss.mean()   # minimize -(E[T_xy] - log(E[exp(T_x_y)]))
```

### Critic Loss (Bellman)
```python
with torch.no_grad():
    a_next, log_prob_next = actor.sample(o_2, g)
    Q1_t, Q2_t = target_critic(o_2, g, a_next)
    Q_target = r_i + gamma * (min(Q1_t, Q2_t) - alpha * log_prob_next)
    Q_target = clip(Q_target, -clip_return, clip_return)

Q1, Q2 = critic(o, g, a)
loss_critic = MSE(Q1, Q_target) + MSE(Q2, Q_target)
```

### Actor Loss
```python
a, log_prob = actor.sample(o, g)
Q1, Q2 = critic(o, g, a)
loss_actor = (alpha * log_prob - min(Q1, Q2)).mean()
```

### Entropy Temperature Loss
```python
target_entropy = -act_dim   # = -4
loss_alpha = (-log_alpha * (log_prob + target_entropy)).mean()
alpha = exp(log_alpha)
```

---

## Training Loop

```
for epoch in range(n_epochs=200):
    for cycle in range(n_cycles=50):

        # 1. Collect rollouts
        for _ in range(rollout_batch_size=2):
            obs, _ = env.reset()
            for t in range(T=50):
                action = actor.get_action(o, g)  + noise
                step env, store transition
            buffer.store_episode(episode)

        # 2. Update normalizer
        normalizer.update_from_buffer(recently_stored_episodes)
        normalizer.recompute_stats()

        # 3. Train MINE  (n_batches=40 steps)
        for _ in range(n_batches):
            o_tau = buffer.sample_mi_pairs(batch_size=256)
            loss = mine_net(o_tau).mean()
            optimizer_mine.zero_grad(); loss.backward(); optimizer_mine.step()

        # 4. Train SAC  (n_batches=40 steps)
        for _ in range(n_batches):
            batch = buffer.sample(batch_size=256)
            r_i = compute_mi_reward(mine_net, batch)
            update critic, actor, alpha
            soft_update(target_critic, critic, polyak=0.95)

    # 5. Evaluate
    evaluate(actor, normalizer, eval_env, n_episodes=10)
```

---

## Hyperparameters (from `SAC+MUSIC.json`)

| Parameter | Value |
|---|---|
| `mi_r_scale` | 5000 |
| `r_scale` | 0 (no task reward) |
| `gamma` | 0.98 |
| `polyak` | 0.95 (τ = 0.05) |
| `hidden` | 256 |
| `layers` | 3 |
| `batch_size` | 256 |
| `replay_k` | 4 |
| `T` (episode length) | 50 |
| `n_epochs` | 200 |
| `n_cycles` | 50 |
| `n_batches` | 40 |
| `rollout_batch_size` | 2 |
| `buffer_size` | 1,000,000 |
| `max_u` | 1.0 |
| `action_l2` | 1.0 |
| `clip_obs` | 200 |
| `clip_return` | 50 |

---

## File Layout

```
musicTorch/
├── music_sac.py          ← single implementation file
├── docs/plans/
│   └── 2026-03-03-music-u-pytorch-design.md
└── baselines/            ← original TF1 reference (unchanged)
```
