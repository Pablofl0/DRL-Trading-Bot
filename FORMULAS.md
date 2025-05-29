# Mathematical Formulations in the Trading Bot

This document explains how the formulas from the research paper "Automated Cryptocurrency Trading Bot Implementing DRL" are implemented in our codebase.

## PPO-CLIP Objective Function

From the paper, the PPO-CLIP objective function is defined as:

$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon)\hat{A}_t)]$$

where $\theta$ represents the parameter values (weights), $\pi_{\theta}$ is the policy, and $\hat{A}_t$ is an estimator of the advantage function at timestep $t$.

### Implementation

This formula is implemented in the `train` method of `PPOAgent` class in `src/models/ppo_agent.py`:

```python
# Calculate surrogate losses for PPO-CLIP objective
surrogate1 = ratio * batch_advantages
surrogate2 = tf.clip_by_value(
    ratio, 1 - self.epsilon, 1 + self.epsilon
) * batch_advantages

# PPO-CLIP objective (negative because we're minimizing)
actor_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
```

## Probability Ratio Calculation

The probability ratio between the updated policy network outputs and the old policy network outputs is given by:

$$r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

### Implementation

This formula is implemented in the `train` method of `PPOAgent` class:

```python
# Calculate ratio r_t(θ) = π_θ(at|st) / π_θ_old(at|st)
ratio = current_action_probs / (old_action_prob_values + 1e-8)
```

## Trading Mechanism Formulas

The paper defines three key formulas for trading:

### 1. Amount Bought

$$\text{Amount bought} = \frac{\text{Current net worth}}{\text{Current crypto closing price}}$$

### Implementation

This formula is implemented in both the environment (`src/env/crypto_env.py`) and live trading (`src/live_trading.py`):

```python
# Environment implementation
net_worth = self.balance + self.crypto_held * price
crypto_bought = net_worth / price

# Live trading implementation
net_worth = self.balance
amount = net_worth / current_price
```

### 2. Amount Sold

$$\text{Amount sold} = \text{Current crypto amount held} \times \text{Current crypto closing price}$$

### Implementation

```python
# Environment implementation
sell_amount = self.crypto_held * price

# Live trading implementation
sell_amount = self.crypto_held * current_price
```

### 3. Reward Function

$$\text{Reward} = \text{Current net worth} - \text{Previous net worth}$$

### Implementation

This is implemented in the `_calculate_reward` method of the `CryptoTradingEnv` class:

```python
new_net_worth = self.balance + self.crypto_held * self._get_current_price()
reward = new_net_worth - self.prev_net_worth
```

## Advantage Estimation

The paper uses advantage estimation to improve the policy updates. We implement Generalized Advantage Estimation (GAE) in the `_compute_advantage` method:

```python
delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
gae = delta + gamma * lam * (1 - dones[t]) * gae
advantages[t] = gae
```

## Notes on Implementation

- We add a small constant (1e-8) to denominators to prevent division by zero.
- The advantage values are normalized to stabilize training.
- We use TensorFlow's gradient tape for automatic differentiation.
- The PPO algorithm uses multiple epochs and mini-batches for each update, as recommended in the original PPO paper. 