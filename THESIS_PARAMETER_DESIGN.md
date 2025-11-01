# Thesis-Appropriate Parameter Sweep Design

## Research Question:
"Under what conditions do PPO, TD3, and A2C converge reliably on benchmark tasks, and how do hyperparameters and architecture influence their stability?"

---

## Key Considerations for Fair Comparison:

### 1. **Shared Hyperparameters** (Test Across All Algorithms)
These parameters exist in all algorithms and should be tested identically:

- **Learning Rate**: Critical for convergence speed and stability
  - Values: `[1e-4, 3e-4, 1e-3]`
  - Rationale: Cover slow, standard, and fast learning

- **Gamma (Discount Factor)**: Affects long-term vs short-term reward balance
  - Values: `[0.99]` (standard) OR `[0.95, 0.99, 0.995]` for more thorough analysis
  - Rationale: 0.99 is standard, but testing range shows stability across horizons

### 2. **Algorithm-Specific Parameters** (Test Individually)

#### PPO-Specific:
- **clip_range**: `[0.1, 0.2, 0.3]` - Controls policy update magnitude
- **ent_coef**: `[0.0, 0.01]` - Exploration vs exploitation
- **batch_size**: `[64, 256]` - Training stability
- **n_steps**: `[2048]` (fixed) - Standard for continuous control

#### TD3-Specific:
- **batch_size**: `[64, 256]` - Training stability (same as PPO for fairness)
- **policy_delay**: `[2]` (fixed) - Standard TD3 setting
- **target_policy_noise**: `[0.1, 0.2]` - Exploration noise
- **tau**: `[0.005]` (fixed) - Standard soft update rate

#### A2C-Specific:
- **n_steps**: `[5, 16, 64]` - Rollout length (replaces batch_size concept)
- **ent_coef**: `[0.0, 0.01]` - Exploration vs exploitation (same as PPO)
- **vf_coef**: `[0.5]` (fixed) - Value function coefficient

#### SAC-Specific:
- **batch_size**: `[64, 256]` - Training stability (same as PPO for fairness)
- **tau**: `[0.005]` (fixed) - Standard soft update rate
- **ent_coef**: `['auto']` (adaptive) OR `[0.01, 0.1]` for testing
- **train_freq**: `[1]` (fixed) - Update every step

### 3. **Environments**
- **Hopper-v5**: Simple bipedal locomotion (tests basic convergence)
- **HalfCheetah-v5**: Faster quadrupedal locomotion (tests stability)

### 4. **Seeds**
- Use 5 seeds `[0, 1, 2, 3, 4]` for statistical significance

---

## Experiment Count:

### Minimal Design (Focused):
- PPO: 2 envs × 3 LR × 2 batch × 2 ent_coef × 5 seeds = **120 experiments**
- TD3: 2 envs × 3 LR × 2 batch × 5 seeds = **60 experiments**
- A2C: 2 envs × 3 LR × 3 n_steps × 2 ent_coef × 5 seeds = **180 experiments**
- SAC: 2 envs × 3 LR × 2 batch × 5 seeds = **60 experiments**

**Total: 420 experiments**

---

## What This Design Tests:

✅ **Convergence Reliability:**
- 5 seeds per configuration → statistical significance
- Multiple learning rates → convergence speed and stability across settings
- Different exploration levels → robustness to exploration strategy

✅ **Hyperparameter Influence:**
- Learning rate: Affects convergence speed and stability
- Batch size (PPO/TD3/SAC) vs n_steps (A2C): Training stability
- Entropy coefficient: Exploration vs exploitation trade-off

✅ **Algorithm-Specific Behavior:**
- PPO: clip_range effects on policy updates
- TD3: target noise effects on exploration
- A2C: rollout length effects on variance
- SAC: automatic entropy tuning vs fixed

✅ **Fair Comparison:**
- Same learning rates tested across all algorithms
- Equivalent "update batch" concepts (batch_size vs n_steps)
- Same environments and seeds
- Matched exploration parameters where applicable

---

## Current vs Recommended:

### Current Design Issues:
1. ❌ A2C uses batch_size (not valid)
2. ❌ Only 2 environments (limited generalization)
3. ⚠️ Fixed gamma (can't test discount factor effects)
4. ✅ Good seed count (5 seeds)
5. ✅ Good learning rate range

### Recommended Fixes:
1. ✅ Remove batch_size from A2C (DONE in train.py)
2. ✅ Use n_steps for A2C instead
3. ✅ Keep same learning rates across algorithms
4. ✅ Keep same seeds across algorithms
5. Optional: Add Walker2d-v5 for better generalization

---

## Implementation Status:

### Already Fixed:
- ✅ A2C batch_size removed in train.py
- ✅ 5 seeds for statistical significance
- ✅ Good learning rate range [1e-4, 3e-4, 1e-3]
- ✅ 2 environments (Hopper, HalfCheetah)

### Needs Verification:
- Check if current configs match this design
- Ensure all algorithm-specific parameters are appropriate
- Verify total experiment count

---

## Thesis Analysis Plan:

With this design, you can analyze:

1. **Convergence Reliability:**
   - Success rate per algorithm per configuration
   - Time to convergence
   - Final performance distribution

2. **Hyperparameter Effects:**
   - Learning rate impact on convergence
   - Batch size / n_steps impact on stability
   - Exploration coefficient impact on final performance

3. **Algorithm Comparisons:**
   - Which algorithm is most robust?
   - Which converges fastest?
   - Which achieves best final performance?
   - How sensitive is each to hyperparameters?

4. **Practical Recommendations:**
   - Best default hyperparameters per algorithm
   - When to use which algorithm
   - Which hyperparameters matter most
