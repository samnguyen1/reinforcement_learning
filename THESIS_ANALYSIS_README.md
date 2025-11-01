# Thesis Analysis Pipeline

This directory contains tools to analyze your RL experiments and answer your thesis question:

**"Under what conditions do PPO, TD3, A2C, and SAC converge reliably on benchmark tasks, and how do hyperparameters and architecture influence their stability?"**

## Quick Start

```bash
# 1. Extract metrics from all wandb runs
python thesis_analysis_pipeline.py

# 2. Create visualizations
python thesis_visualizations.py
```

## What Gets Analyzed

### 1. Convergence Metrics
- **Final reward** (mean ± std across last 10 episodes)
- **Convergence speed** (steps to reach 80% of final performance)
- **Maximum reward achieved**
- **Mean reward throughout training**

### 2. Reliability Metrics
- **Across-seed variance** (how consistent is the algorithm?)
- **Coefficient of Variation** (CV = std / mean, lower = more reliable)
- **Failure rate** (how often does it fail to converge?)

### 3. Hyperparameter Sensitivity
- Performance vs **Learning Rate** (1e-4, 3e-4, 1e-3)
- Performance vs **Batch Size** (64, 256)
- Interaction effects

### 4. Algorithm-Specific Metrics (PPO)
- **KL Divergence** (should stay ~0.01-0.02)
- **Clip Fraction** (should be 0.1-0.3)
- **Value Loss** stability

## Output Files

All files are saved to `thesis_outputs/`:

### CSV Files (Data)
- `all_metrics.csv` - Raw metrics for every run
- `convergence_analysis.csv` - Convergence statistics by config
- `reliability_analysis.csv` - Reliability metrics by algorithm
- `sensitivity_analysis.csv` - Hyperparameter sensitivity scores

### PNG Files (Visualizations)
- `learning_curves_comparison.png` - Algorithm comparison
- `reliability_analysis.png` - Coefficient of variation plots
- `heatmap_[algo]_[env].png` - Hyperparameter sensitivity heatmaps
- `boxplots_final_rewards.png` - Distribution of final rewards
- `convergence_speed.png` - Steps to convergence comparison

## How This Answers Your Thesis Question

### "Under what conditions..."
’ Identified via patterns in success/failure across different hyperparameter configs

**Example findings:**
- PPO converges reliably with LR=3e-4, BS=256 on HalfCheetah
- A2C fails with LR=1e-3 (too high learning rate)
- SAC is robust to batch size but sensitive to learning rate

### "...converge reliably..."
’ Measured by low variance across seeds + stable learning curves

**Metrics used:**
- Coefficient of Variation (CV < 0.1 = reliable)
- Across-seed standard deviation
- Failure rate (% of runs that didn't converge)

### "...hyperparameters influence..."
’ Mapped through performance shifts across parameter sweeps

**Analysis:**
- Heatmaps show performance across LR × BS grid
- Sensitivity scores quantify how much each hyperparameter matters
- Best configs identified for each algorithm + environment

## Example Usage

```python
# Load the analysis results
import pandas as pd

# Get all metrics
df = pd.read_csv('thesis_outputs/all_metrics.csv')

# Find best PPO config for HalfCheetah
ppo_halfcheetah = df[(df['algorithm'] == 'ppo') &
                     (df['environment'] == 'HalfCheetah-v5')]
best_config = ppo_halfcheetah.groupby(['learning_rate', 'batch_size'])['final_reward_mean'].mean().idxmax()
print(f"Best PPO config: LR={best_config[0]}, BS={best_config[1]}")

# Compare algorithm reliability
reliability = df.groupby('algorithm')['final_reward_mean'].std()
print(f"Most reliable algorithm: {reliability.idxmin()}")
```

## Next Steps for Thesis

1. **Run remaining algorithms** (A2C, SAC)
2. **Re-run analysis** after each algorithm completes
3. **Export plots** to your thesis document
4. **Write up findings** using the metrics from CSV files

## Troubleshooting

**Q: "No history for run X" warnings**
A: Some runs may have failed or not logged properly. They'll be skipped automatically.

**Q: Plot shows only one algorithm**
A: You need to complete experiments for multiple algorithms first.

**Q: Heatmap is empty**
A: Need at least 4 different hyperparameter combinations to create a heatmap.

## Current Status

**Completed:**
- PPO: 60 runs (2 environments × 5 seeds × 6 hyperparameter configs)

**Remaining:**
- A2C: 60 runs (~14 hours)
- SAC: 60 runs (~14 hours)
- TD3: 60 runs (~180 hours - consider skipping due to time)

**Total experiments:** 60-180 (depending on whether you include TD3)
