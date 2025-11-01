"""
Comprehensive Thesis Analysis Pipeline
Extracts all metrics to answer: "Under what conditions do PPO, TD3, A2C, and SAC
converge reliably on benchmark tasks, and how do hyperparameters influence stability?"
"""
import wandb
import pandas as pd
import numpy as np
from pathlib import Path

# Create output directory
output_dir = Path("thesis_outputs")
output_dir.mkdir(exist_ok=True)

print("="*70)
print("THESIS ANALYSIS PIPELINE")
print("="*70)

# Connect to wandb
api = wandb.Api()
runs = list(api.runs('samnguyen1/rl1'))

print(f"\nTotal runs found: {len(runs)}")

# ===========================================================================
# STEP 1: Extract All Metrics from Each Run
# ===========================================================================
print("\n" + "="*70)
print("STEP 1: Extracting Metrics from All Runs")
print("="*70)

all_run_data = []

for i, run in enumerate(runs, 1):
    if i % 10 == 0:
        print(f"  Processing run {i}/{len(runs)}...")

    try:
        # Get run metadata
        algo = run.config.get('algorithm', 'unknown')
        env = run.config.get('env_id', 'unknown')
        seed = run.config.get('seed', 0)
        lr = run.config.get('learning_rate', 0)
        bs = run.config.get('batch_size', 0)

        # Get training history
        history = run.history()

        if len(history) == 0:
            print(f"  WARNING: No history for run {run.name}")
            continue

        # Extract episode rewards
        reward_col = None
        for col in ['rollout/ep_rew_mean', 'ep_rew_mean', 'episode_reward', 'reward']:
            if col in history.columns:
                reward_col = col
                break

        if reward_col is None:
            print(f"  WARNING: No reward column found for {run.name}")
            continue

        rewards = history[reward_col].dropna()

        if len(rewards) == 0:
            continue

        # Calculate metrics
        final_rewards = rewards.tail(10)  # Last 10 episodes
        final_mean = final_rewards.mean()
        final_std = final_rewards.std()

        # Convergence speed: steps to reach 80% of final performance
        threshold = final_mean * 0.8
        steps_to_threshold = None
        for idx, reward in enumerate(rewards):
            if reward >= threshold:
                steps_to_threshold = idx * 2048  # Approximate steps
                break

        # Learning curve smoothness (variance in second half of training)
        second_half = rewards[len(rewards)//2:]
        smoothness = second_half.std()

        # Overall statistics
        max_reward = rewards.max()
        min_reward = rewards.min()
        mean_reward = rewards.mean()

        # PPO-specific metrics
        kl_divergence = None
        clip_fraction = None
        value_loss = None

        if 'train/approx_kl' in history.columns:
            kl_divergence = history['train/approx_kl'].dropna().mean()
        if 'train/clip_fraction' in history.columns:
            clip_fraction = history['train/clip_fraction'].dropna().mean()
        if 'train/value_loss' in history.columns:
            value_loss = history['train/value_loss'].dropna().mean()

        # Entropy (for algorithms that track it)
        entropy = None
        if 'train/entropy_loss' in history.columns:
            entropy = history['train/entropy_loss'].dropna().mean()

        all_run_data.append({
            'algorithm': algo,
            'environment': env,
            'seed': seed,
            'learning_rate': lr,
            'batch_size': bs,
            'final_reward_mean': final_mean,
            'final_reward_std': final_std,
            'max_reward': max_reward,
            'min_reward': min_reward,
            'mean_reward': mean_reward,
            'steps_to_threshold': steps_to_threshold,
            'smoothness': smoothness,
            'kl_divergence': kl_divergence,
            'clip_fraction': clip_fraction,
            'value_loss': value_loss,
            'entropy': entropy,
            'total_steps': run.summary.get('global_step', 0),
            'runtime_minutes': run.summary.get('_runtime', 0) / 60,
            'run_name': run.name,
            'run_id': run.id
        })

    except Exception as e:
        print(f"  ERROR processing run {run.name}: {e}")

# Create DataFrame
df = pd.DataFrame(all_run_data)
df.to_csv(output_dir / 'all_metrics.csv', index=False)

print(f"\nExtracted metrics from {len(df)} runs")
print(f"Saved to: {output_dir / 'all_metrics.csv'}")

# ===========================================================================
# STEP 2: Convergence Analysis
# ===========================================================================
print("\n" + "="*70)
print("STEP 2: Convergence Analysis")
print("="*70)

convergence_results = []

for algo in df['algorithm'].unique():
    for env in df['environment'].unique():
        algo_env_data = df[(df['algorithm'] == algo) & (df['environment'] == env)]

        if len(algo_env_data) == 0:
            continue

        # Group by hyperparameters
        for (lr, bs), group in algo_env_data.groupby(['learning_rate', 'batch_size']):
            if len(group) < 2:
                continue

            convergence_results.append({
                'algorithm': algo,
                'environment': env,
                'learning_rate': lr,
                'batch_size': bs,
                'final_reward_mean': group['final_reward_mean'].mean(),
                'final_reward_std': group['final_reward_mean'].std(),
                'across_seed_variance': group['final_reward_mean'].std(),
                'num_seeds': len(group),
                'convergence_rate': group['steps_to_threshold'].mean(),
                'stability_score': 1.0 / (group['smoothness'].mean() + 1e-6)
            })

convergence_df = pd.DataFrame(convergence_results)
convergence_df.to_csv(output_dir / 'convergence_analysis.csv', index=False)

print(f"\nConvergence analysis complete")
print(f"Saved to: {output_dir / 'convergence_analysis.csv'}")

# Print summary
print("\nConvergence Summary by Algorithm:")
for algo in convergence_df['algorithm'].unique():
    algo_data = convergence_df[convergence_df['algorithm'] == algo]
    print(f"\n{algo.upper()}:")
    print(f"  Avg Final Reward: {algo_data['final_reward_mean'].mean():.2f} +/- {algo_data['final_reward_std'].mean():.2f}")
    print(f"  Across-Seed Variance: {algo_data['across_seed_variance'].mean():.2f}")
    print(f"  Configs Tested: {len(algo_data)}")

print("\n" + "="*70)
print("Analysis complete! Files saved to thesis_outputs/")
print("Run thesis_visualizations.py next to create plots")
print("="*70)
