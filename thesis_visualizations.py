"""
Create visualizations for thesis
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Load data
output_dir = Path("thesis_outputs")
df = pd.read_csv(output_dir / 'all_metrics.csv')

print("="*70)
print("CREATING THESIS VISUALIZATIONS")
print("="*70)

# ===========================================================================
# Plot 1: Learning Curves Comparison
# ===========================================================================
print("\n1. Creating learning curves comparison...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Learning Curves by Algorithm and Environment', fontsize=16)

for idx, env in enumerate(df['environment'].unique()):
    ax = axes[idx // 2, idx % 2]

    for algo in df['algorithm'].unique():
        algo_env_data = df[(df['algorithm'] == algo) & (df['environment'] == env)]

        if len(algo_env_data) == 0:
            continue

        # Calculate mean and std across seeds
        mean_rewards = algo_env_data.groupby(['learning_rate', 'batch_size'])['final_reward_mean'].mean()
        std_rewards = algo_env_data.groupby(['learning_rate', 'batch_size'])['final_reward_mean'].std()

        ax.bar(algo, mean_rewards.mean(), yerr=std_rewards.mean(), label=algo.upper(), alpha=0.7)

    ax.set_title(f'{env}')
    ax.set_ylabel('Final Reward')
    ax.set_xlabel('Algorithm')
    ax.legend()

plt.tight_layout()
plt.savefig(output_dir / 'learning_curves_comparison.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {output_dir / 'learning_curves_comparison.png'}")

# ===========================================================================
# Plot 2: Reliability (Variance Across Seeds)
# ===========================================================================
print("\n2. Creating reliability analysis...")

fig, ax = plt.subplots(figsize=(12, 6))

reliability_data = []
for algo in df['algorithm'].unique():
    for env in df['environment'].unique():
        algo_env_data = df[(df['algorithm'] == algo) & (df['environment'] == env)]

        if len(algo_env_data) == 0:
            continue

        # Calculate coefficient of variation
        mean_reward = algo_env_data['final_reward_mean'].mean()
        std_reward = algo_env_data['final_reward_mean'].std()
        cv = std_reward / (mean_reward + 1e-6)

        reliability_data.append({
            'Algorithm': algo.upper(),
            'Environment': env.split('-')[0],
            'CV': cv
        })

reliability_df = pd.DataFrame(reliability_data)

sns.barplot(data=reliability_df, x='Algorithm', y='CV', hue='Environment', ax=ax)
ax.set_title('Reliability: Coefficient of Variation (Lower = More Reliable)')
ax.set_ylabel('Coefficient of Variation')
ax.axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='Target CV=0.1')
ax.legend()

plt.tight_layout()
plt.savefig(output_dir / 'reliability_analysis.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {output_dir / 'reliability_analysis.png'}")

# ===========================================================================
# Plot 3: Hyperparameter Sensitivity
# ===========================================================================
print("\n3. Creating hyperparameter sensitivity heatmaps...")

for algo in df['algorithm'].unique():
    for env in df['environment'].unique():
        algo_env_data = df[(df['algorithm'] == algo) & (df['environment'] == env)]

        if len(algo_env_data) < 4:
            continue

        # Create pivot table
        pivot = algo_env_data.pivot_table(
            values='final_reward_mean',
            index='learning_rate',
            columns='batch_size',
            aggfunc='mean'
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax)
        ax.set_title(f'{algo.upper()} on {env}: Hyperparameter Sensitivity')
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Learning Rate')

        plt.tight_layout()
        filename = f'heatmap_{algo}_{env.split("-")[0]}.png'
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"   Saved: {output_dir / filename}")
        plt.close()

# ===========================================================================
# Plot 4: Box Plots (Distribution of Final Rewards)
# ===========================================================================
print("\n4. Creating boxplots of final rewards...")

fig, axes = plt.subplots(1, len(df['environment'].unique()), figsize=(16, 6))

for idx, env in enumerate(df['environment'].unique()):
    ax = axes[idx] if len(df['environment'].unique()) > 1 else axes

    env_data = df[df['environment'] == env]

    # Prepare data for boxplot
    boxplot_data = []
    labels = []
    for algo in sorted(env_data['algorithm'].unique()):
        algo_rewards = env_data[env_data['algorithm'] == algo]['final_reward_mean'].values
        boxplot_data.append(algo_rewards)
        labels.append(algo.upper())

    ax.boxplot(boxplot_data, labels=labels)
    ax.set_title(f'{env}')
    ax.set_ylabel('Final Reward')
    ax.set_xlabel('Algorithm')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'boxplots_final_rewards.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {output_dir / 'boxplots_final_rewards.png'}")

# ===========================================================================
# Plot 5: Convergence Speed
# ===========================================================================
print("\n5. Creating convergence speed analysis...")

fig, ax = plt.subplots(figsize=(12, 6))

convergence_data = []
for algo in df['algorithm'].unique():
    algo_data = df[df['algorithm'] == algo]
    avg_steps = algo_data['steps_to_threshold'].mean()
    std_steps = algo_data['steps_to_threshold'].std()

    convergence_data.append({
        'Algorithm': algo.upper(),
        'Avg Steps': avg_steps,
        'Std Steps': std_steps
    })

convergence_df = pd.DataFrame(convergence_data)

ax.bar(convergence_df['Algorithm'], convergence_df['Avg Steps'],
       yerr=convergence_df['Std Steps'], capsize=5, alpha=0.7)
ax.set_title('Convergence Speed (Steps to Reach 80% Final Performance)')
ax.set_ylabel('Steps (thousands)')
ax.set_xlabel('Algorithm')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))

plt.tight_layout()
plt.savefig(output_dir / 'convergence_speed.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {output_dir / 'convergence_speed.png'}")

print("\n" + "="*70)
print("All visualizations created successfully!")
print(f"Check the {output_dir}/ folder for all plots")
print("="*70)
