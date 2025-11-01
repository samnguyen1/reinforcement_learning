"""
Export all wandb data for thesis analysis
"""
import wandb
import pandas as pd
from pathlib import Path
import os

print("="*70)
print("EXPORTING THESIS DATA FROM WANDB")
print("="*70)

api = wandb.Api()
runs = list(api.runs('samnguyen1/rl1'))

# Create export directory
export_dir = Path("thesis_data_export")
export_dir.mkdir(exist_ok=True)

print(f"\nExporting {len(runs)} runs to {export_dir}/")
print()

all_summaries = []

for i, run in enumerate(runs, 1):
    if i % 10 == 0:
        print(f"  Exported {i}/{len(runs)} runs...")

    # Get run summary
    summary = {
        'run_name': run.name,
        'run_id': run.id,
        'algorithm': run.config.get('algorithm', 'ppo'),
        'environment': run.config.get('env_id', ''),
        'seed': run.config.get('seed', 0),
        'learning_rate': run.config.get('learning_rate', 0),
        'batch_size': run.config.get('batch_size', 0),
        'total_timesteps': run.summary.get('global_step', 0),
        'runtime_seconds': run.summary.get('_runtime', 0),
        'runtime_minutes': run.summary.get('_runtime', 0) / 60,
        'final_ep_len_mean': run.summary.get('rollout/ep_len_mean', None),
        'state': run.state,
        'created_at': run.created_at,
        'url': run.url
    }
    all_summaries.append(summary)

    # Export full training history
    try:
        history = run.history()
        if len(history) > 0:
            csv_path = export_dir / f"{run.name}_{run.id}_history.csv"
            history.to_csv(csv_path, index=False)
    except Exception as e:
        print(f"  [WARNING] Could not export history for {run.name}: {e}")

# Save summary of all runs
summary_df = pd.DataFrame(all_summaries)
summary_path = export_dir / "all_runs_summary.csv"
summary_df.to_csv(summary_path, index=False)

print(f"\n[OK] Exported all runs!")
print()
print("="*70)
print("EXPORT SUMMARY")
print("="*70)
print(f"Location: {export_dir.absolute()}")
print(f"Total runs: {len(runs)}")
print(f"Files created:")
print(f"  - all_runs_summary.csv (overview of all runs)")
print(f"  - {len(runs)} individual history CSV files")
print()
print("You can now use this data for your thesis analysis!")
print("="*70)

# Print breakdown
print("\nBreakdown by environment:")
for env in summary_df['environment'].unique():
    count = len(summary_df[summary_df['environment'] == env])
    print(f"  {env}: {count} runs")
