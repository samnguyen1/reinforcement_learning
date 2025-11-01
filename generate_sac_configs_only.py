"""
Generate SAC configuration files only

SAC (Soft Actor-Critic) specific features:
- Automatic entropy tuning (ent_coef: 'auto')
- Off-policy algorithm with replay buffer
- Twin Q-networks (like TD3)
- Stochastic policy (unlike TD3's deterministic)
"""
import yaml
from pathlib import Path
from itertools import product

# Output directory
output_dir = Path("thesis_sweep_configs")
output_dir.mkdir(exist_ok=True)

# Experiment settings
ENVIRONMENTS = ["HalfCheetah-v5", "Hopper-v5"]
SEEDS = [0, 1, 2, 3, 4]
TOTAL_TIMESTEPS = 1000000
LEARNING_RATES = [1e-4, 3e-4, 1e-3]
BATCH_SIZES = [64, 256]

def generate_sac_configs():
    """Generate SAC configurations with correct SAC-specific parameters"""
    configs = []

    for env, lr, bs, seed in product(ENVIRONMENTS, LEARNING_RATES, BATCH_SIZES, SEEDS):
        config = {
            'algorithm': 'sac',
            'env_id': env,
            'seed': seed,
            'total_timesteps': TOTAL_TIMESTEPS,
            'policy': 'MlpPolicy',
            'verbose': 0,
            'checkpoint_freq': 0,
            'eval_freq': 0,
            'algo_params': {
                'learning_rate': lr,
                'batch_size': bs,
                'buffer_size': 1000000,
                'learning_starts': 10000,
                'gamma': 0.99,
                'tau': 0.005,
                'train_freq': 1,
                'gradient_steps': 1,
                'ent_coef': 'auto',
                'target_update_interval': 1,
                'use_sde': False,
                'use_sde_at_warmup': False,
            }
        }

        lr_str = f"{lr:.0e}".replace('-0', 'm0').replace('e-', 'em')
        filename = f"sac_{env}_{lr_str}_bs{bs}_s{seed}.yaml"
        configs.append((filename, config))

    return configs

def main():
    print("="*70)
    print("GENERATING SAC CONFIGURATION FILES")
    print("="*70)

    configs = generate_sac_configs()

    print(f"\nGenerating {len(configs)} SAC configs...")

    for filename, config in configs:
        filepath = output_dir / filename
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\nGenerated {len(configs)} SAC configuration files")
    print(f"  Output directory: {output_dir}")

    print("\nBreakdown:")
    print(f"  - 2 environments (HalfCheetah-v5, Hopper-v5)")
    print(f"  - 3 learning rates (1e-4, 3e-4, 1e-3)")
    print(f"  - 2 batch sizes (64, 256)")
    print(f"  - 5 seeds (0-4)")
    print(f"  = {len(configs)} total SAC experiments")

    print("\nSAC-specific parameters:")
    print("  - ent_coef: 'auto' (automatic entropy tuning)")
    print("  - buffer_size: 1M (replay buffer)")
    print("  - learning_starts: 10k (warmup period)")
    print("  - train_freq: 1 (update every step)")
    print("  - gradient_steps: 1 (one gradient step per update)")

if __name__ == "__main__":
    main()
