"""
Generate thesis experiment configuration files.

Creates YAML config files for PPO, A2C, TD3, and SAC with systematic
hyperparameter sweeps across multiple seeds.
"""
import yaml
from pathlib import Path
from itertools import product

output_dir = Path("configs")
output_dir.mkdir(exist_ok=True)

ENVIRONMENTS = ["HalfCheetah-v5", "Hopper-v5"]
SEEDS = [0, 1, 2, 3, 4]
TOTAL_TIMESTEPS = 1000000
LEARNING_RATES = [1e-4, 3e-4, 1e-3]
BATCH_SIZES = [64, 256]

def generate_ppo_configs():
    """Generate PPO configurations."""
    configs = []

    for env, lr, bs, seed in product(ENVIRONMENTS, LEARNING_RATES, BATCH_SIZES, SEEDS):
        config = {
            'algorithm': 'ppo',
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
                'n_steps': 2048,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.0,
                'vf_coef': 0.5,
                'max_grad_norm': 0.5,
            }
        }

        lr_str = f"{lr:.0e}".replace('-0', 'm0').replace('e-', 'em')
        filename = f"ppo_{env}_{lr_str}_bs{bs}_s{seed}.yaml"
        configs.append((filename, config))

    return configs

def generate_a2c_configs():
    """Generate A2C configurations. A2C uses n_steps instead of batch_size."""
    configs = []
    N_STEPS_OPTIONS = [16, 32]

    for env, lr, n_steps, seed in product(ENVIRONMENTS, LEARNING_RATES, N_STEPS_OPTIONS, SEEDS):
        bs_equiv = 64 if n_steps == 16 else 256

        config = {
            'algorithm': 'a2c',
            'env_id': env,
            'seed': seed,
            'total_timesteps': TOTAL_TIMESTEPS,
            'policy': 'MlpPolicy',
            'verbose': 0,
            'checkpoint_freq': 0,
            'eval_freq': 0,
            'algo_params': {
                'learning_rate': lr,
                'n_steps': n_steps,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'ent_coef': 0.01,
                'vf_coef': 0.5,
                'max_grad_norm': 0.5,
            }
        }

        lr_str = f"{lr:.0e}".replace('-0', 'm0').replace('e-', 'em')
        filename = f"a2c_{env}_{lr_str}_bs{bs_equiv}_s{seed}.yaml"
        configs.append((filename, config))

    return configs

def generate_td3_configs():
    """Generate TD3 configurations."""
    configs = []

    for env, lr, bs, seed in product(ENVIRONMENTS, LEARNING_RATES, BATCH_SIZES, SEEDS):
        config = {
            'algorithm': 'td3',
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
                'gamma': 0.99,
                'tau': 0.005,
                'policy_delay': 2,
                'target_policy_noise': 0.2,
            }
        }

        lr_str = f"{lr:.0e}".replace('-0', 'm0').replace('e-', 'em')
        filename = f"td3_{env}_{lr_str}_bs{bs}_s{seed}.yaml"
        configs.append((filename, config))

    return configs

def generate_sac_configs():
    """Generate SAC configurations with automatic entropy tuning."""
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
    algorithms = {
        'PPO': generate_ppo_configs(),
        'A2C': generate_a2c_configs(),
        'TD3': generate_td3_configs(),
        'SAC': generate_sac_configs(),
    }

    total_configs = 0

    for algo_name, configs in algorithms.items():
        print(f"{algo_name}: {len(configs)} configs")

        for filename, config in configs:
            filepath = output_dir / filename
            with open(filepath, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        total_configs += len(configs)

    print(f"\nGenerated {total_configs} configuration files in {output_dir}")
    print(f"4 algorithms × 2 environments × 3 learning rates × 2 batch sizes × 5 seeds")

if __name__ == "__main__":
    main()
