"""
Run RL experiments in parallel across all CPU cores
Much faster than running sequentially!

FIXES:
- Output buffer overflow (was causing experiments to freeze)
- Default workers set to 12
"""
import json
import subprocess
import multiprocessing as mp
from pathlib import Path
import time

# Load generated configs
CONFIGS_DIR = Path("thesis_analysis/configs")
NUM_WORKERS = 12  # FIX: Set to 12 cores as requested

def run_single_experiment(config_info):
    """Run a single experiment with given config."""
    config, index = config_info
    algo = config["algorithm"]
    env = config["env"]
    seed = config["seed"]

    # Create a unique run name
    run_name = f"{algo}_{env}_lr{config['learning_rate']}_g{config['gamma']}_b{config['batch_size']}_s{seed}"

    print(f"[Worker {mp.current_process().name}] Starting: {run_name}")

    # Choose training script based on algorithm
    script_map = {
        "PPO": "experiments/train_ppo.py",
        "TD3": "experiments/train_td3.py",
        "A2C": "experiments/train_a2c.py",
    }

    script = script_map.get(algo)
    if not script:
        print(f"No training script for {algo}")
        return False

    # Build command to run training
    cmd = [
        "python", script,
        f"--env_id={env}",
        f"--seed={seed}",
        f"--learning_rate={config['learning_rate']}",
        f"--gamma={config['gamma']}",
        f"--batch_size={config['batch_size']}",
        f"--total_timesteps={config['total_timesteps']}",
        f"--wandb_project=rl",
        f"--wandb_run_name={run_name}",
    ]

    # Add algorithm-specific params
    if algo == "PPO":
        cmd.extend([
            f"--clip_range={config.get('clip_range', 0.2)}",
            f"--ent_coef={config.get('ent_coef', 0.0)}",
            f"--n_epochs={config.get('n_epochs', 10)}",
        ])
    elif algo == "TD3":
        cmd.extend([
            f"--tau={config.get('tau', 0.005)}",
            f"--policy_delay={config.get('policy_delay', 2)}",
        ])

    # Create log directory
    log_dir = Path("logs/subprocess")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{run_name}.log"

    try:
        start = time.time()

        # FIX: Don't capture output - write to file instead to prevent buffer overflow
        with open(log_file, 'w', buffering=1) as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                  text=True, timeout=7200)  # 2 hour timeout
        elapsed = time.time() - start

        if result.returncode == 0:
            print(f"✓ Completed: {run_name} in {elapsed/60:.1f}m")
            return True
        else:
            # Read last part of log for error
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    error_msg = ''.join(lines[-10:]) if lines else "Unknown"
            except:
                error_msg = "Could not read log"

            print(f"✗ Failed: {run_name}")
            print(f"  Log: {log_file}")
            return False
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout: {run_name} (>2 hours)")
        return False
    except Exception as e:
        print(f"✗ Error: {run_name} - {e}")
        return False


def main():
    print("=" * 70)
    print("PARALLEL EXPERIMENT RUNNER")
    print("=" * 70)

    # Load configs
    algo_configs = {}
    for algo in ["PPO", "TD3", "A2C"]:
        config_file = CONFIGS_DIR / f"{algo.lower()}_configs.json"
        if config_file.exists():
            with open(config_file) as f:
                algo_configs[algo] = json.load(f)
            print(f"Loaded {len(algo_configs[algo])} configs for {algo}")

    # Flatten all configs
    all_configs = []
    for algo, configs in algo_configs.items():
        all_configs.extend([(cfg, i) for i, cfg in enumerate(configs)])

    print(f"\nTotal experiments to run: {len(all_configs)}")
    print(f"Running with {NUM_WORKERS} parallel workers")
    print(f"Estimated time: {len(all_configs) / NUM_WORKERS * 20 / 60:.1f} hours")
    print("  (assuming ~20 min per experiment)")
    print("\nStarting experiments...\n")

    # Run experiments in parallel
    start_time = time.time()
    with mp.Pool(NUM_WORKERS) as pool:
        results = pool.map(run_single_experiment, all_configs)

    # Summary
    elapsed = time.time() - start_time
    successes = sum(results)
    failures = len(results) - successes

    print("\n" + "=" * 70)
    print("EXPERIMENT BATCH COMPLETE")
    print("=" * 70)
    print(f"Total time: {elapsed/3600:.1f} hours")
    print(f"Successes: {successes}")
    print(f"Failures: {failures}")
    print(f"\nData logged to wandb project: rl")


if __name__ == "__main__":
    mp.freeze_support()  # Required for Windows multiprocessing
    main()
