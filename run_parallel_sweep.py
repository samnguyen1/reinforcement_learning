"""
Run hyperparameter sweep in parallel using all CPU cores
Uses your existing train.py script

FIXES:
- Output buffer overflow (was causing SAC to freeze)
- Default workers set to 12
- Better error logging with log files
"""
import os
import sys
import subprocess
import multiprocessing as mp
from pathlib import Path
import time
from datetime import datetime

NUM_WORKERS = 12  # FIX: Set to 12 cores as requested (was mp.cpu_count() - 1)

def run_single_config(config_path):
    """Run training for a single config file."""
    config_path = Path(config_path)
    name = config_path.stem

    print(f"[{mp.current_process().name}] Starting: {name}")

    cmd = [
        sys.executable,  # Use the same Python interpreter as the parent process
        "experiments/train.py",
        "--config", str(config_path),
        "--wandb",
        "--project", "rl1"
    ]

    # Create log directory
    log_dir = Path("logs/subprocess")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}.log"

    try:
        start = time.time()

        # FIX: Don't capture output - write to file instead to prevent buffer overflow
        # capture_output=True was causing SAC experiments to freeze when buffer filled up!
        # With 1M timesteps and verbose logging, this can be gigabytes of text
        with open(log_file, 'w', buffering=1) as f:  # Line buffering
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=21600,  # 6 hour timeout per experiment (SAC takes 3-4 hours)
                cwd=Path.cwd()  # Ensure subprocess runs in correct directory
            )
        elapsed = time.time() - start

        if result.returncode == 0:
            print(f"[OK] [{elapsed/60:.1f}m] {name}")
            return (name, "success", elapsed)
        else:
            # Read last part of log file for error message
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    error_msg = ''.join(lines[-10:]) if lines else "Unknown error"
            except:
                error_msg = "Could not read log file"

            print(f"[FAIL] {name}")
            print(f"   Log: {log_file}")
            return (name, "failed", elapsed)

    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] (>2h): {name}")
        return (name, "timeout", 7200)
    except Exception as e:
        print(f"[ERROR] {name} - {e}")
        return (name, "error", 0)


def main():
    print("=" * 70)
    print("PARALLEL HYPERPARAMETER SWEEP")
    print("=" * 70)

    # Load config list
    config_dir = Path("thesis_sweep_configs")
    config_list_file = config_dir / "config_list.txt"

    if not config_list_file.exists():
        print("ERROR: No config_list.txt found!")
        print("Run: python generate_smart_sweep.py first")
        return

    # Read all config paths
    configs = []
    with open(config_list_file) as f:
        for line in f:
            algo, path = line.strip().split('\t')
            configs.append(Path(path))

    print(f"\nFound {len(configs)} experiment configs")
    print(f"Running with {NUM_WORKERS} parallel workers (12 cores)")

    # Estimate time (adjusted to ~100 min per experiment based on real data)
    avg_time = 100  # minutes per experiment (SAC/TD3 are slower than PPO)
    total_sequential_hours = (len(configs) * avg_time) / 60
    total_parallel_hours = total_sequential_hours / NUM_WORKERS

    print(f"\nEstimated completion time: {total_parallel_hours:.1f} hours")
    print(f"  (Sequential would take: {total_sequential_hours:.1f} hours)")
    print(f"Subprocess logs will be saved to: logs/subprocess/")
    print("\nStarting experiments automatically...")

    # Run experiments in parallel
    print(f"\n{'='*70}")
    print("STARTING EXPERIMENTS")
    print(f"{'='*70}\n")

    start_time = time.time()
    start_datetime = datetime.now()

    # Use imap_unordered for better progress tracking (results arrive as they complete)
    with mp.Pool(NUM_WORKERS) as pool:
        results = list(pool.imap_unordered(run_single_config, configs, chunksize=1))

    # Summary
    elapsed = time.time() - start_time
    successes = sum(1 for _, status, _ in results if status == "success")
    failures = len(results) - successes

    print("\n" + "=" * 70)
    print("SWEEP COMPLETE!")
    print("=" * 70)
    print(f"Started:  {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {elapsed/3600:.1f} hours")
    print(f"\nResults:")
    print(f"  [OK] Successes: {successes}")
    print(f"  [FAIL] Failures:  {failures}")
    print(f"\nData logged to wandb project: rl1")

    # Save results log
    log_file = Path("sweep_results.txt")
    with open(log_file, 'w') as f:
        f.write(f"Sweep completed: {datetime.now()}\n")
        f.write(f"Duration: {elapsed/3600:.2f} hours\n")
        f.write(f"Successes: {successes}\n")
        f.write(f"Failures: {failures}\n\n")
        f.write("Individual results:\n")
        for name, status, time_taken in results:
            f.write(f"{status:10s} {time_taken/60:6.1f}m  {name}\n")

    print(f"\nDetailed results saved to: {log_file}")


if __name__ == "__main__":
    mp.freeze_support()  # Required for Windows multiprocessing
    main()
