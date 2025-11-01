"""
Run experiments for a SINGLE algorithm at a time
This allows you to run one algorithm, verify it works, then run the next

FIXES:
- Output buffer overflow (was causing freezing)
- Default workers set to 12
- Better error logging
"""
import argparse
import os
import sys
import subprocess
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
import time

def run_single_config(config_path):
    """Run training for a single config file."""
    config_path = Path(config_path)
    name = config_path.stem

    print(f"[{mp.current_process().name}] Starting: {name}")

    cmd = [
        sys.executable,
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
        # capture_output=True was causing processes to freeze when buffer filled up!
        # With 1M timesteps and verbose logging, this can be gigabytes of text
        with open(log_file, 'w', buffering=1) as f:  # Line buffering
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=36000,  # 6 hour timeout per experiment (SAC takes 3-4 hours)
                cwd=Path.cwd()
            )
        elapsed = time.time() - start

        if result.returncode == 0:
            print(f"[OK] [{elapsed/60:.1f}m] {name}")
            return {'config': name, 'status': 'success', 'time': elapsed}
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
            return {'config': name, 'status': 'failed', 'error': error_msg[:200], 'time': elapsed, 'log': str(log_file)}

    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] (>2h): {name}")
        return {'config': name, 'status': 'timeout', 'time': 7200, 'log': str(log_file)}
    except Exception as e:
        print(f"[ERROR] {name} - {e}")
        return {'config': name, 'status': 'error', 'error': str(e), 'time': 0}

def main():
    parser = argparse.ArgumentParser(description="Run experiments for a single algorithm")
    parser.add_argument('algorithm', choices=['ppo', 'td3', 'a2c', 'sac'],
                        help='Which algorithm to run (ppo, td3, a2c, or sac)')
    parser.add_argument('--workers', type=int, default=12,
                        help='Number of parallel workers (default: 12)')
    args = parser.parse_args()

    # Get all configs for this algorithm
    config_dir = Path("thesis_sweep_configs")
    all_configs = sorted(config_dir.glob(f"{args.algorithm}_*.yaml"))

    if not all_configs:
        print(f"[ERROR] No configs found for algorithm: {args.algorithm}")
        sys.exit(1)

    print("="*70)
    print(f"RUNNING {args.algorithm.upper()} EXPERIMENTS")
    print("="*70)
    print(f"Algorithm: {args.algorithm.upper()}")
    print(f"Total experiments: {len(all_configs)}")
    print(f"Parallel workers: {args.workers}")
    print(f"Estimated time: {len(all_configs) * 100 / args.workers / 60:.1f} hours")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Subprocess logs: logs/subprocess/")
    print("="*70)
    print()

    start_time = time.time()

    # Run experiments in parallel
    if args.workers == 1:
        # Sequential execution
        results = []
        for config_path in all_configs:
            result = run_single_config(str(config_path))
            results.append(result)
    else:
        # Parallel execution with dynamic load balancing
        # Using imap_unordered with chunksize=1 for immediate work distribution
        with mp.Pool(processes=args.workers) as pool:
            config_paths = [str(p) for p in all_configs]
            results = list(pool.imap_unordered(run_single_config, config_paths, chunksize=1))

    # Summary
    elapsed = time.time() - start_time
    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] != 'success')

    print(f"Algorithm: {args.algorithm.upper()}")
    print(f"Total: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Duration: {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if failed > 0:
        print()
        print("Failed experiments:")
        for r in results:
            if r['status'] != 'success':
                print(f"  - {r['config']}: {r['status']}")
                if 'log' in r:
                    print(f"    Log: {r['log']}")

    print("="*70)

    if failed == 0:
        print(f"\n[OK] All {args.algorithm.upper()} experiments completed successfully!")
        print(f"View results: https://wandb.ai/samnguyen1/rl1")
    else:
        print(f"\n[WARNING] {failed} experiments failed.")
        print(f"Check logs in: logs/subprocess/")

    # Save detailed log
    log_file = Path(f"{args.algorithm}_results.txt")
    with open(log_file, 'w') as f:
        f.write(f"Algorithm: {args.algorithm.upper()}\n")
        f.write(f"Completed: {datetime.now()}\n")
        f.write(f"Duration: {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)\n")
        f.write(f"Successful: {successful}/{len(results)}\n\n")
        f.write("Individual results:\n")
        for r in results:
            status_str = f"{r['status']:10s}"
            time_str = f"{r['time']/60:6.1f}m"
            log_str = f"  (log: {r['log']})" if 'log' in r and r['status'] != 'success' else ""
            f.write(f"{status_str} {time_str}  {r['config']}{log_str}\n")

    print(f"\nDetailed log saved to: {log_file}")

if __name__ == "__main__":
    mp.freeze_support()
    main()
