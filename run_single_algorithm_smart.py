"""
Run experiments for a SINGLE algorithm at a time
SMART VERSION: Skips experiments that have already been completed

This version checks for existing data before running experiments:
- Checks monitor CSVs for completed runs
- Checks tensorboard logs
- Only runs experiments that are missing or incomplete
"""
import argparse
import os
import sys
import subprocess
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
import time
import yaml

def check_if_completed(config_path, min_episodes=50):
    """
    Check if this experiment has already been completed.

    Returns:
        (bool, str): (is_completed, reason)
    """
    config_path = Path(config_path)

    try:
        # Load config to get experiment details
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        env_id = config.get('env_id', 'unknown')
        seed = config.get('seed', 0)
        algo = config.get('algorithm', 'unknown')
        lr = config['algo_params'].get('learning_rate', 0)
        bs = config['algo_params'].get('batch_size', 0)

        # Generate expected directory names
        # Format 1: Simple naming (what exists now)
        simple_name = f"{algo}_{env_id}_seed{seed}"

        # Format 2: Detailed naming (from sweep configs)
        lr_str = f"{lr:.0e}".replace('-0', 'm0').replace('e-', 'em')
        detailed_name = f"{algo}_{env_id}_{lr_str}_bs{bs}_s{seed}"

        possible_names = [simple_name, detailed_name]

        # Check monitor CSVs
        for name in possible_names:
            monitor_csv = Path(f"logs/monitor/{name}/monitor.csv")
            if monitor_csv.exists():
                # Check if it has sufficient data
                try:
                    with open(monitor_csv, 'r') as f:
                        lines = f.readlines()
                        # First line is metadata, second is header, rest are data
                        num_episodes = len(lines) - 2

                        if num_episodes >= min_episodes:
                            return (True, f"Found {num_episodes} episodes in {monitor_csv}")
                        else:
                            return (False, f"Only {num_episodes} episodes in {monitor_csv} (need {min_episodes})")
                except:
                    pass

        # Check tensorboard logs (as secondary check)
        for name in possible_names:
            tb_dir = Path(f"logs/tensorboard/{name}")
            if tb_dir.exists():
                # Check if there are event files
                event_files = list(tb_dir.glob("**/events.out.tfevents.*"))
                if event_files:
                    # Could check file size/modification time, but monitor CSV is better indicator
                    pass

        return (False, "No existing data found")

    except Exception as e:
        return (False, f"Error checking: {e}")

def run_single_config(config_path):
    """Run training for a single config file."""
    config_path = Path(config_path)
    name = config_path.stem

    # Check if already completed
    is_completed, reason = check_if_completed(config_path)

    if is_completed:
        print(f"[SKIP] {name}: {reason}")
        return {'config': name, 'status': 'skipped', 'reason': reason, 'time': 0}
    else:
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
        with open(log_file, 'w', buffering=1) as f:  # Line buffering
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=57600,  # 16 hour timeout per experiment
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
        print(f"[TIMEOUT] (>16h): {name}")
        return {'config': name, 'status': 'timeout', 'time': 57600, 'log': str(log_file)}
    except Exception as e:
        print(f"[ERROR] {name} - {e}")
        return {'config': name, 'status': 'error', 'error': str(e), 'time': 0}

def main():
    parser = argparse.ArgumentParser(description="Run experiments for a single algorithm (SMART: skips completed)")
    parser.add_argument('algorithm', choices=['ppo', 'td3', 'a2c', 'sac'],
                        help='Which algorithm to run (ppo, td3, a2c, or sac)')
    parser.add_argument('--workers', type=int, default=12,
                        help='Number of parallel workers (default: 12)')
    parser.add_argument('--min-episodes', type=int, default=50,
                        help='Minimum episodes to consider run complete (default: 50)')
    parser.add_argument('--force', action='store_true',
                        help='Force rerun all experiments (ignore existing data)')
    args = parser.parse_args()

    # Get all configs for this algorithm
    config_dir = Path("thesis_sweep_configs")
    all_configs = sorted(config_dir.glob(f"{args.algorithm}_*.yaml"))

    if not all_configs:
        print(f"[ERROR] No configs found for algorithm: {args.algorithm}")
        sys.exit(1)

    # Pre-check which experiments need to be run
    if not args.force:
        print("="*70)
        print("CHECKING EXISTING DATA")
        print("="*70)

        to_run = []
        already_done = []

        for config_path in all_configs:
            is_completed, reason = check_if_completed(config_path, args.min_episodes)
            if is_completed:
                already_done.append((config_path.stem, reason))
            else:
                to_run.append(config_path)

        print(f"\nFound {len(already_done)} completed experiments (will skip)")
        print(f"Found {len(to_run)} experiments to run")

        if already_done:
            print("\nAlready completed:")
            for name, reason in already_done[:10]:  # Show first 10
                print(f"  [OK] {name}")
            if len(already_done) > 10:
                print(f"  ... and {len(already_done) - 10} more")

        if not to_run:
            print("\n[INFO] All experiments already completed!")
            print("Use --force to rerun all experiments")
            return

        all_configs = to_run

    print("\n" + "="*70)
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
    skipped = sum(1 for r in results if r['status'] == 'skipped')
    failed = sum(1 for r in results if r['status'] not in ['success', 'skipped'])

    print(f"Algorithm: {args.algorithm.upper()}")
    print(f"Total: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Skipped (already done): {skipped}")
    print(f"Failed: {failed}")
    print(f"Duration: {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if failed > 0:
        print()
        print("Failed experiments:")
        for r in results:
            if r['status'] not in ['success', 'skipped']:
                print(f"  - {r['config']}: {r['status']}")
                if 'log' in r:
                    print(f"    Log: {r['log']}")

    print("="*70)

    if failed == 0 and successful > 0:
        print(f"\n[OK] All new {args.algorithm.upper()} experiments completed successfully!")
        print(f"View results: https://wandb.ai/samnguyen1/rl1")
    elif skipped > 0 and successful == 0 and failed == 0:
        print(f"\n[INFO] All experiments were already completed (skipped {skipped})")
    elif failed > 0:
        print(f"\n[WARNING] {failed} experiments failed.")
        print(f"Check logs in: logs/subprocess/")

    # Save detailed log
    log_file = Path(f"{args.algorithm}_results.txt")
    with open(log_file, 'w') as f:
        f.write(f"Algorithm: {args.algorithm.upper()}\n")
        f.write(f"Completed: {datetime.now()}\n")
        f.write(f"Duration: {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)\n")
        f.write(f"Successful: {successful}/{len(results)}\n")
        f.write(f"Skipped: {skipped}/{len(results)}\n")
        f.write(f"Failed: {failed}/{len(results)}\n\n")
        f.write("Individual results:\n")
        for r in results:
            status_str = f"{r['status']:10s}"
            time_str = f"{r['time']/60:6.1f}m"
            if r['status'] == 'skipped':
                extra = f"  ({r.get('reason', 'already completed')})"
            elif r['status'] not in ['success', 'skipped'] and 'log' in r:
                extra = f"  (log: {r['log']})"
            else:
                extra = ""
            f.write(f"{status_str} {time_str}  {r['config']}{extra}\n")

    print(f"\nDetailed log saved to: {log_file}")

if __name__ == "__main__":
    mp.freeze_support()
    main()
