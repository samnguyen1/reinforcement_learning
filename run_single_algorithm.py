"""Run experiments for a single algorithm with parallel execution."""
import argparse
import os
import sys
import subprocess
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
import time

def run_single_config(config_path):
    """Run training for a single configuration file."""
    config_path = Path(config_path)
    name = config_path.stem

    print(f"Starting: {name}")

    cmd = [
        sys.executable,
        "experiments/train.py",
        "--config", str(config_path),
        "--wandb",
        "--project", "rl1"
    ]

    log_dir = Path("logs/subprocess")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}.log"

    try:
        start = time.time()

        with open(log_file, 'w', buffering=1) as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=36000,
                cwd=Path.cwd()
            )
        elapsed = time.time() - start

        if result.returncode == 0:
            print(f"OK [{elapsed/60:.1f}m] {name}")
            return {'config': name, 'status': 'success', 'time': elapsed}
        else:
            print(f"FAIL {name} -> {log_file}")
            return {'config': name, 'status': 'failed', 'time': elapsed, 'log': str(log_file)}

    except subprocess.TimeoutExpired:
        print(f"TIMEOUT {name}")
        return {'config': name, 'status': 'timeout', 'time': 36000, 'log': str(log_file)}
    except Exception as e:
        print(f"ERROR {name}: {e}")
        return {'config': name, 'status': 'error', 'error': str(e), 'time': 0}

def main():
    parser = argparse.ArgumentParser(description="Run experiments for a single algorithm")
    parser.add_argument('algorithm', choices=['ppo', 'td3', 'a2c', 'sac'],
                        help='Algorithm to run')
    parser.add_argument('--workers', type=int, default=6,
                        help='Number of parallel workers (default: 6)')
    args = parser.parse_args()

    config_dir = Path("configs")
    all_configs = sorted(config_dir.glob(f"{args.algorithm}_*.yaml"))

    if not all_configs:
        print(f"No configs found for {args.algorithm}")
        sys.exit(1)

    print(f"\n{args.algorithm.upper()}: {len(all_configs)} experiments, {args.workers} workers")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}\n")

    start_time = time.time()

    if args.workers == 1:
        results = []
        for config_path in all_configs:
            result = run_single_config(str(config_path))
            results.append(result)
    else:
        with mp.Pool(processes=args.workers) as pool:
            config_paths = [str(p) for p in all_configs]
            results = list(pool.imap_unordered(run_single_config, config_paths, chunksize=1))

    elapsed = time.time() - start_time
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] != 'success')

    print(f"\n{successful}/{len(results)} passed ({elapsed/3600:.1f}h)")

    if failed > 0:
        print("\nFailed:")
        for r in results:
            if r['status'] != 'success':
                print(f"  {r['config']}: {r['status']}")
                if 'log' in r:
                    print(f"    {r['log']}")

    log_file = Path(f"{args.algorithm}_results.txt")
    with open(log_file, 'w') as f:
        f.write(f"{args.algorithm.upper()}\n")
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{elapsed/3600:.1f}h\n")
        f.write(f"{successful}/{len(results)} passed\n\n")
        for r in results:
            status_str = f"{r['status']:10s}"
            time_str = f"{r['time']/60:6.1f}m"
            log_str = f"  {r['log']}" if 'log' in r and r['status'] != 'success' else ""
            f.write(f"{status_str} {time_str}  {r['config']}{log_str}\n")

    print(f"Log: {log_file}")

if __name__ == "__main__":
    mp.freeze_support()
    main()
