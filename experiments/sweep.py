import argparse
import glob
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hyperparameter sweep or multiple experiments.")
    parser.add_argument('--configs', nargs='+', help="List of config files or glob pattern for configs.")
    parser.add_argument('--seeds', type=int, default=5, help="Number of seeds to run for each config.")
    parser.add_argument('--parallel', action='store_true', help="Run experiments in parallel (if not set, runs sequentially).")
    args = parser.parse_args()

    # Gather config files
    config_paths = []
    if args.configs:
        for pattern in args.configs:
            # Support glob patterns or explicit file paths
            config_paths.extend(glob.glob(pattern))
    else:
        # If no configs specified, use all configs in configs/ directory
        config_paths = glob.glob("configs/*.yaml")

    if not config_paths:
        print("No configuration files found for the given pattern.")
        exit(1)

    # Run experiments for each config and seed
    processes = []
    for config in config_paths:
        for seed in range(args.seeds):
            cmd = ["python", "experiments/train.py", "--config", config, "--seed", str(seed)]
            if args.parallel:
                # Launch subprocess and append to list
                processes.append(subprocess.Popen(cmd))
            else:
                # Run sequentially and wait
                result = subprocess.run(cmd)
                if result.returncode != 0:
                    print(f"Experiment failed for {config} seed {seed}. Stopping.")
                    exit(1)
    # If parallel, wait for all to finish
    if args.parallel:
        for p in processes:
            p.wait()
