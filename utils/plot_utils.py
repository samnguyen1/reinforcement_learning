import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_results(log_dir: str) -> pd.DataFrame:
    """
    Load all monitor CSV results from a given directory (recursively).
    Returns a DataFrame with columns for episode reward, length, etc., including the source file.
    """
    pattern = os.path.join(log_dir, "**", "*monitor.csv")
    files = glob.glob(pattern, recursive=True)
    data_frames = []
    for file in files:
        # Each monitor.csv has a header comment line starting with '#', so use comment to skip it
        df = pd.read_csv(file, comment='#', header=0)
        df['file'] = file  # keep track of which run this data came from
        data_frames.append(df)
    if not data_frames:
        print(f"No monitor files found in {log_dir}")
        return pd.DataFrame()  # empty
    # Concatenate all data
    result_df = pd.concat(data_frames, ignore_index=True)
    return result_df

def plot_learning_curve(algorithm_name: str, env_id: str, log_dir: str = "logs/monitor", num_seeds: int = None):
    """
    Plot the learning curve (episode reward vs episode count) for a given algorithm and environment.
    If multiple seeds are present, plots the mean and confidence interval (±1 std) across seeds.
    """
    # Load data
    df = load_results(log_dir)
    if df.empty:
        print("No data to plot.")
        return
    # Filter for the specific algorithm and environment
    # The file path contains e.g. ".../ppo_HalfCheetah-v5_seed0/monitor.csv"
    query_str = f"{algorithm_name}_{env_id}"
    df = df[df['file'].str.contains(query_str)]
    if df.empty:
        print(f"No data for {algorithm_name} on {env_id} in {log_dir}")
        return

    # If num_seeds is not given, infer from data (count unique seed identifiers)
    if num_seeds is None:
        # Extract seed from file path by parsing "...seedX/"
        df['seed'] = df['file'].apply(lambda x: int(x.split('_seed')[-1].split('/')[0]))
        num_seeds = df['seed'].nunique()

    # Sort by episode order within each seed and truncate to equal length for averaging
    rewards_by_seed = []
    for seed in sorted(df['file'].unique()):
        seed_df = df[df['file'] == seed].copy()
        seed_df = seed_df.sort_index()  # ensure in chronological order
        rewards_by_seed.append(seed_df['r'].values)
    # Truncate all runs to the minimum number of episodes (so all have equal length)
    min_episodes = min(len(r) for r in rewards_by_seed)
    rewards_by_seed = [r[:min_episodes] for r in rewards_by_seed if len(r) >= min_episodes]
    rewards_by_seed = np.array(rewards_by_seed)
    # Compute mean and std across seeds
    mean_rewards = rewards_by_seed.mean(axis=0)
    std_rewards = rewards_by_seed.std(axis=0)

    # Plotting
    episodes = np.arange(1, min_episodes + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(episodes, mean_rewards, label=f"{algorithm_name.upper()} on {env_id}")
    plt.fill_between(episodes, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3, label="±1 std dev")
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title(f"Learning Curve: {algorithm_name.upper()} on {env_id}")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_all_algorithms(env_id: str, log_dir: str = "logs/monitor"):
    """
    Plot learning curves for all algorithms present in the monitor logs for a given environment, for comparison.
    Assumes logs are named as <algo>_<env>_seed.
    """
    df = load_results(log_dir)
    if df.empty:
        return
    algos = set()
    # infer algos from file names
    for f in df['file'].unique():
        if env_id in f:
            alg_name = os.path.basename(os.path.dirname(f)).split('_')[0]  # algo part
            algos.add(alg_name)
    plt.figure(figsize=(8,6))
    for algo in sorted(algos):
        sub_df = df[df['file'].str.contains(f"{algo}_{env_id}")]
        if sub_df.empty:
            continue
        # Sort and average as above
        rewards_by_seed = []
        for seed_file in sub_df['file'].unique():
            seed_df = sub_df[sub_df['file'] == seed_file].copy()
            seed_df = seed_df.sort_index()
            rewards_by_seed.append(seed_df['r'].values)
        min_episodes = min(len(r) for r in rewards_by_seed)
        rewards_by_seed = [r[:min_episodes] for r in rewards_by_seed]
        rewards_by_seed = np.array(rewards_by_seed)
        mean_rewards = rewards_by_seed.mean(axis=0)
        episodes = np.arange(1, min_episodes + 1)
        plt.plot(episodes, mean_rewards, label=algo.upper())
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title(f"Learning Curves on {env_id}")
    plt.legend()
    plt.grid(True)
    plt.show()
