# src/aggregate_runs.py
from __future__ import annotations
import argparse, os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common import moving_average

def collect_runs(root: str):
    # Finds .../{algo}/seed_k/episodes.csv
    runs = {}
    for algo_dir in glob.glob(os.path.join(root, "*")):
        algo = os.path.basename(algo_dir)
        seeds = []
        for seed_dir in glob.glob(os.path.join(algo_dir, "seed_*")):
            csv_path = os.path.join(seed_dir, "episodes.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                seeds.append(df["r"].to_numpy())
        if seeds:
            runs[algo] = seeds
    return runs  # dict[str, list[np.ndarray]]

def pad_to_same_length(arrs):
    m = min(len(a) for a in arrs)
    return np.stack([a[:m] for a in arrs], axis=0)

def plot_mean_std(env_id: str, algo: str, rewards_2d: np.ndarray, out_dir: str, ma_window: int):
    # rewards_2d: shape [n_seeds, episodes]
    mean = rewards_2d.mean(axis=0)
    std  = rewards_2d.std(axis=0)

    x = np.arange(1, len(mean)+1)

    plt.figure()
    plt.plot(x, mean, label=f"{algo.upper()} mean")
    plt.fill_between(x, mean-std, mean+std, alpha=0.25, label="±1 std")
    ma = moving_average(mean, ma_window)
    if len(ma) > 0:
        plt.plot(np.arange(ma_window, ma_window + len(ma)), ma, label=f"MA({ma_window})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"{env_id} — {algo.upper()} (mean±std over seeds)")
    plt.legend()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{env_id.lower()}_{algo}_mean_std.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    return out_path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="experiments/cartpole")
    p.add_argument("--env_id", default="CartPole-v1")
    p.add_argument("--ma_window", type=int, default=20)
    p.add_argument("--fig_dir", default="figs/cartpole")
    args = p.parse_args()

    runs = collect_runs(args.root)
    summary_rows = []
    out_figs = []

    for algo, arrs in runs.items():
        rewards_2d = pad_to_same_length(arrs)
        out_path = plot_mean_std(args.env_id, algo, rewards_2d, args.fig_dir, args.ma_window)
        out_figs.append(out_path)

        # Example metrics for Week 5 notes:
        # - sample efficiency: episodes to reach threshold (e.g., >=195)
        threshold = 195
        mean_curve = rewards_2d.mean(axis=0)
        reached = np.argmax(mean_curve >= threshold) if (mean_curve >= threshold).any() else -1

        summary_rows.append({
            "algo": algo,
            "n_seeds": rewards_2d.shape[0],
            "episodes_used": rewards_2d.shape[1],
            "episodes_to_threshold_195": int(reached)
        })

    df_summary = pd.DataFrame(summary_rows)
    os.makedirs(args.fig_dir, exist_ok=True)
    df_summary.to_csv(os.path.join(args.fig_dir, "summary.csv"), index=False)

    print("Figures:")
    for f in out_figs:
        print(" -", f)
    print("Summary:", os.path.join(args.fig_dir, "summary.csv"))

if __name__ == "__main__":
    main()
