# src/run_baseline.py
from __future__ import annotations
import argparse, os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.logger import configure
from common import set_seed, make_env, load_monitor_csv, moving_average, save_json

ALGOS = {"dqn": DQN, "ppo": PPO}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=ALGOS.keys(), required=True)
    p.add_argument("--env_id", default="CartPole-v1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=200)    # short run (Week 4)
    p.add_argument("--long", action="store_true", help="~1000 episodes (Week 5)")
    p.add_argument("--ma_window", type=int, default=20, help="moving average window (episodes)")
    p.add_argument("--out_root", default="experiments/cartpole")
    args = p.parse_args()

    set_seed(args.seed)
    algo = args.algo.lower()
    AlgoCls = ALGOS[algo]
    # Rough timesteps per episode for CartPole is ~200; scale accordingly
    episodes = 1000 if args.long else args.episodes
    total_timesteps = int(episodes * 200)

    run_dir = os.path.join(args.out_root, algo, f"seed_{args.seed}")
    os.makedirs(run_dir, exist_ok=True)

    # Env with monitor
    env = make_env(args.env_id, log_dir=run_dir, seed=args.seed)

    # SB3 logger (kept in run_dir/sb3/)
    new_logger = configure(os.path.join(run_dir, "sb3"), ["csv", "stdout"])

    # Minimal defaults (you can tune later)
    if algo == "dqn":
        model = AlgoCls("MlpPolicy", env, verbose=0, seed=args.seed,
                        learning_rate=1e-3, buffer_size=50_000, learning_starts=1_000,
                        batch_size=64, target_update_interval=250, train_freq=(4, "step"),
                        exploration_fraction=0.1, exploration_final_eps=0.05, gamma=0.99)
    else:
        model = AlgoCls("MlpPolicy", env, verbose=0, seed=args.seed,
                        n_steps=2048, batch_size=64, gae_lambda=0.95, gamma=0.99, clip_range=0.2,
                        learning_rate=3e-4, ent_coef=0.0)

    model.set_logger(new_logger)
    model.learn(total_timesteps=total_timesteps, callback=CallbackList([]))

    # Parse per-episode monitor logs
    monitor_path = os.path.join(run_dir, "monitor.csv")
    df = load_monitor_csv(monitor_path)
    df.to_csv(os.path.join(run_dir, "episodes.csv"), index=False)

    # Plot reward vs episode (+ moving average)
    rewards = df["r"].to_numpy()
    ma = moving_average(rewards, window=args.ma_window)

    plt.figure()
    plt.plot(np.arange(1, len(rewards)+1), rewards, label="Episode reward")
    if len(ma) > 0:
        plt.plot(np.arange(args.ma_window, args.ma_window + len(ma)), ma, label=f"MA({args.ma_window})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"{args.env_id} — {algo.upper()} (seed={args.seed})")
    plt.legend()
    fig_path = os.path.join(run_dir, "learning_curve.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")

    # Metadata for later aggregation
    save_json({
        "algo": algo,
        "env_id": args.env_id,
        "seed": args.seed,
        "episodes": int(df.shape[0]),
        "timesteps": total_timesteps,
        "ma_window": args.ma_window,
        "fig_path": fig_path
    }, os.path.join(run_dir, "meta.json"))

    print(f"Saved: {fig_path} and episodes.csv in {run_dir}")

if __name__ == "__main__":
    main()
