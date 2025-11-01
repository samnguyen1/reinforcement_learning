# src/common.py
from __future__ import annotations
import os, json, math, random
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
from stable_baselines3.common.monitor import Monitor

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def make_env(env_id: str, log_dir: str, seed: int):
    os.makedirs(log_dir, exist_ok=True)
    env = gym.make(env_id)
    env.reset(seed=seed)
    # Monitor writes a CSV (monitor.csv) with episode rewards/lengths
    env = Monitor(env, filename=os.path.join(log_dir, "monitor.csv"))
    # RecordEpisodeStatistics keeps per-episode stats in info (optional)
    env = RecordEpisodeStatistics(env)
    return env

def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    w = np.ones(window) / window
    return np.convolve(x, w, mode="valid")

def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_monitor_csv(path: str) -> pd.DataFrame:
    # Skip header rows starting with '#'
    rows = []
    with open(path, "r") as f:
        for line in f:
            if not line.startswith("#"):
                rows.append(line)
    from io import StringIO
    df = pd.read_csv(StringIO("".join(rows)))
    # Columns typically: r (reward), l (length), t (time)
    return df
