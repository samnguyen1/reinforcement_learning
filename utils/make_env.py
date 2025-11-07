import os
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

def make_env(env_id: str, seed: int = 0, monitor_dir: str = None, n_envs: int = 1):
    """
    Create a Gymnasium environment wrapped with Monitor and VecEnv.
    This ensures episodes are logged and we have a vectorized environment for SB3.

    Args:
        env_id (str): Gym environment ID (e.g., "HalfCheetah-v5").
        seed (int): Random seed for environment.
        monitor_dir (str, optional): Directory to save Monitor logs (if None, logs to memory only).
        n_envs (int): Number of parallel environments (default=1 for compatibility, use 4 for speedup).

    Returns:
        VecEnv: A vectorized environment with n_envs sub-environments.
    """
    # Inner function to create the environment (for VecEnv)
    def _init(rank):
        def _thunk():
            env = gym.make(env_id)
            # Set seed for reproducibility (each env gets unique seed)
            env.reset(seed=seed + rank)
            # Wrap with Monitor to record episode stats
            # For multiple environments, only the first one writes to monitor.csv (rank 0)
            # This avoids race conditions when multiple processes write to the same file
            if monitor_dir and rank == 0:
                os.makedirs(monitor_dir, exist_ok=True)
                monitor_file = os.path.join(monitor_dir, "monitor.csv")
                env = Monitor(env, filename=monitor_file)
            else:
                env = Monitor(env)  # still wrap for stats, but won't record to disk
            return env
        return _thunk

    # Use SubprocVecEnv for true parallelism when n_envs > 1
    if n_envs > 1:
        return SubprocVecEnv([_init(i) for i in range(n_envs)])
    else:
        # Single environment - use DummyVecEnv for simplicity
        return DummyVecEnv([_init(0)])
