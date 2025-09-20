import os
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

def make_env(env_id: str, seed: int = 0, monitor_dir: str = None):
    """
    Create a Gymnasium environment wrapped with Monitor and DummyVecEnv.
    This ensures episodes are logged and we have a vectorized environment for SB3.
    Args:
        env_id (str): Gym environment ID (e.g., "HalfCheetah-v4").
        seed (int): Random seed for environment.
        monitor_dir (str, optional): Directory to save Monitor logs (if None, logs to memory only).
    Returns:
        DummyVecEnv: A vectorized environment with a single sub-environment.
    """
    # Inner function to create the environment (for DummyVecEnv)
    def _init():
        env = gym.make(env_id)
        # Set seed for reproducibility
        env.reset(seed=seed)
        # Wrap with Monitor to record episode stats; save to file if monitor_dir provided
        if monitor_dir:
            os.makedirs(monitor_dir, exist_ok=True)
            monitor_file = os.path.join(monitor_dir, "monitor.csv")
            env = Monitor(env, filename=monitor_file)
        else:
            env = Monitor(env)  # still wrap, but won't record to disk
        return env

    # Return a DummyVecEnv with a single environment instance
    return DummyVecEnv([_init])
