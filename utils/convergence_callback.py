# utils/convergence_callback.py
import time
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class ConvergenceCallback(BaseCallback):
    """
    Logs convergence/stability metrics during training:
      - moving average ep reward
      - time-to-threshold (episodes, timesteps, walltime)
      - plateau detection (slope ~ 0 over a window)
      - post-convergence variance (stability)
    Works with SB3 logger (TensorBoard) and forwards to W&B via TB sync.
    """

    def __init__(
        self,
        reward_threshold=None,
        ma_window=100,
        plateau_window=200,
        plateau_eps_slope=1e-3,
        eval_env=None,
        eval_freq=10000,
        verbose=0,
    ):
        super().__init__(verbose)
        self.reward_threshold = reward_threshold
        self.ma_window = ma_window
        self.plateau_window = plateau_window
        self.plateau_eps_slope = plateau_eps_slope
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self._start_time = None

        self.ep_rewards = []
        self.ep_lengths = []
        self._last_eval = 0
        self._converged = False
        self._time_to_thresh = None
        self._plateaued = False
        self._plateau_at = None

    def _on_training_start(self) -> None:
        self._start_time = time.time()

    def _on_step(self) -> bool:
        # Read latest monitor info if available
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:  # VecEnv style
                self.ep_rewards.append(info["episode"]["r"])
                self.ep_lengths.append(info["episode"]["l"])

        t = self.num_timesteps

        # Moving average reward
        if len(self.ep_rewards) >= 1:
            ma = np.mean(self.ep_rewards[-self.ma_window:]) if len(self.ep_rewards) >= self.ma_window else np.mean(self.ep_rewards)
            self.logger.record("research/ep_reward_ma", float(ma))

        # Convergence threshold detection
        if (not self._converged) and self.reward_threshold is not None and len(self.ep_rewards) >= self.ma_window:
            ma = np.mean(self.ep_rewards[-self.ma_window:])
            if ma >= self.reward_threshold:
                self._converged = True
                wall = time.time() - self._start_time
                episodes = len(self.ep_rewards)
                self._time_to_thresh = dict(
                    timesteps=int(t),
                    episodes=int(episodes),
                    wall_seconds=float(wall),
                    reward_ma=float(ma),
                    threshold=float(self.reward_threshold),
                )
                for k, v in self._time_to_thresh.items():
                    self.logger.record(f"research/time_to_threshold/{k}", v)

        # Plateau detection: linear slope ~ 0 over last plateau_window eps
        if (not self._plateaued) and len(self.ep_rewards) >= self.plateau_window:
            y = np.array(self.ep_rewards[-self.plateau_window:])
            x = np.arange(len(y), dtype=float)
            # slope of least-squares line
            slope = np.polyfit(x, y, 1)[0]
            self.logger.record("research/plateau_slope", float(slope))
            if abs(slope) <= self.plateau_eps_slope:
                self._plateaued = True
                self._plateau_at = dict(
                    episodes=len(self.ep_rewards),
                    timesteps=int(t),
                    slope=float(slope),
                    reward_ma=float(np.mean(y)),
                )
                for k, v in self._plateau_at.items():
                    self.logger.record(f"research/plateau/{k}", v)

        # Stability (variance) over last ma_window after convergence
        if self._converged and len(self.ep_rewards) >= self.ma_window:
            tail = np.array(self.ep_rewards[-self.ma_window:])
            self.logger.record("research/stability/var_tail", float(np.var(tail)))
            self.logger.record("research/stability/std_tail", float(np.std(tail)))

        # Periodic deterministic eval
        if self.eval_env is not None and (t - self._last_eval) >= self.eval_freq:
            self._last_eval = t
            ep_ret = self._eval_once(self.eval_env, deterministic=True)
            self.logger.record("eval/return", float(ep_ret))

        return True

    def _eval_once(self, env, deterministic=True):
        obs, _ = env.reset()
        done = False
        total_r = 0.0
        while not done:
            action, _ = self.model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_r += reward
            done = terminated or truncated
        return total_r
