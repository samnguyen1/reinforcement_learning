from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class EarlyStopCallback(BaseCallback):
    """
    A custom callback that stops training early if the mean reward over the last N episodes exceeds a threshold.
    This uses the training episode rewards (from Monitor info) to decide when to stop.
    """
    def __init__(self, threshold: float, patience_episodes: int = 5, verbose: int = 0):
        super().__init__(verbose)
        self.threshold = threshold
        self.patience = patience_episodes
        self.recent_rewards = []

    def _on_step(self) -> bool:
        # Check if a new episode finished by examining 'dones' and env info
        if 'dones' in self.locals and 'infos' in self.locals:
            dones = self.locals['dones']
            infos = self.locals['infos']
            # If using a VecEnv, there may be multiple entries in dones.
            for i, done in enumerate(dones):
                if done:
                    info = infos[i]
                    # The Monitor wrapper adds 'episode' info when an episode ends
                    if 'episode' in info:
                        episode_reward = info['episode']['r']
                        # Record the episode reward
                        self.recent_rewards.append(episode_reward)
                        if len(self.recent_rewards) > self.patience:
                            # Only keep latest 'patience' number of episodes
                            self.recent_rewards.pop(0)
                        # Check mean of recent episodes
                        mean_recent = np.mean(self.recent_rewards)
                        if self.verbose:
                            print(f"[EarlyStopCallback] Last {len(self.recent_rewards)} episodes mean reward: {mean_recent:.2f}")
                        # If we have enough episodes and threshold condition met, stop training
                        if len(self.recent_rewards) == self.patience and mean_recent >= self.threshold:
                            if self.verbose:
                                print(f"[EarlyStopCallback] Stopping training early: mean reward {mean_recent:.2f} ≥ threshold {self.threshold}")
                            return False  # Returning False instructs SB3 to stop training
        # Continue training
        return True
