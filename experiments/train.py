import os
import yaml
import argparse
import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold
from wandb.integration.sb3 import WandbCallback
import wandb

# Import utility functions and callbacks
from utils.make_env import make_env
from utils.callbacks import EarlyStopCallback  # custom early stopping (optional)

def run_training(config_path: str, seed: int = None, use_wandb: bool = False, wandb_project: str = "rl-convergence"):
    """
    Run a training experiment given a configuration file.
    Args:
        config_path (str): Path to the YAML configuration file.
        seed (int, optional): Random seed to override the config file's seed.
        use_wandb (bool): Whether to use Weights & Biases logging.
        wandb_project (str): W&B project name (if use_wandb is True).
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    # Override seed if provided
    if seed is not None:
        config['seed'] = seed
    # Set random seed for reproducibility (Gym uses this in env reset, SB3 uses internally too)
    run_seed = config.get('seed', 0)

    # Create environment (vectorized with Monitor for logging)
    env_id = config['env_id']
    monitor_dir = os.path.join("logs", "monitor", f"{config['algorithm']}_{env_id}_seed{run_seed}")
    env = make_env(env_id, seed=run_seed, monitor_dir=monitor_dir)

    # Optional: create a separate evaluation environment for EvalCallback
    eval_env = None
    if config.get('eval_freq', 0) > 0 or config.get('reward_threshold', None):
        eval_env = gym.make(env_id)  # unmonitored eval env (similar wrapping as training env if needed)
        eval_env.seed(run_seed)      # ensure deterministic eval if needed

    # Select algorithm class based on config
    algo_name = config.get('algorithm', '').lower()
    if algo_name == 'ppo':
        AlgoClass = PPO
    elif algo_name == 'sac':
        AlgoClass = SAC
    else:
        raise ValueError(f"Unsupported algorithm specified: {algo_name}")

    # Prepare model hyperparameters
    algo_params = config.get('algo_params', {})
    policy = config.get('policy', 'MlpPolicy')
    verbose = config.get('verbose', 1)

    # Set up TensorBoard logging directory
    tb_log_dir = os.path.join("logs", "tensorboard", f"{algo_name}_{env_id}_seed{run_seed}")
    os.makedirs(tb_log_dir, exist_ok=True)

    # Initialize W&B if requested
    if use_wandb:
        wandb.init(project=wandb_project, config=config, sync_tensorboard=True,
                   name=f"{algo_name}_{env_id}_seed{run_seed}", monitor_gym=True, save_code=True)
        # Note: sync_tensorboard will sync TensorBoard logs to W&B
        # monitor_gym will auto-upload monitor logs and optionally video (if enabled via wrappers)
        # We'll use WandbCallback to save model and gradients if desired
    try:
        # Instantiate the RL model
        model = AlgoClass(policy, env, verbose=verbose, tensorboard_log=tb_log_dir, **algo_params)

        # Configure callbacks
        callback_list = []
        # Checkpoint callback (save model every 'checkpoint_freq' steps if specified)
        ckpt_freq = config.get('checkpoint_freq', 0)
        if ckpt_freq and ckpt_freq > 0:
            ckpt_path = os.path.join("models", f"{algo_name}_{env_id}_seed{run_seed}")
            os.makedirs(ckpt_path, exist_ok=True)
            checkpoint_cb = CheckpointCallback(save_freq=max(1, ckpt_freq // env.num_envs), save_path=ckpt_path,
                                               name_prefix=f"{algo_name}_{env_id}")
            callback_list.append(checkpoint_cb)
        # Early stopping on reward threshold using EvalCallback, if specified
        if config.get('reward_threshold', None) is not None:
            # Use EvalCallback to evaluate and trigger StopTrainingOnRewardThreshold
            threshold = config['reward_threshold']
            # Ensure eval_env exists
            if eval_env is None:
                eval_env = gym.make(env_id)
                eval_env.seed(run_seed)
            callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=threshold, verbose=1)
            eval_cb = EvalCallback(eval_env, callback_on_new_best=callback_on_best,
                                   eval_freq=max(10000, config.get('eval_freq', 10000)),  # evaluate at least every 10000 steps
                                   best_model_save_path=None,  # or specify path to save best model
                                   verbose=0)
            callback_list.append(eval_cb)
        # Custom early stopping callback (if using a different criterion)
        # For example, stop if no improvement over N evals (could use StopTrainingOnNoModelImprovement via EvalCallback)
        # or use our EarlyStopCallback for training reward (commented out here):
        # early_stop_cb = EarlyStopCallback(threshold=config['reward_threshold'], patience_episodes=5, verbose=1)
        # callback_list.append(early_stop_cb)

        # WandbCallback for logging (if W&B is enabled)
        if use_wandb:
            wandb_cb = WandbCallback(model_save_path=os.path.join("models", f"{algo_name}_{env_id}_seed{run_seed}"),
                                      model_save_freq=0,  # save model manually or via checkpoint, so set 0
                                      gradient_save_freq=100,  # log gradients every 100 training updates
                                      verbose=1)
            callback_list.append(wandb_cb)

        # Combine callbacks
        callbacks = CallbackList(callback_list) if callback_list else None

        # Start training
        model.learn(total_timesteps=int(config['total_timesteps']), callback=callbacks)

        # Save final model
        os.makedirs("models", exist_ok=True)
        model_path = os.path.join("models", f"{algo_name}_{env_id}_seed{run_seed}_final")
        model.save(model_path)
        if verbose:
            print(f"Training complete. Model saved to {model_path}.")
    finally:
        # Cleanup: ensure environments are closed and W&B finished
        env.close()
        if eval_env is not None:
            eval_env.close()
        if use_wandb:
            wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an RL agent given a config file.")
    parser.add_argument('--config', type=str, required=True, help="Path to the config YAML file.")
    parser.add_argument('--seed', type=int, help="Override random seed from config.")
    parser.add_argument('--wandb', action='store_true', help="Enable Weights & Biases logging.")
    parser.add_argument('--project', type=str, default="rl-convergence", help="W&B project name (if logging enabled).")
    args = parser.parse_args()

    run_training(config_path=args.config, seed=args.seed, use_wandb=args.wandb, wandb_project=args.project)
