import os
import sys
import yaml
import argparse

# FIX: Add project root to sys.path for Windows multiprocessing
# On Windows, subprocess workers don't inherit parent's sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3, A2C
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold
from wandb.integration.sb3 import WandbCallback
import wandb

# Import utility functions and callbacks
from utils.make_env import make_env
from utils.callbacks import EarlyStopCallback  # custom early stopping (optional)
from utils.convergence_callback import ConvergenceCallback

def _to_float(x):
    if isinstance(x, str):
        try:
            return float(x)
        except ValueError:
            return x
    return x

def _to_int(x):
    if isinstance(x, str):
        try:
            return int(x)
        except ValueError:
            return x
    return x

def coerce_algo_params(d: dict) -> dict:
    d = dict(d)  # shallow copy
    # Floats SB3 expects
    for k in ["learning_rate", "clip_range", "gamma", "gae_lambda",
              "ent_coef", "vf_coef", "max_grad_norm", "tau",
              "target_policy_noise", "exploration_noise"]:
        if k in d:
            d[k] = _to_float(d[k])
    # Ints SB3 expects
    for k in ["n_steps", "batch_size", "n_epochs", "policy_delay",
              "target_update_interval", "train_freq", "gradient_steps"]:
        if k in d:
            d[k] = _to_int(d[k])
    return d

def run_training(config_path: str, seed: int = None, use_wandb: bool = False, wandb_project: str = "rl-convergence"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if seed is not None:
        config['seed'] = seed
    run_seed = config.get('seed', 0)

    env_id = config['env_id']

    # Determine algorithm first to set n_envs appropriately
    algo_name = config.get('algorithm', '').lower()
    if algo_name == 'ppo':
        AlgoClass = PPO
    elif algo_name == 'sac':
        AlgoClass = SAC
    elif algo_name == 'td3':
        AlgoClass = TD3
    elif algo_name == 'a2c':
        AlgoClass = A2C
    else:
        raise ValueError(f"Unsupported algorithm specified: {algo_name}")

    # Use parallel environments for off-policy algorithms (TD3, SAC) to speed up data collection
    # This doesn't change learning dynamics but significantly speeds up training
    # On-policy algorithms (PPO, A2C) use n_steps for rollout collection, so keep n_envs=1
    n_envs = 4 if algo_name in ['td3', 'sac'] else 1

    # Get hyperparameters for unique directory naming
    raw_algo_params = config.get('algo_params', {})
    lr = raw_algo_params.get('learning_rate', 3e-4)
    bs = raw_algo_params.get('batch_size', 256)

    # Create unique monitor directory name including hyperparameters
    lr_str = f"{lr:.0e}".replace('-0', 'm0').replace('e-', 'em')
    monitor_dir = os.path.join("logs", "monitor", f"{algo_name}_{env_id}_{lr_str}_bs{bs}_s{run_seed}")
    env = make_env(env_id, seed=run_seed, monitor_dir=monitor_dir, n_envs=n_envs)

    eval_env = None
    if config.get('eval_freq', 0) > 0 or config.get('reward_threshold', None) is not None:
        eval_env = gym.make(env_id)
        # Gymnasium-style seeding:
        try:
            eval_env.reset(seed=run_seed)
        except TypeError:
            # fallback if older API
            pass

    # Coerce algo_params (already read raw_algo_params above for directory naming)
    algo_params = coerce_algo_params(raw_algo_params)

    # FIX: A2C doesn't accept batch_size parameter in Stable-Baselines3
    if algo_name == 'a2c' and 'batch_size' in algo_params:
        algo_params.pop('batch_size')
        print(f"DEBUG: Removed batch_size for A2C (not supported)")

    # device handling: default to 'auto' if not provided
    device = algo_params.pop("device", "auto")

    policy = config.get('policy', 'MlpPolicy')
    verbose = int(config.get('verbose', 1))

    # Use unique directory names including hyperparameters for tensorboard too
    tb_log_dir = os.path.join("logs", "tensorboard", f"{algo_name}_{env_id}_{lr_str}_bs{bs}_s{run_seed}")
    os.makedirs(tb_log_dir, exist_ok=True)

    if use_wandb:
        # mode="offline" prevents blocking on hotspot - all data saved locally, sync later with "wandb sync"
        # This was necessary because PPO ran on fast internet, but now on hotspot wandb.init() blocks
        # Use unique run name including hyperparameters
        wandb.init(project=wandb_project, config=config, mode="offline",
                   name=f"{algo_name}_{env_id}_{lr_str}_bs{bs}_s{run_seed}",
                   group=f"{algo_name}_{env_id}")

    # --- helpful one-time debug prints ---
    print("DEBUG device request:", device)
    print("DEBUG algo_params types:", {k: type(v).__name__ for k, v in algo_params.items()})
    # -------------------------------------

    try:
        model = AlgoClass(
            policy,
            env,
            verbose=verbose,
            tensorboard_log=tb_log_dir,
            device=device,              # <<< ensure GPU when available
            **algo_params
        )

        callback_list = []
        ckpt_freq = int(config.get('checkpoint_freq', 0) or 0)
        if ckpt_freq > 0:
            # Use absolute path for checkpoints to avoid subprocess issues
            # Include hyperparameters in checkpoint path for uniqueness
            ckpt_path = os.path.abspath(os.path.join("models", f"{algo_name}_{env_id}_{lr_str}_bs{bs}_s{run_seed}"))
            os.makedirs(ckpt_path, exist_ok=True)
            # guard num_envs
            num_envs = getattr(env, "num_envs", 1)
            checkpoint_cb = CheckpointCallback(
                save_freq=max(1, ckpt_freq // max(1, num_envs)),
                save_path=ckpt_path,
                name_prefix=f"{algo_name}_{env_id}"
            )
            callback_list.append(checkpoint_cb)

        if config.get('reward_threshold', None) is not None:
            threshold = float(config['reward_threshold'])
            if eval_env is None:
                eval_env = gym.make(env_id)
                try:
                    eval_env.reset(seed=run_seed)
                except TypeError:
                    pass
            callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=threshold, verbose=1)
            eval_cb = EvalCallback(
                eval_env, callback_on_new_best=callback_on_best,
                eval_freq=max(10000, int(config.get('eval_freq', 10000) or 10000)),
                best_model_save_path=None, verbose=0
            )
            callback_list.append(eval_cb)

        if use_wandb:
            wandb_cb = WandbCallback(
                model_save_path=None,  # FIX: Disabled - Windows symlink requires admin. We save models ourselves at line 184
                model_save_freq=0,
                gradient_save_freq=0,  # FIX: Disabled to prevent blocking I/O (was 100)
                verbose=2
            )
            callback_list.append(wandb_cb)

        # ALWAYS attach the convergence/plateau callback (independent of use_wandb)
        conv_cb = ConvergenceCallback(
            reward_threshold=config.get('reward_threshold', None),
            ma_window=100,
            plateau_window=200,
            plateau_eps_slope=1e-3,
            eval_env=eval_env,
            eval_freq=max(5000, int(config.get('eval_freq', 10000) or 10000)),
            verbose=0,
        )
        callback_list.append(conv_cb)

        callbacks = CallbackList(callback_list) if callback_list else None

        model.learn(total_timesteps=int(config['total_timesteps']), callback=callbacks)

        # Use absolute path for models to avoid issues with subprocess working directory
        models_dir = os.path.abspath("models")
        os.makedirs(models_dir, exist_ok=True)
        # Include hyperparameters in final model name for uniqueness
        model_path = os.path.join(models_dir, f"{algo_name}_{env_id}_{lr_str}_bs{bs}_s{run_seed}_final")
        model.save(model_path)
        if verbose:
            print(f"Training complete. Model saved to {model_path}.")
        # confirm actual device of the policy params
        try:
            import torch
            print("DEBUG model param device:", next(model.policy.parameters()).device)
        except Exception:
            pass
    finally:
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
