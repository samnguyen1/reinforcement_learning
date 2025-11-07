import os
import sys
import yaml
import argparse

# Add project root to sys.path for Windows multiprocessing compatibility
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3, A2C
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold
from wandb.integration.sb3 import WandbCallback
import wandb

from utils.make_env import make_env
from utils.callbacks import EarlyStopCallback
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
    """Convert string hyperparameters to appropriate numeric types."""
    d = dict(d)
    float_params = ["learning_rate", "clip_range", "gamma", "gae_lambda",
                    "ent_coef", "vf_coef", "max_grad_norm", "tau",
                    "target_policy_noise", "exploration_noise"]
    int_params = ["n_steps", "batch_size", "n_epochs", "policy_delay",
                  "target_update_interval", "train_freq", "gradient_steps"]

    for k in float_params:
        if k in d:
            d[k] = _to_float(d[k])
    for k in int_params:
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

    # Parallel environments for off-policy algorithms to accelerate data collection
    n_envs = 4 if algo_name in ['td3', 'sac'] else 1
    raw_algo_params = config.get('algo_params', {})
    lr = raw_algo_params.get('learning_rate', 3e-4)

    # A2C uses n_steps, other algorithms use batch_size
    if algo_name == 'a2c':
        batch_param = raw_algo_params.get('n_steps', 16)
        batch_label = f"ns{batch_param}"
    else:
        batch_param = raw_algo_params.get('batch_size', 256)
        batch_label = f"bs{batch_param}"
    lr_str = f"{lr:.0e}".replace('-0', 'm0').replace('e-', 'em')
    monitor_dir = os.path.join("logs", "monitor", f"{algo_name}_{env_id}_{lr_str}_{batch_label}_s{run_seed}")
    env = make_env(env_id, seed=run_seed, monitor_dir=monitor_dir, n_envs=n_envs)

    eval_env = None
    if config.get('eval_freq', 0) > 0 or config.get('reward_threshold', None) is not None:
        eval_env = gym.make(env_id)
        try:
            eval_env.reset(seed=run_seed)
        except TypeError:
            pass
    algo_params = coerce_algo_params(raw_algo_params)

    # A2C doesn't accept batch_size parameter
    if algo_name == 'a2c' and 'batch_size' in algo_params:
        algo_params.pop('batch_size')

    device = algo_params.pop("device", "auto")

    policy = config.get('policy', 'MlpPolicy')
    verbose = int(config.get('verbose', 1))

    tb_log_dir = os.path.join("logs", "tensorboard", f"{algo_name}_{env_id}_{lr_str}_{batch_label}_s{run_seed}")
    os.makedirs(tb_log_dir, exist_ok=True)

    if use_wandb:
        wandb.init(project=wandb_project, config=config, mode="offline",
                   name=f"{algo_name}_{env_id}_{lr_str}_{batch_label}_s{run_seed}",
                   group=f"{algo_name}_{env_id}")

    try:
        model = AlgoClass(
            policy,
            env,
            verbose=verbose,
            tensorboard_log=tb_log_dir,
            device=device,
            **algo_params
        )

        callback_list = []
        ckpt_freq = int(config.get('checkpoint_freq', 0) or 0)
        if ckpt_freq > 0:
            ckpt_path = os.path.abspath(os.path.join("models", f"{algo_name}_{env_id}_{lr_str}_{batch_label}_s{run_seed}"))
            os.makedirs(ckpt_path, exist_ok=True)
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
                model_save_path=None,
                model_save_freq=0,
                gradient_save_freq=0,
                verbose=2
            )
            callback_list.append(wandb_cb)
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

        models_dir = os.path.abspath("models")
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, f"{algo_name}_{env_id}_{lr_str}_{batch_label}_s{run_seed}_final")
        model.save(model_path)
        if verbose:
            print(f"Training complete. Model saved to {model_path}.")
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
