import argparse
from experiments import train  # import the train module

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on a specified environment.")
    parser.add_argument('--env', type=str, default="HalfCheetah-v5", help="Environment ID (Gymnasium env name).")
    parser.add_argument('--config_dir', type=str, default="configs", help="Directory containing config files.")
    parser.add_argument('--seed', type=int, help="Random seed for training.")
    parser.add_argument('--wandb', action='store_true', help="Enable W&B logging.")
    args = parser.parse_args()

    # Construct config file name based on chosen env
    env_key = args.env.split('-')[0].lower()  # e.g., "HalfCheetah-v5" -> "halfcheetah"
    config_file = f"ppo_{env_key}.yaml"
    config_path = f"{args.config_dir}/{config_file}"
    try:
        train.run_training(config_path=config_path, seed=args.seed, use_wandb=args.wandb)
    except Exception as e:
        print(f"Error during PPO training: {e}")
        raise
