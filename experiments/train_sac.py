import argparse
from experiments import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAC on a specified environment.")
    parser.add_argument('--env', type=str, default="HalfCheetah-v4", help="Environment ID to train on.")
    parser.add_argument('--config_dir', type=str, default="configs", help="Directory for config files.")
    parser.add_argument('--seed', type=int, help="Random seed for training.")
    parser.add_argument('--wandb', action='store_true', help="Enable W&B logging.")
    args = parser.parse_args()

    env_key = args.env.split('-')[0].lower()
    config_file = f"sac_{env_key}.yaml"
    config_path = f"{args.config_dir}/{config_file}"
    try:
        train.run_training(config_path=config_path, seed=args.seed, use_wandb=args.wandb)
    except Exception as e:
        print(f"Error during SAC training: {e}")
        raise
