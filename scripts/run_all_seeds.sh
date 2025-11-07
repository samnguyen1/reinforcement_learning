#!/usr/bin/env bash
# Usage: run_all_seeds.sh <config_path> [<num_seeds>]
# Runs the given experiment configuration for multiple seeds sequentially.

CONFIG_PATH="$1"
NUM_SEEDS="${2:-5}"

if [ -z "$CONFIG_PATH" ]; then
  echo "Usage: $0 <config_path> [num_seeds]"
  exit 1
fi

echo "Running experiment $CONFIG_PATH for $NUM_SEEDS seeds..."
for (( SEED=0; SEED<NUM_SEEDS; SEED++ ))
do
  echo "Launching seed $SEED..."
  python experiments/train.py --config "$CONFIG_PATH" --seed $SEED --wandb --project rl
  if [ $? -ne 0 ]; then
    echo "Run failed for seed $SEED, aborting remaining runs."
    exit 1
  fi
done

echo "All $NUM_SEEDS runs completed for config $CONFIG_PATH."
