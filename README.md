# Reinforcement Learning Algorithm Comparison

This repository contains the experimental codebase for comparing four reinforcement learning algorithms (PPO, TD3, A2C, and SAC) on continuous control tasks using MuJoCo environments.

## Installation

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
wandb login
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
wandb login
```


## Repository Structure

```
.
├── experiments/
│   └── train.py              # Main training script
├── configs/                  # Experimental configurations (240 total)
├── utils/                    # Environment creation and callbacks
└── run_single_algorithm.py   # Primary experiment runner
```

## Usage

### Running Experiments

Execute experiments for each algorithm separately:

```bash
# PPO (60 experiments)
python run_single_algorithm.py ppo --workers 6

# TD3 (60 experiments)
python run_single_algorithm.py td3 --workers 6

# A2C (60 experiments)
python run_single_algorithm.py a2c --workers 6

# SAC (60 experiments)
python run_single_algorithm.py sac --workers 6
```

The `--workers` parameter controls parallelisation. Recommended value is `num_cores - 2` for computational efficiency.

## Experiment Tracking

All experiments are logged to Weights & Biases:
- Project: `rl1`
- URL: https://wandb.ai/samnguyen1/rl1

## Environments

**HalfCheetah-v5**: Two-dimensional robot locomotion task with 6 DoF. Observation space is 17-dimensional, action space is 6-dimensional.

**Hopper-v5**: Two-dimensional single-legged robot balancing and locomotion task with 3 DoF. Observation space is 11-dimensional, action space is 3-dimensional.
