# Reinforcement Learning Thesis Experiments

Comprehensive RL experiments comparing PPO, TD3, A2C, and SAC algorithms on MuJoCo environments for thesis research.

## Research Question

Under what conditions do PPO, TD3, and A2C converge reliably on benchmark tasks, and how do hyperparameters and architecture influence their stability?

## Current Progress

✅ **71 PPO experiments completed** (41 HalfCheetah-v5, 30 Hopper-v5)
- All trained to 1M timesteps
- Data available on [W&B Project](https://wandb.ai/samnguyen1/rl1)

## Setup

### Prerequisites

- Python 3.8+
- MuJoCo (for physics simulations)
- CUDA (optional, for GPU acceleration)

### Installation

#### On Windows:

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Login to Weights & Biases
wandb login
```

#### On Mac/Linux:

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Login to Weights & Biases
wandb login
```

### MuJoCo Setup

MuJoCo is now free and integrated with `mujoco-py`. Installation should be automatic with the requirements.txt, but if you encounter issues:

**Mac:**
```bash
brew install gcc
pip install mujoco-py
```

**Linux:**
```bash
sudo apt-get install libosmesa6-dev libgl1-mesa-glx libglfw3
pip install mujoco-py
```

## Project Structure

```
.
├── experiments/
│   ├── train.py              # Main training script
│   └── train_[algo].py       # Algorithm-specific trainers
├── configs/
│   └── *.yaml                # Base config files
├── thesis_sweep_configs/     # 240 experiment configs (60 per algorithm)
├── utils/
│   ├── make_env.py           # Environment creation
│   ├── plot_utils.py         # Plotting utilities
│   └── convergence_callback.py
├── run_single_algorithm.py   # ⭐ Main runner (cross-platform)
├── check_progress.py         # Monitor wandb progress
├── export_thesis_data.py     # Export data for analysis
└── README.md                 # This file
```

## Running Experiments

### Run All Experiments for One Algorithm

**Recommended approach** - Run one algorithm at a time:

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

**Workers:**
- Use `--workers 6` on most machines (balances parallelization and CPU contention)
- Adjust based on your CPU cores (recommended: num_cores - 2)
- Each experiment takes ~100 minutes with 1M timesteps

### Check Progress

```bash
python check_progress.py
```

Shows:
- Number of running/finished/failed experiments
- Breakdown by algorithm
- Currently running experiments

### Export Data for Thesis

```bash
python export_thesis_data.py
```

Exports to `thesis_data_export/`:
- `all_runs_summary.csv` - Overview of all experiments
- Individual history CSV files for each run

## Platform-Specific Notes

### Windows
- Uses PowerShell or CMD
- Paths handled by `pathlib.Path` (automatic)
- Multiprocessing uses 'spawn' mode

### Mac/Linux
- Uses bash/zsh
- Same code works identically
- Multiprocessing uses 'fork' or 'spawn' mode

### Cross-Platform Compatibility

All code is cross-platform:
- ✅ Path handling via `pathlib.Path`
- ✅ Process spawning via `subprocess` and `multiprocessing`
- ✅ No OS-specific commands in main code

## Monitoring

### Weights & Biases

All experiments log to W&B automatically:
- Project: `rl1`
- URL: https://wandb.ai/samnguyen1/rl1

View:
- Training curves
- Hyperparameter comparisons
- System metrics
- Real-time progress

### Local Monitoring

```bash
# Check progress
python check_progress.py

# View results file after completion
cat ppo_results.txt
```

## Expected Runtime

With 6 workers:
- **PPO**: ~14 hours (60 experiments) ✅ DONE
- **TD3**: ~14 hours (60 experiments)
- **A2C**: ~14 hours (60 experiments)
- **SAC**: ~14 hours (60 experiments)
- **Total**: ~56 hours (~2.3 days) for all 240 experiments

## Requirements
See `requirements.txt`. Key libraries:
- Python 3.8+
- Stable-Baselines3
- Gymnasium (with MuJoCo)
- PyTorch
- numpy, pandas, matplotlib
- wandb

## Environments

### HalfCheetah-v5
- 2D robot locomotion
- 6 DOF, 17D observation, 6D action
- Goal: Run forward as fast as possible
- Performance: ~6000+ reward = good

### Hopper-v5
- 2D one-legged robot
- 3 DOF, 11D observation, 3D action
- Goal: Hop forward without falling
- Performance: ~3000+ reward = good

---

**Last Updated**: 2025-10-28
**Status**: 71/240 experiments complete (PPO done, ready for Mac)
