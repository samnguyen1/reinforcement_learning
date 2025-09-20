# RL Convergence Stability Project

This project investigates the convergence stability of reinforcement learning algorithms (specifically PPO and SAC) across multiple continuous control environments. We focus on MuJoCo-based Gymnasium environments like HalfCheetah and Hopper, evaluating performance over multiple random seeds to assess stability and variance.

## Project Structure

- **`environments.md`** – Descriptions and key details of each environment (observation/action space, termination conditions, reward thresholds).
- **`configs/`** – Configuration files (YAML) for different experiments. Each file specifies the algorithm, environment, and hyperparameters.
- **`experiments/`** – Training scripts:
  - `train.py` – General training script that reads a config and runs the specified algorithm.
  - `train_ppo.py` / `train_sac.py` – Convenient wrappers for running PPO or SAC experiments (optional; they call `train.py` with appropriate configs).
  - `sweep.py` – Script to launch multiple runs (e.g., hyperparameter sweeps or multiple seeds).
- **`scripts/`** – Utility shell scripts (for launching multiple seeds, etc.), e.g. `run_all_seeds.sh` to run a given experiment across several random seeds sequentially.
- **`utils/`** – Helper modules:
  - `make_env.py` – Environment factory to create monitored vectorized environments.
  - `plot_utils.py` – Functions for loading results and plotting learning curves with confidence intervals.
  - `callbacks.py` – Custom callbacks (e.g., early stopping or custom logging) for training.
- **`logs/`** – Training logs and outputs (ignored by version control):
  - `wandb/` – Weights & Biases run data (if W&B logging is enabled).
  - `tensorboard/` – TensorBoard log files for training metrics.
  - `monitor/` – Episode Monitor CSV logs for each run (episode rewards, lengths, etc.).
- **`models/`** – Saved model checkpoints.
- **`analysis/`** – Notebooks and scripts for analyzing results:
  - `plot_learning_curves.ipynb` – Jupyter Notebook for visualizing learning curves.
  - `stats_analysis.py` – Script for statistical analysis (e.g., computing AUC, performing t-tests across runs).
- **`requirements.txt`** – Python dependencies required to run the project.

## Getting Started

1. **Install dependencies**: Create a Python virtual environment and install requirements:
   ```bash
   pip install -r requirements.txt
Ensure you have a MuJoCo environment set up if using Gymnasium MuJoCo environments (e.g., install gymnasium[mujoco] and have the MuJoCo engine available).

2. **Running an experiment**: Use the training scripts with a config file. For example, to run PPO on HalfCheetah:
    ```bash
    python experiments/train.py --config configs/ppo_halfcheetah.yaml --seed 0
This will train the agent according to the config, log metrics to TensorBoard (and W&B if enabled), and save the final model to `models/`.

3. **Multiple seeds**: To run the same experiment across several seeds:
    ```bash
    bash scripts/run_all_seeds.sh configs/ppo_halfcheetah.yaml 5
The above runs seeds `0–4` sequentially.

4. **Monitoring training**:
    ```bash
    tensorboard --logdir logs/tensorboard
If Weights & Biases is enabled, metrics and videos will be logged to your W&B project.

5. **Analysing results**:
After training, use the tools in analysis/:
- Run analysis/plot_learning_curves.ipynb to visualize average learning curves with confidence intervals.
- Use analysis/stats_analysis.py to compute metrics like area under the curve (AUC) and perform statistical tests (e.g., t-tests) to compare algorithms or hyperparameter settings.
### Example:
To compare PPO vs SAC on HalfCheetah:
    ```bash
    # Run PPO
    bash scripts/run_all_seeds.sh configs/ppo_halfcheetah.yaml 5
    # Run SAC
    bash scripts/run_all_seeds.sh configs/sac_halfcheetah.yaml 5
    ```
Then use the analysis notebook or scripts to plot curves and compute statistics.

## Project Status
- Algorithms: Implemented using Stable-Baselines3 (PyTorch) for PPO and SAC.
- Logging: Integrated with TensorBoard and optional Weights & Biases.
- Environments: Tested on Gymnasium MuJoCo environments (HalfCheetah, Hopper, etc.).
- Goal: Evaluate convergence stability (variance across runs, speed to reach reward thresholds) for different algorithms and hyperparameter settings.

## Requirements
See `requirements.txt`. Key libraries:
- Python 3.8+
- Stable-Baselines3
- Gymnasium (with MuJoCo or PyBullet for continuous control)
- PyTorch
- numpy, pandas, matplotlib, scipy
- wandb (optional)

## Acknowledgements
Built using OpenAI Gymnasium environments and Stable-Baselines3 RL algorithms. W&B is used for experiment tracking.

## environments.md

# Environments and Benchmarks

This project uses the following continuous control environments (via Gymnasium):

- **HalfCheetah-v4** – A 2D half-cheetah robot that runs forward. 
  - **State (Observation)**: 17-dimensional vector (position/velocity of joints, excluding absolute x-position by default).
  - **Action Space**: 6-dimensional continuous (torques applied to joints).
  - **Episode Termination**: No early termination for falling (the half-cheetah cannot "fall"); episodes are truncated at 1000 timesteps.
  - **Reward**: Combination of forward velocity reward and a control cost:contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}. A well-performing policy can achieve high forward velocities, yielding episode rewards on the order of 5000–8000+.
  - **Performance Benchmark**: There is no official "solved" threshold since the return can grow with faster running speeds, but we consider consistent returns above ~6000 as good performance for HalfCheetah.

- **Hopper-v4** – A 2D one-legged hopper robot that must learn to hop/balance.
  - **State (Observation)**: 11-dimensional vector (position/velocity of the hopper’s parts).
  - **Action Space**: 3-dimensional continuous (torques for the leg joints).
  - **Episode Termination**: Episodes end when the hopper falls (i.e., an unsafe angle) or after 1000 timesteps (truncation):contentReference[oaicite:2]{index=2}.
  - **Reward**: Reward is given for forward hopping velocity, plus a constant alive bonus, minus a penalty for large actions. Successful hopping yields higher returns, up to a few thousand per episode for a good policy.
  - **Performance Benchmark**: Historically, achieving ~3000+ return indicates the hopper is performing well. A perfectly stable hopper can get closer to the maximum (which may be around 3800–4000 for 1000 steps if it never falls).

*Note:* Both environments use MuJoCo physics. Ensure MuJoCo is installed and gymnasium is configured to use the MuJoCo engine for these tasks. The above reward thresholds are rough guidelines to understand performance (they are not hard stops for training, but can be used for early stopping criteria or evaluation).
