# Offline Navigation: BC, TD3 and BCQ

A continuous navigation agent moves across a 10x10 map with a rectangular obstacle.
The state is `[x, y, goal_x - x, goal_y - y]`; actions lie in `[-1, 1]^2`.
Training uses a fixed dataset. Evaluation transitions never enter the training data.

## Setup

Python 3.10+:

```bash
python -m pip install -r requirements.txt
```

CUDA is selected automatically when available. Use `--device cpu` or `--device cuda`
to override it. Shared defaults: batch size 256, Adam learning rate 0.001,
gamma 0.99, tau 0.005, and two hidden layers of 256 units.

## Datasets

```bash
python collect_dataset.py --all
python check_datasets.py
python plot_datasets.py
```

Dataset files are already available in `data/`.

| Dataset | Behavior |
|---|---|
| easy | Waypoint controller; noise 0.3, 5% random actions |
| noisy | Waypoint controller; noise 0.6, 25% random actions |
| mixed | 40% waypoint controller, 60% greedy controller |
| small | Mixed behavior with only 100 episodes |

Training starts vary by +/-0.3. The greedy controller heads toward the goal through
the obstacle. `dones` marks goals and collisions; `timeouts` is separate.
Bellman targets keep bootstrapping at dataset timeouts.

## Training

```bash
python train.py --algo bcq --dataset data/mixed.npz --seed 0 --phi 0.05 --steps 100000
python train.py --algo bc --dataset data/easy.npz --seed 1 --steps 100000
python train.py --algo td3 --dataset data/noisy.npz --seed 2 --steps 100000
```

BC fits dataset actions with MSE. Offline TD3 has no behavior constraint and uses
target smoothing (noise 0.2, clip 0.5), twin critics, and policy delay 2.
BCQ combines a conditional VAE, bounded perturbations and twin critics.
All runs use the configured training budget. The checkpoint with the highest
validation success rate is selected. Mean validation reward breaks ties, and an
exact tie keeps the earlier checkpoint.

Every 5,000 updates and at the final step, the run records:

| JSON field | Meaning |
|---|---|
| `success_rate` | Fraction of successful episodes |
| `collision_rate` | Fraction of collision episodes |
| `average_reward` | Mean undiscounted episode reward |
| `average_steps` | Mean episode length |
| `q_dataset` | Mean Q1(s, policy(s)) on fixed dataset states |
| `q_start` | Mean Q1(s0, a0) at evaluation starts |
| `discounted_return` | Measured sum of gamma^t * reward_t from those starts |

BC has no critic, so its Q fields are `null`. Compare `q_start` with
`discounted_return`; `q_dataset` uses a different state distribution.
Measured returns stop at the episode horizon of 120 steps without a bootstrap.

Validation uses 30 episodes on fixed task seeds. Each selected checkpoint is
tested on 100 independently seeded episodes with the original goal and 100
with new start and goal positions. Evaluation restores the training RNG state.
Run metadata includes the selected step, probe indices, a SHA-256 digest of the
dataset arrays, hyperparameters, and evaluation settings.

```text
results/experiments/runs/bcq_mixed_seed0.json
checkpoints/runs/bcq_mixed_seed0_best.pt
```

Nondefault Phi values add a suffix such as `_phi0.25`. Repeated runs receive
`_repeat1`, `_repeat2`, etc. Logs are written atomically and record
`running`, `completed` or `failed` status.

Other options: `--eval-episodes`, `--eval-interval`, `--test-episodes`, `--batch-size`, `--probe-size`,
`--threads`, `--results-dir`, and `--checkpoint-dir`.

## Experiment Suites

```bash
python run_experiments.py --suite main --steps 100000
python run_experiments.py --suite ablation --steps 100000
```

Reproduce the 2,000-update benchmark:

```bash
python run_experiments.py --suite all --steps 2000 --eval-interval 1000 --device cpu --workers 4 --results-dir results/benchmark/runs --checkpoint-dir checkpoints/benchmark
python make_report.py --runs-dir results/benchmark/runs --output-dir results/benchmark/report --steps 2000
```

The main grid has 36 runs: `{BC, TD3, BCQ} x {easy, noisy, mixed, small} x {0, 1, 2}`.
The ablation evaluates BCQ on mixed data at Phi values `0, 0.05, 0.1, 0.25, 0.5, 1.0`.
Both suites share three Phi=0.05 runs, giving 51 unique experiments.

Use `--dry-run` to inspect the grid. Supply `--device cpu --workers N`
for parallel CPU runs. For a different training budget or validation interval,
use a separate results directory.
Repeating a command skips completed compatible runs. Interrupted runs restart
from scratch under a new filename.

Each final policy is also tested on new tasks: starts in `[0.7, 1.7]^2` and goals
in `[8.3, 9.3]^2`. This tests nearby start/goal pairs, not arbitrary maps.
Training datasets use the fixed goal `(9, 9)`. Results are stored in `generalization`.

## Reports

`make_report.py` writes these files to the report directory:

| Files | Content |
|---|---|
| `results_table.{csv,md,png}` | Main results: mean +/- sample std across seeds |
| `generalization_table.{csv,md,png}` | New start/goal results |
| `phi_table.{csv,md,png}`, `phi_ablation.png` | Phi ablation |
| `q_curves.png` | TD3/BCQ Q estimates and measured return |
| `success_curves.png` | Success rate versus training updates |
| `trajectories.png` | BC, TD3 and BCQ trajectories at a preselected seed |
| `report_notes.txt` | Missing runs and interpretation notes |

Standard deviation uses `ddof=1` across training seeds. One seed gives `N/A`.
Repeated seeds count once, using the latest completed run. Incompatible budgets,
evaluation protocols or dataset contents cannot be mixed. Use `--steps`,
`--runs-dir` and `--output-dir` to select a separate series.

## Notebook

`BCQ_Offline_Navigation_Demo.ipynb` includes Colab setup with `git clone`, a real
transition walkthrough, VAE/actor/critic values before and after updates,
and benchmark tables and plots. Outputs are saved.

```bash
python build_notebook.py --execute
```

## Verification

```bash
python -m unittest -v test_experiments
python run_experiments.py --suite all --dry-run
```

The benchmark run logs and report are in `results/benchmark/`. The training
budget and validation frequency are recorded in each log.

## Source Files

| File | Purpose |
|---|---|
| `env.py`, `gym_env.py` | Navigation environment and Gymnasium adapter |
| `config.py` | Environment and training defaults |
| `collect_dataset.py`, `dataset.py` | Dataset generation and loading |
| `check_datasets.py`, `plot_datasets.py` | Dataset validation and visualization |
| `models.py`, `bcq.py`, `baselines.py` | Networks and offline algorithms |
| `train.py`, `run_experiments.py` | Individual runs and experiment grids |
| `make_report.py` | Aggregate tables and comparison plots |
| `build_notebook.py` | Notebook generation and execution |
| `test_experiments.py` | Experiment regression tests |
| `demo.py` | Historical BCQ animation |
| `evaluate.py`, `evaluate_generalization.py`, `visualize.py` | Historical BCQ evaluation and plots |
