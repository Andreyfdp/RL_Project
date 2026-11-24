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
All runs use their full budget without early stopping; tables use the final policy.

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

Evaluation uses 30 episodes and fixed task seeds. It restores the training RNG state.
Run metadata includes probe indices, dataset SHA256, hyperparameters and evaluation settings.

```text
results/runs/bcq_mixed_seed0.json
checkpoints/runs/bcq_mixed_seed0_final.pt
```

Nondefault Phi values add a suffix such as `_phi0.25`. Repeated runs receive
`_repeat1`, `_repeat2`, etc. Logs are written atomically and record
`running`, `completed` or `failed` status.

Other options: `--eval-episodes`, `--eval-interval`, `--batch-size`, `--probe-size`,
`--threads`, `--results-dir`, and `--checkpoint-dir`.

## Experiment Suites

```bash
python run_experiments.py --suite main --steps 100000
python run_experiments.py --suite ablation --steps 100000
```

Run both suites:

```bash
python run_experiments.py --suite all --steps 100000
python make_report.py
```

The main grid has 36 runs: `{BC, TD3, BCQ} x {easy, noisy, mixed, small} x {0, 1, 2}`.
The ablation evaluates BCQ on mixed data at Phi values `0, 0.05, 0.1, 0.25, 0.5, 1.0`.
Both suites share three Phi=0.05 runs, giving 51 unique experiments.

Use `--dry-run` to inspect the grid. Options include `--seeds`, `--device`,
`--eval-episodes`, `--data-dir`, `--results-dir`, and `--checkpoint-dir`.
Repeating a command skips completed compatible runs. Interrupted runs restart
from scratch under a new filename.

Each final policy is also tested on new tasks: starts in `[0.7, 1.7]^2` and goals
in `[8.3, 9.3]^2`. This tests nearby start/goal pairs, not arbitrary maps.
Training datasets use the fixed goal `(9, 9)`. Results are stored in `generalization`.

## Reports

`make_report.py` writes these files to `results/report/`:

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

## Notebook and Presentation Files

`BCQ_Offline_Navigation_Demo.ipynb` includes Colab setup with `git clone`, a real
transition walkthrough, VAE/actor/critic values before and after updates,
experiment plots, and an embedded navigation animation. Outputs are saved.

```bash
python build_notebook.py --execute
```

The animation uses the historical `checkpoints/bcq_final.pt` when available.
That checkpoint is excluded from the new comparison. If only smoke results exist,
the notebook labels them explicitly.

| Submission item | Project files |
|---|---|
| Presentation slides | Not created; use this README, notebook and result plots |
| Demo video | `demo.py` creates `results/bcq_demo.gif`; MP4 is not created |
| Source code | All `.py` files and `requirements.txt` |
| Algorithm update example | `BCQ_Offline_Navigation_Demo.ipynb` with saved outputs |
| Submission ZIP | Not created |

## Verification

```bash
python -m unittest -v test_experiments
python run_experiments.py --suite all --dry-run
```

All 51 smoke experiments completed with **100 updates and 2 evaluation episodes**.
Artifacts are in `results/smoke_suite/`, ignored by Git. These validate execution,
not converged performance. The full 100,000-update suite has not been run.

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
