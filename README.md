# BCQ for offline 2D navigation

- [Notebook](BCQ_Offline_Navigation_Demo.ipynb)
- [Presentation](presentation/BCQ_Offline_Navigation.pdf)
- [Benchmark results](results/benchmark/report/results_table.md)

Install dependencies and open the notebook from this directory.

```bash
python -m pip install -r requirements.txt
python -m jupyter notebook BCQ_Offline_Navigation_Demo.ipynb
```

Reproduce the benchmark in a new results directory.

```bash
python run_experiments.py --suite all --steps 2000 --eval-interval 1000 --device cpu
```

## 100,000-update BCQ experiment

BCQ trained on the mixed dataset with seed 0 and a perturbation limit of 0.05.
Evaluation ran every 5,000 updates with 30 episodes. Success with the original
goal peaked at 24/30 (80.0%) after 10,000 updates and reached 12/30 (40.0%) at
100,000 updates. The final checkpoint reached 10/30 goals (33.3%) on new starts
and goals. The 2,000-update benchmark compares algorithms across three seeds.

- [Training log](results/runs/bcq_mixed_seed0.json)
- [Final checkpoint](checkpoints/runs/bcq_mixed_seed0_final.pt)
- [Success curves](results/report/success_curves.png)
- [Q-value curves](results/report/q_curves.png)
- [Trajectories](results/report/trajectories.png)

## Reports and tests

```bash
python make_report.py --runs-dir results/benchmark/runs --output-dir results/benchmark/report --steps 2000
python make_report.py --runs-dir results/runs --output-dir results/report --steps 100000
python -m unittest -v test_experiments
```
