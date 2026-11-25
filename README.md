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
