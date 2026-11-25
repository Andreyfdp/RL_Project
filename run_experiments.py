# Run the 36-run benchmark and 18-run mixed-data perturbation ablation.

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from config import CHECKPOINT_DIR, DATA_DIR, EXPERIMENT_DIR, PHI, TRAINING_STEPS
from dataset import dataset_digest
from train import EVAL_EPISODES, EVAL_INTERVAL, TEST_EPISODES, build_parser, train

PHIS = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0)


def experiment_grid(suite, seeds):
    # Yield unique settings; default-phi mixed BCQ runs serve both suites.
    settings = []
    if suite in ("main", "all"):
        settings.extend(
            (algo, dataset, seed, PHI)
            for algo in ("bc", "td3", "bcq")
            for dataset in ("easy", "noisy", "mixed", "small")
            for seed in seeds
        )
    if suite in ("ablation", "all"):
        settings.extend(("bcq", "mixed", seed, phi) for phi in PHIS for seed in seeds)
    return list(dict.fromkeys(settings))


def completed_match(path, args, digest):
    # Skip only completed runs with the same data and evaluation protocol.
    try:
        run = json.loads(path.read_text(encoding="utf-8"))
        return (
            run.get("status") == "completed"
            and run.get("schema_version") == 2
            and run.get("algo") == args.algo
            and run.get("seed") == args.seed
            and run.get("steps") == args.steps
            and run.get("phi") == (args.phi if args.algo == "bcq" else None)
            and run.get("dataset_sha256") == digest
            and run.get("hyperparameters", {}).get("batch_size") == args.batch_size
            and run.get("evaluation", {}).get("episodes") == args.eval_episodes
            and run.get("evaluation", {}).get("interval") == args.eval_interval
            and run.get("evaluation", {}).get("test_episodes") == args.test_episodes
            and (Path(__file__).resolve().parent / run["checkpoint"]).is_file()
        )
    except (KeyError, ValueError, OSError):
        return False


def main():
    # Execute a resumable suite with a common budget and three seeds.
    parser = argparse.ArgumentParser(
        description="Run the 36-run benchmark and 18-run mixed-data perturbation ablation."
    )
    parser.add_argument("--suite", choices=("main", "ablation", "all"), default="all")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=TRAINING_STEPS)
    parser.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL)
    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--test-episodes", type=int, default=TEST_EPISODES)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=EXPERIMENT_DIR / "runs")
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR / "runs")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--dry-run", action="store_true")
    opts = parser.parse_args()
    if opts.workers < 1:
        parser.error("--workers must be positive")
    if opts.workers > 1 and opts.device != "cpu":
        parser.error("--workers > 1 requires --device cpu")
    grid = experiment_grid(opts.suite, opts.seeds)
    print(f"Unique experiments: {len(grid)}; steps per experiment: {opts.steps}", flush=True)
    pending = []
    digests = {}
    for algo, dataset, seed, phi in grid:
        args = build_parser().parse_args([])
        args.algo, args.dataset, args.seed, args.phi = (
            algo,
            opts.data_dir / f"{dataset}.npz",
            seed,
            phi,
        )
        args.steps, args.eval_episodes = opts.steps, opts.eval_episodes
        args.eval_interval, args.test_episodes = opts.eval_interval, opts.test_episodes
        args.results_dir, args.device = opts.results_dir, opts.device
        args.checkpoint_dir = opts.checkpoint_dir
        if opts.dry_run:
            print(f"{algo:3s} {dataset:5s} seed={seed} phi={phi:g}")
            continue
        if args.dataset not in digests:
            digests[args.dataset] = dataset_digest(args.dataset)
        if any(
            completed_match(path, args, digests[args.dataset])
            for path in opts.results_dir.glob(f"{algo}_{dataset}_seed{seed}*.json")
        ):
            print(f"Already completed: {algo} {dataset} seed={seed} phi={phi:g}")
            continue
        pending.append(args)
    if opts.workers == 1:
        for args in pending:
            train(args)
    elif pending:
        with ProcessPoolExecutor(max_workers=opts.workers) as executor:
            for _ in executor.map(train, pending):
                pass


if __name__ == "__main__":
    main()
