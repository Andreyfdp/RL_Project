# Reproducible offline experiments: BC, TD3, BCQ and held-out tasks.

import argparse
import json
import random
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from baselines import BC, TD3
from bcq import BCQ
from config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    DATASET_PATH,
    EXPERIMENT_DIR,
    GAMMA,
    LEARNING_RATE,
    MAX_EPISODE_STEPS,
    NUM_ACTION_SAMPLES,
    PHI,
    SEED,
    START_NOISE,
    TAU,
    TRAINING_STEPS,
)
from dataset import OfflineDataset, dataset_digest
from env import NavigationEnv

EVAL_INTERVAL = 5_000
EVAL_EPISODES = 30
EVAL_SEED = 10_000
TEST_EPISODES = 100
TEST_SEED = 30_000
GENERALIZATION_SEED = 40_000
PROBE_SEED = 20_000


def set_seed(seed):
    # Seed Python, NumPy and PyTorch before constructing any networks.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@contextmanager
def isolated_rng(seed):
    # Repeat diagnostics without consuming the training RNG stream.
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            set_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def make_agent(algo, device="cpu", phi=PHI):
    # Construct algorithms with shared dimensions and hyperparameters.
    cls = {"bc": BC, "td3": TD3, "bcq": BCQ}[algo]
    kwargs = {"phi": phi} if algo == "bcq" else {}
    return cls(4, 2, max_action=1.0, device=device, **kwargs)


@torch.no_grad()
def policy_q(agent, states):
    # Mean Q1(s, pi(s)); BC has no critic and returns None.
    if not hasattr(agent, "critic"):
        return None
    # Chunk the 100 BCQ candidates per state to bound diagnostic memory.
    values = [
        agent.critic.q1(chunk, agent.policy_actions(chunk)).flatten()
        for chunk in states.split(64)
    ]
    value = torch.cat(values).mean().item()
    if not np.isfinite(value):
        raise FloatingPointError("Non-finite policy Q estimate")
    return value


def evaluate(agent, episodes=EVAL_EPISODES, seed=EVAL_SEED, generalization=False):
    # Measure metrics and paired start-state Q/discounted return.
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    rewards, returns, lengths, initial_q = [], [], [], []
    successes = collisions = 0
    trajectory = None
    with isolated_rng(seed):
        rng = np.random.default_rng(seed)
        for episode in range(episodes):
            if generalization:
                start = rng.uniform((0.7, 0.7), (1.7, 1.7))
                goal = rng.uniform((8.3, 8.3), (9.3, 9.3))
                env = NavigationEnv(
                    seed=seed + episode, start_position=start, goal_position=goal
                )
            else:
                env = NavigationEnv(seed=seed + episode, start_noise=START_NOISE)
            state = env.reset()
            points = [env.position.tolist()]
            total = discounted = 0.0
            for t in range(MAX_EPISODE_STEPS):
                action = agent.select_action(state)
                # Q(s0,a0) and the rollout share the actual first action.
                if t == 0 and hasattr(agent, "critic"):
                    with torch.no_grad():
                        s = torch.as_tensor(state, device=agent.device).unsqueeze(0)
                        a = torch.as_tensor(action, device=agent.device).unsqueeze(0)
                        initial_q.append(agent.critic.q1(s, a).item())
                state, reward, done, info = env.step(action)
                total += reward
                discounted += agent.gamma**t * reward
                points.append(env.position.tolist())
                if done:
                    break
            rewards.append(total)
            returns.append(discounted)
            lengths.append(info["steps"])
            successes += int(info["success"])
            collisions += int(info["collision"])
            if episode == 0:
                trajectory = {
                    "positions": points,
                    "goal": env.goal_position.tolist(),
                    "success": bool(info["success"]),
                    "collision": bool(info["collision"]),
                }
    return {
        "episodes": episodes,
        "success_rate": successes / episodes,
        "collision_rate": collisions / episodes,
        "average_reward": float(np.mean(rewards)),
        "average_steps": float(np.mean(lengths)),
        "discounted_return": float(np.mean(returns)),
        "discounted_return_std": float(np.std(returns)),
        "q_start": float(np.mean(initial_q)) if initial_q else None,
        "trajectory": trajectory,
    }


def write_json(path, payload):
    # Atomically replace a log so interrupted writes cannot corrupt it.
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def reserve_run(directory, name):
    # Reserve a fresh filename, including repeated runs of the same seed.
    directory.mkdir(parents=True, exist_ok=True)
    index = 0
    while True:
        suffix = f"_repeat{index}" if index else ""
        path = directory / f"{name}{suffix}.json"
        try:
            with path.open("x", encoding="utf-8") as handle:
                handle.write("{}")
            return path
        except FileExistsError:
            index += 1


def train(args=None):
    # Train for the full budget and save independent experiment artifacts.
    args = build_parser().parse_args() if args is None else args
    for name in (
        "steps",
        "batch_size",
        "eval_interval",
        "eval_episodes",
        "test_episodes",
        "probe_size",
        "threads",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.seed < 0 or not np.isfinite(args.phi) or args.phi < 0:
        raise ValueError("seed and phi must be nonnegative and phi must be finite")
    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = Path(args.dataset).resolve()
    dataset = OfflineDataset(dataset_path, device=device)
    if not len(dataset):
        raise ValueError("Dataset is empty")
    agent = make_agent(args.algo, device, args.phi)
    # Identical probe indices across seeds and algorithms for a given dataset.
    indices = np.random.default_rng(PROBE_SEED).choice(
        len(dataset), min(args.probe_size, len(dataset)), replace=False
    )
    probe = dataset.states[torch.as_tensor(indices, device=device)]
    phi_suffix = f"_phi{args.phi:g}" if args.algo == "bcq" and args.phi != PHI else ""
    name = f"{args.algo}_{dataset_path.stem}_seed{args.seed}{phi_suffix}"
    run_path = reserve_run(Path(args.results_dir), name)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / f"{run_path.stem}_best.pt"
    project_dir = Path(__file__).resolve().parent
    dataset_location = (
        dataset_path.relative_to(project_dir) if dataset_path.is_relative_to(project_dir)
        else dataset_path
    )
    checkpoint_location = checkpoint.resolve()
    if checkpoint_location.is_relative_to(project_dir):
        checkpoint_location = checkpoint_location.relative_to(project_dir)
    run = {
        "schema_version": 2,
        "status": "running",
        "algo": args.algo,
        "dataset": dataset_path.stem,
        "dataset_path": dataset_location.as_posix(),
        "dataset_sha256": dataset_digest(dataset_path),
        "dataset_size": len(dataset),
        "seed": args.seed,
        "phi": args.phi if args.algo == "bcq" else None,
        "steps": args.steps,
        "device": str(device),
        "torch_version": torch.__version__,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "hyperparameters": {
            "batch_size": args.batch_size,
            "gamma": GAMMA,
            "tau": TAU,
            "learning_rate": LEARNING_RATE,
            "hidden_sizes": [256, 256],
        },
        "algorithm_hyperparameters": (
            {
                "phi": args.phi,
                "train_candidates": 10,
                "eval_candidates": NUM_ACTION_SAMPLES,
            }
            if args.algo == "bcq"
            else {"policy_noise": 0.2, "noise_clip": 0.5, "policy_delay": 2}
            if args.algo == "td3"
            else {"objective": "action_mse"}
        ),
        "evaluation": {
            "interval": args.eval_interval,
            "episodes": args.eval_episodes,
            "seed": EVAL_SEED,
            "selection": "success_rate_then_average_reward",
            "test_episodes": args.test_episodes,
            "test_seed": TEST_SEED,
            "start_noise": START_NOISE,
            "generalization_seed": GENERALIZATION_SEED,
            "return_horizon": MAX_EPISODE_STEPS,
            "q_definition": "Q1(s, policy(s))",
        },
        "probe_indices": indices.tolist(),
        "probe_seed": PROBE_SEED,
        "checkpoint": checkpoint_location.as_posix(),
        "history": [],
        "loss_history": [],
    }
    write_json(run_path, run)
    started = time.perf_counter()
    recent = {}
    best_score = (-float("inf"), -float("inf"))
    try:
        for step in range(1, args.steps + 1):
            losses = agent.train(dataset.sample(args.batch_size))
            for key, value in losses.items():
                if not np.isfinite(value):
                    raise FloatingPointError(f"{key} is non-finite at step {step}")
                recent.setdefault(key, []).append(value)
            if step % 1000 == 0 or step == args.steps:
                run["loss_history"].append(
                    {
                        "step": step,
                        **{
                            key: float(np.mean(values))
                            for key, values in recent.items()
                        },
                    }
                )
                recent.clear()
            if step % args.eval_interval == 0 or step == args.steps:
                metrics = evaluate(agent, args.eval_episodes)
                metrics.pop("trajectory")
                with isolated_rng(PROBE_SEED):
                    metrics["q_dataset"] = policy_q(agent, probe)
                metrics["step"] = step
                run["history"].append(metrics)
                score = (metrics["success_rate"], metrics["average_reward"])
                if score > best_score:
                    agent.save(checkpoint)
                    best_score = score
                    run["selected_step"] = step
                write_json(run_path, run)
                print(
                    f"{run_path.stem} | {step}/{args.steps} | "
                    f"success={metrics['success_rate']:.3f} "
                    f"collision={metrics['collision_rate']:.3f} "
                    f"reward={metrics['average_reward']:.3f} "
                    f"steps={metrics['average_steps']:.1f} "
                    f"Q={metrics['q_dataset']} G0={metrics['discounted_return']:.3f}",
                    flush=True,
                )
        agent.load(checkpoint)
        run["final"] = evaluate(agent, args.test_episodes, seed=TEST_SEED)
        run["generalization"] = evaluate(
            agent, args.test_episodes, seed=GENERALIZATION_SEED, generalization=True
        )
        run["status"] = "completed"
    except Exception as error:
        run["status"] = "failed"
        run["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        run["elapsed_seconds"] = time.perf_counter() - started
        write_json(run_path, run)
    return run_path


def build_parser():
    # Expose one CLI for individual runs and experiment suites.
    parser = argparse.ArgumentParser(
        description="Reproducible offline experiments: BC, TD3, BCQ and held-out tasks."
    )
    parser.add_argument("--algo", choices=("bcq", "bc", "td3"), default="bcq")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--phi", type=float, default=PHI)
    parser.add_argument("--steps", type=int, default=TRAINING_STEPS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL)
    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--test-episodes", type=int, default=TEST_EPISODES)
    parser.add_argument("--probe-size", type=int, default=256)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--results-dir", type=Path, default=EXPERIMENT_DIR / "runs")
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR / "runs")
    return parser


if __name__ == "__main__":
    train()
