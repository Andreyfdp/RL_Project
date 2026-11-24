# Collecting offline datasets for the BCQ experiments.

import argparse
import json

import numpy as np

from config import (
    BEHAVIOR_NOISE_STD,
    DATA_DIR,
    DATASET_EPISODES,
    DATASET_MODES,
    DATASET_PATH,
    RANDOM_ACTION_PROB,
    SEED,
    START_NOISE,
    create_project_dirs,
)
from env import NavigationEnv

# Store policy labels for dataset analysis.
GOOD_POLICY = 0
BAD_POLICY = 1


def get_route(rng):
    # Use two possible routes around the obstacle
    if rng.random() < 0.7:
        return [
            np.array([3.7, 1.7], dtype=np.float32),
            np.array([6.3, 1.7], dtype=np.float32),
            np.array([9.0, 9.0], dtype=np.float32),
        ]

    return [
        np.array([3.7, 8.3], dtype=np.float32),
        np.array([6.3, 8.3], dtype=np.float32),
        np.array([9.0, 9.0], dtype=np.float32),
    ]


def add_noise(direction, rng, noise_std, random_prob):
    # Add Gaussian action noise.
    noise = rng.normal(loc=0.0, scale=noise_std, size=2)
    action = direction + noise

    # Occasionally sample a random action.
    if rng.random() < random_prob:
        action = rng.uniform(-1.0, 1.0, size=2)

    return np.clip(action, -1.0, 1.0).astype(np.float32)


def behavior_policy(
    position,
    waypoints,
    waypoint_index,
    rng,
    noise_std=BEHAVIOR_NOISE_STD,
    random_prob=RANDOM_ACTION_PROB,
):
    # Follow waypoints around the obstacle.
    target = waypoints[waypoint_index]

    distance = np.linalg.norm(target - position)

    # Advance when the current waypoint is reached.
    if distance < 0.45 and waypoint_index < len(waypoints) - 1:
        waypoint_index += 1
        target = waypoints[waypoint_index]

    direction = target - position
    norm = np.linalg.norm(direction)

    if norm > 1e-8:
        direction = direction / norm
    else:
        direction = np.zeros(2, dtype=np.float32)

    action = add_noise(direction, rng, noise_std, random_prob)

    return action, waypoint_index


def greedy_policy(
    position, goal, rng, noise_std=BEHAVIOR_NOISE_STD, random_prob=RANDOM_ACTION_PROB
):
    # Head straight toward the goal, ignoring obstacles.
    direction = goal - position
    norm = np.linalg.norm(direction)

    if norm > 1e-8:
        direction = direction / norm
    else:
        direction = np.zeros(2, dtype=np.float32)

    return add_noise(direction, rng, noise_std, random_prob)


def collect_dataset(
    mode=None, episodes=None, out_path=None, seed=SEED, start_noise=None, verbose=True
):
    # Collecting one dataset and returning its statistics.
    create_project_dirs()

    if mode is None:
        settings = {
            "noise_std": BEHAVIOR_NOISE_STD,
            "random_prob": RANDOM_ACTION_PROB,
            "good_fraction": 1.0,
            "episodes": DATASET_EPISODES,
        }
        default_start_noise = 0.0
        default_out = DATASET_PATH
    else:
        if mode not in DATASET_MODES:
            raise ValueError(
                f"Unknown mode {mode!r}, choose from {list(DATASET_MODES)}"
            )
        settings = dict(DATASET_MODES[mode])
        default_start_noise = START_NOISE
        default_out = DATA_DIR / f"{mode}.npz"

    episodes = settings["episodes"] if episodes is None else episodes
    start_noise = default_start_noise if start_noise is None else start_noise
    out_path = default_out if out_path is None else out_path

    noise_std = settings["noise_std"]
    random_prob = settings["random_prob"]
    good_fraction = settings["good_fraction"]

    # Keep policy and environment RNGs separate.
    rng = np.random.default_rng(seed)
    env = NavigationEnv(seed=seed, start_noise=start_noise)

    states, actions, rewards, next_states, dones = [], [], [], [], []
    timeouts, episode_ids, policy_types = [], [], []

    # Track outcomes by policy type.
    outcomes = {
        GOOD_POLICY: {"episodes": 0, "success": 0, "collision": 0, "timeout": 0},
        BAD_POLICY: {"episodes": 0, "success": 0, "collision": 0, "timeout": 0},
    }
    episode_rewards = []

    for episode in range(episodes):
        state = env.reset()

        # Avoid extra RNG draws for the all-good dataset.
        if good_fraction < 1.0 and rng.random() >= good_fraction:
            policy_type = BAD_POLICY
        else:
            policy_type = GOOD_POLICY

        if policy_type == GOOD_POLICY:
            waypoints = get_route(rng)
            waypoint_index = 0

        episode_reward = 0.0

        while True:
            if policy_type == GOOD_POLICY:
                action, waypoint_index = behavior_policy(
                    position=env.position.copy(),
                    waypoints=waypoints,
                    waypoint_index=waypoint_index,
                    rng=rng,
                    noise_std=noise_std,
                    random_prob=random_prob,
                )
            else:
                action = greedy_policy(
                    position=env.position.copy(),
                    goal=env.goal_position,
                    rng=rng,
                    noise_std=noise_std,
                    random_prob=random_prob,
                )

            next_state, reward, done, info = env.step(action)

            # Only goals and collisions are terminal; timeouts still bootstrap.
            terminal = info["success"] or info["collision"]
            timeout = done and not terminal

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(float(terminal))
            timeouts.append(float(timeout))
            episode_ids.append(episode)
            policy_types.append(policy_type)

            state = next_state
            episode_reward += reward

            if done:
                stats = outcomes[policy_type]
                stats["episodes"] += 1
                if info["success"]:
                    stats["success"] += 1
                elif info["collision"]:
                    stats["collision"] += 1
                else:
                    stats["timeout"] += 1
                break

        episode_rewards.append(episode_reward)

    np.savez_compressed(
        out_path,
        states=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32).reshape(-1, 1),
        next_states=np.asarray(next_states, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.float32).reshape(-1, 1),
        # Extra arrays, OfflineDataset is just ignoring them, they are for plots and checks
        timeouts=np.asarray(timeouts, dtype=np.float32).reshape(-1, 1),
        episode_ids=np.asarray(episode_ids, dtype=np.int32),
        policy_types=np.asarray(policy_types, dtype=np.int8),
    )

    # Putting together the summary for this dataset
    total = {
        key: outcomes[GOOD_POLICY][key] + outcomes[BAD_POLICY][key]
        for key in outcomes[GOOD_POLICY]
    }
    summary = {
        "mode": mode or "default",
        "path": str(out_path),
        "seed": seed,
        "episodes": episodes,
        "transitions": len(states),
        "noise_std": noise_std,
        "random_prob": random_prob,
        "good_fraction": good_fraction,
        "start_noise": start_noise,
        "success_rate": total["success"] / episodes,
        "collision_rate": total["collision"] / episodes,
        "timeout_rate": total["timeout"] / episodes,
        "average_reward": float(np.mean(episode_rewards)),
        "good_policy": outcomes[GOOD_POLICY],
        "bad_policy": outcomes[BAD_POLICY],
    }

    if verbose:
        print_summary(summary)

    return summary


def print_summary(s):
    print(f"[{s['mode']}] -> {s['path']}")
    print(f"  episodes: {s['episodes']}  transitions: {s['transitions']}")
    print(
        f"  success: {s['success_rate'] * 100:5.1f}%  "
        f"collision: {s['collision_rate'] * 100:5.1f}%  "
        f"timeout: {s['timeout_rate'] * 100:5.1f}%  "
        f"avg reward: {s['average_reward']:.2f}"
    )
    for name in ("good_policy", "bad_policy"):
        p = s[name]
        if p["episodes"]:
            print(
                f"  {name:11s}: {p['episodes']:4d} ep | success {p['success']:4d} | "
                f"collision {p['collision']:4d} | timeout {p['timeout']:4d}"
            )
    print()


def print_table(summaries):
    # Making a small markdown table, it is easy to paste straight into README or slides
    print(
        "| Dataset | Episodes | Transitions | Good policy | Success | Collision | Timeout | Avg reward |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in summaries:
        print(
            f"| {s['mode']} | {s['episodes']} | {s['transitions']} | "
            f"{s['good_fraction'] * 100:.0f}% | {s['success_rate'] * 100:.1f}% | "
            f"{s['collision_rate'] * 100:.1f}% | {s['timeout_rate'] * 100:.1f}% | "
            f"{s['average_reward']:.2f} |"
        )


def main():
    parser = argparse.ArgumentParser(description="Collecting offline datasets for BCQ.")
    parser.add_argument(
        "--mode",
        choices=list(DATASET_MODES),
        default=None,
        help="Dataset type. Without it the old default dataset is created.",
    )
    parser.add_argument("--all", action="store_true", help="Collect all dataset modes.")
    parser.add_argument(
        "--episodes", type=int, default=None, help="Override number of episodes."
    )
    parser.add_argument("--out", type=str, default=None, help="Output .npz path.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--start-noise",
        type=float,
        default=None,
        help="Random start shift (default: 0.3 for modes, 0 for the old dataset).",
    )
    args = parser.parse_args()

    if args.all:
        if args.mode or args.out:
            parser.error("--all can't be used together with --mode or --out")

        summaries = [
            collect_dataset(
                mode=mode,
                episodes=args.episodes,
                seed=args.seed,
                start_noise=args.start_noise,
            )
            for mode in DATASET_MODES
        ]

        stats_path = DATA_DIR / "dataset_stats.json"
        with open(stats_path, "w") as f:
            json.dump(summaries, f, indent=4)

        print_table(summaries)
        print(f"\nStatistics saved to: {stats_path}")
    else:
        collect_dataset(
            mode=args.mode,
            episodes=args.episodes,
            out_path=args.out,
            seed=args.seed,
            start_noise=args.start_noise,
        )


if __name__ == "__main__":
    main()
