import json
import random

import numpy as np
import torch

from bcq import BCQ
from collect_dataset import behavior_policy, get_route
from config import MODEL_PATH, RESULTS_DIR, SEED, EVALUATION_EPISODES, create_project_dirs
from env import NavigationEnv


def evaluate_random(episodes, seed):
    rng = np.random.default_rng(seed)
    env = NavigationEnv(seed=seed)

    rewards = []
    steps = []
    successes = 0
    collisions = 0
    first_trajectory = None

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0.0
        trajectory = [env.position.copy()]

        while True:
            action = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)

            state, reward, done, info = env.step(action)

            total_reward += reward
            trajectory.append(env.position.copy())

            if done:
                rewards.append(total_reward)
                steps.append(info["steps"])

                successes += int(info["success"])
                collisions += int(info["collision"])

                if episode == 0:
                    first_trajectory = np.asarray(trajectory, dtype=np.float32)

                break

    return create_metrics(rewards, steps, successes, collisions, episodes), first_trajectory


def evaluate_behavior(episodes, seed):
    rng = np.random.default_rng(seed)
    env = NavigationEnv(seed=seed)

    rewards = []
    steps = []
    successes = 0
    collisions = 0
    first_trajectory = None

    for episode in range(episodes):
        state = env.reset()

        waypoints = get_route(rng)
        waypoint_index = 0

        total_reward = 0.0
        trajectory = [env.position.copy()]

        while True:
            action, waypoint_index = behavior_policy(
                env.position.copy(),
                waypoints,
                waypoint_index,
                rng,
            )

            state, reward, done, info = env.step(action)

            total_reward += reward
            trajectory.append(env.position.copy())

            if done:
                rewards.append(total_reward)
                steps.append(info["steps"])

                successes += int(info["success"])
                collisions += int(info["collision"])

                if episode == 0:
                    first_trajectory = np.asarray(trajectory, dtype=np.float32)

                break

    return create_metrics(rewards, steps, successes, collisions, episodes), first_trajectory


def evaluate_bcq(agent, episodes, seed):
    env = NavigationEnv(seed=seed)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    rewards = []
    steps = []
    successes = 0
    collisions = 0
    first_trajectory = None

    for episode in range(episodes):
        state = env.reset()

        total_reward = 0.0
        trajectory = [env.position.copy()]

        while True:
            action = agent.select_action(state)

            state, reward, done, info = env.step(action)

            total_reward += reward
            trajectory.append(env.position.copy())

            if done:
                rewards.append(total_reward)
                steps.append(info["steps"])

                successes += int(info["success"])
                collisions += int(info["collision"])

                if episode == 0:
                    first_trajectory = np.asarray(trajectory, dtype=np.float32)

                break

    return create_metrics(rewards, steps, successes, collisions, episodes), first_trajectory


def create_metrics(rewards, steps, successes, collisions, episodes):
    return {
        "success_rate": successes / episodes,
        "collision_rate": collisions / episodes,
        "average_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "average_steps": float(np.mean(steps)),
        "std_steps": float(np.std(steps)),
    }


def print_metrics(name, metrics):
    print(name)
    print(f"Success rate:   {metrics['success_rate'] * 100:.2f}%")
    print(f"Collision rate: {metrics['collision_rate'] * 100:.2f}%")
    print(f"Average reward: {metrics['average_reward']:.3f}")
    print(f"Reward std:     {metrics['std_reward']:.3f}")
    print(f"Average steps:  {metrics['average_steps']:.2f}")
    print(f"Steps std:      {metrics['std_steps']:.2f}")
    print()


def main():
    create_project_dirs()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    agent = BCQ(
        state_dim=4,
        action_dim=2,
        max_action=1.0,
        device=device,
    )

    agent.load(MODEL_PATH)

    print(f"Model: {MODEL_PATH}")
    print(f"Device: {device}")
    print(f"Episodes per policy: {EVALUATION_EPISODES}")
    print()

    random_metrics, random_trajectory = evaluate_random(
        EVALUATION_EPISODES,
        SEED + 2000,
    )

    behavior_metrics, behavior_trajectory = evaluate_behavior(
        EVALUATION_EPISODES,
        SEED + 3000,
    )

    bcq_metrics, bcq_trajectory = evaluate_bcq(
        agent,
        EVALUATION_EPISODES,
        SEED + 4000,
    )

    print("============================================================")
    print("FINAL EVALUATION")
    print("============================================================")
    print()

    print_metrics("RANDOM POLICY", random_metrics)
    print_metrics("BEHAVIOR POLICY", behavior_metrics)
    print_metrics("BCQ POLICY", bcq_metrics)

    metrics = {
        "random": random_metrics,
        "behavior": behavior_metrics,
        "bcq": bcq_metrics,
    }

    with open(RESULTS_DIR / "evaluation_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    np.savez(
        RESULTS_DIR / "evaluation_trajectories.npz",
        random=random_trajectory,
        behavior=behavior_trajectory,
        bcq=bcq_trajectory,
    )

    print("Results saved:")
    print(RESULTS_DIR / "evaluation_metrics.json")
    print(RESULTS_DIR / "evaluation_trajectories.npz")


if __name__ == "__main__":
    main()