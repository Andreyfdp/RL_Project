# Evaluate BCQ on small, seeded start/goal perturbations.

import argparse
import random

import numpy as np
import torch

from bcq import BCQ
from config import MODEL_PATH, SEED
from env import NavigationEnv


def sample_task(rng):
    # Sample free positions near the original start and goal regions.
    start = rng.uniform(low=(0.7, 0.7), high=(1.7, 1.7)).astype(np.float32)
    goal = rng.uniform(low=(8.3, 8.3), high=(9.3, 9.3)).astype(np.float32)
    return start, goal


def evaluate(agent, episodes, seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    rewards, successes, collisions = [], 0, 0
    for _ in range(episodes):
        start, goal = sample_task(rng)
        env = NavigationEnv(seed=seed, start_position=start, goal_position=goal)
        state = env.reset()
        total_reward = 0.0

        while True:
            action = agent.select_action(state)
            state, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                rewards.append(total_reward)
                successes += int(info["success"])
                collisions += int(info["collision"])
                break

    return {
        "episodes": episodes,
        "success_rate": successes / episodes,
        "collision_rate": collisions / episodes,
        "average_reward": float(np.mean(rewards)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="BCQ robustness check on perturbed tasks."
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=SEED + 5000)
    args = parser.parse_args()

    agent = BCQ(state_dim=4, action_dim=2, max_action=1.0, device="cpu")
    agent.load(MODEL_PATH)
    metrics = evaluate(agent, args.episodes, args.seed)

    print("Perturbed start/goal robustness diagnostic")
    print("This checkpoint was trained on the fixed default task.")
    for name, value in metrics.items():
        print(
            f"{name}: {value:.3f}" if isinstance(value, float) else f"{name}: {value}"
        )


if __name__ == "__main__":
    main()
