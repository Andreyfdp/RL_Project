"""Create a short animated rollout of the trained BCQ policy.

Example:
    python3 demo.py --show
"""

import argparse
import random

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import torch
from matplotlib.patches import Circle

from bcq import BCQ
from config import MODEL_PATH, RESULTS_DIR, SEED, create_project_dirs
from env import NavigationEnv
from visualize import draw_environment


def rollout(agent, seed):
    """Run one seeded policy rollout and return its positions and info."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = NavigationEnv(seed=seed)
    state = env.reset()
    positions = [env.position.copy()]
    total_reward = 0.0

    while True:
        action = agent.select_action(state)
        state, reward, done, info = env.step(action)
        total_reward += reward
        positions.append(env.position.copy())
        if done:
            return np.asarray(positions), total_reward, info


def make_animation(positions, total_reward, info, output_path, fps, show):
    fig, ax = plt.subplots(figsize=(7, 7))
    draw_environment(ax)
    line, = ax.plot([], [], color="#1565c0", linewidth=2.5, label="BCQ trajectory")
    agent_marker = Circle(positions[0], radius=0.18, color="#d32f2f", zorder=5)
    ax.add_patch(agent_marker)
    ax.legend(loc="upper left")
    ax.set_title(
        f"BCQ rollout | reward={total_reward:.2f} | "
        f"success={info['success']} | collision={info['collision']}"
    )

    def update(frame):
        line.set_data(positions[: frame + 1, 0], positions[: frame + 1, 1])
        agent_marker.center = positions[frame]
        return line, agent_marker

    animation = FuncAnimation(
        fig,
        update,
        frames=len(positions),
        interval=1000 / fps,
        blit=True,
        repeat=False,
    )
    animation.save(output_path, writer=PillowWriter(fps=fps), dpi=120)
    print(f"Saved animation: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Animate one trained BCQ rollout.")
    parser.add_argument("--output", default=RESULTS_DIR / "bcq_demo.gif", type=str)
    parser.add_argument("--fps", default=8, type=int)
    parser.add_argument("--seed", default=SEED + 4000, type=int)
    parser.add_argument("--show", action="store_true", help="Also display the animation window.")
    args = parser.parse_args()

    create_project_dirs()
    agent = BCQ(state_dim=4, action_dim=2, max_action=1.0, device="cpu")
    agent.load(MODEL_PATH)

    positions, total_reward, info = rollout(agent, args.seed)
    make_animation(positions, total_reward, info, args.output, args.fps, args.show)


if __name__ == "__main__":
    main()
