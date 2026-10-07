import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

from config import (
    MAP_WIDTH,
    MAP_HEIGHT,
    START_POSITION,
    GOAL_POSITION,
    OBSTACLES,
    AGENT_RADIUS,
    GOAL_RADIUS,
    RESULTS_DIR,
    create_project_dirs,
)


def draw_environment(ax):
    ax.set_xlim(0, MAP_WIDTH)
    ax.set_ylim(0, MAP_HEIGHT)
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Draw obstacles
    for x_min, y_min, x_max, y_max in OBSTACLES:
        obstacle = Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            alpha=0.45,
        )
        ax.add_patch(obstacle)

    # Draw start
    start = Circle(
        START_POSITION,
        AGENT_RADIUS * 1.5,
        alpha=0.8,
    )
    ax.add_patch(start)

    # Draw goal
    goal = Circle(
        GOAL_POSITION,
        GOAL_RADIUS,
        fill=False,
        linewidth=2,
    )
    ax.add_patch(goal)

    ax.text(
        START_POSITION[0],
        START_POSITION[1] - 0.5,
        "START",
        ha="center",
    )

    ax.text(
        GOAL_POSITION[0],
        GOAL_POSITION[1] + 0.7,
        "GOAL",
        ha="center",
    )

    ax.grid(alpha=0.25)


def plot_trajectory(trajectory, title, filename):
    fig, ax = plt.subplots(figsize=(7, 7))

    draw_environment(ax)

    ax.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        marker="o",
        markersize=3,
        linewidth=2,
    )

    ax.scatter(
        trajectory[0, 0],
        trajectory[0, 1],
        s=100,
        label="Start",
    )

    ax.scatter(
        trajectory[-1, 0],
        trajectory[-1, 1],
        s=100,
        label="End",
    )

    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=180)
    plt.close()


def plot_loss(values, title, ylabel, filename):
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(values)

    ax.set_title(title)
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=180)
    plt.close()


def plot_success_history(steps, success):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        steps,
        success * 100,
        marker="o",
    )

    ax.set_title("BCQ Success Rate During Training")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "training_success_rate.png",
        dpi=180,
    )
    plt.close()


def plot_reward_history(steps, rewards):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        steps,
        rewards,
        marker="o",
    )

    ax.set_title("BCQ Average Reward During Training")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Average reward")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "training_average_reward.png",
        dpi=180,
    )
    plt.close()


def plot_policy_success(metrics):
    names = ["Random", "Behavior", "BCQ"]

    values = [
        metrics["random"]["success_rate"] * 100,
        metrics["behavior"]["success_rate"] * 100,
        metrics["bcq"]["success_rate"] * 100,
    ]

    fig, ax = plt.subplots(figsize=(7, 5))

    bars = ax.bar(names, values)

    ax.set_title("Policy Success Rate")
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2,
            f"{value:.0f}%",
            ha="center",
        )

    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "policy_success_comparison.png",
        dpi=180,
    )
    plt.close()


def plot_policy_rewards(metrics):
    names = ["Random", "Behavior", "BCQ"]

    values = [
        metrics["random"]["average_reward"],
        metrics["behavior"]["average_reward"],
        metrics["bcq"]["average_reward"],
    ]

    fig, ax = plt.subplots(figsize=(7, 5))

    bars = ax.bar(names, values)

    ax.set_title("Average Reward Comparison")
    ax.set_ylabel("Average reward")
    ax.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{value:.1f}",
            ha="center",
        )

    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "policy_reward_comparison.png",
        dpi=180,
    )
    plt.close()


def plot_collision_comparison(metrics):
    names = ["Random", "Behavior", "BCQ"]

    values = [
        metrics["random"]["collision_rate"] * 100,
        metrics["behavior"]["collision_rate"] * 100,
        metrics["bcq"]["collision_rate"] * 100,
    ]

    fig, ax = plt.subplots(figsize=(7, 5))

    bars = ax.bar(names, values)

    ax.set_title("Collision Rate Comparison")
    ax.set_ylabel("Collision rate (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2,
            f"{value:.0f}%",
            ha="center",
        )

    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "policy_collision_comparison.png",
        dpi=180,
    )
    plt.close()


def main():
    create_project_dirs()

    history_path = RESULTS_DIR / "training_history.npz"
    metrics_path = RESULTS_DIR / "evaluation_metrics.json"
    trajectories_path = RESULTS_DIR / "evaluation_trajectories.npz"

    history = np.load(history_path)
    trajectories = np.load(trajectories_path)

    with open(metrics_path, "r", encoding="utf-8") as file:
        metrics = json.load(file)

    # Training losses
    plot_loss(
        history["vae_loss"],
        "VAE Loss",
        "Loss",
        "vae_loss.png",
    )

    plot_loss(
        history["critic_loss"],
        "Critic Loss",
        "Loss",
        "critic_loss.png",
    )

    plot_loss(
        history["actor_loss"],
        "Actor Loss",
        "Loss",
        "actor_loss.png",
    )

    # Evaluation during training
    plot_success_history(
        history["eval_steps"],
        history["eval_success"],
    )

    plot_reward_history(
        history["eval_steps"],
        history["eval_rewards"],
    )

    # Final policy comparison
    plot_policy_success(metrics)
    plot_policy_rewards(metrics)
    plot_collision_comparison(metrics)

    # Policy trajectories
    plot_trajectory(
        trajectories["random"],
        "Random Policy Trajectory",
        "trajectory_random.png",
    )

    plot_trajectory(
        trajectories["behavior"],
        "Behavior Policy Trajectory",
        "trajectory_behavior.png",
    )

    plot_trajectory(
        trajectories["bcq"],
        "BCQ Policy Trajectory",
        "trajectory_bcq.png",
    )

    print("Visualization finished.")
    print(f"Results directory: {RESULTS_DIR}")


if __name__ == "__main__":
    main()