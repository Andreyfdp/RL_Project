"""Drawing what is inside every experiment dataset (picture for the slides).

Example:
    python plot_datasets.py                 # -> results/datasets_trajectories.png
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from config import DATA_DIR, DATASET_MODES, RESULTS_DIR, create_project_dirs
from visualize import draw_environment


def split_episodes(data):
    # Cutting the flat list of transitions back into separate episodes. Positions are the
    # first two numbers of the state, and we are adding the last next_state at the end so
    # the line is going all the way to where the episode really finished
    episode_ids = data["episode_ids"]
    boundaries = np.flatnonzero(np.diff(episode_ids)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(episode_ids)]])

    episodes = []
    for start, end in zip(starts, ends):
        positions = np.vstack([data["states"][start:end, :2], data["next_states"][end - 1, :2]])
        reward = data["rewards"][end - 1, 0]
        terminal = data["dones"][end - 1, 0] > 0.5
        # Final reward is +10-ish on success and -5-ish on crash, so its sign is telling us the outcome
        if terminal and reward > 0:
            outcome = "success"
        elif terminal:
            outcome = "collision"
        else:
            outcome = "timeout"
        episodes.append((positions, outcome))

    return episodes


def plot_datasets(modes, max_episodes, output_path):
    create_project_dirs()

    colors = {"success": "#2e7d32", "collision": "#c62828", "timeout": "#757575"}

    fig, axes = plt.subplots(1, len(modes), figsize=(5 * len(modes), 5.4))
    axes = np.atleast_1d(axes)

    for ax, mode in zip(axes, modes):
        data = np.load(DATA_DIR / f"{mode}.npz")
        episodes = split_episodes(data)

        draw_environment(ax)

        # Drawing only a part of episodes, otherwise the picture is just one big blob
        rng = np.random.default_rng(0)
        chosen = rng.choice(len(episodes), size=min(max_episodes, len(episodes)), replace=False)
        for index in chosen:
            positions, outcome = episodes[index]
            ax.plot(positions[:, 0], positions[:, 1], color=colors[outcome], alpha=0.35, linewidth=1)

        success = np.mean([outcome == "success" for _, outcome in episodes]) * 100
        ax.set_title(f"{mode}: {len(episodes)} ep, success {success:.0f}%")

    # One shared legend for all panels
    handles = [plt.Line2D([0], [0], color=c, lw=2, label=name) for name, c in colors.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plotting trajectories from the experiment datasets.")
    parser.add_argument("--modes", nargs="+", default=list(DATASET_MODES), choices=list(DATASET_MODES))
    parser.add_argument("--max-episodes", type=int, default=80)
    parser.add_argument("--output", type=str, default=str(RESULTS_DIR / "datasets_trajectories.png"))
    args = parser.parse_args()

    plot_datasets(args.modes, args.max_episodes, args.output)


if __name__ == "__main__":
    main()
