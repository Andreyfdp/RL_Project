# Aggregate independent seeds and plot offline navigation experiments.

import argparse
import csv
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from config import MAP_HEIGHT, MAP_WIDTH, OBSTACLES, PHI, RESULTS_DIR

METRICS = (
    "success_rate",
    "collision_rate",
    "average_reward",
    "average_steps",
    "discounted_return",
)
COLORS = {"bc": "tab:green", "td3": "tab:red", "bcq": "tab:blue"}


def load_runs(directory, steps=None):
    # Load complete runs, reject incompatible protocols, deduplicate repeats.
    selected = {}
    protocols = set()
    for path in sorted(Path(directory).glob("*.json")):
        try:
            run = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            warnings.warn(f"Skipping unreadable run: {path}")
            continue
        if run.get("status") != "completed" or (
            steps is not None and run["steps"] != steps
        ):
            continue
        protocol = (
            run["steps"],
            json.dumps(run["hyperparameters"], sort_keys=True),
            json.dumps(run["evaluation"], sort_keys=True),
        )
        protocols.add(protocol)
        key = (run["algo"], run["dataset"], run["seed"], run["phi"])
        if key in selected:
            warnings.warn(f"Repeated seed: using the latest log for {key}")
        if key not in selected or run["started_at"] > selected[key]["started_at"]:
            selected[key] = run
    if len(protocols) > 1:
        raise ValueError(
            "Incompatible training/evaluation protocols. Select --steps or separate run directories."
        )
    runs = list(selected.values())
    for dataset in {run["dataset"] for run in runs}:
        if (
            len({run["dataset_sha256"] for run in runs if run["dataset"] == dataset})
            > 1
        ):
            raise ValueError(
                f"Different dataset contents under the same name: {dataset}"
            )
    return runs


def mean_std(values):
    # Sample standard deviation across training seeds, never across episodes.
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
    return mean, std


def summarize(runs, section="final"):
    # Build one aggregate row per algorithm, dataset and BCQ phi.
    groups = {}
    for run in runs:
        key = (run["dataset"], run["algo"], run["phi"])
        groups.setdefault(key, []).append(run)
    rows = []
    for (dataset, algo, phi), group in sorted(groups.items()):
        row = {"dataset": dataset, "algo": algo, "phi": phi, "n_seeds": len(group)}
        for metric in METRICS:
            mean, std = mean_std([run[section][metric] for run in group])
            row[f"{metric}_mean"], row[f"{metric}_std"] = mean, std
        rows.append(row)
    return rows


def save_table(rows, output, name):
    # Export a machine-readable CSV and a readable mean ± std table.
    if not rows:
        return
    with (output / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "| Dataset | Algorithm | Phi | Seeds | Success | Collision | Reward | Steps | Return |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    cells = []
    for row in rows:
        values = []
        for metric in METRICS:
            mean, std = row[f"{metric}_mean"], row[f"{metric}_std"]
            values.append(
                f"{mean:.3f} ± {std:.3f}" if np.isfinite(std) else f"{mean:.3f} ± N/A"
            )
        label = [
            row["dataset"],
            row["algo"].upper(),
            str(row["phi"]),
            str(row["n_seeds"]),
        ]
        cells.append(label + values)
        lines.append("| " + " | ".join(label + values) + " |")
    (output / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(16, max(2, len(cells) * 0.42 + 1)))
    ax.axis("off")
    table = ax.table(
        cellText=cells,
        colLabels=["Dataset", "Algo", "Phi", "Seeds", *METRICS],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    fig.savefig(output / f"{name}.png", bbox_inches="tight", dpi=160)
    plt.close(fig)


def plot_curve(ax, runs, metric, label, color, linestyle="-"):
    # Plot the seed mean and ±1 standard deviation at common logged steps.
    if not runs:
        return
    by_run = [
        {entry["step"]: entry.get(metric) for entry in run["history"]} for run in runs
    ]
    steps = sorted(set.intersection(*(set(entries) for entries in by_run)))
    steps = [
        step for step in steps if all(entries[step] is not None for entries in by_run)
    ]
    if not steps:
        return
    values = np.array([[entries[step] for step in steps] for entries in by_run])
    mean = values.mean(axis=0)
    ax.plot(
        steps,
        mean,
        label=label,
        color=color,
        linestyle=linestyle,
        marker="o" if len(steps) == 1 else None,
    )
    if len(runs) > 1:
        std = values.std(axis=0, ddof=1)
        ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.12)


def make_report(directory, output, steps=None):
    # Produce result tables, Q diagnostics, learning curves and trajectories.
    runs = load_runs(directory, steps)
    if not runs:
        raise ValueError(f"No completed runs in {directory}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    main = [run for run in runs if run["algo"] != "bcq" or run["phi"] == PHI]
    save_table(summarize(main), output, "results_table")
    save_table(summarize(main, "generalization"), output, "generalization_table")
    save_table(
        summarize([r for r in runs if r["algo"] == "bcq" and r["dataset"] == "mixed"]),
        output,
        "phi_table",
    )
    datasets = sorted({run["dataset"] for run in main})
    for kind in ("success", "q") if datasets else ():
        fig, axes = plt.subplots(
            1, len(datasets), figsize=(6 * len(datasets), 4), squeeze=False
        )
        for ax, dataset in zip(axes[0], datasets):
            for algo, color in COLORS.items():
                group = [
                    r for r in main if r["dataset"] == dataset and r["algo"] == algo
                ]
                if kind == "success":
                    plot_curve(ax, group, "success_rate", algo.upper(), color)
                elif algo != "bc":
                    plot_curve(
                        ax, group, "q_dataset", f"{algo.upper()} Q(dataset)", color, ":"
                    )
                    plot_curve(ax, group, "q_start", f"{algo.upper()} Q(start)", color)
                    plot_curve(
                        ax,
                        group,
                        "discounted_return",
                        f"{algo.upper()} measured G0",
                        color,
                        "--",
                    )
            ax.set(
                title=dataset,
                xlabel="Training steps",
                ylabel="Success rate" if kind == "success" else "Q / discounted return",
            )
            if kind == "success":
                ax.set_ylim(-0.03, 1.03)
            ax.grid(alpha=0.2)
            if ax.lines:
                ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / f"{kind}_curves.png", dpi=160)
        plt.close(fig)
    ablation = [
        row
        for row in summarize(runs)
        if row["dataset"] == "mixed" and row["algo"] == "bcq"
    ]
    if ablation:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.errorbar(
            [r["phi"] for r in ablation],
            [r["success_rate_mean"] for r in ablation],
            yerr=[r["success_rate_std"] if r["n_seeds"] > 1 else 0 for r in ablation],
            fmt="o-",
            capsize=4,
        )
        ax.set(
            xlabel="Phi",
            ylabel="Final success rate",
            title="BCQ on mixed data",
            ylim=(-0.05, 1.05),
        )
        fig.tight_layout()
        fig.savefig(output / "phi_ablation.png", dpi=160)
        plt.close(fig)
    plot_trajectories(main, datasets, output)
    missing = [
        (a, d, s)
        for a in COLORS
        for d in ("easy", "noisy", "mixed", "small")
        for s in (0, 1, 2)
        if not any(
            r["algo"] == a and r["dataset"] == d and r["seed"] == s for r in main
        )
    ]
    missing_phi = [
        (phi, seed)
        for phi in (0, 0.05, 0.1, 0.25, 0.5, 1)
        for seed in (0, 1, 2)
        if not any(
            r["algo"] == "bcq"
            and r["dataset"] == "mixed"
            and r["seed"] == seed
            and r["phi"] == phi
            for r in runs
        )
    ]
    (output / "report_notes.txt").write_text(
        f"Completed unique runs: {len(runs)}\nMissing main-grid runs: {missing}\n"
        f"Missing ablation runs (phi, seed): {missing_phi}\n"
        f"Training steps per run: {runs[0]['steps']}; evaluation episodes: {runs[0]['evaluation']['episodes']}\n"
        "Tables: final policy, mean ± sample std across seeds. One seed: std=N/A.\n"
        "Q(dataset) uses dataset states; compare Q(start) with measured G0.\n"
        "G0 stops at the 120-step horizon; Bellman targets bootstrap at dataset timeouts.\n"
        "Generalization: new start/goal pairs near the original regions, not arbitrary maps.\n",
        encoding="utf-8",
    )
    print(
        f"Report saved to {output}; unique runs: {len(runs)}, missing main runs: {len(missing)}"
    )
    return output


def plot_trajectories(runs, datasets, output):
    # Plot one preselected seed per algorithm, including failed trajectories.
    if not datasets:
        return
    fig, axes = plt.subplots(
        1, len(datasets), figsize=(5 * len(datasets), 5), squeeze=False
    )
    for ax, dataset in zip(axes[0], datasets):
        for x0, y0, x1, y1 in OBSTACLES:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color="gray", alpha=0.5))
        for algo, color in COLORS.items():
            group = sorted(
                [r for r in runs if r["dataset"] == dataset and r["algo"] == algo],
                key=lambda r: r["seed"],
            )
            if not group:
                continue
            # Use the lowest seed for every algorithm; do not cherry-pick successes.
            run = group[0]
            trajectory = run["final"]["trajectory"]
            points = np.array(trajectory["positions"])
            ax.plot(*points.T, color=color, label=f"{algo.upper()} seed {run['seed']}")
            ax.scatter(*points[0], color=color, marker="o", s=20)
            ax.scatter(
                *trajectory["goal"], color="gold", edgecolor="black", marker="*", s=120
            )
        ax.set(xlim=(0, MAP_WIDTH), ylim=(0, MAP_HEIGHT), title=dataset, aspect="equal")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "trajectories.png", dpi=160)
    plt.close(fig)


def main():
    # CLI for reports from full experiments or isolated smoke runs.
    parser = argparse.ArgumentParser(
        description="Aggregate independent seeds and plot offline navigation experiments."
    )
    parser.add_argument("--runs-dir", type=Path, default=RESULTS_DIR / "runs")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "report")
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()
    make_report(args.runs_dir, args.output_dir, args.steps)


if __name__ == "__main__":
    main()
