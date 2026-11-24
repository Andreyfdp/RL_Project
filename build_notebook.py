"""Build and optionally execute the presentation notebook with real outputs."""

import argparse
import textwrap
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent


def build_notebook():
    """Create a self-contained walkthrough of the actual BCQ update equations."""
    cells = []

    def markdown(source):
        """Append a readable explanation to the notebook."""
        cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(source).strip()))

    def code(source):
        """Append executable code without incidental builder indentation."""
        cells.append(nbf.v4.new_code_cell(textwrap.dedent(source).strip()))

    markdown(r"""
    # Offline navigation: BC, TD3 и BCQ

    Ноутбук показывает реальный переход из `data/mixed.npz`, все стадии BCQ и
    изменение сетей после одного шага. Сохранённые outputs вычислены на CPU.
    Демонстрационные короткие обучения не являются результатами полного эксперимента.
    """)
    markdown("""
    ## 1. Google Colab: клонирование и установка

    В Colab выполните ячейку ниже. Локально она только сообщает текущий каталог.
    Для GPU выберите Runtime → Change runtime type → GPU.
    Новые локальные изменения появятся при клонировании после их публикации в репозитории.
    """)
    code("""
    import importlib.util
    import os
    from pathlib import Path
    import subprocess
    import sys

    if importlib.util.find_spec("google.colab") is not None:
        project = Path("/content/RL_Project")
        if not project.exists():
            subprocess.run([
                "git", "clone", "https://github.com/Andreyfdp/RL_Project.git", str(project)
            ], check=True)
        os.chdir(project)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
    print("Project:", Path.cwd())
    """)
    code("""
    %matplotlib inline
    import json
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    from IPython.display import HTML, Image, Markdown, display
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Rectangle

    from bcq import BCQ
    from config import MODEL_PATH, OBSTACLES, RESULTS_DIR
    from dataset import OfflineDataset
    from train import build_parser, evaluate, isolated_rng, make_agent, set_seed, train

    torch.set_num_threads(1)
    set_seed(0)
    dataset = OfflineDataset("data/mixed.npz", device="cpu")
    print("Transitions:", len(dataset))
    print("PyTorch:", torch.__version__, "CUDA:", torch.cuda.is_available())
    """)
    markdown(r"""
    ## 2. Один реальный переход

    Состояние содержит `[x, y, goal_x - x, goal_y - y]`.
    `done` обозначает цель или столкновение. Ограничение длины эпизода хранится
    отдельно в `timeouts` и не зануляет Bellman bootstrap.
    Для наглядности обновляем сети на одном переходе; обычное обучение использует batch=256.
    """)
    code("""
    index = 0
    batch = tuple(t[index:index + 1].clone() for t in (
        dataset.states, dataset.actions, dataset.rewards, dataset.next_states, dataset.dones
    ))
    states, actions, rewards, next_states, dones = batch
    agent = BCQ(4, 2, phi=0.05, device="cpu")
    print("Dataset index:", index)
    for name, value in zip(("s", "a", "r", "s_next", "done"), batch):
        print(name, "=", value.numpy().round(5))
    with np.load("data/mixed.npz") as raw:
        print("timeout =", raw["timeouts"][index] if "timeouts" in raw else "not provided")
    """)
    markdown(r"""
    ## 3. VAE: encode → sample → decode → обновление

    $z=\mu+\sigma\epsilon$, $L_{VAE}=MSE(\hat a,a)+0.5 KL$.
    Одни и те же $\epsilon$ и latent probe используются до и после обновления,
    чтобы изменение реконструкции не объяснялось новой случайной выборкой.
    """)
    code("""
    epsilon = torch.randn(1, agent.vae.latent_dim)
    probe_z = torch.randn_like(epsilon).clamp(-0.5, 0.5)
    mean, log_std = agent.vae.encode(states, actions)
    std = log_std.exp()
    reconstructed = agent.vae.decode(states, mean + std * epsilon)
    reconstruction_loss = F.mse_loss(reconstructed, actions)
    kl = -0.5 * torch.mean(1 + torch.log(std.square() + 1e-8) - mean.square() - std.square())
    vae_loss = reconstruction_loss + 0.5 * kl
    reconstruction_before = reconstructed.detach().clone()
    with torch.no_grad():
        candidate_before = agent.vae.decode(states, probe_z)
    agent.vae_optimizer.zero_grad()
    vae_loss.backward()
    agent.vae_optimizer.step()
    with torch.no_grad():
        mean_after, log_std_after = agent.vae.encode(states, actions)
        reconstruction_after = agent.vae.decode(states, mean_after + log_std_after.exp() * epsilon)
        candidate = agent.vae.decode(states, probe_z)
    print("mu:", mean.detach().numpy().round(5), "std:", std.detach().numpy().round(5))
    print(f"reconstruction loss={reconstruction_loss.item():.6f}; KL={kl.item():.6f}; total={vae_loss.item():.6f}")
    print("Real action:", actions.numpy().round(5))
    print("Reconstruction before:", reconstruction_before.numpy().round(5))
    print("Reconstruction after: ", reconstruction_after.numpy().round(5))
    print("Fixed latent candidate before/after:", candidate_before.numpy(), candidate.numpy())
    """)
    markdown(r"""
    ## 4. Perturbation → Q1, Q2

    $\tilde a=clip(a_{VAE}+\Phi\tanh(\xi(s,a_{VAE})), -1,1)$.
    Коррекция каждой координаты ограничена Φ=0.05. Затем обе Q-сети оценивают действие.
    """)
    code("""
    with torch.no_grad():
        perturbed_before = agent.actor(states, candidate)
        candidate_q1, candidate_q2 = agent.critic(states, perturbed_before)
    print("VAE action:", candidate.numpy().round(5))
    print("Perturbed action:", perturbed_before.numpy().round(5))
    print("Correction:", (perturbed_before - candidate).numpy().round(5))
    print("Q1, Q2:", candidate_q1.item(), candidate_q2.item())
    """)
    markdown(r"""
    ## 5. Target y из десяти кандидатов следующего состояния

    $Q_{mix}=0.75\min(Q_1',Q_2')+0.25\max(Q_1',Q_2')$,
    $y=r+\gamma(1-done)\max_i Q_{mix}(s',a_i')$.
    Target рассчитывается без градиентов, после обновления VAE, как в `BCQ.train`.
    """)
    code("""
    with torch.no_grad():
        repeated_next = next_states.repeat_interleave(10, dim=0)
        next_candidates = agent.vae.decode(repeated_next)
        next_actions = agent.actor_target(repeated_next, next_candidates)
        target_q1, target_q2 = agent.critic_target(repeated_next, next_actions)
        mixed_q = 0.75 * torch.minimum(target_q1, target_q2) + 0.25 * torch.maximum(target_q1, target_q2)
        best_q = mixed_q.view(1, 10).max(dim=1, keepdim=True).values
        target_y = rewards + (1 - dones) * agent.gamma * best_q
    print("i      VAE a                   perturbed a               Q1        Q2       Qmix")
    for i in range(10):
        print(f"{i}  {next_candidates[i].numpy().round(4)}  {next_actions[i].numpy().round(4)}  "
              f"{target_q1[i].item():9.5f} {target_q2[i].item():9.5f} {mixed_q[i].item():9.5f}")
    print(f"y = {rewards.item():.6f} + {agent.gamma} * (1 - {dones.item():.0f}) * {best_q.item():.6f} = {target_y.item():.6f}")
    """)
    markdown(r"""
    ## 6. Обновление критика: до / после

    Минимизируем $(Q_1(s,a)-y)^2+(Q_2(s,a)-y)^2$ на **действии датасета**.
    После шага сравниваем с тем же фиксированным y.
    """)
    code("""
    current_q1, current_q2 = agent.critic(states, actions)
    critic_loss = F.mse_loss(current_q1, target_y) + F.mse_loss(current_q2, target_y)
    q_before = [current_q1.item(), current_q2.item()]
    agent.critic_optimizer.zero_grad()
    critic_loss.backward()
    agent.critic_optimizer.step()
    with torch.no_grad():
        after_q1, after_q2 = agent.critic(states, actions)
        critic_loss_after = F.mse_loss(after_q1, target_y) + F.mse_loss(after_q2, target_y)
    q_after = [after_q1.item(), after_q2.item()]
    print("Target y:", target_y.item())
    print("Q1, Q2 before:", q_before)
    print("Q1, Q2 after: ", q_after)
    print("Critic loss before / after:", critic_loss.item(), critic_loss_after.item())
    """)
    markdown(r"""
    ## 7. Обновление актёра и target-сетей: до / после

    $L_{actor}=-Q_1(s,\xi(s,a_{VAE}))$. VAE-кандидат отделён от графа.
    Сравниваем до/после на одном кандидате и уже обновлённом критике.
    Один шаг Adam не гарантирует улучшение каждой отдельной оценки.
    """)
    code("""
    fixed_candidate = candidate.detach()
    before_action = agent.actor(states, fixed_candidate)
    actor_q_before = agent.critic.q1(states, before_action)
    actor_loss = -actor_q_before.mean()
    agent.actor_optimizer.zero_grad()
    actor_loss.backward()
    agent.actor_optimizer.step()
    with torch.no_grad():
        after_action = agent.actor(states, fixed_candidate)
        actor_q_after = agent.critic.q1(states, after_action)
    agent._soft_update(agent.critic, agent.critic_target)
    agent._soft_update(agent.actor, agent.actor_target)
    print("Action before:", before_action.detach().numpy().round(6))
    print("Action after: ", after_action.numpy().round(6))
    print("Q1 before / after actor step:", actor_q_before.item(), actor_q_after.item())
    print("Actor loss before / after:", actor_loss.item(), -actor_q_after.item())
    print("Soft update tau:", agent.tau)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    x = np.arange(2)
    axes[0].bar(x - 0.18, q_before, 0.36, label="Before")
    axes[0].bar(x + 0.18, q_after, 0.36, label="After")
    axes[0].axhline(target_y.item(), color="black", linestyle="--", label="Target y")
    axes[0].set(xticks=x, xticklabels=["Q1(s,a)", "Q2(s,a)"], title="Critic update")
    axes[0].legend()
    axes[1].bar(["Before", "After"], [actor_q_before.item(), actor_q_after.item()])
    axes[1].set(title="Actor update: Q1 at fixed VAE candidate")
    plt.tight_layout()
    plt.show()
    """)
    markdown("""
    ## 8. Короткий пример общего train.py

    Ниже 100 шагов BC на mixed — проверка интерфейса, не обучение до сходимости.
    Файл сохраняется отдельно от основной серии. Для полного эксперимента:

    ```bash
    python run_experiments.py --suite all --steps 100000
    python make_report.py
    ```

    Это 36 основных запусков и ещё 15 уникальных запусков ablation: Φ=0.05 на mixed
    переиспользуется. Обобщение на новых стартах и целях оценивается после каждого запуска.
    """)
    code("""
    args = build_parser().parse_args([
        "--algo", "bc", "--dataset", "data/mixed.npz", "--seed", "0", "--steps", "100",
        "--eval-episodes", "2", "--probe-size", "32",
        "--results-dir", "results/notebook/runs", "--checkpoint-dir", "results/notebook/checkpoints"
    ])
    demo_run_path = train(args)
    demo_run = json.loads(demo_run_path.read_text(encoding="utf-8"))
    print("Saved:", demo_run_path)
    print(json.dumps(demo_run["history"][-1], indent=2))
    print("BC has no critic: q_dataset and q_start are null.")
    """)
    markdown("""
    ## 9. Таблицы и графики серии

    Показываем полный отчёт, если он существует; иначе — явно обозначенный отчёт
    проверочной серии. `mean ± std` считается между seeds, а не между эпизодами.
    Q на состояниях датасета и return из старта имеют разные распределения состояний:
    непосредственно сравниваем `Q(start)` с `G0`.
    """)
    code("""
    from make_report import make_report

    main_runs = Path("results/runs")
    smoke_runs = Path("results/smoke_suite/runs")
    completed = lambda directory: any(
        json.loads(p.read_text(encoding="utf-8")).get("status") == "completed"
        for p in directory.glob("*.json")
    ) if directory.exists() else False
    if completed(main_runs):
        report_dir = make_report(main_runs, Path("results/report"))
        print("Main experiment report")
    elif completed(smoke_runs):
        report_dir = make_report(smoke_runs, Path("results/smoke_suite/report"))
        print("SMOKE TEST ONLY: 100 training steps, 2 evaluation episodes; not a scientific result")
    else:
        report_dir = None
        print("Run run_experiments.py to obtain result tables and learning curves.")
    if report_dir is not None:
        display(Markdown((report_dir / "results_table.md").read_text(encoding="utf-8")))
        display(Markdown((report_dir / "generalization_table.md").read_text(encoding="utf-8")))
        for name in ("success_curves.png", "q_curves.png", "phi_ablation.png", "trajectories.png"):
            path = report_dir / name
            if path.exists():
                display(Image(filename=str(path)))
    """)
    markdown("""
    ## 10. Анимация сохранённой BCQ-политики

    Используем существующий исторический checkpoint для иллюстрации движения.
    Он не входит в новую таблицу BC/TD3/BCQ. Если его нет, коротко обучаем
    демонстрационную модель и явно указываем это в выводе.
    """)
    code("""
    from demo import rollout

    visual_agent = make_agent("bcq", device="cpu")
    if MODEL_PATH.exists():
        visual_agent.load(MODEL_PATH)
        print("Historical demonstration checkpoint:", MODEL_PATH)
    else:
        print("No historical checkpoint; training 100 demonstration updates")
        for _ in range(100):
            visual_agent.train(dataset.sample(256))
    positions, total_reward, info = rollout(visual_agent, seed=4042)
    print("Reward:", round(total_reward, 3), "Success:", info["success"], "Collision:", info["collision"])

    fig, ax = plt.subplots(figsize=(5, 5))
    for x0, y0, x1, y1 in OBSTACLES:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color="gray", alpha=0.6))
    ax.scatter(9, 9, marker="*", s=180, color="gold", edgecolor="black")
    ax.set(xlim=(0, 10), ylim=(0, 10), aspect="equal", title="BCQ demonstration")
    line, = ax.plot([], [], color="tab:blue")
    marker, = ax.plot([], [], "o", color="tab:red")

    def update(frame):
        '''Draw the observed trajectory up to the current time step.'''
        line.set_data(positions[:frame + 1, 0], positions[:frame + 1, 1])
        marker.set_data([positions[frame, 0]], [positions[frame, 1]])
        return line, marker

    animation = FuncAnimation(fig, update, frames=len(positions), interval=100, blit=True)
    html = animation.to_jshtml()
    plt.close(fig)
    display(HTML(html))
    """)
    markdown("""
    ## Интерпретация

    Рост Q сам по себе не означает улучшение политики: проверяйте success, collision
    и измеренный дисконтированный return. Возврат ограничен горизонтом 120 шагов;
    Bellman targets продолжают bootstrap на timeouts датасета, поэтому для застрявших
    политик это дополнительный источник расхождения. Обобщение здесь означает новые
    пары старт/цель в окрестностях исходных областей, не новые карты.
    """)
    return nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
        },
    )


def main():
    """Write the notebook, optionally execute every cell and retain its outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    notebook = build_notebook()
    if args.execute:
        from nbclient import NotebookClient

        NotebookClient(
            notebook,
            timeout=600,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        ).execute()
    destination = ROOT / "BCQ_Offline_Navigation_Demo.ipynb"
    nbf.write(notebook, destination)
    print(f"Saved {destination}; executed={args.execute}")


if __name__ == "__main__":
    main()
