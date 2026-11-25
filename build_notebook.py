# Build and optionally execute the presentation notebook with real outputs.

import argparse
import textwrap
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent


def build_notebook():
    # Create a self-contained walkthrough of the actual BCQ update equations.
    cells = []

    def markdown(source):
        # Append a readable explanation to the notebook.
        cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(source).strip()))

    def code(source):
        # Append executable code without incidental builder indentation.
        cells.append(nbf.v4.new_code_cell(textwrap.dedent(source).strip()))

    markdown(
        r"""
        # Offline navigation: BC, TD3 and BCQ

        This notebook follows one real transition from `data/mixed.npz` through a BCQ update.
        The saved outputs were computed on CPU.
        """
    )
    markdown(
        r"""
        ## 1. Google Colab setup

        Run the next cell in Colab to clone the project and install dependencies.
        For local execution, start Jupyter in the project directory.
        """
    )
    code("""
    from importlib.util import find_spec
    import os
    from pathlib import Path
    import subprocess
    import sys

    if find_spec("google") is not None and find_spec("google.colab") is not None:
        project = Path("/content/RL_Project")
        if not project.exists():
            subprocess.run([
                "git", "clone", "https://github.com/Andreyfdp/RL_Project.git", str(project)
            ], check=True)
        os.chdir(project)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
    """)
    code("""
    %matplotlib inline
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    from IPython.display import Image, Markdown, display

    from bcq import BCQ
    from dataset import OfflineDataset
    from train import set_seed

    torch.set_num_threads(1)
    set_seed(0)
    dataset = OfflineDataset("data/mixed.npz", device="cpu")
    print("Transitions:", len(dataset))
    print("PyTorch:", torch.__version__, "CUDA:", torch.cuda.is_available())
    """)
    markdown(
        r"""
        ## 2. One real transition

        The state is `[x, y, goal_x - x, goal_y - y]`.
        `done` marks a goal or collision. Time limits are stored separately in `timeouts`
        and keep the Bellman bootstrap. This example updates one transition; training uses batch=256.
        """
    )
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
    markdown(
        r"""
        ## 3. VAE: encode, sample, decode and update

        $z=\mu+\sigma\epsilon$, $L_{VAE}=MSE(\hat a,a)+0.5 KL$.
        Use the same $\epsilon$ and latent probe before and after the update to isolate
        parameter changes from sampling noise.
        """
    )
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
    markdown(
        r"""
        ## 4. Perturbation and twin Q estimates

        $\tilde a=clip(a_{VAE}+\Phi\tanh(\xi(s,a_{VAE})), -1,1)$.
        Each action correction is bounded by Phi=0.05. Both critics then score the action.
        """
    )
    code("""
    with torch.no_grad():
        perturbed_before = agent.actor(states, candidate)
        candidate_q1, candidate_q2 = agent.critic(states, perturbed_before)
    print("VAE action:", candidate.numpy().round(5))
    print("Perturbed action:", perturbed_before.numpy().round(5))
    print("Correction:", (perturbed_before - candidate).numpy().round(5))
    print("Q1, Q2:", candidate_q1.item(), candidate_q2.item())
    """)
    markdown(
        r"""
        ## 5. Target y from ten next-state candidates

        $Q_{mix}=0.75\min(Q_1',Q_2')+0.25\max(Q_1',Q_2')$,
        $y=r+\gamma(1-done)\max_i Q_{mix}(s',a_i')$.
        Compute the target without gradients after the VAE update, as in `BCQ.train`.
        """
    )
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
    markdown(
        r"""
        ## 6. Critic update: before and after

        Minimize $(Q_1(s,a)-y)^2+(Q_2(s,a)-y)^2$ on the **dataset action**.
        Compare both critics against the same fixed target after the update.
        """
    )
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
    markdown(
        r"""
        ## 7. Actor and target updates: before and after

        $L_{actor}=-Q_1(s,\xi(s,a_{VAE}))$. Detach the VAE candidate from its computation graph.
        Compare the same candidate before and after the actor step using the updated critic.
        One Adam step does not guarantee that every individual Q estimate improves.
        """
    )
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
    markdown(
        r"""
        ## 8. Benchmark results

        The benchmark runs BC, offline TD3 and BCQ on four datasets with training seeds 0–2,
        plus six BCQ perturbation limits on the mixed dataset. Each run uses 2,000 updates.
        The checkpoint with the highest validation success rate is selected, using mean reward
        to break ties. Each selected policy is tested on 100 episodes per task distribution.
        Tables report the mean and sample standard deviation across training seeds.
        """
    )
    code("""
    report = Path("results/benchmark/report")
    if not (report / "results_table.md").is_file():
        raise FileNotFoundError("Run the benchmark and build its report using the README commands.")

    for filename in ("results_table.md", "generalization_table.md", "phi_table.md"):
        display(Markdown((report / filename).read_text(encoding="utf-8")))
    for filename in ("success_curves.png", "q_curves.png", "phi_ablation.png", "trajectories.png"):
        display(Image(filename=str(report / filename)))
    """)
    markdown(
        r"""
        ## Interpretation

        Increasing Q alone does not establish policy improvement. Check success, collisions
        and measured discounted return. Returns stop at the 120-step horizon, while dataset
        timeouts retain Bellman bootstrapping. This can also cause Q/return differences.
        Generalization uses new start/goal pairs near the original regions, not new maps.
        """
    )
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
    # Write the notebook, optionally execute every cell and retain its outputs.
    parser = argparse.ArgumentParser(
        description="Build and optionally execute the presentation notebook with real outputs."
    )
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
