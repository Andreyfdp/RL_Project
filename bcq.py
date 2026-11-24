import copy

import numpy as np
import torch
import torch.nn.functional as F

from config import GAMMA, LEARNING_RATE, NUM_ACTION_SAMPLES, PHI, TAU
from models import VAE, Critic, PerturbationActor


class BCQ:
    def __init__(self, state_dim, action_dim, max_action=1.0, device="cpu", phi=PHI):
        """Create BCQ with an explicit perturbation bound for ablations."""
        self.device = torch.device(device)
        self.max_action = max_action

        self.vae = VAE(state_dim, action_dim, max_action).to(self.device)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor = PerturbationActor(state_dim, action_dim, max_action, phi).to(
            self.device
        )
        self.actor_target = copy.deepcopy(self.actor)

        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=LEARNING_RATE)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=LEARNING_RATE
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=LEARNING_RATE
        )

        self.gamma = GAMMA
        self.tau = TAU

    def train(self, batch):
        """Update the VAE, twin critic, perturbation actor and target networks."""
        states, actions, rewards, next_states, dones = batch

        # Train VAE
        reconstructed_actions, mean, std = self.vae(states, actions)

        reconstruction_loss = F.mse_loss(reconstructed_actions, actions)

        kl_loss = -0.5 * torch.mean(
            1 + torch.log(std.pow(2) + 1e-8) - mean.pow(2) - std.pow(2)
        )

        vae_loss = reconstruction_loss + 0.5 * kl_loss

        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()

        # Create target Q-values
        with torch.no_grad():
            batch_size = next_states.shape[0]
            num_samples = 10

            repeated_next_states = next_states.repeat_interleave(num_samples, dim=0)

            sampled_actions = self.vae.decode(repeated_next_states)
            sampled_actions = self.actor_target(repeated_next_states, sampled_actions)

            target_q1, target_q2 = self.critic_target(
                repeated_next_states, sampled_actions
            )

            mixed_q = 0.75 * torch.min(target_q1, target_q2) + 0.25 * torch.max(
                target_q1, target_q2
            )

            mixed_q = mixed_q.view(batch_size, num_samples)
            best_q = mixed_q.max(dim=1, keepdim=True).values

            target_q = rewards + (1.0 - dones) * self.gamma * best_q

        # Train critic
        current_q1, current_q2 = self.critic(states, actions)

        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(
            current_q2, target_q
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Train perturbation actor
        sampled_actions = self.vae.decode(states).detach()
        perturbed_actions = self.actor(states, sampled_actions)

        actor_loss = -self.critic.q1(states, perturbed_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        self._soft_update(self.critic, self.critic_target)
        self._soft_update(self.actor, self.actor_target)

        return {
            "vae_loss": vae_loss.item(),
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
        }

    @torch.no_grad()
    def policy_actions(self, states):
        """Select the best VAE candidate independently for every state."""
        repeated = states.repeat_interleave(NUM_ACTION_SAMPLES, dim=0)
        actions = self.actor(repeated, self.vae.decode(repeated))
        scores = self.critic.q1(repeated, actions).view(-1, NUM_ACTION_SAMPLES)
        candidates = actions.view(len(states), NUM_ACTION_SAMPLES, -1)
        return candidates[
            torch.arange(len(states), device=self.device), scores.argmax(1)
        ]

    def select_action(self, state):
        """Return one environment action using the same policy as Q diagnostics."""
        states = torch.as_tensor(
            np.asarray(state), dtype=torch.float32, device=self.device
        )
        return self.policy_actions(states.unsqueeze(0))[0].cpu().numpy()

    def _soft_update(self, source, target):
        """Move target parameters toward online parameters by tau."""
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )

    def save(self, path):
        """Save networks and phi so ablation checkpoints retain their policy."""
        torch.save(
            {
                "vae": self.vae.state_dict(),
                "critic": self.critic.state_dict(),
                "actor": self.actor.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "phi": self.actor.phi,
            },
            path,
        )

    def load(self, path):
        """Restore networks; accept historical checkpoints without phi metadata."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.phi = checkpoint.get("phi", self.actor.phi)
        self.actor_target.phi = self.actor.phi

        self.vae.load_state_dict(checkpoint["vae"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor.load_state_dict(checkpoint["actor"])

        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    agent = BCQ(
        state_dim=4,
        action_dim=2,
        max_action=1.0,
        device=device,
    )

    batch_size = 256

    batch = (
        torch.randn(batch_size, 4, device=device),
        torch.rand(batch_size, 2, device=device) * 2.0 - 1.0,
        torch.randn(batch_size, 1, device=device),
        torch.randn(batch_size, 4, device=device),
        torch.zeros(batch_size, 1, device=device),
    )

    losses = agent.train(batch)

    test_state = np.array([1.0, 1.0, 8.0, 8.0], dtype=np.float32)
    action = agent.select_action(test_state)

    print("BCQ initialized successfully.")
    print("Device:", device)
    print("VAE loss:", losses["vae_loss"])
    print("Critic loss:", losses["critic_loss"])
    print("Actor loss:", losses["actor_loss"])
    print("Selected action:", action)
