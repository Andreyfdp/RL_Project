"""Behavior cloning and unconstrained offline TD3 on the shared BCQ networks."""

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from config import GAMMA, LEARNING_RATE, TAU
from models import Critic


class Actor(nn.Module):
    """Two 256-unit hidden layers, matching BCQ's actor capacity and bounds."""

    def __init__(self, state_dim, action_dim, max_action=1.0):
        super().__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, states):
        """Map observations to bounded continuous actions."""
        return self.max_action * torch.tanh(self.net(states))


class BC:
    """Supervised action regression; no reward optimization or learned critic."""

    def __init__(self, state_dim, action_dim, max_action=1.0, device="cpu"):
        self.device = torch.device(device)
        self.max_action = max_action
        self.gamma = GAMMA
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=LEARNING_RATE
        )

    def train(self, batch):
        """Fit dataset actions with the same optimizer and LR as BCQ."""
        states, actions, *_ = batch
        loss = F.mse_loss(self.actor(states), actions)
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        return {"actor_loss": loss.item()}

    @torch.no_grad()
    def policy_actions(self, states):
        """Return a batch of policy actions for evaluation."""
        return self.actor(states)

    def select_action(self, state):
        """Return a NumPy action compatible with NavigationEnv."""
        states = torch.as_tensor(
            np.asarray(state), dtype=torch.float32, device=self.device
        )
        return self.policy_actions(states.unsqueeze(0))[0].cpu().numpy()

    def save(self, path):
        """Save inference weights; experiment metadata lives in the run JSON."""
        torch.save({"actor": self.actor.state_dict()}, path)

    def load(self, path):
        """Load inference weights on the selected device."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])


class TD3(BC):
    """TD3 trained only on offline batches, without a behavior constraint."""

    def __init__(self, state_dim, action_dim, max_action=1.0, device="cpu"):
        super().__init__(state_dim, action_dim, max_action, device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=LEARNING_RATE
        )
        self.tau = TAU
        self.policy_noise = 0.2 * max_action
        self.noise_clip = 0.5 * max_action
        self.policy_delay = 2
        self.updates = 0

    def train(self, batch):
        """Use clipped double Q, target smoothing, and delayed actor updates."""
        states, actions, rewards, next_states, dones = batch
        self.updates += 1
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions = (self.actor_target(next_states) + noise).clamp(
                -self.max_action, self.max_action
            )
            q1, q2 = self.critic_target(next_states, next_actions)
            # Dataset dones contain true terminals; timeouts still bootstrap.
            target = rewards + (1 - dones) * self.gamma * torch.minimum(q1, q2)
        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        losses = {"critic_loss": critic_loss.item()}
        if self.updates % self.policy_delay == 0:
            actor_loss = -self.critic.q1(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            losses["actor_loss"] = actor_loss.item()
            with torch.no_grad():
                for source, target_net in (
                    (self.actor, self.actor_target),
                    (self.critic, self.critic_target),
                ):
                    for param, target_param in zip(
                        source.parameters(), target_net.parameters()
                    ):
                        target_param.lerp_(param, self.tau)
        return losses

    def save(self, path):
        """Save policy and both online/target critics for diagnostics."""
        torch.save(
            {
                name: getattr(self, name).state_dict()
                for name in ("actor", "actor_target", "critic", "critic_target")
            },
            path,
        )

    def load(self, path):
        """Restore inference and target networks."""
        checkpoint = torch.load(path, map_location=self.device)
        for name in ("actor", "actor_target", "critic", "critic_target"):
            getattr(self, name).load_state_dict(checkpoint[name])
