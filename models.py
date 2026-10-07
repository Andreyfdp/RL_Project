import torch
import torch.nn as nn
import torch.nn.functional as F

from config import LATENT_DIM, PHI


class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, max_action=1.0, latent_dim=LATENT_DIM):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.latent_dim = latent_dim

        # Encoder
        self.encoder_fc1 = nn.Linear(state_dim + action_dim, 256)
        self.encoder_fc2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, latent_dim)
        self.log_std = nn.Linear(256, latent_dim)

        # Decoder
        self.decoder_fc1 = nn.Linear(state_dim + latent_dim, 256)
        self.decoder_fc2 = nn.Linear(256, 256)
        self.decoder_output = nn.Linear(256, action_dim)

    def encode(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.encoder_fc1(x))
        x = F.relu(self.encoder_fc2(x))

        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), -4.0, 15.0)

        return mean, log_std

    def decode(self, state, z=None):
        # Sample latent vector if not provided
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim), device=state.device)
            z = torch.clamp(z, -0.5, 0.5)

        x = torch.cat([state, z], dim=1)
        x = F.relu(self.decoder_fc1(x))
        x = F.relu(self.decoder_fc2(x))

        return self.max_action * torch.tanh(self.decoder_output(x))

    def forward(self, state, action):
        mean, log_std = self.encode(state, action)
        std = torch.exp(log_std)

        z = mean + std * torch.randn_like(std)
        reconstructed_action = self.decode(state, z)

        return reconstructed_action, mean, std


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        # First Q-network
        self.q1_fc1 = nn.Linear(state_dim + action_dim, 256)
        self.q1_fc2 = nn.Linear(256, 256)
        self.q1_output = nn.Linear(256, 1)

        # Second Q-network
        self.q2_fc1 = nn.Linear(state_dim + action_dim, 256)
        self.q2_fc2 = nn.Linear(256, 256)
        self.q2_output = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)

        q1 = F.relu(self.q1_fc1(x))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_output(q1)

        q2 = F.relu(self.q2_fc1(x))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_output(q2)

        return q1, q2

    def q1(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.q1_fc1(x))
        x = F.relu(self.q1_fc2(x))

        return self.q1_output(x)


class PerturbationActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action=1.0, phi=PHI):
        super().__init__()

        self.max_action = max_action
        self.phi = phi

        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.output = nn.Linear(256, action_dim)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        perturbation = self.phi * self.max_action * torch.tanh(self.output(x))
        modified_action = action + perturbation

        return torch.clamp(modified_action, -self.max_action, self.max_action)


if __name__ == "__main__":
    state_dim = 4
    action_dim = 2
    batch_size = 256

    states = torch.randn(batch_size, state_dim)
    actions = torch.randn(batch_size, action_dim).clamp(-1.0, 1.0)

    vae = VAE(state_dim, action_dim)
    critic = Critic(state_dim, action_dim)
    actor = PerturbationActor(state_dim, action_dim)

    reconstructed_actions, mean, std = vae(states, actions)
    generated_actions = vae.decode(states)
    perturbed_actions = actor(states, generated_actions)
    q1, q2 = critic(states, perturbed_actions)

    print("Models initialized successfully.")
    print("Reconstructed actions:", reconstructed_actions.shape)
    print("Latent mean:", mean.shape)
    print("Latent std:", std.shape)
    print("Generated actions:", generated_actions.shape)
    print("Perturbed actions:", perturbed_actions.shape)
    print("Q1:", q1.shape)
    print("Q2:", q2.shape)
    print("Generated action range:", generated_actions.min().item(), generated_actions.max().item())
    print("Perturbed action range:", perturbed_actions.min().item(), perturbed_actions.max().item())