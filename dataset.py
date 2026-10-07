import numpy as np
import torch


class OfflineDataset:
    def __init__(self, path, device="cpu"):
        # Load dataset from disk
        data = np.load(path)

        self.states = torch.tensor(
            data["states"],
            dtype=torch.float32,
            device=device,
        )

        self.actions = torch.tensor(
            data["actions"],
            dtype=torch.float32,
            device=device,
        )

        self.rewards = torch.tensor(
            data["rewards"],
            dtype=torch.float32,
            device=device,
        )

        self.next_states = torch.tensor(
            data["next_states"],
            dtype=torch.float32,
            device=device,
        )

        self.dones = torch.tensor(
            data["dones"],
            dtype=torch.float32,
            device=device,
        )

        self.device = device
        self.size = self.states.shape[0]

    def __len__(self):
        return self.size

    def sample(self, batch_size):
        # Sample random transition indices
        indices = torch.randint(
            low=0,
            high=self.size,
            size=(batch_size,),
            device=self.device,
        )

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )


if __name__ == "__main__":
    from config import DATASET_PATH, BATCH_SIZE

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    dataset = OfflineDataset(
        path=DATASET_PATH,
        device=device,
    )

    print("Dataset loaded successfully.")
    print(f"Device: {device}")
    print(f"Transitions: {len(dataset)}")
    print()

    states, actions, rewards, next_states, dones = dataset.sample(
        BATCH_SIZE
    )

    print("Batch shapes:")
    print("States:", states.shape)
    print("Actions:", actions.shape)
    print("Rewards:", rewards.shape)
    print("Next states:", next_states.shape)
    print("Dones:", dones.shape)

    print()
    print("Example transition:")
    print("State:", states[0])
    print("Action:", actions[0])
    print("Reward:", rewards[0])
    print("Next state:", next_states[0])
    print("Done:", dones[0])