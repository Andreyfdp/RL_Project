# Sanity checks for the experiment datasets and the start noise.

import numpy as np

from config import DATA_DIR, DATASET_MODES, START_NOISE, START_POSITION, STEP_SIZE
from dataset import OfflineDataset
from env import NavigationEnv


def check_env_start_noise():
    # Starts are fixed when noise is disabled.
    env = NavigationEnv(seed=0)
    starts = np.array([env.reset()[:2] for _ in range(20)])
    assert np.allclose(starts, START_POSITION), "start without noise must be fixed"

    # Noisy starts must stay within the configured bounds.
    env = NavigationEnv(seed=0, start_noise=START_NOISE)
    starts = np.array([env.reset()[:2] for _ in range(500)])
    shift = np.abs(starts - np.asarray(START_POSITION))
    assert shift.max() <= START_NOISE + 1e-6, "start went outside the noise square"
    assert len(np.unique(starts, axis=0)) > 400, "starts are not random enough"

    # Same seed -> same starts, so datasets are reproducible
    a = NavigationEnv(seed=5, start_noise=START_NOISE)
    b = NavigationEnv(seed=5, start_noise=START_NOISE)
    assert all(np.allclose(a.reset(), b.reset()) for _ in range(20)), (
        "start noise is not reproducible"
    )
    print("OK  env start noise")


def check_dataset(mode):
    path = DATA_DIR / f"{mode}.npz"
    data = np.load(path)
    n = len(data["states"])

    # Shapes are the ones the training code is expecting
    assert data["states"].shape == (n, 4) and data["next_states"].shape == (n, 4)
    assert data["actions"].shape == (n, 2)
    for key in ("rewards", "dones", "timeouts"):
        assert data[key].shape == (n, 1), f"{key} has a wrong shape"

    # No NaNs and actions are inside [-1, 1]
    for key in ("states", "actions", "rewards", "next_states"):
        assert np.isfinite(data[key]).all(), f"{key} has NaN/inf"
    assert np.abs(data["actions"]).max() <= 1.0 + 1e-6, "action outside [-1, 1]"

    # Adjacent transitions must share the same intermediate state.
    same_episode = data["episode_ids"][1:] == data["episode_ids"][:-1]
    assert np.allclose(
        data["next_states"][:-1][same_episode], data["states"][1:][same_episode]
    ), "transitions are not chained correctly"

    # Only the final transition ends each episode.
    last_step = np.append(~same_episode, True)
    ended = (data["dones"][:, 0] + data["timeouts"][:, 0]) > 0
    assert np.array_equal(ended, last_step), "episode ends are wrong"
    assert not np.any((data["dones"][:, 0] > 0) & (data["timeouts"][:, 0] > 0)), (
        "done and timeout at once"
    )

    # Movement cannot exceed the action-scaled step size.
    moved = np.linalg.norm(data["next_states"][:, :2] - data["states"][:, :2], axis=1)
    assert moved.max() <= STEP_SIZE * np.sqrt(2) + 1e-5, "agent jumped too far"

    # Episode starts must stay inside the start region.
    first_step = np.insert(~same_episode, 0, True)
    starts = data["states"][first_step, :2]
    assert np.abs(starts - np.asarray(START_POSITION)).max() <= START_NOISE + 1e-6

    # Number of episodes is matching the config
    episodes = len(np.unique(data["episode_ids"]))
    assert episodes == DATASET_MODES[mode]["episodes"], "wrong number of episodes"

    # Check the training batch shapes.
    batch = OfflineDataset(path).sample(64)
    assert [tuple(x.shape) for x in batch] == [
        (64, 4),
        (64, 2),
        (64, 1),
        (64, 4),
        (64, 1),
    ]

    print(f"OK  {mode:6s} ({episodes} episodes, {n} transitions)")


if __name__ == "__main__":
    check_env_start_noise()
    for mode in DATASET_MODES:
        check_dataset(mode)
    print("\nAll checks passed")
