import numpy as np

from config import (
    DATASET_PATH,
    DATASET_EPISODES,
    BEHAVIOR_NOISE_STD,
    SEED,
    create_project_dirs,
)

from env import NavigationEnv


def get_route(rng):
    # Use two possible routes around the obstacle
    if rng.random() < 0.7:
        return [
            np.array([3.7, 1.7], dtype=np.float32),
            np.array([6.3, 1.7], dtype=np.float32),
            np.array([9.0, 9.0], dtype=np.float32),
        ]

    return [
        np.array([3.7, 8.3], dtype=np.float32),
        np.array([6.3, 8.3], dtype=np.float32),
        np.array([9.0, 9.0], dtype=np.float32),
    ]


def behavior_policy(position, waypoints, waypoint_index, rng):
    # Move to the next waypoint
    target = waypoints[waypoint_index]

    distance = np.linalg.norm(target - position)

    if distance < 0.45 and waypoint_index < len(waypoints) - 1:
        waypoint_index += 1
        target = waypoints[waypoint_index]

    direction = target - position
    norm = np.linalg.norm(direction)

    if norm > 1e-8:
        direction = direction / norm
    else:
        direction = np.zeros(2, dtype=np.float32)

    # Add exploration noise
    noise = rng.normal(
        loc=0.0,
        scale=BEHAVIOR_NOISE_STD,
        size=2,
    )

    action = direction + noise

    # Occasionally use a completely random action
    if rng.random() < 0.05:
        action = rng.uniform(-1.0, 1.0, size=2)

    action = np.clip(
        action,
        -1.0,
        1.0,
    )

    return action.astype(np.float32), waypoint_index


def collect_dataset():
    create_project_dirs()

    rng = np.random.default_rng(SEED)

    env = NavigationEnv(seed=SEED)

    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []

    successful_episodes = 0
    collision_episodes = 0
    timeout_episodes = 0

    total_reward = 0.0

    for episode in range(DATASET_EPISODES):
        state = env.reset()

        waypoints = get_route(rng)
        waypoint_index = 0

        episode_reward = 0.0

        while True:
            action, waypoint_index = behavior_policy(
                position=env.position.copy(),
                waypoints=waypoints,
                waypoint_index=waypoint_index,
                rng=rng,
            )

            next_state, reward, done, info = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(float(done))

            state = next_state
            episode_reward += reward

            if done:
                if info["success"]:
                    successful_episodes += 1

                elif info["collision"]:
                    collision_episodes += 1

                else:
                    timeout_episodes += 1

                break

        total_reward += episode_reward

        if (episode + 1) % 100 == 0:
            print(
                f"Episode {episode + 1}/{DATASET_EPISODES} | "
                f"Transitions: {len(states)} | "
                f"Success: {successful_episodes} | "
                f"Collision: {collision_episodes} | "
                f"Timeout: {timeout_episodes}"
            )

    states = np.asarray(states, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1, 1)
    next_states = np.asarray(next_states, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32).reshape(-1, 1)

    np.savez_compressed(
        DATASET_PATH,
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
    )

    print()
    print("Dataset collection finished.")
    print(f"Saved to: {DATASET_PATH}")
    print(f"Transitions: {len(states)}")
    print(f"States shape: {states.shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"Rewards shape: {rewards.shape}")
    print(f"Next states shape: {next_states.shape}")
    print(f"Dones shape: {dones.shape}")
    print()
    print(f"Successful episodes: {successful_episodes}")
    print(f"Collision episodes: {collision_episodes}")
    print(f"Timeout episodes: {timeout_episodes}")
    print(
        f"Success rate: "
        f"{successful_episodes / DATASET_EPISODES * 100:.2f}%"
    )
    print(
        f"Average reward: "
        f"{total_reward / DATASET_EPISODES:.3f}"
    )


if __name__ == "__main__":
    collect_dataset()