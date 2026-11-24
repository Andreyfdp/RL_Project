import numpy as np

from config import (
    MAP_WIDTH,
    MAP_HEIGHT,
    START_POSITION,
    GOAL_POSITION,
    OBSTACLES,
    STEP_SIZE,
    AGENT_RADIUS,
    GOAL_RADIUS,
    MAX_EPISODE_STEPS,
    PROGRESS_REWARD_SCALE,
    STEP_PENALTY,
    GOAL_REWARD,
    COLLISION_PENALTY,
    SEED,
)


class NavigationEnv:
    """Core navigation environment with a lightweight, dependency-free API.

    ``GymnasiumNavigationEnv`` in ``gym_env.py`` adapts this class to the
    standard Gymnasium API.  Keeping this core class free of Gymnasium makes
    dataset creation and the existing training code easy to run as scripts.
    """

    def __init__(self, seed=SEED, start_position=None, goal_position=None, start_noise=0.0):
        self.rng = np.random.default_rng(seed)

        # Here we are keeping the start noise: if it is bigger than zero, then on every
        # reset the agent is starting not exactly at the start point, but somewhere in a
        # small square around it, so basically episodes are not all the same anymore.
        # By default it is zero, so all the old scripts and the old checkpoint are working
        # exactly like before
        if start_noise < 0:
            raise ValueError(f"start_noise must be >= 0, got {start_noise}")
        self.start_noise = float(start_noise)

        self.start_position = self._as_valid_position(
            START_POSITION if start_position is None else start_position,
            "start_position",
        )
        self.goal_position = self._as_valid_position(
            GOAL_POSITION if goal_position is None else goal_position,
            "goal_position",
        )

        self.position = self.start_position.copy()
        self.steps = 0
        self.done = False

    @property
    def state_dim(self):
        return 4

    @property
    def action_dim(self):
        return 2

    @property
    def max_action(self):
        return 1.0

    def _as_valid_position(self, position, name):
        position = np.asarray(position, dtype=np.float32)

        if position.shape != (2,):
            raise ValueError(f"{name} must have shape (2,), got {position.shape}")

        if self._is_outside_map(position) or self._collides_with_obstacle(position):
            raise ValueError(f"{name} must be inside the free space, got {position}")

        return position.copy()

    def set_task(self, start_position, goal_position):
        """Set a valid start/goal pair for the next reset.

        This is useful for robustness experiments.  It deliberately does not
        randomize the default task, so existing dataset and checkpoint results
        remain exactly reproducible.
        """
        self.start_position = self._as_valid_position(start_position, "start_position")
        self.goal_position = self._as_valid_position(goal_position, "goal_position")

    def reset(self, start_position=None, goal_position=None):
        if (start_position is None) != (goal_position is None):
            raise ValueError("Provide both start_position and goal_position, or neither.")

        if start_position is not None:
            self.set_task(start_position, goal_position)

        self.position = self._sample_start()
        self.steps = 0
        self.done = False

        return self._get_state()

    def _sample_start(self):
        # Without noise we are just returning the usual fixed start, nothing special here
        if self.start_noise == 0.0:
            return self.start_position.copy()

        # Otherwise taking random shifts in [-noise, noise] for x and y and checking that the
        # agent is not starting inside a wall or an obstacle (if it is, we are simply trying again)
        for _ in range(100):
            shift = self.rng.uniform(-self.start_noise, self.start_noise, size=2)
            candidate = (self.start_position + shift).astype(np.float32)

            if not self._is_collision(candidate):
                return candidate

        # Practically never happening, but just in case falling back to the normal start
        return self.start_position.copy()

    def _get_state(self):
        goal_delta = self.goal_position - self.position

        return np.array(
            [
                self.position[0],
                self.position[1],
                goal_delta[0],
                goal_delta[1],
            ],
            dtype=np.float32,
        )

    def _distance_to_goal(self, position=None):
        if position is None:
            position = self.position

        return float(np.linalg.norm(self.goal_position - position))

    def _is_outside_map(self, position):
        x, y = position

        return (
            x - AGENT_RADIUS < 0.0
            or x + AGENT_RADIUS > MAP_WIDTH
            or y - AGENT_RADIUS < 0.0
            or y + AGENT_RADIUS > MAP_HEIGHT
        )

    def _collides_with_obstacle(self, position):
        x, y = position

        for x_min, y_min, x_max, y_max in OBSTACLES:
            closest_x = np.clip(x, x_min, x_max)
            closest_y = np.clip(y, y_min, y_max)

            dx = x - closest_x
            dy = y - closest_y

            if dx * dx + dy * dy <= AGENT_RADIUS * AGENT_RADIUS:
                return True

        return False

    def _is_collision(self, position):
        return self._is_outside_map(position) or self._collides_with_obstacle(position)

    def _goal_reached(self, position=None):
        if position is None:
            position = self.position

        return self._distance_to_goal(position) <= GOAL_RADIUS

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode is finished. Call reset() before step().")

        action = np.asarray(action, dtype=np.float32)

        if action.shape != (2,):
            raise ValueError(f"Action must have shape (2,), got {action.shape}")

        action = np.clip(action, -1.0, 1.0)

        old_distance = self._distance_to_goal()

        next_position = self.position + action * STEP_SIZE

        collision = self._is_collision(next_position)
        reached_goal = False

        if not collision:
            self.position = next_position
            reached_goal = self._goal_reached()

        new_distance = self._distance_to_goal()

        progress = old_distance - new_distance

        reward = (
            PROGRESS_REWARD_SCALE * progress
            - STEP_PENALTY
        )

        self.steps += 1

        if collision:
            reward -= COLLISION_PENALTY
            self.done = True

        elif reached_goal:
            reward += GOAL_REWARD
            self.done = True

        elif self.steps >= MAX_EPISODE_STEPS:
            self.done = True

        info = {
            "success": reached_goal,
            "collision": collision,
            "distance_to_goal": new_distance,
            "steps": self.steps,
        }

        return self._get_state(), float(reward), self.done, info


if __name__ == "__main__":
    env = NavigationEnv()

    state = env.reset()

    print("Initial state:", state)

    for _ in range(10):
        action = np.array([1.0, 1.0], dtype=np.float32)

        state, reward, done, info = env.step(action)

        print(
            f"state={state}, "
            f"reward={reward:.3f}, "
            f"done={done}, "
            f"info={info}"
        )

        if done:
            break
