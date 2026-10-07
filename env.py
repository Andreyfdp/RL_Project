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
    def __init__(self, seed=SEED):
        self.rng = np.random.default_rng(seed)

        self.start_position = np.array(START_POSITION, dtype=np.float32)
        self.goal_position = np.array(GOAL_POSITION, dtype=np.float32)

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

    def reset(self):
        self.position = self.start_position.copy()
        self.steps = 0
        self.done = False

        return self._get_state()

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