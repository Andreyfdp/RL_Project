"""Gymnasium adapter for the project's continuous 2D navigation task."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import GOAL_POSITION, MAP_HEIGHT, MAP_WIDTH, SEED, START_POSITION
from env import NavigationEnv


class GymnasiumNavigationEnv(gym.Env):
    """Expose :class:`NavigationEnv` through Gymnasium's reset/step contract.

    ``options`` may contain a ``start_position`` and a ``goal_position``.  Both
    must be supplied together and must be collision-free positions.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 8}

    def __init__(self, render_mode=None):
        super().__init__()
        if render_mode not in (None, "ansi"):
            raise ValueError("Only render_mode='ansi' is supported.")

        self.render_mode = render_mode
        self.core = NavigationEnv(seed=SEED)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -MAP_WIDTH, -MAP_HEIGHT], dtype=np.float32),
            high=np.array([MAP_WIDTH, MAP_HEIGHT, MAP_WIDTH, MAP_HEIGHT], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}

        has_start = "start_position" in options
        has_goal = "goal_position" in options
        if has_start != has_goal:
            raise ValueError(
                "options must provide both start_position and goal_position, or neither."
            )

        start = options.get("start_position", START_POSITION)
        goal = options.get("goal_position", GOAL_POSITION)
        observation = self.core.reset(start_position=start, goal_position=goal)
        info = {
            "success": False,
            "collision": False,
            "distance_to_goal": self.core._distance_to_goal(),
            "steps": 0,
        }
        return observation, info

    def step(self, action):
        observation, reward, done, info = self.core.step(action)
        terminated = bool(info["success"] or info["collision"])
        truncated = bool(done and not terminated)
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "ansi":
            return None
        return (
            f"position={self.core.position.tolist()}, "
            f"goal={self.core.goal_position.tolist()}, steps={self.core.steps}"
        )
