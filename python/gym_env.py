from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env_client import RLGodotClient


class GodotGymEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9000,
        obs_dim: int = 8,
        timeout: float = 10.0,
    ) -> None:
        super().__init__()
        self.client = RLGodotClient(host=host, port=port, timeout=timeout)
        self.obs_dim = obs_dim
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        obs = self.client.reset(seed=seed)
        if obs.shape != (self.obs_dim,):
            raise ValueError(
                f"Unexpected observation shape: {obs.shape}, expected {(self.obs_dim,)}"
            )
        return obs.astype(np.float32), {}

    def step(self, action: np.ndarray):
        obs, reward, done = self.client.step(action)
        if obs.shape != (self.obs_dim,):
            raise ValueError(
                f"Unexpected observation shape: {obs.shape}, expected {(self.obs_dim,)}"
            )
        terminated = done
        truncated = False
        return obs.astype(np.float32), float(reward), terminated, truncated, {}

    def close(self) -> None:
        self.client.close()
