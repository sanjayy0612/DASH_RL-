"""Adapter that exposes a Stable-Baselines3 PPO policy as an ABR controller."""

from pathlib import Path
from typing import Any

import numpy as np

from .base import ABRController


class PPOController(ABRController):
    """Wrap a trained PPO model without coupling callers to SB3's API."""

    def __init__(self, model: Any, action_count: int, deterministic: bool = True) -> None:
        if action_count <= 0:
            raise ValueError("action_count must be positive")
        self._model = model
        self._action_count = action_count
        self._deterministic = deterministic

    @classmethod
    def from_path(
        cls, model_path: str | Path, action_count: int, deterministic: bool = True
    ) -> "PPOController":
        """Load a Stable-Baselines3 PPO model lazily."""
        from stable_baselines3 import PPO

        return cls(PPO.load(str(model_path)), action_count, deterministic)

    def select_bitrate(self, observation: np.ndarray) -> int:
        action, _ = self._model.predict(observation, deterministic=self._deterministic)
        action_index = int(action)
        if not 0 <= action_index < self._action_count:
            raise ValueError(
                f"PPO returned invalid action {action_index}; expected [0, {self._action_count})"
            )
        return action_index
