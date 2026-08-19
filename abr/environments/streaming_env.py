"""Gymnasium environment for the legacy chunk-based ABR simulation."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from abr.config import StreamingConfig
from abr.metrics.qoe import calculate_qoe
from abr.traces.synthetic import NetworkTrace, SyntheticTrace


class StreamingEnv(gym.Env[np.ndarray, int]):
    """ABR simulator with legacy transition and reward semantics.

    Observation order is ``[bandwidth_kbps, buffer_seconds, last_quality]``.
    Actions are indices into ``config.bitrates_kbps``. With default settings,
    its declared spaces match the legacy PPO checkpoint.
    """

    metadata = {"render_modes": []}

    def __init__(
        self, config: StreamingConfig | None = None, trace: NetworkTrace | None = None
    ) -> None:
        super().__init__()
        self.config = config or StreamingConfig()
        if not self.config.bitrates_kbps:
            raise ValueError("bitrates_kbps must contain at least one bitrate")
        self.trace = trace or SyntheticTrace(seed=self.config.seed)
        self.action_space = spaces.Discrete(len(self.config.bitrates_kbps))
        # Preserve legacy declared bounds for existing model compatibility.
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0], dtype=np.float32),
            high=np.array(
                [5000, self.config.max_buffer_seconds, len(self.config.bitrates_kbps) - 1],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )
        self.state = self._initial_state()
        self.chunks_left = self.config.episode_length

    def _initial_state(self) -> np.ndarray:
        return np.array(
            [
                self.config.initial_bandwidth_kbps,
                self.config.initial_buffer_seconds,
                0,
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.trace.reset(seed=seed)
        self.state = self._initial_state()
        self.chunks_left = self.config.episode_length
        return self.state.copy(), {}

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid bitrate action: {action}")

        action_index = int(action)
        previous_speed, previous_buffer, previous_quality = self.state
        # Keep the legacy NumPy float32 arithmetic path for checkpoint-compatible
        # transitions; casting to Python float changes low-order results.
        new_speed = self.trace.next_bandwidth(previous_speed)
        chosen_bitrate = self.config.bitrates_kbps[action_index]
        qoe = calculate_qoe(
            bitrate_kbps=chosen_bitrate,
            previous_quality=int(previous_quality),
            new_quality=action_index,
            previous_buffer_seconds=float(previous_buffer),
            bandwidth_kbps=new_speed,
            chunk_duration_seconds=self.config.chunk_duration_seconds,
            max_buffer_seconds=self.config.max_buffer_seconds,
        )
        self.state = np.array([new_speed, qoe.buffer_seconds, action_index], dtype=np.float32)
        self.chunks_left -= 1
        terminated = self.chunks_left <= 0
        info = {
            "rebuffer": qoe.rebuffer_seconds,
            "bitrate_kbps": chosen_bitrate,
            "buffer_seconds": qoe.buffer_seconds,
        }
        return self.state.copy(), qoe.reward, terminated, False, info
