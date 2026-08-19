"""Common interface for controllers evaluated in the streaming environment."""

from abc import ABC, abstractmethod

import numpy as np


class ABRController(ABC):
    """Selects an encoded-bitrate index from an environment observation."""

    @abstractmethod
    def select_bitrate(self, observation: np.ndarray) -> int:
        """Return a valid discrete bitrate action for ``observation``."""
