"""Synthetic network source preserving the legacy bandwidth random walk."""

from __future__ import annotations

from abc import ABC, abstractmethod
import random


class NetworkTrace(ABC):
    """Supplies the next observed bandwidth for a streaming chunk."""

    @abstractmethod
    def reset(self, seed: int | None = None) -> None:
        """Reset state; an explicit seed must produce repeatable samples."""

    @abstractmethod
    def next_bandwidth(self, previous_bandwidth_kbps: float) -> float:
        """Return the bandwidth measurement for the next chunk."""


class SyntheticTrace(NetworkTrace):
    """Legacy ±20% multiplicative bandwidth random walk.

    The unbounded random walk is deliberately preserved. Separation only makes
    later LTE, 5G, Wi-Fi, mobility, and OOD trace sources pluggable.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = random.Random(seed)

    def next_bandwidth(self, previous_bandwidth_kbps: float) -> float:
        return previous_bandwidth_kbps * self._rng.uniform(0.8, 1.2)
