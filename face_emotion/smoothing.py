"""Temporal smoothing for frame-by-frame probabilities."""

from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np


class ProbabilitySmoother:
    """Simple moving average over the latest valid probability vectors."""

    def __init__(self, window_size: int) -> None:
        self.window_size = window_size
        self._history: deque[np.ndarray] = deque(maxlen=window_size)

    def update(self, probabilities: Iterable[float]) -> np.ndarray:
        vector = np.array(list(probabilities), dtype=np.float32)
        self._history.append(vector)
        return self.value

    @property
    def value(self) -> np.ndarray:
        if not self._history:
            raise ValueError("The smoother has no probability history.")
        return np.mean(np.stack(self._history, axis=0), axis=0)

    def reset(self) -> None:
        self._history.clear()

    @property
    def is_empty(self) -> bool:
        return not self._history
