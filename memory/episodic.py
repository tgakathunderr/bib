"""
BIB Memory: Prioritized Experience Replay (PER)
================================================
Schaul et al. (2016) — Prioritized Experience Replay
O'Neill et al. (2010) — Reactivation of experience-dependent neural activity

Episodes are stored with priority = |TD-error| + ε.
High TD-error episodes (surprising, important) are replayed more during sleep.
This replaces BIM4's flat chronological buffer and fixes equal-priority replay.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from bib.config import PER_ALPHA, PER_BETA, PER_CAPACITY, PER_EPSILON


@dataclass
class Episode:
    """One timestep of experience."""
    sdr: np.ndarray          # cortical SDR (Association Cortex L3)
    action: int              # chosen action index
    reward: float            # external reward
    td_error: float          # |δ| = |r + γV(s') - V(s)|
    ach: float               # ACh at time of experience
    da: float                # DA at time of experience
    priority: float = 1.0   # sampling priority


class EpisodicBuffer:
    """
    Prioritized circular buffer of episodes.
    Priority = (|TD-error| + PER_EPSILON) ^ PER_ALPHA.
    Sampling: proportional to priority (not uniform like BIM4's FIFO).
    """

    def __init__(self, capacity: int = PER_CAPACITY) -> None:
        self.capacity = capacity
        self._buffer: list[Optional[Episode]] = [None] * capacity
        self._cursor: int = 0
        self._size: int = 0

    def store(
        self,
        sdr: np.ndarray,
        action: int,
        reward: float,
        td_error: float,
        ach: float,
        da: float,
    ) -> None:
        """Store one episode. Overwrites least-priority slot when full."""
        priority = (abs(td_error) + PER_EPSILON) ** PER_ALPHA
        ep = Episode(
            sdr=sdr.copy(),
            action=action,
            reward=reward,
            td_error=td_error,
            ach=ach,
            da=da,
            priority=priority,
        )
        self._buffer[self._cursor] = ep
        self._cursor = (self._cursor + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample_prioritized(self, n: int) -> list[Episode]:
        """
        Sample n episodes proportional to their priority.
        High |TD-error| episodes are sampled more frequently.
        """
        if self._size == 0:
            return []
        n = min(n, self._size)

        filled = [ep for ep in self._buffer if ep is not None]
        priorities = np.array([ep.priority for ep in filled], dtype=np.float64)
        probs = priorities / priorities.sum()

        chosen_idx = np.random.choice(len(filled), size=n, replace=False, p=probs)
        return [filled[i] for i in chosen_idx]

    def sample_chronological(self, n: int) -> list[Episode]:
        """Return episodes in chronological order (for NREM N1/N2 replay)."""
        filled = [ep for ep in self._buffer if ep is not None]
        return filled[:n] if n < len(filled) else filled

    def get_top_k(self, k: int) -> list[Episode]:
        """Return the k highest-priority episodes (most surprising)."""
        filled = [ep for ep in self._buffer if ep is not None]
        filled.sort(key=lambda e: e.priority, reverse=True)
        return filled[:k]

    def update_priority(self, episode: Episode, new_td_error: float) -> None:
        """Re-prioritize an episode after its TD-error is recomputed during replay."""
        episode.priority = (abs(new_td_error) + PER_EPSILON) ** PER_ALPHA

    @property
    def size(self) -> int:
        return self._size

    def __len__(self) -> int:
        return self._size
