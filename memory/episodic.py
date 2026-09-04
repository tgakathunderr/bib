from dataclasses import dataclass
import numpy as np

from bib.config import PER_ALPHA, PER_CAPACITY, PER_EPSILON, SDR_SIZE


@dataclass
class Episode:
    sdr: np.ndarray
    action: int
    reward: float
    td_error: float
    ach: float
    da: float
    priority: float = 1.0


class EpisodicBuffer:
    def __init__(self, capacity: int = PER_CAPACITY) -> None:
        self.capacity = capacity
        self._sdrs = np.zeros((capacity, SDR_SIZE), dtype=np.int16)
        self._actions = np.zeros(capacity, dtype=np.int32)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._td_errors = np.zeros(capacity, dtype=np.float32)
        self._achs = np.zeros(capacity, dtype=np.float32)
        self._das = np.zeros(capacity, dtype=np.float32)
        self._priorities = np.zeros(capacity, dtype=np.float64)
        self._filled = np.zeros(capacity, dtype=bool)
        self._lengths = np.zeros(capacity, dtype=np.int32)
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
        if len(sdr) == 0:
            return
        slot = self._cursor
        self._sdrs[slot].fill(0)
        max_len = min(len(sdr), SDR_SIZE)
        self._sdrs[slot, :max_len] = np.asarray(sdr[:max_len], dtype=np.int16)
        self._lengths[slot] = max_len
        self._actions[slot] = int(action)
        self._rewards[slot] = float(reward)
        self._td_errors[slot] = float(td_error)
        self._achs[slot] = float(ach)
        self._das[slot] = float(da)
        self._priorities[slot] = (abs(td_error) + PER_EPSILON) ** PER_ALPHA
        self._filled[slot] = True
        self._cursor = (self._cursor + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def _build_episode(self, i: int) -> Episode:
        n = int(self._lengths[i])
        return Episode(
            sdr=np.asarray(self._sdrs[i, :n], dtype=np.int64),
            action=int(self._actions[i]),
            reward=float(self._rewards[i]),
            td_error=float(self._td_errors[i]),
            ach=float(self._achs[i]),
            da=float(self._das[i]),
            priority=float(self._priorities[i]),
        )

    def sample_prioritized(self, n: int) -> list[Episode]:
        if self._size == 0:
            return []
        n = min(n, self._size)
        idxs = np.where(self._filled)[0]
        probs = self._priorities[idxs] / self._priorities[idxs].sum()
        chosen = np.random.choice(np.arange(len(idxs)), size=n, replace=False, p=probs)
        return [self._build_episode(idxs[i]) for i in chosen]

    def sample_chronological(self, n: int) -> list[Episode]:
        idxs = np.where(self._filled)[0]
        if n < len(idxs):
            idxs = idxs[:n]
        return [self._build_episode(i) for i in idxs]

    def get_top_k(self, k: int) -> list[Episode]:
        idxs = np.where(self._filled)[0]
        if len(idxs) == 0:
            return []
        order = idxs[np.argsort(-self._priorities[idxs])][:k]
        return [self._build_episode(i) for i in order]

    def update_priority(self, slot: int, new_td_error: float) -> None:
        if 0 <= slot < self.capacity and self._filled[slot]:
            self._priorities[slot] = (abs(new_td_error) + PER_EPSILON) ** PER_ALPHA

    @property
    def size(self) -> int:
        return self._size

    def __len__(self) -> int:
        return self._size
