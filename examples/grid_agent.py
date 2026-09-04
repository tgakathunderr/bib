"""
BIB Example: Grid-World Agent
=============================
A minimal reinforcement-learning benchmark for the BIB brain.

GridWorld: a 5x5 grid, a fixed target, and an agent that starts at (0,0).
The brain drives the agent with TD(λ) basal-ganglia learning on a dense,
distance-based shaped reward.

Run: python -m bib.examples.grid_agent
"""
from __future__ import annotations

import numpy as np

from bib.brain import BIB
from bib.config import MODALITY_INPUT_DIMS, N_ACTIONS


class GridWorld:
    def __init__(self, size: int = 5, seed: int = 0) -> None:
        self.size = size
        rng = np.random.default_rng(seed)
        self.target = (int(rng.integers(0, size)), int(rng.integers(0, size)))
        self.reset()

    def reset(self) -> tuple[int, int]:
        self.agent = (0, 0)
        self._prev_agent = (0, 0)
        return self.agent

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        dx, dy = self._move_vector(action)
        ax, ay = self.agent
        nx = min(self.size - 1, max(0, ax + dx))
        ny = min(self.size - 1, max(0, ay + dy))
        self._prev_agent = self.agent
        self.agent = (nx, ny)
        return self._sensors(), self.shaped_reward(), self.agent == self.target

    def _move_vector(self, action: int) -> tuple[int, int]:
        moves = {
            0: (0, 0),
            1: (0, 0),
            2: (0, 0),
            3: (0, 0),
            4: (1, 0),
            5: (-1, 0),
            6: (0, 1),
            7: (0, -1),
        }
        return moves.get(action, (0, 0))

    def shaped_reward(self) -> float:
        ax, ay = self.agent
        tx, ty = self.target
        prev_dist = abs(self._prev_agent[0] - tx) + abs(self._prev_agent[1] - ty)
        new_dist = abs(ax - tx) + abs(ay - ty)
        shaping = 0.2 * (prev_dist - new_dist)
        if self.agent == self.target:
            shaping += 1.0
        return float(np.clip(shaping, -1.0, 1.0))

    def _sensors(self) -> dict[str, np.ndarray]:
        ax, ay = self.agent
        tx, ty = self.target
        vision = np.zeros(MODALITY_INPUT_DIMS["vision"], dtype=np.float32)
        vision[0] = ax / self.size
        vision[1] = ay / self.size
        vision[2] = tx / self.size
        vision[3] = ty / self.size
        touch = np.zeros(MODALITY_INPUT_DIMS["touch"], dtype=np.float32)
        touch[0] = 1.0 if self.agent == self.target else 0.0
        proprioception = np.zeros(MODALITY_INPUT_DIMS["proprioception"], dtype=np.float32)
        proprioception[0] = ax / self.size
        proprioception[1] = ay / self.size
        chemoreception = np.zeros(MODALITY_INPUT_DIMS["chemoreception"], dtype=np.float32)
        chemoreception[0] = 1.0 - abs(ax - tx) / self.size
        interoception = np.zeros(MODALITY_INPUT_DIMS["interoception"], dtype=np.float32)
        auditory = np.zeros(MODALITY_INPUT_DIMS["auditory"], dtype=np.float32)
        return {
            "vision": vision,
            "touch": touch,
            "proprioception": proprioception,
            "chemoreception": chemoreception,
            "interoception": interoception,
            "auditory": auditory,
        }


def run_episodes(
    n_episodes: int = 50,
    max_steps: int = 40,
    size: int = 5,
    seed: int = 0,
) -> tuple[list[int], BIB]:
    brain = BIB()
    env = GridWorld(size=size, seed=seed)
    steps_to_goal = []
    for _ in range(n_episodes):
        env.reset()
        prev_reward = 0.0
        steps = 0
        for _ in range(max_steps):
            sensors = env._sensors()
            action = brain.tick(sensors, reward=prev_reward)
            _, reward, done = env.step(action)
            prev_reward = reward
            steps += 1
            if brain.needs_sleep():
                brain.sleep()
            if done:
                break
        steps_to_goal.append(steps if done else max_steps)
    return steps_to_goal, brain


if __name__ == "__main__":
    print("running grid-world sweep...")
    steps, brain = run_episodes()
    print(f"episode steps-to-goal: {steps}")
    print(f"first-10 mean: {sum(steps[:10])/10:.1f}  last-10 mean: {sum(steps[-10:])/10:.1f}")
