import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bib.examples.grid_agent import GridWorld, run_episodes


def test_grid_world_step_returns_valid_shapes():
    env = GridWorld(size=4, seed=0)
    env.reset()
    sensors, reward, done = env.step(4)
    assert set(sensors.keys()) == {
        "vision", "touch", "proprioception",
        "chemoreception", "interoception", "auditory",
    }
    assert -1.0 <= reward <= 1.0
    assert isinstance(done, bool)


def test_brain_learns_from_environment_interaction():
    steps, brain = run_episodes(n_episodes=20, max_steps=25, size=4, seed=1)
    actor_scale = max(
        float(brain.basal_ganglia.actor_weights.max()),
        abs(float(brain.basal_ganglia.actor_weights.min())),
    )
    assert actor_scale > 0.001


def test_brain_completes_at_least_one_episode():
    steps, brain = run_episodes(n_episodes=20, max_steps=25, size=4, seed=2)
    assert any(s < 25 for s in steps)


def test_episodic_buffer_fills_during_environment_run():
    _, brain = run_episodes(n_episodes=5, max_steps=15, size=4, seed=3)
    assert len(brain.episodic_buffer) > 0
