import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bib.brain import BIB
from bib.config import N_ACTIONS, MODALITY_INPUT_DIMS, SDR_SIZE


def make_sensors(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "vision": rng.random(MODALITY_INPUT_DIMS["vision"]).astype("float32"),
        "touch": rng.random(MODALITY_INPUT_DIMS["touch"]).astype("float32"),
        "proprioception": rng.random(MODALITY_INPUT_DIMS["proprioception"]).astype("float32"),
        "chemoreception": rng.random(MODALITY_INPUT_DIMS["chemoreception"]).astype("float32"),
        "interoception": rng.random(MODALITY_INPUT_DIMS["interoception"]).astype("float32"),
        "auditory": rng.random(MODALITY_INPUT_DIMS["auditory"]).astype("float32"),
    }


def test_tick_returns_valid_action():
    brain = BIB()
    sensors = make_sensors()
    for _ in range(10):
        action = brain.tick(sensors)
        assert 0 <= action < N_ACTIONS


def test_chemistry_in_bounds():
    brain = BIB()
    sensors = make_sensors()
    for _ in range(5):
        brain.tick(sensors, reward=0.5)
    chem = brain.get_chemistry()
    assert -1.0 <= chem["DA"] <= 1.0
    assert 0.0 <= chem["NE"] <= 1.0
    assert 0.0 <= chem["5-HT"] <= 1.0
    assert 0.0 <= chem["ACh"] <= 1.0
    assert 0.0 <= chem["Cortisol"] <= 1.0


def test_td_error_changes_with_reward():
    brain_a = BIB()
    brain_b = BIB()
    sensors = make_sensors(1)
    for _ in range(5):
        brain_a.tick(sensors, reward=0.0)
        brain_b.tick(sensors, reward=1.0)
    assert brain_a.get_telemetry()["last_td_error"] != brain_b.get_telemetry()["last_td_error"]


def test_no_nan_in_weights():
    brain = BIB()
    sensors = make_sensors(2)
    for _ in range(5):
        brain.tick(sensors)
    assert not np.isnan(brain.basal_ganglia.actor_weights).any()
    assert not np.isnan(brain.hippocampus.ca3_weights).any()
    assert not np.isnan(brain.amygdala.valence_weights).any()


def test_sleep_resets_fatigue():
    brain = BIB()
    sensors = make_sensors(3)
    for _ in range(10):
        brain.tick(sensors)
    brain.hypothalamus.fatigue = 0.9
    assert brain.needs_sleep()
    report = brain.sleep()
    assert brain.hypothalamus.fatigue == 0.0
    assert report["sws_replays"] >= 0


def test_save_load_roundtrip(tmp_path):
    brain = BIB()
    sensors = make_sensors(4)
    for _ in range(5):
        brain.tick(sensors)
    brain.save(str(tmp_path))
    loaded = BIB()
    loaded.load(str(tmp_path))
    np.testing.assert_array_equal(loaded.basal_ganglia.actor_weights, brain.basal_ganglia.actor_weights)
    np.testing.assert_array_equal(loaded.hippocampus.ca3_weights, brain.hippocampus.ca3_weights)


class NoBabbleRng:
    def random(self):
        return 1.0

    def integers(self, low, high, endpoint=None):
        return low


def test_cerebellum_correction_changes_action_selection():
    brain = BIB()
    sensors = make_sensors(5)
    for _ in range(10):
        brain.tick(sensors)

    large_correction = np.zeros(N_ACTIONS, dtype=np.float32)
    forced_action = 1
    large_correction[forced_action] = 1000.0
    brain.basal_ganglia._rng = NoBabbleRng()

    forced_actions = []
    for _ in range(5):
        action, _ = brain.basal_ganglia.select_action(
            brain.association_cortex.get_world_state_sdr(),
            da=brain.brainstem.da,
            ach=brain.brainstem.ach,
            serotonin=brain.brainstem.sht,
            fear_salience=0.0,
            cerebellum_correction=large_correction,
        )
        forced_actions.append(action)
    assert all(a == forced_action for a in forced_actions)


def test_pfc_bias_flows_into_association_cortex():
    brain = BIB()
    sensors = make_sensors(7)
    for _ in range(5):
        brain.tick(sensors, reward=0.5)

    target_col = brain.association_cortex.get_world_state_sdr()[0]
    bias = np.zeros(SDR_SIZE, dtype=np.float32)
    bias[target_col] = 0.25

    brain.association_cortex.apply_pfc_bias(bias)
    pred_cols = brain.association_cortex.cortex.layers[0].get_predictive_columns()

    assert len(pred_cols) > 0
    assert target_col in set(int(c) for c in pred_cols)


def test_episodic_buffer_stores_and_samples():
    brain = BIB()
    sensors = make_sensors(6)
    for _ in range(20):
        brain.tick(sensors, reward=0.1)
    buf = brain.episodic_buffer
    assert len(buf) == 20
    sampled = buf.sample_prioritized(5)
    assert len(sampled) == 5
    chrono = buf.sample_chronological(3)
    assert len(chrono) == 3
    for ep in sampled + chrono:
        assert len(ep.sdr) > 0
