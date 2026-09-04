import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bib.brain import BIB
from bib.config import ACTION_NAMES, N_ACTIONS
import numpy as np

brain = BIB()

sensors = {
    "vision": np.random.rand(256).astype("float32"),
    "touch":  np.zeros(64, dtype="float32"),
    "proprioception": np.zeros(32, dtype="float32"),
    "chemoreception": np.zeros(32, dtype="float32"),
    "interoception":  np.zeros(16, dtype="float32"),
    "auditory":       np.zeros(128, dtype="float32"),
}

for i in range(3):
    reward = 0.1 if i == 1 else 0.0
    action = brain.tick(sensors, reward=reward)
    assert 0 <= action < N_ACTIONS, f"action {action} out of range"
    chem = brain.get_chemistry()
    name = brain.get_action_name(action)
    assert name in ACTION_NAMES, f"unknown action name {name}"
    assert -1.0 <= chem["DA"] <= 1.0, "DA out of range"
    assert 0.0 <= chem["NE"] <= 1.0, "NE out of range"
    assert 0.0 <= chem["5-HT"] <= 1.0, "5-HT out of range"
    assert 0.0 <= chem["ACh"] <= 1.0, "ACh out of range"
    assert 0.0 <= chem["Cortisol"] <= 1.0, "Cortisol out of range"
    print(f"  Tick {i+1}: action={name}, DA={chem['DA']:.3f}, ACh={chem['ACh']:.3f}")

assert isinstance(brain.needs_sleep(), bool), "needs_sleep must return bool"
tele = brain.get_telemetry()
assert {"tick", "chemistry", "homeostasis", "hippocampus", "prefrontal", "episodic_buffer_size", "last_action", "last_td_error"} <= set(tele.keys()), "telemetry missing keys"
assert isinstance(tele["episodic_buffer_size"], int), "buffer size must be int"
assert len(brain.basal_ganglia.actor_weights) == N_ACTIONS, "actor weights wrong rows"
assert not np.isnan(brain.basal_ganglia.critic_weights).any(), "critic has NaN"
assert not np.isnan(brain.hippocampus.ca3_weights).any(), "CA3 has NaN"
assert not np.isnan(brain.amygdala.valence_weights).any(), "amygdala has NaN"

print(f"  Blocks: CA3 binds={tele['hippocampus']['binds']}, PER={tele['episodic_buffer_size']}")
print("SMOKE TEST PASSED")
