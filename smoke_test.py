import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bib.brain import BIB
import numpy as np

print("Initializing BIB...")
brain = BIB()
print("BIB initialized. Running 3 ticks...")

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
    chem = brain.get_chemistry()
    name = brain.get_action_name(action)
    da = chem["DA"]
    ach = chem["ACh"]
    print(f"  Tick {i+1}: action={name}, DA={da:.3f}, ACh={ach:.3f}")

print("needs_sleep:", brain.needs_sleep())
tele = brain.get_telemetry()
print("hippo binds:", tele["hippocampus"]["binds"])
print("PER buffer:", tele["episodic_buffer_size"])
print("")
print("SMOKE TEST PASSED")
