# BIB — Usage Guide

How to install, drive, and extend the BIB brain. 
For the shorthand overview, see the
[README](../README.md).

## Requirements

- Python 3.10+ (the code uses PEP 604 `X | None` type syntax).
- `numpy`, `numba` (JIT kernels), `rich` (REPL rendering). Listed in `setup.py`.

## Install

```bash
pip install -e .
```

## The core interface: `brain.tick()`

Everything happens through one method on `bib.brain.BIB`:

```python
action = brain.tick(sensors, reward=0.0, homeostatic_state=None)
```

`sensors` maps each of the 6 modalities to a `float32` array whose length
matches `MODALITY_INPUT_DIMS` in `config.py`:

| modality        | length |
|-----------------|--------|
| `vision`        | 256    |
| `touch`         | 64     |
| `proprioception`| 32     |
| `chemoreception`| 32     |
| `interoception` | 16     |
| `auditory`      | 128    |

Missing or all-zero modalities are handled gracefully: the Thalamus encodes
them as silent (empty) SDRs. This matters — a constant/zero sensor stream must
not dominate the world-state representation.

`reward` is a scalar in `[-1, 1]`. `homeostatic_state` is an optional dict with
`hunger`, `thirst`, `pain`, and `energy` floats.

The return value is an action index in `[0, 7]` — one of the MRS-GREN
life-process names in `config.ACTION_NAMES`:

| index | name          |
|-------|---------------|
| 0     | MOVEMENT      |
| 1     | REPRODUCTION  |
| 2     | SENSITIVITY   |
| 3     | GROWTH        |
| 4     | EXCRETION     |
| 5     | NUTRITION     |
| 6     | IDLE          |
| 7     | COMMUNICATE   |

Use `brain.get_action_name(action)` to resolve an index to a name.

### Minimal example

```python
import numpy as np
from bib.brain import BIB

brain = BIB()

sensors = {
    "vision": np.zeros(256, dtype=np.float32),
    "touch": np.zeros(64, dtype=np.float32),
    "proprioception": np.zeros(32, dtype=np.float32),
    "chemoreception": np.zeros(32, dtype=np.float32),
    "interoception": np.zeros(16, dtype=np.float32),
    "auditory": np.zeros(128, dtype=np.float32),
}

for _ in range(100):
    action = brain.tick(sensors, reward=0.0)
    print(brain.get_action_name(action))
```

### Reward semantics

The basal ganglia learns from the reward prediction error
`δ = r + γ·V(s′) − V(s)`. Because reward for a transition is delivered on the
*tick after* the action, **pass `reward=0.0` on the first tick** and then the
reward produced by the previous action on each subsequent tick. See the
`grid_agent.py` loop for the deferred-reward pattern:

```python
prev_reward = 0.0
for step in range(max_steps):
    action = brain.tick(sensors, reward=prev_reward)
    _, reward, done = env.step(action)
    prev_reward = reward
```

## The REPL

`repl.py` runs an interactive mock organism with live chemistry charts.

```bash
python repl.py
```

Commands:

| command        | effect                              |
|----------------|-------------------------------------|
| `[REWARD:x]`   | inject graded reward signal         |
| `[PAIN:x]`     | inject graded pain signal           |
| `[SLEEP]`      | force a sleep cycle                 |
| `[STATS]`      | dump full telemetry                 |
| `[SAVE path]`  | save a brain snapshot to `path`     |
| `[LOAD path]`  | load a brain snapshot from `path`   |
| `exit` / `quit`| exit                                |

## Persistence

Save and load the whole brain (weight matrices for BG, hippocampus CA3,
amygdala, PFC, cerebellum, and all cortical layers) to a directory:

```python
brain.save("./snapshot/")
brain.load("./snapshot/")
```

Snapshots are directory-based, one `.npy` file per weight matrix.

## Telemetry

```python
brain.get_chemistry()   # {DA, NE, 5-HT, ACh, Cortisol}
brain.get_telemetry()   # tick, chemistry, homeostasis, hippocampus,
                        # prefrontal, episodic_buffer_size, last_action,
                        # last_td_error, is_exploring, last_motor_error, last_surprise
brain.needs_sleep()     # True when fatigue exceeds SLEEP_THRESHOLD (0.85)
```

## Sleep

When `brain.needs_sleep()` returns `True`, run a consolidation cycle:

```python
if brain.needs_sleep():
    report = brain.sleep()
    # report: sws_replays, rem_chains, n1n2_ticks,
    #         synaptic_renorm, goal_consolidated
```

Sleep has three stages (see Architecture.md). It resets fatigue afterward.

## End-to-end example: grid navigation

The repo ships a working RL loop in `examples/grid_agent.py`:

```python
from bib.examples.grid_agent import run_episodes

steps_to_goal, brain = run_episodes(n_episodes=40, max_steps=40, size=5)
```

`GridWorld` is a plain class you can reuse as a template: it implements
`_sensors()` (builds the 6 modality arrays), `step(action)` (applies the move,
returns `(sensors, reward, done)`), and `shaped_reward()` (distance-based
shaping). The `run_episodes` driver shows the deferred-reward tick loop and
sleep handling.

To use BIB with your own environment, implement the same contract:
`_sensors()` → modality dict, and a reward that reflects your task.

## Running the tests

```bash
python -m pytest tests/ -q
```

23 tests across `tests/test_brain.py`, `tests/test_properties.py`, and
`tests/test_environment.py` (see the README for what each covers).

## Extending BIB

### Tune parameters

All constants live in `config.py`, each annotated with the neuroscience
source. Change e.g. `SDR_SPARSITY`, `LR_ACTOR`, `LAMBDA_TRACE`, or
`HIPPO_CA3_SIZE` there — nothing is hard-coded in module bodies.

Two parameters are load-bearing for learning (see Architecture.md):

- `LR_CRITIC` — keep small (0.01). High values make linear TD diverge under
  overlapping SDR features.
- The world-state that feeds the actor is the association cortex's **L1
  winner cells** over raw modal SDRs. If you change `association_cortex.step`
  to export a deeper (L3) representation, the actor-critic will lose the
  spatial discriminability it needs.

### Add an organ or signal

- Add a new module under `core/` (or `action/`, `chemistry/`, `memory/`) and a
  reference on the `BIB` class in `brain.py`.
- Wire its `tick`-time update into `BIB.tick()` at the appropriate numbered
  step, and add any top-down/bottom-up projection to the relevant cortex via
  `apply_apical_bias` / `apply_hippo_bias` / `apply_pfc_bias`.
- Add telemetry under `BIB.get_telemetry()`.

## Tip for first-time users

Drive the brain from a *real* environment, not `tick(sensors, 0.0)` forever
with static zero sensors. The brain learns from temporal structure and reward.
The grid-agent example is the smallest working demonstration of that.
