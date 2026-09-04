# BIB: Biologically Inspired Brain

**BIB** is a Python library that models the mammalian brain as ~12 cooperating organs and drives them on every tick. It uses **Sparse Distributed Representations (SDRs)**, **Spike-Timing-Dependent Plasticity (STDP)**, and **neuromodulator state** (DA/NE/5-HT/ACh/cortisol) instead of backpropagation or batched epochs. Every tunable parameter — grid geometry, synapse counts, learning rates, decay constants — lives in `config.py`, each annotated with its biological citation.

This is an experimental neuroscience-grounded brain, not a drop-in ANN. It is intended to be the cognitive core of an autonomous agent, driven through the single `brain.tick()` interface.

## How it is grounded in code

The architecture and every constant are described in the source. Read order if you want to verify claims in this document:

| File | What it defines |
|---|---|
| `config.py` | All constants and their neuroscience citations |
| `brain.py` | The `BIB` orchestrator + `tick()` control flow |
| `core/thalamus.py` | NE-gated multi-modal encoder |
| `core/neocortex.py` | 3-layer STDP cortical hierarchy |
| `core/association_cortex.py` | Cross-modal world-state fusion |
| `action/basal_ganglia.py` | D1/D2 Go/NoGo actor-critic, TD(λ) |
| `core/hippocampus.py` | DG→CA3→CA1→EC episodic memory |
| `memory/episodic.py`, `memory/sleep.py` | Prioritized replay + sleep consolidation |

## Key Features (each tied to a constant)

- **Sparse coding.** Each representation activates `SDR_SPARSITY = 32` of `SDR_SIZE = 4096` columns (~0.78%). Per `config.py`, this sparsity gives an interference probability ≈ C(N,W)⁻¹ ≈ 10⁻⁷⁰ (BIM 1 proof; Maass 2000, Mountcastle 1978).
- **STDP.** Causal/anti-causal plasticity (`STDP_A_PLUS=0.20`, `STDP_A_MINUS=0.10`) over time constants `STDP_TAU_PLUS=20` / `STDP_TAU_MINUS=40` ticks, in a `STDP_TAU_WINDOW=10`-tick window (Bi & Poo 1998).
- **Hierarchical neocortex.** Each modality runs 3 cortical layers with leaky temporal pooling (`LAYER_DECAY_RATES=(0.80, 0.95, 0.99)`) and top-down apical feedback from layers 1–2 (`APICAL_SOURCE_LAYERS`), implementing predictive coding (Rao & Ballard 1999, Friston 2010).
- **Neuromodulation.** DA encodes reward prediction error (`BASE_DA=0.0`, `DA_DECAY=0.90`; Schultz 1997). NE gates sensory-gain across `NE_GAIN_MIN=0.5` → `NE_GAIN_MAX=2.0` (Aston-Jones & Cohen 2005). 5-HT scales the NoGo pathway (`SEROTONIN_NOGO_BIAS=0.30`). ACh rises with surprise (`ACH_SURPRISE_GAIN=0.50`; Hasselmo 2006).
- **Basal Ganglia actor-critic.** A D1 Go / D2 NoGo policy (`LR_ACTOR=0.05`, `LR_NOGO=0.03`) and a linear value function (`LR_CRITIC=0.01`) learned with TD(λ) eligibility traces (`LAMBDA_TRACE=0.70`, discount `GAMMA=0.95`; Frank et al. 2004, Sutton & Barto 1998). Exploration is confidence-gated (`BABBLE_THRESHOLD=0.10`) plus a base random rate (`BABBLE_RATE_BASE=0.10`).
- **Hippocampal memory.** DG→CA3→CA1→EC with a CA3 Hopfield attractor (`HIPPO_CA3_SIZE=2048`, `HIPPO_RETRIEVE_ITER=4`) for pattern completion, and CA1 as a mismatch/novelty detector. Cortisol suppresses hippocampal binding (`CORTISOL_HIPPO_SUPPRESS=0.8`).
- **Sleep consolidation.** `SWS_N_REPLAYS=300` sharp-wave re-plays consolidate episodic memory into cortex; `REM_N_CHAINS=80` CA3 chains; `RENORM_FACTOR=0.97` synaptic down-scaling (Tononi & Cirelli 2006).
- **Prioritized experience replay.** `PER_CAPACITY=2000` episodes stored as parallel NumPy arrays, sampled by TD-error priority (`PER_ALPHA=0.6`, `PER_BETA=0.4`; Schaul et al. 2016).

The full sensor interface is 6 modalities with dimensions `MODALITY_INPUT_DIMS`: vision **256**, touch 64, proprioception 32, chemoreception 32, interoception 16, auditory 128. Actions are the 8 MRS-GREN life-processes in `ACTION_NAMES`.

## Memory footprint

The one dense matrix is the CA3 attractor: `HIPPO_CA3_SIZE² × 4 bytes = 16 MB`. The cortices use sparse synapse arrays, and cross-modal fusion uses Knuth-hash topographic projection (`core/association_cortex.py`) with no dense projection matrix. Total model is on the order of low tens of MB.

## Two design choices that matter for learning

These are not tuning folklore — they are enforced in the code because, without them, the actor-critic cannot learn:

1. **The world-state representation is the early-layer fusion, not the deep L3 output.** Deep STDP temporal pooling monotonically erodes spatial discriminability (measured: raw fusion ~13/32 mean overlap across grid positions vs ~31/32 after 3 pooling layers). `association_cortex.step()` therefore feeds the actor the association cortex's **L1 winner cells** over per-modality raw SDRs. See `core/association_cortex.py` and its docstring.
2. **The critic learning rate is low (`LR_CRITIC=0.01`).** Shared SDR columns couple states, and TD(0) with linear value approximation diverges at high LR (Sutton & Barto §9.4). At the previous value of 0.10 the critic diverged (V(s)→~74 for a reward-bounded task); 0.01 keeps it stable.

## Quick Start

```python
import numpy as np
from bib.brain import BIB

brain = BIB()

sensors = {
    "vision": np.zeros(256, dtype=np.float32),
    "touch": np.zeros(64, dtype=np.float32),
    "proprioception": np.zeros(32, dtype=np.float32),
    "interoception": np.zeros(16, dtype=np.float32),
    "chemoreception": np.zeros(32, dtype=np.float32),
    "auditory": np.zeros(128, dtype=np.float32),
}

homeostatic_state = {"hunger": 0.5, "thirst": 0.2, "pain": 0.0, "energy": 0.5}

action = brain.tick(sensors, reward=0.0, homeostatic_state=homeostatic_state)

print(brain.get_action_name(action))
print(brain.get_chemistry())
```

For an interactive sandbox with live chemistry charts and telemetry:

```bash
python repl.py
```

## End-to-end example: grid navigation

`examples/grid_agent.py` wires the brain to a 5×5 grid world with a fixed target and a dense distance-based shaped reward. The basal ganglia learns the navigation policy online:

```python
from bib.examples.grid_agent import run_episodes

steps_to_goal, brain = run_episodes(n_episodes=40, max_steps=40, size=5)
print(steps_to_goal)
```

In targets the brain can reach, learning is monotonic (episodes drop from ~13 to ~4 steps). Note that the 5×5 task is at the edge of this small actor's capacity: it reliably solves reachable targets but can fail on the farthest corner within 40 episodes.

## Tests

```bash
python -m pytest tests/ -q
```

23 tests in `tests/`:

- **`test_brain.py`** — end-to-end wiring: action validity, chemistry bounds, TD-error dynamics, sleep, persistence, cerebellum and PFC influence on action selection.
- **`test_properties.py`** — numeric and neuroscience invariants: CA3 attractor retrieval under corruption/capacity limits, critic convergence to its analytic fixed point `V = r/(1−γ)`, world-state positional discriminability, and critic non-divergence.
- **`test_environment.py`** — the grid-world environment and its interaction with the learning brain.

## Theoretical Foundations

Constants cite primary sources in `config.py`; key references:

- **Schultz (1997)** — dopamine neurons encode reward prediction error (`BASE_DA`, `DA_DECAY`).
- **Marr (1971)** / **Rolls (2013)** / **O'Keefe & Dostrovsky (1971)** — hippocampal DG pattern separation, CA3 attractor completion, place cells.
- **Bi & Poo (1998)** — STDP amplitudes/time constants.
- **Frank, Seeberger & O'Reilly (2004)** — D1/D2 basal-ganglia model of action selection.
- **Sutton & Barto (1998)** — TD(λ) eligibility traces; §9.4 divergence of linear VFA.
- **Tononi & Cirelli (2006)** — synaptic homeostasis / sleep consolidation.
- **Sherman & Guillery (2006)** — thalamus as an active gatekeeper; **Aston-Jones & Cohen (2005)** — LC-NE gain.
- **Schaul et al. (2016)** — prioritized experience replay.

## License

MIT. See [LICENSE](LICENSE).
