# BIB: Biologically Inspired Brain

**BIB** is a 1:1 neurobiologically grounded, software-based artificial brain architecture. 

Unlike traditional Artificial Neural Networks (ANNs) that rely on backpropagation and massive datasets, BIB utilizes **Sparse Distributed Representations (SDRs)**, **Spike-Timing Dependent Plasticity (STDP)**, and dynamic **Neurochemistry** to learn continuously in real-time. It is designed to act as the cognitive core for autonomous agents and synthetic digital organisms.

## Key Features

- **No Backpropagation or Epochs:** BIB learns continuously and asynchronously. There is no distinction between "training" and "inference."
- **Real Neurochemistry:** The brain is modulated by realistic chemical gradients:
  - **Dopamine (DA):** Encodes Reward Prediction Error (RPE) for reinforcement.
  - **Acetylcholine (ACh):** Encodes surprise/novelty, modulating learning rates and curiosity.
  - **Serotonin (5-HT):** Regulates patience and behavioral inhibition.
  - **Norepinephrine (NE):** Controls sensory gain and alertness (attentional spotlight).
- **Basal Ganglia with TD(λ):** A D1/D2 Go/NoGo pathway implementation featuring Temporal Difference learning with Eligibility Traces for complex action-credit assignment.
- **Hippocampal Sleep Consolidation:** Features a working 3-stage sleep cycle where short-term episodic memories (Hippocampus) are consolidated into long-term semantic memory (Neocortex) via Sharp-Wave Ripples.
- **Constant Memory Footprint:** Employs Knuth-hash topographical projections to prevent 0(N^2) memory explosions across cortical regions, enabling complex multimodal fusion on consumer hardware.

## Architecture

BIB accurately models the macro-structures of the mammalian brain:

1. **Thalamus:** An active sensory gatekeeper that compresses raw multimodal floats (Vision, Touch, Proprioception, etc.) into Sparse Distributed Representations (SDRs).
2. **Neocortex (L1, L2, L3):** 6 hierarchical modal cortices + 1 Association Cortex. Handles predictive coding and pattern generalization.
3. **Prefrontal Cortex (PFC):** Maintains working memory and broadcasts top-down goal representations.
4. **Hippocampus (DG, CA3, CA1):** Acts as a fast-learning episodic buffer using Hopfield-like attractor networks.
5. **Amygdala:** Handles fast fear-conditioning and emotional salience.
6. **Basal Ganglia:** The primary action-selection engine (Actor-Critic).
7. **Brainstem (VTA, LC, DRN, NBM):** The neuromodulatory hub calculating basic drives and chemical RPEs.

## Installation

BIB requires Python 3.10+ and uses `numpy` and `numba` for JIT-compiled high-performance tensor operations.

```bash
git clone https://github.com/tgakathunderr/bib.git
cd bib
pip install numpy numba
```

## Quick Start

```python
import numpy as np
from bib.brain import BIB

# 1. Initialize the brain
brain = BIB()

# 2. Construct raw sensory floats (e.g., from a simulation or robot)
sensors = {
    "vision": np.zeros(256, dtype=np.float32),
    "touch": np.zeros(64, dtype=np.float32),
    "proprioception": np.zeros(32, dtype=np.float32),
    "interoception": np.zeros(16, dtype=np.float32),
    "chemoreception": np.zeros(32, dtype=np.float32),
    "auditory": np.zeros(128, dtype=np.float32)
}

# 3. Define the body's homeostatic state
homeostatic_state = {
    "hunger": 0.5,
    "thirst": 0.2,
    "pain": 0.0,
    "energy": 0.5
}

# 4. Tick the brain (returns an action index 0-7)
action = brain.tick(sensors, reward=0.0, homeostatic_state=homeostatic_state)

# 5. Access internal chemistry (optional)
chemistry = brain.get_chemistry()
print(f"Dopamine Level: {chemistry['DA']}")
```

## Theoretical Foundations

BIB is heavily grounded in peer-reviewed neuroscience. Key implementations are based on:
- **Schultz (1997)**: Dopamine neurons encode reward prediction error.
- **Sherman & Guillery (2006)**: The thalamus as an active gatekeeper.
- **Frank, Seeberger & O'Reilly (2004)**: By carrot or by stick: cognitive reinforcement learning in parkinsonism (D1/D2 BG model).
- **Tononi & Cirelli (2006)**: Sleep and synaptic homeostasis hypothesis.
- **Sutton & Barto (1998)**: Reinforcement Learning: An Introduction (TD-learning).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
