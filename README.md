# BIB: Biologically Inspired Brain

[![Research Paper](https://img.shields.io/badge/Research-Paper-05f094.svg)](PAPER.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Numba JIT](https://img.shields.io/badge/JIT-Numba-orange.svg)](https://numba.pydata.org/)
[![Lab](https://img.shields.io/badge/Lab-UnikAI-00e1ff.svg)](https://www.unikai.in)

**BIB** is a 1:1 neurobiologically grounded, software-based artificial brain architecture developed by **UnikAI Lab**. 

Unlike traditional Artificial Neural Networks (ANNs) that rely on dense matrix multiplication, backpropagation, and massive static datasets, BIB utilizes **Sparse Distributed Representations (SDRs)**, **Spike-Timing Dependent Plasticity (STDP)**, and dynamic **Neurochemistry** to learn continuously in real time. It is designed to act as the unified cognitive substrate for autonomous synthetic digital species.

📄 **Read the complete academic paper:** [Biologically Inspired Brain (BIB): A 1:1 Neurobiologically Grounded Architecture for Continuous Learning in Autonomous Digital Organisms](PAPER.md).

---

## 🌟 Key Features

* **No Backpropagation or Epochs:** Learns continuously and asynchronously. There is no artificial split between "training" and "inference."
* **Mathematical Resistance to Catastrophic Forgetting:** Operates on 4,096-column Sparse Distributed Representations with exactly 32 active columns (0.78% active sparsity). Orthogonal hyper-dimensional projection ensures new learning does not overwrite older memories.
* **5 Dynamic Neuromodulators & HPA Axis:**
  * **Dopamine (DA):** Encodes Reward Prediction Error (RPE) via Ventral Tegmental Area (VTA) and Substantia Nigra (SNc) dynamics for reinforcement.
  * **Acetylcholine (ACh):** Encodes novelty and surprise (Nucleus Basalis of Meynert), scaling causal cortical plasticity (LTP).
  * **Serotonin (5-HT):** Modulates behavioral patience, emotional tone, and scales striatal NoGo pathway inhibition.
  * **Norepinephrine (NE):** Controls attentional gain and thalamic sensory gating via the Locus Coeruleus (LC).
  * **Cortisol:** Regulates HPA axis stress responses and suppresses hippocampal CA3 binding under prolonged deficit.
* **8-Organ Integrated Neuro-Architecture:** Full macro-structural replication of Thalamus, 6 Modal Neocortices, Association Cortex, Prefrontal Cortex, Hippocampus, Amygdala, Basal Ganglia, Cerebellum, Brainstem, and Hypothalamus.
* **Basal Ganglia with $\text{TD}(\lambda)$:** Dual-pathway D1 (Go) and D2 (NoGo) striatal actor-critic model with eligibility traces ($\lambda = 0.70$) for multi-step credit assignment.
* **Cerebellar Motor Forward Model:** Continuously generates sensory forward predictions to correct motor execution errors.
* **3-Stage Sleep Memory Consolidation:** Features an authentic biological sleep cycle (NREM $\to$ Slow-Wave Sleep [SWS] with Sharp-Wave Ripples $\to$ REM), transferring episodic CA3 traces into cortical long-term weights and executing synaptic homeostasis downscaling (Tononi SHY).
* **Consumer CPU Performance (Zero GPU Required):** JIT-compiled through NumPy and Numba, sustaining a full 8-organ cycle at 60 FPS on a single CPU thread with constant $O(1)$ memory footprint.

---

## 🧠 Macro-Architecture

BIB models the mammalian brain pipeline through synchronous update ticks ($\Delta t$):

```
                      +------------------------------------------+
                      |         Sensory Environment              |
                      | (Vision, Touch, Proprio, Intero, etc.)   |
                      +--------------------+---------------------+
                                           | Raw Floats (528-dim)
                                           v
+-----------------------------------------------------------------------------------------+
|                                    THALAMUS                                             |
|              NE-Gated Spatial Pooler & Competitive Winner-Take-All                      |
|                       Output: 2,048-bit SDR (0.78% Sparsity)                            |
+------------------------------------------+----------------------------------------------+
                                           |
                    +----------------------+----------------------+
                    |                                             |
                    v                                             v
+----------------------------------------+   +--------------------------------------------+
|         MODAL NEOCORTICES (x6)         |   |                  AMYGDALA                  |
| L1 (0.80) -> L2 (0.95) -> L3 (0.99)    |   | Salience Gating, Threat & Fear Value       |
| Predictive Coding & Apical Bias        |   | Conditioning (BLA/CeA Nuclei)              |
+-------------------+--------------------+   +---------------------+----------------------+
                    |                                              |
                    v                                              |
+----------------------------------------+                         |
|           ASSOCIATION CORTEX           |                         |
| Multi-Modal Fusion -> World-State SDR  |                         |
+-------------------+--------------------+                         |
                    |                                              |
     +--------------+--------------+                               |
     |                             |                               |
     v                             v                               |
+-----------------------+  +----------------------+                |
|   PREFRONTAL CORTEX   |  |     HIPPOCAMPUS      |                |
| Working Memory (k=4)  |  | DG -> CA3 -> CA1     |                |
| Top-Down Goal Signals |  | Attractor Retrieval  |                |
+-----------+-----------+  +-----------+----------+                |
            |                          |                           |
            +------------+-------------+                           |
                         |                                         |
                         v                                         v
+-----------------------------------------------------------------------------------------+
|                                  BASAL GANGLIA                                          |
|            Striatal Competition: D1 (Go) vs D2 (NoGo) Pathways                          |
|             Actor-Critic Reinforcement with TD(lambda) Traces                           |
+------------------------------------+----------------------------------------------------+
                                     | Motor Output Proposal
                                     v
+-----------------------------------------------------------------------------------------+
|                                   CEREBELLUM                                            |
|                 Forward Predictive Model & Error Adjustment Correction                  |
+------------------------------------+----------------------------------------------------+
                                     | Final Selected Action (0 - 7)
                                     v
                      +------------------------------------------+
                      |               Motor Action               |
                      |   (Movement, Nutrition, Mating, etc.)    |
                      +------------------------------------------+
```

1. **Thalamus:** Active gatekeeper compressing 528 raw sensory floats into 2,048-bit SDRs via competitive Winner-Take-All inhibition and Locus Coeruleus NE gain.
2. **Neocortex (L1, L2, L3):** 6 hierarchical sensory cortices + 1 cross-modal Association Cortex operating with local STDP plasticity and top-down apical feedback.
3. **Prefrontal Cortex (PFC):** Maintains 4 working memory slots and broadcasts emergent goal representations.
4. **Hippocampus (DG, CA3, CA1, EC):** Fast-learning episodic attractor network; DG pattern separation, CA3 Hopfield auto-association, and CA1 mismatch detection.
5. **Amygdala:** Real-time fear-conditioning, threat salience, and emotional valence attribution.
6. **Basal Ganglia:** Go (D1) / NoGo (D2) striatal actor-critic engine with $\text{TD}(\lambda)$ eligibility traces.
7. **Cerebellum:** Forward predictive model calculating motor prediction errors for fine motor adjustment.
8. **Brainstem & Hypothalamus:** Neuromodulator engine ($\text{DA}, \text{5-HT}, \text{ACh}, \text{NE}, \text{Cortisol}$) driving hunger, thirst, fatigue, pain, and sleep onset.
9. **Sleep Orchestrator:** 3-stage NREM, SWS (Sharp-Wave Ripples, synaptic downscaling $\times 0.97$), and REM consolidation.

---

## 🔬 Input / Output Specifications

### 1. Sensory Compression (528 Input Dimensions)
Every tick, the organism's body compiles sensory readings into a dictionary of 6 modalities:
* **Vision (`vision`):** 256-dimensional float vector (retinal / FOV raycasts).
* **Touch (`touch`):** 64-dimensional float vector (surface tactile & boundary sensors).
* **Proprioception (`proprioception`):** 32-dimensional float vector (body velocity, angle, joint positions).
* **Chemoreception (`chemoreception`):** 32-dimensional float vector (environmental pheromones/scent).
* **Interoception (`interoception`):** 16-dimensional float vector (hunger, hydration, waste, internal energy).
* **Auditory (`auditory`):** 128-dimensional float vector (spectral frequency bands).

### 2. MRS GREN Biological Action Primitives
The Basal Ganglia maps internal and cortical states into one of 8 universal biological motor actions:

| Action Index | Action Name | Biological Life Process | Description |
| :---: | :--- | :--- | :--- |
| **0** | `MOVEMENT` | Locomotion | Forward velocity displacement |
| **1** | `REPRODUCTION` | Reproduction | Declares mating intent / gamete transmission |
| **2** | `SENSITIVITY` | Irritability | Rotates attention heading and angular orientation |
| **3** | `GROWTH` | Nutrition / Growth | Allocates biomass to physical expansion & capacity |
| **4** | `EXCRETION` | Excretion | Eliminates metabolic waste |
| **5** | `NUTRITION` | Nutrition | Ingests adjacent food or water |
| **6** | `IDLE` | Homeostasis | Conserves energy (cuts metabolic burn by 50%) |
| **7** | `COMMUNICATE` | Social signaling | Emits communicative chemical/auditory signals |

---

## 📦 Installation

BIB requires Python 3.10+ and relies on `numpy` and `numba` for JIT-accelerated tensor execution:

```bash
git clone https://github.com/tgakathunderr/bib.git
cd bib
pip install numpy numba
```

---

## ⚡ Quick Start

```python
import numpy as np
from bib.brain import BIB

# 1. Initialize the 8-organ brain
brain = BIB()

# 2. Construct raw sensory vectors (528 float dimensions)
sensors = {
    "vision": np.zeros(256, dtype=np.float32),
    "touch": np.zeros(64, dtype=np.float32),
    "proprioception": np.zeros(32, dtype=np.float32),
    "interoception": np.zeros(16, dtype=np.float32),
    "chemoreception": np.zeros(32, dtype=np.float32),
    "auditory": np.zeros(128, dtype=np.float32)
}

# 3. Report the body's homeostatic drives
homeostatic_state = {
    "hunger": 0.4,
    "thirst": 0.2,
    "fatigue": 0.1,
    "pain": 0.0,
    "energy": 0.8
}

# 4. Tick the brain (returns MRS GREN action index 0-7)
action_idx = brain.tick(sensors, reward=0.0, homeostatic_state=homeostatic_state)
print(f"Selected Action: {action_idx}")

# 5. Check if the organism requires sleep consolidation
if brain.needs_sleep():
    sleep_report = brain.sleep()
    print("Sleep completed. Consolidating memories into cortex:", sleep_report)

# 6. Read real-time neurochemistry
chemistry = brain.get_chemistry()
print(f"Dopamine (DA): {chemistry['DA']:.3f} | Serotonin (5-HT): {chemistry['5HT']:.3f}")
```

---

## 🌐 Ecosystem Implementations

The BIB cognitive substrate powers the synthetic biology projects across UnikAI Lab:

* **[Project Big Bang](https://github.com/tgakathunderr/project_big_bang)**: An open-source, continuous 2D evolutionary sandbox where organisms learn to forage, mate, and adapt across seasons via Hebbian plasticity and natural selection. Read the [Project Big Bang Story](https://www.unikai.in/blog/project-big-bang).
* **[Project Cambrian](https://github.com/tgakathunderr/project_cambrian)**: A standalone desktop application (v1.0.0) simulating a living digital terrarium with live 8-organ telemetry, terrain painting, life biographies, and lineage family trees. Read the [Project Cambrian Announcement](https://www.unikai.in/blog/project-cambrian) or visit the [Cambrian Release Site](https://cambrian.unikai.in).

---

## 📚 Theoretical Foundations

BIB is directly grounded in peer-reviewed neuroscience literature:
* **Schultz (1997)**: Dopamine neurons encode reward prediction error (*Science*).
* **Bi & Poo (1998)**: Synaptic modifications in cultured hippocampal neurons: dependence on spike timing (*J. Neurosci*).
* **Frank, Seeberger & O'Reilly (2004)**: By carrot or by stick: cognitive reinforcement learning in parkinsonism (D1/D2 model) (*Science*).
* **Tononi & Cirelli (2006)**: Sleep and synaptic homeostasis hypothesis (*Brain Res. Bull.*).
* **Sherman & Guillery (2006)**: *Exploring the Thalamus and Its Role in Cortical Function* (MIT Press).
* **Friston (2010)**: The free-energy principle: a unified brain theory? (*Nat. Rev. Neurosci.*).
* **Rolls (2013)**: The mechanisms for pattern completion and pattern separation in the hippocampus (*Front. Syst. Neurosci.*).
* **Sutton & Barto (1998)**: *Reinforcement Learning: An Introduction* (MIT Press).

---

## 📖 Citation

If you use BIB in your research or applications, please cite:

```bibtex
@article{uniki_bib_2026,
  title   = {Biologically Inspired Brain (BIB): A 1:1 Neurobiologically Grounded Architecture for Continuous Learning in Autonomous Digital Organisms},
  author  = {{UnikAI Lab}},
  journal = {UnikAI Lab Technical Reports},
  year    = {2026},
  url     = {https://github.com/tgakathunderr/bib}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
