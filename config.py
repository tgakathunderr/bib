"""
BIB: Biologically Inspired Brain — Configuration
=================================================
All constants are grounded in real neuroscience.
Citations are provided per constant group.

Nothing is hard-coded in module bodies. Every tunable parameter lives here.
"""

from __future__ import annotations
import numpy as np

# ---------------------------------------------------------------------------
# SDR Geometry
# Mountcastle (1978) — cortical column theory; Maass (2000) — sparse coding
# 0.39% sparsity: probability of interference ≈ C(N,W)^-1 ≈ 10^-70 (BIM 1 proof)
# ---------------------------------------------------------------------------
SDR_SIZE: int = 4_096  # columns per cortical region
SDR_SPARSITY: int = 32  # active columns per representation (0.78%)
CELLS_PER_COLUMN: int = 4  # minicolumn depth
MAX_SYNAPSES: int = 16  # distal synapses per cell
SYNAPSE_THRESHOLD: float = 0.50  # permanence threshold for connection
SYNAPSE_INITIAL: float = 0.51  # just above threshold on creation
TOTAL_CELLS: int = SDR_SIZE * CELLS_PER_COLUMN  # 16,384 cells per region

# ---------------------------------------------------------------------------
# STDP — Spike-Timing Dependent Plasticity
# Bi & Poo (1998) — first empirical STDP characterization in hippocampal neurons
# Song, Miller & Abbott (2000) — competitive STDP model
# ---------------------------------------------------------------------------
STDP_A_PLUS: float = 0.20  # LTP amplitude (causal: pre before post)
STDP_A_MINUS: float = 0.10  # LTD amplitude (anti-causal: post before pre)
STDP_TAU_PLUS: float = 20.0  # LTP time constant (ticks, ~20ms biological)
STDP_TAU_MINUS: float = 40.0  # LTD time constant (ticks, ~40ms biological)
STDP_TAU_WINDOW: int = 10  # max tick delta where STDP has effect
SYNAPSE_PERM_MIN: float = 0.0
SYNAPSE_PERM_MAX: float = 1.0

# ---------------------------------------------------------------------------
# Hierarchical Cortex — 3 layers per neocortex instance
# Rao & Ballard (1999) — predictive coding in visual cortex
# Friston (2010) — free energy principle & hierarchical predictive coding
# ---------------------------------------------------------------------------
N_CORTICAL_LAYERS: int = 3
LAYER_DECAY_RATES: tuple[float, ...] = (0.80, 0.95, 0.99)
#   L1 (primary):    fast, byte/sample-level patterns     (decay 0.80)
#   L2 (secondary):  medium, feature/word-level patterns  (decay 0.95)
#   L3 (association): slow, abstract/context patterns     (decay 0.99)
APICAL_SOURCE_LAYERS: tuple[int, ...] = (1, 2)  # layers that project top-down
APICAL_BIAS_THRESHOLD: float = 0.05

# ---------------------------------------------------------------------------
# Multi-Modal Thalamus
# Sherman & Guillery (2006) — thalamus as active gatekeeper, not relay
# Aston-Jones & Cohen (2005) — LC-NE system and thalamic gain control
# ---------------------------------------------------------------------------
MODALITY_NAMES: tuple[str, ...] = (
    "vision",
    "touch",
    "proprioception",
    "chemoreception",
    "interoception",
    "auditory",
)
N_MODALITIES: int = len(MODALITY_NAMES)  # 6 sensory channels

# Per-modality input float vector dimensions (raw sensor readings)
MODALITY_INPUT_DIMS: dict[str, int] = {
    "vision": 256,  # e.g. flattened pixel patch or retinal cell activations
    "touch": 64,  # pressure + temperature sensors across body surface
    "proprioception": 32,  # joint angles, velocity, acceleration
    "chemoreception": 32,  # olfactory + taste receptor activations
    "interoception": 16,  # hunger, thirst, pain, arousal, heart-rate analog
    "auditory": 128,  # frequency-band energy spectrum
}

NE_GAIN_MIN: float = 0.5  # column activation gain under low NE (drowsy)
NE_GAIN_MAX: float = 2.0  # column activation gain under high NE (alert)
THAL_SUPPRESSION: float = 0.4  # how much top-down prediction cancels bottom-up

# ---------------------------------------------------------------------------
# Hippocampus — DG → CA3 → CA1 → EC
# O'Keefe & Dostrovsky (1971) — place cells; Rolls (2013) — CA3 attractor theory
# McClelland et al. (1995) — complementary learning systems (hippocampus vs cortex)
# Marr (1971) — archicortex as pattern separator (DG) + completer (CA3)
# ---------------------------------------------------------------------------
HIPPO_DG_SIZE: int = 4_096  # Dentate Gyrus
HIPPO_DG_SPARSITY: int = 8  # ~0.2% active
HIPPO_CA3_SIZE: int = 2_048  # CA3 — Hopfield attractor network
HIPPO_CA3_SPARSITY: int = 64  # ~3% active
HIPPO_CA1_SIZE: int = 1_024  # CA1 — mismatch detector
HIPPO_CA1_SPARSITY: int = 32  # ~3% active
HIPPO_CA3_LR: float = 0.50  # CA3 Hebbian binding rate
HIPPO_CA3_LTD: float = 0.02  # CA3 LTD rate
HIPPO_RETRIEVE_ITER: int = 4  # CA3 attractor settle iterations (fewer for speed)
HIPPO_BIND_THRESHOLD: float = 0.60  # surprise threshold to trigger binding
HIPPO_RENORM_FACTOR: float = 0.97  # SWS synaptic renormalization (Tononi 2006)
HIPPO_WEIGHT_MAX: float = 1.0
EC_TIME_WINDOW: int = 64  # Entorhinal Cortex time-cell buffer length

# CA3 Hopfield capacity: ~0.14 × N ≈ 1147 patterns. Renorm prevents saturation.
CORTISOL_HIPPO_SUPPRESS: float = 0.8  # at max cortisol, binding multiplied by 0.2

# ---------------------------------------------------------------------------
# Amygdala — Basolateral (BLA) + Central (CeA) nuclei
# LeDoux (2000) — amygdala and fear conditioning
# Phelps & LeDoux (2005) — contributions of BLA to emotional memory
# ---------------------------------------------------------------------------
AMYGDALA_SDR_SIZE: int = SDR_SIZE
AMYGDALA_VALENCE_LR: float = 0.30
AMYGDALA_EXTINCTION_RATE: float = 0.001
AMYGDALA_NE_THRESHOLD: float = 0.70
AMYGDALA_HIPPO_THRESHOLD: float = 0.40

# ---------------------------------------------------------------------------
# Brainstem Neuromodulator Nuclei
# Schultz (1997) — dopamine neurons encode reward prediction error (Nobel basis)
# Aston-Jones & Cohen (2005) — LC-NE adaptive gain theory
# Dayan & Huys (2009) — serotonin and patience/inhibition
# Hasselmo (2006) — acetylcholine and cortical network dynamics
# ---------------------------------------------------------------------------

# Dopamine — Ventral Tegmental Area (VTA) + Substantia Nigra pars compacta (SNc)
BASE_DA: float = 0.0
MAX_DA: float = 1.0
MIN_DA: float = -1.0
DA_DECAY: float = 0.90  # fast decay (DA signals are phasic)

# Norepinephrine — Locus Coeruleus (LC)
BASE_NE: float = 0.10
MAX_NE: float = 1.0
NE_DECAY: float = 0.92

# Serotonin — Dorsal Raphe Nucleus (DRN)
BASE_5HT: float = 0.50
MAX_5HT: float = 1.0
HT5_DECAY: float = 0.98  # slow tonic modulator

# Acetylcholine — Nucleus Basalis of Meynert (NBM) + Septal nucleus
BASE_ACH: float = 0.10
MAX_ACH: float = 1.0
ACH_DECAY: float = 0.95
ACH_SURPRISE_GAIN: float = 0.50  # how much surprise elevates ACh

# Cortisol — HPA Axis (Hypothalamus → Pituitary → Adrenal cortex)
# McEwen (2007) — glucocorticoids and hippocampal vulnerability
BASE_CORTISOL: float = 0.0
MAX_CORTISOL: float = 1.0
CORTISOL_RISE: float = 0.0002  # slow rise under sustained stress
CORTISOL_DECAY: float = 0.0001  # very slow — stress hormones linger
CORTISOL_THRESHOLD: float = 0.70  # deficit level that triggers HPA axis

# ---------------------------------------------------------------------------
# Hypothalamus — Allostatic Drives + HPA Axis
# McEwen (1998) — allostasis and allostatic load
# Saper et al. (2005) — hypothalamus and sleep–wake switch
# ---------------------------------------------------------------------------
HUNGER_RATE: float = 0.0001  # per tick (without eating)
THIRST_RATE: float = 0.00015  # per tick (water more urgent than food)
FATIGUE_RATE: float = 0.00005  # per tick (slow accumulation)
PAIN_DECAY: float = 0.998  # pain load decays slowly per tick
SLEEP_THRESHOLD: float = 0.85  # fatigue above this → needs_sleep() = True
HOMEOSTATIC_WEIGHTS: tuple[float, ...] = (0.30, 0.35, 0.25, 0.10)
# Weighted deficit = 0.30×hunger + 0.35×thirst + 0.25×fatigue + 0.10×pain

# ---------------------------------------------------------------------------
# Prefrontal Cortex — Working Memory & Goal Representation
# Fuster (2001) — PFC and working memory
# Miller & Cohen (2001) — integrative theory of PFC function
# ---------------------------------------------------------------------------
PFC_WORKING_MEMORY_K: int = 4  # slots of L3 SDR history
PFC_GOAL_LR: float = 0.10
PFC_GOAL_DECAY: float = 0.001
PFC_APICAL_GAIN: float = 0.20

# ---------------------------------------------------------------------------
# Basal Ganglia — Go / NoGo Pathways
# Frank et al. (2004) — D1/D2 model of BG action selection
# Sutton & Barto (1998) — TD learning; Mnih et al. (2015) — eligibility traces
# ---------------------------------------------------------------------------
N_ACTIONS: int = 8  # 7 MRS GREN + Idle
GAMMA: float = 0.95  # temporal discount factor
LR_ACTOR: float = 0.05  # D1 Go pathway learning rate
LR_NOGO: float = 0.03  # D2 NoGo pathway learning rate
LR_CRITIC: float = 0.10  # Value function (Critic) learning rate
LAMBDA_TRACE: float = 0.70  # eligibility trace decay (TD-λ)
BABBLE_THRESHOLD: float = 0.10  # confidence floor before exploration
BABBLE_RATE_BASE: float = 0.10  # base random exploration probability
SEROTONIN_NOGO_BIAS: float = 0.30  # 5-HT scaling of NoGo pathway strength
CURIOSITY_ACH_BONUS: float = 0.20  # bonus for high-ACh novel-state actions

# Action index → MRS GREN life process mapping
ACTION_NAMES: tuple[str, ...] = (
    "MOVEMENT",  # 0 — locomotion, physical displacement
    "REPRODUCTION",  # 1 — mating, cell division signals
    "SENSITIVITY",  # 2 — heightened sensory attention mode
    "GROWTH",  # 3 — anabolic signal: allocate energy to growth
    "EXCRETION",  # 4 — waste removal signal
    "NUTRITION",  # 5 — eat / drink (ingest)
    "IDLE",  # 6 — do nothing, conserve energy
    "COMMUNICATE",  # 7 — social signal / pheromone release
)
# NOTE: RESPIRATION is always-on background (brainstem/medulla), never in BG.

# ---------------------------------------------------------------------------
# Cerebellum — Forward Model
# Wolpert & Kawato (1998) — MOSAIC model of motor learning via cerebellum
# Ito (2008) — the cerebellum and adaptive control
# ---------------------------------------------------------------------------
CEREB_FORWARD_LR: float = 0.10  # forward model learning rate
CEREB_ERROR_SCALE: float = 0.30  # how much cerebellum adjusts BG output
CEREB_PREDICTION_DECAY: float = 0.95  # temporal pooling in forward model

# ---------------------------------------------------------------------------
# Memory — Prioritized Experience Replay
# Schaul et al. (2016) — prioritized experience replay (PER)
# O'Neill et al. (2010) — hippocampal replay and memory consolidation
# ---------------------------------------------------------------------------
PER_CAPACITY: int = 2_000
PER_ALPHA: float = 0.6  # priority exponent (0=uniform, 1=full priority)
PER_BETA: float = 0.4  # importance-sampling correction
PER_EPSILON: float = 0.01  # floor priority (prevents zero-probability episodes)

# ---------------------------------------------------------------------------
# Sleep Architecture
# Tononi & Cirelli (2006) — Synaptic Homeostasis Hypothesis (SHY)
# Stickgold (2005) — sleep-dependent memory consolidation
# Hobson & McCarley (1977) — activation-synthesis hypothesis of REM
# Buzsáki (1989) — two-stage model of memory consolidation (hippocampus → cortex)
# ---------------------------------------------------------------------------
SLEEP_N1N2_FRACTION: float = 0.15
SLEEP_SWS_FRACTION: float = 0.45
SLEEP_REM_FRACTION: float = 0.40

# Chemistry during sleep stages (Hobson 2002)
SWS_ACH: float = 0.05  # ACh is LOW during SWS (NBM suppressed)
SWS_NE: float = 0.00  # LC silent during SWS
SWS_5HT: float = 0.15  # Raphe mostly quiet during SWS
REM_ACH: float = 0.95  # ACh is HIGH during REM (NBM fully active)
REM_NE: float = 0.00  # LC completely silent during REM
REM_5HT: float = 0.00  # Raphe completely silent during REM

SWS_N_REPLAYS: int = 300  # sharp-wave ripple replay events during SWS
REM_N_CHAINS: int = 80  # CA3 self-association chains during REM
RENORM_FACTOR: float = 0.97  # SWS synaptic downscaling (Tononi SHY)

# ---------------------------------------------------------------------------
# Data Types (Numba JIT compatibility)
# ---------------------------------------------------------------------------
SDR_DTYPE = np.uint16  # SDR indices: 0–16383 fits in uint16
PERM_DTYPE = np.float32  # permanences: float32 sufficient
WEIGHT_DTYPE = np.float32  # weight matrices
TIME_DTYPE = np.int64  # tick timestamps
