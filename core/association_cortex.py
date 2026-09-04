"""
BIB Core: Association Cortex
==============================
The 7th neocortex instance — receives L3 outputs from all 6 modal cortices
and builds a unified cross-modal world-state representation.

Biological analog: parietal + temporal association cortex.
  Mesulam (1998) — from sensation to cognition: the role of association cortex
  Felleman & Van Essen (1991) — distributed hierarchical processing in primate cortex

Fusion mechanism: hash-based topographic projection (zero memory overhead).
Each modality's L3 SDR columns are hashed into the shared SDR_SIZE column
space using offset + Knuth hash, then competition selects top-SDR_SPARSITY.
No dense projection matrix needed — biologically, parietal convergence zones
use sparse topographic projections, not all-to-all dense weights.
"""
from __future__ import annotations

import numpy as np

from bib.config import (
    CELLS_PER_COLUMN,
    MODALITY_NAMES,
    N_MODALITIES,
    SDR_SIZE,
    SDR_SPARSITY,
)
from bib.core.neocortex import NeocortexInstance

# Unique prime offsets per modality for hash separation
_MODALITY_HASH_OFFSETS: tuple[int, ...] = (
    1779033703,   # vision
    2654435761,   # touch
    1844674407,   # proprioception
    3045258175,   # chemoreception
    2447445141,   # interoception
    3681393503,   # auditory
)


class AssociationCortex:
    """
    Cross-modal integration layer.
    Receives L3 SDR output from each of the 6 modal cortices.
    Produces a unified world-state SDR consumed by PFC and hippocampus.
    Memory: O(NeocortexInstance) only — no dense projection matrix.
    """

    def __init__(self) -> None:
        self.cortex = NeocortexInstance(name="association")
        self._world_state_cache: np.ndarray = np.array([], dtype=np.int64)

    def step(
        self,
        modal_sdrs: dict[str, np.ndarray],
        ach: float,
        learn: bool = True,
    ) -> tuple[np.ndarray, float]:
        """
        Fuse per-modality SDRs into one cross-modal world-state representation.

        Modalities are hashed into the shared column space, then the fused
        columns are run through the association neocortex for STDP sequence
        learning and surprise. The exported world-state is the association
        cortex's L1 live winner cells — the topographic, unsmoothed fusion —
        because deeper STDP layers monotonically erode the spatial
        discriminability the actor-critic depends on.

        Args:
            modal_sdrs: dict[modality → int64[:SDR_SPARSITY]] — per-modality
                        active column indices (raw or L1-level, not collapsed L3)
            ach: acetylcholine level (gates STDP plasticity)
            learn: whether to apply STDP

        Returns:
            (world_state_sdr, surprise)  —  int64[:SDR_SPARSITY], float
        """
        # Hash-project each modality into shared column space
        activations = np.zeros(SDR_SIZE, dtype=np.float32)
        for i, name in enumerate(MODALITY_NAMES):
            sdr = modal_sdrs.get(name, np.array([], dtype=np.int64))
            if len(sdr) == 0:
                continue
            offset = _MODALITY_HASH_OFFSETS[i]
            hashed = ((sdr.astype(np.int64) * offset) & 0xFFFFFFFF) % SDR_SIZE
            activations[hashed] += 1.0

        if activations.max() < 1e-8:
            l1_cols = np.arange(SDR_SPARSITY, dtype=np.int64)
        else:
            l1_cols = np.argpartition(activations, -SDR_SPARSITY)[-SDR_SPARSITY:].astype(np.int64)

        surprise = self.cortex.step(l1_cols, ach=ach, learn=learn)

        # World-state = cortical L1 winners (discriminative), not L3 (collapsed).
        winner = self.cortex.layers[0].winner_cells
        if winner.sum() == 0:
            world = l1_cols
        else:
            world = np.where(winner)[0] // CELLS_PER_COLUMN
        self._world_state_cache = world.astype(np.int64)
        return self._world_state_cache, float(surprise)

    def replay(
        self,
        sdr: np.ndarray,
        ach: float,
        learn: bool = True,
    ) -> float:
        surprise = self.cortex.step(sdr, ach=ach, learn=learn)
        self._world_state_cache = self.cortex.get_l3_sdr()
        return float(surprise)

    def get_world_state_sdr(self) -> np.ndarray:
        """Returns the last computed cross-modal world-state SDR."""
        return self._world_state_cache

    def get_l1_predictive_columns(self) -> np.ndarray:
        return self.cortex.get_l1_predictive_columns()

    def apply_hippo_bias(self, bias: np.ndarray) -> None:
        """Hippocampal top-down apical feedback → association cortex L1."""
        self.cortex.layers[0].apply_apical_bias(bias)

    def apply_pfc_bias(self, bias: np.ndarray) -> None:
        self.cortex.layers[0].apply_apical_bias(bias)

    def apply_temporal_bias(self, temporal_cols: np.ndarray, gain: float = 0.2) -> None:
        if len(temporal_cols) == 0:
            return
        bias = np.zeros(SDR_SIZE, dtype=np.float32)
        valid = temporal_cols[(temporal_cols >= 0) & (temporal_cols < SDR_SIZE)]
        if len(valid) > 0:
            bias[valid] = gain
        self.cortex.layers[0].apply_apical_bias(bias)

    def reset_context(self) -> None:
        self.cortex.reset_context()
        self._l3_cache = np.array([], dtype=np.int64)
