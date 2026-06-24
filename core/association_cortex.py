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
        self._l3_cache: np.ndarray = np.array([], dtype=np.int64)

    def step(
        self,
        modal_l3_sdrs: dict[str, np.ndarray],
        ach: float,
        learn: bool = True,
    ) -> tuple[np.ndarray, float]:
        """
        Fuse 6 modal L3 SDRs into one cross-modal representation.

        Uses Knuth multiplicative hash projection (zero extra memory):
        Each modality's active columns are hashed into [0, SDR_SIZE) with
        a unique per-modality offset to prevent hash collisions between modalities.
        Activation scores are accumulated; top-SDR_SPARSITY columns win.

        Args:
            modal_l3_sdrs: dict[modality → int64[:SDR_SPARSITY]]
            ach: acetylcholine level (gates STDP plasticity)
            learn: whether to apply STDP

        Returns:
            (unified_sdr, surprise)  —  int64[:SDR_SPARSITY], float
        """
        # Hash-project each modality's L3 SDR into shared column space
        activations = np.zeros(SDR_SIZE, dtype=np.float32)
        for i, name in enumerate(MODALITY_NAMES):
            sdr = modal_l3_sdrs.get(name, np.array([], dtype=np.int64))
            if len(sdr) == 0:
                continue
            offset = _MODALITY_HASH_OFFSETS[i]
            # Knuth hash: each SDR column → deterministic SDR_SIZE slot
            hashed = ((sdr.astype(np.int64) * offset) & 0xFFFFFFFF) % SDR_SIZE
            activations[hashed] += 1.0

        # Top-K column selection — competitive inhibition
        if activations.max() < 1e-8:
            l1_cols = np.arange(SDR_SPARSITY, dtype=np.int64)
        else:
            top_k = np.argpartition(activations, -SDR_SPARSITY)[-SDR_SPARSITY:]
            l1_cols = top_k.astype(np.int64)

        # Run through the association neocortex
        surprise = self.cortex.step(l1_cols, ach=ach, learn=learn)

        # Cache L3 output
        self._l3_cache = self.cortex.get_l3_sdr()
        return self._l3_cache, float(surprise)

    def get_world_state_sdr(self) -> np.ndarray:
        """Returns the last computed cross-modal L3 SDR."""
        return self._l3_cache

    def get_l1_predictive_columns(self) -> np.ndarray:
        return self.cortex.get_l1_predictive_columns()

    def apply_hippo_bias(self, bias: np.ndarray) -> None:
        """Hippocampal top-down apical feedback → association cortex L1."""
        self.cortex.layers[0].apply_apical_bias(bias)

    def reset_context(self) -> None:
        self.cortex.reset_context()
        self._l3_cache = np.array([], dtype=np.int64)
