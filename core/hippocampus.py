"""
BIB Core: Hippocampus — DG → CA3 → CA1 → EC Pipeline
=======================================================
Biological basis:
  O'Keefe & Dostrovsky (1971) — place cells in hippocampus
  Marr (1971) — archicortex as pattern separator (DG) and completer (CA3)
  McClelland, McNaughton & O'Reilly (1995) — complementary learning systems
  Rolls (2013) — CA3 as auto-associative attractor network
  Hasselmo & Schnell (1994) — CA1 as novelty / mismatch detector
  Tononi & Cirelli (2006) — SWS synaptic renormalization (CA3 maintenance)
  McEwen (2007) — cortisol suppresses hippocampal encoding (stress → amnesia)

Pipeline:
  Cortical L3 SDR
    → [DG]  Pattern separation (extreme sparsity, orthogonalization)
    → [CA3] Hopfield attractor (binding + retrieval from partial cues)
    → [CA1] Mismatch detection (CA3 completion vs fresh EC input → novelty signal)
    → [EC]  Time-cell buffer (temporal context: when did this happen?)
    → Apical bias back → Association Cortex L1
"""
from __future__ import annotations

import numpy as np

from bib.config import (
    CORTISOL_HIPPO_SUPPRESS,
    EC_TIME_WINDOW,
    HIPPO_BIND_THRESHOLD,
    HIPPO_CA1_SIZE,
    HIPPO_CA1_SPARSITY,
    HIPPO_CA3_LR,
    HIPPO_CA3_LTD,
    HIPPO_CA3_SIZE,
    HIPPO_CA3_SPARSITY,
    HIPPO_DG_SIZE,
    HIPPO_DG_SPARSITY,
    HIPPO_RENORM_FACTOR,
    HIPPO_RETRIEVE_ITER,
    HIPPO_WEIGHT_MAX,
    SDR_SIZE,
    SDR_SPARSITY,
)
from bib.kernels.hippo_kernels import (
    ca1_mismatch_jit,
    ca3_hebbian_bind_jit,
    ca3_renormalize_jit,
    ca3_retrieve_jit,
    dg_project_jit,
)


class Hippocampus:
    """
    Full DG→CA3→CA1→EC hippocampal pipeline.

    State:
      ca3_weights  [CA3_SIZE, CA3_SIZE] float32   — Hopfield attractor matrix
      ec_buffer    [EC_TIME_WINDOW, SDR_SIZE] int8 — time-cell SDR ring buffer
      binds        int   — total CA3 bindings performed
      retrievals   int   — total CA3 retrievals performed
    """

    def __init__(self) -> None:
        # CA3 Hopfield weight matrix (256 MB)
        self.ca3_weights: np.ndarray = np.zeros(
            (HIPPO_CA3_SIZE, HIPPO_CA3_SIZE), dtype=np.float32
        )
        # Entorhinal Cortex time-cell buffer
        self.ec_buffer: np.ndarray = np.zeros(
            (EC_TIME_WINDOW, SDR_SIZE), dtype=np.int8
        )
        self._ec_cursor: int = 0

        # Statistics
        self.binds: int = 0
        self.retrievals: int = 0
        self._last_ca3_cells: np.ndarray = np.array([], dtype=np.int64)

    # ────────────────────────────────────────────────────────────────────────
    # DG: Pattern Separation
    # ────────────────────────────────────────────────────────────────────────
    def _dg_separate(self, cortex_cols: np.ndarray) -> np.ndarray:
        """
        Project cortical columns through Dentate Gyrus.
        Extreme sparsity (0.1%) ensures near-orthogonal DG representations,
        preventing interference between similar cortical inputs in CA3.
        """
        return dg_project_jit(
            cortex_cols.astype(np.int64), HIPPO_DG_SIZE, HIPPO_DG_SPARSITY
        )

    # ────────────────────────────────────────────────────────────────────────
    # CA3: Hopfield Attractor (Binding + Retrieval)
    # ────────────────────────────────────────────────────────────────────────
    def maybe_bind(
        self,
        dg_cells: np.ndarray,    # DG-separated pattern
        ach: float,              # ACh gates encoding (high ACh = encode mode)
        amygdala_salience: float,
        cortisol: float,
    ) -> bool:
        """
        Conditionally bind a DG pattern into CA3 memory.

        Binding occurs when:
          - ach is high (NBM active = encode mode, not retrieve mode)
          - amygdala_salience is above threshold (emotionally significant)
          - cortisol does NOT suppress the binding

        Cortisol suppression: McEwen (2007) — chronic stress → hippocampal atrophy.
        High cortisol reduces binding rate proportionally.
        """
        bind_threshold = HIPPO_BIND_THRESHOLD
        if amygdala_salience > 0.4:
            bind_threshold *= 0.7   # emotional salience lowers threshold

        if ach < bind_threshold:
            return False

        # Cortisol suppression: at max cortisol, binding × 0.2
        cortisol_factor = 1.0 - CORTISOL_HIPPO_SUPPRESS * cortisol
        effective_lr = HIPPO_CA3_LR * cortisol_factor

        if effective_lr < 0.05:
            return False   # cortisol completely suppressing encoding

        ca3_cells = self._dg_to_ca3(dg_cells)
        ca3_hebbian_bind_jit(
            self.ca3_weights, ca3_cells, effective_lr, HIPPO_CA3_LTD, HIPPO_WEIGHT_MAX
        )
        self.binds += 1
        self._last_ca3_cells = ca3_cells
        return True

    def retrieve(self, cortex_cols: np.ndarray) -> np.ndarray:
        """
        Pattern-complete from a partial cortical cue.
        DG orthogonalizes the cue → CA3 attractor settles → returns CA3 cells.
        """
        if len(cortex_cols) == 0:
            return np.array([], dtype=np.int64)
        dg_cue = self._dg_separate(cortex_cols)
        ca3_cue = self._dg_to_ca3(dg_cue)
        result = ca3_retrieve_jit(
            self.ca3_weights, ca3_cue,
            HIPPO_CA3_SIZE, HIPPO_CA3_SPARSITY, HIPPO_RETRIEVE_ITER
        )
        self._last_ca3_cells = result
        self.retrievals += 1
        return result

    # ────────────────────────────────────────────────────────────────────────
    # CA1: Mismatch Detection (novelty signal)
    # ────────────────────────────────────────────────────────────────────────
    def ca1_mismatch(self, ca3_cells: np.ndarray, cortex_cols: np.ndarray) -> float:
        """
        CA1 compares CA3 pattern completion against the raw EC input.
        High mismatch → novelty → NE spike + re-bind.
        Hasselmo & Schnell (1994): CA1 novelty = re-encoding signal.
        """
        # Project cortex cols to CA3 space for comparison
        ec_as_ca3 = self._dg_to_ca3(
            dg_project_jit(cortex_cols.astype(np.int64), HIPPO_DG_SIZE, HIPPO_DG_SPARSITY)
        )
        return float(ca1_mismatch_jit(ca3_cells, ec_as_ca3, HIPPO_CA3_SIZE, HIPPO_CA3_SPARSITY))

    # ────────────────────────────────────────────────────────────────────────
    # EC: Time-Cell Buffer
    # ────────────────────────────────────────────────────────────────────────
    def update_ec(self, cortex_cols: np.ndarray) -> np.ndarray:
        """
        Entorhinal Cortex time-cell ring buffer.
        Each slot = one timestep's cortical SDR.
        Provides temporal context: 'what happened T ticks ago'.
        Returns temporal_context SDR (union of recent episodes).
        """
        slot = np.zeros(SDR_SIZE, dtype=np.int8)
        for col in cortex_cols:
            if 0 <= col < SDR_SIZE:
                slot[col] = 1
        self.ec_buffer[self._ec_cursor] = slot
        self._ec_cursor = (self._ec_cursor + 1) % EC_TIME_WINDOW

        # Temporal context = union of recent slots (weighted by recency)
        context_sum = np.zeros(SDR_SIZE, dtype=np.float32)
        for i in range(EC_TIME_WINDOW):
            age = (self._ec_cursor - i - 1) % EC_TIME_WINDOW
            weight = float(np.exp(-i / 20.0))
            context_sum += self.ec_buffer[age].astype(np.float32) * weight

        # Top-K columns as temporal context SDR
        top_k = min(SDR_SPARSITY, SDR_SIZE)
        ctx_cols = np.argpartition(context_sum, -top_k)[-top_k:].astype(np.int64)
        return ctx_cols

    # ────────────────────────────────────────────────────────────────────────
    # Full pipeline tick
    # ────────────────────────────────────────────────────────────────────────
    def tick(
        self,
        cortex_cols: np.ndarray,    # Association Cortex L3 SDR
        ach: float,
        amygdala_salience: float,
        cortisol: float,
        retrieve: bool = True,
    ) -> tuple[np.ndarray, float, float]:
        """
        Run full DG→CA3→CA1→EC pass for one brain tick.

        Args:
            cortex_cols: current cortical SDR (association cortex L3)
            ach: acetylcholine (gates encoding)
            amygdala_salience: emotional significance
            cortisol: stress suppression factor
            retrieve: whether to also do CA3 pattern retrieval

        Returns:
            (temporal_context_sdr, ca1_novelty, ca3_apical_bias_vec)
            - temporal_context_sdr: EC temporal context for association cortex
            - ca1_novelty: 0-1 novelty (mismatch between CA3 completion and input)
            - hippo_bias: float32[SDR_SIZE] — apical bias for association cortex
        """
        if len(cortex_cols) == 0:
            empty = np.array([], dtype=np.int64)
            return empty, 0.0, np.zeros(SDR_SIZE, dtype=np.float32)

        dg_cells = self._dg_separate(cortex_cols)

        # CA3 retrieval: pattern complete from DG projection
        ca3_cells = np.array([], dtype=np.int64)
        ca1_novelty = 0.0
        if retrieve:
            ca3_cue = self._dg_to_ca3(dg_cells)
            ca3_cells = ca3_retrieve_jit(
                self.ca3_weights, ca3_cue,
                HIPPO_CA3_SIZE, HIPPO_CA3_SPARSITY, HIPPO_RETRIEVE_ITER
            )
            self.retrievals += 1

        # CA1: novelty signal
        if len(ca3_cells) > 0:
            ca1_novelty = self.ca1_mismatch(ca3_cells, cortex_cols)

        # CA3 binding (conditional)
        self.maybe_bind(dg_cells, ach, amygdala_salience, cortisol)

        # EC time cells
        temporal_ctx = self.update_ec(cortex_cols)

        # Build apical bias for association cortex
        hippo_bias = self._build_apical_bias(ca3_cells)

        return temporal_ctx, ca1_novelty, hippo_bias

    # ────────────────────────────────────────────────────────────────────────
    # SWS: Synaptic Renormalization (Tononi 2006)
    # ────────────────────────────────────────────────────────────────────────
    def sws_renormalize(self) -> None:
        """
        Apply synaptic downscaling to CA3 weight matrix during SWS.
        All weights × RENORM_FACTOR:
          - Weak attractors (occasional) fade → forgetting the unimportant
          - Strong attractors (repeated) survive → long-term memory
        """
        ca3_renormalize_jit(self.ca3_weights, HIPPO_RENORM_FACTOR)

    # ────────────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────────────
    def _dg_to_ca3(self, dg_cells: np.ndarray) -> np.ndarray:
        """Map DG cell indices to CA3 space via second hash."""
        if len(dg_cells) == 0:
            return np.empty(0, dtype=np.int64)
        ca3 = (dg_cells.astype(np.int64) * 1779033703) % HIPPO_CA3_SIZE
        unique = np.unique(ca3)
        if len(unique) >= HIPPO_CA3_SPARSITY:
            return unique[:HIPPO_CA3_SPARSITY]
        return unique

    def _build_apical_bias(self, ca3_cells: np.ndarray, gain: float = 0.2) -> np.ndarray:
        """Build apical bias for association cortex from CA3 retrieved cells."""
        bias = np.zeros(SDR_SIZE, dtype=np.float32)
        if len(ca3_cells) == 0:
            return bias
        # Reverse-map CA3 → cortex columns via inverse hash approximation
        cortex_approx = (ca3_cells * 2654435761) % SDR_SIZE
        bias[cortex_approx] = gain
        np.clip(bias, 0.0, 0.3, out=bias)
        return bias

    def report(self) -> dict:
        return {
            "binds": self.binds,
            "retrievals": self.retrievals,
            "ca3_max_weight": float(self.ca3_weights.max()),
            "ca3_mean_weight": float(self.ca3_weights.mean()),
        }
