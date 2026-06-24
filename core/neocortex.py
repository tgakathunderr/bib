"""
BIB Core: NeocortexInstance
============================
A single sensory cortical hierarchy with 3 layers (L1→L2→L3).
Instantiated 7 times: once per sensory modality (6) + Association Cortex (1).

Biological basis:
  Mountcastle (1978) — cortical column as fundamental processing unit
  Rao & Ballard (1999) — predictive coding in visual cortex
  Friston (2010) — hierarchical predictive coding / free energy principle
  Bi & Poo (1998) — STDP characterization

Plasticity: real STDP with per-cell last-fire timestamps.
No BILM code is used here — built from scratch.
"""
from __future__ import annotations

import numpy as np

from bib.config import (
    APICAL_BIAS_THRESHOLD,
    APICAL_SOURCE_LAYERS,
    CELLS_PER_COLUMN,
    LAYER_DECAY_RATES,
    MAX_SYNAPSES,
    N_CORTICAL_LAYERS,
    SDR_DTYPE,
    SDR_SIZE,
    SDR_SPARSITY,
    STDP_A_MINUS,
    STDP_A_PLUS,
    STDP_TAU_MINUS,
    STDP_TAU_PLUS,
    STDP_TAU_WINDOW,
    SYNAPSE_INITIAL,
    SYNAPSE_THRESHOLD,
    SYNAPSE_PERM_MAX,
    SYNAPSE_PERM_MIN,
    TIME_DTYPE,
    TOTAL_CELLS,
)
from bib.kernels.stdp_kernels import (
    grow_synapses_jit,
    select_winner_cell_jit,
    stdp_update_jit,
    update_last_fire_times_jit,
)
from bib.kernels.cortex_kernels import (
    apply_apical_bias_jit,
    generate_predictions_jit,
    get_predictive_columns_jit,
    top_k_indices_jit,
)


class CorticalLayer:
    """
    One layer of a neocortex instance.
    Maintains sparse synapse arrays and STDP state per cell.

    Data structures (per layer):
      connected_targets  [TOTAL_CELLS, MAX_SYNAPSES]  int32  — synapse targets
      permanences        [TOTAL_CELLS, MAX_SYNAPSES]  float32
      synapse_counts     [TOTAL_CELLS]                int32
      active_cells       [TOTAL_CELLS]                bool
      winner_cells       [TOTAL_CELLS]                bool
      predictive_cells   [TOTAL_CELLS]                bool
      prev_winner_cells  [TOTAL_CELLS]                bool
      cell_usage         [TOTAL_CELLS]                int32  — for homeostatic fairness
      last_fire_time     [TOTAL_CELLS]                int64  — STDP timestamps
    """

    def __init__(self) -> None:
        self.connected_targets = np.full(
            (TOTAL_CELLS, MAX_SYNAPSES), -1, dtype=np.int32
        )
        self.permanences = np.zeros(
            (TOTAL_CELLS, MAX_SYNAPSES), dtype=np.float32
        )
        self.synapse_counts = np.zeros(TOTAL_CELLS, dtype=np.int32)

        self.active_cells    = np.zeros(TOTAL_CELLS, dtype=bool)
        self.winner_cells    = np.zeros(TOTAL_CELLS, dtype=bool)
        self.predictive_cells = np.zeros(TOTAL_CELLS, dtype=bool)
        self.prev_winner_cells = np.zeros(TOTAL_CELLS, dtype=bool)
        self.cell_usage      = np.zeros(TOTAL_CELLS, dtype=np.int32)

        # STDP state — per-cell last-fire timestamp
        self.last_fire_time  = np.full(TOTAL_CELLS, -999, dtype=TIME_DTYPE)
        self.current_tick: int = 0

    def step(
        self,
        active_cols: np.ndarray,   # int64[:] — SDR column indices entering this layer
        ach: float,                 # ACh level — gates STDP LTP amplitude
        learn: bool = True,
    ) -> float:
        """
        Process one SDR step through this cortical layer.
        Returns Jaccard surprise (0 = perfect prediction, 1 = total mismatch).
        """
        self.prev_winner_cells[:] = self.winner_cells
        self.active_cells.fill(False)
        self.winner_cells.fill(False)

        # ── 1. Surprise: predicted vs actual column overlap ──────────────────
        pred_cols_set = set(self.get_predictive_columns().tolist())
        actual_cols_set = set(int(c) for c in active_cols)
        if pred_cols_set or actual_cols_set:
            inter = len(pred_cols_set & actual_cols_set)
            union = len(pred_cols_set | actual_cols_set)
            surprise = 1.0 - (inter / union) if union > 0 else 1.0
        else:
            surprise = 0.0

        prev_winner_idx = np.where(self.prev_winner_cells)[0].astype(np.int64)

        # ── 2. Activate / select winner cells ────────────────────────────────
        for col_idx in active_cols:
            col_idx = int(col_idx)
            start = col_idx * CELLS_PER_COLUMN
            end = start + CELLS_PER_COLUMN
            predicted = self.predictive_cells[start:end]

            if predicted.any():
                # Predicted column: activate predicted cells (sequence memory)
                for k in range(CELLS_PER_COLUMN):
                    if predicted[k]:
                        self.active_cells[start + k] = True
                        self.winner_cells[start + k] = True
            else:
                # Unpredicted: select winner via best-match + least-used
                winner = select_winner_cell_jit(
                    start, end,
                    prev_winner_idx,
                    self.connected_targets,
                    self.permanences,
                    self.synapse_counts,
                    self.cell_usage,
                    SYNAPSE_THRESHOLD,
                )
                self.active_cells[winner] = True
                self.winner_cells[winner] = True
                self.cell_usage[winner] += 1

        curr_winner_idx = np.where(self.winner_cells)[0].astype(np.int64)

        # ── 3. STDP update ───────────────────────────────────────────────────
        if learn and len(prev_winner_idx) > 0:
            stdp_update_jit(
                prev_winner_idx,
                self.winner_cells,
                self.predictive_cells,
                self.connected_targets,
                self.permanences,
                self.synapse_counts,
                self.last_fire_time,
                self.current_tick,
                ach,                     # ACh gates LTP amplitude
                STDP_A_PLUS,
                STDP_A_MINUS,
                STDP_TAU_PLUS,
                STDP_TAU_MINUS,
                STDP_TAU_WINDOW,
                SYNAPSE_PERM_MIN,
                SYNAPSE_PERM_MAX,
            )
            # Grow new synapses for unpredicted winner cells
            for cell in curr_winner_idx:
                if not self.predictive_cells[cell]:
                    grow_synapses_jit(
                        int(cell),
                        prev_winner_idx,
                        self.connected_targets,
                        self.permanences,
                        self.synapse_counts,
                        MAX_SYNAPSES,
                        SYNAPSE_INITIAL,
                    )

        # ── 4. Update STDP timestamps ────────────────────────────────────────
        if learn and len(curr_winner_idx) > 0:
            update_last_fire_times_jit(
                curr_winner_idx,
                self.last_fire_time,
                self.current_tick,
            )

        # ── 5. Generate predictions for next timestep ────────────────────────
        if len(curr_winner_idx) > 0:
            self.predictive_cells = generate_predictions_jit(
                curr_winner_idx,
                self.connected_targets,
                self.permanences,
                self.synapse_counts,
                TOTAL_CELLS,
                SYNAPSE_THRESHOLD,
            )
        else:
            self.predictive_cells.fill(False)

        self.current_tick += 1
        return float(surprise)

    def apply_apical_bias(self, bias: np.ndarray) -> None:
        """Top-down apical feedback from higher layer or hippocampus."""
        high_bias_cols = np.where(bias > APICAL_BIAS_THRESHOLD)[0].astype(np.int64)
        if len(high_bias_cols) == 0:
            return
        apply_apical_bias_jit(
            self.predictive_cells,
            high_bias_cols,
            self.winner_cells,
            CELLS_PER_COLUMN,
            APICAL_BIAS_THRESHOLD,
            bias,
        )

    def get_predictive_columns(self) -> np.ndarray:
        return get_predictive_columns_jit(
            self.predictive_cells, CELLS_PER_COLUMN, SDR_SIZE
        )

    def reset_context(self) -> None:
        self.active_cells.fill(False)
        self.winner_cells.fill(False)
        self.predictive_cells.fill(False)
        self.prev_winner_cells.fill(False)

    def total_synapses(self) -> int:
        return int(self.synapse_counts.sum())


class NeocortexInstance:
    """
    A complete 3-layer predictive cortical hierarchy for one sensory modality
    (or the association cortex).

    Layers:
      L1: fast (decay 0.80) — primary sensory: sample/byte-level patterns
      L2: medium (decay 0.95) — secondary: feature/word-level patterns
      L3: slow (decay 0.99) — associative: abstract/context-level patterns

    L3 output is consumed by the Association Cortex (for modal cortices)
    or by the PFC (for the Association Cortex itself).

    Top-down apical feedback flows L3→L2→L1 after each bottom-up pass.
    """

    def __init__(self, name: str = "unnamed") -> None:
        self.name = name
        self.layers: list[CorticalLayer] = [
            CorticalLayer() for _ in range(N_CORTICAL_LAYERS)
        ]
        self.temporal_pools = np.zeros(
            (N_CORTICAL_LAYERS, SDR_SIZE), dtype=np.float32
        )
        self.decay_rates = np.asarray(LAYER_DECAY_RATES, dtype=np.float32)

    def step(
        self,
        l1_input_cols: np.ndarray,  # int64[:] — encoded SDR column indices for L1
        ach: float,
        learn: bool = True,
    ) -> float:
        """
        Full bottom-up + top-down cortical pass.
        Returns L1 surprise (primary sensory mismatch signal).
        """
        active_cols = np.asarray(l1_input_cols, dtype=np.int64)
        l1_surprise = 0.0

        # ── Bottom-up pass: L1 → L2 → L3 ────────────────────────────────────
        for i in range(N_CORTICAL_LAYERS):
            surprise = self.layers[i].step(active_cols, ach=ach, learn=learn)
            if i == 0:
                l1_surprise = surprise

            # Temporal pooling: accumulate + decay
            self.temporal_pools[i] *= self.decay_rates[i]
            if len(active_cols) > 0:
                self.temporal_pools[i][active_cols] += 1.0

            # Pass top-K pooled columns to next layer
            if i + 1 < N_CORTICAL_LAYERS:
                active_cols = top_k_indices_jit(
                    self.temporal_pools[i], SDR_SPARSITY
                )

        # ── Top-down apical feedback: L3→L2, L2→L1 ──────────────────────────
        for src in APICAL_SOURCE_LAYERS:
            dst = src - 1
            if src >= N_CORTICAL_LAYERS or dst < 0:
                continue
            pred_cols = self.layers[src].get_predictive_columns()
            if len(pred_cols) == 0:
                continue
            bias = self._build_apical_bias(src, pred_cols)
            self.layers[dst].apply_apical_bias(bias)

        # Subnormal guard
        self.temporal_pools[self.temporal_pools < 1e-30] = 0.0
        return l1_surprise

    def get_l3_sdr(self) -> np.ndarray:
        """L3 abstract SDR — consumed by Association Cortex or PFC."""
        return top_k_indices_jit(self.temporal_pools[2], SDR_SPARSITY)

    def get_l1_predictive_columns(self) -> np.ndarray:
        return self.layers[0].get_predictive_columns()

    def reset_context(self) -> None:
        for layer in self.layers:
            layer.reset_context()
        self.temporal_pools.fill(0.0)

    def total_synapses(self) -> list[int]:
        return [layer.total_synapses() for layer in self.layers]

    def _build_apical_bias(
        self, src_layer: int, src_pred_cols: np.ndarray
    ) -> np.ndarray:
        bias = np.zeros(SDR_SIZE, dtype=np.float32)
        pool = self.temporal_pools[src_layer]
        bias[src_pred_cols] = pool[src_pred_cols]
        max_val = float(bias.max())
        if max_val > 1e-8:
            bias /= max_val
        np.clip(bias, 0.0, 0.30, out=bias)
        return bias
