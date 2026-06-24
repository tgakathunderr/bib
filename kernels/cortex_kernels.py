"""
BIB Kernels: Cortex — Prediction Generation
============================================
Generates predictive cell sets from current winner cells via synapse traversal.
Applies apical bias from higher layers / hippocampus.
"""
from __future__ import annotations
import numpy as np
from numba import njit


@njit(cache=True)
def generate_predictions_jit(
    winner_indices: np.ndarray,      # int64[:] — currently active winner cells
    connected_targets: np.ndarray,   # int32[:,:]
    permanences: np.ndarray,         # float32[:,:]
    synapse_counts: np.ndarray,      # int32[:]
    total_cells: int,
    threshold: float,
) -> np.ndarray:
    """
    Forward-pass: for each winner cell, follow synapses above threshold
    and mark their targets as predictive for the next timestep.
    Returns bool mask of shape (total_cells,).
    """
    predictive = np.zeros(total_cells, dtype=np.bool_)
    for cell in winner_indices:
        n = synapse_counts[cell]
        for s in range(n):
            target = connected_targets[cell, s]
            if target >= 0 and permanences[cell, s] >= threshold:
                predictive[target] = True
    return predictive


@njit(cache=True)
def apply_apical_bias_jit(
    predictive_cells: np.ndarray,   # bool[:] — modified in-place
    bias_cols: np.ndarray,          # int64[:] — column indices to bias
    winner_cells: np.ndarray,       # bool[:] — current winners (to find cells)
    cells_per_column: int,
    bias_threshold: float,
    bias_vals: np.ndarray,          # float32[:] — bias strength per column
) -> None:
    """
    Top-down apical feedback: if a column's bias exceeds threshold,
    mark winner cells in that column as predictive (top-down prediction).
    Biologically: layer-6 feedback modulates layer-1 receptive fields.
    """
    for col_idx in bias_cols:
        if bias_vals[col_idx] < bias_threshold:
            continue
        start = col_idx * cells_per_column
        for k in range(cells_per_column):
            c = start + k
            if winner_cells[c]:
                predictive_cells[c] = True


@njit(cache=True)
def get_predictive_columns_jit(
    predictive_cells: np.ndarray,   # bool[:]
    cells_per_column: int,
    n_columns: int,
) -> np.ndarray:
    """Extract unique column indices that have at least one predictive cell."""
    col_mask = np.zeros(n_columns, dtype=np.bool_)
    for cell in range(len(predictive_cells)):
        if predictive_cells[cell]:
            col_mask[cell // cells_per_column] = True
    count = 0
    for c in range(n_columns):
        if col_mask[c]:
            count += 1
    result = np.empty(count, dtype=np.int64)
    idx = 0
    for c in range(n_columns):
        if col_mask[c]:
            result[idx] = c
            idx += 1
    return result


@njit(cache=True)
def top_k_indices_jit(pool: np.ndarray, k: int) -> np.ndarray:
    """
    Return indices of top-k values in pool using partial selection (O(N)).
    Used for temporal pooling between cortical layers.
    """
    n = len(pool)
    if n == 0 or k == 0:
        return np.empty(0, dtype=np.int64)
    k = min(k, n)
    # Partial sort: find k-th largest threshold
    threshold = np.partition(pool, n - k)[n - k]
    result = np.empty(k, dtype=np.int64)
    count = 0
    for i in range(n):
        if pool[i] >= threshold and count < k:
            result[count] = i
            count += 1
    return result[:count]
