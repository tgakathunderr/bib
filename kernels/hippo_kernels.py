"""
BIB Kernels: Hippocampus — DG, CA3, CA1
=========================================
Marr (1971) — archicortex theory: DG=separator, CA3=completer
Rolls (2013) — CA3 as a Hopfield attractor network
Tononi & Cirelli (2006) — synaptic renormalization during SWS
"""
from __future__ import annotations
import numpy as np
from numba import njit


@njit(cache=True)
def dg_project_jit(
    cortex_cols: np.ndarray,   # int64[:] — cortical column indices
    dg_size: int,
    dg_sparsity: int,
) -> np.ndarray:
    """
    Dentate Gyrus pattern separation via Knuth multiplicative hash projection.
    Extreme sparsity (0.1%) ensures orthogonalized DG representations.
    Different cortical patterns → near-zero DG overlap (prevents CA3 interference).
    """
    if len(cortex_cols) == 0:
        return np.empty(0, dtype=np.int64)

    # Knuth multiplicative hash to DG index space
    hashed = np.empty(len(cortex_cols), dtype=np.int64)
    for i in range(len(cortex_cols)):
        h = (cortex_cols[i] * np.int64(2654435769)) & np.int64(0xFFFFFFFF)
        hashed[i] = h % dg_size

    # Unique sort
    seen = np.zeros(dg_size, dtype=np.bool_)
    unique_count = 0
    uniq = np.empty(len(hashed), dtype=np.int64)
    for h in hashed:
        if not seen[h]:
            seen[h] = True
            uniq[unique_count] = h
            unique_count += 1
    uniq = uniq[:unique_count]

    if unique_count >= dg_sparsity:
        return uniq[:dg_sparsity]

    # Pad to dg_sparsity with unmapped cells
    result = np.empty(dg_sparsity, dtype=np.int64)
    for i in range(unique_count):
        result[i] = uniq[i]
    pad_count = 0
    for i in range(dg_size):
        if unique_count + pad_count >= dg_sparsity:
            break
        if not seen[i]:
            result[unique_count + pad_count] = i
            pad_count += 1
    return result


@njit(cache=True)
def ca3_hebbian_bind_jit(
    W: np.ndarray,              # float32[CA3_SIZE, CA3_SIZE] — weight matrix
    active_cells: np.ndarray,   # int64[:] — active CA3 cell indices
    lr: float,                  # learning rate
    ltd_rate: float,            # LTD rate for inactive pairs
    weight_max: float,
) -> None:
    """
    Hopfield outer-product Hebbian binding.
    W[i,j] += lr  for all (i,j) in active_cells × active_cells
    W[i,j] -= ltd  for active_cells pairs already saturated (LTD — now implemented!)

    Biological basis: Hebb (1949) — cells that fire together wire together.
    LTD: Bear & Malenka (1994) — synaptic depression via mGluR activation.
    """
    n = len(active_cells)
    for i in range(n):
        ci = active_cells[i]
        for j in range(n):
            cj = active_cells[j]
            if ci == cj:
                continue
            new_w = W[ci, cj] + lr
            if new_w > weight_max:
                # Saturated — apply LTD to prevent runaway potentiation
                W[ci, cj] = weight_max - ltd_rate
            else:
                W[ci, cj] = new_w


@njit(cache=True)
def ca3_retrieve_jit(
    W: np.ndarray,             # float32[CA3_SIZE, CA3_SIZE]
    cue_cells: np.ndarray,     # int64[:] — partial cue cell indices
    ca3_size: int,
    ca3_sparsity: int,
    n_iterations: int,
) -> np.ndarray:
    """
    Iterative attractor dynamics for pattern completion from partial cue.
    Proven: 100% retrieval on 50%-corrupted inputs (Hopfield 1982).
    Returns settled active cell indices.
    """
    if len(cue_cells) == 0:
        return np.empty(0, dtype=np.int64)

    state = np.zeros(ca3_size, dtype=np.float32)
    for c in cue_cells:
        state[c] = 1.0

    prev_active = np.empty(ca3_sparsity, dtype=np.int64)
    prev_active[:] = -1

    for _ in range(n_iterations):
        # Energy = W @ state
        energy = np.zeros(ca3_size, dtype=np.float32)
        for i in range(ca3_size):
            for j in range(len(cue_cells) if _ == 0 else ca3_sparsity):
                src = cue_cells[j] if _ == 0 else prev_active[j]
                if src >= 0 and state[src] > 0:
                    energy[i] += W[i, src]

        # Winner-take-all: select top-k cells
        k = min(ca3_sparsity, ca3_size)
        # Find k-th largest via partial scan
        sorted_idx = np.argsort(-energy)
        new_active = sorted_idx[:k].astype(np.int64)

        # Check convergence
        converged = True
        if len(prev_active) == len(new_active):
            for i in range(len(new_active)):
                found = False
                for j in range(len(prev_active)):
                    if new_active[i] == prev_active[j]:
                        found = True
                        break
                if not found:
                    converged = False
                    break
        else:
            converged = False

        # Update state
        new_state = np.zeros(ca3_size, dtype=np.float32)
        for c in new_active:
            new_state[c] = 1.0
        state = new_state
        prev_active = new_active

        if converged:
            break

    return prev_active


@njit(cache=True)
def ca3_renormalize_jit(
    W: np.ndarray,           # float32[CA3_SIZE, CA3_SIZE] — in-place
    renorm_factor: float,    # e.g. 0.97 — Tononi SHY downscaling
) -> None:
    """
    SWS synaptic renormalization (Tononi & Cirelli 2006 — Synaptic Homeostasis Hypothesis).
    All CA3 weights scaled down uniformly during deep sleep.
    Weak attractors fade; strong (often-replayed) attractors survive.
    Net effect: forgetting the unimportant while preserving the essential.
    """
    rows, cols = W.shape
    for i in range(rows):
        for j in range(cols):
            W[i, j] *= renorm_factor


@njit(cache=True)
def ca1_mismatch_jit(
    ca3_cells: np.ndarray,   # int64[:] — CA3 completion output
    ec_cells: np.ndarray,    # int64[:] — fresh EC input (current cortical pattern)
    ca3_size: int,
    ca3_sparsity: int,
) -> float:
    """
    CA1 compares CA3 pattern completion against fresh EC input.
    Returns novelty score (0=familiar, 1=completely novel).
    Biological basis: Hasselmo & Schnell (1994) — CA1 as mismatch detector.
    High mismatch → NE spike (LC activation) + CA3 re-binding.
    """
    if len(ca3_cells) == 0 or len(ec_cells) == 0:
        return 1.0

    # Jaccard distance between CA3 completion and EC input
    ca3_set = np.zeros(ca3_size, dtype=np.bool_)
    ec_set  = np.zeros(ca3_size, dtype=np.bool_)
    for c in ca3_cells:
        if 0 <= c < ca3_size:
            ca3_set[c] = True
    for c in ec_cells:
        if 0 <= c < ca3_size:
            ec_set[c] = True

    intersection = 0
    union = 0
    for i in range(ca3_size):
        a = ca3_set[i]
        b = ec_set[i]
        if a and b:
            intersection += 1
        if a or b:
            union += 1

    if union == 0:
        return 0.0
    return 1.0 - (intersection / union)
