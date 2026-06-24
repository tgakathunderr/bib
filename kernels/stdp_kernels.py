"""
BIB Kernels: STDP (Spike-Timing Dependent Plasticity)
======================================================
All inner loops JIT-compiled via Numba @njit.

Biological basis:
  Bi & Poo (1998) — asymmetric STDP window in hippocampal neurons
  Song, Miller & Abbott (2000) — competitive STDP leads to sparse selectivity

Per-cell last-fire timestamps track when each cell last won.
Delta-t between pre and post determines LTP vs LTD via exponential kernels.
"""
from __future__ import annotations
import numpy as np
from numba import njit


@njit(cache=True)
def stdp_update_jit(
    prev_winner_indices: np.ndarray,    # int64[:] — cells that fired last tick
    curr_winner_mask: np.ndarray,       # bool[:] — cells firing this tick
    predictive_mask: np.ndarray,        # bool[:] — cells predicted but may not fire
    connected_targets: np.ndarray,      # int32[:,:] — synapse target cell indices
    permanences: np.ndarray,            # float32[:,:] — synapse permanence values
    synapse_counts: np.ndarray,         # int32[:] — active synapses per cell
    last_fire_time: np.ndarray,         # int64[:] — tick of last win per cell
    current_tick: int,                  # current global tick counter
    ach_scale: float,                   # ACh level gates LTP amplitude
    a_plus: float,                      # LTP base amplitude
    a_minus: float,                     # LTD base amplitude
    tau_plus: float,                    # LTP time constant
    tau_minus: float,                   # LTD time constant
    tau_window: int,                    # max delta_t with STDP effect
    perm_min: float,
    perm_max: float,
) -> None:
    """
    STDP update pass over all synapses from recently-active cells.

    For each synapse (pre_cell → post_cell):
      - If post_cell fired this tick (curr_winner_mask) AND pre fired recently:
          delta_t = current_tick - last_fire_time[pre_cell]
          If 0 < delta_t <= tau_window: LTP
      - If post_cell was predictive but did NOT fire (anti-causal):
          LTD (weighted by how recently pre fired)
    """
    for pre_cell in prev_winner_indices:
        delta_t = current_tick - last_fire_time[pre_cell]
        if delta_t <= 0 or delta_t > tau_window:
            continue

        # Exponential LTP and LTD kernels
        ltp_factor = a_plus * ach_scale * np.exp(-delta_t / tau_plus)
        ltd_factor = a_minus * np.exp(-delta_t / tau_minus)

        n_syn = synapse_counts[pre_cell]
        for s in range(n_syn):
            post_cell = connected_targets[pre_cell, s]
            if post_cell < 0:
                continue

            if curr_winner_mask[post_cell]:
                # Causal: pre fired before post → LTP
                new_perm = permanences[pre_cell, s] + ltp_factor
                if new_perm > perm_max:
                    permanences[pre_cell, s] = perm_max
                else:
                    permanences[pre_cell, s] = new_perm

            elif predictive_mask[post_cell]:
                # Anti-causal: predicted but not activated → LTD
                new_perm = permanences[pre_cell, s] - ltd_factor
                if new_perm < perm_min:
                    permanences[pre_cell, s] = perm_min
                else:
                    permanences[pre_cell, s] = new_perm


@njit(cache=True)
def update_last_fire_times_jit(
    winner_indices: np.ndarray,   # int64[:] — cells that just fired
    last_fire_time: np.ndarray,   # int64[:] — to be updated in-place
    current_tick: int,
) -> None:
    """Stamp all winner cells with the current tick."""
    for cell in winner_indices:
        last_fire_time[cell] = current_tick


@njit(cache=True)
def select_winner_cell_jit(
    col_start: int,
    col_end: int,
    prev_winner_indices: np.ndarray,  # int64[:]
    connected_targets: np.ndarray,    # int32[:,:]
    permanences: np.ndarray,          # float32[:,:]
    synapse_counts: np.ndarray,       # int32[:]
    cell_usage: np.ndarray,           # int32[:] — usage counts for tie-breaking
    threshold: float,
) -> int:
    """
    Select the winner cell within a column that is NOT predicted.
    Strategy: cell with most connected synapses to prev_winners (matching),
    tie-broken by least-used cell (homeostatic fairness).
    Returns the global cell index.
    """
    best_cell = col_start
    best_score = -1
    best_usage = cell_usage[col_start]

    for cell in range(col_start, col_end):
        score = 0
        n_syn = synapse_counts[cell]
        for s in range(n_syn):
            target = connected_targets[cell, s]
            if target < 0 or permanences[cell, s] < threshold:
                continue
            for pw in prev_winner_indices:
                if target == pw:
                    score += 1
                    break

        if score > best_score or (score == best_score and cell_usage[cell] < best_usage):
            best_score = score
            best_cell = cell
            best_usage = cell_usage[cell]

    return best_cell


@njit(cache=True)
def grow_synapses_jit(
    cell: int,
    prev_winner_indices: np.ndarray,
    connected_targets: np.ndarray,
    permanences: np.ndarray,
    synapse_counts: np.ndarray,
    max_synapses: int,
    initial_perm: float,
    max_grow: int = 4,
) -> None:
    """
    Grow new synapses from a winner cell to random prev_winner cells.
    Simulates dendritic spine formation on learning (Bhatt et al. 2009).
    """
    if len(prev_winner_indices) == 0:
        return
    current_count = synapse_counts[cell]
    if current_count >= max_synapses:
        return

    grown = 0
    for pw in prev_winner_indices:
        if current_count + grown >= max_synapses:
            break
        # Check not already connected
        already = False
        for s in range(current_count):
            if connected_targets[cell, s] == pw:
                already = True
                break
        if not already:
            idx = current_count + grown
            connected_targets[cell, idx] = pw
            permanences[cell, idx] = initial_perm
            grown += 1

    synapse_counts[cell] += grown
