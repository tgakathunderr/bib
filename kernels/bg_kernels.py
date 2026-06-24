"""
BIB Kernels: Basal Ganglia — Go/NoGo + TD(λ) + Eligibility Traces
===================================================================
Frank et al. (2004) — D1/D2 receptor model of BG action selection
Sutton (1988) — TD(λ) with eligibility traces
"""
from __future__ import annotations
import numpy as np
from numba import njit


@njit(cache=True)
def score_actions_jit(
    weights: np.ndarray,          # float32[N_ACTIONS, N] — Go or NoGo weights
    sdr_indices: np.ndarray,      # int64[:] — active cortical column indices
    n_actions: int,
) -> np.ndarray:
    """
    Sparse dot product: sum weights[:, active_cols] across actions.
    Returns float32[N_ACTIONS] expected-reward vector.
    O(N_ACTIONS × |SDR|) — fast sparse computation.
    """
    scores = np.zeros(n_actions, dtype=np.float32)
    for idx in sdr_indices:
        for a in range(n_actions):
            scores[a] += weights[a, idx]
    return scores


@njit(cache=True)
def critic_score_jit(
    critic_weights: np.ndarray,   # float32[N] — value function weights
    sdr_indices: np.ndarray,      # int64[:] — active cortical column indices
) -> float:
    """
    Sparse dot product for critic: V(s) = sum critic_weights[active_cols].
    """
    value = 0.0
    for idx in sdr_indices:
        value += critic_weights[idx]
    return value


@njit(cache=True)
def update_eligibility_traces_jit(
    traces: np.ndarray,           # float32[N_ACTIONS, N] — eligibility traces
    sdr_indices: np.ndarray,      # int64[:] — active cortical column indices
    chosen_action: int,           # action taken this tick
    lambda_decay: float,          # trace decay rate
    n_actions: int,
) -> None:
    """
    Update eligibility traces (TD-λ):
    1. Decay all traces by lambda
    2. Increment trace for (chosen_action, active_columns)

    This propagates credit assignment backward through recent (s,a) pairs,
    fixing the 'only last byte gets credit' bug from BIM4.
    """
    # Decay all traces
    for a in range(n_actions):
        for i in range(traces.shape[1]):
            traces[a, i] *= lambda_decay

    # Set trace for current (action, state)
    for idx in sdr_indices:
        traces[chosen_action, idx] = 1.0


@njit(cache=True)
def apply_td_update_jit(
    actor_weights: np.ndarray,    # float32[N_ACTIONS, N] — Go pathway
    nogo_weights: np.ndarray,     # float32[N_ACTIONS, N] — NoGo pathway
    critic_weights: np.ndarray,   # float32[N] — value function
    traces: np.ndarray,           # float32[N_ACTIONS, N] — eligibility traces
    sdr_indices: np.ndarray,      # int64[:] — current active columns
    td_error: float,              # δ = r + γV(s') - V(s)
    lr_actor: float,
    lr_nogo: float,
    lr_critic: float,
    n_actions: int,
    weight_max: float = 10.0,
) -> None:
    """
    Apply TD(λ) weight updates:
    - Actor (Go): weights += lr_actor × δ × traces  (when δ > 0: D1 strengthened)
    - NoGo: weights += lr_nogo × (-δ) × traces     (when δ < 0: D2 strengthened)
    - Critic: weights[active] += lr_critic × δ
    """
    for a in range(n_actions):
        for i in range(traces.shape[1]):
            if traces[a, i] > 0.0:
                # Go pathway (D1): activated by positive RPE
                new_go = actor_weights[a, i] + lr_actor * td_error * traces[a, i]
                if new_go > weight_max:
                    actor_weights[a, i] = weight_max
                elif new_go < -weight_max:
                    actor_weights[a, i] = -weight_max
                else:
                    actor_weights[a, i] = new_go

                # NoGo pathway (D2): activated by negative RPE
                new_nogo = nogo_weights[a, i] + lr_nogo * (-td_error) * traces[a, i]
                if new_nogo > weight_max:
                    nogo_weights[a, i] = weight_max
                elif new_nogo < 0.0:
                    nogo_weights[a, i] = 0.0
                else:
                    nogo_weights[a, i] = new_nogo

    # Critic update (only on active columns)
    for idx in sdr_indices:
        new_val = critic_weights[idx] + lr_critic * td_error
        if new_val > weight_max:
            critic_weights[idx] = weight_max
        elif new_val < -weight_max:
            critic_weights[idx] = -weight_max
        else:
            critic_weights[idx] = new_val
