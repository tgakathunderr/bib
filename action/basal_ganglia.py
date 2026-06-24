"""
BIB Action: Basal Ganglia — Go/NoGo with TD(λ) + Eligibility Traces
=====================================================================
Biological basis:
  Frank, Seeberger & O'Reilly (2004) — by carrot or by stick: D1/D2 BG model
  Schultz (1997) — dopamine neurons code prediction error
  Sutton (1988) — learning to predict by TD methods
  Daw, Niv & Dayan (2005) — uncertainty-based competition between BG systems

Architecture:
  Striatum D1 (Go pathway):   actor_weights[N_ACTIONS, SDR_SIZE]
    - Strengthened by positive DA (RPE > 0)
    - Learns: "in state S, do action A (it was better than expected)"
  Striatum D2 (NoGo pathway): nogo_weights[N_ACTIONS, SDR_SIZE]
    - Strengthened by negative DA (RPE < 0)
    - Learns: "in state S, do NOT do action A (it was worse than expected)"
  Action selected: argmax(Go_score - NoGo_score)

  Critic: critic_weights[SDR_SIZE] → scalar V(s) for TD-RPE computation
  Eligibility traces: fix BIM4's 'last byte only' credit assignment bug.

  5-HT modulation: high serotonin → scale up NoGo weights → conservative behavior
  ACh curiosity: high ACh → bonus for least-often-chosen actions (explore novel)
"""
from __future__ import annotations

import numpy as np

from bib.config import (
    ACTION_NAMES,
    BABBLE_RATE_BASE,
    BABBLE_THRESHOLD,
    CURIOSITY_ACH_BONUS,
    GAMMA,
    LAMBDA_TRACE,
    LR_ACTOR,
    LR_CRITIC,
    LR_NOGO,
    N_ACTIONS,
    SDR_SIZE,
    SEROTONIN_NOGO_BIAS,
)
from bib.kernels.bg_kernels import (
    apply_td_update_jit,
    critic_score_jit,
    score_actions_jit,
    update_eligibility_traces_jit,
)


class BasalGanglia:
    """
    Go/NoGo Actor-Critic with TD(λ) eligibility traces.

    Outputs:
      action_index (int 0-7 → MRS GREN action)
      is_exploring (bool)
    """

    def __init__(self) -> None:
        # Striatum D1 — Go pathway
        self.actor_weights: np.ndarray = np.zeros(
            (N_ACTIONS, SDR_SIZE), dtype=np.float32
        )
        # Striatum D2 — NoGo pathway (non-negative)
        self.nogo_weights: np.ndarray = np.zeros(
            (N_ACTIONS, SDR_SIZE), dtype=np.float32
        )
        # Critic — value function V(s)
        self.critic_weights: np.ndarray = np.zeros(SDR_SIZE, dtype=np.float32)

        # Eligibility traces — temporal credit assignment
        self.traces: np.ndarray = np.zeros(
            (N_ACTIONS, SDR_SIZE), dtype=np.float32
        )

        # Statistics
        self._action_counts: np.ndarray = np.zeros(N_ACTIONS, dtype=np.int32)
        self._last_action: int = 0
        self._v_current: float = 0.0

        self._rng = np.random.default_rng(seed=42)

    # ────────────────────────────────────────────────────────────────────────
    # Value estimation
    # ────────────────────────────────────────────────────────────────────────
    def estimate_value(self, world_state_sdr: np.ndarray) -> float:
        """V(s) — current state value from Critic."""
        if len(world_state_sdr) == 0:
            return 0.0
        v = critic_score_jit(self.critic_weights, world_state_sdr.astype(np.int64))
        self._v_current = float(v)
        return self._v_current

    # ────────────────────────────────────────────────────────────────────────
    # Action selection
    # ────────────────────────────────────────────────────────────────────────
    def select_action(
        self,
        world_state_sdr: np.ndarray,  # int64[:] — Association Cortex L3
        da: float,
        ach: float,
        serotonin: float,
        fear_salience: float,         # from amygdala — suppresses risky actions
    ) -> tuple[int, bool]:
        """
        Select motor action via Go/NoGo competition.

        Returns (action_index, is_exploring).
        """
        if len(world_state_sdr) == 0:
            return self._rng.integers(0, N_ACTIONS), True

        sdr = world_state_sdr.astype(np.int64)
        go_scores   = score_actions_jit(self.actor_weights, sdr, N_ACTIONS)
        nogo_scores = score_actions_jit(self.nogo_weights, sdr, N_ACTIONS)

        # 5-HT bias: high serotonin scales NoGo pathway (patience / inhibition)
        effective_nogo = nogo_scores * (1.0 + SEROTONIN_NOGO_BIAS * serotonin)

        # Fear suppression: amygdala fear → boost NoGo for last action if bad
        if fear_salience > 0.5:
            effective_nogo[self._last_action] += fear_salience * 0.5

        # Net Q-scores: Go - NoGo
        net_scores = go_scores - effective_nogo

        best_action = int(np.argmax(net_scores))
        max_score = float(net_scores[best_action])

        # ── Exploration logic ──────────────────────────────────────────────
        # Curiosity: high ACh (surprise) → bonus toward least-tried actions
        curiosity_bonus = np.zeros(N_ACTIONS, dtype=np.float32)
        if ach > 0.5:
            visit_freq = self._action_counts.astype(np.float32)
            if visit_freq.sum() > 0:
                inv_freq = 1.0 / (visit_freq + 1.0)
                curiosity_bonus = inv_freq / inv_freq.sum()
                curiosity_bonus *= CURIOSITY_ACH_BONUS * ach
            net_scores_curious = net_scores + curiosity_bonus
            best_action = int(np.argmax(net_scores_curious))
            max_score = float(net_scores_curious[best_action])

        # Babble: random if confidence low OR probabilistic exploration
        babble_rate = BABBLE_RATE_BASE * (1.0 - 0.5 * serotonin)  # 5-HT reduces babble
        is_exploring = False
        if max_score < BABBLE_THRESHOLD or self._rng.random() < babble_rate:
            best_action = int(self._rng.integers(0, N_ACTIONS))
            is_exploring = True

        self._last_action = best_action
        self._action_counts[best_action] += 1
        return best_action, is_exploring

    # ────────────────────────────────────────────────────────────────────────
    # Learning
    # ────────────────────────────────────────────────────────────────────────
    def learn(
        self,
        world_state_sdr: np.ndarray,  # current state SDR
        action: int,
        td_error: float,              # δ from Brainstem (Schultz RPE)
    ) -> None:
        """
        Apply TD(λ) update to Actor, NoGo, and Critic weights.
        Eligibility traces propagate credit to ALL recent (s,a) pairs —
        fixing BIM4's 'only last byte gets credit' bug.
        """
        if len(world_state_sdr) == 0:
            return
        sdr = world_state_sdr.astype(np.int64)

        # Update eligibility traces
        update_eligibility_traces_jit(
            self.traces, sdr, action, LAMBDA_TRACE, N_ACTIONS
        )

        # Apply TD update using traces
        apply_td_update_jit(
            self.actor_weights,
            self.nogo_weights,
            self.critic_weights,
            self.traces,
            sdr,
            td_error,
            LR_ACTOR, LR_NOGO, LR_CRITIC,
            N_ACTIONS,
        )

    # ────────────────────────────────────────────────────────────────────────
    # Reports
    # ────────────────────────────────────────────────────────────────────────
    def get_confidence(self, world_state_sdr: np.ndarray) -> float:
        """Returns max Go-NoGo net score for current state (BG confidence)."""
        if len(world_state_sdr) == 0:
            return 0.0
        sdr = world_state_sdr.astype(np.int64)
        go   = score_actions_jit(self.actor_weights, sdr, N_ACTIONS)
        nogo = score_actions_jit(self.nogo_weights, sdr, N_ACTIONS)
        return float((go - nogo).max())

    def report(self) -> dict:
        return {
            "action_counts": {
                ACTION_NAMES[i]: int(self.action_counts[i])
                for i in range(N_ACTIONS)
            },
            "trace_active_synapses": int((self.traces > 0.01).sum()),
        }

    @property
    def action_counts(self) -> np.ndarray:
        return self._action_counts
