"""
BIB Core: Prefrontal Cortex — Working Memory & Emergent Goal
=============================================================
Biological basis:
  Fuster (2001) — the prefrontal cortex: an update
  Miller & Cohen (2001) — an integrative theory of PFC function
  Goldman-Rakic (1995) — cellular basis of working memory

Working memory: ring buffer of last K Association Cortex L3 SDRs.
  Enables multi-step temporal reasoning (holding the recent past online).

Goal SDR: emergent via high-DA state consolidation.
  Repeated high-DA cortical states get consolidated into goal_sdr.
  No external injection — the brain learns what to want through experience.
  (Consistent with Miller & Cohen 2001: PFC selects biasing signals based on reward history.)

Top-down bias: goal_sdr projects onto Association Cortex as an apical bias.
  Biases cortex predictions toward goal-relevant patterns.
  Drives goal-directed behavior without explicit programming.
"""
from __future__ import annotations

import numpy as np

from bib.config import (
    PFC_APICAL_GAIN,
    PFC_GOAL_DECAY,
    PFC_GOAL_LR,
    PFC_WORKING_MEMORY_K,
    SDR_SIZE,
    SDR_SPARSITY,
)


class PrefrontalCortex:
    """
    Prefrontal Cortex: working memory + emergent goal representation.

    State:
      working_memory  [K, SDR_SIZE]  float32  — ring buffer of recent L3 SDRs
      goal_sdr        [SDR_SIZE]     float32  — goal activation vector
      _wm_cursor      int            — ring buffer write head
    """

    def __init__(self) -> None:
        self.working_memory: np.ndarray = np.zeros(
            (PFC_WORKING_MEMORY_K, SDR_SIZE), dtype=np.float32
        )
        self.goal_sdr: np.ndarray = np.zeros(SDR_SIZE, dtype=np.float32)
        self._wm_cursor: int = 0

    def tick(
        self,
        world_state_sdr: np.ndarray,   # int64[:] — Association Cortex L3
        da: float,                      # Dopamine (RPE signal)
    ) -> np.ndarray:
        """
        Update working memory with new world state.
        Update goal SDR based on DA signal (emergent goal formation).
        Returns apical_bias[SDR_SIZE] for Association Cortex.

        Goal learning rule:
          If DA > 0 (reward / better than expected):
            goal_sdr += lr × current_state_activation
            (this state is desirable → encode as goal)
          If DA < 0 (punishment):
            goal_sdr -= lr × |DA| × current_state_activation
            (avoid this state)
          Always: goal_sdr *= (1 - decay)  [slow drift without reinforcement]
        """
        # ── 1. Working memory update ────────────────────────────────────────
        slot = np.zeros(SDR_SIZE, dtype=np.float32)
        if len(world_state_sdr) > 0:
            slot[world_state_sdr] = 1.0
        self.working_memory[self._wm_cursor] = slot
        self._wm_cursor = (self._wm_cursor + 1) % PFC_WORKING_MEMORY_K

        # ── 2. Emergent goal formation (Miller & Cohen 2001) ────────────────
        if len(world_state_sdr) > 0:
            if da > 0.05:
                # Positive RPE: consolidate current state toward goal
                self.goal_sdr[world_state_sdr] += PFC_GOAL_LR * da
            elif da < -0.05:
                # Negative RPE: suppress this state from goal
                self.goal_sdr[world_state_sdr] -= PFC_GOAL_LR * abs(da)

        # Global goal decay — goals drift without reinforcement
        self.goal_sdr *= (1.0 - PFC_GOAL_DECAY)
        np.clip(self.goal_sdr, -1.0, 1.0, out=self.goal_sdr)

        # ── 3. Apical bias = goal SDR projected top-down ────────────────────
        return self._build_apical_bias(world_state_sdr)

    def get_active_goal_cols(self) -> np.ndarray:
        """Returns the top-K goal-relevant column indices."""
        k = min(SDR_SPARSITY, SDR_SIZE)
        top_k = np.argpartition(self.goal_sdr, -k)[-k:].astype(np.int64)
        return top_k[self.goal_sdr[top_k] > 0.0]

    def get_context_sdr(self) -> np.ndarray:
        """
        Concatenated working memory summary (time-averaged).
        Used during REM sleep for goal consolidation replay.
        """
        return self.working_memory.mean(axis=0)

    def consolidate_goal_during_rem(self, da: float = 0.3) -> None:
        """
        REM sleep: goal SDR reinforced from working memory.
        Biological: REM replays emotional + goal-relevant episodes
        (Hobson 2002; Walker & Stickgold 2004).
        """
        ctx = self.get_context_sdr()
        strong_cols = np.where(ctx > 0.3)[0]
        if len(strong_cols) > 0:
            self.goal_sdr[strong_cols] += PFC_GOAL_LR * da
        np.clip(self.goal_sdr, -1.0, 1.0, out=self.goal_sdr)

    def _build_apical_bias(self, world_state_sdr: np.ndarray) -> np.ndarray:
        """Scale goal SDR by apical gain to produce top-down bias vector."""
        bias = self.goal_sdr * PFC_APICAL_GAIN
        # Working memory context adds slight activation boost
        wm_avg = self.working_memory.mean(axis=0)
        bias += wm_avg * (PFC_APICAL_GAIN * 0.5)
        np.clip(bias, 0.0, 0.3, out=bias)
        return bias.astype(np.float32)

    def report(self) -> dict:
        active_goal = self.get_active_goal_cols()
        return {
            "active_goal_cols": len(active_goal),
            "goal_max": float(self.goal_sdr.max()),
            "goal_mean": float(self.goal_sdr[self.goal_sdr > 0].mean()) if self.goal_sdr.max() > 0 else 0.0,
            "wm_occupancy": float((self.working_memory > 0.1).mean()),
        }
