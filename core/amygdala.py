"""
BIB Core: Amygdala — Emotional Salience & Fear Conditioning
===========================================================
Basolateral Amygdala (BLA): fear/reward conditioning via Pavlovian association.
Central Nucleus (CeA): drives autonomic arousal outputs (NE spike, avoidance).

Biological basis:
  LeDoux (2000) — emotion circuits in the brain; amygdala and fear conditioning
  Phelps & LeDoux (2005) — contributions of BLA to emotional memory
  Pare, Quirk & LeDoux (2004) — new vistas on amygdala networks in conditioned fear

Mechanism:
  - Maintains valence_weights[SDR_SIZE]: emotional significance per cortical column
  - Positive reward → LTP on active columns → positive valence memory
  - Pain/fear → LTP on active columns → negative valence memory (fear conditioning)
  - Extinction: repeated exposure without pain/reward → slow valence decay
  - Output: salience scalar → NE (arousal if high), Hippocampus (lower bind threshold)
"""
from __future__ import annotations

import numpy as np

from bib.config import (
    AMYGDALA_EXTINCTION_RATE,
    AMYGDALA_NE_THRESHOLD,
    AMYGDALA_SDR_SIZE,
    AMYGDALA_VALENCE_LR,
)


class Amygdala:
    """
    Emotional salience tagger and fear/reward conditioning organ.

    Inputs:  current cortical SDR columns, reward signal, pain signal
    Outputs: salience (float 0-1), valence (float -1 to +1)
    """

    def __init__(self) -> None:
        # Valence weights per cortical column (-1 = fear, +1 = reward)
        # LeDoux (2000): BLA synaptic weights encode emotional significance
        self.valence_weights: np.ndarray = np.zeros(AMYGDALA_SDR_SIZE, dtype=np.float32)

    def tick(
        self,
        active_cols: np.ndarray,   # int64[:] — current cortical active columns
        reward: float,             # +1.0 = reward, 0 = neutral
        pain: float,               # +1.0 = max pain (converted to -1.0 valence)
    ) -> tuple[float, float, bool]:
        """
        Update amygdala conditioning and compute emotional output.

        Returns:
          (salience, valence, ne_spike_triggered)
          - salience: 0-1, how emotionally significant this moment is
          - valence: -1 to +1, positive = appetitive, negative = aversive
          - ne_spike_triggered: True if salience > AMYGDALA_NE_THRESHOLD
        """
        if len(active_cols) == 0:
            return 0.0, 0.0, False

        # ── 1. Compute current valence from active columns ───────────────────
        raw_valence = float(self.valence_weights[active_cols].mean())

        # ── 2. Conditioning: Pavlovian LTP (Hebb + reward signal) ───────────
        if reward != 0.0 or pain != 0.0:
            # Net emotional signal: positive reward, negative pain
            emotional_signal = float(np.clip(reward - pain, -1.0, 1.0))

            # LTP: strengthen valence association for active columns
            lr = AMYGDALA_VALENCE_LR
            self.valence_weights[active_cols] += lr * emotional_signal
            np.clip(self.valence_weights, -1.0, 1.0, out=self.valence_weights)

        # ── 3. Extinction: passive decay on all columns (including active) ───
        # Biological: extinction = new learning, not unlearning (Quirk 2002)
        self.valence_weights -= AMYGDALA_EXTINCTION_RATE
        np.clip(self.valence_weights, -1.0, 1.0, out=self.valence_weights)

        # ── 4. Compute salience = absolute magnitude of valence ──────────────
        # CeA responds to both fear and strong positive valence (absolute value)
        salience = float(abs(raw_valence))
        salience = float(np.clip(salience, 0.0, 1.0))

        # ── 5. NE spike trigger (CeA → LC) ──────────────────────────────────
        ne_spike = salience > AMYGDALA_NE_THRESHOLD

        return salience, raw_valence, ne_spike

    def get_fear_salience(self, active_cols: np.ndarray) -> float:
        """
        Returns negative valence magnitude for current columns.
        Used by BG NoGo pathway to suppress dangerous actions.
        """
        if len(active_cols) == 0:
            return 0.0
        v = float(self.valence_weights[active_cols].mean())
        return float(np.clip(-v, 0.0, 1.0))  # only negative valence

    def get_reward_salience(self, active_cols: np.ndarray) -> float:
        """
        Returns positive valence magnitude for current columns.
        Used by PFC goal formation.
        """
        if len(active_cols) == 0:
            return 0.0
        v = float(self.valence_weights[active_cols].mean())
        return float(np.clip(v, 0.0, 1.0))   # only positive valence
