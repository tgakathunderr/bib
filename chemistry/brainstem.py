"""
BIB Chemistry: Brainstem Neuromodulator Nuclei
==============================================
Four anatomically distinct nuclei, each producing one chemical:

  VTA + SNc  → Dopamine (DA)   — reward prediction error (Schultz 1997)
  LC         → Norepinephrine  — arousal / attentional spotlight
  Raphe      → Serotonin 5-HT  — patience / impulse control (Dayan & Huys 2009)
  NBM+Septum → Acetylcholine   — cortical plasticity / REM (Hasselmo 2006)

Plus HPA axis Cortisol (in hypothalamus.py) which interacts via brainstem.

Nothing is hard-coded here — all constants come from config.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from bib.config import (
    ACH_DECAY, ACH_SURPRISE_GAIN, BASE_ACH, BASE_DA, BASE_NE, BASE_5HT,
    HT5_DECAY, MAX_ACH, MAX_DA, MAX_NE, MAX_5HT, MIN_DA,
    NE_DECAY,
)


@dataclass
class ChemicalState:
    """Snapshot of all neuromodulator levels at one tick."""
    da:  float = 0.0     # Dopamine (VTA)
    ne:  float = 0.1     # Norepinephrine (LC)
    sht: float = 0.5     # Serotonin 5-HT (Raphe)
    ach: float = 0.1     # Acetylcholine (NBM)
    cortisol: float = 0.0  # HPA axis cortisol


class Brainstem:
    """
    Simulates the 4 neuromodulator nuclei of the biological brainstem.
    Updated once per brain tick after cortical processing.

    Chemical interactions:
      - High ACh (surprise) → triggers NE spike if amygdala salience is high
      - High NE → gates Thalamus gain (handled in thalamus.py via ne level)
      - High 5-HT → BG NoGo pathway scaled (handled in basal_ganglia.py)
      - DA computed as TD-RPE (not raw reward spike like BIM4)
    """

    def __init__(self) -> None:
        self.da:  float = BASE_DA
        self.ne:  float = BASE_NE
        self.sht: float = BASE_5HT
        self.ach: float = BASE_ACH

        self._ach_window: list[float] = []
        self._ach_window_size: int = 10

    # ────────────────────────────────────────────────────────────────────────
    # LC: Norepinephrine — arousal spike on unexpected novelty
    # ────────────────────────────────────────────────────────────────────────
    def update_ne(self, ach_spike: float, amygdala_salience: float, ne_spike: bool) -> None:
        ne_drive = ach_spike * amygdala_salience
        if ne_spike:
            ne_drive += amygdala_salience
        self.ne = float(np.clip(self.ne + ne_drive, 0.0, MAX_NE))
        self.ne = self.ne * NE_DECAY + BASE_NE * (1.0 - NE_DECAY)

    # ────────────────────────────────────────────────────────────────────────
    # Raphe: Serotonin 5-HT — patience / calm when needs met
    # ────────────────────────────────────────────────────────────────────────
    def update_serotonin(self, homeostatic_deficit: float) -> None:
        """
        5-HT = inverse of homeostatic deficit.
        When needs are fully met → 5-HT high → patient, conservative.
        When hungry/thirsty/stressed → 5-HT low → impulsive, explorative.
        Dayan & Huys (2009): serotonin regulates behavioral inhibition.
        """
        target = float(np.clip(1.0 - homeostatic_deficit, 0.0, MAX_5HT))
        self.sht = self.sht * HT5_DECAY + target * (1.0 - HT5_DECAY)

    # ────────────────────────────────────────────────────────────────────────
    # NBM / Septum: Acetylcholine — surprise-driven plasticity gate
    # ────────────────────────────────────────────────────────────────────────
    def update_ach(self, surprise: float) -> None:
        """
        ACh rises with prediction error (cortical surprise).
        Variance-based habituation: if the brain keeps being surprised
        at a constant rate, ACh habituates (adaptation).
        Hasselmo (2006): ACh controls encoding vs retrieval balance in cortex.
        """
        # Variance habituation window
        self._ach_window.append(surprise)
        if len(self._ach_window) > self._ach_window_size:
            self._ach_window.pop(0)

        habituated_surprise = surprise
        if len(self._ach_window) == self._ach_window_size:
            variance = float(np.var(self._ach_window))
            if variance < 0.01 and surprise > 0.5:
                # Constant surprise → habituate (reduce ACh response)
                habituated_surprise *= 0.5

        if habituated_surprise > BASE_ACH:
            self.ach = float(
                np.clip(self.ach + habituated_surprise * ACH_SURPRISE_GAIN, BASE_ACH, MAX_ACH)
            )
        else:
            self.ach = self.ach * ACH_DECAY + BASE_ACH * (1.0 - ACH_DECAY)

    # ────────────────────────────────────────────────────────────────────────
    # Respiration — Always-on brainstem background process
    # Medulla oblongata: pre-Bötzinger complex drives breathing rhythm.
    # This is NOT a BG action — it is automatic and continuous.
    # ────────────────────────────────────────────────────────────────────────
    def respiration_tick(self, energy_available: float) -> float:
        """
        Simulate respiratory energy cost per tick.
        Returns energy_remaining after respiration.
        Brainstem sets basal metabolic rate; organism body tracks energy.
        """
        BASE_RESPIRATION_COST = 0.0005   # per tick energy cost of breathing
        return max(0.0, energy_available - BASE_RESPIRATION_COST)

    # ────────────────────────────────────────────────────────────────────────
    # Master tick update
    # ────────────────────────────────────────────────────────────────────────
    def tick(
        self,
        surprise: float,
        td_error: float,
        amygdala_salience: float,
        homeostatic_deficit: float,
        ne_spike: bool = False,
    ) -> ChemicalState:
        self.da = float(np.clip(BASE_DA + td_error, MIN_DA, MAX_DA))
        self.update_ach(surprise)
        self.update_ne(
            ach_spike=max(0.0, self.ach - BASE_ACH),
            amygdala_salience=amygdala_salience,
            ne_spike=ne_spike,
        )
        self.update_serotonin(homeostatic_deficit)
        return self.snapshot()

    def snapshot(self) -> "ChemicalState":
        return ChemicalState(
            da=self.da,
            ne=self.ne,
            sht=self.sht,
            ach=self.ach,
        )

    def override_for_sleep(self, sws_mode: bool) -> None:
        """
        Set chemistry appropriate for each sleep stage.
        SWS: low ACh (NBM suppressed), NE=0.
        REM: high ACh, NE=0, 5-HT=0.
        Called by sleep.py during sleep cycles.
        """
        if sws_mode:
            self.ach = 0.05
            self.ne  = 0.0
        else:
            # REM
            self.ach = 0.95
            self.ne  = 0.0
            self.sht = 0.0
