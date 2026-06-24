"""
BIB Chemistry: Hypothalamus + HPA Axis
=======================================
Allostatic drive tracking and cortisol stress response.

Biological basis:
  McEwen (1998) — allostasis and allostatic load
  Saper et al. (2005) — hypothalamic sleep-wake switch
  McEwen (2007) — glucocorticoids and hippocampal vulnerability
  Sterling & Eyer (1988) — allostasis: a new paradigm for stress

Drives:
  hunger  — increases each tick; reset by organism's NUTRITION action
  thirst  — increases faster than hunger; reset by organism's NUTRITION action
  fatigue — accumulates; drives organic sleep onset; reset by sleep()
  pain    — injected by organism; decays slowly

Cortisol:
  Rises slowly when homeostatic_deficit > CORTISOL_THRESHOLD for sustained ticks.
  Falls slowly always.
  High cortisol → suppresses hippocampal CA3 binding (stress impairs memory).
  High cortisol → suppresses 5-HT (serotonin falls under chronic stress).
"""
from __future__ import annotations

import numpy as np

from bib.config import (
    CORTISOL_DECAY,
    CORTISOL_RISE,
    CORTISOL_THRESHOLD,
    FATIGUE_RATE,
    HOMEOSTATIC_WEIGHTS,
    HUNGER_RATE,
    MAX_CORTISOL,
    PAIN_DECAY,
    SLEEP_THRESHOLD,
    THIRST_RATE,
)


class Hypothalamus:
    """
    Tracks 4 allostatic drive variables and the HPA Cortisol axis.

    Ownership (per grill-me decisions):
      - fatigue: owned by BIB (neural fatigue = a brain-internal process)
      - hunger, thirst, pain: reported by organism body each tick
    """

    def __init__(self) -> None:
        # Internal drives [0, 1]
        self.hunger:  float = 0.0
        self.thirst:  float = 0.0
        self.fatigue: float = 0.0
        self.pain:    float = 0.0

        # HPA axis
        self.cortisol: float = 0.0
        self._deficit_streak: int = 0   # consecutive ticks above cortisol threshold

    def tick(
        self,
        hunger: float | None = None,
        thirst: float | None = None,
        pain: float | None = None,
    ) -> float:
        """
        Update all drives for one tick.
        Organism body reports current hunger/thirst/pain.
        Fatigue is accumulated internally.
        Returns homeostatic_deficit [0, 1].
        """
        # Organism-reported drives (clipped to [0,1])
        if hunger is not None:
            self.hunger = float(np.clip(hunger, 0.0, 1.0))
        else:
            self.hunger = min(1.0, self.hunger + HUNGER_RATE)

        if thirst is not None:
            self.thirst = float(np.clip(thirst, 0.0, 1.0))
        else:
            self.thirst = min(1.0, self.thirst + THIRST_RATE)

        if pain is not None:
            self.pain = float(np.clip(self.pain + pain, 0.0, 1.0))
        self.pain *= PAIN_DECAY  # pain decays passively each tick

        # Fatigue: brain-owned, accumulates each tick
        self.fatigue = min(1.0, self.fatigue + FATIGUE_RATE)

        # Compute weighted homeostatic deficit
        deficit = (
            HOMEOSTATIC_WEIGHTS[0] * self.hunger
            + HOMEOSTATIC_WEIGHTS[1] * self.thirst
            + HOMEOSTATIC_WEIGHTS[2] * self.fatigue
            + HOMEOSTATIC_WEIGHTS[3] * self.pain
        )
        deficit = float(np.clip(deficit, 0.0, 1.0))

        # HPA axis: cortisol rises under sustained deficit
        if deficit > CORTISOL_THRESHOLD:
            self._deficit_streak += 1
            self.cortisol = min(
                MAX_CORTISOL,
                self.cortisol + CORTISOL_RISE * self._deficit_streak
            )
        else:
            self._deficit_streak = 0
            self.cortisol = max(0.0, self.cortisol - CORTISOL_DECAY)

        return deficit

    def needs_sleep(self) -> bool:
        """
        Returns True when fatigue has accumulated past the sleep threshold.
        Biological: hypothalamic sleep-wake switch flips when adenosine
        accumulation (proxy = fatigue) exceeds threshold.
        """
        return self.fatigue >= SLEEP_THRESHOLD

    def after_sleep(self) -> None:
        """
        Reset fatigue and partially clear cortisol after a full sleep cycle.
        Biological: sleep clears adenosine buildup and reduces HPA activity.
        """
        self.fatigue = 0.0
        self.cortisol = max(0.0, self.cortisol - 0.3)   # cortisol partially cleared
        self._deficit_streak = 0

    def consume_food(self, amount: float = 1.0) -> None:
        """Organism ate — reduce hunger drive."""
        self.hunger = max(0.0, self.hunger - amount)

    def consume_water(self, amount: float = 1.0) -> None:
        """Organism drank — reduce thirst drive."""
        self.thirst = max(0.0, self.thirst - amount)

    def report(self) -> dict:
        return {
            "hunger":  round(self.hunger, 3),
            "thirst":  round(self.thirst, 3),
            "fatigue": round(self.fatigue, 3),
            "pain":    round(self.pain, 3),
            "cortisol": round(self.cortisol, 3),
            "deficit": round(
                HOMEOSTATIC_WEIGHTS[0] * self.hunger
                + HOMEOSTATIC_WEIGHTS[1] * self.thirst
                + HOMEOSTATIC_WEIGHTS[2] * self.fatigue
                + HOMEOSTATIC_WEIGHTS[3] * self.pain,
                3,
            ),
            "needs_sleep": self.needs_sleep(),
        }
