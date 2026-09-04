"""
BIB Memory: Sleep Architecture — NREM N1/N2 + SWS + REM
=========================================================
Biological basis:
  Tononi & Cirelli (2006) — Synaptic Homeostasis Hypothesis (SHY)
  Buzsáki (1989) — two-stage hippocampal→cortical memory consolidation
  Stickgold (2005) — sleep-dependent memory consolidation review
  Hobson & McCarley (1977) — activation-synthesis model of REM
  Walker & Stickgold (2004) — sleep, memory, and plasticity

Three stages with different chemistry and mechanisms:
  N1/N2 (15%): Light sleep. Thalamocortical spindles. Early synaptic homeostasis.
  SWS  (45%): Deep sleep. Low ACh. Hippocampal sharp-wave ripples (SWRs).
               Systems consolidation: hippocampus → cortex transfer.
               CA3 synaptic renormalization (Tononi SHY).
  REM  (40%): High ACh. NE=0. 5-HT=0. CA3 self-chaining.
               Novel cross-domain association formation.
               PFC goal SDR consolidation.
               Amygdala emotional memory integration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from bib.config import (
    REM_ACH, REM_N_CHAINS, REM_NE, REM_5HT,
    RENORM_FACTOR,
    SLEEP_N1N2_FRACTION, SLEEP_REM_FRACTION, SLEEP_SWS_FRACTION,
    SWS_ACH, SWS_N_REPLAYS, SWS_NE, SWS_5HT,
)

if TYPE_CHECKING:
    from bib.core.hippocampus import Hippocampus
    from bib.core.association_cortex import AssociationCortex
    from bib.core.prefrontal import PrefrontalCortex
    from bib.core.amygdala import Amygdala
    from bib.action.basal_ganglia import BasalGanglia
    from bib.chemistry.brainstem import Brainstem
    from bib.memory.episodic import EpisodicBuffer


@dataclass
class SleepReport:
    sws_replays: int = 0
    rem_chains: int = 0
    n1n2_ticks: int = 0
    synaptic_renorm_applied: bool = False
    goal_consolidated: bool = False


class SleepOrchestrator:
    """
    Runs a full sleep cycle with 3 stages.
    Called by BIB.sleep() when brain.needs_sleep() is True.
    """

    def run(
        self,
        hippocampus: "Hippocampus",
        association_cortex: "AssociationCortex",
        prefrontal: "PrefrontalCortex",
        amygdala: "Amygdala",
        basal_ganglia: "BasalGanglia",
        brainstem: "Brainstem",
        episodic_buffer: "EpisodicBuffer",
        total_steps: int = 1000,
    ) -> SleepReport:
        """
        Run full sleep cycle.
        total_steps is divided across N1/N2, SWS, and REM proportionally.
        """
        report = SleepReport()

        n_n1n2 = int(total_steps * SLEEP_N1N2_FRACTION)
        n_sws  = int(total_steps * SLEEP_SWS_FRACTION)
        n_rem  = int(total_steps * SLEEP_REM_FRACTION)

        # ── Stage 1: NREM N1/N2 ──────────────────────────────────────────────
        brainstem.ach = SWS_ACH * 4.0    # slightly higher than SWS (transitional)
        brainstem.ne  = 0.05
        brainstem.sht = 0.30

        episodes_n1n2 = episodic_buffer.sample_chronological(n_n1n2)
        for ep in episodes_n1n2:
            if len(ep.sdr) > 0:
                association_cortex.replay(ep.sdr, ach=brainstem.ach, learn=True)
            report.n1n2_ticks += 1

        # ── Stage 2: SWS — Systems Consolidation ────────────────────────────
        brainstem.override_for_sleep(sws_mode=True)

        # Sharp-wave ripple replay: high-priority episodes → cortex
        swr_episodes = episodic_buffer.sample_prioritized(SWS_N_REPLAYS)
        for ep in swr_episodes:
            if len(ep.sdr) > 0:
                hippocampus.retrieve(ep.sdr)
                association_cortex.replay(ep.sdr, ach=brainstem.ach, learn=True)
                # Reinforce BG weights with historical TD error
                basal_ganglia.learn(ep.sdr, ep.action, ep.td_error)
                report.sws_replays += 1

        # CA3 Synaptic Renormalization (Tononi & Cirelli 2006 SHY)
        hippocampus.sws_renormalize()
        report.synaptic_renorm_applied = True

        # ── Stage 3: REM — Association & Goal Consolidation ──────────────────
        brainstem.override_for_sleep(sws_mode=False)  # ACh=0.95, NE=0, 5-HT=0

        if hippocampus.binds > 0:
            report.rem_chains = hippocampus.ca3_chain(min(REM_N_CHAINS, n_rem))

        # PFC goal consolidation during REM
        prefrontal.consolidate_goal_during_rem(da=0.3)
        report.goal_consolidated = True

        # Amygdala emotional memory integration:
        # slightly decay all valences (extinction processing during REM)
        amygdala.valence_weights *= 0.99

        # Reset cortical context (sleep clears short-term cortical state)
        association_cortex.reset_context()

        return report
