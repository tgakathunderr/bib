"""
BIB: Biologically Inspired Brain
=================================
Top-level orchestration class. Wires all organs together.

Interface (environment-agnostic):
    brain = BIB()
    action = brain.tick(sensors, reward=0.0, homeostatic_state=None)
    if brain.needs_sleep():
        report = brain.sleep()
    brain.save("./brain_snapshot/")
    brain.load("./brain_snapshot/")

Sensory input: dict of float arrays per modality (vision, touch, etc.)
Motor output:  integer action index 0-7 (MRS GREN actions)

The organism body:
  - Provides sensor readings each tick
  - Reports homeostatic state (hunger, thirst, pain)
  - Executes the returned action index in its world
  - BIB is the brain. The body is something else.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

from bib.config import (
    ACTION_NAMES,
    MODALITY_NAMES,
    N_ACTIONS,
    SDR_SIZE,
)
from bib.core.thalamus import Thalamus
from bib.core.neocortex import NeocortexInstance
from bib.core.association_cortex import AssociationCortex
from bib.core.prefrontal import PrefrontalCortex
from bib.core.hippocampus import Hippocampus
from bib.core.amygdala import Amygdala
from bib.action.basal_ganglia import BasalGanglia
from bib.action.cerebellum import Cerebellum
from bib.chemistry.brainstem import Brainstem, ChemicalState
from bib.chemistry.hypothalamus import Hypothalamus
from bib.memory.episodic import EpisodicBuffer
from bib.memory.sleep import SleepOrchestrator


class BIB:
    """
    Biologically Inspired Brain.

    Organs:
      Thalamus           — 6-modality NE-gated sensory encoder
      Neocortex ×6       — modal cortical hierarchies (one per sense)
      AssociationCortex  — 7th cortex: cross-modal world-state integration
      PrefrontalCortex   — working memory + emergent goal representation
      Hippocampus        — DG→CA3→CA1→EC episodic memory
      Amygdala           — fear/reward conditioning + salience tagging
      BasalGanglia       — Go/NoGo action selection + TD(λ)
      Cerebellum         — forward model + motor error correction
      Brainstem          — 4 neuromodulator nuclei (DA/NE/5-HT/ACh)
      Hypothalamus       — allostatic drives + HPA cortisol + sleep onset
      EpisodicBuffer     — prioritized experience replay
      SleepOrchestrator  — 3-stage NREM+REM sleep

    All organs communicate through the tick() method.
    Nothing is hard-coded — all parameters in config.py.
    """

    def __init__(self) -> None:
        # ── Sensory system ──────────────────────────────────────────────────
        self.thalamus = Thalamus()
        self.modal_cortices: dict[str, NeocortexInstance] = {
            name: NeocortexInstance(name=name) for name in MODALITY_NAMES
        }
        self.association_cortex = AssociationCortex()

        # ── Higher brain ────────────────────────────────────────────────────
        self.prefrontal = PrefrontalCortex()
        self.hippocampus = Hippocampus()
        self.amygdala = Amygdala()

        # ── Motor system ────────────────────────────────────────────────────
        self.basal_ganglia = BasalGanglia()
        self.cerebellum = Cerebellum()

        # ── Chemistry ───────────────────────────────────────────────────────
        self.brainstem = Brainstem()
        self.hypothalamus = Hypothalamus()

        # ── Memory ──────────────────────────────────────────────────────────
        self.episodic_buffer = EpisodicBuffer()
        self._sleep_orchestrator = SleepOrchestrator()

        # ── State tracking ──────────────────────────────────────────────────
        self._tick_count: int = 0
        self._prev_world_state: np.ndarray = np.array([], dtype=np.int64)
        self._prev_action: int = 0
        self._prev_td_error: float = 0.0

    # ────────────────────────────────────────────────────────────────────────
    # Main tick
    # ────────────────────────────────────────────────────────────────────────
    def tick(
        self,
        sensors: dict[str, np.ndarray],
        reward: float = 0.0,
        homeostatic_state: Optional[dict] = None,
    ) -> int:
        """
        One brain tick. Returns chosen MRS GREN action index (0-7).

        Args:
            sensors: dict[modality → float array] — raw sensor readings.
                     Missing modalities default to zero vectors.
            reward:  external reward signal (-1.0 to +1.0)
            homeostatic_state: dict with optional keys:
                     'hunger' (float), 'thirst' (float), 'pain' (float)

        Returns:
            action_index: int 0-7 (see config.ACTION_NAMES)
        """
        self._tick_count += 1
        hs = homeostatic_state or {}

        # ── 1. Hypothalamus: update allostatic drives ────────────────────────
        homeostatic_deficit = self.hypothalamus.tick(
            hunger=hs.get("hunger"),
            thirst=hs.get("thirst"),
            pain=hs.get("pain"),
        )

        # ── 2. Thalamus: encode sensory input → 6 modal SDRs ─────────────────
        suppression = {
            name: self.modal_cortices[name].get_l1_predictive_columns()
            for name in MODALITY_NAMES
        }
        modal_sdrs = self.thalamus.encode_all(
            sensors=sensors,
            ne_level=self.brainstem.ne,
            suppressed_cols_per_modality=suppression,
        )

        # ── 3. 6 Modal Cortices: L1→L3 processing per modality ───────────────
        modal_l3_sdrs: dict[str, np.ndarray] = {}
        total_surprise = 0.0
        for name in MODALITY_NAMES:
            surprise = self.modal_cortices[name].step(
                modal_sdrs[name], ach=self.brainstem.ach
            )
            modal_l3_sdrs[name] = self.modal_cortices[name].get_l3_sdr()
            total_surprise += surprise
        avg_surprise = total_surprise / max(len(MODALITY_NAMES), 1)

        # ── 4. Association Cortex: fuse 6×L3 → unified world-state SDR ───────
        world_state_sdr, assoc_surprise = self.association_cortex.step(
            modal_l3_sdrs, ach=self.brainstem.ach
        )
        surprise = (avg_surprise + assoc_surprise) * 0.5

        # ── 5. Amygdala: emotional salience tagging ───────────────────────────
        pain_signal = float(hs.get("pain", 0.0) or 0.0)
        salience, valence, ne_spike = self.amygdala.tick(
            world_state_sdr,
            reward=max(0.0, reward),
            pain=max(0.0, pain_signal),
        )

        # ── 6. Hippocampus: DG→CA3→CA1→EC episodic memory ────────────────────
        temporal_ctx, ca1_novelty, hippo_bias = self.hippocampus.tick(
            world_state_sdr,
            ach=self.brainstem.ach,
            amygdala_salience=salience,
            cortisol=self.hypothalamus.cortisol,
        )
        # Inject hippocampal apical bias into association cortex
        self.association_cortex.apply_hippo_bias(hippo_bias)

        # ── 7. Basal Ganglia: estimate V(s) ──────────────────────────────────
        v_current = self.basal_ganglia.estimate_value(world_state_sdr)

        # ── 8. Brainstem: update all 5 chemicals ─────────────────────────────
        # Need V(s') — use current V as approximation until next tick
        v_next = v_current   # updated after action taken, next tick
        td_error, chem = self.brainstem.tick(
            reward=reward,
            surprise=surprise,
            v_current=self.basal_ganglia.estimate_value(self._prev_world_state)
                if len(self._prev_world_state) > 0 else 0.0,
            v_next=v_current,
            amygdala_salience=salience,
            homeostatic_deficit=homeostatic_deficit,
        )
        # Inject cortisol from hypothalamus
        chem.cortisol = self.hypothalamus.cortisol

        # ── 9. Respiration: always-on brainstem background ───────────────────
        energy_in_hs = float(hs.get("energy", 1.0) or 1.0)
        self.brainstem.respiration_tick(energy_in_hs)

        # ── 10. Prefrontal: working memory + goal SDR update ──────────────────
        pfc_bias = self.prefrontal.tick(world_state_sdr, da=self.brainstem.da)

        # ── 11. Cerebellum: predict next state, get correction ────────────────
        cereb_correction = self.cerebellum.get_score_correction(world_state_sdr)

        # ── 12. Basal Ganglia: action selection (Go/NoGo + curiosity) ─────────
        fear_salience = self.amygdala.get_fear_salience(world_state_sdr)
        action, is_exploring = self.basal_ganglia.select_action(
            world_state_sdr,
            da=self.brainstem.da,
            ach=self.brainstem.ach,
            serotonin=self.brainstem.sht,
            fear_salience=fear_salience,
        )

        # ── 13. Cerebellum: forward prediction for this action ────────────────
        self.cerebellum.predict_next(world_state_sdr, action)

        # ── 14. BG learning on PREVIOUS step's TD error ───────────────────────
        if len(self._prev_world_state) > 0:
            self.basal_ganglia.learn(
                self._prev_world_state, self._prev_action, td_error
            )
            # Cerebellum: update forward model with what actually happened
            motor_error = self.cerebellum.learn(
                self._prev_world_state, self._prev_action, world_state_sdr
            )

        # ── 15. Store episode in PER buffer ───────────────────────────────────
        if len(world_state_sdr) > 0:
            self.episodic_buffer.store(
                sdr=world_state_sdr,
                action=action,
                reward=reward,
                td_error=td_error,
                ach=self.brainstem.ach,
                da=self.brainstem.da,
            )

        # ── 16. Handle MRS GREN action side-effects ───────────────────────────
        self._handle_mrs_gren_action(action, hs)

        # ── 17. Save state for next tick ──────────────────────────────────────
        self._prev_world_state = world_state_sdr
        self._prev_action = action
        self._prev_td_error = td_error

        return action

    # ────────────────────────────────────────────────────────────────────────
    # Sleep
    # ────────────────────────────────────────────────────────────────────────
    def needs_sleep(self) -> bool:
        """True when fatigue > SLEEP_THRESHOLD. Organism should trigger sleep."""
        return self.hypothalamus.needs_sleep()

    def sleep(self) -> dict:
        """
        Run full 3-stage sleep cycle (NREM N1/N2 → SWS → REM).
        Returns consolidation report dict.
        Should be called by the organism when the environment is safe.
        """
        report = self._sleep_orchestrator.run(
            hippocampus=self.hippocampus,
            association_cortex=self.association_cortex,
            prefrontal=self.prefrontal,
            amygdala=self.amygdala,
            basal_ganglia=self.basal_ganglia,
            brainstem=self.brainstem,
            episodic_buffer=self.episodic_buffer,
        )
        # Reset fatigue after sleep
        self.hypothalamus.after_sleep()
        return {
            "sws_replays": report.sws_replays,
            "rem_chains": report.rem_chains,
            "n1n2_ticks": report.n1n2_ticks,
            "synaptic_renorm": report.synaptic_renorm_applied,
            "goal_consolidated": report.goal_consolidated,
        }

    # ────────────────────────────────────────────────────────────────────────
    # Telemetry
    # ────────────────────────────────────────────────────────────────────────
    def get_chemistry(self) -> dict:
        return {
            "DA":       round(self.brainstem.da, 3),
            "NE":       round(self.brainstem.ne, 3),
            "5-HT":     round(self.brainstem.sht, 3),
            "ACh":      round(self.brainstem.ach, 3),
            "Cortisol": round(self.hypothalamus.cortisol, 3),
        }

    def get_telemetry(self) -> dict:
        return {
            "tick": self._tick_count,
            "chemistry": self.get_chemistry(),
            "homeostasis": self.hypothalamus.report(),
            "hippocampus": self.hippocampus.report(),
            "prefrontal": self.prefrontal.report(),
            "episodic_buffer_size": len(self.episodic_buffer),
            "last_action": ACTION_NAMES[self._prev_action],
            "last_td_error": round(self._prev_td_error, 4),
        }

    def get_action_name(self, action_idx: int) -> str:
        return ACTION_NAMES[action_idx] if 0 <= action_idx < N_ACTIONS else "UNKNOWN"

    # ────────────────────────────────────────────────────────────────────────
    # Persistence (directory-based brain snapshot)
    # ────────────────────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        """
        Save full brain state to directory.
        One .npy file per organ's weight matrix.
        """
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)

        # BG
        np.save(p / "bg_actor.npy", self.basal_ganglia.actor_weights)
        np.save(p / "bg_nogo.npy",  self.basal_ganglia.nogo_weights)
        np.save(p / "bg_critic.npy", self.basal_ganglia.critic_weights)
        # Hippocampus CA3
        np.save(p / "hippo_ca3.npy", self.hippocampus.ca3_weights)
        # Amygdala
        np.save(p / "amygdala.npy", self.amygdala.valence_weights)
        # PFC
        np.save(p / "pfc_goal.npy", self.prefrontal.goal_sdr)
        np.save(p / "pfc_wm.npy",   self.prefrontal.working_memory)
        # Cerebellum
        np.save(p / "cereb_fw.npy", self.cerebellum.fw_weights)

        # Cortex synaptic weights per modal cortex
        for name, ctx in self.modal_cortices.items():
            for i, layer in enumerate(ctx.layers):
                np.save(p / f"cortex_{name}_L{i+1}_perms.npy", layer.permanences)
                np.save(p / f"cortex_{name}_L{i+1}_targets.npy", layer.connected_targets)

        print(f"[BIB] Brain state saved to {p}")

    def load(self, path: str) -> None:
        """Load brain state from directory (inverse of save)."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Brain snapshot not found: {p}")

        def _load(fname: str) -> np.ndarray:
            return np.load(p / fname)

        self.basal_ganglia.actor_weights[:] = _load("bg_actor.npy")
        self.basal_ganglia.nogo_weights[:]  = _load("bg_nogo.npy")
        self.basal_ganglia.critic_weights[:] = _load("bg_critic.npy")
        self.hippocampus.ca3_weights[:]      = _load("hippo_ca3.npy")
        self.amygdala.valence_weights[:]     = _load("amygdala.npy")
        self.prefrontal.goal_sdr[:]          = _load("pfc_goal.npy")
        self.prefrontal.working_memory[:]    = _load("pfc_wm.npy")
        self.cerebellum.fw_weights[:]        = _load("cereb_fw.npy")

        for name, ctx in self.modal_cortices.items():
            for i, layer in enumerate(ctx.layers):
                pf = p / f"cortex_{name}_L{i+1}_perms.npy"
                tf = p / f"cortex_{name}_L{i+1}_targets.npy"
                if pf.exists():
                    layer.permanences[:]        = np.load(pf)
                    layer.connected_targets[:]  = np.load(tf)

        print(f"[BIB] Brain state loaded from {p}")

    # ────────────────────────────────────────────────────────────────────────
    # MRS GREN action side effects (brain-level bookkeeping)
    # ────────────────────────────────────────────────────────────────────────
    def _handle_mrs_gren_action(self, action: int, hs: dict) -> None:
        """
        When BG selects a life-process action, update internal state accordingly.
        The organism body handles the physical execution; BIB handles the neural side.
        """
        action_name = ACTION_NAMES[action]
        if action_name == "NUTRITION":
            # If organism eats/drinks, reset corresponding drives
            if hs.get("has_food"):
                self.hypothalamus.consume_food(amount=0.5)
            if hs.get("has_water"):
                self.hypothalamus.consume_water(amount=0.5)
