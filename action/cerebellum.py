"""
BIB Action: Cerebellum — Forward Model & Motor Error Correction
==============================================================
Biological basis:
  Wolpert & Kawato (1998) — MOSAIC model: multiple paired forward-inverse models
  Ito (2008) — the cerebellum and adaptive motor control
  Shadmehr & Mussa-Ivaldi (1994) — adaptive representation of motor dynamics

The cerebellum predicts the sensory consequence of a motor command BEFORE execution.
When the prediction is wrong, Purkinje cells generate an error signal that corrects
the ongoing motor command. Cerebellar damage → ataxia: movement becomes uncoordinated.

BIB forward model:
  fw_weights[N_ACTIONS, SDR_SIZE] → predicts next cortical state SDR given (action)
  Motor error = Jaccard distance between predicted next SDR and actual next SDR
  Correction: delta applied to BG's confidence scores before action selection
"""
from __future__ import annotations

import numpy as np

from bib.config import (
    CEREB_ERROR_SCALE,
    CEREB_FORWARD_LR,
    CEREB_PREDICTION_DECAY,
    N_ACTIONS,
    SDR_SIZE,
    SDR_SPARSITY,
)


class Cerebellum:
    """
    Forward model: predicts next sensory SDR given current state + chosen action.
    Corrects BG's motor output based on prediction error.

    Data:
      fw_weights[N_ACTIONS, SDR_SIZE] float32  — forward model weights
      _last_prediction: np.ndarray             — last predicted next-state columns
    """

    def __init__(self) -> None:
        self.fw_weights: np.ndarray = np.zeros(
            (N_ACTIONS, SDR_SIZE), dtype=np.float32
        )
        self._last_prediction: np.ndarray = np.array([], dtype=np.int64)
        self._last_action: int = 0
        self._temporal_pool: np.ndarray = np.zeros(SDR_SIZE, dtype=np.float32)

    def predict_next(
        self,
        world_state_sdr: np.ndarray,  # current Association Cortex L3
        action: int,
    ) -> np.ndarray:
        """
        Forward model: predict which SDR columns will be active next tick.
        Purkinje cells fire if this prediction is wrong.
        Returns predicted_next_sdr (top-K indices).
        """
        if len(world_state_sdr) == 0:
            self._last_prediction = np.array([], dtype=np.int64)
            return self._last_prediction

        # Sparse forward: sum fw_weights[action, active_cols]
        activations = np.zeros(SDR_SIZE, dtype=np.float32)
        activations[world_state_sdr] = self.fw_weights[action, world_state_sdr]

        # Temporal pooling
        self._temporal_pool = self._temporal_pool * CEREB_PREDICTION_DECAY + activations

        k = min(SDR_SPARSITY, SDR_SIZE)
        top_k = np.argpartition(self._temporal_pool, -k)[-k:].astype(np.int64)
        predicted = top_k[self._temporal_pool[top_k] > 0.01]

        self._last_prediction = predicted
        self._last_action = action
        return predicted

    def compute_motor_error(self, actual_next_sdr: np.ndarray) -> float:
        """
        Purkinje cell error signal: Jaccard distance between predicted and actual.
        High error = wrong forward model → learning signal.
        Returns motor_error [0, 1].
        """
        if len(self._last_prediction) == 0 or len(actual_next_sdr) == 0:
            return 0.0

        pred_set = set(int(c) for c in self._last_prediction)
        actual_set = set(int(c) for c in actual_next_sdr)
        inter = len(pred_set & actual_set)
        union = len(pred_set | actual_set)
        if union == 0:
            return 0.0
        return 1.0 - (inter / union)

    def learn(
        self,
        prev_state_sdr: np.ndarray,   # state when action was taken
        action: int,
        actual_next_sdr: np.ndarray,  # actual next cortical state observed
    ) -> float:
        """
        Update forward model weights using motor error (Purkinje cell gradient).
        fw_weights[action, active_cols in actual] += lr
        fw_weights[action, predicted but wrong] -= lr * error
        Returns motor_error.
        """
        if len(prev_state_sdr) == 0 or len(actual_next_sdr) == 0:
            return 0.0

        motor_error = self.compute_motor_error(actual_next_sdr)

        # Hebbian forward model update (Widrow-Hoff delta rule)
        # Target: actual_next activation at 1.0, everything else at 0.0
        target = np.zeros(SDR_SIZE, dtype=np.float32)
        target[actual_next_sdr] = 1.0

        # Prediction: current forward weights at prev state
        prediction = np.zeros(SDR_SIZE, dtype=np.float32)
        if len(self._last_prediction) > 0:
            prediction[self._last_prediction] = 1.0

        delta = (target - prediction) * CEREB_FORWARD_LR
        self.fw_weights[action, prev_state_sdr] += delta[prev_state_sdr].mean()
        np.clip(self.fw_weights, -5.0, 5.0, out=self.fw_weights)

        return motor_error

    def get_score_correction(
        self,
        world_state_sdr: np.ndarray,
    ) -> np.ndarray:
        """
        Returns per-action correction scores based on forward model confidence.
        Actions whose predicted next-states align with goal get a bonus.
        This nudges BG action selection toward globally-better outcomes.
        """
        if len(world_state_sdr) == 0:
            return np.zeros(N_ACTIONS, dtype=np.float32)

        corrections = np.zeros(N_ACTIONS, dtype=np.float32)
        for a in range(N_ACTIONS):
            # Confidence = mean forward weight on active columns
            corrections[a] = float(
                self.fw_weights[a, world_state_sdr].mean()
            )

        # Scale correction by CEREB_ERROR_SCALE
        corrections *= CEREB_ERROR_SCALE
        return corrections

    def report(self) -> dict:
        return {
            "last_motor_error": float(
                self.compute_motor_error(np.array([], dtype=np.int64))
            ),
            "fw_weights_max": float(self.fw_weights.max()),
            "fw_weights_mean": float(self.fw_weights.mean()),
        }
