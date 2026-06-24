"""
BIB Core: Thalamus
==================
Sensory relay with NE-gated attentional gain control.
6 modalities, each encoded as float vector → SDR column indices.

Biological basis:
  Sherman & Guillery (2006) — thalamus as active gatekeeper, not passive relay
  Aston-Jones & Cohen (2005) — LC-NE adaptive gain theory (thalamic modulation)
  Jones (2001) — thalamic matrix and core pathways

Each modality has its own SDR projection matrix (learned fixed hash).
NE level scales column activation gain: low NE = diffuse, high NE = sharp spotlight.
Top-down cortical suppression: predicted columns arrive at reduced gain (prediction cancellation).
"""
from __future__ import annotations

import numpy as np

from bib.config import (
    MODALITY_INPUT_DIMS,
    MODALITY_NAMES,
    N_MODALITIES,
    NE_GAIN_MAX,
    NE_GAIN_MIN,
    SDR_SIZE,
    SDR_SPARSITY,
    THAL_SUPPRESSION,
)


class ThalamicEncoder:
    """
    Encodes a single modality's float sensor vector into SDR column indices.
    Uses a fixed random projection matrix (hash-seeded per modality).
    """

    def __init__(self, modality: str, seed: int) -> None:
        self.modality = modality
        input_dim = MODALITY_INPUT_DIMS[modality]

        rng = np.random.default_rng(seed)
        # Fixed projection: input_dim → SDR_SIZE (random weights, normalized)
        self.projection: np.ndarray = rng.standard_normal(
            (input_dim, SDR_SIZE)
        ).astype(np.float32)
        # L2-normalize columns so each SDR dimension is equally weighted
        col_norms = np.linalg.norm(self.projection, axis=0, keepdims=True)
        col_norms[col_norms < 1e-8] = 1.0
        self.projection /= col_norms

        self.input_dim = input_dim

    def encode(
        self,
        sensor_vec: np.ndarray,    # float[:input_dim] — raw sensor readings
        ne_level: float,           # 0.0–1.0 — NE gain modulation
        suppressed_cols: np.ndarray | None = None,  # top-down prediction suppression
    ) -> np.ndarray:
        """
        Project sensor vector → SDR column indices.
        NE gates the gain (high NE → sharper, sparser activation).
        Predicted columns are suppressed (sensory prediction cancellation).
        Returns int64[:SDR_SPARSITY] active column indices.
        """
        if len(sensor_vec) != self.input_dim:
            # Pad or truncate to match expected dim
            buf = np.zeros(self.input_dim, dtype=np.float32)
            n = min(len(sensor_vec), self.input_dim)
            buf[:n] = sensor_vec[:n]
            sensor_vec = buf
        else:
            sensor_vec = np.asarray(sensor_vec, dtype=np.float32)

        # Linear projection: (input_dim,) @ (input_dim, SDR_SIZE) → (SDR_SIZE,)
        activations = sensor_vec @ self.projection

        # NE-gated gain: interpolate between flat (low NE) and sharp (high NE)
        gain = NE_GAIN_MIN + (NE_GAIN_MAX - NE_GAIN_MIN) * float(ne_level)
        activations *= gain

        # Top-down suppression: reduce predicted columns
        if suppressed_cols is not None and len(suppressed_cols) > 0:
            activations[suppressed_cols] *= (1.0 - THAL_SUPPRESSION)

        # Select top-K columns (winner-take-all)
        if SDR_SPARSITY >= SDR_SIZE:
            return np.arange(SDR_SIZE, dtype=np.int64)
        top_k = np.argpartition(activations, -SDR_SPARSITY)[-SDR_SPARSITY:]
        return top_k.astype(np.int64)


class Thalamus:
    """
    Multi-modal thalamic relay.
    Maintains one ThalamicEncoder per sensory modality.
    Applies NE-gated gain and top-down cortical suppression.

    Usage:
        sdrs = thalamus.encode_all(
            sensors={
                'vision': np.array([...], dtype=np.float32),
                'touch':  np.array([...], dtype=np.float32),
                ...
            },
            ne_level=0.4,
            suppressed_cols_per_modality={
                'vision': cortex_vision.get_l1_predictive_columns(),
                ...
            }
        )
    """

    def __init__(self) -> None:
        self.encoders: dict[str, ThalamicEncoder] = {}
        for i, name in enumerate(MODALITY_NAMES):
            # Each modality gets a unique seed for its projection matrix
            self.encoders[name] = ThalamicEncoder(modality=name, seed=100 + i)

    def encode_all(
        self,
        sensors: dict[str, np.ndarray],
        ne_level: float = 0.1,
        suppressed_cols_per_modality: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Encode all provided sensor readings into SDRs.
        Missing modalities are treated as zero-vector inputs.
        Returns dict[modality_name → int64[:SDR_SPARSITY]].
        """
        result: dict[str, np.ndarray] = {}
        for name in MODALITY_NAMES:
            sensor_vec = sensors.get(name, np.zeros(MODALITY_INPUT_DIMS[name], dtype=np.float32))
            suppressed = (
                suppressed_cols_per_modality.get(name)
                if suppressed_cols_per_modality
                else None
            )
            result[name] = self.encoders[name].encode(
                sensor_vec=sensor_vec,
                ne_level=ne_level,
                suppressed_cols=suppressed,
            )
        return result

    def encode_one(
        self,
        modality: str,
        sensor_vec: np.ndarray,
        ne_level: float = 0.1,
        suppressed_cols: np.ndarray | None = None,
    ) -> np.ndarray:
        """Encode a single modality."""
        return self.encoders[modality].encode(sensor_vec, ne_level, suppressed_cols)
