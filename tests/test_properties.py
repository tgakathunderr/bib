import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bib.brain import BIB
from bib.config import (
    GAMMA,
    HIPPO_CA3_SIZE,
    HIPPO_CA3_SPARSITY,
    HIPPO_CA3_LR,
    HIPPO_CA3_LTD,
    HIPPO_WEIGHT_MAX,
    SDR_SPARSITY,
)
from bib.examples.grid_agent import GridWorld
from bib.kernels.hippo_kernels import (
    ca3_hebbian_bind_jit,
    ca3_retrieve_jit,
)


def overlap(a, b):
    return len(set(int(x) for x in a) & set(int(x) for x in b))


class TestCA3AttractorRetrieval:
    def _build(self, rng, n_patterns):
        W = np.zeros((HIPPO_CA3_SIZE, HIPPO_CA3_SIZE), dtype=np.float32)
        patterns = []
        for _ in range(n_patterns):
            pat = rng.choice(HIPPO_CA3_SIZE, size=HIPPO_CA3_SPARSITY, replace=False).astype(np.int64)
            ca3_hebbian_bind_jit(W, pat, HIPPO_CA3_LR, HIPPO_CA3_LTD, HIPPO_WEIGHT_MAX)
            patterns.append(pat)
        return W, patterns

    def test_retrieval_recovers_heavily_corrupted_cue(self):
        rng = np.random.default_rng(0)
        W, patterns = self._build(rng, 3)
        for pat in patterns:
            keep = max(1, int(len(pat) * 0.5))
            corrupted = pat[:keep]
            retrieved = ca3_retrieve_jit(W, corrupted, HIPPO_CA3_SIZE, HIPPO_CA3_SPARSITY, 4)
            assert overlap(retrieved, pat) >= int(HIPPO_CA3_SPARSITY * 0.9)

    def test_retrieval_perfect_within_capacity(self):
        rng = np.random.default_rng(1)
        W, patterns = self._build(rng, 5)
        for pat in patterns:
            retrieved = ca3_retrieve_jit(W, pat, HIPPO_CA3_SIZE, HIPPO_CA3_SPARSITY, 4)
            assert overlap(retrieved, pat) == HIPPO_CA3_SPARSITY

    def test_single_cell_cue_recovers_memory(self):
        rng = np.random.default_rng(2)
        W, patterns = self._build(rng, 5)
        for pat in patterns:
            cue = pat[:1]
            retrieved = ca3_retrieve_jit(W, cue, HIPPO_CA3_SIZE, HIPPO_CA3_SPARSITY, 4)
            assert overlap(retrieved, pat) >= int(HIPPO_CA3_SPARSITY * 0.9)

    def test_retrieval_degrades_gracefully_over_capacity(self):
        rng = np.random.default_rng(3)
        W, patterns = self._build(rng, 20)
        recovered = []
        for pat in patterns:
            retrieved = ca3_retrieve_jit(W, pat, HIPPO_CA3_SIZE, HIPPO_CA3_SPARSITY, 4)
            recovered.append(overlap(retrieved, pat))
        mean_recovery = sum(recovered) / len(recovered)
        assert mean_recovery >= HIPPO_CA3_SPARSITY * 0.5


class TestCriticConvergence:
    def _fixed_sdr(self):
        return np.arange(32, dtype=np.int64)

    def test_critic_converges_to_analytic_fixed_point(self):
        brain = BIB()
        sdr = self._fixed_sdr()
        brain._prev_world_state = sdr.copy()
        v = brain.basal_ganglia.estimate_value(sdr)
        for _ in range(200):
            td = 0.5 + GAMMA * v - v
            brain.basal_ganglia.critic_weights[sdr] += 0.10 * td
            v = brain.basal_ganglia.estimate_value(sdr)
        fixed_point = 0.5 / (1.0 - GAMMA)
        assert abs(v - fixed_point) < 0.05
        assert abs(td) < 0.05

    def test_da_positive_for_underpredicted_reward(self):
        brain = BIB()
        sdr = self._fixed_sdr()
        brain.basal_ganglia.critic_weights[...] = 0.0
        v = brain.basal_ganglia.estimate_value(sdr)
        td = 1.0 + GAMMA * v - v
        assert td > 0.0

    def test_da_negative_for_overpredicted_reward(self):
        brain = BIB()
        sdr = self._fixed_sdr()
        brain.basal_ganglia.critic_weights[...] = 10.0
        v = brain.basal_ganglia.estimate_value(sdr)
        td = 0.0 + GAMMA * v - v
        assert td < 0.0

    def test_da_zero_when_prediction_matches_reward(self):
        brain = BIB()
        sdr = self._fixed_sdr()
        target_v = 0.5 / (1.0 - GAMMA)
        per_col = target_v / len(sdr)
        brain.basal_ganglia.critic_weights[sdr] = per_col
        v = brain.basal_ganglia.estimate_value(sdr)
        td = 0.5 + GAMMA * v - v
        assert abs(td) < 1e-6


class TestWorldStateDiscriminability:
    def _run_positions(self, size: int = 4):
        brain = BIB()
        env = GridWorld(size=size, seed=3)
        env.reset()
        prev_r = 0.0
        for _ in range(3):
            act = brain.tick(env._sensors(), reward=prev_r)
            _, r, d = env.step(act)
            prev_r = r
            if d:
                env.reset()
        keys = [(x, y) for x in range(size) for y in range(size)]
        sdrs = {}
        for (x, y) in keys:
            env.agent = (x, y)
            env._prev_agent = (x, y)
            brain.tick(env._sensors(), reward=0.0)
            sdrs[(x, y)] = brain.association_cortex.get_world_state_sdr()
        return sdrs

    def test_world_state_is_position_discriminative(self):
        sdrs = self._run_positions()
        distinct = len({int(c) for s in sdrs.values() for c in s})
        assert distinct >= SDR_SPARSITY * 2
        keys = list(sdrs)
        overlaps = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a = sdrs[keys[i]]
                b = sdrs[keys[j]]
                if len(a) == 0 or len(b) == 0:
                    continue
                inter = len(set(int(c) for c in a) & set(int(c) for c in b))
                overlaps.append(inter)
        mean_ov = sum(overlaps) / len(overlaps)
        assert mean_ov < len(next(iter(sdrs.values()))) * 0.8


class TestCriticNonDivergence:
    def test_critic_value_stays_bounded_under_learning(self):
        brain = BIB()
        sdr = np.arange(32, dtype=np.int64)
        v = 0.0
        for _ in range(500):
            td = 0.1 + GAMMA * v - v
            brain.basal_ganglia.critic_weights[sdr] += 0.01 * td
            v = brain.basal_ganglia.estimate_value(sdr)
        assert abs(v) < abs(0.1 / (1.0 - GAMMA)) * 1.5
