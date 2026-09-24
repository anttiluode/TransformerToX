import dataclasses
import numpy as np
import pytest

from transformer_to_x.pretrained_shadow import (
    DECAY_CONFIGS,
    RESONANT_CONFIGS,
    PretrainedHeadGeometry,
    evaluate_contexts,
    reconstruct_head,
    resident_address_query,
    resident_value_offset,
    resident_write_operator,
    run_shadow_context,
    select_temporal_config,
)
from transformer_to_x.trace_bank import DecayMemory, ResonantMemory, StaticMemory, assert_equal_state_budget


def tiny_geometry() -> PretrainedHeadGeometry:
    return PretrainedHeadGeometry(
        d_model=4,
        d_head=2,
        w_q=np.array([[1, 2], [3, 5], [7, 11], [13, 17]], dtype=float) / 17,
        w_k=np.array([[2, 1], [5, 3], [11, 7], [17, 13]], dtype=float) / 19,
        w_v=np.array([[1, 0], [0, 2], [3, 1], [1, 4]], dtype=float) / 5,
        b_q=np.array([0.2, -0.1]),
        b_k=np.array([0.7, -0.4]),
        b_v=np.array([0.3, -0.2]),
        w_o=np.array([[1, 2, 0, -1], [0.5, -1, 2, 1]], dtype=float),
        score_scale=1 / np.sqrt(2.0),
    )


def identity_geometry(d_model=4):
    eye = np.eye(d_model)
    return PretrainedHeadGeometry(
        d_model=d_model,
        d_head=d_model,
        w_q=eye,
        w_k=eye,
        w_v=eye,
        b_q=np.zeros(d_model),
        b_k=np.zeros(d_model),
        b_v=np.zeros(d_model),
        w_o=eye,
        score_scale=1.0,
    )


def test_reference_head_is_causal_and_key_bias_cancels():
    x = np.arange(12, dtype=float).reshape(3, 4) / 10
    g = tiny_geometry()
    run = reconstruct_head(g, x)
    assert run.attention.shape == (3, 3)
    assert run.mixture.shape == (3, 2)
    assert run.contribution.shape == (3, 4)
    assert np.allclose(np.triu(run.attention, k=1), 0.0)
    assert np.allclose(run.attention.sum(axis=1), 1.0)
    no_key_bias = dataclasses.replace(g, b_k=np.zeros(2))
    np.testing.assert_allclose(
        run.attention,
        reconstruct_head(no_key_bias, x).attention,
        atol=1e-14,
        rtol=1e-14,
    )


def test_address_write_and_value_offset_follow_compiler_geometry():
    g = tiny_geometry()
    x_t = np.ones(4)
    np.testing.assert_allclose(
        resident_address_query(g, x_t),
        (x_t @ g.w_q + g.b_q) @ g.w_k.T,
    )
    np.testing.assert_allclose(resident_write_operator(g), g.w_v @ g.w_o)
    np.testing.assert_allclose(resident_value_offset(g), g.b_v @ g.w_o)


def test_geometry_and_reference_reject_malformed_inputs():
    g = tiny_geometry()
    with pytest.raises(ValueError):
        dataclasses.replace(g, w_o=np.zeros((3, 4)))
    bad = g.w_q.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        dataclasses.replace(g, w_q=bad)
    with pytest.raises(ValueError):
        dataclasses.replace(g, score_scale=0.0)
    for bad_x in (np.empty((0, 4)), np.ones((2, 3)), np.ones(4)):
        with pytest.raises(ValueError):
            reconstruct_head(g, bad_x)
    nonfinite = np.ones((2, 4))
    nonfinite[0, 0] = np.inf
    with pytest.raises(ValueError):
        reconstruct_head(g, nonfinite)


class SpyMemory:
    scalar_state_budget = 4

    def __init__(self):
        self.calls = []
        self.current = np.zeros(4)

    def reset(self):
        self.calls.append(("reset", None))
        self.current[:] = 0

    def update(self, x_t):
        self.calls.append(("update", x_t.copy()))
        self.current = x_t.copy()

    def read(self, query):
        self.calls.append(("read", query.copy()))
        return self.current.copy()


def test_shadow_uses_inclusive_update_then_read_order():
    g = identity_geometry()
    x = np.eye(4)[:3]
    memory = SpyMemory()
    run = run_shadow_context(g, x, x.copy(), memory)
    assert run.prediction.shape == x.shape
    assert [name for name, _ in memory.calls] == [
        "reset", "update", "read", "update", "read", "update", "read"
    ]


def test_canonical_candidate_budgets_are_exactly_eight_vectors():
    d_model = 768
    memories = [StaticMemory(d_model=d_model, state_slots=8)]
    memories += [DecayMemory(d_model=d_model, rhos=config) for config in DECAY_CONFIGS]
    memories += [ResonantMemory(d_model=d_model, modes=config) for config in RESONANT_CONFIGS]
    assert all(memory.scalar_state_budget == 6144 for memory in memories)
    assert not any(hasattr(memory, "history") for memory in memories)
    with pytest.raises(ValueError):
        assert_equal_state_budget([
            StaticMemory(d_model=d_model, state_slots=7),
            DecayMemory(d_model=d_model, rhos=DECAY_CONFIGS[0]),
        ])


def test_select_temporal_config_uses_selection_contexts_only_and_tie_breaks_by_order():
    g = identity_geometry(d_model=2)
    x = np.array([[1.0, 0.0], [0.0, 1.0]])
    reference = x.copy()

    class ScaleMemory:
        def __init__(self, scale):
            self.scale = scale
            self.scalar_state_budget = 2
            self.current = np.zeros(2)

        def reset(self):
            self.current[:] = 0

        def update(self, event):
            self.current = event.copy()

        def read(self, query):
            return self.current * self.scale

    selected = select_temporal_config(
        configs=(1.0, 0.5),
        selection_contexts=(g, [(x, reference)]),
        memory_factory_for_config=lambda scale: ScaleMemory(scale),
    )
    assert selected == 1.0
    held = evaluate_contexts(g, [(x, 0.5 * reference)], lambda: ScaleMemory(selected))
    assert held["relative_rmse"] > 0.0
    assert selected == 1.0


def test_shadow_rejects_bad_reference_memory_output_and_budget():
    g = identity_geometry()
    x = np.eye(4)[:2]
    with pytest.raises(ValueError):
        run_shadow_context(g, x, np.ones((2, 3)), SpyMemory())
    bad_ref = x.copy()
    bad_ref[0, 0] = np.nan
    with pytest.raises(ValueError):
        run_shadow_context(g, x, bad_ref, SpyMemory())

    class BadRead(SpyMemory):
        def read(self, query):
            return np.zeros(3)

    with pytest.raises(ValueError):
        run_shadow_context(g, x, x, BadRead())

    class BadBudget(SpyMemory):
        scalar_state_budget = 0

    with pytest.raises(ValueError):
        run_shadow_context(g, x, x, BadBudget())

    with pytest.raises(ValueError):
        evaluate_contexts(g, [], lambda: SpyMemory())
