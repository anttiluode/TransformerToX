import numpy as np
import pytest

from transformer_to_x.trace_bank import (
    DecayMemory,
    ResonantMemory,
    RytmiWindowMemory,
    StaticMemory,
    assert_equal_state_budget,
)


def test_decay_memory_has_fixed_state_independent_of_sequence_length():
    mem = DecayMemory(d_model=12, rhos=(0.25, 0.75))
    initial_shape = mem.state_shape
    for _ in range(100):
        mem.update(np.ones(12))
    assert mem.state_shape == initial_shape == (2, 12)
    assert mem.scalar_state_budget == 24
    assert not hasattr(mem, "history")


def test_decay_recurrence_matches_declared_equation():
    mem = DecayMemory(d_model=3, rhos=(0.25, 0.75))
    mem.update(np.ones(3))
    mem.update(np.zeros(3))
    np.testing.assert_allclose(
        mem.read_components(),
        np.array([[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]]),
        atol=0.0,
        rtol=0.0,
    )


def test_resonant_memory_counts_real_and_imaginary_state_and_rotates():
    rho = 0.9
    omega = 0.4
    mem = ResonantMemory(d_model=2, modes=((rho, omega),))
    assert mem.scalar_state_budget == 4
    mem.update(np.ones(2))
    mem.update(np.zeros(2))
    expected = np.array(
        [
            [rho * np.cos(omega), rho * np.cos(omega)],
            [rho * np.sin(omega), rho * np.sin(omega)],
        ]
    )
    np.testing.assert_allclose(mem.read_components(), expected, atol=1e-15, rtol=1e-15)


def test_unstable_or_nonfinite_resonance_is_rejected():
    with pytest.raises(ValueError):
        ResonantMemory(d_model=12, modes=((1.01, 0.4),))
    with pytest.raises(ValueError):
        ResonantMemory(d_model=12, modes=((0.9, np.nan),))


def test_rytmi_window_memory_exposes_fast_minus_slow_and_slow_at_fixed_budget():
    mem = RytmiWindowMemory(
        d_model=2,
        pairs=((0.25, 0.75), (0.5, 0.9)),
    )
    assert mem.scalar_state_budget == 8
    assert mem.state_shape == (4, 2)

    mem.update(np.ones(2))
    mem.update(np.zeros(2))
    expected = np.array(
        [
            [-0.50, -0.50],
            [0.75, 0.75],
            [-0.40, -0.40],
            [0.90, 0.90],
        ]
    )
    np.testing.assert_allclose(mem.read_components(), expected, atol=0.0, rtol=0.0)
    assert not hasattr(mem, "history")


def test_rytmi_slow_only_ablation_keeps_same_stored_budget_but_hides_direction_components():
    full = RytmiWindowMemory(d_model=3, pairs=((0.25, 0.75), (0.5, 0.9)))
    slow_only = RytmiWindowMemory(
        d_model=3,
        pairs=((0.25, 0.75), (0.5, 0.9)),
        read_mode="slow_only",
    )
    for event in (np.ones(3), np.zeros(3)):
        full.update(event)
        slow_only.update(event)

    assert full.scalar_state_budget == slow_only.scalar_state_budget == 12
    np.testing.assert_allclose(
        slow_only.read_components(),
        np.array([[0.75, 0.75, 0.75], [0.90, 0.90, 0.90]]),
        atol=0.0,
        rtol=0.0,
    )
    assert full.read_components().shape == (4, 3)
    assert slow_only.read_components().shape == (2, 3)


def test_rytmi_raw_fast_slow_control_exposes_same_traces_without_coordinate_subtraction():
    raw = RytmiWindowMemory(
        d_model=2,
        pairs=((0.25, 0.75), (0.5, 0.9)),
        read_mode="raw_fast_slow",
    )
    direction = RytmiWindowMemory(
        d_model=2,
        pairs=((0.25, 0.75), (0.5, 0.9)),
        read_mode="direction",
    )
    for event in (np.ones(2), np.zeros(2)):
        raw.update(event)
        direction.update(event)

    assert raw.scalar_state_budget == direction.scalar_state_budget == 8
    np.testing.assert_allclose(
        raw.read_components(),
        np.array(
            [
                [0.25, 0.25],
                [0.75, 0.75],
                [0.50, 0.50],
                [0.90, 0.90],
            ]
        ),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        raw.read_components()[1::2],
        direction.read_components()[1::2],
        atol=0.0,
        rtol=0.0,
    )


def test_rytmi_window_pairs_require_faster_trace_to_decay_faster_than_slow_trace():
    with pytest.raises(ValueError, match="rho_fast"):
        RytmiWindowMemory(d_model=12, pairs=((0.9, 0.5),))
    with pytest.raises(ValueError):
        RytmiWindowMemory(d_model=12, pairs=((0.5, 1.0),))
    with pytest.raises(ValueError):
        RytmiWindowMemory(d_model=12, pairs=((0.5, np.nan),))
    with pytest.raises(ValueError, match="read_mode"):
        RytmiWindowMemory(d_model=12, pairs=((0.5, 0.9),), read_mode="oracle")


def test_static_memory_is_order_invariant_non_temporal_attacker():
    first = StaticMemory(d_model=4, state_slots=2)
    second = StaticMemory(d_model=4, state_slots=2)
    a = np.array([1.0, -0.5, 0.25, 2.0])
    b = np.array([-0.25, 1.5, 0.5, -1.0])
    first.update(a)
    first.update(b)
    second.update(b)
    second.update(a)
    np.testing.assert_allclose(
        first.read_components(), second.read_components(), atol=1e-15, rtol=1e-15
    )


def test_all_memories_read_fixed_size_summary_from_address_query():
    memories = [
        StaticMemory(d_model=12, state_slots=2),
        DecayMemory(d_model=12, rhos=(0.25, 0.9)),
        ResonantMemory(d_model=12, modes=((0.75, 0.25),)),
        RytmiWindowMemory(d_model=12, pairs=((0.25, 0.75),)),
    ]
    for mem in memories:
        mem.update(np.arange(12, dtype=float) / 12.0)
        summary = mem.read(np.linspace(-0.3, 0.4, 12))
        assert summary.shape == (12,)
        assert np.all(np.isfinite(summary))
        assert not hasattr(mem, "history")


def test_equal_budget_guard_accepts_parity_and_rejects_mismatch():
    matched = [
        StaticMemory(d_model=12, state_slots=2),
        DecayMemory(d_model=12, rhos=(0.25, 0.75)),
        ResonantMemory(d_model=12, modes=((0.9, 0.4),)),
        RytmiWindowMemory(d_model=12, pairs=((0.25, 0.75),)),
    ]
    assert assert_equal_state_budget(matched) == 24

    with pytest.raises(ValueError):
        assert_equal_state_budget(
            [
                StaticMemory(d_model=12, state_slots=1),
                DecayMemory(d_model=12, rhos=(0.25, 0.75)),
            ]
        )


def test_memories_reject_bad_update_and_query_shapes_and_nonfinite_values():
    mem = DecayMemory(d_model=12, rhos=(0.25, 0.75))
    with pytest.raises(ValueError):
        mem.update(np.zeros(11))
    bad = np.zeros(12)
    bad[3] = np.inf
    with pytest.raises(ValueError):
        mem.update(bad)
    with pytest.raises(ValueError):
        mem.read(np.zeros(11))
