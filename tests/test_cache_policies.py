import numpy as np
import pytest

from transformer_to_x.cache_policies import PolicySpec, policy_attention
from transformer_to_x.gpt2_numpy import MemorySpec, memory_attention


def _slow(q, k, v, scale, spec):
    """Plain-Python per-head reference of the same policies."""
    H, T, d = q.shape
    out = np.zeros_like(q)
    B, R = spec.slots, spec.recent_size
    for h in range(H):
        slots = []  # dicts: k, v, pos, score, mass
        for t in range(T):
            if len(slots) == B:
                nonrecent = [i for i, s in enumerate(slots) if s["pos"] <= t - R]
                vi = min(nonrecent, key=lambda i: (slots[i]["score"], i))
                victim = slots[vi]
                if spec.policy == "merge":
                    others = [i for i in nonrecent if i != vi]
                    if others:
                        cos = lambda a, b: a @ b / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12))
                        ti = max(others, key=lambda i: (cos(slots[i]["k"], victim["k"]), -i))
                        tg = slots[ti]
                        tot = tg["mass"] + victim["mass"]
                        tg["k"] = (tg["mass"] * tg["k"] + victim["mass"] * victim["k"]) / tot
                        tg["v"] = (tg["mass"] * tg["v"] + victim["mass"] * victim["v"]) / tot
                        tg["mass"] = tot
                        tg["score"] += victim["score"]
                slots[vi] = {"k": k[h, t].copy(), "v": v[h, t].copy(), "pos": t, "score": 0.0, "mass": 1.0}
            else:
                slots.append({"k": k[h, t].copy(), "v": v[h, t].copy(), "pos": t, "score": 0.0, "mass": 1.0})
            lg = np.array([scale * s["k"] @ q[h, t] + (np.log(s["mass"]) if spec.policy == "merge" else 0.0)
                           for s in slots])
            p = np.exp(lg - lg.max())
            p /= p.sum()
            for s, pi in zip(slots, p):
                s["score"] += pi
            out[h, t] = sum(pi * s["v"] for s, pi in zip(slots, p))
    return out


@pytest.mark.parametrize("policy", ["h2o", "merge"])
@pytest.mark.parametrize("slots,recent", [(6, None), (5, 2), (8, 1)])
def test_vectorised_policy_matches_slow_reference(policy, slots, recent):
    rng = np.random.default_rng(slots * 7 + (recent or 0))
    q, k, v = (rng.normal(size=(3, 30, 4)) for _ in range(3))
    spec = PolicySpec(policy, slots, recent)
    np.testing.assert_allclose(policy_attention(q, k, v, 0.6, spec), _slow(q, k, v, 0.6, spec), atol=1e-12)


@pytest.mark.parametrize("policy", ["h2o", "merge"])
def test_policy_with_room_for_everything_is_full_attention(policy):
    rng = np.random.default_rng(0)
    q, k, v = (rng.normal(size=(2, 15, 4)) for _ in range(3))
    np.testing.assert_allclose(policy_attention(q, k, v, 0.5, PolicySpec(policy, 15)),
                               memory_attention(q, k, v, 0.5, MemorySpec()), atol=1e-12)


def test_budgets_and_validation():
    assert PolicySpec("h2o", 48).scalar_budget(64) == 6144
    assert PolicySpec("merge", 47).scalar_budget(64) == 6063
    assert PolicySpec("h2o", 128).scalar_budget(64) == 16384
    assert PolicySpec("merge", 127).scalar_budget(64) == 16383
    with pytest.raises(ValueError):
        PolicySpec("h2o", 4, recent=4)


def test_h2o_keeps_a_token_everyone_attends_to():
    # token 0's key aligns with every query: H2O must never evict it.
    rng = np.random.default_rng(1)
    d = 4
    q = np.tile(np.array([1.0, 0, 0, 0]), (1, 40, 1)) + rng.normal(size=(1, 40, d)) * 0.01
    k = rng.normal(size=(1, 40, d)) * 0.1
    k[0, 0] = [20.0, 0, 0, 0]
    v = rng.normal(size=(1, 40, d))
    out = policy_attention(q, k, v, 1.0, PolicySpec("h2o", 6))
    np.testing.assert_allclose(out[0, -1], v[0, 0], atol=1e-3)
