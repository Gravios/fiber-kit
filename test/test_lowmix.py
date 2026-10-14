"""fiber_lowmix: the split finds a two-source mixture under spatially correlated noise and leaves a
single source alone (the negative control), the local-rate refractory test separates one neuron from
two, and the verdict rules keep the unit's core."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from fiber_kit import fiber_lowmix as lm  # noqa: E402

NS, NC, SR = 42, 8, 32552.0


def _src(t0, width, chans, sign=-1.0, amp=400.0):
    t = np.arange(NS)
    w = sign * np.exp(-0.5 * ((t - t0) / width) ** 2) - 0.4 * sign * np.exp(-0.5 * ((t - t0 - 3 * width) / (2 * width)) ** 2)
    prof = np.exp(-0.5 * ((np.arange(NC) - chans) / 1.5) ** 2)
    return amp * np.outer(w, prof)


def _noise(n, rng, rms=120.0):
    # spatially and temporally correlated background: smoothed white noise mixed across channels
    z = rng.standard_normal((n, NS + 8, NC))
    k = np.exp(-0.5 * (np.arange(-4, 5) / 1.5) ** 2); k /= k.sum()
    z = np.apply_along_axis(lambda v: np.convolve(v, k, "valid"), 1, z)
    M = np.exp(-np.abs(np.subtract.outer(np.arange(NC), np.arange(NC))) / 2.0)
    z = z @ M
    return (rms * z / z.std()).astype(np.float32)


def _setup(seed=0):
    rng = np.random.default_rng(seed)
    P = lm.smoother(NS, 20)
    noise = lm.prep(_noise(6000, rng), P).reshape(6000, -1).astype(np.float64)
    return rng, P, noise, lm.whitener(noise)


def test_smoother_keeps_low_order_polynomials():
    P = lm.smoother(NS, 6)
    x = np.linspace(-1, 1, NS)
    y = 3 * x ** 5 - x ** 2 + 0.5
    assert np.allclose(P @ y, y, atol=1e-8)
    assert np.allclose(lm.smoother(NS, 0), np.eye(NS))


def test_split_finds_two_sources_and_leaves_one_alone():
    rng, P, noise, Wh = _setup()
    A, B = _src(20, 1.5, 5), _src(18, 3.0, 2, amp=350.0)
    n = 1200
    amp = rng.gamma(20, 1 / 20, n)[:, None, None]
    lab = rng.integers(0, 2, n)
    X = np.where(lab[:, None, None] == 0, A, B) * amp + _noise(n, rng)
    s = lm.split_unit(lm.prep(X, P), noise, Wh, seed=0)
    assert s["split"], s
    from sklearn.metrics import adjusted_rand_score
    assert adjusted_rand_score(lab, s["labels"]) > 0.9
    # negative control: ONE source with the same amplitude spread and noise
    X1 = A * amp + _noise(n, rng)
    s1 = lm.split_unit(lm.prep(X1, P), noise, Wh, seed=0)
    assert not s1["split"], s1


def _train(rate, T, rng, refrac_s=0.002):
    t = np.cumsum(rng.exponential(1 / rate, int(rate * T * 1.3)) + refrac_s)
    return np.round(t[t < T] * SR).astype(np.int64)


def test_local_refractory_one_vs_two_neurons():
    rng = np.random.default_rng(1)
    one = _train(20, 600, rng)
    half = rng.random(one.size) < 0.3
    r = lm.local_refractory(one[half], one[~half], SR)
    assert r["verdict"] == "allow", r
    other = _train(20, 600, rng)
    r2 = lm.local_refractory(other[:3000], one, SR)
    assert r2["verdict"] == "veto", r2
    r3 = lm.local_refractory(other[:5], one[:50], SR)
    assert r3["verdict"] == "abstain", r3


def test_decide_keeps_the_core_and_gates_copies_on_refractory():
    g = [dict(n=500, cos=0.80, refrac="abstain"), dict(n=300, cos=0.97, refrac="allow"),
         dict(n=200, cos=0.96, refrac="abstain"), dict(n=100, cos=0.92, refrac="abstain"),
         dict(n=80, cos=0.70, refrac="abstain")]
    v = [x["verdict"] for x in lm.decide([dict(d) for d in g])]
    assert v == ["keep", "copy", "new", "new", "novel"], v
    v2 = [x["verdict"] for x in lm.decide([dict(d) for d in g], copy_on_abstain=True)]
    assert v2 == ["keep", "copy", "copy", "new", "novel"], v2
    # all copies: no core is invented
    v3 = [x["verdict"] for x in lm.decide([dict(n=50, cos=0.99, refrac="allow")])]
    assert v3 == ["copy"]


def test_noise_snippets_avoid_spikes():
    rng = np.random.default_rng(2)
    sig = rng.standard_normal((200000, NC)).astype(np.float32)
    res = np.sort(rng.choice(np.arange(100, 199000), 2000, replace=False))
    snips_t = []

    def read(a, b):
        snips_t.append(a)
        return sig[a:b]
    S = lm.noise_snippets(read, sig.shape[0], res, NS, 500, rng)
    assert S.shape == (500, NS, NC)
    t = np.array(snips_t)
    j = np.searchsorted(res, t)
    d_prev = t - res[np.maximum(j - 1, 0)]
    d_next = res[np.minimum(j, res.size - 1)] - t
    assert ((d_prev > NS) | (j == 0)).all() and ((d_next > 2 * NS) | (j == res.size)).all()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("ok", fn.__name__)
    print(f"\n{len(fns)} passed")
