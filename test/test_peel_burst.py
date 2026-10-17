"""fiber_ccg.burst_evidence and fiber-peel's burst pass.

Simulated 20-min trains (32552 Hz):
  positives  a complex-spike bursting cell cut by amplitude (first spike ->
             'big', later burst spikes -> 'small'), plain and theta-modulated;
  negatives  independent cells; theta co-modulated cells; monosynaptic
             excitation (+1.5-3 ms); the big fragment of one bursting cell
             against the small fragment of another; two theta interneurons.

Checks:
  * burst_evidence says 'burst' on the positives and not on the independent,
    theta, cross-cell or interneuron pairs;
  * the monosynaptic pair DOES show the forward peak -- which is why the gate
    needs the refractory dip as well -- and the composite gate refuses it;
  * composite gate (refractory 'allow' AND 'burst'): every positive passes,
    every negative fails;
  * peel_agglomerate with fragments at footprint cosine 0.85 (below
    foot_lo 0.90): without the burst pass nothing merges (NEGATIVE CONTROL);
    with burst_lo 0.80 the burst pair merges and neither the independent
    nor the monosynaptic look-alike does.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fiber_kit import fiber_ccg as cg       # noqa: E402
from fiber_kit import fiber_peel as fp      # noqa: E402

SR = 32552.0
DUR = int(20 * 60 * SR)
MS = SR / 1000.0


def poisson(rate, rng, refr=2.0, mod=None):
    rmax = rate * (1 + (mod[0] if mod else 0))
    t = np.sort(rng.uniform(0, DUR, rng.poisson(rmax * DUR / SR)))
    if mod:
        m, f, ph = mod
        t = t[rng.uniform(0, 1, t.size) < (1 + m * np.cos(2 * np.pi * f * t / SR + ph)) / (1 + m)]
    out, last = [], -1e18
    for x in t:
        if x - last > refr * MS:
            out.append(x); last = x
    return np.array(out)


def bursting(rate, rng, pb=0.4, mod=None):
    big, small = [], []
    for t in poisson(rate, rng, refr=20, mod=mod):
        big.append(t)
        if rng.random() < pb:
            tb = t
            for _ in range(rng.integers(1, 4)):
                tb += rng.uniform(3, 8) * MS
                small.append(tb)
    return np.array(big), np.sort(np.array(small))


def ints(t):
    return np.sort(np.asarray(t).astype(np.int64))


def composite(a, b):
    a, b = ints(a), ints(b)
    rg = cg.refractory_gate(a, b, DUR, cg.refrac_samples(2.0, SR), thr=0.3, min_exp=5.0)
    be = cg.burst_evidence(a, b, SR)
    return rg["verdict"] == "allow" and be["verdict"] == "burst", rg, be


def main():
    ok = 0

    def check(name, cond):
        nonlocal ok
        assert cond, name
        ok += 1
        print(f"  ok  {name}")

    rng = np.random.default_rng(0)
    pos = {"bursting cell cut by amplitude": bursting(2.0, rng),
           "theta-modulated bursting cell": bursting(2.0, rng, mod=(0.8, 8.0, 0))}
    a10 = poisson(10, rng)
    sel = rng.random(a10.size) < 0.3
    b1, _ = bursting(2.0, rng); _, s2 = bursting(2.0, rng)
    neg = {"independent 2 / 5 Hz": (poisson(2, rng), poisson(5, rng)),
           "theta co-modulated 5 / 5 Hz": (poisson(5, rng, mod=(0.9, 8, 0)), poisson(5, rng, mod=(0.9, 8, 0.3))),
           "monosynaptic 10 Hz -> +1.5-3 ms": (a10, np.sort(np.r_[a10[sel] + rng.uniform(1.5, 3.0, sel.sum()) * MS,
                                                                   poisson(3, rng)])),
           "big of cell A vs small of cell B": (b1, s2),
           "two theta interneurons 20 Hz": (poisson(20, rng, mod=(0.5, 8, 0)), poisson(20, rng, mod=(0.5, 8, 0)))}

    for name, (a, b) in pos.items():
        allowed, rg, be = composite(a, b)
        check(f"{name}: burst signature (fwd {be['forward']}, bwd {be['backward']}, exp {be['expected']:.0f})",
              be["verdict"] == "burst")
        check(f"{name}: composite gate allows the merge", allowed)
    for name, (a, b) in neg.items():
        allowed, rg, be = composite(a, b)
        if name.startswith("monosynaptic"):
            check(f"{name}: forward peak present (burst_evidence alone would say yes)", be["verdict"] == "burst")
            check(f"{name}: refractory vetoes it (ratio {rg['ratio']:.2f})", rg["verdict"] == "veto")
        else:
            check(f"{name}: no burst signature ({be['verdict']})", be["verdict"] != "burst")
        check(f"{name}: composite gate refuses the merge", not allowed)

    # ── peel_agglomerate: fragments at cosine 0.85, below foot_lo 0.90 ──
    d = 200
    u = rng.normal(size=d); u /= np.linalg.norm(u)

    def at_cos(c, amp):
        v = rng.normal(size=d); v -= (v @ u) * u; v /= np.linalg.norm(v)
        return amp * (c * u + np.sqrt(1 - c * c) * v)

    big, small = pos["bursting cell cut by amplitude"]
    mono_a, mono_b = neg["monosynaptic 10 Hz -> +1.5-3 ms"]
    foot = [u * 1.0, at_cos(0.85, 0.6),            # 0, 1: one bursting cell
            at_cos(0.85, 0.9),                     # 2: an independent cell, same site
            at_cos(0.85, 1.2), at_cos(0.85, 0.7)]  # 3, 4: a monosynaptic pair (3 drives 4)
    times = [ints(big), ints(small), ints(poisson(4, rng)), ints(mono_a), ints(mono_b)]
    # the monosynaptic pair's footprints are 0.85 to u, so ~0.72 to each other: give pair 3-4 their own axis
    w = rng.normal(size=d); w -= (w @ u) * u; w /= np.linalg.norm(w)
    foot[3] = 1.2 * w
    vv = rng.normal(size=d); vv -= (vv @ u) * u + (vv @ w) * w; vv /= np.linalg.norm(vv)
    foot[4] = 0.7 * (0.85 * w + np.sqrt(1 - 0.85 ** 2) * vv)
    counts = [len(t) for t in times]
    lab0 = fp.peel_agglomerate(foot, times, counts, DUR, SR)
    check("negative control: without the burst pass nothing below foot_lo merges", len(set(lab0)) == 5)
    st = {}
    lab1 = fp.peel_agglomerate(foot, times, counts, DUR, SR, burst_lo=0.80, stats=st)
    check("burst pass merges the bursting cell's two fragments", lab1[0] == lab1[1])
    check("burst pass leaves the independent same-site cell alone", lab1[2] not in (lab1[0], lab1[3], lab1[4]))
    check("burst pass leaves the monosynaptic pair apart", lab1[3] != lab1[4])
    check("stats report exactly one burst merge", st.get("n_burst_merges") == 1)
    check("above foot_lo the anneal is unchanged by the burst pass",
          fp.peel_agglomerate([u, u * 0.99], times[:2], counts[:2], DUR, SR) ==
          fp.peel_agglomerate([u, u * 0.99], times[:2], counts[:2], DUR, SR, burst_lo=0.8))

    print(f"test_peel_burst: {ok} checks passed")


if __name__ == "__main__":
    main()
