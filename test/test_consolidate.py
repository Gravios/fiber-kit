#!/usr/bin/env python3
# test_consolidate.py — the per-spike consolidation pass (fiber_consolidate).
#
# Three claims, each on synthetic waveforms with known ground truth:
#   1. METRIC PARITY — the vectorised strip score equals a literal per-spike
#      transcription of the Klusters loop (klustersdoc_strip.cpp): kernel
#      w=|T|, D = sqrt(Σw(x−T)²/Σw)/denom, g = Σw·x·T/Σw·T², per-channel
#      worst distance over the ≥5%-energy channels.
#   2. STRIP BEHAVIOUR — planted contaminant spikes move to the template
#      cluster; the margin stops a template from strip-mining an
#      energy-level look-alike's core (and moving WOULD happen without it);
#      reserve labels never move; chunks never exchange spikes.
#   3. KNN BEHAVIOUR — fiber_refine._knn_apply through the knn_pass wrapper
#      folds a planted mislabelled bucket back into its true cluster.
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import numpy as np  # noqa: E402  (path shim above, as every test here does)

try:
    from fiber_kit import fiber_consolidate as fc  # noqa: E402
except ImportError:
    sys.path.insert(0, os.path.join(HERE, "..", "src", "fiber_kit"))
    import fiber_consolidate as fc  # noqa: E402

NSAMP, NCH = 32, 4
rng = np.random.default_rng(7)
fails = 0
ran = 0


def check(ok, what):
    global fails, ran
    ran += 1
    print(("  ok:   " if ok else "  FAIL: ") + what)
    if not ok:
        fails += 1


def bump(center, width, chans, amp):
    t = np.zeros((NSAMP, NCH))
    x = np.arange(NSAMP)
    for c, a in zip(chans, amp):
        t[:, c] = a * np.exp(-0.5 * ((x - center) / width) ** 2)
    return t


TPL_A = bump(12, 2.5, (0, 1), (80.0, -55.0))
TPL_B = bump(18, 4.0, (2, 3), (-70.0, 40.0))


def spikes(tpl, n, noise=3.0, scale=1.0, seed=0):
    r = np.random.default_rng(seed)
    return tpl[None] * scale + r.normal(0, noise, (n, NSAMP, NCH))


# ── 1. metric parity against the literal Klusters loop ──────────────────────
def naive_scores(X, T):
    """Per-spike transcription of the C++ scoring loop (all channels gated
    the same way the strip gates them)."""
    w = np.abs(T)
    W = w.sum()
    E = (w * T * T).sum()
    denom = np.sqrt(E / W)
    Wc = w.sum(0)
    Ec = (w * T * T).sum(0)
    gate = (Ec >= 0.05 * Ec.max()) & (Wc > 0)
    D = np.empty(len(X))
    g = np.empty(len(X))
    worst = np.zeros(len(X))
    for i, x in enumerate(X):
        d = x - T
        q = (w * d * d).sum()
        D[i] = np.sqrt(q / W) / denom
        g[i] = (w * x * T).sum() / E
        for c in np.flatnonzero(gate):
            qc = (w[:, c] * d[:, c] * d[:, c]).sum()
            worst[i] = max(worst[i], np.sqrt(qc / Wc[c]) / np.sqrt(Ec[c] / Wc[c]))
    return D, g, worst


X = rng.normal(0, 30.0, (64, NSAMP, NCH)) + TPL_A[None] * rng.uniform(0.3, 1.6, (64, 1, 1))
t = fc._tpl_terms(TPL_A)
Dv, gv = fc._score_block(X.reshape(64, -1), t)
Dn, gn, wn = naive_scores(X, TPL_A)
check(np.allclose(Dv, Dn, atol=1e-9), "vectorised D equals the per-spike Klusters loop")
check(np.allclose(gv, gn, atol=1e-9), "vectorised g equals the per-spike Klusters loop")
check(np.allclose(fc._chan_worst(X.reshape(64, -1), t), wn, atol=1e-9),
      "per-channel worst distance equals the per-spike loop")

# ── 2. strip behaviour ───────────────────────────────────────────────────────
# Chunk 0: cluster 2 = A (200 spikes, of which 20 planted B-shaped), cluster
# 3 = B (150).  Cluster 4 = 0.55·A, an energy-level look-alike of A.  Chunk 1:
# cluster 5 = a second pure-B population that must NOT trade with chunk 0.
# Cluster 0 = noise-floor reserve, excluded.
parts = [
    (0, spikes(np.zeros((NSAMP, NCH)), 40, noise=25.0, seed=1)),           # 0: reserve
    (2, np.concatenate([spikes(TPL_A, 180, seed=2), spikes(TPL_B, 20, seed=3)])),
    (3, spikes(TPL_B, 150, seed=4)),
    (4, spikes(TPL_A, 120, scale=0.55, noise=1.5, seed=5)),
    (5, spikes(TPL_B, 140, seed=6)),
]
WAVES = np.concatenate([w for _, w in parts]).astype(np.float32)
LAB = np.concatenate([np.full(len(w), c) for c, w in parts])
CHUNK = np.concatenate([np.full(len(w), 1 if c == 5 else 0) for c, w in parts])
planted = np.arange(40 + 180, 40 + 200)                                     # the 20 B spikes inside 2


def get_waves(idx):
    return WAVES[np.asarray(idx)]


new, moved = fc.strip_pass(get_waves, LAB, np.flatnonzero(CHUNK == 0),
                           exclude=(0,), margin=0.85, rng=rng)
check((new[planted] == 3).all(), "strip moves every planted B spike from cluster 2 to 3")
core = np.arange(40, 40 + 180)
check((new[core] == 2).sum() >= 175, "cluster 2's own A core stays (margin protects it)")
check((new[LAB == 0] == 0).all(), "reserve spikes never move")
check((new[LAB == 4] == 4).all(), "the 0.55·A look-alike keeps its core against A's template")

nomargin, _ = fc.strip_pass(get_waves, LAB, np.flatnonzero(CHUNK == 0),
                            exclude=(0,), margin=1e9, rng=rng)
check((nomargin[LAB == 4] == 2).any(), "…and WITHOUT the margin A does strip-mine it "
                                       "(the margin is load-bearing)")

both, tot = fc.consolidate(get_waves, LAB, CHUNK, mode="strip", exclude=(0,),
                           strip_kw=dict(margin=0.85), log=None)
check((both[LAB == 5] == 5).all() and (both[planted] == 3).all(),
      "driver: chunk 1's B population never trades with chunk 0's")
check(tot["strip"] == int((both != LAB).sum()), "driver: reported strip count matches the moves")

# ── 3. knn behaviour through the wrapper ─────────────────────────────────────
k_lab = LAB.copy()
k_planted = np.arange(40 + 200, 40 + 200 + 15)                              # 15 true-B spikes...
k_lab[k_planted] = 2                                                        # ...mislabelled into 2
k_new, k_moved, k_fresh, nid = fc.knn_pass(get_waves, k_lab, np.flatnonzero(CHUNK == 0),
                                           k=10, thr=0.3, minref=30, minnew=5,
                                           fold_thr=0.7, dims=5, exclude=(0,), rng=rng)
check((k_new[k_planted] == 3).all(), "knn folds the mislabelled bucket back into cluster 3")
check(k_moved >= 15, "knn reports at least the planted moves")
check(nid == int(k_lab.max()) + 1 + k_fresh, "new-cluster ids allocated contiguously")

print(f"\n{ran} checks, {fails} failed")
sys.exit(1 if fails else 0)
