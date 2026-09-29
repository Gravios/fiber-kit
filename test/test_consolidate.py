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
# The calibrated radius (tgt_q=0.90) claims spikes that fit the target like
# its own core — and ~10% of the target's OWN spikes sit outside that radius
# by construction, so a statistically exchangeable contaminant moves at ~90%,
# never at 100%.  All movers must land on 3; nothing may go elsewhere.
n_pl = int((new[planted] == 3).sum())
check(n_pl >= 17 and set(new[planted].tolist()) <= {2, 3},
      f"strip moves the planted B spikes to 3 ({n_pl}/20; rest stay, none mislabelled)")
core = np.arange(40, 40 + 180)
check((new[core] == 2).sum() >= 175, "cluster 2's own A core stays (margin protects it)")
check((new[LAB == 0] == 0).all(), "reserve spikes never move")
check((new[LAB == 4] == 4).all(), "the 0.55·A look-alike keeps its core against A's template")

twin_a = spikes(TPL_A, 150, noise=8.0, seed=11)
twin_b = spikes(TPL_A * 0.97, 150, noise=8.0, seed=12)          # near-identical sibling
TW = np.concatenate([twin_a, twin_b]).astype(np.float32)
TL = np.concatenate([np.full(150, 2), np.full(150, 3)])
tw_new, tw_moved = fc.strip_pass(lambda ix: TW[np.asarray(ix)], TL, np.arange(300),
                                 exclude=(0,), rng=rng)
check(tw_moved == 0, "calibration + twin gate: near-identical twins trade nothing at defaults")

# Energy ladder: a cluster spanning amplitudes 0.3–1.3 of A next to a pure
# 0.35·A sibling — the amplitude-sensitive D wants to hand the big cluster's
# low-energy tail to the sibling; the twin gate (same shape at some scale)
# must refuse the pair, and demonstrably WOULD trade without it.
r13 = np.random.default_rng(13)
lad = (TPL_A[None] * r13.uniform(0.3, 1.3, (200, 1, 1))
       + r13.normal(0, 2.0, (200, NSAMP, NCH)))
sib = spikes(TPL_A, 120, scale=0.35, noise=1.5, seed=14)
LW = np.concatenate([lad, sib]).astype(np.float32)
LL = np.concatenate([np.full(200, 2), np.full(120, 3)])
def lgw(ix):
    return LW[np.asarray(ix)]
lad_new, lad_moved = fc.strip_pass(lgw, LL, np.arange(320), exclude=(0,), rng=rng)
check(lad_moved == 0, "twin gate: an energy ladder never sheds its tail to a scaled sibling")
lad_off, lad_off_moved = fc.strip_pass(lgw, LL, np.arange(320), exclude=(0,),
                                       own_q=0.5, tgt_q=1.0, twin_thr=None, rng=rng)
check(lad_off_moved > 0, "…and without the gate the ladder DOES shed (the gate is load-bearing)")

nomargin, _ = fc.strip_pass(get_waves, LAB, np.flatnonzero(CHUNK == 0),
                            exclude=(0,), margin=1e9, own_q=0.0, tgt_q=1.0, twin_thr=None, rng=rng)
check((nomargin[LAB == 4] == 2).any(), "…and WITHOUT the margin A does strip-mine it "
                                       "(the margin is load-bearing)")

both, tot = fc.consolidate(get_waves, LAB, CHUNK, mode="strip", exclude=(0,),
                           strip_kw=dict(margin=0.85), log=None)
check((both[LAB == 5] == 5).all(), "driver: chunk 1's B population never trades with chunk 0's")
# The driver realigns each chunk to its MIXED median; a few planted B spikes
# keep residual jitter against that A-dominated reference and score 0.5–0.7,
# just over the Klusters-default radius.  Most move, none mislabel; the
# stragglers are the price of the conservative default, tunable per run.
check(int((both[planted] == 3).sum()) >= 15 and set(both[planted]) <= {2, 3},
      "driver: planted contamination moves (conservative default; no mislabels)")
check(tot["strip"] == int((both != LAB).sum()), "driver: reported strip count matches the moves")

# ── 3. knob resolution: global yaml + stage-scoped mode override ─────────────
import argparse  # noqa: E402

for _k in [k for k in os.environ if k.startswith("FK_CONS_") or k.endswith("_CONS_MODE")]:
    del os.environ[_k]                                  # the resolution below must see only the dict
gy = {"FK_CONS_MODE": "strip", "FK_INTRA_CONS_MODE": "off", "FK_CONS_OWN_Q": "0.8"}
p_shared = argparse.ArgumentParser(); fc.add_consolidate_args(p_shared, gy)
p_intra = argparse.ArgumentParser(); fc.add_consolidate_args(p_intra, gy, stage="INTRA")
p_sess = argparse.ArgumentParser(); fc.add_consolidate_args(p_sess, gy, stage="SESSION")
check(p_shared.parse_args([]).cons_mode == "strip"
      and p_shared.parse_args([]).cons_own_q == 0.8,
      "FK_CONS_* in the global yaml reaches the parser defaults")
check(p_intra.parse_args([]).cons_mode == "off",
      "FK_<stage>_CONS_MODE overrides the shared mode for its own stage")
check(p_sess.parse_args([]).cons_mode == "strip",
      "a stage without a scoped key inherits the shared FK_CONS_MODE")
check(p_intra.parse_args(["--cons-mode", "both"]).cons_mode == "both",
      "CLI still beats every yaml key")
p_bool = argparse.ArgumentParser()
fc.add_consolidate_args(p_bool, {"FK_CONS_MODE": False})    # yaml parses a bare `off` as boolean
check(p_bool.parse_args([]).cons_mode == "off",
      "a bare yaml off (boolean False) reads as mode 'off'")

# ── 4. knn behaviour through the wrapper ─────────────────────────────────────
k_lab = LAB.copy()
k_planted = np.arange(40 + 200, 40 + 200 + 15)                              # 15 true-B spikes...
k_lab[k_planted] = 2                                                        # ...mislabelled into 2
k_new, k_moved, k_fresh, nid = fc.knn_pass(get_waves, k_lab, np.flatnonzero(CHUNK == 0),
                                           k=10, thr=0.3, minref=30, minnew=5,
                                           fold_thr=0.7, dims=5, exclude=(0,), rng=rng)
check((k_new[k_planted] == 3).all(), "knn folds the mislabelled bucket back into cluster 3")
check(k_moved >= 15, "knn reports at least the planted moves")
check(nid == int(k_lab.max()) + 1 + k_fresh, "new-cluster ids allocated contiguously")

# ── 5. fiber-session parking of consolidation moves ──────────────────────────
try:
    from fiber_kit.fiber_session import _park_consolidated  # noqa: E402
except ImportError:
    from fiber_session import _park_consolidated  # noqa: E402

pk_clu = np.array([0, 2, 2, 3, 3, 2, 3])
pk_new = np.array([0, 2, 3, 3, 3, 3, 2])          # spikes 2,5: 2->3 ; spike 6: 3->2
pk_child = np.array([0, 7, 7, 8, 8, 7, 8])
pk_parent = {7: 2, 8: 3}
pk_out, pk_nxt = _park_consolidated(pk_clu, pk_new, pk_child, pk_parent, 9)
check((pk_out == pk_new).all() and pk_out.dtype == np.int32, "parking returns the consolidated clu")
check(pk_child[1] == 7 and pk_child[3] == 8 and pk_child[0] == 0,
      "unmoved spikes keep their atoms")
check(pk_child[2] == pk_child[5] and pk_child[2] >= 9,
      "a moved (source atom -> dest fiber) bucket becomes ONE new atom")
check(pk_child[6] >= 9 and pk_child[6] != pk_child[2], "a different bucket gets its own atom")
check(pk_parent[int(pk_child[2])] == 3 and pk_parent[int(pk_child[6])] == 2,
      "new atoms parent to the destination fiber")
check(pk_nxt == 11 and set(pk_parent) == {7, 8, 9, 10}, "atom ids allocated contiguously")
_, pk_same = _park_consolidated(pk_new.copy(), pk_new, pk_child.copy(), dict(pk_parent), 11)
check(pk_same == 11, "a no-move pass allocates nothing")

print(f"\n{ran} checks, {fails} failed")
sys.exit(1 if fails else 0)
