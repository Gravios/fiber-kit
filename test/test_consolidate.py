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

# ── 3b. the curator's loop: iterative rounds + amplitude-aware cap ───────────
# Iteration: strip, re-template on the result, strip again — every round
# refits templates AND calibration quantiles, so the per-cluster distance
# self-adjusts.  Rounds must accumulate (>= single pass) and stop early on a
# converged fixture.
it1, t1 = fc.consolidate(get_waves, LAB, CHUNK, mode="strip", exclude=(0,),
                         strip_kw=dict(margin=0.85, iters=1), log=None)
it2, t2 = fc.consolidate(get_waves, LAB, CHUNK, mode="strip", exclude=(0,),
                         strip_kw=dict(margin=0.85, iters=2), log=None)
check(t2["strip"] >= t1["strip"], "iters=2 accumulates at least the single pass's moves")
check(int((it2[planted] == 3).sum()) >= int((it1[planted] == 3).sum())
      and set(it2[planted].tolist()) <= {2, 3},
      "a second round only adds planted recoveries, never mislabels")
_, t_conv = fc.consolidate(lambda ix: TW[np.asarray(ix)], TL, np.zeros(300, int),
                           mode="strip", exclude=(0,), strip_kw=dict(iters=5), log=None)
check(t_conv["strip"] == 0, "iteration stops early: a converged (twin) fixture trades nothing x5")

# cap_amp: D is normalized by the template's kernel RMS, so a FAINT cluster's
# distances run large and a fixed cap gates it far below its own core, while a
# BRIGHT cluster's cap sits near its core.  cap_amp refers the cap to the
# chunk's median template amplitude: faint caps loosen, bright caps tighten.
def at_D(tpl, n, targets, seed):
    """Spikes at EXACT strip distance from `tpl`: D(tpl + a*u) is linear in a,
    so each unit-noise direction is rescaled onto its target distance."""
    r = np.random.default_rng(seed)
    u = r.normal(0, 1.0, (n, NSAMP, NCH))
    t = fc._tpl_terms(tpl)
    Du, _ = fc._score_block((tpl[None] + u).reshape(n, -1), t)
    return tpl[None] + u * (np.asarray(targets) / Du)[:, None, None]

FAINT = TPL_A * 0.10
r21 = np.random.default_rng(21)
f_own = at_D(FAINT, 160, r21.uniform(0.20, 0.90, 160), seed=21)   # faint: D runs big
f_pl = at_D(FAINT, 25, r21.uniform(0.55, 0.70, 25), seed=22)      # beyond the 0.5 cap...
b_own = spikes(TPL_B, 300, noise=3.0, seed=23)                    # ...planted inside bright B
CW = np.concatenate([f_own, b_own, f_pl]).astype(np.float32)
CL = np.concatenate([np.full(160, 2), np.full(325, 3)])
c_pl = np.arange(160 + 300, 160 + 325)
def cgw(ix):
    return CW[np.asarray(ix)]
t_f = fc._tpl_terms(np.median(f_own, 0))
Dpl, _ = fc._score_block(f_pl.reshape(25, -1), t_f)
check(Dpl.min() > 0.5, "fixture: faint-shaped contamination sits BEYOND the fixed cap "
                       f"(min D_target {Dpl.min():.2f})")
cap0, _ = fc.strip_pass(cgw, CL, np.arange(485), exclude=(0,), twin_thr=None, rng=rng)
capA, _ = fc.strip_pass(cgw, CL, np.arange(485), exclude=(0,), twin_thr=None,
                        cap_amp=1.0, rng=rng)
check((cap0[c_pl] == 3).all(), "fixed cap: the faint cluster cannot claim its own spikes back")
check((capA[c_pl] == 2).sum() >= 15 and set(capA.tolist()) <= {2, 3},
      f"cap_amp=1 unlocks the faint claim ({int((capA[c_pl] == 2).sum())}/25 recovered)")
check((capA[:160] == 2).all() and (capA[160:460] == 3).all(),
      "cap_amp moves nothing else in the fixture")
# ...and the bright side TIGHTENS: donor spikes at exact D 0.36-0.44 from B are
# claimable under the fixed cap 0.5 but not once cap_amp pulls B's cap below.
band = at_D(TPL_B, 40, np.random.default_rng(31).uniform(0.36, 0.44, 40), seed=31)
BW = np.concatenate([b_own, f_own, band]).astype(np.float32)
BL = np.concatenate([np.full(len(b_own), 3), np.full(len(f_own), 2), np.full(len(band), 4)])
b_band = np.arange(len(BL) - len(band), len(BL))
def bgw(ix):
    return BW[np.asarray(ix)]
# min_tpl 50 leaves the band untemplated (no own column) and tgt_scale 1e9
# makes the CAP the binding radius, isolating the knob under test.
tight0, _ = fc.strip_pass(bgw, BL, np.arange(len(BL)), exclude=(0,), twin_thr=None,
                          tgt_scale=1e9, tgt_q=1.0, min_tpl=50, rng=rng)
tightA, _ = fc.strip_pass(bgw, BL, np.arange(len(BL)), exclude=(0,), twin_thr=None,
                          tgt_scale=1e9, tgt_q=1.0, min_tpl=50, cap_amp=1.0, rng=rng)
check((tight0[b_band] == 3).sum() > 0, "fixed cap: bright B claims the 0.35-0.45 band")
check((tightA[b_band] == 3).sum() == 0,
      "cap_amp=1 tightens the bright cap below the band (no claim)")

# ── 3b2. per-cluster template self-alignment (the curator's realign step) ────
ta, ta_n = fc.strip_pass(get_waves, LAB, np.flatnonzero(CHUNK == 0),
                         exclude=(0,), margin=0.85, tpl_align=True, rng=rng)
check(int((ta[planted] == 3).sum()) >= 17 and set(ta[planted].tolist()) <= {2, 3},
      "tpl_align: planted contamination still moves, nothing mislabels")
check((ta[LAB == 0] == 0).all() and (ta[LAB == 4] == 4).all(),
      "tpl_align: reserves and the look-alike's core still hold")

# ── 3c. the pure-claimant gate (only pure clusters strip) ────────────────────
# Cluster 2 = X, an A-cell CONTAMINATED with off-channel B spikes at spread
# scales: its own-gain IQR is wide, so with pure_iqr set it may not claim.
# Cluster 3 = BB, pure B: tight gain, a licensed claimant.  Cluster 4 = pure
# C holding 15 planted A spikes that only X could claim back.  Round 1: BB
# strips its B spikes out of X (gated clusters stay strippable FROM); X's
# gain tightens; round 2: X has EARNED claiming rights and recovers its
# planted spikes — the gate is recomputed every round.
TPL_C = bump(8, 3.0, (0, 3), (60.0, 45.0))
x_a = spikes(TPL_A, 150, noise=3.0, seed=41)
x_b = (TPL_B[None] * np.random.default_rng(42).uniform(0.7, 1.3, (60, 1, 1))
       + np.random.default_rng(43).normal(0, 3.0, (60, NSAMP, NCH)))
bb = spikes(TPL_B, 200, noise=3.0, seed=44)
cc = spikes(TPL_C, 120, noise=3.0, seed=45)
c_apl = spikes(TPL_A, 15, noise=3.0, seed=46)               # A spikes planted in C
PW = np.concatenate([x_a, x_b, bb, cc, c_apl]).astype(np.float32)
PL = np.concatenate([np.full(210, 2), np.full(200, 3), np.full(135, 4)])
p_bcont = np.arange(150, 210)                               # B contamination inside X
p_apl = np.arange(410 + 120, 410 + 135)                     # planted A inside C
def pgw(ix):
    return PW[np.asarray(ix)]
PKW = dict(own_q=0.5, pure_iqr=0.2)
one, _ = fc.consolidate(pgw, PL, np.zeros(len(PL), int), mode="strip",
                        strip_kw=dict(PKW, iters=1), exclude=(0,), log=None)
# BB's calibrated radius reaches only the near-unit-scale part of the planted
# ladder in one round — enough to push X's contamination under the IQR's
# quartile breakdown point (25%), which is what flips the gate next round.
check((one[p_bcont] == 3).sum() >= 15, "round 1: pure BB strips its spikes out of impure X "
                                       f"({int((one[p_bcont] == 3).sum())}/60)")
check((one[p_apl] == 4).all(), "round 1: impure X is gated and cannot claim its planted spikes")
two, _ = fc.consolidate(pgw, PL, np.zeros(len(PL), int), mode="strip",
                        strip_kw=dict(PKW, iters=3), exclude=(0,), log=None)
check((two[p_apl] == 2).sum() >= 12, "later rounds: X, now purified, has EARNED claiming rights "
                                     f"({int((two[p_apl] == 2).sum())}/15 recovered)")
check((two[:150] == 2).all() and (two[210:410] == 3).all() and (two[410:530] == 4).all(),
      "the gate moves nothing else")
ung, _ = fc.consolidate(pgw, PL, np.zeros(len(PL), int), mode="strip",
                        strip_kw=dict(own_q=0.5, iters=1), exclude=(0,), log=None)
check((ung[p_apl] == 2).sum() >= 12, "pure_iqr=0 (off): the impure X claims immediately, as before")

# ── 3d. the aggregate pool: dissolve hopeless clusters, kk reseeds them ──────
# A cluster whose remainder is hopeless (own-gain IQR above agg_iqr) dissolves
# into ONE per-chunk aggregate; the pool never templates but stays claimable;
# KlustaKwik over the pool seeds candidate templates later rounds vacuum toward.
mixA = (TPL_A[None] * np.random.default_rng(51).uniform(0.5, 1.5, (120, 1, 1))
        + np.random.default_rng(52).normal(0, 3.0, (120, NSAMP, NCH)))
mixB = spikes(TPL_B, 120, noise=3.0, seed=53)
mix2 = (TPL_A[None] * np.random.default_rng(54).uniform(0.5, 1.5, (100, 1, 1))
        + np.random.default_rng(55).normal(0, 3.0, (100, NSAMP, NCH)))
pureD = spikes(TPL_C, 140, noise=3.0, seed=56)
pureD2 = spikes(TPL_C, 140, noise=3.0, seed=57)
GW_ = np.concatenate([np.concatenate([mixA, mixB]), pureD, mix2, pureD2]).astype(np.float32)
GL_ = np.concatenate([np.full(240, 6), np.full(140, 2), np.full(100, 6), np.full(140, 2)])
GC_ = np.concatenate([np.zeros(380, int), np.ones(240, int)])
g_h0 = np.arange(0, 240)                                    # chunk-0 hopeless members
g_h1 = np.arange(380, 480)                                  # chunk-1 hopeless members
def ggw(ix):
    return GW_[np.asarray(ix)]
# dissolve only (reseed starved by a huge minnew): one NEW aggregate id per chunk
agg_lab, agg_tot = fc.consolidate(ggw, GL_, GC_, mode="strip", exclude=(0,),
                                  strip_kw=dict(iters=1, agg_iqr=0.5,
                                                agg_minnew=10 ** 6), log=None)
a0 = set(agg_lab[g_h0].tolist()) - {2, 6}
a1 = set(agg_lab[g_h1].tolist()) - {2, 6}
check(len(a0) == 1 and min(a0) > 6, "a hopeless cluster dissolves into ONE new aggregate id")
check(len(a1) == 1 and a1 != a0, "each chunk gets its OWN aggregate")
check((agg_lab[GL_ == 2] == 2).all(), "pure clusters are untouched by the dissolve")
check(agg_tot["agg"] >= 300 and agg_tot["agg_new"] == 0,
      "dissolve is counted; the starved reseed makes nothing")
# full arc (vacuum ON): dissolve -> kk finds the pooled shapes -> rounds vacuum
arc_lab, arc_tot = fc.consolidate(ggw, GL_, np.where(GC_ == 0, 0, -1), mode="strip",
                                  exclude=(0,),
                                  strip_kw=dict(iters=3, agg_iqr=0.5, agg_every=1,
                                                agg_minnew=30, agg_vac=1), log=None)
check(arc_tot["agg_new"] >= 2, f"kk reseeds the pool ({arc_tot['agg_new']} candidate template(s))")
new_ids = [c for c in np.unique(arc_lab[g_h0]) if c > 6]
coh = 0
for c in new_ids:
    m = np.flatnonzero(arc_lab[:480] == c)
    if m.size < 20: continue
    fa = np.mean(m < 120)                                    # fraction from the A-scale group
    coh += (fa >= 0.8 or fa <= 0.2)
check(coh >= 2, f"reseeded clusters are shape-coherent ({coh} of {len(new_ids)} pure by origin)")

# default = QUARANTINE: without agg_vac the pool's spikes are never claimed,
# even by a pure cluster their shapes match; with it, the vacuum runs.
qA = spikes(TPL_A, 150, noise=3.0, seed=61)
qmix = np.concatenate([
    TPL_A[None] * np.random.default_rng(62).uniform(0.5, 1.5, (120, 1, 1))
    + np.random.default_rng(63).normal(0, 3.0, (120, NSAMP, NCH)),
    spikes(TPL_B, 120, noise=3.0, seed=64)])
QW_ = np.concatenate([qA, qmix]).astype(np.float32)
QL_ = np.concatenate([np.full(150, 2), np.full(240, 6)])
q_mix = np.arange(150, 390)
def qgw(ix):
    return QW_[np.asarray(ix)]
QKW = dict(iters=3, agg_iqr=0.5, agg_minnew=10 ** 6)
qv0, t_v0 = fc.consolidate(qgw, QL_, np.zeros(390, int), mode="strip", exclude=(0,),
                           strip_kw=dict(QKW), log=None)
qv1, t_v1 = fc.consolidate(qgw, QL_, np.zeros(390, int), mode="strip", exclude=(0,),
                           strip_kw=dict(QKW, agg_vac=1), log=None)
pool0 = int((qv0[q_mix] > 6).sum()); pool1 = int((qv1[q_mix] > 6).sum())
nA0 = int((qv0[q_mix] == 2).sum()); nA1 = int((qv1[q_mix] == 2).sum())
check(t_v0["agg"] > 0 and pool0 > pool1,
      f"quarantine (default) holds the pool ({pool0} pooled vs {pool1} with the vacuum)")
check(nA1 > nA0, f"agg_vac=1 vacuums matching pool spikes back ({nA0} -> {nA1} into the pure cell)")

# excluded ids are HARD inert: even a template-perfect spike in cluster 0 stays
ex_pl = spikes(TPL_A, 6, noise=1.0, seed=58)
EW_ = np.concatenate([spikes(TPL_A, 150, noise=3.0, seed=59),
                      spikes(TPL_B, 150, noise=3.0, seed=60), ex_pl]).astype(np.float32)
EL_ = np.concatenate([np.full(150, 2), np.full(150, 3), np.zeros(6, int)])
ex_new, _ = fc.strip_pass(lambda ix: EW_[np.asarray(ix)], EL_, np.arange(306),
                          exclude=(0,), rng=rng)
check((ex_new[300:] == 0).all(), "excluded spikes never move, even template-perfect ones")

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

# 5b. park_min: sub-floor buckets collapse into ONE sweep atom per destination
#     sources 20/21/22 each donate ONE spike to fiber 3 (three 1-spike buckets),
#     source 23 donates three spikes (a real bucket), source 24 one spike to fiber 4.
pm_clu = np.array([2, 2, 2, 2, 2, 2, 2, 2])
pm_new = np.array([3, 3, 3, 3, 3, 3, 4, 2])
pm_child = np.array([20, 21, 22, 23, 23, 23, 24, 25])
pm_parent = {20: 2, 21: 2, 22: 2, 23: 2, 24: 2, 25: 2}
pm_out, pm_nxt = _park_consolidated(pm_clu, pm_new, pm_child, pm_parent, 30, park_min=2)
check((pm_out == pm_new).all(), "park_min: flat clu unchanged by the floor")
check(pm_child[0] == pm_child[1] == pm_child[2],
      "park_min: three 1-spike buckets to one dest share ONE sweep atom")
check(pm_child[3] == pm_child[4] == pm_child[5] and pm_child[3] != pm_child[0],
      "park_min: an at-floor bucket keeps its own atom")
check(pm_child[6] not in (pm_child[0], pm_child[3]),
      "park_min: a sub-floor bucket to ANOTHER dest gets that dest's sweep atom")
check(pm_parent[int(pm_child[0])] == 3 and pm_parent[int(pm_child[6])] == 4,
      "park_min: sweep atoms parent to their destination fiber")
check(pm_nxt == 33, "park_min: 3 atoms allocated (sweep@3, bucket 23, sweep@4)")
# floor off reproduces per-bucket parking on the same input
pm_child0 = np.array([20, 21, 22, 23, 23, 23, 24, 25])
_, pm_nxt0 = _park_consolidated(pm_clu, pm_new, pm_child0, dict(pm_parent), 30, park_min=0)
check(pm_nxt0 == 35, "park_min=0: historical per-bucket parking (5 atoms)")

print(f"\n{ran} checks, {fails} failed")
sys.exit(1 if fails else 0)
