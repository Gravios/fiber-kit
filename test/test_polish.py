"""fiber-polish: the strip -> pool loop, the island pass, the reverse-strip merge, the CLI.

  * gain_residual: a scaled copy of a template is no distance; another cell is; no template-shaped
    signal is inf (never a small number);
  * kd_leaves partitions every point into leaves of at most `leaf`;
  * merge_pass unions two fragments of one cell and leaves a different cell alone; a pair is scored
    under the CLEANER cluster's tails, so a cluster whose own tails are inflated by contamination
    does not accept a different cell (the failure that collapsed a whole sort at threshold 0.3);
  * strip_pool_pass parks displaced intruders, pools them with the sub-`pool_max` clusters, and
    leaves NOTHING in cluster 0 or 1 -- the remainder is a real residue cluster below the reserved block;
  * island_pass moves a foreign island to its own cell and leaves a homogeneous cluster whole
    (skipped without the [consensus] extra);
  * the CLI reads a session, writes <base>.clu.<method>.<group>.<in>_polish, and scores against --gt-clu.
"""
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from fiber_kit import fiber_chunks as fch               # noqa: E402
from fiber_kit import fiber_polish as fpo               # noqa: E402
from fiber_kit import neuro_io as nio                   # noqa: E402

fails = ran = 0


def check(ok, what):
    global fails, ran
    ran += 1
    print(("  ok:   " if ok else "  FAIL: ") + what)
    if not ok:
        fails += 1


S, C = 40, 8
t = np.arange(S)
bump = np.exp(-((t - 20) ** 2) / 8.0)
wide = np.exp(-((t - 21) ** 2) / 30.0)


def cell(n, chans, amps, shape, rng, noise=3.0, jitter=0.15):
    w = np.zeros((n, S, C))
    g = 1.0 + jitter * rng.standard_normal(n)
    for c, a in zip(chans, amps):
        w[:, :, c] = -a * g[:, None] * shape[None, :]
    return w + rng.normal(0, noise, w.shape)


rng = np.random.default_rng(3)
A = cell(600, (0, 1), (100, 60), bump, rng)             # cell A, to be split into two fragments
B = cell(300, (4, 5), (90, 70), bump, rng)              # cell B, elsewhere on the probe
D = cell(300, (0, 1), (100, 60), wide, rng)             # cell D: A's channels, a different shape
W = np.concatenate([A, B, D]).astype(np.float32)
lab0 = np.concatenate([np.full(300, 2), np.full(300, 3), np.full(300, 4), np.full(300, 5)]).astype(np.int64)
raw = lambda ix: np.asarray(W[np.asarray(ix)], float)   # noqa: E731
feats = lambda ix: W[np.asarray(ix)].reshape(len(ix), -1)[:, ::7].astype(float)   # noqa: E731

# ── gain_residual ──
TA, TB, TD = np.median(A, 0), np.median(B, 0), np.median(D, 0)
check(fpo.gain_residual(0.6 * TA, TA) < 1e-9, "gain_residual: a scaled copy of the template is no distance")
check(fpo.gain_residual(TD, TA) > 20 * fpo.gain_residual(np.median(A[:300], 0), np.median(A[300:], 0)),
      "gain_residual: another shape on the same channels is far; two halves of one cell are not")
check(np.isinf(fpo.gain_residual(TB, TA)), "gain_residual: no template-shaped signal is inf, never a small number")

# ── kd_leaves ──
P = rng.standard_normal((1000, 2))
lv = fpo.kd_leaves(P, 300)
check(sorted(np.concatenate(lv).tolist()) == list(range(1000)) and max(len(x) for x in lv) <= 300,
      "kd_leaves: a partition of every point into leaves of at most `leaf`")

# ── merge ──
m_lab, m_un = fpo.merge_pass(lab0, raw, thr=0.7, rng=np.random.default_rng(0))
check(m_lab[0] == m_lab[599] and m_un == 1, "merge: the two fragments of cell A are unioned, in one union")
check(len({int(m_lab[0]), int(m_lab[600]), int(m_lab[900])}) == 3, "merge: cells B and D are left alone")
# a contaminated cluster (A + 45% D) has inflated tails; under ITS tails D would be accepted
mix = np.concatenate([np.arange(0, 300), np.arange(900, 1150)])
eA = fpo._entry(raw(np.arange(300, 600)), 3.0)
eMix = fpo._entry(raw(mix), 3.0)
eD = fpo._entry(raw(np.arange(1150, 1200)), 3.0)
check(eMix["thr"][0] > 3 * eA["thr"][0], "merge: a contaminated cluster's own on-channel tail is inflated")
check(fpo.pair_acceptance(eMix, eA) < 0.7,
      "merge: scored under the CLEANER cluster's tails, the contaminated one is not merged into the clean one")
check(0.0 <= fpo.pair_acceptance(eA, eD) < 0.3, "merge: a different shape on the same channels is not accepted")

# ── strip -> pool ──
sp = lab0.copy()
sp[1190:] = 9                                            # a 10-spike fragment: below pool_max
intr = W.copy()
intr[:30, :, 4] += -80 * bump                            # 30 spikes of A1 carry a second cell on channel 4
intr[:30, :, 5] += -70 * bump
rawi = lambda ix: np.asarray(intr[np.asarray(ix)], float)   # noqa: E731
stub = lambda F, max_clusters, min_size, seed: np.full(len(F), -1)   # noqa: E731  (no pool cluster is formed)
s_lab, s_st, res_id = fpo.strip_pool_pass(sp, rawi, feats, strip_min=100, pool_max=50, rkk=stub,
                                          rng=np.random.default_rng(0))
check((s_lab[:30] == res_id).sum() >= 27, "strip: the displaced intruders are parked out of their cluster")
check((s_lab[1190:] == res_id).all(), "pool: a cluster below pool_max joins the pool whole")
check(res_id > 1 and not fch.is_reserved(np.array([res_id]))[0] and not np.isin(s_lab, (0, 1)).any(),
      "pool: the remainder is a real residue cluster -- nothing is left in cluster 0 or 1")
two = lambda F, max_clusters, min_size, seed: (np.arange(len(F)) % 2)   # noqa: E731
b_lab, b_st, _ = fpo.strip_pool_pass(sp, rawi, feats, strip_min=100, pool_max=50, pool_min=5, rkk=two,
                                     rng=np.random.default_rng(0))
check(b_st["born"] == 2 and b_st["residue"] == 0, "pool: the rkk's clusters become new clusters with fresh ids")

# ── the driver ──
stages = []
p_lab, p_res = fpo.polish(sp, rawi, feats, loops=1, rounds=1, island=0, merge=1, rkk=stub,
                          on_stage=lambda tag, lab: stages.append(tag))
check(not np.isin(p_lab, (0, 1)).any() and len(stages) == 2, "polish: loop then merge, and no spike ends in cluster 0 or 1")
check(p_lab[100] == p_lab[500], "polish: the fragments of cell A end in one cluster")
z_lab, _ = fpo.polish(lab0, raw, feats, loops=0, rounds=0)
check((z_lab == lab0).all(), "polish: loops 0 and rounds 0 change nothing")

# ── islands (needs the [consensus] extra) ──
try:
    import networkx  # noqa: F401
    import openTSNE  # noqa: F401
    have = True
except ImportError:
    have = False
if have:
    il = np.concatenate([np.full(600, 2), np.full(300, 4), np.full(300, 5)]).astype(np.int64)
    il[1150:1200] = 2                                    # 50 spikes of cell D sit in cluster A
    F = W.reshape(len(W), -1)[:, ::5].astype(float)
    i_lab, i_st = fpo.island_pass(il, raw, F, leaf=300, runs=4, min_clu=60, margin=0.5,
                                  rng=np.random.default_rng(0))
    check((i_lab[1150:1200] == 5).mean() >= 0.8, "islands: a foreign island moves to its own cell")
    check((i_lab[:600] == 2).mean() >= 0.97 and (i_lab[600:900] == 4).all(),
          "islands: the homogeneous clusters keep their spikes (an island moves only on a better fit)")
else:
    print("  skip: island pass (openTSNE / networkx not installed)")

# ── score_against ──
truth = np.concatenate([np.full(600, 7), np.full(300, 8), np.full(300, 9)])
nb, pur, good, comp = fpo.score_against(lab0, truth)
check(nb == 4 and pur == 1.0 and good == 1.0 and abs(comp - 0.75) < 1e-9,
      "score_against: an over-split but pure sort is 100% pure and good, 75% complete")

# ── the CLI ──
with tempfile.TemporaryDirectory() as d:
    base = os.path.join(d, "sess")
    open(base + ".yaml", "w").write(
        "acquisitionSystem:\n  nChannels: 8\n  samplingRate: 32552\n  nBits: 16\n"
        "spikeDetection:\n  channelGroups:\n  - channels: [0, 1, 2, 3, 4, 5, 6, 7]\n"
        f"    nSamples: {S}\n    peakSampleIndex: 20\n    nFeatures: 3\n")
    nio.write_res_file(base + ".res.1", np.arange(len(W)) * 100)
    W.astype(np.int16).tofile(base + ".spk.standard.1")
    nio.write_clu(base, 1, sp, variant="standard", tag="in")
    nio.write_clu_file(os.path.join(d, "gt.clu"), truth)
    old = os.getcwd()
    os.chdir(d)
    try:
        fpo.main(["sess", "1", "--clu-method", "standard", "--clu-stage", "in", "--island", "0",
                  "--loops", "1", "--rounds", "1", "--gt-clu", os.path.join(d, "gt.clu")])
    finally:
        os.chdir(old)
    out = base + ".clu.standard.1.in_polish"
    check(os.path.exists(out), "CLI: writes <base>.clu.<method>.<group>.<in>_polish")
    if os.path.exists(out):
        _, o = nio.read_clu_file(out, n_spikes=len(W))
        o = np.asarray(o)
        check(not np.isin(o, (0, 1)).any() and o[100] == o[500],
              "CLI: no spike in cluster 0 or 1, and cell A's fragments are one cluster")

print(f"\n{ran} checks, {fails} failed")
sys.exit(1 if fails else 0)
