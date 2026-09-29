#!/usr/bin/env python3
# test_amp_band.py — the post-linkage amplitude-band partition (fiber_intrachunk).
#
# The stderiv correlation gates are near-blind to amplitude, so same-shape
# cells of different sizes weld; the partition splits a unit's atoms where
# their top-cohort CEILINGS (raw waveforms) fall in separated log2 bands.
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import numpy as np  # noqa: E402  (path shim above, as every test here does)

try:
    from fiber_kit.fiber_intrachunk import _amp_band_partition  # noqa: E402
except ImportError:
    sys.path.insert(0, os.path.join(HERE, "..", "src", "fiber_kit"))
    from fiber_intrachunk import _amp_band_partition  # noqa: E402

NSAMP, NCH = 32, 4
fails = 0
ran = 0


def check(ok, what):
    global fails, ran
    ran += 1
    print(("  ok:   " if ok else "  FAIL: ") + what)
    if not ok:
        fails += 1


def bump(amp):
    t = np.zeros((NSAMP, NCH))
    t[:, 0] = amp * np.exp(-0.5 * ((np.arange(NSAMP) - 12) / 2.5) ** 2)
    t[:, 1] = -0.6 * t[:, 0]
    return t


def spikes(amp, n, seed):
    r = np.random.default_rng(seed)
    return bump(amp)[None] + r.normal(0, 2.0, (n, NSAMP, NCH))


# One welded unit (id 5): two big atoms at ceiling ~800, two at ~300 (2.7x),
# plus a tiny 10-spike atom near the low band.  A second, clean unit (id 6)
# at a single ceiling must be untouched, as must reserve ids 0/1.
W = np.concatenate([
    spikes(800, 60, 1), spikes(780, 50, 2),        # unit 5, high band (atoms 11, 12)
    spikes(300, 55, 3), spikes(310, 45, 4),        # unit 5, low band  (atoms 13, 14)
    spikes(295, 10, 5),                            # unit 5, tiny atom 15 -> attaches low
    spikes(500, 70, 6), spikes(505, 40, 7),        # unit 6 (atoms 16, 17)
    spikes(400, 8, 8),                             # reserve 1 (atom 18)
]).astype(np.float32)
U = np.concatenate([np.full(210, 5, np.int64), np.full(220 - 210, 5, np.int64),
                    np.full(110, 6, np.int64), np.full(8, 1, np.int64)])
CH = np.concatenate([np.full(60, 11), np.full(50, 12), np.full(55, 13), np.full(45, 14),
                     np.full(10, 15), np.full(70, 16), np.full(40, 17), np.full(8, 18)])


def gw(ix):
    return W[np.asarray(ix)]


new, n_extra = _amp_band_partition(U, CH, gw, top_frac=0.2, gap=0.5, min_n=25)
check(n_extra == 1, f"the welded unit splits into exactly two bands ({n_extra + 1})")
hi = new[:110]; lo = new[110:220]
check(len(set(hi.tolist())) == 1 and len(set(lo.tolist())) == 1 and hi[0] != lo[0],
      "high- and low-ceiling atoms land in different units, each band whole")
check({hi[0], lo[0]} >= {5}, "the most populous band keeps the original unit id")
check(len(set(new[210:220].tolist())) == 1 and new[210] == lo[0],
      "a sub-anchor atom attaches to the NEAREST band, never anchors one")
check((new[220:330] == 6).all(), "a single-band unit is untouched")
check((new[330:] == 1).all(), "reserve ids never band")

# Gap below the threshold: the same unit stays whole (1.34x < 2^0.5).
W2 = np.concatenate([spikes(800, 60, 9), spikes(600, 60, 10)]).astype(np.float32)
U2 = np.full(120, 5, np.int64)
CH2 = np.concatenate([np.full(60, 21), np.full(60, 22)])
new2, n2 = _amp_band_partition(U2, CH2, lambda ix: W2[np.asarray(ix)],
                               top_frac=0.2, gap=0.5, min_n=25)
check(n2 == 0 and (new2 == 5).all(), "a ceiling gap under the threshold never splits")

# A three-band ladder cuts at BOTH gaps (banding, not transitive pairwise).
W3 = np.concatenate([spikes(200, 50, 11), spikes(420, 50, 12),
                     spikes(900, 50, 13)]).astype(np.float32)
U3 = np.full(150, 7, np.int64)
CH3 = np.concatenate([np.full(50, 31), np.full(50, 32), np.full(50, 33)])
new3, n3 = _amp_band_partition(U3, CH3, lambda ix: W3[np.asarray(ix)],
                               top_frac=0.2, gap=0.5, min_n=25)
check(n3 == 2 and len(set(new3.tolist())) == 3,
      "a three-band ladder becomes three units (no transitive chaining)")

print(f"\n{ran} checks, {fails} failed")
sys.exit(1 if fails else 0)
