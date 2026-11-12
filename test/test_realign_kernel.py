#!/usr/bin/env python3
"""fiber_realign's correlation window (--kernel / FK_REALIGN_KERNEL) is Klusters' (patch 0971,
klustersshared xcorr_weighted_test): the same template, the same seven spikes -- four clean at jitters
0, 2, -3, 1, two with a larger spike 14 samples after their peak, one all-zero -- give the same lags:
the whole window loses both collided spikes, a Gaussian of half-width 4 aligns them; weights of 1
everywhere reproduce the unweighted kernel; the zero spike never moves."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from fiber_kit import fiber_realign as fr   # noqa: E402

fails = 0


def check(ok, what):
    global fails
    print(("  ok:   " if ok else "  FAIL: ") + what); fails += not ok


N, C, P = 32, 2, 12
d = np.arange(N) - P
tmpl = np.stack([np.rint((-1000 * np.exp(-0.5 * d * d / 2.25) + 250 * np.exp(-0.5 * (d - 6) ** 2 / 6.0)) * g)
                 for g in (1.0, 0.6)], 1).astype(np.int16)                      # (N, C), the C++ test's template


def spike(jitter, collide):
    w = tmpl[(np.arange(N) + jitter) % N].astype(np.float64)                  # spike[s] = tmpl[s + jitter]
    if collide:
        dd = np.arange(N) - (P + 14)
        w += (-1600 * np.exp(-0.5 * dd * dd / 2.25))[:, None] * np.array([1.0, 0.9])[None, :]
    return np.rint(w).astype(np.int16)


jit = [0, 2, -3, 1]
spk = np.stack([spike(j, False) for j in jit] + [spike(2, True), spike(-3, True), np.zeros((N, C), np.int16)])
# one pass, the template given as the cluster mean would be contaminated; the C++ test scores against the
# fixed template, so do the same here through the kernel's scoring on a 1-iteration, template-only cluster:
# klusters_offsets builds its template from the spikes, so hand it the template as 40 extra clean spikes.
clean = np.stack([tmpl] * 40)
W = np.concatenate([spk, clean]); labels = np.full(len(W), 2)
off0, io0 = fr.klusters_offsets(W, labels, P, max_shift=6, iters=1, min_n=1, min_score=0.5)
offG, ioG = fr.klusters_offsets(W, labels, P, max_shift=6, iters=1, min_n=1, min_score=0.5, kernel="gauss", kernel_width=4)
offT, ioT = fr.klusters_offsets(W, labels, P, max_shift=6, iters=1, min_n=1, min_score=0.5, kernel="template")
want = [-j for j in jit]
check(list(io0[:4]) == want, f"whole window: the clean spikes' lags are minus their jitters ({[int(x) for x in io0[:4]]})")
check(list(ioG[:4]) == want, f"gauss +-4: the clean spikes' lags too ({[int(x) for x in ioG[:4]]})")
print(f"    collided: whole {[int(x) for x in io0[4:6]]}, gauss {[int(x) for x in ioG[4:6]]}, template {[int(x) for x in ioT[4:6]]} (expected [-2, 3])")
check(list(io0[4:6]) != [-2, 3], "whole window: the collision pulls a collided spike off its lag (or below threshold)")
check(list(ioG[4:6]) == [-2, 3], "gauss +-4: both collided spikes aligned on their own peak -- the C++ test's result")
check(list(ioT[4:6]) == [-2, 3], "template energy: both aligned as well")
check(io0[6] == 0 and ioG[6] == 0 and ioT[6] == 0, "the zero spike never moves")
w1 = fr.realign_kernel_weights("gauss", 4, P, N)
check(w1.shape == (N,) and abs(w1[P] - 1) < 1e-12 and abs(w1[P + 4] - np.exp(-2)) < 1e-12, "gauss: half-width = 2 sigma, 1 at the peak")
wh = fr.realign_kernel_weights("hann", 4, P, N)
check(wh[P] == 1 and wh[P + 4] == 0 and wh[P + 5] == 0 and 0 < wh[P + 2] < 1, "hann: raised cosine over +-width, 0 outside")
wt = fr.realign_kernel_weights("template", 0, P, N, tmpl)
check(wt.max() == 1 and wt.min() >= 0.02 and int(np.argmax(wt)) == P, "template: energy profile, max 1 at the peak, floor 0.02")
check(fr.realign_kernel_weights(None, 4, P, N) is None and fr.realign_kernel_weights("none", 4, P, N) is None, "none / None: no window")
print(f"test_realign_kernel: {'PASS' if not fails else 'FAIL'} ({fails} failed)")
sys.exit(1 if fails else 0)
