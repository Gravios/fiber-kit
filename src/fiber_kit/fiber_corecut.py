#!/usr/bin/env python3
# ══════════════════════════════════════════════════════════════════════════════
#  fiber_corecut.py  —  the core of a cluster, and the spikes that are not it
#
#  A numpy port of neurosuite-3's src/klusters/src/corecut.h (Klusters: Actions ->
#  Split Off Spurious Spikes), step for step, so fiber-kit can make the same cut
#  without Klusters.  The header is the reference; where the two could differ --
#  which median of an even count, which of two equal minima, how a half is
#  rounded -- this file does what the header does.  What it cannot reproduce is
#  the summation order of the float products in the ordering loop (BLAS here, a
#  plain loop there), so two spikes at almost the same distance from the running
#  mean can swap places.  Measured on sirotaA-jg-000005-20120312 group 8 against
#  the header (test/test_clean.py states the figures).
#
#  One cluster's spikes are put in the order in which each next spike changes the
#  cumulative mean waveform least, and the order is cut where spikes unlike the
#  ones already taken start to come in:
#    1. CENTRE   every spike is moved (sub-sample, cubic) so that the point where
#                its falling flank crosses half its trough depth, on the cluster's
#                primary channel, lands where the median waveform's does
#    2. WINDOW   a tapered window around that crossing (short rise, flat over the
#                trough, longer fall)
#    3. ALL CHANNELS, unweighted
#    4. ORDER    seed = the spike nearest the sample-wise median, then always the
#                spike nearest the running mean
#    5. CUT      per channel, the excess kurtosis of the spikes taken so far and
#                its change over a span of 2 % of the spikes; the onset on a
#                channel is the first rank past half the order where that change
#                exceeds onset_factor x its mean size between 20 % and 50 %; the
#                cut is the earliest onset over the channels
#  Spikes past the cut are the ones to split off.  The dominant unit is assumed
#  to be more than half the cluster.  The order is quadratic in the spikes: above
#  max_ordered an evenly spaced subset is ordered and cut, and the others are
#  kept when they lie within the cut's own step of the kept subset's mean.
# ══════════════════════════════════════════════════════════════════════════════
import math

import numpy as np

DEFAULTS = dict(rise_start_ms=-0.154, flat_start_ms=-0.031, flat_end_ms=0.184, fall_end_ms=0.430,
                max_shift_ms=0.184, search_ms=0.280, baseline_ms=0.246, onset_factor=5.0,
                min_spikes=200, max_ordered=40000)


def _lround(x):
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


def _upper_median(a, axis=0):
    """The element std::nth_element leaves at size // 2 (the upper of the two middle ones)."""
    mid = a.shape[axis] // 2
    return np.take(np.partition(a, mid, axis=axis), mid, axis=axis)


def _taper(x, r0, f0, f1, r1):
    w = np.zeros_like(x, dtype=np.float64)
    w[(x >= f0) & (x <= f1)] = 1.0
    q = (x > r0) & (x < f0)
    w[q] = 0.5 - 0.5 * np.cos(np.pi * (x[q] - r0) / (f0 - r0))
    q = (x > f1) & (x < r1)
    w[q] = 0.5 + 0.5 * np.cos(np.pi * (x[q] - f1) / (r1 - f1))
    return w


def first_half_crossing(w, lo, hi, n_base):
    """Per row of w (n, nSamp): the fractional sample where the falling flank
    crosses half the trough depth (trough looked for in [lo, hi], depth from the
    mean of the first n_base samples).  Returns (crossing, ok)."""
    w = np.asarray(w, np.float64)
    n, ns = w.shape
    wb = w - w[:, :n_base].sum(1, keepdims=True) / n_base
    k = lo + np.argmin(wb[:, lo:hi + 1], axis=1)                 # the first of equal minima
    rows = np.arange(n)
    v = wb[rows, k]
    above = wb >= (v / 2.0)[:, None]
    above[np.arange(ns)[None, :] > k[:, None]] = False
    has = above.any(1)
    a = ns - 1 - np.argmax(above[:, ::-1], axis=1)               # the last sample at or above half depth, <= k
    a = np.where(has, a, 0)
    ok = (v < 0.0) & has & (a < k) & (a + 1 < ns)
    wa = wb[rows, a]; wn = wb[rows, np.minimum(a + 1, ns - 1)]
    ok &= wa != wn
    with np.errstate(divide="ignore", invalid="ignore"):
        cr = a + (wa - v / 2.0) / (wa - wn)
    return np.where(ok, cr, 0.0), ok


def _cubic_at(W, t):
    """W (n, nSamp, nCh) float32 at the fractional samples t (n, nWin): Catmull-Rom, the ends held.
    Returns (n, nWin, nCh) float32."""
    n, ns, _ = W.shape
    i = np.clip(np.floor(t).astype(np.int64), 0, ns - 1)
    u = (t - i).astype(np.float32)[:, :, None]
    r = np.arange(n)[:, None]
    p0 = W[r, np.maximum(i - 1, 0)]; p1 = W[r, i]
    p2 = W[r, np.minimum(i + 1, ns - 1)]; p3 = W[r, np.minimum(i + 2, ns - 1)]
    f2, f5, f4, f3, h = np.float32(2), np.float32(5), np.float32(4), np.float32(3), np.float32(0.5)
    out = p1 + h * u * (p2 - p0 + u * (f2 * p0 - f5 * p1 + f4 * p2 - p3 + u * (f3 * (p1 - p2) + p3 - p0)))
    lo = (t <= 0.0)[:, :, None]; hi = (t >= ns - 1)[:, :, None]
    out = np.where(lo, W[:, :1, :], out)
    return np.where(hi, W[:, -1:, :], out).astype(np.float32)


def order_spikes(X, seed):
    """The ordering loop: (order, step) with order[0] == seed and step[k] = |x - m| at which the k-th
    spike was taken.  Ties go to the lower index."""
    X = np.ascontiguousarray(X, np.float32)
    n = X.shape[0]
    x2 = (X.astype(np.float64) ** 2).sum(1)
    xm = np.zeros(n)
    m = X[seed].astype(np.float64)
    order = np.empty(n, np.int64); step = np.zeros(n)
    order[0] = seed
    j = int(seed)
    for k in range(1, n):
        xj = X[j]
        xm += (X.dot(xj).astype(np.float64) - xm) / k
        x2[j] = np.inf                                           # taken
        key = x2 - 2.0 * xm
        best = int(np.argmin(key))
        if k > 1:
            m += (xj - m) / k
        order[k] = best
        step[k] = math.sqrt(max(float(key[best]) + float(m @ m), 0.0))
        j = best
    return order, step


def analyse(wav, sample_rate_hz, **params):
    """wav: (n, nSamp, nCh) waveforms of ONE cluster (the .spk layout).  Returns a dict: ok, reason,
    primary_channel, cut_channel, ordered, cut, cut_step, order (indices into wav, in order) and
    keep (n,) bool -- False for the spikes to split off.  ok False: the analysis did not run and
    everything is kept; ok True with cut == ordered: nothing to split off."""
    p = dict(DEFAULTS); p.update(params)
    W = np.asarray(wav, np.float32)
    n = W.shape[0] if W.ndim == 3 else 0
    R = dict(ok=False, reason="", primary_channel=-1, cut_channel=-1, ordered=0, cut=0, cut_step=0.0,
             order=np.zeros(0, np.int64), keep=np.ones(n, bool))
    if W.ndim != 3 or W.shape[1] < 8 or W.shape[2] < 1:
        R["reason"] = "no waveforms"; return R
    if n < p["min_spikes"]:
        R["reason"] = "too few spikes"; return R
    _, ns, nch = W.shape
    per_ms = float(sample_rate_hz) / 1000.0

    med = _upper_median(W[::max(1, n // 2000)], axis=0)          # (nSamp, nCh)
    prim = int(np.argmax(med.max(0) - med.min(0)))
    R["primary_channel"] = prim
    trough = int(np.argmin(med[:, prim]))
    n_base = min(max(_lround(p["baseline_ms"] * per_ms), 2), ns // 2)
    search = max(2, _lround(p["search_ms"] * per_ms))
    lo, hi = max(1, trough - search), min(ns - 2, trough + search)
    if lo >= hi:
        R["reason"] = "the median waveform has no trough to centre on"; return R
    tgt, okm = first_half_crossing(med[None, :, prim], lo, hi, n_base)
    if not okm[0]:
        R["reason"] = "the median waveform has no trough to centre on"; return R
    target = float(tgt[0])
    max_shift = p["max_shift_ms"] * per_ms

    tw = _taper(np.arange(ns) - target, p["rise_start_ms"] * per_ms, p["flat_start_ms"] * per_ms,
                p["flat_end_ms"] * per_ms, p["fall_end_ms"] * per_ms)
    win_t = np.flatnonzero(tw > 0.0)
    if win_t.size < 4:
        R["reason"] = "the window falls outside the snippet"; return R
    win_w = tw[win_t].astype(np.float32)
    nwin = win_t.size
    d = nwin * nch

    X = np.empty((n, d), np.float32)                             # [s * nCh + c]
    for a in range(0, n, 8192):
        Wc = W[a:a + 8192]
        cr, ok = first_half_crossing(Wc[:, :, prim], lo, hi, n_base)
        shift = np.where(ok, np.clip(target - cr, -max_shift, max_shift), 0.0)
        C = _cubic_at(Wc, win_t[None, :] - shift[:, None])
        X[a:a + 8192] = (C * win_w[None, :, None]).reshape(len(Wc), d)

    n_ord = min(n, max(int(p["max_ordered"]), int(p["min_spikes"])))
    pick = np.arange(n) if n_ord == n else (np.arange(n_ord, dtype=np.float64) * n / n_ord).astype(np.int64)
    Xo = X if n_ord == n else X[pick]

    med_x = _upper_median(Xo[::max(1, n_ord // 4000)], axis=0)
    seed = int(np.argmin(((Xo.astype(np.float64) - med_x) ** 2).sum(1)))
    order, step = order_spikes(Xo, seed)

    span = max(20, n_ord // 50)
    half, fifth = n_ord // 2, n_ord // 5
    kk = np.arange(1, n_ord + 1, dtype=np.float64)[:, None]
    onset = np.full(nch, n_ord, np.int64)
    Xs = Xo[order].reshape(n_ord, nwin, nch)
    for c in range(nch):
        Y = Xs[:, :, c].astype(np.float64)
        m1 = np.cumsum(Y, 0) / kk; m2 = np.cumsum(Y ** 2, 0) / kk
        m3 = np.cumsum(Y ** 3, 0) / kk; m4 = np.cumsum(Y ** 4, 0) / kk
        var = np.maximum(m2 - m1 * m1, 1e-9)
        K = ((m4 - 4.0 * m1 * m3 + 6.0 * m1 * m1 * m2 - 3.0 * m1 ** 4) / (var * var) - 3.0).sum(1) / nwin
        dK = np.zeros(n_ord); dK[span:] = K[span:] - K[:-span]
        base = np.abs(dK[fifth:half]).sum() / max(1, half - fifth)
        f = np.flatnonzero(dK[max(half, span):] > p["onset_factor"] * base)
        if f.size:
            onset[c] = max(1, max(half, span) + int(f[0]) - span // 2)
    cut = int(onset.min())
    R.update(ok=True, ordered=int(n_ord), cut=cut, order=pick[order])
    if cut >= n_ord:
        R["reason"] = "no change point: the spikes are of one kind"; return R
    R["cut_channel"] = int(np.argmin(onset))
    near = step[max(1, cut - span):cut]
    if near.size:
        R["cut_step"] = float(np.partition(near, near.size // 2)[near.size // 2])
    keep = np.ones(n, bool)
    keep[pick[order[cut:]]] = False
    if n_ord != n:
        core = Xo[order[:cut]].astype(np.float64).sum(0) / cut
        rest = np.ones(n, bool); rest[pick] = False
        idx = np.flatnonzero(rest)
        for a in range(0, idx.size, 16384):
            ii = idx[a:a + 16384]
            keep[ii] = ((X[ii].astype(np.float64) - core) ** 2).sum(1) <= R["cut_step"] ** 2
    R["keep"] = keep
    return R


def residual_variance(waves, min_spikes=30):
    """neurosuite-3's waveformquality.h (the waveform view's quality dot): how tightly a cluster's
    spikes sit on their median waveform.  waves (n, nSamp, nCh).

        value = sum over the samples off the window's edges, all channels, of
                (mean over spikes of (w - median)^2  -  noise[c])   /   sum of median^2
        noise[c] = mean of w^2 over the spikes and the EDGE samples (first and last tenth) of channel c

    0 = the spikes differ from their median by the background only.  NaN below min_spikes, for a
    window under 5 samples, or for a flat median."""
    W = np.asarray(waves, np.float64)
    if W.ndim != 3 or W.shape[0] < min_spikes or W.shape[1] < 5:
        return float("nan")
    ns = W.shape[1]
    edge = max(2, ns // 10)
    if ns - 2 * edge < 1:
        return float("nan")
    ed = np.r_[0:edge, ns - edge:ns]
    noise = (W[:, ed, :] ** 2).mean((0, 1))
    med = np.median(W, 0)
    v = ((W - med) ** 2).mean(0)[edge:ns - edge]
    energy = float((med[edge:ns - edge] ** 2).sum())
    return float((v - noise).sum() / energy) if energy > 0.0 else float("nan")
