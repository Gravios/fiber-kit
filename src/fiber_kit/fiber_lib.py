# ════════════════════════════════════════════════════════════════════════════
#  fiber_lib.py  —  validated primitives for CA1 fiber reorganization
#  (neurosuite-3, session 2026-06-03; group 5 = Buzsaki64L shank 5, ch 32-39)
#
#  Every function here was empirically validated on real chunk data this
#  session.  See HANDOFF.md for the validation evidence behind each constant.
# ════════════════════════════════════════════════════════════════════════════
import numpy as np
from sklearn.covariance import LedoitWolf

SR = 32552.0
MASK_FULL  = np.arange(11, 26)   # WIDE trough+rebound+shoulders (tracking/morph); default.
                                 #   empirically best for whiteness assignment across energy bands;
                                 #   at low energy fiber discrimination lives in the broad shape, not
                                 #   the (converged) trough core, so DO NOT narrow at low E.
MASK_NARROW = np.arange(13, 24)  # former default (13-23); ~0.01 worse at low/mid energy
MASK_CORE  = np.arange(13, 17)   # tight core; WORST for assignment in every band — seeds only, with care
EXTRACT_OFFSET = 15              # .spkD window = [res-15, res+17]; detection peak at sample 15

# ── confirmed transform: spkD = Δt(ALLPAIRS(fil)), verified cos 0.997 ───────
def fil_to_spkD_space(fil_trace):
    """fil_trace (T, nCh) raw voltage -> detection/spkD space (T, nCh).
    T1 = n*x - sum_ch(x)  (ALLPAIRS common-mode rejection, n=nCh)
    then temporal first-difference along time."""
    n = fil_trace.shape[1]
    T1 = n * fil_trace - fil_trace.sum(1, keepdims=True)
    T2 = np.empty_like(T1); T2[1:] = T1[1:] - T1[:-1]; T2[0] = 0
    return T2

# ── per-chunk whitener from occupancy-masked .fil baseline (in spkD space) ──
def chunk_whitener(fil_trace, spike_samples_rel, mask=MASK_FULL, n_base=6000, guard=24, seed=0):
    """Noise whitener + mean in the masked spkD space, from off-spike baseline.
    fil_trace: (T,nCh) raw voltage; spike_samples_rel: spike times relative to trace start."""
    T2 = fil_to_spkD_space(fil_trace); T = T2.shape[0]
    forb = np.zeros(T, bool)
    for sp in spike_samples_rel:
        forb[max(0, sp-guard):min(T, sp+guard)] = True
    rng = np.random.default_rng(seed); base=[]; tries=0
    while len(base) < n_base and tries < 50*n_base:
        s = int(rng.integers(0, T-32)); tries += 1
        if not forb[s:s+32].any(): base.append(T2[s:s+32])
    base = np.array(base)
    bm = base[:, mask, :].reshape(len(base), -1); nmean = bm.mean(0)
    C = LedoitWolf().fit(bm - nmean).covariance_
    ev, Vv = np.linalg.eigh(C); ev = np.maximum(ev, 1e-9)
    W = Vv @ np.diag(1/np.sqrt(ev)) @ Vv.T
    return W, nmean, len(base)

# ── per-spike trough realignment (mandatory: ~3.5-sample jitter) ────────────
def realign(waveforms, lo=6, hi=26, maxlag=4):
    """Rigid trough-lag align each (nSamp,nCh) spike to the cluster-mean dom channel."""
    m = waveforms.mean(0); dom = int(np.argmax(m.max(0) - m.min(0))); ref = m[:, dom]
    out = np.empty_like(waveforms)
    for i, w in enumerate(waveforms):
        best = (-1e18, 0)
        for lag in range(-maxlag, maxlag+1):
            c = np.dot(np.roll(w[:, dom], lag)[lo:hi], ref[lo:hi])
            if c > best[0]: best = (c, lag)
        out[i] = np.roll(w, best[1], axis=0)
    return out

# ── feature pipeline: realign -> mask -> whiten -> polar (radius, direction) ─
def features(waveforms, W, nmean, mask=MASK_FULL):
    W_al = realign(waveforms)
    X = (W_al[:, mask, :].reshape(len(W_al), -1) - nmean) @ W
    r = np.linalg.norm(X, axis=1)
    return X, r, X / r[:, None]

def location_cy(waveforms, y_um):
    """Energy-weighted depth centroid (µm) from the mean template PTP."""
    m = waveforms.mean(0); ptp = m.max(0) - m.min(0); ptp = np.maximum(ptp, 0)
    return float((ptp * y_um).sum() / ptp.sum())
