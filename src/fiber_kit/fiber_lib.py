# ════════════════════════════════════════════════════════════════════════════
#  fiber_lib.py  —  validated primitives for CA1 fiber reorganization
#  (neurosuite-3, session 2026-06-03; group 5 = Buzsaki64L shank 5, ch 32-39)
#
#  Every function here was empirically validated on real chunk data this
#  session.  See HANDOFF.md for the validation evidence behind each constant.
# ════════════════════════════════════════════════════════════════════════════
import numpy as np
from sklearn.covariance import LedoitWolf
try:
    from . import backend as _bk
except ImportError:
    import backend as _bk

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
def _fit_whitener(base, mask):
    """Fit the masked-space mean + ZCA whitener from collected baseline snippets."""
    if len(base) == 0:
        raise RuntimeError("chunk_whitener: no off-spike baseline snippets found "
                           "(span too short, or spikes cover it). Lower n_base/guard or widen the span.")
    bm = np.asarray(base)[:, mask, :].reshape(len(base), -1).astype(np.float64)
    nmean = bm.mean(0)
    C = LedoitWolf().fit(bm - nmean).covariance_
    ev, Vv = np.linalg.eigh(C); ev = np.maximum(ev, 1e-9)
    W = Vv @ np.diag(1.0 / np.sqrt(ev)) @ Vv.T
    return W, nmean, len(base)


def _collect_baseline(read_window, T, spike_rel, n_base, guard, seed, win=32, pad=1):
    """Draw up to n_base off-spike 32-sample snippets WITHOUT materializing the
    whole span.  `read_window(r0, r1)` returns the raw (rows, nCh) float window;
    only sampled windows are read and stderiv-transformed.  Forbidden zones are
    tested by searchsorted on the sorted spike times (no T-length mask)."""
    sp = np.sort(np.asarray(spike_rel, dtype=np.int64))
    rng = np.random.default_rng(seed)
    base = []; tries = 0; maxtries = 50 * n_base
    hi_start = max(1, T - win)
    while len(base) < n_base and tries < maxtries:
        tries += 1
        start = int(rng.integers(0, hi_start))
        # reject if any spike falls in [start-guard, start+win+guard)
        i = np.searchsorted(sp, start - guard, "left")
        j = np.searchsorted(sp, start + win + guard, "right")
        if j > i:
            continue
        r0 = max(0, start - pad)                 # 1 extra row so the temporal diff is exact
        seg = np.asarray(read_window(r0, start + win), dtype=np.float32)
        if seg.shape[0] < win:
            continue
        base.append(np.asarray(fil_to_spkD_space(seg)[-win:], dtype=np.float32))
    return base


def chunk_whitener(fil_trace, spike_samples_rel, mask=MASK_FULL, n_base=6000, guard=24, seed=0):
    """Noise whitener + mean in the masked spkD space, from off-spike baseline.
    fil_trace: (T,nCh) raw voltage (already restricted to the group's channels);
    spike_samples_rel: spike times relative to trace start.  Memory-frugal: only
    the sampled snippets are stderiv-transformed, never the whole trace."""
    fil_trace = np.asarray(fil_trace)
    T = fil_trace.shape[0]

    def read_window(r0, r1):
        return fil_trace[r0:r1]

    base = _collect_baseline(read_window, T, spike_samples_rel, n_base, guard, seed)
    return _fit_whitener(base, mask)


def chunk_whitener_mm(filmm, gch, s0, s1, spike_abs, mask=MASK_FULL, n_base=6000, guard=24, seed=0):
    """Memmap whitener: identical statistics to chunk_whitener but reads ONLY the
    sampled baseline windows from `filmm` (shape (Ttot, nTotalCh)), so peak memory
    is O(n_base * 32 * nGroupCh) regardless of span length.  Contiguous channel
    groups are sliced as a view; non-contiguous fall back to per-window fancy index."""
    s0 = max(0, int(s0)); s1 = min(filmm.shape[0], int(s1)); T = s1 - s0
    gch = np.asarray(gch, dtype=np.int64)
    contiguous = gch.size > 0 and bool((np.diff(gch) == 1).all())
    lo, hi = int(gch[0]), int(gch[-1]) + 1
    spike_rel = np.asarray(spike_abs, dtype=np.int64) - s0

    def read_window(r0, r1):
        if contiguous:
            return filmm[s0 + r0: s0 + r1, lo:hi]
        return filmm[s0 + r0: s0 + r1, :][:, gch]

    base = _collect_baseline(read_window, T, spike_rel, n_base, guard, seed)
    return _fit_whitener(base, mask)

# ── per-spike trough realignment (mandatory: ~3.5-sample jitter) ────────────
def realign(waveforms, lo=6, hi=26, maxlag=4):
    """Rigid trough-lag align each (nSamp,nCh) spike to the cluster-mean dom channel.

    Vectorized: cross-correlate every spike's dominant channel against the mean
    reference at all lags in one batched pass (memory-frugal — one lag's gather
    held at a time), pick the first-max lag (identical tie-break to the former
    -maxlag..+maxlag loop), then circularly shift each full waveform by its
    chosen lag via a single take_along_axis.  Bit-identical to the old double
    loop on the numpy path (np.roll wrap reproduced as (t-lag) mod T); ~14x
    faster on 20k spikes.  Runs on GPU when backend.gpu_enabled() (CuPy);
    numpy is the default."""
    xp = _bk.xp()
    W = _bk.asarray(waveforms)
    nspk, T, _ = W.shape
    m = W.mean(0); dom = int(xp.argmax(m.max(0) - m.min(0))); refw = m[lo:hi, dom]
    lags = xp.arange(-maxlag, maxlag + 1)
    win = xp.arange(lo, hi)
    src = (win[None, :] - lags[:, None]) % T          # roll(x,lag)[w] = x[(w-lag) mod T]
    dom_sig = W[:, :, dom]                            # (nspk, T)
    corr = xp.empty((nspk, lags.size), dtype=xp.float64)
    for k in range(lags.size):                        # small fixed loop; vectorized over spikes
        corr[:, k] = dom_sig[:, src[k]] @ refw
    chosen = lags[corr.argmax(1)]                     # argmax = first max = old behaviour
    full_src = (xp.arange(T)[None, :] - chosen[:, None]) % T
    return _bk.asnumpy(xp.take_along_axis(W, full_src[:, :, None], axis=1))

# ── feature pipeline: realign -> mask -> whiten -> polar (radius, direction) ─
def features(waveforms, W, nmean, mask=MASK_FULL):
    W_al = realign(waveforms)
    xp = _bk.xp()
    Xd = (_bk.asarray(W_al[:, mask, :].reshape(len(W_al), -1)) - _bk.asarray(nmean)) \
        @ _bk.asarray(W)
    X = _bk.asnumpy(Xd); r = _bk.asnumpy(xp.linalg.norm(Xd, axis=1))
    return X, r, X / r[:, None]

def location_cy(waveforms, y_um):
    """Energy-weighted depth centroid (µm) from the mean template PTP."""
    m = waveforms.mean(0); ptp = m.max(0) - m.min(0); ptp = np.maximum(ptp, 0)
    return float((ptp * y_um).sum() / ptp.sum())
