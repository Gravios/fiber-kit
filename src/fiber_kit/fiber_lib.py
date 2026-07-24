# ════════════════════════════════════════════════════════════════════════════
#  fiber_lib.py  —  validated primitives for CA1 fiber reorganization
#  (neurosuite-3, session 2026-06-03; group 5 = Buzsaki64L shank 5, ch 32-39)
#
#  Every function here was empirically validated on real chunk data this
#  session.  See HANDOFF.md for the validation evidence behind each constant.
# ════════════════════════════════════════════════════════════════════════════
import numpy as np
import os as _os
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

# The masks above and realign's lo/hi window are calibrated for the 32-sample/peak-15 window.
# Their spans are physical (fixed in SAMPLES, since the sampling rate is constant), so for any
# other window they should sit at the SAME sample offsets relative to the detection peak.
# build_masks rebuilds them peak-relative for an arbitrary (nsamp, peak); at (32, 15) it returns
# byte-identical masks/offsets to the constants above, so 32-sample sessions are unchanged.
from collections import namedtuple as _namedtuple

Masks = _namedtuple("Masks", "full narrow core offset realign_lo realign_hi")

_FULL_REL    = (-4, 11)    # MASK_FULL   = arange(peak-4,  peak+11)  -> arange(11, 26) at peak 15
_NARROW_REL  = (-2, 9)     # MASK_NARROW = arange(peak-2,  peak+9)   -> arange(13, 24)
_CORE_REL    = (-2, 2)     # MASK_CORE   = arange(peak-2,  peak+2)   -> arange(13, 17)
_REALIGN_REL = (-9, 11)    # realign search window lo/hi             -> (6, 26)


def build_masks(nsamp, peak=None):
    """Build the feature masks + extraction offset + realign window for a spike window of `nsamp`
    samples whose detection peak sits at index `peak` (the YAML spikeDetection peak / fiber-session
    'peak').  Reproduces the historical 32-sample/peak-15 constants exactly when (nsamp, peak) ==
    (32, 15); otherwise it shifts the same physical spans onto the new peak so non-32 windows get a
    correct mask instead of the mis-placed 32-sample default.

    If `peak` is None it is inferred (15 for nsamp 32, else round(nsamp * 15/32)).  Returns a
    Masks namedtuple: (full, narrow, core, offset, realign_lo, realign_hi).  Every span is clipped
    to the valid [0, nsamp) range, so a window too short to hold the full span degrades gracefully
    rather than indexing out of bounds."""
    if peak is None:
        peak = EXTRACT_OFFSET if int(nsamp) == 32 else int(round(int(nsamp) * EXTRACT_OFFSET / 32.0))
    peak = int(peak); nsamp = int(nsamp)
    if not (0 <= peak < nsamp):
        raise ValueError(f"build_masks: peak {peak} outside the window [0, {nsamp})")

    def span(lohi):
        lo, hi = lohi
        return np.arange(max(0, peak + lo), min(nsamp, peak + hi))

    return Masks(span(_FULL_REL), span(_NARROW_REL), span(_CORE_REL), peak,
                 max(0, peak + _REALIGN_REL[0]), min(nsamp, peak + _REALIGN_REL[1]))

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
# ── feature-alignment mode switch (A/B lever for the centroid hypothesis) ────
# "xcorr"    : centroid-seed + iterated xcorr refine (converges to the trough/template alignment).
# "centroid" : PURE centroid, no refine -> keeps trough-position-vs-asymmetry structure in the
#              feature space.  realign() and every splitter that calls align_xcorr respect this; the
#              committing aligners (klusters_offsets/template_offsets) do NOT use align_xcorr and are
#              unaffected (they must keep the trough on Klusters' canonical sample).
# Initial value is read from the FIBER_ALIGN env var so it reaches forked/spawned pool workers; the
# fiber-session / fiber-refine --feature-align flag sets both the env var and set_feature_align().
_FEATURE_ALIGN = _os.environ.get("FIBER_ALIGN", "xcorr")
if _FEATURE_ALIGN not in ("xcorr", "centroid"):     # ignore a malformed env value
    _FEATURE_ALIGN = "xcorr"


def set_feature_align(mode):
    """Select the feature-building alignment: 'xcorr' (default) or 'centroid' (pure, no refine)."""
    global _FEATURE_ALIGN
    if mode not in ("xcorr", "centroid"):
        raise ValueError("feature align mode must be 'xcorr' or 'centroid'")
    _FEATURE_ALIGN = mode


def get_feature_align():
    return _FEATURE_ALIGN


# Per-spike SUB-SAMPLE refine for realign().  align_xcorr already estimates a 3-pt parabolic
# sub-sample lag on top of its integer xcorr peak; realign() historically discarded it
# (subsample=False), snapping every spike to the integer sample grid.  That quantisation is a
# per-spike ~+-0.5-sample jitter that inflates within-unit residual variance.  Because
# mutual_center applies a single COMMON roll to the whole population, the per-spike sub-sample
# refine survives it -- so enabling it tightens the features at native sample count, no upsampling.
# Default ON (validated: pooled cosine AUC 0.971 -> 0.985 on real g5, best-or-tied in 4/5 windows).
# Turn OFF for the legacy integer-grid behaviour with FIBER_SUBSAMPLE=0 (reaches forked/spawned pool
# workers), set_realign_subsample(False), or realign(..., subsample=False).
_REALIGN_SUBSAMPLE = _os.environ.get("FIBER_SUBSAMPLE", "1").strip().lower() in ("1", "true", "yes", "on")


def set_realign_subsample(on):
    """Enable/disable realign()'s per-spike sub-sample (parabolic) refine.  See _REALIGN_SUBSAMPLE."""
    global _REALIGN_SUBSAMPLE
    _REALIGN_SUBSAMPLE = bool(on)


def realign_subsample():
    return _REALIGN_SUBSAMPLE


def realign(waveforms, lo=6, hi=26, maxlag=4, iters=6, ref="median", subsample=None):
    """Realign each (nSamp,nCh) spike to a robust cluster reference.

    Thin wrapper over align_xcorr (the shared aligner core): FULL multichannel channel-summed
    cross-correlation against the cluster MEDIAN.  Inherits align_xcorr's default init='centroid', so
    each spike is first seeded by its reference-free circular centroid and then refined by the xcorr --
    fast (the seed needs only one or two refine passes) and robust to large initial jitter.  This
    replaces the former dominant-channel single-pass trough lock.

    Honours set_feature_align('centroid') -> pure centroid, no refine.  Because the centroid seed is
    sub-sample, the returned waveforms are sub-sample (Fourier) aligned rather than exact integer rolls
    -- which is what the feature/template builders that call realign want anyway.  For an exact integer
    roll use align_xcorr(..., init='cold', subsample=False).  `lo`/`hi` are retained for signature
    compatibility and unused; `maxlag` bounds the refine search.

    subsample selects whether the xcorr refine keeps its 3-pt parabolic sub-sample lag (True) or
    snaps to the integer grid (False).  None (the default) follows the FIBER_SUBSAMPLE module lever
    (set_realign_subsample); that lever now defaults to True (sub-sample on) -- set FIBER_SUBSAMPLE=0
    or realign(subsample=False) for the legacy integer-grid behaviour.  On real g5 across the session,
    sub-sample alignment lifts the split-half
    same/different cosine AUC ~0.97 -> ~0.985 (pooled) at native sample count -- matching or beating
    2x Fourier upsampling without doubling the feature length."""
    sub = _REALIGN_SUBSAMPLE if subsample is None else bool(subsample)
    return align_xcorr(waveforms, ref=ref, iters=iters, maxlag=maxlag, subsample=sub)


def centroid_shift(waves, peak, weight="energy"):
    """Reference-free per-spike alignment shift via the circular centroid of the energy envelope.

    Treat each channel's per-sample energy e[t] (W**2 for weight='energy', |W| for 'abs') as a mass
    distribution on the circle theta_t = 2*pi*t/T.  The first DFT bin Z = sum_t e[t] exp(i theta_t) is
    the complex resultant: its angle is the circular center-of-mass (sub-sample), its magnitude the
    resultant length (concentration).  Summing Z over channels weights each channel by its own
    resultant length, so sharply-peaked channels dominate and energy-spread channels contribute little.
    The shift is the signed circular distance from that centroid to `peak`.  Closed-form, single pass,
    template-free, sub-sample, and -- unlike a template xcorr -- invariant to how large the initial
    jitter is (no reference to blur).  Note it centres the ENERGY CENTROID, which for asymmetric
    waveforms sits a constant offset from the trough; that offset is uniform across a consistent shape
    so it tightens cleanly, but a committing aligner that must hit the trough sample should calibrate it.

    Returns sh (N,) float32 signed shift in samples; applying a circular roll of -sh lands the centroid
    on `peak`.  Backend-aware: runs on GPU when backend.gpu_enabled() (CuPy), numpy otherwise."""
    W = _bk.asarray(np.asarray(waves, float)); T = W.shape[1]
    pos = _centroid_pos(W, weight)                            # backend array, centroid sample in [0,T)
    sh = peak - pos
    return _bk.asnumpy((sh + T / 2.0) % T - T / 2.0).astype(np.float32)   # signed circular distance


def _centroid_pos(W, weight="energy"):
    """Per-spike circular-centroid sample position in [0,T) on the active backend.  `W` is an xp array
    (n,T,C); see centroid_shift for the math.  Shared by centroid_shift and align_xcorr's centroid init
    so the GPU path avoids a host round-trip."""
    xp = _bk.xp(); T = W.shape[1]
    th = 2.0 * np.pi * xp.arange(T) / T
    e = (W * W) if weight == "energy" else xp.abs(W)
    Z = (e * xp.exp(1j * th)[None, :, None]).sum(1).sum(1)     # (n,) channel-summed complex resultant
    return (xp.angle(Z) % (2.0 * np.pi)) * T / (2.0 * np.pi)


def align_xcorr(waves, ref="median", iters=6, maxlag=6, subsample=True, tol=1e-3,
                return_shifts=False, init="centroid", peak=None):
    """Circularly align each (nSamp,nCh) spike to the cluster mean/median waveform
    by FULL channel-summed cross-correlation, ITERATING until the residual
    variance stops dropping -- so the variance that remains is the shape variance.

    This is the single shared alignment core for fiber-kit: realign() wraps it with
    subsample=False, the session/refine splitters call it directly.  Median reference by default:
    robust to the sub-units we are about to split.  The cross-correlation is the RAW channel-summed
    correlation (a matched filter) -- argmax over |lag|<=maxlag, no per-lag normalisation -- so there
    is no large-lag bias.  Circular (Fourier) shifts are exact and harmless because the spkD/.fil
    waveforms are high-pass filtered (window edges ~zero).  Runs on GPU when backend.gpu_enabled().

    init='centroid' (DEFAULT) seeds the per-spike shift from the reference-free circular centroid
    (_centroid_pos) before the xcorr refinement.  Because that seed is invariant to the initial jitter
    magnitude, the refinement converges in one or two passes (tol early-stop) even on heavily-jittered
    data where a cold mean/median template would be blurred -- so it is both faster and at least as tight
    as a cold start.  When `peak` is None the seed targets the population's CIRCULAR-MEAN centroid, i.e.
    a pure relative de-jitter that leaves the population's absolute position where a cold start would
    converge (so masked features are unchanged, only the convergence path is shorter).  Pass peak to
    target an absolute sample, or iters=0 for the pure centroid alignment with no xcorr refinement.
    init='cold' starts at zero shift and is BIT-IDENTICAL to the pre-centroid implementation.

    Returns aligned waves (+ per-spike shifts if return_shifts)."""
    xp = _bk.xp()
    W = _bk.asarray(np.asarray(waves, float)); n, T, C = W.shape
    if n < 3:
        out = _bk.asnumpy(W)
        return (out, np.zeros(n)) if return_shifts else out
    FW = xp.fft.fft(W, axis=1); f = xp.fft.fftfreq(T)
    if _FEATURE_ALIGN == "centroid":
        # A/B lever: PURE centroid alignment, no xcorr refinement, to a FIXED reference (peak if
        # given else T//2) so different fragments' direction profiles are positioned comparably --
        # this preserves the trough-position-vs-asymmetry structure the refine would erase.
        pos = _centroid_pos(W); tgt = float(peak) if peak is not None else float(T // 2)
        total = ((pos - tgt + T / 2.0) % T - T / 2.0)
        cur = xp.fft.ifft(FW * xp.exp(2j * np.pi * f[None, :, None] * total[:, None, None]), axis=1).real
        out = _bk.asnumpy(cur)
        return (out, _bk.asnumpy(total)) if return_shifts else out
    lag = ((xp.arange(T) + T // 2) % T) - T // 2          # signed circular lags
    inwin = xp.abs(lag) <= maxlag
    if init == "centroid":
        pos = _centroid_pos(W)                                         # centroid sample in [0,T)
        if peak is not None:
            target = float(peak)
        else:
            # circular-mean centroid: relative de-jitter that preserves the population's absolute
            # position (the basin a cold start converges to), so masked features are unchanged
            ang = xp.angle((xp.exp(2j * np.pi * pos / T)).mean())
            target = (float(ang) % (2.0 * np.pi)) * T / (2.0 * np.pi)
        # cur = ifft(FW exp(2pi i f total)) == roll(W, -total); to move pos -> target, total = pos - target
        total = ((pos - target + T / 2.0) % T - T / 2.0)              # shortest signed circular shift
        cur = xp.fft.ifft(FW * xp.exp(2j * np.pi * f[None, :, None] * total[:, None, None]), axis=1).real
    else:
        total = xp.zeros(n); cur = W.copy()
    prev = float("inf")
    for it in range(iters):
        templ = xp.median(cur, 0) if ref == "median" else cur.mean(0)
        rv = float(((cur - templ) ** 2).mean())
        if it > 0 and prev - rv < tol * prev:
            break
        prev = rv
        Ft = xp.fft.fft(templ, axis=0)
        xc = xp.fft.ifft(xp.fft.fft(cur, axis=1) * xp.conj(Ft)[None], axis=1).real.sum(2)  # (n,T)
        xcm = xp.where(inwin[None, :], xc, -np.inf); k = xcm.argmax(1); d = lag[k].astype(float)
        if subsample:
            r = xp.arange(n); km = (k - 1) % T; kp = (k + 1) % T
            y0 = xc[r, km]; y1 = xc[r, k]; y2 = xc[r, kp]; den = y0 - 2 * y1 + y2
            ok = xp.abs(den) > 1e-9                       # parabolic vertex only where the peak is curved
            dsafe = xp.where(ok, den, 1.0)                # keep the divide finite: where() evaluates both arms,
            d += xp.where(ok, 0.5 * (y0 - y2) / dsafe, 0.0).clip(-0.5, 0.5)   # so /den would warn on flat triples
        total += d
        cur = xp.fft.ifft(FW * xp.exp(2j * np.pi * f[None, :, None] * total[:, None, None]), axis=1).real
    cur = _bk.asnumpy(cur); total = _bk.asnumpy(total)
    return (cur, total) if return_shifts else cur

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


# ── chunk whitener, memmap path ──────────────────────────────────────────────
# Moved out of fiber_session: it is an adapter over chunk_whitener_mm just below,
# so its home is here rather than in a CLI stage that four other modules had to
# import to reach it.  `nsamp` is accepted and unused -- kept in the signature
# because every existing call site passes it positionally.
def fil_chunk_whitener(filmm, gch, s0, s1, spike_abs, nsamp, mask):
    # memmap path: reads only sampled baseline windows, never the whole span.
    return chunk_whitener_mm(filmm, gch, s0, s1, spike_abs, mask=mask)
