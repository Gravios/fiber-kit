#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════════════════════
#  fiber_geometry.py — fine-geometry fiber signature + strict link veto.
#
#  The cross-chunk linker must NOT regroup units the per-chunk sort separated.
#  A single mean-template vector is too coarse: distinct units routinely reach
#  cosine 0.6-0.85 (worst 0.98 == the same-unit median), so no template
#  threshold is both strict and useful.  The fiber GEOMETRY -- the unit-waveform
#  direction d(r) traced over spike energy r, as a CURVE rather than one point --
#  separates same-unit-over-time from different-unit at AUC ~0.99 and admits a
#  strict cut (curve distance <= ~1.3 -> 95% same-unit recall, ~1.5% false-merge
#  on real g5 data; the residual false merges are genuinely near-identical units).
#
#  Curves are built from UN-WHITENED masked templates (per-chunk whiteners
#  differ; raw footprints are comparable across chunks), then embedded in a
#  canonical PCA(3) direction basis fit on the pooled curve points at link time.
#  Distance is scale-free (mean per-point tip distance / mean curve length) so it
#  does not reward high-energy units.  Intended use: a VETO on top of the
#  overlap-anchor backbone (shared spikes = the positive identity evidence) --
#  never union two per-chunk fibers whose curves disagree, which also blocks the
#  union-find from chaining two geometrically-distinct fibers through one
#  spurious adjacent link.
# ════════════════════════════════════════════════════════════════════════════
import numpy as np
from collections import namedtuple
from scipy.ndimage import gaussian_filter1d

DEFAULT_NQ = 5
DEFAULT_GEO_THR = 1.3        # calibrated on real g5: 95% recall, ~1.5% false-merge
DEFAULT_SMOOTH_SIGMA = 1.0   # Gaussian temporal denoise; see denoise() calibration


def denoise(waveforms, sigma=DEFAULT_SMOOTH_SIGMA):
    """Strip the high-frequency noise floor off RAW footprints before building
    the curve, by Gaussian-smoothing along the sample axis (-2), per channel.
    waveforms: (..., nsamp, nchan) realigned (un-whitened) waveforms.

    The geometry curve is built from raw footprints (cross-chunk comparable,
    unlike whitened features), so it carries the recording noise floor; that
    floor is what spreads same-unit curves and caps the strict link recall.
    Calibrated on real g5 (curated identity, temporal split): sigma~1.0 lifts the
    perfectly-separable same-unit fraction 0.68 -> 0.95 AND pushes the nearest
    different-unit pair 1.23 -> 1.61 (it denoises without collapsing the fine
    timing structure that separates near-duplicate units).  A linear Gaussian
    beats a 5-pt median here because the noise is ~white and the median's
    nonlinear peak/trough clipping distorts the discriminative shape.  Over-
    smoothing is the failure mode: sigma>=2 starts merging fine-structure
    near-duplicates (the different-unit floor falls back below the unfiltered
    value), so keep sigma ~1.  sigma<=0 disables."""
    wf = np.asarray(waveforms, float)
    return wf if sigma <= 0 else gaussian_filter1d(wf, sigma, axis=-2)


def mutual_center(template, ref_sample=16):
    """Circularly shift a single (nsamp, nchan) template so its dominant-channel
    trough sits at ref_sample.  fl.realign aligns spikes only to their OWN cluster
    mean, so a constant whole-cluster time-offset between two clusters survives and
    wrecks any cross-cluster cosine / curve comparison (a 4-sample shift can read
    cosine -0.46 where the centred templates read +0.95).  Centring every cluster's
    template to a common trough sample removes that nuisance shift; the inter-channel
    offsets are shift-invariant and unaffected, so this only fixes the comparisons
    that need it.  This is the cross-cluster counterpart to realign's within-cluster
    alignment, and is required before template cosine is used as an identity gate."""
    t = np.asarray(template, float)
    dom = int(np.argmax(t.max(0) - t.min(0)))
    return np.roll(t, ref_sample - int(np.argmin(t[:, dom])), axis=0)


def mutual_center_spikes(waveforms, ref_sample=16):
    """mutual_center applied to a (nspk, nsamp, nchan) stack: shift every spike by
    the single offset that brings the cluster-mean dominant trough to ref_sample
    (rigid whole-cluster shift; preserves within-cluster structure realign set up)."""
    w = np.asarray(waveforms, float)
    m = w.mean(0); dom = int(np.argmax(m.max(0) - m.min(0)))
    return np.roll(w, ref_sample - int(np.argmin(m[:, dom])), axis=1)


def fiber_curve(masked_templates_by_spike, energy, nq=DEFAULT_NQ):
    """d(r) curve as a stack of UN-WHITENED mean templates over energy quantiles.

    masked_templates_by_spike: (nspike, P) per-spike realigned masked waveforms
        flattened over (samples, channels) -- raw .spk space, not whitened.
    energy: (nspike,) per-spike energy (e.g. L2 norm of the masked waveform).
    Returns (nq, P): the mean raw template in each energy quantile bin (low->high
    energy).  Bins with too few spikes fall back to the fiber-wide mean."""
    Xs = np.asarray(masked_templates_by_spike, float)
    r = np.asarray(energy, float)
    edges = np.quantile(r, np.linspace(0, 1, nq + 1))
    edges[0] = -np.inf; edges[-1] = np.inf
    glob = Xs.mean(0)
    cur = np.empty((nq, Xs.shape[1]))
    for k in range(nq):
        m = (r >= edges[k]) & (r < edges[k + 1])
        cur[k] = Xs[m].mean(0) if m.sum() >= 5 else glob
    return cur


def geometry_basis(curves, k=3):
    """Canonical PCA(k) direction basis over pooled, unit-normalised curve points.
    `curves`: iterable of (nq, P) arrays.  Returns (mean: (P,), P_basis: (P, k))."""
    pts = np.vstack(list(curves))
    pts = pts / (np.linalg.norm(pts, axis=1, keepdims=True) + 1e-12)   # directions
    mu = pts.mean(0)
    _, _, Vt = np.linalg.svd(pts - mu, full_matrices=False)
    return mu, Vt[:k].T


def _embed(curve, mu, P):
    d = curve / (np.linalg.norm(curve, axis=1, keepdims=True) + 1e-12)
    return (d - mu) @ P


def _clen(emb):
    return float(np.mean([np.linalg.norm(emb[i + 1] - emb[i]) for i in range(len(emb) - 1)])) + 1e-9


def curve_distance(curve_i, curve_j, mu, P):
    """Scale-free distance between two fiber curves in canonical PCA(k): mean
    per-quantile tip distance normalised by the mean curve length.  Lower = more
    likely the same unit; ~0.96 same-unit vs ~5.6 different-unit on real g5."""
    ei, ej = _embed(curve_i, mu, P), _embed(curve_j, mu, P)
    return float(np.mean(np.linalg.norm(ei - ej, axis=1)) / (0.5 * (_clen(ei) + _clen(ej))))


def geo_veto(curve_i, curve_j, mu, P, thr=DEFAULT_GEO_THR):
    """True == REFUSE the link (curves too distinct to be the same unit)."""
    return curve_distance(curve_i, curve_j, mu, P) > thr


# ── inter-channel temporal offsets ──────────────────────────────────────────
# The drift-robust half of identity: WHEN the spike reaches each channel, not
# the amplitude footprint.  Two units at different positions differ in their
# per-channel timing even when per-channel shapes match; and unlike amplitude,
# the timing pattern is intrinsic to the unit, so it survives drift across
# chunks.  Complementary to the curve distance (each catches the other's
# near-duplicates), so a STRICT link requires both to agree.
DEFAULT_OFF_THR = 1.8          # samples RMS; STRICT primary gate (drift-robust)
DEFAULT_GEO_THR_BACKBONE = 2.5 # amplitude footprint is drift-FRAGILE -> loose secondary on anchored links

def group_delay_profile(template, sr=32552.0, band=(300.0, 9000.0), amp_frac=0.3):
    """Per-channel GROUP DELAY (samples, relative to the dominant channel) of a template, from the
    cross-spectrum phase slope:  gd_c = -d/domega arg(F[c] * conj(F[ref])).  This is the per-channel
    delay ('warp') of the Omlor-Giese anechoic mixing model x_c(t)=alpha_c*s(t-tau_c) -- a neuron's
    octrode footprint is one delayed source, so gd_c is its spatial-temporal signature.  Uses the
    whole phase spectrum, so it is steadier than a single trough/lag.  Channels below amp_frac of the
    dominant peak-to-peak carry no reliable phase -> NaN.  Best on RAW templates."""
    T = np.asarray(template, float); nt, nc = T.shape
    F = np.fft.rfft(T - T.mean(0), axis=0); fr = np.fft.rfftfreq(nt, 1.0 / sr); w = 2 * np.pi * fr
    p2p = T.max(0) - T.min(0); ref = int(np.argmax(p2p))
    m = (fr >= band[0]) & (fr <= band[1]) & (np.abs(F[:, ref]) > 1e-9)
    gd = np.full(nc, np.nan)
    if m.sum() < 3:
        return gd
    for c in range(nc):
        if p2p[c] < amp_frac * p2p[ref]:
            continue
        X = F[:, c] * np.conj(F[:, ref]); ph = np.unwrap(np.angle(X[m]))
        gd[c] = -np.polyfit(w[m], ph, 1, w=np.sqrt(np.abs(X[m]) + 1e-12))[0] * sr
    return gd - gd[ref]


def warp_correlation(gd_a, gd_b):
    """Cross-channel Pearson correlation of two group-delay profiles.  ~1 when the per-channel delay
    structure matches (same neuron -- a fixed geometric signature, drift-robust); low/incoherent for
    two different co-located cells.  Complements cosine: it catches high-cosine look-alikes (validated
    on g5: same-neuron ~0.93, high-cosine-different ~0.67), so a relaxed cosine + warp gate recovers
    the last few real merges without the false ones."""
    m = ~np.isnan(gd_a) & ~np.isnan(gd_b)
    if m.sum() < 3 or np.std(gd_a[m]) < 1e-6 or np.std(gd_b[m]) < 1e-6:
        return 0.0
    return float(np.corrcoef(gd_a[m], gd_b[m])[0, 1])


CrossSpecMatch = namedtuple("CrossSpecMatch", "coherence delay")


def cross_spectrum_match(temp_a, temp_b, sr=32552.0, band=(300.0, 9000.0), amp_frac=0.3):
    """Complex-embedding match of two mean templates via the per-channel cross-spectrum phasor.

    For each channel c, D_c = sum_xi F_a[c,xi] * conj(F_b[c,xi]) over the band -- one complex number
    whose MODULUS is the matched-filter shape/amplitude agreement and whose ANGLE is the net timing
    offset (the Omlor-Giese A(xi)=alpha*exp(-2pi i tau xi) phasor, band-collapsed).  Each channel's
    phase is referenced to the dominant channel, and the phasors are summed magnitude-weighted into a
    resultant R:
        coherence = |R| / sum_c |D_c|   in [0,1]   (1 == identical per-channel timing structure)
        delay     = arg(D_dom) / (2 pi f_bar) * sr  (mean global delay in samples; >0: a leads b)

    A GLOBAL delay (drift) multiplies every channel's spectrum by the same exp(-2pi i xi tau), so it
    rotates all phasors equally and is removed by the dominant-channel reference -- `coherence` is
    drift-invariant by construction.  Unlike warp_correlation it needs no per-channel slope fit, so it
    stays sharp at low spike count (synthetic high-cosine/different-timing pairs under random drift:
    same-vs-different AUC ~1.0 by 100 spikes, vs ~0.70 for warp_correlation and ~chance for cosine).
    It bundles the cosine (modulus) and warp (angle) signatures in one object.  Best on RAW,
    FULL-window templates -- do not truncate the tails, the phase lives there.  `coherence` is only
    APPROXIMATELY invariant to amplitude-reweighting drift (the magnitude weighting absorbs moderate
    reweighting).  Returns CrossSpecMatch(coherence, delay); coherence is the same-neuron scalar.

    REAL-DATA CAVEAT (g5 180-210 min, split-half SAME vs cross-cluster DIFFERENT): as a same-vs-
    different gate this UNDERPERFORMS plain cosine -- coherence AUC ~0.88 vs cosine ~0.98, because real
    co-located different units share enough per-channel phase structure that coherence stays high
    (DIFFERENT median ~0.94, poorly specific).  The synthetic edge appears only on shape-matched /
    timing-different pairs.  Keep this as a shift-invariant utility, NOT as a replacement for the
    cosine identity gate.
    """
    A = np.asarray(temp_a, float); B = np.asarray(temp_b, float); nt, nc = A.shape
    FA = np.fft.rfft(A - A.mean(0), axis=0); FB = np.fft.rfft(B - B.mean(0), axis=0)
    fr = np.fft.rfftfreq(nt, 1.0 / sr); m = (fr >= band[0]) & (fr <= band[1])
    p2pA = A.max(0) - A.min(0); p2pB = B.max(0) - B.min(0)
    dom = int(np.argmax(p2pA + p2pB))
    if m.sum() < 3:
        return CrossSpecMatch(0.0, float("nan"))
    D = (FA[m] * np.conj(FB[m])).sum(0)                       # (nc,) per-channel cross-spectrum
    keep = (p2pA >= amp_frac * p2pA[dom]) & (p2pB >= amp_frac * p2pB[dom])
    if keep.sum() < 3 or abs(D[dom]) < 1e-12:
        return CrossSpecMatch(0.0, float("nan"))
    phi = np.angle(D) - np.angle(D[dom])                      # reference out the global drift
    w = np.abs(D)
    R = (w[keep] * np.exp(1j * phi[keep])).sum() / w[keep].sum()
    fbar = float((fr[m] * np.abs(FA[m, dom])).sum() / (np.abs(FA[m, dom]).sum() + 1e-12))
    delay = float(np.angle(D[dom]) / (2 * np.pi * fbar) * sr) if fbar > 0 else float("nan")
    return CrossSpecMatch(float(np.abs(R)), delay)


def temporal_offset(ta, tb, mask=None, maxlag=6):
    """Global sub-sample temporal offset (samples) of template `tb` relative to `ta`: the lag that,
    applied to tb (tb -> tb shifted by +offset), best aligns it to ta.  Channel-summed cross-
    correlation (matched filter over the whole footprint) with a 3-pt parabolic sub-sample refine --
    the accurate estimator for a single GLOBAL shift (sharper than the band-collapsed phase of
    cross_spectrum_match, which biases low for multi-sample shifts).  Intended for align-at-merge:
    once cross_spectrum_match.coherence confirms two fragments are the same unit despite an offset,
    this gives the value to shift one onto the other before combining.  mask: optional (T,C) bool to
    restrict to footprint channels."""
    A = np.asarray(ta, float); B = np.asarray(tb, float); T = A.shape[0]
    if mask is not None:
        A = np.where(mask, A, 0.0); B = np.where(mask, B, 0.0)
    A = A - A.mean(0); B = B - B.mean(0)
    xc = np.fft.irfft(np.fft.rfft(A, axis=0) * np.conj(np.fft.rfft(B, axis=0)), n=T, axis=0).real.sum(1)
    xc = np.roll(xc, T // 2)                                   # zero lag at T//2
    lag0 = T // 2
    lo = max(lag0 - maxlag, 1); hi = min(lag0 + maxlag + 1, T - 1)
    k = lo + int(np.argmax(xc[lo:hi]))
    d = float(k - lag0)
    y0, y1, y2 = xc[k - 1], xc[k], xc[k + 1]; den = y0 - 2 * y1 + y2
    if abs(den) > 1e-9:
        d += float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5))
    return d


def interchannel_offsets(template, amp_frac=0.3, method="trough", up=8, maxlag=None):
    """Per-channel sub-sample timing (samples) of each channel relative to the dominant
    channel.  Channels below amp_frac of the dominant peak-to-peak carry no reliable
    timing -> NaN.  template: (nsamp, nchan) realigned (ideally denoised) mean template.

    method="trough" (default, unchanged): parabolic sub-sample minimum of the trough.
        Cheap, but the trough sample is jittery at low spike count and on the stderiv
        waveform (multi-extremum) -- measured split-half noise ~8 samples on a 27-spike
        cluster, which swamps the genuine sub-sample inter-channel timing.
    method="xcorr": upsampled cross-correlation LAG of the FULL channel waveform against
        the dominant channel (a matched filter -- uses the whole shape, not one sample).
        Far more stable: on g5 the same-neuron split-half noise drops to ~0.1 samples on
        a RAW template (use raw, not stderiv -- the stderiv trough is noisy), so a real
        0.26-sample inter-channel difference between two co-located cells becomes a clean
        2-3 sigma discriminator even at ~30 spikes.  NOTE the lag scale differs from the
        trough scale, so off_thr must be re-calibrated (~0.2-0.3, not 1.0) when using it."""
    T = np.asarray(template, float); nt, nc = T.shape
    p2p = T.max(0) - T.min(0); dom = int(np.argmax(p2p))
    if method == "xcorr":
        x = T - T.mean(0); L = nt * int(up)
        X = np.fft.rfft(x, n=L, axis=0)
        cc = np.roll(np.fft.irfft(X * np.conj(X[:, dom:dom + 1]), n=L, axis=0), L // 2, axis=0)
        if maxlag is not None:
            lo = L // 2 - int(maxlag * up); hi = L // 2 + int(maxlag * up) + 1
            k = lo + cc[lo:hi].argmax(0)
        else:
            k = cc.argmax(0)
        off = (k - L // 2) / float(up)
        off[p2p < amp_frac * p2p[dom]] = np.nan
        return off - off[dom]
    if method == "xcorr_fp":
        # CORRECT Fourier upsampling.  method "xcorr" zero-pads in the TIME domain (rfft(x, n=L) on an
        # nt-length signal merely appends zeros -> no interpolation), so its peak sits at the original-
        # sample lag and the /up shrinks every offset by a factor of `up` (true delay returned at 1/up
        # scale).  Here the spectrum is zero-padded in the FREQUENCY domain (a real Fourier resample),
        # so the inverse is interpolated on a 1/up grid and offsets are returned in TRUE samples, plus
        # a 3-pt parabolic refine for sub-1/up resolution.  Opt-in / non-breaking: "xcorr" is left as
        # is.  NOTE off_thr must be recalibrated for this scale (~up x the value tuned against "xcorr").
        x = T - T.mean(0); L = nt * int(up)
        Xf = np.fft.rfft(x, axis=0)
        Xp = np.zeros((L // 2 + 1, nc), complex); Xp[:Xf.shape[0]] = Xf      # pad in FREQUENCY -> interpolate
        xu = np.fft.irfft(Xp, n=L, axis=0) * up
        Xu = np.fft.rfft(xu, axis=0)
        cc = np.roll(np.fft.irfft(Xu * np.conj(Xu[:, dom:dom + 1]), n=L, axis=0), L // 2, axis=0)
        ml = int((maxlag if maxlag is not None else nt // 2) * up)
        lo = max(L // 2 - ml, 1); hi = min(L // 2 + ml + 1, L - 1)
        k = lo + cc[lo:hi].argmax(0)
        r = np.arange(nc); km = (k - 1) % L; kp = (k + 1) % L
        y0 = cc[km, r]; y1 = cc[k, r]; y2 = cc[kp, r]; den = y0 - 2 * y1 + y2
        d = (k - L // 2).astype(float)
        d += np.where(np.abs(den) > 1e-9, 0.5 * (y0 - y2) / den, 0.0).clip(-0.5, 0.5)
        off = d / float(up)
        off[p2p < amp_frac * p2p[dom]] = np.nan
        return off - off[dom]
    off = np.full(nc, np.nan)
    for ch in range(nc):
        if p2p[ch] < amp_frac * p2p[dom]:
            continue
        i = int(np.argmin(T[:, ch]))
        if 0 < i < nt - 1:
            a, b, c = T[i - 1, ch], T[i, ch], T[i + 1, ch]; dn = a - 2 * b + c
            off[ch] = i + (0.5 * (a - c) / dn if abs(dn) > 1e-9 else 0.0)   # parabolic sub-sample
        else:
            off[ch] = i
    return off - off[dom]

def offset_distance(o1, o2):
    """RMS inter-channel-offset difference (samples) over channels reliable in both."""
    m = ~np.isnan(o1) & ~np.isnan(o2)
    return float(np.sqrt(np.nanmean((o1[m] - o2[m]) ** 2))) if m.sum() >= 2 else np.inf

def link_veto(curve_i, off_i, curve_j, off_j, mu, P, geo_thr=DEFAULT_GEO_THR, off_thr=DEFAULT_OFF_THR):
    """True == REFUSE: link only if BOTH amplitude footprint AND inter-channel
    timing agree.  On real g5 this gives 0 false-merge at 93% same-unit recall."""
    return (curve_distance(curve_i, curve_j, mu, P) > geo_thr) or (offset_distance(off_i, off_j) > off_thr)


# ── strict overlap-anchor linker with combined veto (drop-in for link_chunks) ─
def link_chunks_strict(ext_idx, ext_lab, waves, mask, *, min_anchor=20, frac=0.5,
                       geo_thr=DEFAULT_GEO_THR_BACKBONE, off_thr=DEFAULT_OFF_THR,
                       sigma=DEFAULT_SMOOTH_SIGMA, nq=DEFAULT_NQ):
    """Overlap-anchor union-find (shared spikes = positive identity), but every
    union must also pass the combined geometry+timing veto, so one spurious
    adjacent overlap link can no longer CHAIN two geometrically-distinct fibers
    into a mega-cluster (the catastrophic-linkage failure mode).  Self-contained
    drop-in for fiber_session.link_chunks: pass `waves`/`mask` and it computes the
    per-fiber curve + inter-channel offsets internally.  Returns (gid, nglob)."""
    from collections import defaultdict, Counter
    try:
        from . import fiber_lib as fl
    except ImportError:
        import fiber_lib as fl
    cur, off = {}, {}                                          # per (chunk, localid) signatures
    for c in range(len(ext_idx)):
        if len(ext_idx[c]) == 0:
            continue
        for l in {int(x) for x in ext_lab[c] if x >= 0}:
            sel = ext_idx[c][ext_lab[c] == l]
            al = denoise(fl.realign(waves[sel]), sigma)
            wf = al[:, mask, :].reshape(len(sel), -1)
            cur[(c, l)] = fiber_curve(wf, np.linalg.norm(wf, axis=1), nq)
            off[(c, l)] = interchannel_offsets(al.mean(0))
    mu, P = geometry_basis(list(cur.values()))
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[rb] = ra
    for c in range(len(ext_idx)):
        for l in {int(x) for x in ext_lab[c] if x >= 0}: find((c, l))
    for c in range(len(ext_idx) - 1):
        A = {int(g): int(l) for g, l in zip(ext_idx[c], ext_lab[c]) if l >= 0}
        Bd = {int(g): int(l) for g, l in zip(ext_idx[c + 1], ext_lab[c + 1]) if l >= 0}
        shared = set(A) & set(Bd)
        if not shared: continue
        ab = defaultdict(Counter); ba = defaultdict(Counter)
        for s in shared: ab[A[s]][Bd[s]] += 1; ba[Bd[s]][A[s]] += 1
        for f, row in ab.items():
            g, cnt = row.most_common(1)[0]
            if cnt < min_anchor or cnt < frac * sum(row.values()): continue
            f2, cnt2 = ba[g].most_common(1)[0]
            if f2 != f or cnt2 < frac * sum(ba[g].values()): continue
            if link_veto(cur[(c, f)], off[(c, f)], cur[(c + 1, g)], off[(c + 1, g)],
                         mu, P, geo_thr, off_thr):              # geometry/timing disagree -> refuse
                continue
            union((c, f), (c + 1, g))
    roots = {}; gid = {}
    for c in range(len(ext_idx)):
        for l in {int(x) for x in ext_lab[c] if x >= 0}:
            r = find((c, l)); roots.setdefault(r, len(roots)); gid[(c, l)] = roots[r]
    return gid, len(roots)


# ── curl: intra-chunk fiber fingerprint from the inter-channel timing field ──
# Treat per-channel cross-correlation lags as a 1-form on the channel graph and
# take its CURL (the circulation / non-integrable part, via discrete Helmholtz):
# the part of the pairwise timing NOT explained by any single consistent
# wavefront.  It is a reproducible per-fiber fingerprint that distinguishes
# fibers whose median templates are similar but whose spatiotemporal timing
# differs -- exactly the template-near-duplicates an envelope/template metric
# merges (on real g5: same-fiber curl distance ~0.74 vs template-similar pairs
# 3.4-26; intra-chunk match AUC 0.83).  INTRA-CHUNK ONLY: it is computed from the
# median waveform and is NOT drift-invariant, so it discriminates fibers within
# one chunk's frame, not across chunks.  Geometry-free (graph Helmholtz); pair
# with interchannel_offsets for the gradient/slowness half if positions help.
def curl_feature(template, maxlag=5):
    """Curl 1-form of the inter-channel CC-lag field of a (denoised) median
    template (nsamp, nchan).  Returns a vector over the nchan*(nchan-1)/2 channel
    pairs; compare two fibers with curl_distance.  Denoise the template first."""
    T = np.asarray(template, float); nch = T.shape[1]
    pairs = [(i, j) for i in range(nch) for j in range(i + 1, nch)]
    lag = np.zeros(len(pairs))
    for k, (i, j) in enumerate(pairs):
        wi, wj = T[:, i], T[:, j]
        cc = [float(wi @ np.roll(wj, t)) for t in range(-maxlag, maxlag + 1)]
        t0 = int(np.argmax(cc)) - maxlag
        cm, c0, cp = float(wi @ np.roll(wj, t0 - 1)), float(wi @ np.roll(wj, t0)), float(wi @ np.roll(wj, t0 + 1))
        dn = cm - 2 * c0 + cp
        lag[k] = t0 + (0.5 * (cm - cp) / dn if abs(dn) > 1e-9 else 0.0)
    inc = np.zeros((len(pairs), nch))                       # graph incidence
    for k, (i, j) in enumerate(pairs): inc[k, i] = 1; inc[k, j] = -1
    phi, *_ = np.linalg.lstsq(np.vstack([inc, np.eye(1, nch)]), np.append(lag, 0.0), rcond=None)
    return lag - inc @ phi                                  # Helmholtz curl part (gradient removed)


def curl_distance(c1, c2):
    """Euclidean distance between two curl 1-forms (same channel set)."""
    return float(np.linalg.norm(np.asarray(c1, float) - np.asarray(c2, float)))
