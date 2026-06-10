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
    pts = np.vstack([c for c in curves])
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

def interchannel_offsets(template, amp_frac=0.3):
    """Per-channel sub-sample trough time relative to the dominant channel.
    template: (nsamp, nchan) realigned (ideally denoised) mean template.  Channels
    below amp_frac of the dominant peak-to-peak carry no reliable timing -> NaN."""
    T = np.asarray(template, float); p2p = T.max(0) - T.min(0); dom = int(np.argmax(p2p))
    off = np.full(T.shape[1], np.nan)
    for ch in range(T.shape[1]):
        if p2p[ch] < amp_frac * p2p[dom]:
            continue
        i = int(np.argmin(T[:, ch]))
        if 0 < i < T.shape[0] - 1:
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
        for l in set(int(x) for x in ext_lab[c] if x >= 0):
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
        for l in set(int(x) for x in ext_lab[c] if x >= 0): find((c, l))
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
        for l in set(int(x) for x in ext_lab[c] if x >= 0):
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
