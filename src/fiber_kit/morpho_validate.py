#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════════════════════
#  morpho_validate.py — check the model's claims against a curated sort.
#
#  Everything upstream of this module is a forward model.  A forward model that
#  is never confronted with the recording is an opinion, and the confrontation
#  has to be a measurement someone could lose.  This module is the confrontation:
#  it takes a curated .clu and asks the four questions the model made claims
#  about, each of which can come back against the model.
#
#    recovery   amplitude and shape as a function of the PRECEDING ISI, pooled
#               over a unit and its candidate fragments.  This is the direct
#               test of morpho_envelope.Recovery, and on the reference session
#               it FALSIFIED it -- see the note on that class.
#
#    budget     how far a unit's own template moves across the session, split by
#               time block.  This is the within-unit budget that actually
#               matters, and it turns out to dwarf the firing-state term.
#
#    noise      split-half template distance per cluster.  Without it a
#               fragment's "deviation" cannot be distinguished from having been
#               estimated from 300 spikes, and most of them cannot.
#
#    timing     latency enrichment of each fragment after a main-cluster spike,
#               against the chance level implied by the main rate, plus the
#               refractory cost of the merge.  This is evidence INDEPENDENT of
#               waveform, which is the only kind that can break the tie when
#               shape distances sit inside the noise.
#
#  The module deliberately computes no verdict.  Every quantity here has a
#  failure mode that a number alone hides -- the refractory test has almost no
#  power at a few hundred spikes, latency enrichment is shared by a burst
#  continuation and by a synaptically driven partner, and a small shape distance
#  can mean identity or can mean two cells at the same point.  Printing
#  "MERGE" over that would be a false precision.
# ════════════════════════════════════════════════════════════════════════════
import numpy as np

try:
    from . import neuro_io as nio
except ImportError:
    import neuro_io as nio

# The .res -> LFP sample mapping lives in neuro_io and is re-exported, not
# reimplemented: a second copy would be a second place the convention is
# decided, and being one sample out is invisible in a rate ratio.
lfp_index = nio.lfp_index


def cos_dist(a, b):
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= 0 or nb <= 0:
        return np.nan
    return 1.0 - float(a @ b / (na * nb))


def p2p(t):
    t = np.asarray(t, float)
    return float((t.max(0) - t.min(0)).max())


class Sort:
    """A curated group: spike times, waveforms and cluster ids, loaded together.

    Loaded through neuro_io rather than by direct path composition so that the
    variant/tag grammar has exactly one implementation.  The .spk stays a memmap:
    a group of this session is 284 MB and every statistic below touches only a
    cluster's rows.
    """

    def __init__(self, base, elec, nsamp, nchan, variant="", tag="", prefer=None):
        self.base, self.elec = base, elec
        self.nsamp, self.nchan = int(nsamp), int(nchan)
        # .res and .spk are Shared-class artifacts: ONE physical copy whatever
        # method produced it, so they are resolved with resolve_any rather than
        # by walking a preference list.  In this session they do not even carry
        # the same token -- the .res is written `stderiv` while the .spk is
        # `stderiv.C5.D34` -- and a preference walk would simply miss one of them.
        rr = nio.resolve_any(base, "res", elec, preferred=variant)
        rs = nio.resolve_any(base, "spk", elec, preferred=variant)
        for r, what in ((rr, "res"), (rs, "spk")):
            if not r.found:
                raise FileNotFoundError(f"no .{what} for group {elec}; expected {r.path}")
        self.res_path, self.spk_path = rr.path, rs.path
        self.res_variant, self.spk_variant = rr.variant, rs.variant
        self.res = np.asarray(nio.read_res_file(rr.path), np.int64)
        self.spk = nio.open_spk_file(rs.path, self.nsamp, self.nchan)
        # The .clu is MethodSpecific: strict, no fallback, pinned by variant+tag.
        if tag:
            out = nio.read_clu_at(base, elec, variant=variant, tag=tag,
                                  n_spikes=len(self.res))
        else:
            out = nio.read_clu(base, elec, n_spikes=len(self.res),
                               prefer=(prefer if prefer is not None
                                       else ([variant] if variant else None)))
        self.nclu, self.clu = (out if isinstance(out, tuple) else (None, out))
        self.clu = np.asarray(self.clu, np.int64)
        # Report which physical copies were opened.  The dedup unit is the STAGE,
        # so a res/clu/spk trio that came from different runs is the failure to
        # catch here, before any statistic is computed on a half-aligned stage.
        self.provenance = dict(res=(rr.path, rr.variant), spk=(rs.path, rs.variant),
                               clu=(nio.session_path(base, "clu", elec, variant, tag),
                                    variant))
        if not (len(self.res) == len(self.clu) == len(self.spk)):
            raise ValueError(f"length mismatch: res={len(self.res)} clu={len(self.clu)} "
                             f"spk={len(self.spk)} — the .clu/.res/.spk are not one stage")

    def sizes(self):
        u, c = np.unique(self.clu, return_counts=True)
        return dict(zip(u.tolist(), c.tolist()))

    def idx(self, k):
        return np.flatnonzero(self.clu == k)

    def template(self, k, idx=None):
        i = self.idx(k) if idx is None else idx
        return np.asarray(self.spk[i], np.float64).mean(0)

    def minutes(self, k, sr):
        return self.res[self.idx(k)].astype(np.float64) / float(sr) / 60.0


# ── the four measurements ───────────────────────────────────────────────────
def split_half_noise(sort, k):
    """Template distance between a cluster's first and second half, in time.

    First half against second half, not a random split, because a random split
    averages drift into both halves and reports a noise floor that is too
    optimistic — which would then make every fragment look significant.  This
    number therefore contains drift as well as estimation error, and is an upper
    bound on estimation error alone.
    """
    i = sort.idx(k)
    if len(i) < 8:
        return np.nan
    h = len(i) // 2
    return cos_dist(sort.template(k, i[:h]), sort.template(k, i[h:]))


def time_budget(sort, k, nblock=6):
    """Template distances between time blocks of ONE accepted cluster.

    The point of this number: it is the within-unit variation a curator has
    already accepted as a single cell, measured on their own sort.  Any
    candidate fragment closer than this is not asking for a tolerance the sort
    does not already extend to itself.
    """
    i = sort.idx(k)
    if len(i) < nblock * 8:
        return np.full((nblock, nblock), np.nan)
    t = sort.res[i].astype(np.float64)
    e = np.percentile(t, np.linspace(0, 100, nblock + 1))
    T = []
    for b in range(nblock):
        m = (t >= e[b]) & (t <= e[b + 1])
        T.append(sort.template(k, i[m]) if m.sum() >= 4 else None)
    D = np.full((nblock, nblock), np.nan)
    for a in range(nblock):
        for b in range(nblock):
            if T[a] is not None and T[b] is not None:
                D[a, b] = cos_dist(T[a], T[b])
    return D


def latency_enrichment(sort, main, frag, sr, window_ms=10.0):
    """P(fragment spike within `window` after a main spike), and chance level.

    Chance is the Poisson expectation from the main cluster's own mean rate,
    1 - exp(-rate * window).  A fragment that is the later spikes of the main
    cell's bursts is enriched many-fold; two independent cells are not.  What
    this CANNOT separate is a burst continuation from a synaptically driven
    partner cell, which shows the same enrichment for a different reason.
    """
    tm = np.sort(sort.res[sort.idx(main)].astype(np.float64)) / float(sr)
    tk = np.sort(sort.res[sort.idx(frag)].astype(np.float64)) / float(sr)
    if len(tm) < 2 or len(tk) < 1:
        return np.nan, np.nan
    j = np.searchsorted(tm, tk) - 1
    ok = j >= 0
    if not ok.any():
        return np.nan, np.nan
    lat = tk[ok] - tm[j[ok]]
    obs = float((lat < window_ms * 1e-3).mean())
    rate = len(tm) / max(tm[-1] - tm[0], 1e-9)
    return obs, float(1.0 - np.exp(-rate * window_ms * 1e-3))


def refractory(sort, ks, sr, ref_ms=2.0):
    """Fraction of ISIs below `ref_ms` for the union of the given clusters.

    Reported against the main cluster alone so the MERGE COST is visible rather
    than the absolute rate.  Note the power: at a few hundred fragment spikes
    against tens of thousands of main spikes, a genuinely independent cell adds
    a violation fraction far below what this can resolve, so a clean result here
    is weak evidence and must not be read as a clean bill.
    """
    m = np.isin(sort.clu, np.atleast_1d(ks))
    t = np.sort(sort.res[m].astype(np.float64)) / float(sr)
    if len(t) < 2:
        return np.nan
    return float((np.diff(t) < ref_ms * 1e-3).mean())


def recovery_curve(sort, ks, sr, bins=None, isolated_ms=200.0):
    """Amplitude and shape versus PRECEDING ISI, pooled over clusters.

    Pooling the main cluster with its candidate fragments is the whole point: if
    the fragments are the cell's short-ISI spikes, splitting them out is exactly
    what hides the dependence this measures.  Amplitude is normalised to the
    median of spikes preceded by more than `isolated_ms`, and shape is measured
    against the template of those same isolated spikes, so both columns are
    relative to the cell's own rested state.
    """
    bins = bins or [(2, 4), (4, 6), (6, 8), (8, 12), (12, 20), (20, 40),
                    (40, 100), (100, 200), (200, np.inf)]
    sel = np.flatnonzero(np.isin(sort.clu, np.atleast_1d(ks)))
    t = sort.res[sel].astype(np.float64) / float(sr)
    o = np.argsort(t)
    sel, t = sel[o], t[o]
    w = np.asarray(sort.spk[sel], np.float32)
    amp = (w.max(1) - w.min(1)).max(1)
    isi = np.full(len(t), np.inf)
    isi[1:] = np.diff(t) * 1e3
    iso = isi > isolated_ms
    if iso.sum() < 20:
        raise ValueError("too few isolated spikes to normalise against")
    base = float(np.median(amp[iso]))
    T0 = w[iso].astype(np.float64).mean(0)
    rows = []
    for lo, hi in bins:
        m = (isi >= lo) & (isi < hi)
        if m.sum() < 20:
            continue
        rows.append((lo, hi, int(m.sum()), float(np.median(amp[m])) / base,
                     cos_dist(w[m].astype(np.float64).mean(0), T0)))
    return rows, base, int(iso.sum())


# ── dispersion ──────────────────────────────────────────────────────────────
def feature_radius(features, ids, key):
    """RMS distance of a cluster's spikes from its own centroid, in feature space.

    A scalar, deliberately.  The covariance-based tests above need n >> d and
    are useless on the small clusters where the question is sharpest -- which was
    never a sample-size problem to work around but a sign that a location
    statistic was the wrong tool.  Dispersion is what distinguishes a fragment
    from a cell, and dispersion has one useful number in it.  This is estimable
    from ~20 spikes because it averages n*d squared residuals, not n of them.
    """
    f = np.asarray(features, float)
    i = np.flatnonzero(np.asarray(ids) == key)
    if len(i) < 3:
        return np.nan, len(i)
    X = f[i]
    return float(np.sqrt(((X - X.mean(0)) ** 2).sum(1).mean())), len(i)


def dispersion_table(features, ids, min_spikes=200):
    """Per-cluster feature radius, plus the template energy it should be
    independent of.

    Returns (keys, radius, n, energy).  Energy is carried so a caller can CHECK
    the constancy rather than assume it: if radius tracked energy the whole
    approach would be wrong, and that is a one-line regression to run.
    """
    f = np.asarray(features, float)
    ids = np.asarray(ids)
    u, c = np.unique(ids, return_counts=True)
    keep = u[c >= min_spikes]
    r, n, e = [], [], []
    for k in keep:
        i = np.flatnonzero(ids == k)
        X = f[i]
        r.append(float(np.sqrt(((X - X.mean(0)) ** 2).sum(1).mean())))
        n.append(len(i)); e.append(float(np.linalg.norm(X.mean(0))))
    return keep, np.asarray(r), np.asarray(n), np.asarray(e)


def constancy(radius, energy):
    """Log-log slope and R^2 of radius against template energy.

    Slope ~0 with R^2 ~0 is the claim: one cell's feature-space span is a
    CONSTANT, not a fraction of its amplitude.  Measured on g5 this is slope
    +0.057, R^2 0.020 -- the feature-space form of the constant absolute scatter
    radius already established on raw amplitudes.
    """
    r, e = np.asarray(radius, float), np.asarray(energy, float)
    m = np.isfinite(r) & np.isfinite(e) & (r > 0) & (e > 0)
    if m.sum() < 4:
        return np.nan, np.nan
    x, y = np.log10(e[m]), np.log10(r[m])
    sl = float(np.polyfit(x, y, 1)[0])
    return sl, float(np.corrcoef(x, y)[0, 1] ** 2)


def isi_share(features, times_s, bins_ms=(4, 8, 16, 32, 64, 128, 256)):
    """Fraction of a cell's feature variance that ISI bin membership explains.

    Run on a cell's spikes POOLED across its fragments -- splitting them first is
    what hides the dependence.  On g5 this is 0.21% of variance, a radius of 41
    inside a cell radius of 897: the ISI-dependent changes really do live inside
    the constant span rather than adding to it.
    """
    X = np.asarray(features, float)
    t = np.asarray(times_s, float)
    o = np.argsort(t); X, t = X[o], t[o]
    isi = np.full(len(t), np.inf)
    isi[1:] = np.diff(t) * 1e3
    g = np.digitize(isi, list(bins_ms))
    mu = X.mean(0)
    tot = float(((X - mu) ** 2).sum(1).mean())
    btw = sum((g == k).sum() * float(((X[g == k].mean(0) - mu) ** 2).sum())
              for k in np.unique(g)) / len(X)
    return (btw / max(tot, 1e-30)), float(np.sqrt(tot)), float(np.sqrt(btw))


def dispersion_verdict(radius, expected, lo=0.80, hi=1.25):
    """Classify a cluster by how its span compares with one cell's.

    UNDER-dispersed is the over-split signature: a fragment is a compact
    sub-region of a cell, so it is TIGHTER than a cell, not merely nearer.
    OVER-dispersed is contamination or a merge.  The thresholds are conventions
    and are exposed; what is measured is that a curated cell sits at 1.0 and its
    14 fragments all sat between 0.80 and 0.96.
    """
    if not np.isfinite(radius) or not np.isfinite(expected) or expected <= 0:
        return "unknown", np.nan
    q = radius / expected
    return ("under (fragment?)" if q < lo else
            "over (contaminated?)" if q > hi else "one cell"), q


# ── fast / slow decomposition ───────────────────────────────────────────────
def fast_slow(features, times_s, max_gap_s=1.0, nblock=12):
    """Split a cluster's feature variance into fast, drift and slow-residual.

    The estimator uses only TEMPORAL ADJACENCY, and that is the point.  A pair of
    spikes from the same cell separated by under a second shares the electrode
    position, the dendritic state and the behavioural context, so whatever
    differs between them is spike-independent:

        V_fast = E[ |F_i - F_j|^2 ] / 2   over adjacent pairs

    Nothing about the waveform, the baseline or a noise model enters.  That
    matters because the obvious alternative -- estimating noise from the
    pre-spike samples of a .spk window -- is unsafe on detected spikes: the
    window's leading samples already contain the spike's rising phase, and in a
    dense band they routinely contain other units' spikes, so it measures
    multi-unit activity and calls it a noise floor.

    V_fast is NOT "recording noise".  It is electrode noise plus spike
    superposition plus cluster contamination, three things a merge gate should
    treat differently and which this decomposition does not separate.  Only the
    claim that it is spike-independent is supported.

    Returns dict with v_total, v_fast, v_drift (between nblock time blocks),
    v_slow (total - fast), and the corresponding radii normalised by the
    template norm.
    """
    X = np.asarray(features, float)
    t = np.asarray(times_s, float)
    o = np.argsort(t); X, t = X[o], t[o]
    mu = X.mean(0)
    tn = float(np.linalg.norm(mu))
    V = float(((X - mu) ** 2).sum(1).mean())
    d = np.diff(t)
    m = d < max_gap_s
    if m.sum() < 40:
        return dict(v_total=V, v_fast=np.nan, v_drift=np.nan, v_slow=np.nan,
                    n_pairs=int(m.sum()), template_norm=tn)
    vf = float(((X[1:][m] - X[:-1][m]) ** 2).sum(1).mean() / 2.0)
    e = np.percentile(t, np.linspace(0, 100, nblock + 1))
    vd = 0.0
    for j in range(nblock):
        b = (t >= e[j]) & (t <= e[j + 1])
        if b.sum():
            vd += b.sum() * float(((X[b].mean(0) - mu) ** 2).sum())
    vd /= len(X)
    vs = max(V - vf, 0.0)
    return dict(v_total=V, v_fast=vf, v_drift=vd, v_slow=vs, n_pairs=int(m.sum()),
                template_norm=tn,
                q_fast=np.sqrt(vf) / max(tn, 1e-30),
                q_drift=np.sqrt(vd) / max(tn, 1e-30),
                q_slow=np.sqrt(vs) / max(tn, 1e-30),
                q_slow_nodrift=np.sqrt(max(vs - vd, 0.0)) / max(tn, 1e-30))


def tail_index(features):
    """99.9th percentile of the squared residual norm, in units of its mean.

    Gaussian residuals in d dimensions give chi2_d, whose 99.9th percentile is
    about 2.0 for d = 32.  A larger value means a minority of spikes sit far out,
    which is the signature of superposition or contamination rather than of
    additive noise, and it is a reason not to call V_fast a noise floor.
    """
    X = np.asarray(features, float)
    n2 = ((X - X.mean(0)) ** 2).sum(1)
    return float(np.percentile(n2, 99.9) / max(n2.mean(), 1e-30))


# ── state axes ──────────────────────────────────────────────────────────────
def ccg_asymmetry(values, times_s, win=0.05, lag_lo=0.0, lag_hi=None, split=None):
    """Asymmetry of the cross-correlogram between the two halves of a split.

    Split a cluster along a feature direction and cross-correlate the halves.
    Noise cannot produce an asymmetry: which half a spike lands in is
    independent of when it fired.  A STATE variable can, because the state
    evolves in time, so one half systematically precedes the other.  This is
    therefore a POSITIVE detector of within-cell structure, where every other
    measure in this module works by subtraction.

    Returns (index, n_pairs) with index = (n_forward - n_backward) / n_total
    over the lag band [lag_lo, lag_hi].
    """
    v = np.asarray(values, float)
    t = np.asarray(times_s, float)
    o = np.argsort(t); v, t = v[o], t[o]
    s = v - (np.median(v) if split is None else split)
    a = t[s > 0]; b = t[s <= 0]
    if len(a) < 200 or len(b) < 200:
        return np.nan, 0
    hi_l = lag_hi if lag_hi is not None else win
    lo = np.searchsorted(b, a - win); hi = np.searchsorted(b, a + win)
    parts = [b[lo[k]:hi[k]] - a[k] for k in range(len(a)) if hi[k] > lo[k]]
    if not parts:
        return np.nan, 0
    L = np.concatenate(parts)
    L = L[np.abs(L) > 1e-9]
    p = int(((L > lag_lo) & (L <= hi_l)).sum())
    n = int(((L < -lag_lo) & (L >= -hi_l)).sum())
    return ((p - n) / max(p + n, 1)), p + n


def local_center(features, times_s, win_s=60.0):
    """Subtract a running mean over +-win_s, so drift cannot masquerade as state.

    Without this a slowly drifting feature separates early spikes from late
    ones, and the split halves then differ in TIME as well as in feature -- which
    produces a CCG asymmetry with no physiology in it at all.
    """
    X = np.asarray(features, float)
    t = np.asarray(times_s, float)
    o = np.argsort(t); X, t = X[o], t[o]
    lo = np.searchsorted(t, t - win_s); hi = np.searchsorted(t, t + win_s)
    C = np.cumsum(np.vstack([np.zeros(X.shape[1]), X]), 0)
    return X - (C[hi] - C[lo]) / np.maximum((hi - lo)[:, None], 1)


def shared_state_variance(values, times_s, near=(0.010, 0.050), far=(0.300, 1.0)):
    """Variance along an axis that spikes close in time SHARE.

    A state variable changes slowly enough that two spikes tens of milliseconds
    apart hold similar values, so their squared difference is smaller than for
    spikes a second apart.  A noise axis shows no such gradient.  The difference
    between the two bands is the state variance carried by this axis, and unlike
    a total-minus-noise subtraction it never goes negative for a reason that
    cannot be checked.
    """
    v = np.asarray(values, float)
    t = np.asarray(times_s, float)
    o = np.argsort(t); v, t = v[o], t[o]
    d = np.diff(t)
    dv = (v[1:] - v[:-1]) ** 2 / 2.0
    mn = (d >= near[0]) & (d < near[1])
    mf = (d >= far[0]) & (d < far[1])
    if mn.sum() < 200 or mf.sum() < 200:
        return np.nan, int(mn.sum()), int(mf.sum())
    return float(dv[mf].mean() - dv[mn].mean()), int(mn.sum()), int(mf.sum())


def state_axes(features, times_s, ncomp=8, win_s=60.0, lag=(0.010, 0.050), rng=None):
    """Rank the residual's principal axes by how much STATE they carry.

    Two independent signatures per axis, and an axis is only interesting if it
    shows both: a CCG asymmetry beyond its own shuffled control, and a positive
    shared-state variance.  Either alone is weak -- asymmetry can come from
    residual drift the local centring missed, and a short-gap variance deficit
    can come from refractory-correlated amplitude effects.
    """
    rng = rng or np.random.default_rng(0)
    X = local_center(features, times_s, win_s)
    t = np.sort(np.asarray(times_s, float))
    R = X - X.mean(0)
    U, S, Vt = np.linalg.svd(R, full_matrices=False)
    tot = float((R ** 2).sum(1).mean())
    out = []
    for k in range(min(ncomp, len(S))):
        p = R @ Vt[k]
        a, npair = ccg_asymmetry(p, t, win=lag[1], lag_lo=lag[0], lag_hi=lag[1])
        sh = float(np.nanmean([ccg_asymmetry(rng.permutation(p), t, win=lag[1],
                                             lag_lo=lag[0], lag_hi=lag[1])[0]
                               for _ in range(3)]))
        sv, _, _ = shared_state_variance(p, t, near=lag)
        out.append(dict(pc=k, var_frac=float(S[k] ** 2 / (S ** 2).sum()),
                        asym=a, asym_shuffled=sh, n_pairs=npair,
                        state_var=sv, state_frac=(sv / tot if np.isfinite(sv) else np.nan),
                        direction=Vt[k]))
    return out


# ── LFP phase ───────────────────────────────────────────────────────────────
def bipolar(lfp, a=0, b=1):
    """Difference of two LFP channels: the local voltage gradient.

    Subtracting two nearby sites cancels the volume-conducted far field, which
    is common to both, and keeps the local gradient, which is not.  Measured on
    g5's shank-7 pair (140 um apart): the bipolar theta SD is 1.88x the
    common-mode SD, so the derivation is doing real work rather than just
    halving the signal.

    It does NOT by itself identify a layer.  Two sites 140 um apart constrain
    the gradient between them; naming the generator needs the fissure located,
    which that span cannot do.
    """
    x = np.asarray(lfp)
    return x[:, a].astype(np.float64) - x[:, b].astype(np.float64)


def band_phase(x, sr, lo=5.0, hi=11.0, order=3, chunk=1_000_000, overlap=20_000):
    """Instantaneous phase and amplitude in a band, computed in chunks.

    Chunked because a Hilbert transform of a full session at 1250 Hz is a
    26-million-point FFT and will exhaust memory -- it did, on the first run.
    Chunks overlap by `overlap` samples and only their interiors are kept, so
    the filter and Hilbert edge transients never reach the output; at 1250 Hz
    the default overlap is 16 s, far longer than a 5 Hz filter's ringing.
    """
    from scipy import signal as _sig
    x = np.asarray(x, np.float64)
    n = len(x)
    b, a = _sig.butter(order, [lo / (sr / 2), hi / (sr / 2)], btype="band")
    ph = np.empty(n, np.float32); am = np.empty(n, np.float32)
    for s in range(0, n, chunk):
        p = max(s - overlap, 0); q = min(s + chunk + overlap, n)
        seg = x[p:q] - x[p:q].mean()
        h = _sig.hilbert(_sig.filtfilt(b, a, seg))
        i0, i1 = s - p, min(s + chunk, n) - p
        ph[s:s + (i1 - i0)] = np.angle(h[i0:i1])
        am[s:s + (i1 - i0)] = np.abs(h[i0:i1])
    return ph, am


def circ_lin_corr(y, phase):
    """Circular-linear correlation of a linear variable with a phase.

    Note what it does NOT do: a value of 0.03 at n = 24,000 is many standard
    errors from zero and still explains under 0.1% of the variance.  Report the
    shuffled control alongside it, which is why phase_dependence does.
    """
    y = np.asarray(y, float); p = np.asarray(phase, float)
    c = np.corrcoef(y, np.cos(p))[0, 1]
    s = np.corrcoef(y, np.sin(p))[0, 1]
    rcs = np.corrcoef(np.cos(p), np.sin(p))[0, 1]
    return float(np.sqrt(max(c * c + s * s - 2 * c * s * rcs, 0.0) / max(1 - rcs * rcs, 1e-12)))


def phase_modulation(phase, nbin=18):
    """Depth and max/min of a count histogram over phase — the positive control.

    If the spikes themselves are not phase modulated, the phase estimate is
    wrong or the band is empty, and nothing downstream of it means anything.
    """
    h, _ = np.histogram(np.asarray(phase, float), bins=nbin, range=(-np.pi, np.pi))
    m = h.mean()
    return dict(depth=float((h.max() - h.min()) / max(m, 1e-12)),
                ratio=float(h.max() / max(h.min(), 1)), counts=h)


def phase_dependence(features, times_s, phase, amp=None, amp_pct=60.0, ncomp=6,
                     win_s=60.0, rng=None, nshuffle=10):
    """Do the residual's principal axes track LFP phase?

    Restricted to high-amplitude epochs by default: phase is meaningless where
    the rhythm is absent, and including those samples dilutes a real effect with
    noise rather than guarding against a false one.

    Each axis is reported with BOTH its phase correlation and its shared-state
    variance, because the interesting question is whether they are the same
    axes.  On g5 cluster 2103 they are not -- the axis with the strongest phase
    correlation carries the least state.
    """
    rng = rng or np.random.default_rng(0)
    X = local_center(features, times_s, win_s)
    t = np.sort(np.asarray(times_s, float))
    p = np.asarray(phase, float)
    keep = np.ones(len(p), bool) if amp is None else \
        (np.asarray(amp, float) > np.percentile(np.asarray(amp, float), amp_pct))
    R = X - X.mean(0)
    U, S, Vt = np.linalg.svd(R, full_matrices=False)
    tot = float((R ** 2).sum(1).mean())
    out = []
    for k in range(min(ncomp, len(S))):
        y = R @ Vt[k]
        sv, _, _ = shared_state_variance(y, t)
        r = circ_lin_corr(y[keep], p[keep])
        sh = float(np.mean([circ_lin_corr(y[keep], rng.permutation(p[keep]))
                            for _ in range(nshuffle)]))
        out.append(dict(pc=k, var_frac=float(S[k] ** 2 / (S ** 2).sum()),
                        state_frac=(sv / tot if np.isfinite(sv) else np.nan),
                        r_phase=r, r_shuffled=sh, direction=Vt[k]))
    return out
