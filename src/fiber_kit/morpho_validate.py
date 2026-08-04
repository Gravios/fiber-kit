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
