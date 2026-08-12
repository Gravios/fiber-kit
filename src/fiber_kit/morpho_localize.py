#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════════════════════
#  morpho_localize.py — where is each cluster, and does it contain one cell?
#
#  A forward model turns a cluster's per-channel amplitude profile into a
#  POSITION.  That is a merge and link criterion built on none of the evidence
#  the existing ones use: not waveform cosine, not refractory statistics, not
#  feature-space distance.  Two clusters that localise to the same point are
#  the same cell; two that do not, are not.
#
#  Measured on g5 group 5, with a fitted CA1 basket cell:
#
#    resolution        split-half floor 2.5 um at ~4000 spikes; the RMSE
#                      landscape triples over 5 um of depth
#    co-localisation   clusters 262 and 263 agree to a median 2.5 um across
#                      eight 24-minute blocks -- exactly the floor
#    drift             one cluster's atoms trace 152.5 -> 175 um over 5 hours
#    two cells in one  within a single chunk, two atoms of cluster 263 sit
#                      26.6 um and 21.4 um apart, against floors of 5.0 and 2.5
#
#  WHY IT IS AFFORDABLE.  The extracellular transfer matrix is position-
#  dependent and time-independent, so one simulation per morphology supports the
#  whole position grid.  Building a table is seconds; fitting a cluster to it is
#  a table search.  Nothing here re-simulates per cluster.
#
#  WHAT IT CANNOT DO, stated because it bounds every use below.  It is blind to
#  genuinely co-located cells -- two somata at the same point produce the same
#  profile whatever their spikes look like -- and that is exactly the population
#  left over once the spatial criteria are exhausted.  It also inherits the
#  model's systematic error: a good fit here is RMSE ~0.011 against a split-half
#  floor of ~0.002, so the model is ~6x from describing the data perfectly, and
#  a reported position is only as good as the morphology being roughly right.
# ════════════════════════════════════════════════════════════════════════════
import numpy as np


def profile(waves):
    """Normalised per-channel peak-to-peak of a mean waveform.

    Normalised because the fit must not depend on absolute amplitude: the same
    cell at the same place recorded through a different gain, or with a
    different spike count, must land in the same position.  Shape carries the
    geometry; scale does not.
    """
    w = np.asarray(waves, np.float64)
    if w.ndim == 3:
        w = w.mean(0)
    p = np.ptp(w, axis=0)
    m = p.max()
    return p / m if m > 0 else p


class PositionTable:
    """Precomputed (morphology, rotation, depth, lateral) -> profile lookup.

    Built once per probe geometry and reused for every cluster in the session.
    Entries carry their generating parameters so a fit is reportable, not just
    a number.
    """

    __slots__ = ("profiles", "grid", "names", "meta")

    def __init__(self, profiles, grid, names, meta=None):
        self.profiles = np.asarray(profiles, float)   # (N, nchan)
        self.grid = np.asarray(grid, float)           # (N, 3) rot, dy, lat
        self.names = list(names)                      # (N,) morphology per row
        self.meta = meta or {}

    def __len__(self):
        return len(self.profiles)

    def fit(self, prof):
        """Nearest entry to a profile.  Returns dict with position and RMSE."""
        p = np.asarray(prof, float)
        e = np.sqrt(((self.profiles - p) ** 2).mean(1))
        j = int(np.argmin(e))
        rot, dy, lat = self.grid[j]
        return dict(morphology=self.names[j], rot=float(rot), depth=float(dy),
                    lateral=float(lat), rmse=float(e[j]), index=j)

    def save(self, path):
        np.savez_compressed(path, profiles=self.profiles, grid=self.grid,
                            names=np.array(self.names),
                            meta=np.array(sorted(self.meta.items()), dtype=object))

    @staticmethod
    def load(path):
        z = np.load(path, allow_pickle=True)
        return PositionTable(z["profiles"], z["grid"], [str(s) for s in z["names"]],
                             dict(z["meta"].tolist()) if "meta" in z else {})


def _one_morphology(args):
    """Load, simulate and sweep ONE morphology.  Top-level so it can be pickled.

    Returns (label, profiles, grid) rather than writing into shared state, which
    is what lets this run in a worker process.
    """
    (path, kind, xy, rotations, depths, laterals, d_lambda, max_comp,
     dt, t_stop, stim_amp, sigma) = args
    try:
        from . import morpho_geom as mg, morpho_cable as mc, morpho_eap as me
        from . import morpho_chan_ca1 as mca
    except ImportError:
        import morpho_geom as mg, morpho_cable as mc, morpho_eap as me
        import morpho_chan_ca1 as mca
    kw = {} if sigma is None else dict(sigma=sigma)
    c = mg.orient(mg.compartmentalize(mg.load(path), d_lambda=d_lambda,
                                      max_comp=max_comp))
    im = mc.simulate(mc.Cell(c, mca.biophys(kind)), dt=dt, t_stop=t_stop,
                     stim_amp=stim_amp, record_v=False)["im"]
    label = str(path).split("/")[-1].replace(".CNG.swc", "").replace(".swc", "")
    P, G = [], []
    for rot in rotations:
        cr = mg.rotate_z(c, rot)
        for dy in depths:
            for lat in laterals:
                sites = me.sites_3d(xy - np.array([0.0, dy]), z=lat)
                p = np.ptp(im @ me.transfer_matrix(cr, sites, **kw), 0)
                if p.max() <= 0:
                    continue
                P.append(p / p.max()); G.append((rot, dy, lat))
    return label, P, G


def build_table(morphologies, site_xy, kinds=None, rotations=(0, 90, 180, 270),
                depths=None, laterals=None, d_lambda=0.5, max_comp=550,
                dt=0.02, t_stop=9.0, stim_amp=8.0, sigma=None, progress=None,
                jobs=1):
    """Simulate each morphology once, then sweep positions with matmuls.

    site_xy must be the group's own geometry RE-REFERENCED to its first site.
    A .probe file gives absolute array coordinates -- group 5 of a Buzsaki64L
    sits near x = 1000 um -- and using them unshifted puts the electrode a
    millimetre from a cell simulated at the origin.
    """
    depths = np.arange(0, 201, 5.0) if depths is None else np.asarray(depths, float)
    laterals = np.arange(5, 71, 5.0) if laterals is None else np.asarray(laterals, float)
    xy = np.asarray(site_xy, float)
    xy = xy - xy[0]
    # One task per morphology.  Loading dominates the per-cell cost (3.2 s of
    # 3.7 s excluding the sweep), and every stage -- load, simulate, sweep -- is
    # independent between cells, so this parallelises without any shared state.
    tasks = [(q, (kinds or {}).get(q, "pvbasket"), xy, tuple(rotations),
              np.asarray(depths, float), np.asarray(laterals, float),
              d_lambda, max_comp, dt, t_stop, stim_amp, sigma)
             for q in morphologies]
    P, Gd, nm = [], [], []
    jobs = max(1, int(jobs))
    if jobs > 1 and len(tasks) > 1:
        import multiprocessing as _mp
        # A worker inherits the parent's BLAS thread count, so N processes each
        # spawning N threads oversubscribes the machine badly.  The matmuls here
        # are small (a few hundred by 8) and gain nothing from threading anyway.
        import os as _os
        _prev = {v: _os.environ.get(v) for v in
                 ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
        for v in _prev:
            _os.environ[v] = "1"
        try:
            # fork where available, spawn as a fallback.  spawn re-imports the
            # parent's __main__ in every worker, which costs a full numpy import
            # per worker -- material at 64 of them -- and outright FAILS when the
            # parent was started from stdin, because the worker then tries to
            # re-execute a "<stdin>" path that does not exist.  fork has neither
            # problem; the usual fork hazard is inherited threads, and BLAS
            # threads are already pinned to 1 just above.
            try:
                _ctx = _mp.get_context("fork")
            except ValueError:                        # not available (Windows)
                _ctx = _mp.get_context("spawn")
            with _ctx.Pool(jobs) as pool:
                for k, (label, p, g) in enumerate(
                        pool.imap(_one_morphology, tasks, chunksize=1)):
                    P.extend(p); Gd.extend(g); nm.extend([label] * len(p))
                    if progress:
                        progress(k + 1, len(tasks), label)
        finally:
            for v, old_v in _prev.items():
                if old_v is None:
                    _os.environ.pop(v, None)
                else:
                    _os.environ[v] = old_v
    else:
        for k, t in enumerate(tasks):
            label, p, g = _one_morphology(t)
            P.extend(p); Gd.extend(g); nm.extend([label] * len(p))
            if progress:
                progress(k + 1, len(tasks), label)
    if not P:
        raise ValueError("no usable positions — check site_xy and the grid")
    return PositionTable(P, Gd, nm, meta=dict(n_morph=str(len(morphologies)),
                                              n_pos=str(len(P))))


# ── uncertainty ─────────────────────────────────────────────────────────────
def split_half_floor(table, waves, n_rep=8, rng=None):
    """Position scatter from splitting one population in half.

    This is the resolution the data supports, and it is what every distance
    below must be compared against.  It is not a property of the grid: a
    cluster of 400 spikes and one of 40,000 have very different floors, and a
    fixed micrometre threshold would be far too strict for one and useless for
    the other.
    """
    rng = rng or np.random.default_rng(0)
    w = np.asarray(waves, np.float64)
    if len(w) < 40:
        return np.nan
    d = []
    for _ in range(int(n_rep)):
        q = rng.permutation(len(w)); h = len(q) // 2
        a = table.fit(profile(w[q[:h]])); b = table.fit(profile(w[q[h:]]))
        d.append(np.hypot(a["depth"] - b["depth"], a["lateral"] - b["lateral"]))
    return float(np.median(d))


def separation(a, b):
    """Distance between two fits, in the probe's own micrometres."""
    return float(np.hypot(a["depth"] - b["depth"], a["lateral"] - b["lateral"]))


# ── the two tests ───────────────────────────────────────────────────────────
def colocalised(table, waves_a, waves_b, k=3.0, floor_min=5.0, rng=None):
    """Do two populations occupy the same position?  A merge criterion.

    Passing means only that the two are spatially indistinguishable, which is
    necessary for a merge and not sufficient -- co-located cells pass trivially.
    Combine with refractory and waveform evidence; the value here is that this
    is independent of both.
    """
    fa = table.fit(profile(waves_a)); fb = table.fit(profile(waves_b))
    fl = max(x for x in (split_half_floor(table, waves_a, rng=rng),
                         split_half_floor(table, waves_b, rng=rng), 0.0)
             if np.isfinite(x))
    sep = separation(fa, fb)
    thr = max(k * fl, floor_min)
    return dict(same=bool(sep <= thr), separation=sep, floor=fl, threshold=thr,
                fit_a=fa, fit_b=fb)


def within_chunk_split(table, atom_waves, k=3.0, floor_min=7.5, rng=None):
    """Do several atoms from the SAME time window sit at different positions?

    Comparing atoms ACROSS chunks measures drift and will flag every drifting
    unit -- that mistake was made once and is the reason this function insists
    the caller supply atoms from one window.  Within a window drift cannot
    account for a separation, so a large one means two cells in one cluster.
    """
    fits = [table.fit(profile(w)) for w in atom_waves]
    floors = [split_half_floor(table, w, rng=rng) for w in atom_waves]
    fl = max([f for f in floors if np.isfinite(f)] or [0.0])
    worst, pair = 0.0, (0, 0)
    for i in range(len(fits)):
        for j in range(i + 1, len(fits)):
            s = separation(fits[i], fits[j])
            if s > worst:
                worst, pair = s, (i, j)
    thr = max(k * fl, floor_min)
    return dict(split=bool(worst > thr), separation=worst, pair=pair,
                floor=fl, threshold=thr, fits=fits)


def trajectory(table, blocks, rng=None):
    """Position per time block — a drift trace in micrometres.

    blocks is an iterable of (label, waves).  Returns one record per block with
    the fit, its floor, and the step from the previous block, so a trajectory
    can be read against its own resolution rather than eyeballed.
    """
    out, prev = [], None
    for lab, w in blocks:
        f = table.fit(profile(w))
        fl = split_half_floor(table, w, rng=rng)
        step = np.nan if prev is None else separation(prev, f)
        out.append(dict(label=lab, n=len(np.asarray(w)), floor=fl, step=step, **f))
        prev = f
    return out
