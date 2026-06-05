#!/usr/bin/env python3
# fiber_refine.py - iterative refinement of a fine spike sort into clean units.
#
# The detector imposes a refractory period (spikeDetection.refractoryPeriod, in
# samples; or refractoryMs * sr/1000): no two events in a group can be closer
# than that.  Sub-floor inter-spike intervals are therefore DUPLICATE detections
# of one physical event (most sit at exactly ISI=0), not contamination -- no
# split can fix a duplicate, only a dedup pass.  Real contamination lives in the
# band [floor, window): above the detection floor, below the biological / ISI-
# violation refractory (the quality-feature refractoryMs, default 2.0 ms).  So
# fiber-refine:
#
#   1. dedup at the imposed floor (keep the largest-amplitude spike per
#      coincidence run);
#   2. iterate, on the LARGE clusters of the current labelling:
#        variance-driven split (peaked per-channel residual profile)
#          -> gated cascade  rkk (CEM) -> dipsplit -> isolate, each sub-piece
#             kept only if it LOWERS the per-channel residual variance AND does
#             not worsen the [floor, window) refractory contamination;
#        then knn-peel: per-spike K-NN majority vote against the pool of OTHER
#        clusters' spikes; a peeled bucket whose median waveform matches the
#        target by NON-normalised (amplitude-sensitive) xcorr >= fold-thr is
#        folded in, otherwise kept as a new energy-level cluster.
#
# The measure is only meaningful after iterated circular-xcorr alignment to the
# cluster median (fiber_lib.align_xcorr), and must be the residual to the
# energy-local template in raw (un-whitened) channel space -- both handled by
# fiber_tracer.channel_residual_profile.  Gating bounds fragmentation the way the
# Klusters knn-split threshold+residual does; isolate retires a cluster that no
# method can clean.

import argparse
import os
import time
import numpy as np
from collections import namedtuple
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

try:
    from . import fiber_lib as fl
    from . import fiber_tracer as ft
    from . import fiber_session as fs
    from . import session_yaml as sy
    from . import neuro_io as nio
    from . import backend as _bk
    from .klustakwik import klustakwik as _rkk
except ImportError:                                   # script / flat-layout fallback
    import fiber_lib as fl
    import fiber_tracer as ft
    import fiber_session as fs
    import session_yaml as sy
    import neuro_io as nio
    import backend as _bk
    from klustakwik import klustakwik as _rkk

# scoring context threaded through the helpers: whitener, masked window, rate,
# imposed detection floor (samples) and contamination window upper bound (samples).
Ctx = namedtuple("Ctx", "W nmean mask sr floor window")


# ── dedup at the imposed detection refractory ────────────────────────────────
def dedup_spikes(res, ptp, floor):
    """Greedy dedup: within any run of spikes closer than `floor` samples keep
    only the largest-`ptp` one (sub-floor coincidences are duplicate detections
    of a single event).  Returns sorted indices into `res` to keep.  floor<=0
    keeps everything."""
    res = np.asarray(res)
    order = np.argsort(res, kind="mergesort")
    rs = res[order]
    pa = np.asarray(ptp)[order]
    keep = np.zeros(len(res), bool)
    last_t = -(1 << 62)
    last_i = -1
    for i in range(len(rs)):
        if rs[i] - last_t >= floor:
            keep[i] = True; last_t = rs[i]; last_i = i
        elif pa[i] > pa[last_i]:                       # same coincidence, bigger peak wins
            keep[last_i] = False; keep[i] = True; last_t = rs[i]; last_i = i
    return np.sort(order[keep])


# ── contamination measures (band = real, dup = sub-floor artifact) ───────────
def band_pct(res_sub, ctx):
    """% of ISIs in [floor, window): contamination above the detection floor and
    below the biological refractory."""
    if len(res_sub) < 10:
        return np.nan
    s = np.diff(np.sort(res_sub))
    return float(((s >= ctx.floor) & (s < ctx.window)).mean()) * 100.0


def dup_pct(res_sub, ctx):
    """% of ISIs below the imposed floor (duplicate detections)."""
    if len(res_sub) < 10:
        return np.nan
    return float((np.diff(np.sort(res_sub)) < ctx.floor).mean()) * 100.0


def _radius(w, ctx):
    al = fl.realign(w)
    return np.linalg.norm((al[:, ctx.mask, :].reshape(len(w), -1) - ctx.nmean) @ ctx.W, axis=1)


def _pcv(w, ctx):
    return ft.channel_residual_profile(w, ctx.W, ctx.nmean, ctx.mask)["mean"]


def _feats(w, ctx, d):
    """Low-dim features of a single cluster after iterated circular-xcorr
    alignment to its own median (so within-cluster jitter doesn't masquerade as
    structure)."""
    al = fl.align_xcorr(w, ref="median", iters=4)
    wc = al[:, ctx.mask, :].reshape(len(w), -1)
    wc = wc - wc.mean(0)
    U, S, _ = np.linalg.svd(wc, full_matrices=False)
    return U[:, :d] * S[:d]


# ── gated cascade: rkk -> dipsplit -> isolate ────────────────────────────────
def _gated_partition(si, sub, pv, pb, waves, res, ctx, mg, vmargin, btol, pmed=None, scorr=1.0):
    """Accept the sub-labels `sub` over cluster `si` only where a piece lowers the
    per-channel residual variance by >= vmargin AND keeps the [floor,window)
    refractory within btol of the parent AND is SHAPE-DISTINCT from the parent
    (normalised median-waveform corr < scorr) -- the last gate stops a high-rate
    unit being shattered into energy-level pieces that share its waveform shape;
    everything else falls into a residual core.  Returns a list of >=2 index
    arrays, or None if nothing qualified."""
    keep, fail = [], []
    for s in np.unique(sub):
        pc = si[sub == s]
        if len(pc) < mg:
            fail.append(pc); continue
        shape_dup = pmed is not None and _ncorr(_med(pc, waves), pmed) >= scorr
        if (not shape_dup) and _pcv(waves[pc], ctx) < pv * (1.0 - vmargin) and band_pct(res[pc], ctx) <= pb + btol:
            keep.append(pc)
        else:
            fail.append(pc)
    if not keep:
        return None
    resid = np.concatenate(fail) if fail else None
    final = list(keep) + ([resid] if (resid is not None and len(resid) >= mg) else [])
    if resid is not None and len(resid) < mg and keep:        # fold the dust into the biggest kept piece
        b = int(np.argmax([len(k) for k in keep]))
        final[b] = np.concatenate([final[b], resid])
    return final if len(final) > 1 else None


def _gated_split(si, waves, res, ctx, mg, vmargin, btol, scorr=1.0):
    """Try rkk, then dipsplit, gating each; isolate if neither cleans it."""
    pv = _pcv(waves[si], ctx)
    pb = band_pct(res[si], ctx)
    pmed = _med(si, waves) if scorr < 1.0 else None
    sub = _rkk(_feats(waves[si], ctx, 6), max_clusters=12, min_size=mg, seed=42)
    fp = _gated_partition(si, sub, pv, pb, waves, res, ctx, mg, vmargin, btol, pmed, scorr)
    if fp is not None:
        return fp, "rkk"
    if fs._HAVE_DIP:
        pcs = fs._dipsplit_rec(_feats(waves[si], ctx, 4), np.arange(len(si)), mg, 0.05)
        if len(pcs) > 1:
            sub = np.zeros(len(si), int)
            for k, p in enumerate(pcs):
                sub[p] = k
            fp = _gated_partition(si, sub, pv, pb, waves, res, ctx, mg, vmargin, btol, pmed, scorr)
            if fp is not None:
                return fp, "dip"
    return [si], "iso"


def _split_all(lab, isol, waves, res, ctx, large, mg, vmargin, btol, vpeak, vdepth, scorr=1.0):
    out = np.full(len(lab), -1, int)
    nid = 0
    nr = nd = ni = 0
    newi = isol.copy()
    for c in np.unique(lab[lab >= 0]):
        idx = np.flatnonzero(lab == c)
        if isol[idx].mean() > 0.7 or len(idx) < 2 * mg:            # retired or too small -> pass through
            out[idx] = nid; nid += 1; continue
        pcs = fs._variance_split(waves[idx], ctx.W, ctx.nmean, ctx.mask,
                                 40, vpeak, 0.10, mg, 6, max_depth=vdepth)
        for p in pcs:
            si = idx[p]
            if len(si) >= large:
                fp, how = _gated_split(si, waves, res, ctx, mg, vmargin, btol, scorr)
                nr += how == "rkk"; nd += how == "dip"; ni += how == "iso"
                if how == "iso":
                    newi[si] = True
                for q in fp:
                    out[q] = nid; nid += 1
            else:
                out[si] = nid; nid += 1
    return out, newi, nr, nd, ni


# ── knn-peel against the pool of other clusters' spikes ──────────────────────
def _gfeat(waves, ctx, d):
    al = fl.realign(waves)
    X = (al[:, ctx.mask, :].reshape(len(waves), -1) - ctx.nmean) @ ctx.W
    return PCA(n_components=min(d, X.shape[1]), random_state=0).fit_transform(X)


def _med(idx, waves):
    return np.median(fl.align_xcorr(waves[idx], ref="median", iters=4), 0)


def _ncorr(a, b, ml=4):
    """Shape-only (amplitude-normalised) best-lag correlation in [-1,1]: high when
    waveform SHAPE matches regardless of scale, so energy levels of one neuron
    (same fiber direction) score high and merge into one cell."""
    A = a.ravel(); A = A / (np.linalg.norm(A) + 1e-9); best = -1.0
    for L in range(-ml, ml + 1):
        B = np.roll(b, L, axis=0).ravel(); B = B / (np.linalg.norm(B) + 1e-9)
        best = max(best, float(A @ B))
    return best


def _match(a, b, ml=4):
    """NON-normalised (amplitude-sensitive) best-lag xcorr ratio in [0,1]: high
    only when both shape AND scale match, so energy levels are NOT collapsed."""
    A = a.ravel(); aa = float((A * A).sum()); best = 0.0
    for L in range(-ml, ml + 1):
        B = np.roll(b, L, axis=0).ravel(); num = float((A * B).sum())
        best = max(best, min(num / (aa + 1e-9), num / (float((B * B).sum()) + 1e-9)))
    return best


def _knn_apply(lab, F, waves, res, ctx, K, thr, minref, minnew, hi, scorr=1.0):
    pool = np.flatnonzero(lab >= 0)
    sz = np.bincount(lab[pool])
    big = np.flatnonzero(sz >= minref)
    pm = pool[np.isin(lab[pool], big)]
    if len(pm) <= K:
        return lab, 0, 0
    pl = lab[pm]
    nn = NearestNeighbors(n_neighbors=min(K * 4, len(pm))).fit(F[pm])
    _, ind = nn.kneighbors(F)
    vmin = int(np.ceil(thr * K))
    win = np.full(len(lab), -1, int)
    for i in np.flatnonzero(lab >= 0):
        cc = pl[ind[i]]
        own = lab[i]
        cc = cc[cc != own][:K]
        if len(cc) < K:
            continue
        v, n = np.unique(cc, return_counts=True)
        b = n.argmax()
        if n[b] >= vmin and v[b] != own:
            win[i] = v[b]
    new = lab.copy()
    nid = int(lab.max()) + 1
    mc = {}
    fo = ke = 0
    for c in np.unique(lab[lab >= 0]):
        s = np.flatnonzero(lab == c)
        w = win[s]
        cmed = None
        for ww in np.unique(w[w >= 0]):
            bk = s[w == ww]
            if len(bk) < minnew:
                continue
            bmed = _med(bk, waves)
            if scorr < 1.0:                            # SHAPE-distinctness gate
                if cmed is None:
                    cmed = _med(s, waves)
                if _ncorr(bmed, cmed) >= scorr:        # same shape as its own cluster -> not a contaminant, keep
                    continue
            if ww not in mc:
                mc[ww] = _med(np.flatnonzero(lab == ww), waves)
            if _match(bmed, mc[ww]) >= hi or (scorr < 1.0 and _ncorr(bmed, mc[ww]) >= scorr):
                new[bk] = ww; fo += 1                  # shape/amplitude match -> fold into the target
            else:
                new[bk] = nid; nid += 1; ke += 1       # distinct from source AND target -> new cluster
    return new, fo, ke


def _drop_tiny(lab, mg):
    if not (lab >= 0).any():
        return lab
    sz = np.bincount(lab[lab >= 0])
    tiny = np.flatnonzero(sz < mg)
    lab = np.where(np.isin(lab, tiny), -1, lab)
    u = np.unique(lab[lab >= 0])
    rm = {c: i for i, c in enumerate(u)}
    return np.array([rm.get(int(x), -1) for x in lab], int)


# ── per-iteration statistics ─────────────────────────────────────────────────
def merge_back(lab, waves, res, ctx, *, budget=1.0, min_sim=0.90,
               mode="normalized", verbose=True):
    """Contamination-gated agglomerative merge of an over-split sort back down to
    a reasonable count.  Greedily merges the most-similar cluster pair (by median
    waveform: shape-only when mode='normalized' -> merges energy levels of one
    neuron onto its single d(r) fiber; amplitude-sensitive when mode='amplitude'
    -> keeps energy levels apart) while similarity >= min_sim AND the merged
    cluster's [floor, window) refractory stays <= budget %.  The gate auto-finds
    the per-cluster knee, so distinct cells (whose merge would cross the
    refractory) are never fused.  Returns 0-based labels (-1 noise)."""
    import heapq
    u = [int(c) for c in np.unique(lab[lab >= 0])]
    if len(u) < 2:
        return lab.copy()
    groups = {c: np.flatnonzero(lab == c) for c in u}
    med = {c: _med(groups[c], waves) for c in u}
    sizes = {c: len(groups[c]) for c in u}
    active = set(u)
    nextid = max(u) + 1
    simfn = _ncorr if mode == "normalized" else _match
    heap = []
    for i in range(len(u)):
        for j in range(i + 1, len(u)):
            s = simfn(med[u[i]], med[u[j]])
            if s >= min_sim:
                heapq.heappush(heap, (-s, u[i], u[j]))
    nmerge = nrej = 0
    while heap:
        negs, a, b = heapq.heappop(heap)
        if a not in active or b not in active:
            continue                                   # stale entry (one side already merged)
        midx = np.concatenate([groups[a], groups[b]])
        if band_pct(res[midx], ctx) > budget:          # would over-merge distinct cells -> keep apart
            nrej += 1; continue
        active.discard(a); active.discard(b)
        nid = nextid; nextid += 1
        groups[nid] = midx
        sizes[nid] = sizes[a] + sizes[b]
        med[nid] = (sizes[a] * med[a] + sizes[b] * med[b]) / sizes[nid]   # cheap merged template
        active.add(nid); nmerge += 1
        for c in active:
            if c == nid:
                continue
            s = simfn(med[nid], med[c])
            if s >= min_sim:
                heapq.heappush(heap, (-s, nid, c))
    out = np.full(len(lab), -1, int)
    for k, c in enumerate(sorted(active)):
        out[groups[c]] = k
    if verbose:
        print(f"merge-back: {len(u)} -> {len(active)} clusters "
              f"({nmerge} merges, {nrej} gated; mode={mode}, budget={budget}%, min_sim={min_sim})")
    return out


def _iter_stats(name, lab, waves, res, ctx):
    u = np.unique(lab[lab >= 0])
    rb, du, sz, cv = [], [], [], []
    for c in u:
        idx = np.flatnonzero(lab == c)
        rb.append(band_pct(res[idx], ctx)); du.append(dup_pct(res[idx], ctx)); sz.append(len(idx))
        r = _radius(waves[idx], ctx); cv.append(r.std() / (r.mean() + 1e-9))
    rb = np.array(rb, float); du = np.array(du, float); sz = np.array(sz); cv = np.array(cv)
    swb = float(np.nansum(rb * sz) / np.nansum(sz[~np.isnan(rb)])) if len(sz) else np.nan
    swd = float(np.nansum(du * sz) / np.nansum(sz[~np.isnan(du)])) if len(sz) else np.nan
    return dict(it=name, nfib=int(len(u)),
                medBand=float(np.nanmedian(rb)) if len(rb) else np.nan,
                pct2=100.0 * float(np.nanmean(rb < 2)) if len(rb) else np.nan,
                swBand=swb, swDup=swd,
                enCV=float(np.mean(cv)) if len(cv) else np.nan,
                nbig=int((sz >= 800).sum()), rkk=0, dip=0, iso=0, fold=0, kept=0)


_HDR = f"{'iter':>5}{'nfib':>6}{'medBand':>9}{'%<2%':>6}{'swBand':>8}{'swDup':>8}{'enCV':>7}{'#big':>5}{'rkk':>4}{'dip':>4}{'iso':>4}"


def _row(s):
    return (f"{s['it']:>5}{s['nfib']:>6}{s['medBand']:>9.2f}{s['pct2']:>6.0f}"
            f"{s['swBand']:>8.2f}{s['swDup']:>8.3f}{s['enCV']:>7.3f}{s['nbig']:>5}"
            f"{s['rkk']:>4}{s['dip']:>4}{s['iso']:>4}")


# ── driver ───────────────────────────────────────────────────────────────────
def _refit_reassign(lab, waves, W, nmean, mask, mg):
    """Fit a fiber (per-cluster trajectory) from every current cluster, then
    REASSIGN each labelled spike to the fiber with the smallest whiteness
    residual (ft.run_from_seeds).  Noise (-1) is preserved; tiny clusters are
    dropped.  Re-derives membership from the cleaned templates so the next pass
    seeds off a self-consistent labelling."""
    groups = {int(c): np.flatnonzero(lab == c) for c in np.unique(lab[lab >= 0])
              if (lab == c).sum() >= 50}
    if len(groups) < 2:
        return lab
    out = ft.run_from_seeds(waves, groups, W, nmean, mask=mask)
    keymap = {k: i for i, k in enumerate(out["keys"])}
    new = lab.copy()
    nn = np.flatnonzero(lab >= 0)
    new[nn] = [keymap.get(h, -1) if h is not None else -1 for h in out["hard"][nn]]
    return _drop_tiny(new, mg)


# ── fiber-geometry tracking across iterations ────────────────────────────────
_GEOM_KEYS = ("n", "r_mean", "r_cv", "r_skew", "r_bimod",
              "cone_med", "cone_p95", "resid_med", "resid_mad",
              "traj_bend", "traj_smooth")


def geometry_tracks(snaps, waves, W, nmean, mask, n_grid=40, min_n=40):
    """Follow each FINAL fiber's geometry back through the refine snapshots.

    `snaps` is the list of (tag, labels) recorded per step (labels are over the
    SAME spikes throughout, so identity links by spike overlap -- no template
    matching needed).  For every final cluster and snapshot, the host cluster
    holding the majority of that fiber's spikes is found and its
    fiber_shape_stats reported.  Returns {final_fiber: [(tag, host, purity,
    stats), ...]} -- the time series of radius/cone/smoothness/bend as the loop
    refines, exposing structure (e.g. r_bimod and cone collapsing exactly when a
    real sub-unit separates, or a bend that stays high because it is two cells)."""
    per_snap = []
    for tag, lab in snaps:
        sc = {}
        for c in np.unique(lab[lab >= 0]):
            idx = np.flatnonzero(lab == c)
            if len(idx) >= min_n:
                sc[int(c)] = ft.fiber_shape_stats(waves[idx], W, nmean, mask, n_grid=n_grid)
        per_snap.append((tag, lab, sc))
    final_lab = per_snap[-1][1]
    tracks = {}
    for fc in np.unique(final_lab[final_lab >= 0]):
        spk = np.flatnonzero(final_lab == fc); series = []
        for tag, lab, sc in per_snap:
            sub = lab[spk]; sub = sub[sub >= 0]
            if len(sub) == 0:
                continue
            host = int(np.bincount(sub).argmax()); frac = float((sub == host).mean())
            if host in sc:
                series.append((tag, host, frac, sc[host]))
        tracks[int(fc)] = series
    return tracks


def write_geometry_tracks(tracks, path):
    """Lossless npz of the per-iteration geometry tracks (one row per
    (final_fiber, snapshot)).  Long format, no object arrays, so a viewer loads
    it straight into columns and groups by `fiber`:
        fiber[N] int, iter[N] str, host[N] int, purity[N] f8,
        stats[N,11] f8, keys[11] str (= _GEOM_KEYS, the stats column order).
    Rows are ordered fine -> ... -> final within each fiber."""
    fib, it, host, pur, rows = [], [], [], [], []
    for fc, series in sorted(tracks.items()):
        for tag, h, frac, s in series:
            fib.append(fc); it.append(str(tag)); host.append(h); pur.append(frac)
            rows.append([s[k] for k in _GEOM_KEYS])
    np.savez_compressed(path,
                        fiber=np.asarray(fib, int), iter=np.asarray(it),
                        host=np.asarray(host, int), purity=np.asarray(pur, float),
                        stats=np.asarray(rows, float).reshape(-1, len(_GEOM_KEYS)),
                        keys=np.asarray(_GEOM_KEYS))
    return path


def refine(waves, res_abs, W, nmean, mask, sr, *,
           floor=16, window_ms=2.0, iters=4, large=800, min_group=40,
           var_margin=0.05, brr_tol=0.30, var_peak=2.0, var_depth=4, split_min_corr=0.93,
           knn_k=20, knn_thr=0.3, knn_minref=50, knn_minnew=30,
           knn_dims=16, fold_thr=0.9, init_labels=None,
           conv_tol=0.0, conv_patience=2, reseed=0,
           merge_back_enable=True, merge_budget=1.0, merge_min_sim=0.92,
           merge_mode="normalized", fine_method="gmm", coarse_mg=150,
           snaps_out=None, verbose=True):
    """Iteratively refine a fine sort.  Returns (labels, stats) where labels is
    0-based (-1 = noise) over `waves`/`res_abs` and stats is the per-iteration
    list of dicts.  `init_labels` (0-based, -1 noise) is refined in place; if
    None, a fine sort is produced first with cluster_chunk_fine.

    `iters` caps the splitting phase; conv_tol>0 stops it early once nfib (within
    conv_tol fraction), swBand and enCV have all held for conv_patience iters.
    `split_min_corr` is the shape-distinctness gate: a split piece or peeled
    energy bucket whose normalised median waveform correlates >= this with its
    parent is NOT carved off (stops high-rate units being shattered into
    energy-level clones of one waveform).  merge_back_enable adds a final
    contamination-gated merge_back() after each pass.  `reseed` re-runs the whole
    loop using the refined (consolidated) labels as the seed for the next pass,
    up to `reseed` extra passes, stopping early when the pass leaves nfib/swBand
    steady -- the cleaned fibers seed a better next pass than the raw input."""
    window = int(round(window_ms * sr / 1000.0))
    ctx = Ctx(W, nmean, mask, sr, int(floor), window)
    if init_labels is None:
        lab, _ = fs.cluster_chunk_fine(waves, res_abs, W, nmean, coarse_mg, mask, sr,
                                       method=fine_method, var_split=0.0)
    else:
        lab = np.asarray(init_labels, int).copy()
    stats = [_iter_stats("fine", lab, waves, res_abs, ctx)]
    if snaps_out is not None:
        snaps_out.append(("fine", lab.copy()))
    if verbose:
        print(f"contamination window = [{floor/sr*1000:.2f}, {window_ms:.2f}] ms "
              f"([{int(floor)}, {window}] samples); split_min_corr={split_min_corr}, reseed={reseed}")
        print(_HDR); print(_row(stats[-1]))
    npass = max(1, reseed + 1)
    prev_pass = None
    for p in range(npass):
        isol = np.zeros(len(lab), bool)               # fresh isolation view per (re)seed
        stable = 0
        for it in range(iters):
            lab, isol, nr, nd, ni = _split_all(lab, isol, waves, res_abs, ctx,
                                               large, min_group, var_margin, brr_tol,
                                               var_peak, var_depth, split_min_corr)
            F = _gfeat(waves, ctx, knn_dims)
            lab, fo, ke = _knn_apply(lab, F, waves, res_abs, ctx,
                                     knn_k, knn_thr, knn_minref, knn_minnew, fold_thr, split_min_corr)
            lab = _drop_tiny(lab, min_group)
            tag = f"{p+1}.{it+1}" if reseed else str(it + 1)
            st = _iter_stats(tag, lab, waves, res_abs, ctx)
            st.update(rkk=nr, dip=nd, iso=ni, fold=fo, kept=ke)
            prev = stats[-1]; stats.append(st)
            if snaps_out is not None:
                snaps_out.append((tag, lab.copy()))
            if verbose:
                print(_row(st))
            if conv_tol > 0:
                steady = (abs(st["nfib"] - prev["nfib"]) <= conv_tol * max(prev["nfib"], 1)
                          and abs(st["swBand"] - prev["swBand"]) <= 0.01
                          and abs(st["enCV"] - prev["enCV"]) <= 0.002)
                stable = stable + 1 if steady else 0
                if stable >= conv_patience:
                    if verbose:
                        print(f"[split phase converged at iter {it + 1}]")
                    break
        if merge_back_enable:
            lab = merge_back(lab, waves, res_abs, ctx, budget=merge_budget,
                             min_sim=merge_min_sim, mode=merge_mode, verbose=verbose)
            st = _iter_stats(f"{p+1}.merge" if reseed else "merge", lab, waves, res_abs, ctx)
            stats.append(st)
            if snaps_out is not None:
                snaps_out.append((f"{p+1}.merge" if reseed else "merge", lab.copy()))
            if verbose:
                print(_row(st))
        if reseed:
            # fit new fibers from the merged clusters and REASSIGN every spike by
            # whiteness residual (membership cleanup), then loop back to splitting.
            lab = _refit_reassign(lab, waves, W, nmean, mask, min_group)
            st = _iter_stats(f"{p+1}.reasgn", lab, waves, res_abs, ctx)
            stats.append(st)
            if snaps_out is not None:
                snaps_out.append((f"{p+1}.reasgn", lab.copy()))
            if verbose:
                print(_row(st))
            cur = (st["nfib"], st["swBand"])            # outer re-seed convergence
            if (prev_pass is not None
                    and abs(cur[0] - prev_pass[0]) <= conv_tol * max(prev_pass[0], 1)
                    and abs(cur[1] - prev_pass[1]) <= 0.01):
                if verbose:
                    print(f"[reseed converged at pass {p + 1}]")
                break
            prev_pass = cur
    return lab, stats


# ── chunked / drift-aware driver ─────────────────────────────────────────────
def _chunk_bounds(res, sr, chunk_min, overlap_min):
    """Tile the session into disjoint CORE windows [lo,hi) (every spike lands in
    exactly one) plus EXTended windows [lo-ov, hi+ov) that overlap their
    neighbours -- the overlap spikes are the same physical events in both and are
    used only to link per-window fibers (overlap-anchor)."""
    chunk_s = chunk_min * 60.0 * sr; ov_s = overlap_min * 60.0 * sr
    t_min, t_max = int(res.min()), int(res.max())
    nchunks = max(1, int(np.ceil((t_max - t_min) / chunk_s)))
    chunks = []
    for c in range(nchunks):
        lo = t_min + c * chunk_s; hi = t_min + (c + 1) * chunk_s
        ext = np.flatnonzero((res >= lo - ov_s) & (res < hi + ov_s))
        core = np.flatnonzero((res >= lo) & (res < hi))
        chunks.append(dict(c=c, lo=lo, hi=hi, ext=ext, core=core,
                           tmin=(lo - t_min) / sr / 60.0))
    return chunks, nchunks


def _chunk_geometry(chunks, glab, waves, chunk_W, chunk_nm, mask, min_n=40):
    """Per global fiber, its fiber_shape_stats in EACH chunk it occupies, each
    measured in that chunk's OWN whitened frame -- the time series is the drift
    signature (radius/cone/bend evolving over the session)."""
    tracks = {}
    for ck in chunks:
        c = ck["c"]
        if c not in chunk_W:
            continue
        core = ck["core"]; gl = glab[core]
        for g in np.unique(gl[gl >= 0]):
            sel = core[gl == g]
            if len(sel) < min_n:
                continue
            s = ft.fiber_shape_stats(waves[sel], chunk_W[c], chunk_nm[c], mask)
            tracks.setdefault(int(g), []).append((c, ck["tmin"], s))
    return tracks


def write_chunk_geometry(tracks, path):
    """Lossless npz of the cross-window (drift) geometry, one row per
    (global_fiber, chunk).  Long format, no object arrays:
        fiber[N] int, chunk[N] int, t_min[N] f8, stats[N,11] f8,
        keys[11] str (= _GEOM_KEYS).  Group by `fiber`, order by `t_min` for the
    drift time series."""
    fib, ck, tmin, rows = [], [], [], []
    for g in sorted(tracks):
        for c, t, s in tracks[g]:
            fib.append(g); ck.append(c); tmin.append(t)
            rows.append([s[k] for k in _GEOM_KEYS])
    np.savez_compressed(path,
                        fiber=np.asarray(fib, int), chunk=np.asarray(ck, int),
                        t_min=np.asarray(tmin, float),
                        stats=np.asarray(rows, float).reshape(-1, len(_GEOM_KEYS)),
                        keys=np.asarray(_GEOM_KEYS))
    return path


def load_geometry(path):
    """Load a .geom/.geomchunk npz into a dict with the index columns plus each
    stat as its own named column (so a viewer/report can do g['cone_med'] without
    knowing the stats-matrix layout).  Works for both writers."""
    z = np.load(path, allow_pickle=False)
    keys = [str(k) for k in z["keys"]]
    out = {k: z[k] for k in z.files if k != "stats"}
    out["keys"] = keys
    for j, k in enumerate(keys):
        out[k] = z["stats"][:, j]
    return out


def refine_chunked(waves, res, base, elec, ntotal, nsamp, nchan, gch, mask, sr,
                   chunk_min, overlap_min, *, init=None, refine_kw=None,
                   min_group=40, track_geometry=False, make_bundles=False, verbose=True):
    """Drift-aware refine: window the session, fit a SEPARATE whitener + run the
    full refine loop INSIDE each window (so each window is quasi-stationary),
    then link per-window fibers by overlap-anchor (fs.link_chunks: same physical
    spikes in adjacent windows' overlap prove identity).  Final per-spike label
    comes from each spike's CORE window.  Returns (global_labels, n_global,
    chunk_tracks|None, bundles|None).  This is the correct way to run on a long
    drifting session -- pooling the whole session into one trajectory smears it."""
    refine_kw = dict(refine_kw or {})
    chunks, nchunks = _chunk_bounds(res, sr, chunk_min, overlap_min)
    filmm = nio.open_signal(f"{base}.fil", ntotal)
    ext_idx = [np.array([], int)] * nchunks
    ext_lab = [np.array([], int)] * nchunks
    chunk_W = {}; chunk_nm = {}
    for ck in chunks:
        c, ext = ck["c"], ck["ext"]
        if len(ext) < 2 * min_group:
            if verbose:
                print(f"[chunk {c+1}/{nchunks}] {len(ck['core'])} core ({len(ext)} ext) -> skipped (small)")
            continue
        s0 = int(res[ext].min()) - nsamp; s1 = int(res[ext].max()) + nsamp + 1
        Wc, nmc, _ = fs.fil_chunk_whitener(filmm, gch, s0, s1, res[ext], nsamp, mask)
        init_c = init[ext] if init is not None else None
        labc, _ = refine(waves[ext], res[ext], Wc, nmc, mask, sr,
                         init_labels=init_c, min_group=min_group, verbose=False, **refine_kw)
        ext_idx[c] = ext; ext_lab[c] = labc; chunk_W[c] = Wc; chunk_nm[c] = nmc
        if verbose:
            print(f"[chunk {c+1}/{nchunks}] t={ck['tmin']:.1f}m  {len(ck['core'])} core "
                  f"({len(ext)} ext) -> {len(np.unique(labc[labc >= 0]))} fibers")
    gid, nglob = fs.link_chunks(ext_idx, ext_lab)
    glab = np.full(len(res), -1, int)                       # final label by CORE window
    for ck in chunks:
        c, core = ck["c"], ck["core"]
        if len(ext_idx[c]) == 0 or len(core) == 0:
            continue
        ii = np.searchsorted(ext_idx[c], core)              # ext_idx sorted, core subset of ext
        labs = ext_lab[c][ii]
        glab[core] = [gid.get((c, int(l)), -1) if l >= 0 else -1 for l in labs]
    if verbose:
        print(f"[chunked] {nglob} global fibers across {nchunks} windows "
              f"(overlap-anchor linked); {int((glab >= 0).sum())}/{len(glab)} spikes assigned")
    tracks = _chunk_geometry(chunks, glab, waves, chunk_W, chunk_nm, mask) if track_geometry else None
    bundles = _chunk_bundles(chunks, glab, waves, chunk_W, chunk_nm, mask) if make_bundles else None
    return glab, nglob, tracks, bundles


def _chunk_bundles(chunks, glab, waves, chunk_W, chunk_nm, mask, npos=50, min_n=40):
    """For every (global fiber, chunk) fit the fiber trajectory in that chunk's
    whitened frame and UN-whiten it to a template curve r*d(r) in raw feature
    space -- comparable across chunks despite their different whiteners, so a
    bundle's per-chunk curves can be drawn together and their spread read as
    drift.  Returns long-format arrays for the .bundles npz."""
    fib, ch, tmin, cnt, curves = [], [], [], [], []
    for ck in chunks:
        c = ck["c"]
        if c not in chunk_W:
            continue
        Wc, nmc = chunk_W[c], chunk_nm[c]; Winv = np.linalg.pinv(Wc)
        core = ck["core"]; gl = glab[core]
        for g in np.unique(gl[gl >= 0]):
            sel = core[gl == g]
            if len(sel) < min_n:
                continue
            Wal = fl.realign(waves[sel])
            X = (Wal[:, mask, :].reshape(len(sel), -1) - nmc) @ Wc
            r = np.linalg.norm(X, axis=1); tr = ft.trajectory(X)
            rg = np.linspace(np.quantile(r, 0.05), np.quantile(r, 0.95), npos)
            tmpl = (rg[:, None] * ft.predict_many(tr, rg)) @ Winv + nmc   # un-whitened curve
            fib.append(int(g)); ch.append(c); tmin.append(ck["tmin"]); cnt.append(len(sel))
            curves.append(tmpl)
    return (np.asarray(fib, int), np.asarray(ch, int), np.asarray(tmin, float),
            np.asarray(cnt, int), np.asarray(curves, float))


def write_bundles(arrays, path):
    """Write the .bundles npz consumed by fiber_view.load_bundles_npz:
        fiber[N] int, chunk[N] int, t_min[N] f8, count[N] int,
        curves[N, NPOS, nfeat] f8 (un-whitened template curves)."""
    fib, ch, tmin, cnt, curves = arrays
    np.savez_compressed(path, fiber=fib, chunk=ch, t_min=tmin, count=cnt,
                        curves=curves if len(curves) else np.zeros((0, 0, 0)))
    return path


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        prog="fiber-refine",
        description="Dedup at the imposed refractory, then iteratively split/peel a "
                    "fine sort into clean units; writes a refined .clu (+ deduped .res).")
    ap.add_argument("session", help="session basename or folder (finds <session>.yaml)")
    ap.add_argument("group", type=int, help="1-based spike group")
    ap.add_argument("--channels", default=None, help="override: comma-separated physical channels")
    ap.add_argument("--ntotal", type=int, default=None)
    ap.add_argument("--nsamp", type=int, default=None)
    ap.add_argument("--nchan", type=int, default=None)
    ap.add_argument("--sr", type=float, default=None)
    ap.add_argument("--in-clu", default=None,
                    help="path to the input sort to refine; "
                         "default = canonical .clu if present, else a fresh fine sort")
    ap.add_argument("--out-variant", default="refine", help="variant tag for the output .clu/.res")
    ap.add_argument("--refr-floor", type=int, default=None,
                    help="imposed detection refractory (samples); default = from yaml")
    ap.add_argument("--refr-window-ms", type=float, default=2.0,
                    help="biological/ISI-violation window upper bound (ms); contamination is [floor, window)")
    ap.add_argument("--no-dedup", action="store_true", help="skip the sub-floor dedup pass")
    ap.add_argument("--iters", type=int, default=10, help="max splitting iterations (cap)")
    ap.add_argument("--converge", dest="converge", action="store_true", default=True,
                    help="stop the splitting phase early once nfib/swBand/enCV are steady (default on)")
    ap.add_argument("--no-converge", dest="converge", action="store_false")
    ap.add_argument("--converge-tol", type=float, default=0.01,
                    help="nfib change (fraction) below which an iteration counts as steady")
    ap.add_argument("--converge-patience", type=int, default=2,
                    help="number of consecutive steady iters required to stop")
    ap.add_argument("--merge-back", dest="merge_back", action="store_true", default=True,
                    help="final contamination-gated merge-back to a reasonable count (default on)")
    ap.add_argument("--no-merge-back", dest="merge_back", action="store_false")
    ap.add_argument("--merge-budget", type=float, default=1.0,
                    help="max merged-cluster [floor,window) band%% to accept a merge")
    ap.add_argument("--merge-min-sim", type=float, default=0.92,
                    help="min median-waveform similarity to consider a merge")
    ap.add_argument("--merge-mode", choices=["normalized", "amplitude"], default="normalized",
                    help="normalized = merge energy levels (neuron count); amplitude = keep them")
    ap.add_argument("--split-min-corr", type=float, default=0.93,
                    help="shape-distinctness gate: do NOT carve off a split piece / energy bucket whose "
                         "normalised median waveform correlates >= this with its parent (stops over-fragmenting "
                         "high-rate units into energy-level clones); 1.0 disables")
    ap.add_argument("--reseed", type=int, default=0,
                    help="re-run the whole loop (split -> merge -> refit fibers -> reassign) using the "
                         "refined labels as the next seed, up to N extra passes (e.g. 1 = 2 passes); 0 = single pass")
    ap.add_argument("--track-geometry", action="store_true",
                    help="record per-fiber geometry (radius/cone/smoothness/bend) at every iteration and "
                         "write <base>.geom.<group>.npz tracking each final fiber back through the loop")
    ap.add_argument("--large", type=int, default=800, help="only clusters >= this are split each iter")
    ap.add_argument("--min-group", type=int, default=40)
    ap.add_argument("--var-margin", type=float, default=0.05,
                    help="min per-channel residual-variance reduction to accept a gated sub-split")
    ap.add_argument("--brr-tol", type=float, default=0.30,
                    help="max allowed increase (pp) in [floor,window) refractory for a gated sub-split")
    ap.add_argument("--var-peak", type=float, default=2.0, help="var-split trigger (max/median channel variance)")
    ap.add_argument("--var-depth", type=int, default=4)
    ap.add_argument("--knn-k", type=int, default=20)
    ap.add_argument("--knn-thr", type=float, default=0.3, help="K-NN majority fraction to peel a spike")
    ap.add_argument("--knn-minref", type=int, default=50)
    ap.add_argument("--knn-minnew", type=int, default=30)
    ap.add_argument("--knn-dims", type=int, default=16)
    ap.add_argument("--fold-thr", type=float, default=0.9,
                    help="non-normalised median-xcorr above which a peeled bucket is folded (else kept as new)")
    ap.add_argument("--fine-method", choices=["gmm", "fiber", "none"], default="gmm",
                    help="method for the initial fine sort when no --in-clu is given")
    ap.add_argument("--chunk-minutes", type=float, default=0.0,
                    help="drift-aware mode: window the session into CORE chunks of this many minutes, "
                         "refine each in its own whitened frame, and link fibers across windows by "
                         "overlap-anchor; 0 = single whole-session pass (assumes stationary)")
    ap.add_argument("--chunk-overlap-minutes", type=float, default=1.0,
                    help="overlap between adjacent windows used for overlap-anchor linking (drift-aware mode)")
    ap.add_argument("--bundles", action="store_true",
                    help="drift-aware mode: also write <base>.bundles.<group>.npz (per-chunk un-whitened "
                         "template curves per global fiber) for the fiber-view-gui bundle table")
    ap.add_argument("--gpu", action="store_true")
    a = ap.parse_args()

    if a.gpu:
        on = _bk.use_gpu(True)
        print(f"[fiber_refine] GPU requested: backend = {_bk.backend_name()}"
              + ("" if on else " (CuPy/CUDA unavailable -> CPU)"))

    cfg = sy.resolve_session_params(a.session, a.group, channels=a.channels, ntotal=a.ntotal,
                                    nchan=a.nchan, nsamp=a.nsamp, sr=a.sr)
    base = cfg["base"]; elec = a.group
    ntotal = cfg["ntotal"]; nchan = cfg["nchan"]; nsamp = cfg["nsamp"]; sr = cfg["sr"]
    gch = np.array(cfg["channels"], int)
    mask = fl.MASK_FULL

    floor = a.refr_floor
    if floor is None:
        floor, src = sy.refractory_period_samples(a.session, a.group, sr=sr)
        print(f"[fiber_refine] imposed refractory = {floor} samples "
              f"({floor / sr * 1000:.2f} ms) [{src}]")
    else:
        print(f"[fiber_refine] imposed refractory = {floor} samples "
              f"({floor / sr * 1000:.2f} ms) [--refr-floor]")

    t0 = time.time()
    res = fs.read_res(base, elec)
    spk, spkpath = fs.open_spkD(base, elec, nsamp, nchan)
    assert spk.shape[0] == len(res), f".res {len(res)} vs {spkpath} {spk.shape[0]}"
    waves = np.asarray(spk[:], dtype=float)
    init = None
    if a.in_clu is not None and os.path.exists(a.in_clu):
        _, ids = nio.read_clu_file(a.in_clu, n_spikes=len(res))
        init = ids.astype(int) - 1                     # NeuroSuite 1-based -> 0-based, 0/noise -> -1
        init[init < 0] = -1
    else:
        try:
            _, ids = nio.read_clu(base, elec, n_spikes=len(res), prefer=nio.prefer_canonical())
            init = ids.astype(int) - 1; init[init < 0] = -1
        except FileNotFoundError:
            init = None
    print(f"loaded {len(res)} spikes ({spkpath}); "
          f"{'input sort '+str(len(np.unique(init[init>=0])))+' clusters' if init is not None else 'no input .clu -> fresh fine sort'}")

    if not a.no_dedup and floor > 0:
        ptp = np.ptp(waves.reshape(len(waves), -1), axis=1)
        keep = dedup_spikes(res, ptp, floor)
        n_dup = len(res) - len(keep)
        n_exact = int((np.diff(np.sort(res)) == 0).sum())
        res = res[keep]; waves = waves[keep]
        if init is not None:
            init = init[keep]
        print(f"dedup: kept {len(keep)} of {len(keep)+n_dup} "
              f"({n_dup} removed; {n_exact} exact ISI=0)")

    if a.chunk_minutes and a.chunk_minutes > 0:
        refine_kw = dict(floor=floor, window_ms=a.refr_window_ms, iters=a.iters,
                         large=a.large, min_group=a.min_group, var_margin=a.var_margin,
                         brr_tol=a.brr_tol, var_peak=a.var_peak, var_depth=a.var_depth,
                         split_min_corr=a.split_min_corr, knn_k=a.knn_k, knn_thr=a.knn_thr,
                         knn_minref=a.knn_minref, knn_minnew=a.knn_minnew, knn_dims=a.knn_dims,
                         fold_thr=a.fold_thr, conv_tol=(a.converge_tol if a.converge else 0.0),
                         conv_patience=a.converge_patience, reseed=a.reseed,
                         merge_back_enable=a.merge_back, merge_budget=a.merge_budget,
                         merge_min_sim=a.merge_min_sim, merge_mode=a.merge_mode,
                         fine_method=a.fine_method)
        glab, nglob, tracks, bundles = refine_chunked(
            waves, res, base, elec, ntotal, nsamp, nchan, gch, mask, sr,
            a.chunk_minutes, a.chunk_overlap_minutes, init=init, refine_kw=refine_kw,
            min_group=a.min_group, track_geometry=a.track_geometry,
            make_bundles=a.bundles, verbose=True)
        ids = np.where(glab < 0, 0, glab + 1).astype(np.int64)
        clu_path = nio.write_clu(base, elec, ids, variant=a.out_variant)
        res_path = nio.write_res(base, elec, res, variant=a.out_variant)
        print(f"wrote {clu_path}\n      {res_path}")
        if tracks is not None:
            gpath = write_chunk_geometry(tracks, f"{base}.geomchunk.{elec}.npz")
            print(f"      {gpath}  ({len(tracks)} fibers across windows)")
        if bundles is not None:
            bpath = write_bundles(bundles, f"{base}.bundles.{elec}.npz")
            print(f"      {bpath}  ({len(np.unique(bundles[0]))} bundles)  [view with fiber-view-gui]")
        print(f"[done] {nglob} global fibers; t={time.time()-t0:.0f}s")
        return

    # whitener from the .fil baseline over the (deduped) spike span
    filmm = nio.open_signal(f"{base}.fil", ntotal)
    s0 = int(res.min()) - nsamp; s1 = int(res.max()) + nsamp + 1
    W, nmean, _ = fs.fil_chunk_whitener(filmm, gch, s0, s1, res, nsamp, mask)

    snaps = [] if a.track_geometry else None
    lab, stats = refine(waves, res, W, nmean, mask, sr,
                        floor=floor, window_ms=a.refr_window_ms, iters=a.iters,
                        large=a.large, min_group=a.min_group,
                        var_margin=a.var_margin, brr_tol=a.brr_tol,
                        var_peak=a.var_peak, var_depth=a.var_depth, split_min_corr=a.split_min_corr,
                        knn_k=a.knn_k, knn_thr=a.knn_thr, knn_minref=a.knn_minref,
                        knn_minnew=a.knn_minnew, knn_dims=a.knn_dims,
                        fold_thr=a.fold_thr, init_labels=init,
                        conv_tol=(a.converge_tol if a.converge else 0.0),
                        conv_patience=a.converge_patience, reseed=a.reseed,
                        merge_back_enable=a.merge_back, merge_budget=a.merge_budget,
                        merge_min_sim=a.merge_min_sim, merge_mode=a.merge_mode,
                        fine_method=a.fine_method, snaps_out=snaps, verbose=True)

    ids = np.where(lab < 0, 0, lab + 1).astype(np.int64)   # 0 = noise, clusters 1..K
    clu_path = nio.write_clu(base, elec, ids, variant=a.out_variant)
    res_path = nio.write_res(base, elec, res, variant=a.out_variant)
    tsv = f"{base}.refine.{elec}.tsv"
    with open(tsv, "w") as f:
        f.write("iter\tnfib\tmedBand\tpct<2\tswBand\tswDup\tenCV\tnbig\trkk\tdip\tiso\tfold\tkept\n")
        for s in stats:
            f.write(f"{s['it']}\t{s['nfib']}\t{s['medBand']:.3f}\t{s['pct2']:.1f}\t"
                    f"{s['swBand']:.3f}\t{s['swDup']:.3f}\t{s['enCV']:.4f}\t{s['nbig']}\t"
                    f"{s['rkk']}\t{s['dip']}\t{s['iso']}\t{s['fold']}\t{s['kept']}\n")
    print(f"wrote {clu_path}\n      {res_path}\n      {tsv}")
    if snaps is not None:
        tracks = geometry_tracks(snaps, waves, W, nmean, mask)
        gpath = write_geometry_tracks(tracks, f"{base}.geom.{elec}.npz")
        print(f"      {gpath}  ({len(tracks)} fibers x {len(snaps)} snapshots)")
    print(f"[done] t={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
