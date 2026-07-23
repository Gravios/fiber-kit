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

_LP = "\u25b8 fiber-refine"
_IND = " " * (len(_LP) + 3)
def _log(m=""): print(f"{_LP} \u00b7 {m}" if m else _LP)
def _det(k, v, w=13): print(f"{_IND}{k:<{w}} {v}")
from collections import namedtuple
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

try:
    from . import fiber_lib as fl
    from . import fiber_tracer as ft
    from . import fiber_session as fs
    from . import session_yaml as sy
    from . import neuro_io as nio
    from . import fiber_pca as _fpca
    from . import fiber_split as fsp
    from . import fiber_geometry as fg
    from . import backend as _bk
    from . import fiber_ccg as ccg
    from .klustakwik import klustakwik as _rkk
except ImportError:                                   # script / flat-layout fallback
    import fiber_lib as fl
    import fiber_tracer as ft
    import fiber_session as fs
    import session_yaml as sy
    import neuro_io as nio
    import fiber_pca as _fpca
    import fiber_split as fsp
    import fiber_geometry as fg
    import backend as _bk
    import fiber_ccg as ccg
    from klustakwik import klustakwik as _rkk

# scoring context threaded through the helpers: whitener, masked window, rate,
# imposed detection floor (samples) and contamination window upper bound (samples).
Ctx = namedtuple("Ctx", "W nmean mask sr floor window basis")
Ctx.__new__.__defaults__ = (None,)        # `basis`: optional global ndm_pca basis for shape features


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
    structure).  SHAPE features: if ctx.basis (the global ndm_pca basis) is set, project
    the aligned waveforms onto it (shared basis across chunks); else local SVD -> top d."""
    al = fl.align_xcorr(w, ref="median", iters=4)
    if ctx.basis is not None:
        F = _fpca.cluster_features(al, ctx.basis, realign=False, dims=d)
        if F is not None:
            return F
    wc = al[:, ctx.mask, :].reshape(len(w), -1)
    wc = wc - wc.mean(0)
    U, S, _ = np.linalg.svd(wc, full_matrices=False)
    return U[:, :d] * S[:d]


def _dipsplit_realign(si, waves, ctx, mg, alpha, dim=4, depth=0, maxd=4):
    """Per-node realign dipsplit (refine side of --dip-realign): re-align + re-featurize via
    _feats (which aligns to the node's OWN median) at EACH bisection, so a deeper split is judged
    on its own alignment, not the parent's.  fs._dipsplit_rec features the whole cluster once and
    recurses on fixed scores; this re-derives them per node.  Returns index arrays of positions
    within si."""
    n = len(si)
    if not fs._HAVE_DIP or n < 2 * mg or depth > maxd:
        return [np.arange(n)]
    F = _feats(waves[si], ctx, dim)
    km = fs.KMeans(2, n_init=4, random_state=0).fit(F)
    a = np.flatnonzero(km.labels_ == 0); b = np.flatnonzero(km.labels_ == 1)
    if len(a) < mg or len(b) < mg:
        return [np.arange(n)]
    dr = km.cluster_centers_[1] - km.cluster_centers_[0]; dr /= np.linalg.norm(dr) + 1e-9
    _, p = fs._diptest.diptest(np.ascontiguousarray(F @ dr))
    if p >= alpha:
        return [np.arange(n)]
    out = []
    for loc in (a, b):
        for piece in _dipsplit_realign(si[loc], waves, ctx, mg, alpha, dim, depth + 1, maxd):
            out.append(loc[piece])
    return out


def _rkk_realign(si, waves, ctx, dims, max_clusters, mg, iters=2, delete=True):
    """Per-cluster realign rkk (refine side of --rkk-realign): seed with rkk on the cluster's
    aligned features, then iterate {realign EACH sub-cluster to its own median -> re-featurize ->
    re-cluster}, stopping when the count stabilises, so the final CEM runs on consistently
    per-cluster-aligned features.  Returns per-spike sub-labels over si."""
    W = waves[si]
    F = _feats(W, ctx, dims)
    lab = _rkk(F, max_clusters=max_clusters, min_size=mg, seed=42, delete=delete)
    for _ in range(max(0, iters)):
        Wal = np.array(W, dtype=float)
        for c in np.unique(lab):
            ix = np.flatnonzero(lab == c)
            if len(ix) >= 8:
                Wal[ix] = fl.align_xcorr(W[ix], ref="median", iters=6, maxlag=6)
        wc = Wal[:, ctx.mask, :].reshape(len(W), -1); wc = wc - wc.mean(0)
        F = _fpca.cluster_features(Wal, ctx.basis, realign=False, dims=dims) if ctx.basis is not None else None
        if F is None:
            U, S, _ = np.linalg.svd(wc, full_matrices=False); F = U[:, :dims] * S[:dims]
        new = _rkk(F, max_clusters=max_clusters, min_size=mg, seed=42, delete=delete)
        stop = len(np.unique(new)) == len(np.unique(lab))
        lab = new
        if stop:
            break
    return lab


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


def _gated_split(si, waves, res, ctx, mg, vmargin, btol, scorr=1.0,
                 dip_realign=True, rkk_realign=True, rkk_iters=2, dip_first=True,
                 ccg_refrac_ms=0.0, rkk_delete=True):
    """Try dip-bisection and rkk, gating each; isolate if neither cleans it.

    dip_first (default) runs the dip-axis bisection BEFORE rkk.  On GT the welds
    that survive refine are near-linearly separable along a single dip axis
    (median best-axis Cohen's d ~3), which dipsplit targets directly; rkk's
    full-cov CEM instead optimises penalised likelihood and tends to peel an
    outlier tail rather than bisect the two cells -- and an rkk mis-split that
    clears _gated_partition would stop the cascade before dip is ever reached.
    _gated_partition gates either splitter (variance drop + refractory + shape-
    distinct), so the reorder cannot over-split a clean cell -- it only changes
    which splitter gets first crack."""
    pv = _pcv(waves[si], ctx)
    pb = band_pct(res[si], ctx)
    pmed = _med(si, waves) if scorr < 1.0 else None

    def _ccg_ok(fp):
        # Curation-INDEPENDENT veto: reject a split the refractory cross-correlogram
        # calls spurious (one neuron) WHERE IT HAS POWER; abstain (keep) otherwise.
        # The timing test is independent of the waveform space the split was made on,
        # but only fires at sufficient rate*duration -- at low rates it abstains and
        # the waveform gates in _gated_partition stand.
        if not ccg_refrac_ms or ccg_refrac_ms <= 0 or fp is None or len(fp) < 2:
            return fp
        refr = ccg.refrac_samples(ccg_refrac_ms, ctx.sr)
        dur = float(res.max() - res.min())
        big = sorted(fp, key=len, reverse=True)[:2]
        ta = np.sort(res[big[0]].astype(np.int64)); tb = np.sort(res[big[1]].astype(np.int64))
        if ccg.split_gate(ta, tb, dur, refr).get("verdict") == "spurious":
            return None                                   # powered + refractory dip -> one cell
        return fp

    def _try_rkk():
        sub = (_rkk_realign(si, waves, ctx, 6, 12, mg, rkk_iters, delete=rkk_delete) if rkk_realign
               else _rkk(_feats(waves[si], ctx, 6), max_clusters=12, min_size=mg, seed=42, delete=rkk_delete))
        return _ccg_ok(_gated_partition(si, sub, pv, pb, waves, res, ctx, mg, vmargin, btol, pmed, scorr))

    def _try_dip():
        if not fs._HAVE_DIP:
            return None
        pcs = (_dipsplit_realign(si, waves, ctx, mg, 0.05, 4) if dip_realign
               else fs._dipsplit_rec(_feats(waves[si], ctx, 4), np.arange(len(si)), mg, 0.05))
        if len(pcs) <= 1:
            return None
        sub = np.zeros(len(si), int)
        for k, p in enumerate(pcs):
            sub[p] = k
        return _ccg_ok(_gated_partition(si, sub, pv, pb, waves, res, ctx, mg, vmargin, btol, pmed, scorr))

    order = ((("dip", _try_dip), ("rkk", _try_rkk)) if dip_first
             else (("rkk", _try_rkk), ("dip", _try_dip)))
    for how, fn in order:
        fp = fn()
        if fp is not None:
            return fp, how
    return [si], "iso"


def _feat_var_top3(idx, waves, ctx, k=3):
    """Mean of the top-k whitened-feature variances of a cluster -- the curator's split trigger.
    On g5 split-source clusters carry feat_var_top3 ~3.6x the population median (p50 5.84M vs
    1.64M), while kurtosis/skewness/ISI are NOT discriminative -- so feature variance is the
    signal that says 'mixture, split me'."""
    Wf = (waves[idx][:, ctx.mask, :].reshape(len(idx), -1) - ctx.nmean) @ ctx.W
    return float(np.sort(Wf.var(0))[::-1][:k].mean())


def _split_all(lab, isol, waves, res, ctx, large, mg, vmargin, btol, vpeak, vdepth, scorr=1.0,
               dip_realign=True, rkk_realign=True, rkk_iters=2, var_mult=0.0, dip_first=True, ccg_refrac_ms=0.0,
               rkk_delete=True,
               nudge=False, nudge_max=3, nudge_amp_pct=40.0, nudge_min_ch=4, nudge_alpha=0.01):
    out = np.full(len(lab), -1, int)
    nid = 0
    nr = nd = ni = 0
    newi = isol.copy()
    amp_thr = np.inf
    fvar = {}; vmed = 0.0                                  # curator split-variance trigger
    if var_mult and var_mult > 0:
        for c in np.unique(lab[lab >= 0]):
            idx = np.flatnonzero(lab == c)
            if isol[idx].mean() <= 0.7 and len(idx) >= 2 * mg:
                fvar[c] = _feat_var_top3(idx, waves, ctx)
        vmed = float(np.median(list(fvar.values()))) if fvar else 0.0
    if nudge:                                              # low-amp gate threshold over the active clusters
        amps = []
        for c in np.unique(lab[lab >= 0]):
            idx = np.flatnonzero(lab == c)
            if isol[idx].mean() <= 0.7 and len(idx) >= 2 * mg:
                amps.append(fs._amp_spread(waves[idx], ctx.mask)[0])
        if amps:
            amp_thr = float(np.percentile(amps, nudge_amp_pct))
    for c in np.unique(lab[lab >= 0]):
        idx = np.flatnonzero(lab == c)
        if isol[idx].mean() > 0.7 or len(idx) < 2 * mg:            # retired or too small -> pass through
            out[idx] = nid; nid += 1; continue
        if var_mult and var_mult > 0 and vmed > 0 and fvar.get(c, np.inf) < var_mult * vmed:
            out[idx] = nid; nid += 1; continue                    # below curator split-variance bar -> leave intact
        pcs = fs._variance_split(waves[idx], ctx.W, ctx.nmean, ctx.mask,
                                 40, vpeak, 0.10, mg, 6, max_depth=vdepth)
        for p in pcs:
            si = idx[p]
            if len(si) >= large:
                fp, how = _gated_split(si, waves, res, ctx, mg, vmargin, btol, scorr,
                                       dip_realign, rkk_realign, rkk_iters, dip_first, ccg_refrac_ms,
                                       rkk_delete=rkk_delete)
                if how == "iso" and nudge and len(si) >= 2 * mg:    # offset-overlay pass on what isolated
                    amp, nch = fs._amp_spread(waves[si], ctx.mask)
                    if amp <= amp_thr and nch >= nudge_min_ch:
                        npc = fs._nudge_split(waves[si], ctx.mask, 4, mg, nudge_alpha, nudge_max)
                        if len(npc) > 1:                            # accept directly (residual-neutral)
                            fp = [si[q] for q in npc]; how = "nudge"
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


def _knn_apply(lab, F, waves, res, ctx, K, thr, minref, minnew, hi, scorr=1.0, off_thr=None):
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
            fold = _match(bmed, mc[ww]) >= hi or (scorr < 1.0 and _ncorr(bmed, mc[ww]) >= scorr)
            if fold and off_thr is not None:                 # INTER-CHANNEL TIMING veto: a small group that
                ob = fg.interchannel_offsets(bmed, method="xcorr")   # is amplitude/shape-similar to the target
                ot = fg.interchannel_offsets(mc[ww], method="xcorr") # but timing-distinct is a DIFFERENT cell --
                if fg.offset_distance(ob, ot) > off_thr:             # don't shatter it into the look-alike (294/295)
                    fold = False
            if fold:
                new[bk] = ww; fo += 1                  # shape/amplitude (and timing) match -> fold into the target
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
def _amp_profile_corr(ta, tb):
    """Cross-channel correlation of two templates' per-channel peak-to-peak AMPLITUDE
    profiles -- the magnitude term of the Omlor-Giese anechoic model (eq. 10)."""
    a = ta.max(0) - ta.min(0); b = tb.max(0) - tb.min(0)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def merge_back(lab, waves, res, ctx, *, budget=1.0, min_sim=0.90,
               mode="normalized", warp_thr=None, warp_recall=None, warp_resid_thr=None, amp_thr=0.7,
               warp_resid_thr_int=None, warp_resid_thr_pyr=None, raw_waves=None, sr=32552.0, verbose=True):
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
    gd = ({c: fg.group_delay_profile(med[c]) for c in u}
          if (warp_thr is not None or warp_recall is not None or warp_resid_thr is not None
              or warp_resid_thr_int is not None or warp_resid_thr_pyr is not None) else None)
    # cell-type-aware (dual) warp-residual threshold.  Interneurons have far stiffer per-channel timing
    # than pyramidal cells (offset RMS ~0.23 vs ~0.72), so a single-channel group-delay residual that is
    # benign jitter for a pyr is a real different-source signature for an int -> it wants a TIGHTER veto.
    # Type each cluster from its RAW median (classify_celltype: trough-to-peak width; stderiv breaks it, so
    # this needs raw_waves); a pair touching an interneuron uses the int threshold.  Absent the dual thrs
    # or raw_waves, falls back to the single warp_resid_thr (unchanged behaviour).
    dual = raw_waves is not None and (warp_resid_thr_int is not None or warp_resid_thr_pyr is not None)
    ctype = {c: fg.classify_celltype(_med(groups[c], raw_waves), sr) for c in u} if dual else {}
    def _resid_thr(a, b):
        if not dual:
            return warp_resid_thr
        t_int = warp_resid_thr_int if warp_resid_thr_int is not None else warp_resid_thr
        t_pyr = warp_resid_thr_pyr if warp_resid_thr_pyr is not None else warp_resid_thr
        return t_int if (ctype.get(a) == "int" or ctype.get(b) == "int") else t_pyr
    def _warp_ok(a, b):   # spatio-temporal WARP coherence gate on the (clean) median templates
        return warp_thr is None or fg.warp_correlation(gd[a], gd[b]) >= warp_thr
    def _warp_resid_ok(a, b):   # single-channel warp-incongruity SUB-gate (layers on the warp gate/recall):
        # among coherent-overall pairs, reject when one channel's group-delay residual betrays a different
        # co-located source; warp_channel_incongruity returns 0.0 when warp < 0.85, so it only acts on
        # already-coherent pairs and is a no-op below that.  The threshold is cell-type-aware (see above).
        thr = _resid_thr(a, b)
        return thr is None or fg.warp_channel_incongruity(gd[a], gd[b]) <= thr
    def _admit(a, b, sim):
        if sim >= min_sim and _warp_ok(a, b) and _warp_resid_ok(a, b):
            return True                      # cosine-selected merge (energy levels), warp-gated
        if (warp_recall is not None and fg.warp_correlation(gd[a], gd[b]) >= warp_recall
                and _amp_profile_corr(med[a], med[b]) >= amp_thr and _warp_resid_ok(a, b)):
            return True                      # DRIFT-FRAGMENT recall: low cosine but coherent warp + amplitude
        return False
    sizes = {c: len(groups[c]) for c in u}
    active = set(u)
    nextid = max(u) + 1
    simfn = _ncorr if mode == "normalized" else _match
    heap = []
    for i in range(len(u)):
        for j in range(i + 1, len(u)):
            s = simfn(med[u[i]], med[u[j]])
            if _admit(u[i], u[j], s):
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
        if gd is not None:
            gd[nid] = fg.group_delay_profile(med[nid])
        if dual:                                 # a merge touching an interneuron stays int (the stiffer gate)
            ctype[nid] = "int" if (ctype.get(a) == "int" or ctype.get(b) == "int") else "pyr"
        active.add(nid); nmerge += 1
        for c in active:
            if c == nid:
                continue
            s = simfn(med[nid], med[c])
            if _admit(nid, c, s):
                heapq.heappush(heap, (-s, nid, c))
    out = np.full(len(lab), -1, int)
    for k, c in enumerate(sorted(active)):
        out[groups[c]] = k
    if verbose:
        _log(f"merge-back: {len(u):,} → {len(active):,} clusters  ({nmerge:,} merged · {nrej:,} gated)")
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


def _residual_refine(lab, X, min_group, margin=0.02, max_depth=4, verbose=False):
    """Final cleanup: split any over-merged fiber on the residual to its single
    shared d(r) (fiber_split), accepting a split only when it lowers held-out
    residual energy beyond a random split of the same cluster.  Returns
    (labels, n_clusters_split)."""
    out = lab.copy()
    nxt = (int(lab.max()) + 1) if (lab >= 0).any() else 0
    nsplit = 0
    for c in sorted({int(x) for x in lab if x >= 0}):
        idx = np.where(lab == c)[0]
        if len(idx) < 2 * min_group:
            continue
        sub = fsp.recursive_split(X[idx], min_n=min_group, max_depth=max_depth, margin=margin)
        k = int(sub.max()) + 1
        if k <= 1:
            continue
        for ch in range(1, k):                            # child 0 keeps id c; rest get new ids
            out[idx[sub == ch]] = nxt; nxt += 1
        nsplit += 1
    if verbose and nsplit:
        _log(f"residual-split: {nsplit:,} cluster(s) refined → {nxt:,} fibers")
    return out, nsplit


def _tmpl_scores(WB, tmpl, mask, ml=4):
    """Per-spike best-lag SHAPE correlation of waveforms WB (n,T,C) to a single template (T,C),
    restricted to the high-signal `mask` samples and vectorised over spikes.  MEAN-SUBTRACTED
    (Pearson), amplitude-normalised, lag-tolerant (±ml).  The mask is ESSENTIAL: a single stderiv
    spike over the full extraction window is noise-dominated and a spike barely out-scores a
    foreign template there (own ~0.57); on the peak/high-signal window it separates cleanly.  The
    template is rolled on the FULL window THEN masked so the ±ml shift stays a true time lag."""
    n = len(WB)
    Wm = WB[:, mask, :].reshape(n, -1); Wm = Wm - Wm.mean(1, keepdims=True)
    Wm = Wm / (np.linalg.norm(Wm, axis=1, keepdims=True) + 1e-9)
    best = np.full(n, -1.0)
    for L in range(-ml, ml + 1):
        t = np.roll(tmpl, L, axis=0)[mask].ravel(); t = t - t.mean()
        t = t / (np.linalg.norm(t) + 1e-9)
        best = np.maximum(best, Wm @ t)
    return best


def _ab_reclaim(lab, waves, res, ctx, *, distinct=0.93, abs_thr=0.50, margin=0.05,
                min_reclaim=10, pair_lo=0.5, clean_band=0.30, band_tol=0.30, sigcap=2000, jobs=1, verbose=True):
    """Targeted A/B contamination reclaim -- the curator's "merge a contaminated cluster onto a
    clean, DISTINCT neighbour and re-sort, and the clean cell pulls its own spikes back" made into
    a pass.  For each host cluster B and each clean donor A (whose template is at least loosely
    similar to B's, corr >= pair_lo, else nothing of A's could be hiding in B), a spike of B is
    moved into A iff, realigned, its shape matches A's template by >= abs_thr AND beats its match
    to B by >= margin.  The move is then gated TWICE: (1) the host RESIDUAL (B minus the matched
    spikes) must stay DISTINCT from A (corr < distinct) -- if removing the matched spikes collapses
    B onto A, the two were ONE cell and this is a merge (merge_back's job), not a reclaim.  Crucially
    distinctness is judged on the residual, NOT B's contaminated median, which the foreign spikes
    would pull toward A and so defeat the gate.  (2) A's [floor,window) refractory must stay clean
    afterwards (<= max(A's current, band_tol)) -- reclaimed spikes that don't form a refractory-
    consistent train with A are not A's.  Per-spike commit (not a wholesale 2-way re-partition)
    bounds the false-grab of B's own spikes.

    This contamination is INVISIBLE to B's refractory (a non-co-firing foreign cell leaves B's ISI
    untouched -- the 403 case), which is why the gate is shape + distinctness, not B's own band.
    Validated on g5 ground-truth injection (refine_latest, inject A-spikes into B): ~100% recall of
    the foreign spikes at template corr < 0.93, degrading into the continuum above it -- so the
    distinctness gate is load-bearing, and proper per-subcluster realignment (here _med / align_xcorr
    to own median, NOT a joint median) is what makes the reclaim work at all.  Reclaim-only into
    EXISTING donors -- never spawns clusters -- so it cannot over-fragment.  Default OFF.  Returns lab."""
    lab = lab.copy()
    cl = list(np.unique(lab[lab >= 0]))
    if len(cl) < 2:
        return lab
    idxof = {c: np.flatnonzero(lab == c) for c in cl}
    _rng = np.random.default_rng(0)
    def _cap(idx):                                              # estimate a cluster's median TEMPLATE from a sample,
        if sigcap and len(idx) > sigcap:                        #   not its full membership: the iterated align in _med
            return _rng.choice(idx, sigcap, replace=False)      #   is the bottleneck on whole-session merged clusters
        return idx                                              #   (tens of thousands of spikes); a ~2000-spike sample
    capidx = {c: _cap(idxof[c]) for c in cl}                    # sample sequentially (deterministic _rng order) so the
    def _tpl(c):                                                #   template is reproducible and INDEPENDENT of worker count;
        return c, _med(capidx[c], waves), band_pct(res[idxof[c]], ctx)   # the parallel work below is then pure.
    if jobs and jobs > 1 and len(cl) > 2:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=jobs) as _ex:      # _med is numpy/BLAS-heavy (releases the GIL), so threads
            _tpls = list(_ex.map(_tpl, cl))                     #   parallelise the per-cluster align with no pickling and
    else:                                                       #   waves stays shared -- the precompute is the bottleneck on
        _tpls = [_tpl(c) for c in cl]                           #   whole-session sorts (clusters span chunks, so the axis is
    med = {c: m for c, m, _b in _tpls}                          #   per-CLUSTER, not per-chunk).  Result is jobs-invariant.
    band = {c: _b for c, m, _b in _tpls}
    moved = pairs = 0
    for B in cl:
        bidx = idxof[B]
        if len(bidx) < 2 * min_reclaim:
            continue
        # score the RAW (extractor peak-aligned) spikes against each candidate template -- the lag
        # search inside _tmpl_scores aligns every spike to the template it is tested against, so a
        # B-spike is judged in the DONOR's frame, not pre-warped to B's median (which would pull a
        # foreign A-spike toward B and hide it).
        WBraw = waves[bidx]
        sB = _tmpl_scores(WBraw, med[B], ctx.mask, ml=6)
        for A in cl:
            if A == B or band[A] > clean_band:                  # donor must be refractory-clean
                continue
            tc = _ncorr(med[A], med[B])
            if tc < pair_lo:                                    # unrelated shape -> nothing of A's hides in B
                continue
            sA = _tmpl_scores(WBraw, med[A], ctx.mask, ml=6)
            take = (sA >= abs_thr) & (sA >= sB + margin)
            nt = int(take.sum())
            if nt < min_reclaim or len(bidx) - nt < 8:          # need a residual host to verify distinctness
                continue
            cand = bidx[take]
            resid = bidx[~take]
            if _ncorr(med[A], _med(_cap(resid), waves)) >= distinct:  # host RESIDUAL == donor -> one cell -> merge_back's job
                continue                                        # (measured on the residual, NOT the contaminated host
            if band_pct(res[np.concatenate([idxof[A], cand])], ctx) > max(band[A], band_tol):
                continue                                        #  median -- the contamination would inflate that corr)
            lab[cand] = A; moved += len(cand); pairs += 1
            idxof[A] = np.concatenate([idxof[A], cand])
            med[A] = _med(_cap(idxof[A]), waves); band[A] = band_pct(res[idxof[A]], ctx)
            keep = ~take; bidx = bidx[keep]; idxof[B] = bidx; WBraw = WBraw[keep]; sB = sB[keep]
            if len(bidx) < 2 * min_reclaim:
                break
            med[B] = _med(_cap(bidx), waves); sB = _tmpl_scores(WBraw, med[B], ctx.mask, ml=6)
    if verbose and moved:
        _log(f"a/b reclaim · {moved} foreign spike(s) -> clean donor across {pairs} pair(s)")
    return lab


def refine(waves, res_abs, W, nmean, mask, sr, *,
           floor=16, window_ms=2.0, iters=4, large=800, min_group=40,
           var_margin=0.05, brr_tol=0.30, var_peak=2.0, var_depth=4, split_min_corr=0.93, split_var_mult=0.0,
           knn_k=20, knn_thr=0.3, knn_minref=50, knn_minnew=30,
           knn_dims=16, fold_thr=0.9, fold_off_thr=None, init_labels=None,
           conv_tol=0.0, conv_patience=2, reseed=0,
           merge_back_enable=True, merge_budget=1.0, merge_min_sim=0.92, merge_warp_thr=None,
           merge_warp_recall=None, merge_warp_resid_thr=None, merge_amp_thr=0.7,
           merge_warp_resid_thr_int=None, merge_warp_resid_thr_pyr=None, raw_waves=None,
           merge_mode="normalized", fine_method="gmm", coarse_mg=150,
           residual_split=True, residual_margin=0.02,
           ab_reclaim=False, ab_distinct=0.93, ab_abs=0.50, ab_margin=0.05, ab_min=10, ab_sigcap=2000, ab_jobs=1,
           dip_realign=True, rkk_realign=True, rkk_iters=2, dip_first=True, ccg_refrac_ms=0.0,
           rkk_delete=True, drop_min=None,
           nudge_split=False, nudge_max=3, nudge_amp_pct=40.0, nudge_min_ch=4, nudge_alpha=0.01,
           snaps_out=None, verbose=True, basis=None):
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
    ctx = Ctx(W, nmean, mask, sr, int(floor), window, basis)
    drop_eff = min_group if drop_min is None else int(drop_min)   # cluster-KEEP floor, decoupled
                                                                  # from min_group (the split-PIECE floor)
    if init_labels is None:
        lab, _ = fs.cluster_chunk_fine(waves, res_abs, W, nmean, coarse_mg, mask, sr,
                                       method=fine_method, var_split=0.0, basis=basis)
    else:
        lab = np.asarray(init_labels, int).copy()
    stats = [_iter_stats("fine", lab, waves, res_abs, ctx)]
    if snaps_out is not None:
        snaps_out.append(("fine", lab.copy()))
    if verbose:
        _log(f"contamination window [{floor/sr*1000:.2f}, {window_ms:.2f}] ms  ·  "
             f"split_min_corr={split_min_corr} · reseed={reseed}")
        print(_HDR); print(_row(stats[-1]))
    npass = max(1, reseed + 1)
    prev_pass = None
    for p in range(npass):
        isol = np.zeros(len(lab), bool)               # fresh isolation view per (re)seed
        stable = 0
        for it in range(iters):
            lab, isol, nr, nd, ni = _split_all(lab, isol, waves, res_abs, ctx,
                                               large, min_group, var_margin, brr_tol,
                                               var_peak, var_depth, split_min_corr,
                                               dip_realign, rkk_realign, rkk_iters,
                                               var_mult=split_var_mult, dip_first=dip_first, ccg_refrac_ms=ccg_refrac_ms,
                                               rkk_delete=rkk_delete,
                                               nudge=nudge_split, nudge_max=nudge_max,
                                               nudge_amp_pct=nudge_amp_pct,
                                               nudge_min_ch=nudge_min_ch, nudge_alpha=nudge_alpha)
            F = _gfeat(waves, ctx, knn_dims)
            lab, fo, ke = _knn_apply(lab, F, waves, res_abs, ctx,
                                     knn_k, knn_thr, knn_minref, knn_minnew, fold_thr, split_min_corr, off_thr=fold_off_thr)
            lab = _drop_tiny(lab, drop_eff)
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
                        _log(f"split phase converged at iter {it + 1}")
                    break
        if merge_back_enable:
            lab = merge_back(lab, waves, res_abs, ctx, budget=merge_budget,
                             min_sim=merge_min_sim, mode=merge_mode, warp_thr=merge_warp_thr,
                             warp_recall=merge_warp_recall, warp_resid_thr=merge_warp_resid_thr,
                             warp_resid_thr_int=merge_warp_resid_thr_int,
                             warp_resid_thr_pyr=merge_warp_resid_thr_pyr, raw_waves=raw_waves, sr=sr,
                             amp_thr=merge_amp_thr, verbose=verbose)
            st = _iter_stats(f"{p+1}.merge" if reseed else "merge", lab, waves, res_abs, ctx)
            stats.append(st)
            if snaps_out is not None:
                snaps_out.append((f"{p+1}.merge" if reseed else "merge", lab.copy()))
            if verbose:
                print(_row(st))
        if ab_reclaim:
            lab = _ab_reclaim(lab, waves, res_abs, ctx, distinct=ab_distinct, abs_thr=ab_abs,
                              margin=ab_margin, min_reclaim=ab_min, sigcap=ab_sigcap, jobs=ab_jobs, verbose=verbose)
            st = _iter_stats(f"{p+1}.abrcl" if reseed else "abrcl", lab, waves, res_abs, ctx)
            stats.append(st)
            if snaps_out is not None:
                snaps_out.append((f"{p+1}.abrcl" if reseed else "abrcl", lab.copy()))
            if verbose:
                print(_row(st))
        if reseed:
            # fit new fibers from the merged clusters and REASSIGN every spike by
            # whiteness residual (membership cleanup), then loop back to splitting.
            lab = _refit_reassign(lab, waves, W, nmean, mask, drop_eff)
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
                    _log(f"reseed converged at pass {p + 1}")
                break
            prev_pass = cur
    if residual_split:                                    # opt-out final cleanup: clean clusters
        Xw = (fl.realign(waves)[:, mask, :].reshape(len(waves), -1) - nmean) @ W
        lab, nsp = _residual_refine(lab, Xw, min_group, margin=residual_margin, verbose=verbose)
        st = _iter_stats("rsplit", lab, waves, res_abs, ctx); st["rsplit"] = nsp
        stats.append(st)
        if snaps_out is not None:
            snaps_out.append(("rsplit", lab.copy()))
        if verbose:
            print(_row(st))
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


def _chunk_fiber_features(ext_idx, ext_lab, waves, mask, chpos=None):
    """Per (chunk, localid) features for the continuity fallback.  Mirrors the
    `_chunk_bundles` convention: waves are (nspike, nsamp, nchan); realign, then
    keep the informative samples via `mask`.  Yields:
      depth -- energy-weighted centroid over CHANNELS (per-channel energy summed
               over the masked samples); a geometry-free drift coordinate (uses
               chpos if given, else channel index), and
      sig   -- the masked mean template flattened (drift-robust identity
               signature for the cosine gate).
    Returns (depth: {(c,l): float}, sig: {(c,l): vector})."""
    depth, sig = {}, {}
    for c in range(len(ext_idx)):
        if len(ext_idx[c]) == 0:
            continue
        for l in np.unique(ext_lab[c][ext_lab[c] >= 0]):
            sel = ext_idx[c][ext_lab[c] == l]
            tmpl = fl.realign(waves[sel])[:, mask, :].mean(0)    # (n_masked_samp, nchan)
            e = (tmpl ** 2).sum(0)                               # per-channel energy
            pos = np.asarray(chpos, float) if chpos is not None else np.arange(tmpl.shape[1], dtype=float)
            depth[(c, int(l))] = float((pos * e).sum() / (e.sum() + 1e-12))
            sig[(c, int(l))] = tmpl.ravel()
    return depth, sig


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


_RCTX = {}


def _init_refine_pool_worker(cfg):
    """Pool initializer for a POOL WORKER: pin BLAS to one thread, then the usual
    context.  Separate from _init_refine_worker, which also runs the serial path --
    a single process should keep its threads."""
    from .fiber_session import _limit_worker_threads
    _limit_worker_threads()
    _init_refine_worker(cfg)


def _init_refine_worker(cfg):
    """Pool initializer (also run for the serial path): stash the static config and open the
    .spkD / .fil memmaps once per worker process.  Mirrors fiber_session._init_chunk_worker."""
    _RCTX.clear(); _RCTX.update(cfg)
    _RCTX["spk"], _ = fs.open_spkD(cfg["base"], cfg["elec"], cfg["nsamp"], cfg["nchan"])
    _RCTX["filmm"] = nio.open_signal(f'{cfg["base"]}.fil', cfg["ntotal"])
    if cfg.get("need_raw"):                     # standard .spk for the cell-type-aware warp-resid gate
        _RCTX["raw"], _ = nio.open_spk(cfg["base"], cfg["elec"], cfg["nsamp"], cfg["nchan"], prefer=["standard"])


def _refine_one_chunk(task):
    """Refine ONE chunk -- task = (c, ext, res_e, init_e); returns (c, ext, labc, Wc, nmc).
    Reads the memmaps + params from _RCTX.  Each chunk is independent (its own whitener + refine,
    written to its own slot), so this runs unchanged whether dispatched serially or across a
    ProcessPool.  Workers read .spkD from disk -- apply_spike_keep keeps it row-aligned with the
    in-memory waves even after a dedup, so a worker slice equals waves[ext]."""
    c, ext, res_e, init_e = task
    ctx = _RCTX
    waves_e = np.asarray(ctx["spk"][ext], dtype=float)
    raw_e = np.asarray(ctx["raw"][ext], dtype=float) if "raw" in ctx else None
    s0 = int(res_e.min()) - ctx["nsamp"]; s1 = int(res_e.max()) + ctx["nsamp"] + 1
    Wc, nmc, _ = fs.fil_chunk_whitener(ctx["filmm"], ctx["gch"], s0, s1, res_e, ctx["nsamp"], ctx["mask"])
    labc, _ = refine(waves_e, res_e, Wc, nmc, ctx["mask"], ctx["sr"],
                     init_labels=init_e, min_group=ctx["min_group"], raw_waves=raw_e, verbose=False, **ctx["refine_kw"])
    return c, ext, labc, Wc, nmc


def refine_chunked(waves, res, base, elec, ntotal, nsamp, nchan, gch, mask, sr,
                   chunk_min, overlap_min, *, init=None, refine_kw=None,
                   min_group=40, track_geometry=False, make_bundles=False,
                   strict_link=True, link_min_anchor=20,
                   link_continuity=False, continuity_kw=None, chpos=None, jobs=1, verbose=True):
    """Drift-aware refine: window the session, fit a SEPARATE whitener + run the
    full refine loop INSIDE each window (so each window is quasi-stationary),
    then link per-window fibers by overlap-anchor (fs.link_chunks: same physical
    spikes in adjacent windows' overlap prove identity).  Final per-spike label
    comes from each spike's CORE window.  Returns (global_labels, n_global,
    chunk_tracks|None, bundles|None).  This is the correct way to run on a long
    drifting session -- pooling the whole session into one trajectory smears it."""
    refine_kw = dict(refine_kw or {})
    refine_kw.pop("min_group", None)        # passed explicitly below; avoid double-keyword to refine()
    chunks, nchunks = _chunk_bounds(res, sr, chunk_min, overlap_min)
    ext_idx = [np.array([], int)] * nchunks
    ext_lab = [np.array([], int)] * nchunks
    chunk_W = {}; chunk_nm = {}
    tmin_of = {ck["c"]: ck["tmin"] for ck in chunks}
    ncore_of = {ck["c"]: len(ck["core"]) for ck in chunks}

    def _store(result):
        c, ext, labc, Wc, nmc = result
        ext_idx[c] = ext; ext_lab[c] = labc; chunk_W[c] = Wc; chunk_nm[c] = nmc
        if verbose:
            print(f"{_IND}chunk {c+1:>3}/{nchunks}  t={tmin_of[c]:>5.1f}m   {ncore_of[c]:>7,} core   →  {len(np.unique(labc[labc >= 0])):>4} fibers")

    tasks = []                                                  # one task per chunk big enough to refine
    for ck in chunks:
        c, ext = ck["c"], ck["ext"]
        if len(ext) < 2 * min_group:
            if verbose:
                print(f"{_IND}chunk {c+1:>3}/{nchunks}   {ncore_of[c]:>7,} core   →  skipped (small)")
            continue
        tasks.append((c, ext, res[ext], (init[ext] if init is not None else None)))

    need_raw = (refine_kw.get("merge_warp_resid_thr_int") is not None
                or refine_kw.get("merge_warp_resid_thr_pyr") is not None)
    cfg = dict(base=base, elec=elec, ntotal=ntotal, nsamp=nsamp, nchan=nchan, gch=gch,
               mask=mask, sr=sr, min_group=min_group, refine_kw=refine_kw, need_raw=need_raw)
    jobs = max(1, int(jobs))
    if jobs == 1 or len(tasks) <= 1:                            # chunks are independent; serial == the former inline loop
        _init_refine_worker(cfg)
        for task in tasks:
            _store(_refine_one_chunk(task))
    else:
        from concurrent.futures import ProcessPoolExecutor
        nworkers = min(jobs, len(tasks))
        if verbose:
            _log(f"refining {len(tasks)} chunks on {nworkers} processes")
        with ProcessPoolExecutor(max_workers=nworkers,
                                 initializer=_init_refine_pool_worker, initargs=(cfg,)) as ex:
            for result in ex.map(_refine_one_chunk, tasks):     # map preserves task order -> ordered logs
                _store(result)
    if strict_link:                                        # geometry+timing veto blocks chaining
        gid, nglob = fg.link_chunks_strict(ext_idx, ext_lab, waves, mask, min_anchor=link_min_anchor)
    else:
        gid, nglob = fs.link_chunks(ext_idx, ext_lab)
    if link_continuity:
        depth, sig = _chunk_fiber_features(ext_idx, ext_lab, waves, mask, chpos)
        ckw = dict(continuity_kw or {})
        gid, ng2 = fs.link_continuity(gid, nglob, depth, sig, **ckw)
        if verbose:
            _log(f"continuity fallback: {nglob:,} → {ng2:,} global fibers (drift-predicted, signature-gated)")
        nglob = ng2
    glab = np.full(len(res), -1, int)                       # final label by CORE window
    for ck in chunks:
        c, core = ck["c"], ck["core"]
        if len(ext_idx[c]) == 0 or len(core) == 0:
            continue
        ii = np.searchsorted(ext_idx[c], core)              # ext_idx sorted, core subset of ext
        labs = ext_lab[c][ii]
        glab[core] = [gid.get((c, int(l)), -1) if l >= 0 else -1 for l in labs]
    if verbose:
        _log(f"{nglob:,} global fibers across {nchunks} windows (overlap-anchor linked)")
        _det("assigned", f"{int((glab >= 0).sum()):,}/{len(glab):,} spikes")
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
def emit_identity_hierarchy(base, group, ids, *, variant, tag):
    """Process refine's written flat .clu (per-spike `ids`) with the microfiberize IDENTITY
    lift, emitting the matching .clc (atom layer = the clu) and .clp (identity child->parent)
    so the refine output is a full .clu/.clc/.clp triple that intrachunk/link can consume as a
    hierarchy.  Not tracked during refine; derived from the final labelling here.  renumber=False
    keeps the ids exactly as written, so the triple stays consistent with the .clu (the .clu
    itself is left untouched).  Returns (clc_path, clp_path)."""
    try:
        from .fiber_microfiberize import lift_identity
        from .fiber_refiberize import FiberHierarchy
    except ImportError:
        from fiber_microfiberize import lift_identity
        from fiber_refiberize import FiberHierarchy
    child, parent = lift_identity(np.asarray(ids))
    _, clc, clp = FiberHierarchy(child, parent).refiberize(renumber=False)
    cp = nio.session_path(base, "clc", group, variant=variant, tag=tag)
    pp = nio.session_path(base, "clp", group, variant=variant, tag=tag)
    nio.write_clu_file(cp, clc); nio.write_clu_file(pp, clp)
    return cp, pp


def main():
    ap = argparse.ArgumentParser(
        prog="fiber-refine",
        description="Dedup at the imposed refractory, then iteratively split/peel a "
                    "fine sort into clean units; writes a refined .clu (+ deduped .res).")
    sy.add_session_args(ap)
    ap.add_argument("--in-clu", default=None,
                    help="path to the input sort to refine; "
                         "default = canonical .clu if present, else a fresh fine sort")
    ap.add_argument("--out-method", default="stderiv",
                    help="feature space written BEFORE the group (standard|stderiv|...); "
                         "refine operates in stderiv space, so default stderiv")
    ap.add_argument("--no-cluster-basis", action="store_true",
                    help="ignore the global .pca basis for the split-stage shape features and use a "
                         "per-call local SVD (legacy behaviour)")
    ap.add_argument("--out-stage", dest="out_stage", default="refine",
                    help="fiber STAGE written AFTER the group, e.g. 'refine' -> "
                         "<base>.clu.<out-method>.<group>.refine  ('' for none). "
                         "(was --out-variant; renamed so --out-variant is free for the "
                         "feature variant, as in fiber-realign)")
    ap.add_argument("--emit-hierarchy", dest="emit_hierarchy", action=argparse.BooleanOptionalAction, default=True,
                    help="after writing the .clu, also emit the .clc/.clp microfiber triple via the "
                         "fiber-microfiberize identity lift (each fiber = one atom) so the refine output is "
                         "a hierarchy intrachunk/link can consume; --no-emit-hierarchy writes the flat .clu only")
    ap.add_argument("--refr-floor", type=int, default=None,
                    help="imposed detection refractory (samples); default = from yaml")
    ap.add_argument("--refr-window-ms", type=float, default=2.0,
                    help="biological/ISI-violation window upper bound (ms); contamination is [floor, window)")
    ap.add_argument("--dedup", action=argparse.BooleanOptionalAction, default=False,
                    help="run the sub-floor dedup pass (drops near-coincident sub-threshold duplicate "
                         "detections, ~200-300 spikes).  OFF by default: dedup re-indexes the spike list so "
                         "a deduped clu no longer aligns 1:1 with the canonical .res, which breaks round-"
                         "tripping of curated clu files.  --no-dedup is the explicit off; --dedup re-enables.")
    ap.add_argument("--no-residual-split", dest="residual_split", action="store_false",
                    help="disable the final residual-split cleanup (on by default): each output "
                         "fiber is split on the residual to its shared d(r) when that lowers "
                         "held-out residual energy beyond a random split of the same cluster")
    ap.set_defaults(residual_split=True)
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
    ap.add_argument("--merge-warp-thr", type=float, default=None,
                    help="final-merge WARP gate (Omlor-Giese group delay): require the cross-channel "
                         "correlation of the two clusters' median-template group-delay profiles >= this. "
                         "On clean well-populated clusters the warp is a very clean signature (g5: same "
                         "neuron ~0.99, different ~0; it morphs continuously with drift, adjacent-bin change "
                         "~0.004), so ~0.9 is safe -- lower --merge-min-sim and set this to recover the last "
                         "merges without false ones. None (default) = off.")
    ap.add_argument("--merge-warp-recall", type=float, default=None,
                    help="DRIFT-FRAGMENT recall path for merge-back: also admit a merge when the median-template "
                         "group-delay (WARP) correlation >= this AND the per-channel amplitude-profile correlation "
                         ">= --merge-amp-thr, EVEN IF cosine is low. Drift changes waveform shape (dropping cosine) "
                         "but preserves the per-channel delay+amplitude structure, so this recovers same-neuron "
                         "fragments cosine misses (g5 over-clusters: warp>=0.9 & amp>=0.7 recovered 323/395 such "
                         "merges at >=0.976 precision). Refractory budget remains the final gate. None (default) = off; ~0.9.")
    ap.add_argument("--merge-amp-thr", type=float, default=0.7,
                    help="amplitude-profile correlation floor for the --merge-warp-recall path (Omlor-Giese magnitude term)")
    ap.add_argument("--merge-warp-resid-thr", type=float, default=None,
                    help="single-channel warp-incongruity SUB-gate layered on --merge-warp-thr / --merge-warp-recall: "
                         "among pairs whose overall group-delay (warp) correlation is already coherent (>=0.85), veto "
                         "the merge if any ONE channel's group-delay residual (robust Theil-Sen per-channel delay line) "
                         "exceeds this many samples. warp_correlation is a cross-channel Pearson, so a couple of strong "
                         "channels can hold it high while one channel betrays a different co-located source; this catches "
                         "that (especially on the low-cosine warp-recall admits). g5: vetoes ~9%% of warp-coherent merge-"
                         "admissible pairs at ~1.0. None (default) = off.")
    ap.add_argument("--merge-warp-resid-thr-int", type=float, default=None,
                    help="cell-type-aware warp-incongruity SUB-gate: threshold for pairs touching an INTERNEURON "
                         "(narrow trough-to-peak). Interneurons fire fast so their per-channel timing is very stable "
                         "(offset RMS ~0.23), making a single-channel group-delay residual a real different-source "
                         "signature -> tighter than pyr (~0.7). Needs raw .spk for cell-typing. Set BOTH _INT and _PYR "
                         "to enable the dual gate (each falls back to --merge-warp-resid-thr if only one is set).")
    ap.add_argument("--merge-warp-resid-thr-pyr", type=float, default=None,
                    help="cell-type-aware warp-incongruity SUB-gate: threshold for PYRAMIDAL pairs (wide trough-to-peak). "
                         "Pyramidal timing jitters more (offset RMS ~0.72), so a larger single-channel residual is benign "
                         "(~1.3). A pair touching an interneuron uses the stricter _INT threshold.")
    ap.add_argument("--merge-mode", choices=["normalized", "amplitude"], default="normalized",
                    help="normalized = merge energy levels (neuron count); amplitude = keep them")
    ap.add_argument("--split-min-corr", type=float, default=0.93,
                    help="shape-distinctness gate: do NOT carve off a split piece / energy bucket whose "
                         "normalised median waveform correlates >= this with its parent (stops over-fragmenting "
                         "high-rate units into energy-level clones); 1.0 disables")
    ap.add_argument("--fold-off-thr", type=float, default=None,
                    help="inter-channel TIMING veto on the knn contaminant-fold: when set, a small group is NOT "
                         "folded into an amplitude/shape-similar target if their robust (raw xcorr-lag) inter-channel "
                         "offset profiles differ by more than this many samples -- protects small but timing-distinct "
                         "cells (e.g. 294 vs 295) from being shattered into a look-alike. g5 calibration: same-cell "
                         "~0.11, distinct co-located cells ~0.26, so ~0.2-0.25 catches them. None (default) = off.")
    ap.add_argument("--reseed", type=int, default=0,
                    help="re-run the whole loop (split -> merge -> refit fibers -> reassign) using the "
                         "refined labels as the next seed, up to N extra passes (e.g. 1 = 2 passes); 0 = single pass")
    ap.add_argument("--track-geometry", action="store_true",
                    help="record per-fiber geometry (radius/cone/smoothness/bend) at every iteration and "
                         "write <base>.geom.<group>.npz tracking each final fiber back through the loop")
    ap.add_argument("--large", type=int, default=800, help="only clusters >= this are split each iter")
    ap.add_argument("--min-group", type=int, default=40)
    ap.add_argument("--drop-min", type=int, default=None,
                    help="cluster-KEEP floor: clusters smaller than this are dropped to the artifact bin "
                         "each iteration.  Decoupled from --min-group (the split-PIECE floor).  Default "
                         "None = use --min-group (legacy).  Set LOW (e.g. 5) so session's small over-split "
                         "fragments survive to be stitched by intrachunk instead of being discarded.")
    ap.add_argument("--var-margin", type=float, default=0.05,
                    help="min per-channel residual-variance reduction to accept a gated sub-split")
    ap.add_argument("--brr-tol", type=float, default=0.30,
                    help="max allowed increase (pp) in [floor,window) refractory for a gated sub-split")
    ap.add_argument("--var-peak", type=float, default=2.0, help="var-split trigger (max/median channel variance)")
    ap.add_argument("--split-var-mult", type=float, default=0.0,
                    help="curator split-variance gate: only attempt to split a cluster whose top-3 whitened-feature variance exceeds this multiple of the median over splittable clusters. From the g5 curation log, split-source clusters carry ~3.6x the median feature variance (and kurtosis/ISI are NOT the trigger), so ~1.5-2.0 matches the curator; below it clusters pass through unsplit. 0 (default) = off (no variance prefilter).")
    ap.add_argument("--var-depth", type=int, default=4)
    ap.add_argument("--dip-realign", dest="dip_realign", action="store_true", default=True,
                    help="realign each dipsplit node to its own median (per-step; default on)")
    ap.add_argument("--no-dip-realign", dest="dip_realign", action="store_false")
    ap.add_argument("--rkk-realign", dest="rkk_realign", action="store_true", default=True,
                    help="interleave rkk (CEM) with per-cluster realignment (default on)")
    ap.add_argument("--no-rkk-realign", dest="rkk_realign", action="store_false")
    ap.add_argument("--rkk-realign-iters", dest="rkk_iters", type=int, default=2)
    ap.add_argument("--rkk-delete", dest="rkk_delete", action="store_true", default=True,
                    help="rkk (CEM) culls sub-min-group clusters during the split (default on)")
    ap.add_argument("--no-rkk-delete", dest="rkk_delete", action="store_false",
                    help="keep small (non-singular) rkk sub-clusters instead of dissolving them -- "
                         "stops the size-cull from shedding spikes that the cascade then sends to the "
                         "residual/artifact bin. Use when too many spikes land in the artifact cluster.")
    ap.add_argument("--ccg-refrac-ms", dest="ccg_refrac_ms", type=float, default=0.0,
                    help="curation-independent veto: reject a split the refractory cross-correlogram\n"
                         "calls spurious (one neuron) where it has power; 0 = off. ~1.5 to enable. "
                         "Abstains (no effect) at low firing rates.")
    ap.add_argument("--ab-reclaim", dest="ab_reclaim", action=argparse.BooleanOptionalAction, default=False,
                    help="after merge_back, run the targeted A/B contamination reclaim: move a host\n"
                         "cluster's spikes into a clean, DISTINCT donor when their realigned shape matches\n"
                         "the donor (refractory-safe, per-spike, reclaim-only).  Recovers foreign spikes a\n"
                         "non-co-firing contaminant leaves invisible to the refractory metric.  Default off.")
    ap.add_argument("--ab-distinct", dest="ab_distinct", type=float, default=0.93,
                    help="A/B reclaim: max donor/host template shape-corr to treat them as two cells "
                         "(>= this is one cell -> merge_back, not reclaim). Validated knee ~0.93.")
    ap.add_argument("--ab-abs", dest="ab_abs", type=float, default=0.50,
                    help="A/B reclaim: absolute shape-corr a spike must reach to the donor template to move.")
    ap.add_argument("--ab-margin", dest="ab_margin", type=float, default=0.05,
                    help="A/B reclaim: how much better a spike must match the donor than its host to move "
                         "(bounds false-grab of the host's own spikes).")
    ap.add_argument("--ab-min", dest="ab_min", type=int, default=10,
                    help="A/B reclaim: minimum spikes a donor must reclaim from one host to commit the move.")
    ap.add_argument("--ab-sigcap", dest="ab_sigcap", type=int, default=2000,
                    help="A/B reclaim: cap on spikes used to estimate each cluster's median TEMPLATE (not its "
                         "membership/scoring).  A ~2000-spike sample matches the full template to ~1e-3 while "
                         "avoiding the iterated align on whole-session merged clusters (tens of thousands of "
                         "spikes) -- the pass's bottleneck in whole-session mode.  0 = no cap (use all spikes).")
    ap.add_argument("--ab-jobs", dest="ab_jobs", type=int, default=1,
                    help="A/B reclaim: worker threads for the per-cluster template precompute (the align bottleneck). "
                         "Result is identical for any value (templates are sampled before the parallel work). 1 = serial. "
                         "Helps in WHOLE-SESSION mode (big clusters); in chunked mode clusters are small so it barely matters.")
    ap.add_argument("--rkk-first", dest="dip_first", action="store_false", default=True,
                    help="restore the old cascade order (rkk before dip-bisection); default is "
                         "dip-first, which targets the single high-margin dip axis welds separate on")
    ap.add_argument("--nudge-split", dest="nudge_split", action="store_true", default=False,
                    help="split temporally-offset overlaid units in low-amp clusters by alignment lag "
                         "(residual-neutral; for from-scratch/coarse sorts, default off)")
    ap.add_argument("--nudge-max", type=int, default=3)
    ap.add_argument("--nudge-amp-pct", type=float, default=40.0)
    ap.add_argument("--nudge-min-channels", dest="nudge_min_ch", type=int, default=4)
    ap.add_argument("--nudge-alpha", type=float, default=0.01)
    ap.add_argument("--knn-k", type=int, default=20)
    ap.add_argument("--knn-thr", type=float, default=0.3, help="K-NN majority fraction to peel a spike")
    ap.add_argument("--knn-minref", type=int, default=50)
    ap.add_argument("--knn-minnew", type=int, default=30)
    ap.add_argument("--knn-dims", type=int, default=16)
    ap.add_argument("--fold-thr", type=float, default=0.9,
                    help="non-normalised median-xcorr above which a peeled bucket is folded (else kept as new)")
    ap.add_argument("--fine-algo", "--fine-method", dest="fine_method", choices=["gmm", "fiber", "none"], default="gmm",
                    help="method for the initial fine sort when no --in-clu is given")
    ap.add_argument("--chunk-minutes", "--chunk-min", type=float, default=0.0,
                    help="drift-aware mode: window the session into CORE chunks of this many minutes, "
                         "refine each in its own whitened frame, and link fibers across windows by "
                         "overlap-anchor; 0 = single whole-session pass (assumes stationary)")
    ap.add_argument("--chunk-overlap-minutes", type=float, default=1.0,
                    help="overlap between adjacent windows used for overlap-anchor linking (drift-aware mode)")
    ap.add_argument("--chunk-jobs", dest="chunk_jobs", type=int, default=1,
                    help="parallel worker PROCESSES over chunks in drift-aware mode (default 1 = serial). "
                         "Chunks are independent (own whitener + refine), so this is the main speedup for a "
                         "chunked run; the cross-window link runs serially after.  Workers re-open the .spkD/.fil "
                         "memmaps, so memory is bounded.  No effect in whole-session mode (--chunk-minutes 0).")
    ap.add_argument("--bundles", action="store_true",
                    help="drift-aware mode: also write <base>.bundles.<group>.npz (per-chunk un-whitened "
                         "template curves per global fiber) for the fiber-view-gui bundle table")
    ap.add_argument("--legacy-link", action="store_true",
                    help="use the old overlap-only link_chunks (no geometry/timing veto)")
    ap.add_argument("--min-anchor", type=int, default=20,
                    help="min shared overlap spikes to link two per-chunk fibers (strict linker)")
    ap.add_argument("--link-continuity", action="store_true",
                    help="drift-aware mode: after overlap-anchor linking, bridge sparse fibers that share too "
                         "few overlap spikes using a drift-predicted, signature-gated continuity fallback")
    ap.add_argument("--continuity-sig-thr", type=float, default=0.6,
                    help="min template cosine to allow a continuity bridge (signature gate; default 0.6)")
    ap.add_argument("--continuity-depth-gate", type=float, default=14.0,
                    help="max drift-predicted depth error (per chunk of gap) for a continuity bridge")
    ap.add_argument("--continuity-max-gap", type=int, default=2,
                    help="max chunk gap a continuity bridge may span (default 2)")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--feature-align", dest="feature_align", choices=["xcorr", "centroid"], default=None,
                    help="feature-building alignment: xcorr (default) or centroid (pure, no refine -- "
                         "adds the trough-position-vs-asymmetry structure to the clustering/linking "
                         "features).  Does NOT touch committing alignment or fiber-realign.  Overrides "
                         "the FIBER_ALIGN env var.")
    ap.add_argument("--dedup-strict", dest="dedup_strict", action=argparse.BooleanOptionalAction, default=True,
                    help="when a dedup removes spikes, every live per-spike file of the group is subset too. "
                         "A live file at a third row count (neither pre- nor post-dedup) is a stale leftover "
                         "from an earlier extraction that CANNOT be subset by this mask; by default it is "
                         "quarantined aside (.stalebkp) and regenerated. --no-dedup-strict leaves such files "
                         "in place instead; --dedup-stale error restores a hard failure.")
    ap.add_argument("--dedup-stale", choices=("error", "skip", "quarantine"), default=None,
                    help="explicit policy for stale leftover per-spike files at a third row count. Default "
                         "(unset) QUARANTINES them aside as <file>.stalebkp -- non-destructive, leaves the "
                         "group consistent, and the stage regenerates them. error = hard-fail (old strict "
                         "behavior); skip = leave them (= --no-dedup-strict). "
                         "Default follows --dedup-strict.")
    ap.add_argument("--subsample", dest="subsample", action=argparse.BooleanOptionalAction, default=None,
                    help="enable (--subsample) or disable (--no-subsample) realign's per-spike "
                         "sub-sample (parabolic) refine in the feature build; default leaves the "
                         "FIBER_SUBSAMPLE env var / lever untouched (off).  Reaches pool workers.")
    a = ap.parse_args()
    if a.feature_align:
        os.environ["FIBER_ALIGN"] = a.feature_align    # reach any spawned workers
        fl.set_feature_align(a.feature_align)            # this process
    if a.subsample is not None:
        os.environ["FIBER_SUBSAMPLE"] = "1" if a.subsample else "0"    # reach any spawned workers
        fl.set_realign_subsample(a.subsample)                           # this process
    _gpu_line = None
    if a.gpu:
        _on = _bk.use_gpu(True)
        _gpu_line = _bk.backend_name() + ("" if _on else "  (unavailable -> CPU)")

    cfg = sy.resolve_session_params(a.session, a.group, channels=a.channels, ntotal=a.ntotal,
                                    nchan=a.nchan, nsamp=a.nsamp, sr=a.sr)
    base = cfg["base"]; elec = a.group
    ntotal = cfg["ntotal"]; nchan = cfg["nchan"]; nsamp = cfg["nsamp"]; sr = cfg["sr"]
    gch = np.array(cfg["channels"], int)
    mask = fl.build_masks(nsamp, cfg["peak"]).full
    # SHAPE features for the split stages: the GLOBAL ndm_pca basis (shared across chunks);
    # None -> per-call local SVD fallback.  Variant matches the extracted waveforms (out-method).
    cluster_basis = None if a.no_cluster_basis else _fpca.read_cluster_basis(base, elec, a.out_method)

    floor = a.refr_floor
    if floor is None:
        floor, src = sy.refractory_period_samples(a.session, a.group, sr=sr)
        _refr = f"{floor} samples ({floor / sr * 1000:.2f} ms) [{src}]"
    else:
        _refr = f"{floor} samples ({floor / sr * 1000:.2f} ms) [--refr-floor]"

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
    _log(f"group {elec} · {len(res):,} spikes")
    _det("source", spkpath)
    _det("input", f"{len(np.unique(init[init>=0])):,} clusters" if init is not None else "no input .clu → fresh fine sort")
    _det("refractory", _refr)
    _det("feature-align", fl.get_feature_align())
    _det("realign", "sub-sample" if fl.realign_subsample() else "whole-sample")
    if _gpu_line: _det("GPU", _gpu_line)

    if a.dedup and floor > 0:
        n_orig = len(res)
        ptp = np.ptp(waves.reshape(len(waves), -1), axis=1)
        keep = dedup_spikes(res, ptp, floor)
        n_dup = n_orig - len(keep)
        n_exact = int((np.diff(np.sort(res)) == 0).sum())
        res = res[keep]; waves = waves[keep]
        if init is not None:
            init = init[keep]
        _det("dedup", f"kept {len(keep):,} of {n_orig:,}  ({n_dup:,} removed · {n_exact:,} exact ISI=0)")
        if n_dup > 0:
            # a dedup removes physical spikes from the GROUP, so every per-spike file
            # (res/clu/spk/spkD/fet/fetD, across all variants and stages) must drop the
            # same rows -- otherwise any later load hits a spike-count mismatch.
            nio.apply_spike_keep(base, elec, keep, n_orig, nsamp, nchan, strict=a.dedup_strict, stale=a.dedup_stale)

    if a.chunk_minutes and a.chunk_minutes > 0:
        refine_kw = dict(floor=floor, window_ms=a.refr_window_ms, iters=a.iters,
                         large=a.large, min_group=a.min_group, drop_min=a.drop_min, var_margin=a.var_margin,
                         brr_tol=a.brr_tol, var_peak=a.var_peak, var_depth=a.var_depth, split_var_mult=a.split_var_mult,
                         split_min_corr=a.split_min_corr, dip_realign=a.dip_realign,
                         rkk_realign=a.rkk_realign, rkk_iters=a.rkk_iters, dip_first=a.dip_first, ccg_refrac_ms=a.ccg_refrac_ms, rkk_delete=a.rkk_delete, nudge_split=a.nudge_split,
                         nudge_max=a.nudge_max, nudge_amp_pct=a.nudge_amp_pct,
                         nudge_min_ch=a.nudge_min_ch, nudge_alpha=a.nudge_alpha,
                         knn_k=a.knn_k, knn_thr=a.knn_thr,
                         knn_minref=a.knn_minref, knn_minnew=a.knn_minnew, knn_dims=a.knn_dims,
                         fold_thr=a.fold_thr, fold_off_thr=a.fold_off_thr, conv_tol=(a.converge_tol if a.converge else 0.0),
                         conv_patience=a.converge_patience, reseed=a.reseed,
                         merge_back_enable=a.merge_back, merge_budget=a.merge_budget, merge_warp_thr=a.merge_warp_thr,
                         merge_warp_recall=a.merge_warp_recall, merge_warp_resid_thr=a.merge_warp_resid_thr, merge_amp_thr=a.merge_amp_thr,
                         merge_warp_resid_thr_int=a.merge_warp_resid_thr_int, merge_warp_resid_thr_pyr=a.merge_warp_resid_thr_pyr,
                         merge_min_sim=a.merge_min_sim, merge_mode=a.merge_mode,
                         fine_method=a.fine_method, residual_split=a.residual_split,
                         ab_reclaim=a.ab_reclaim, ab_distinct=a.ab_distinct, ab_abs=a.ab_abs,
                         ab_margin=a.ab_margin, ab_min=a.ab_min, ab_sigcap=a.ab_sigcap, ab_jobs=a.ab_jobs, basis=cluster_basis)
        glab, nglob, tracks, bundles = refine_chunked(
            waves, res, base, elec, ntotal, nsamp, nchan, gch, mask, sr,
            a.chunk_minutes, a.chunk_overlap_minutes, init=init, refine_kw=refine_kw,
            min_group=a.min_group, track_geometry=a.track_geometry,
            make_bundles=a.bundles, jobs=a.chunk_jobs,
            strict_link=not a.legacy_link, link_min_anchor=a.min_anchor,
            link_continuity=a.link_continuity,
            continuity_kw=dict(sig_thr=a.continuity_sig_thr, depth_gate=a.continuity_depth_gate,
                               max_gap=a.continuity_max_gap),
            verbose=True)
        ids = np.where(glab < 0, 0, glab + 1).astype(np.int64)
        clu_path = nio.write_clu(base, elec, ids, variant=a.out_method, tag=a.out_stage)
        res_path = nio.write_res(base, elec, res, variant=a.out_method, tag=a.out_stage)
        _log("wrote")
        _det("clu", clu_path); _det("res", res_path)
        if a.emit_hierarchy:
            _cp, _pp = emit_identity_hierarchy(base, elec, ids, variant=a.out_method, tag=a.out_stage)
            _det("clc", _cp); _det("clp", _pp)
        if tracks is not None:
            gpath = write_chunk_geometry(tracks, f"{base}.geomchunk.{elec}.npz")
            _det("geom", f"{gpath}   ({len(tracks):,} fibers across windows)")
        if bundles is not None:
            bpath = write_bundles(bundles, f"{base}.bundles.{elec}.npz")
            _det("bundles", f"{bpath}   ({len(np.unique(bundles[0])):,} bundles) [fiber-view-gui]")
        _log(f"done · {nglob:,} global fibers · {time.time()-t0:.0f}s")
        return

    # whitener from the .fil baseline over the (deduped) spike span
    filmm = nio.open_signal(f"{base}.fil", ntotal)
    s0 = int(res.min()) - nsamp; s1 = int(res.max()) + nsamp + 1
    W, nmean, _ = fs.fil_chunk_whitener(filmm, gch, s0, s1, res, nsamp, mask)

    snaps = [] if a.track_geometry else None
    raw_waves_ws = None
    if a.merge_warp_resid_thr_int is not None or a.merge_warp_resid_thr_pyr is not None:
        _rspk, _ = nio.open_spk(base, elec, nsamp, nchan, prefer=["standard"])   # cell-type-aware warp-resid gate
        raw_waves_ws = np.asarray(_rspk[:], dtype=float)                          # aligned to waves (post-dedup)
    lab, stats = refine(waves, res, W, nmean, mask, sr,
                        floor=floor, window_ms=a.refr_window_ms, iters=a.iters,
                        large=a.large, min_group=a.min_group, drop_min=a.drop_min,
                        var_margin=a.var_margin, brr_tol=a.brr_tol,
                        var_peak=a.var_peak, var_depth=a.var_depth, split_min_corr=a.split_min_corr, split_var_mult=a.split_var_mult,
                        dip_realign=a.dip_realign, rkk_realign=a.rkk_realign, rkk_iters=a.rkk_iters, dip_first=a.dip_first, ccg_refrac_ms=a.ccg_refrac_ms, rkk_delete=a.rkk_delete,
                        nudge_split=a.nudge_split, nudge_max=a.nudge_max, nudge_amp_pct=a.nudge_amp_pct,
                        nudge_min_ch=a.nudge_min_ch, nudge_alpha=a.nudge_alpha,
                        knn_k=a.knn_k, knn_thr=a.knn_thr, knn_minref=a.knn_minref,
                        knn_minnew=a.knn_minnew, knn_dims=a.knn_dims,
                        fold_thr=a.fold_thr, fold_off_thr=a.fold_off_thr, init_labels=init,
                        conv_tol=(a.converge_tol if a.converge else 0.0),
                        conv_patience=a.converge_patience, reseed=a.reseed,
                        merge_back_enable=a.merge_back, merge_budget=a.merge_budget,
                        merge_min_sim=a.merge_min_sim, merge_mode=a.merge_mode,
                        merge_warp_thr=a.merge_warp_thr, merge_warp_recall=a.merge_warp_recall,
                        merge_warp_resid_thr=a.merge_warp_resid_thr, merge_amp_thr=a.merge_amp_thr,
                        merge_warp_resid_thr_int=a.merge_warp_resid_thr_int,
                        merge_warp_resid_thr_pyr=a.merge_warp_resid_thr_pyr, raw_waves=raw_waves_ws,
                        fine_method=a.fine_method, residual_split=a.residual_split,
                        ab_reclaim=a.ab_reclaim, ab_distinct=a.ab_distinct, ab_abs=a.ab_abs,
                        ab_margin=a.ab_margin, ab_min=a.ab_min, ab_sigcap=a.ab_sigcap, ab_jobs=a.ab_jobs,
                        snaps_out=snaps, verbose=True, basis=cluster_basis)

    ids = np.where(lab < 0, 0, lab + 1).astype(np.int64)   # 0 = noise, clusters 1..K
    clu_path = nio.write_clu(base, elec, ids, variant=a.out_method, tag=a.out_stage)
    res_path = nio.write_res(base, elec, res, variant=a.out_method, tag=a.out_stage)
    tsv = f"{base}.refine.{elec}.tsv"
    with open(tsv, "w") as f:
        f.write("iter\tnfib\tmedBand\tpct<2\tswBand\tswDup\tenCV\tnbig\trkk\tdip\tiso\tfold\tkept\n")
        for s in stats:
            f.write(f"{s['it']}\t{s['nfib']}\t{s['medBand']:.3f}\t{s['pct2']:.1f}\t"
                    f"{s['swBand']:.3f}\t{s['swDup']:.3f}\t{s['enCV']:.4f}\t{s['nbig']}\t"
                    f"{s['rkk']}\t{s['dip']}\t{s['iso']}\t{s['fold']}\t{s['kept']}\n")
    _log("wrote")
    _det("clu", clu_path); _det("res", res_path); _det("stats", tsv)
    if a.emit_hierarchy:
        _cp, _pp = emit_identity_hierarchy(base, elec, ids, variant=a.out_method, tag=a.out_stage)
        _det("clc", _cp); _det("clp", _pp)
    if snaps is not None:
        tracks = geometry_tracks(snaps, waves, W, nmean, mask)
        gpath = write_geometry_tracks(tracks, f"{base}.geom.{elec}.npz")
        _det("geom", f"{gpath}   ({len(tracks):,} fibers × {len(snaps)} snapshots)")
    _log(f"done · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
