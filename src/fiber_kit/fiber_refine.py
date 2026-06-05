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
def _gated_partition(si, sub, pv, pb, waves, res, ctx, mg, vmargin, btol):
    """Accept the sub-labels `sub` over cluster `si` only where a piece lowers the
    per-channel residual variance by >= vmargin AND keeps the [floor,window)
    refractory within btol of the parent; everything else falls into a residual
    core.  Returns a list of >=2 index arrays, or None if nothing qualified."""
    keep, fail = [], []
    for s in np.unique(sub):
        pc = si[sub == s]
        if len(pc) < mg:
            fail.append(pc); continue
        if _pcv(waves[pc], ctx) < pv * (1.0 - vmargin) and band_pct(res[pc], ctx) <= pb + btol:
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


def _gated_split(si, waves, res, ctx, mg, vmargin, btol):
    """Try rkk, then dipsplit, gating each; isolate if neither cleans it."""
    pv = _pcv(waves[si], ctx)
    pb = band_pct(res[si], ctx)
    sub = _rkk(_feats(waves[si], ctx, 6), max_clusters=12, min_size=mg, seed=42)
    fp = _gated_partition(si, sub, pv, pb, waves, res, ctx, mg, vmargin, btol)
    if fp is not None:
        return fp, "rkk"
    if fs._HAVE_DIP:
        pcs = fs._dipsplit_rec(_feats(waves[si], ctx, 4), np.arange(len(si)), mg, 0.05)
        if len(pcs) > 1:
            sub = np.zeros(len(si), int)
            for k, p in enumerate(pcs):
                sub[p] = k
            fp = _gated_partition(si, sub, pv, pb, waves, res, ctx, mg, vmargin, btol)
            if fp is not None:
                return fp, "dip"
    return [si], "iso"


def _split_all(lab, isol, waves, res, ctx, large, mg, vmargin, btol, vpeak, vdepth):
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
                fp, how = _gated_split(si, waves, res, ctx, mg, vmargin, btol)
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


def _match(a, b, ml=4):
    """NON-normalised (amplitude-sensitive) best-lag xcorr ratio in [0,1]: high
    only when both shape AND scale match, so energy levels are NOT collapsed."""
    A = a.ravel(); aa = float((A * A).sum()); best = 0.0
    for L in range(-ml, ml + 1):
        B = np.roll(b, L, axis=0).ravel(); num = float((A * B).sum())
        best = max(best, min(num / (aa + 1e-9), num / (float((B * B).sum()) + 1e-9)))
    return best


def _knn_apply(lab, F, waves, res, ctx, K, thr, minref, minnew, hi):
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
        for ww in np.unique(w[w >= 0]):
            bk = s[w == ww]
            if len(bk) < minnew:
                continue
            if ww not in mc:
                mc[ww] = _med(np.flatnonzero(lab == ww), waves)
            if _match(_med(bk, waves), mc[ww]) >= hi:
                new[bk] = ww; fo += 1                  # amplitude-matched -> fold in
            else:
                new[bk] = nid; nid += 1; ke += 1       # distinct energy level -> new cluster
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
def refine(waves, res_abs, W, nmean, mask, sr, *,
           floor=16, window_ms=2.0, iters=4, large=800, min_group=40,
           var_margin=0.05, brr_tol=0.30, var_peak=2.0, var_depth=4,
           knn_k=20, knn_thr=0.3, knn_minref=50, knn_minnew=30,
           knn_dims=16, fold_thr=0.9, init_labels=None,
           fine_method="gmm", coarse_mg=150, verbose=True):
    """Iteratively refine a fine sort.  Returns (labels, stats) where labels is
    0-based (-1 = noise) over `waves`/`res_abs` and stats is the per-iteration
    list of dicts.  `init_labels` (0-based, -1 noise) is refined in place; if
    None, a fine sort is produced first with cluster_chunk_fine."""
    window = int(round(window_ms * sr / 1000.0))
    ctx = Ctx(W, nmean, mask, sr, int(floor), window)
    if init_labels is None:
        lab, _ = fs.cluster_chunk_fine(waves, res_abs, W, nmean, coarse_mg, mask, sr,
                                       method=fine_method, var_split=0.0)
    else:
        lab = np.asarray(init_labels, int).copy()
    isol = np.zeros(len(lab), bool)
    stats = [_iter_stats("fine", lab, waves, res_abs, ctx)]
    if verbose:
        print(f"contamination window = [{floor/sr*1000:.2f}, {window_ms:.2f}] ms "
              f"([{int(floor)}, {window}] samples)")
        print(_HDR); print(_row(stats[-1]))
    for it in range(iters):
        lab, isol, nr, nd, ni = _split_all(lab, isol, waves, res_abs, ctx,
                                           large, min_group, var_margin, brr_tol,
                                           var_peak, var_depth)
        F = _gfeat(waves, ctx, knn_dims)
        lab, fo, ke = _knn_apply(lab, F, waves, res_abs, ctx,
                                 knn_k, knn_thr, knn_minref, knn_minnew, fold_thr)
        lab = _drop_tiny(lab, min_group)
        st = _iter_stats(str(it + 1), lab, waves, res_abs, ctx)
        st.update(rkk=nr, dip=nd, iso=ni, fold=fo, kept=ke)
        stats.append(st)
        if verbose:
            print(_row(st))
    return lab, stats


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
    ap.add_argument("--iters", type=int, default=4)
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

    # whitener from the .fil baseline over the (deduped) spike span
    filmm = nio.open_signal(f"{base}.fil", ntotal)
    s0 = int(res.min()) - nsamp; s1 = int(res.max()) + nsamp + 1
    W, nmean, _ = fs.fil_chunk_whitener(filmm, gch, s0, s1, res, nsamp, mask)

    lab, stats = refine(waves, res, W, nmean, mask, sr,
                        floor=floor, window_ms=a.refr_window_ms, iters=a.iters,
                        large=a.large, min_group=a.min_group,
                        var_margin=a.var_margin, brr_tol=a.brr_tol,
                        var_peak=a.var_peak, var_depth=a.var_depth,
                        knn_k=a.knn_k, knn_thr=a.knn_thr, knn_minref=a.knn_minref,
                        knn_minnew=a.knn_minnew, knn_dims=a.knn_dims,
                        fold_thr=a.fold_thr, init_labels=init,
                        fine_method=a.fine_method, verbose=True)

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
    print(f"wrote {clu_path}\n      {res_path}\n      {tsv}\n[done] t={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
