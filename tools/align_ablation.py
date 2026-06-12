#!/usr/bin/env python3
"""align_ablation.py -- A/B the feature-building alignment (xcorr refine vs PURE centroid) on the
linker's signature separability, to test whether centroid alignment adds separable structure to the
fiber feature space (i.e. widens the sig_thr margin).

It flips fiber_lib.set_feature_align(mode) and, for each mode, rebuilds every fragment's direction
profile with the EXACT signature the linker gates on (fiber_session._sliding_pairs: per-energy-window
mean whitened direction, pairwise weighted-mean cosine), then reports how well that signature tells
same-unit fragment pairs from different-unit pairs.

Ground truth:
  (default)        time-split self-continuity: each refine cluster's spikes are split into two time
                   halves -> two fragments of the SAME true unit.  Needs no curation; runs anywhere.
  --gt-clu PATH    curated merge map: fragments are the refine pieces themselves, and a piece's truth
                   label is the curated unit it majority-maps to.  This is the production validation
                   against your hand-curated merges (within-unit pairs = pieces you merged).

Whitener defaults to identity (isolates the alignment's effect on the raw masked direction profile);
--pooled-zca whitens by the pooled-spike covariance instead.  The centroid-vs-xcorr COMPARISON is the
result; absolute AUC shifts with the whitener but the ordering is what the hypothesis predicts.
"""
import argparse, numpy as np
from fiber_kit import fiber_lib as fl, neuro_io as nio


def _read_clu(path):
    a = np.fromfile(path, dtype="<i4")
    return a[1:]                                   # drop nClusters header (NeuroSuite convention)


def build_fragments(lab, res, sel, gt=None, min_spikes=80, max_units=150, seed=0):
    """Return (fragments[list of spike-index arrays], truth[unit label per fragment])."""
    rng = np.random.default_rng(seed)
    lab_s = lab[sel]
    if gt is not None:
        # production: each refine piece is a fragment; its truth = majority curated unit over its spikes
        frags, truth = [], []
        for u in np.unique(lab_s):
            if u == 0:
                continue
            ii = sel[lab_s == u]
            if len(ii) < min_spikes:
                continue
            g = gt[ii]; g = g[g != 0]
            if len(g) == 0:
                continue
            cu = np.bincount(g).argmax()
            frags.append(ii); truth.append(int(cu))
        order = rng.permutation(len(frags))[:max_units * 4]
        return [frags[i] for i in order], np.array([truth[i] for i in order])
    # default: time-split each cluster into two halves = two fragments of the same unit
    units = [u for u in np.unique(lab_s) if u != 0 and (lab_s == u).sum() >= min_spikes]
    rng.shuffle(units); units = units[:max_units]
    frags, truth = [], []
    for u in units:
        ii = sel[lab_s == u]; ii = ii[np.argsort(res[ii])]; h = len(ii) // 2
        frags += [ii[:h], ii[h:]]; truth += [int(u), int(u)]
    return frags, np.array(truth)


def sig_profiles(frags, spk, mask, n_win=14, min_spikes=10, W=None, nmean=None):
    Xs = []
    for ii in frags:
        w = np.asarray(spk[ii], float)
        al = fl.realign(w)                                  # alignment honours set_feature_align
        X = al[:, mask, :].reshape(len(w), -1)
        X = X - (nmean if nmean is not None else X.mean(0))
        if W is not None:
            X = X @ W
        Xs.append(X)
    K = len(Xs); p = Xs[0].shape[1]
    rs = [np.linalg.norm(X, axis=1) for X in Xs]
    ds = [X / (r[:, None] + 1e-12) for X, r in zip(Xs, rs)]
    allr = np.concatenate(rs)
    edges = np.linspace(np.percentile(allr, 1), np.percentile(allr, 99), n_win + 1)
    prof = np.zeros((K, n_win, p)); cnt = np.zeros((K, n_win)); conc = np.zeros((K, n_win))
    for k in range(K):
        wi = np.clip(np.searchsorted(edges, rs[k]) - 1, 0, n_win - 1)
        for wv in range(n_win):
            m = wi == wv
            if int(m.sum()) >= min_spikes:
                mr = ds[k][m].mean(0); R = np.linalg.norm(mr)
                prof[k, wv] = mr / (R + 1e-12); cnt[k, wv] = int(m.sum()); conc[k, wv] = R
    return prof, cnt, conc


def pair_cos(prof, cnt, conc, min_shared=2):
    K = len(prof); C = np.full((K, K), np.nan)
    for i in range(K):
        for j in range(i + 1, K):
            sh = (cnt[i] > 0) & (cnt[j] > 0)
            if sh.sum() < min_shared:
                continue
            wts = np.minimum(cnt[i], cnt[j])[sh] * conc[i][sh] * conc[j][sh]
            if wts.sum() < 1e-9:
                continue
            C[i, j] = C[j, i] = (wts * (prof[i][sh] * prof[j][sh]).sum(1)).sum() / wts.sum()
    return C


def margin(C, truth):
    K = len(truth); iu = np.triu_indices(K, 1)
    same = truth[iu[0]] == truth[iu[1]]; v = C[iu]
    ok = ~np.isnan(v); v, same = v[ok], same[ok]
    win, cross = v[same], v[~same]
    cs = np.sort(cross)
    auc = float(np.mean([np.searchsorted(cs, x) / len(cs) for x in win])) if len(win) and len(cross) else float("nan")
    return dict(auc=auc, med_same=float(np.median(win)) if len(win) else float("nan"),
                cross_p95=float(np.percentile(cross, 95)) if len(cross) else float("nan"),
                gap=(float(np.median(win)) - float(np.percentile(cross, 95))) if len(win) and len(cross) else float("nan"),
                n_same=int(len(win)), n_cross=int(len(cross)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base"); ap.add_argument("group", type=int)
    ap.add_argument("--nsamp", type=int, default=32); ap.add_argument("--nchan", type=int, default=8)
    ap.add_argument("--res"); ap.add_argument("--refine-clu"); ap.add_argument("--gt-clu")
    ap.add_argument("--sr", type=float, default=32552.0); ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--modes", default="xcorr,centroid")
    ap.add_argument("--min-spikes", type=int, default=80); ap.add_argument("--max-units", type=int, default=150)
    ap.add_argument("--n-win", type=int, default=14); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pooled-zca", action="store_true")
    a = ap.parse_args()
    B, g = a.base, a.group
    spk, _ = nio.open_spk_raw(B, g, a.nsamp, a.nchan)
    res = np.fromfile(a.res or f"{B}.res.{g}", dtype="<i8")
    lab = _read_clu(a.refine_clu or f"{B}.clu.stderiv.{g}.refine")
    gt = _read_clu(a.gt_clu) if a.gt_clu else None
    nn = min(spk.shape[0], len(lab), len(res))
    lab, res = lab[:nn], res[:nn]
    if gt is not None:
        gt = gt[:nn]
    sel = np.flatnonzero(lab != 0) if a.minutes <= 0 else \
        np.flatnonzero((res >= res.max() - int(a.minutes * 60 * a.sr)) & (lab != 0))
    frags, truth = build_fragments(lab, res, sel, gt=gt, min_spikes=a.min_spikes,
                                   max_units=a.max_units, seed=a.seed)
    mask = np.arange(a.nsamp)
    W = nmean = None
    if a.pooled_zca:
        pool = np.concatenate([np.asarray(spk[ii], float)[:, mask, :].reshape(len(ii), -1)
                               for ii in frags[:200]])
        nmean = pool.mean(0); Xc = pool - nmean
        cov = Xc.T @ Xc / len(Xc); ev, V = np.linalg.eigh(cov)
        W = V @ np.diag(1.0 / np.sqrt(np.maximum(ev, 1e-6))) @ V.T
    gtname = a.gt_clu if gt is not None else "time-split self-continuity"
    print(f"# {len(set(truth.tolist()))} units / {len(frags)} fragments  | ground truth: {gtname}")
    print(f"# whitener: {'pooled-ZCA' if a.pooled_zca else 'identity'}  | signature: _sliding_pairs weighted cosine\n")
    print(f"{'mode':10s} {'AUC(same>cross)':>16s} {'median_same':>12s} {'cross_p95':>10s} {'gap':>8s}  pairs(same/cross)")
    for mode in a.modes.split(","):
        fl.set_feature_align(mode.strip())
        pr, cn, co = sig_profiles(frags, spk, mask, n_win=a.n_win, W=W, nmean=nmean)
        m = margin(pair_cos(pr, cn, co), truth)
        print(f"{mode:10s} {m['auc']:16.4f} {m['med_same']:12.3f} {m['cross_p95']:10.3f} {m['gap']:+8.3f}  {m['n_same']}/{m['n_cross']}")
    fl.set_feature_align("xcorr")


if __name__ == "__main__":
    main()
