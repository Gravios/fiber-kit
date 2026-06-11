# ════════════════════════════════════════════════════════════════════════════
#  fiber_trajectory.py — iterative trajectory refinement of linked bundles.
#
#  After the overlap-backbone + co-gate linker has built bundles (one per tracked
#  neuron), each bundle is a sequence of per-chunk units that should lie on a single
#  smooth path as the neuron drifts: a depth trajectory y0(t) in physical space and a
#  smooth curve F(t) in template-PCA feature space (plus a near-flat logA(t), since A
#  is the drift-invariant amplitude anchor).  We fit those trajectories and use them as
#  a model to do two things the pairwise linker cannot:
#
#    1. RESOLVE same-chunk conflicts.  A bundle that claims >=2 units in one chunk is a
#       provable mis-merge (one neuron is one unit per chunk).  We predict that chunk's
#       position/feature from the OTHER chunks and keep the unit closest to the
#       prediction, evicting the rest -- they fall back to singletons and may re-attach.
#    2. EXTEND linkage.  A singleton (or small-bundle) unit that lies on a bundle's
#       predicted trajectory in BOTH depth and PCA-feature space (and logA), in a chunk
#       the bundle does not yet occupy, is attached -- bridging dropouts by interpolation.
#
#  Tolerances are calibrated from the existing multi-chunk members' own residuals (a
#  quantile), so the gate adapts to the session's drift/noise rather than a magic number.
#  Iterate conflict-resolve -> attach -> refit until no change.
#
#  This is a linking-stage refinement: it operates on the per-unit signatures the linker
#  already uses, and changes only bundle membership (no re-alignment, no re-clustering).
#  Validated on the 350-min g5 session: 31 same-chunk-conflict bundles -> 0, span>=3
#  bundles 72 -> 92, with attached units fitting tighter (leave-one-out depth resid
#  3.2/10.6 median/95p) than existing members (3.6/29.8) and zero extrapolation.
# ════════════════════════════════════════════════════════════════════════════
import numpy as np


def feature_coords(templates, K=4):
    """Stack centered unit templates and project onto their top-K PCA axes.
    templates: sequence of (T,C) arrays (or an (U,T,C)/(U,D) array).  Returns
    F (U,K) coordinates and per-axis scales (for a unit-free residual)."""
    T = np.stack([np.ravel(t) for t in templates]).astype(np.float64)
    Tc = T - T.mean(0)
    _, _, Vt = np.linalg.svd(Tc, full_matrices=False)
    K = min(K, Vt.shape[0])
    F = Tc @ Vt[:K].T
    return F, F.std(0) + 1e-9


def _robust_polyfit(t, v):
    """Low-order polynomial of v(t) with one IRLS reweighting round (Cauchy weights).
    Degree scales with support: 2 for >=4 points, 1 for 2-3, constant for 1."""
    deg = 2 if len(t) >= 4 else (1 if len(t) >= 2 else 0)
    if deg == 0:
        return np.array([float(v[0])])
    c = np.polyfit(t, v, deg)
    r = v - np.polyval(c, t)
    s = np.median(np.abs(r)) * 1.4826 + 1e-9
    w = 1.0 / (1.0 + (r / (3 * s)) ** 2)
    return np.polyfit(t, v, deg, w=w)


def _trajectory(members, tmin_arr, y0, logA, F):
    t = tmin_arr[members]
    return dict(y=_robust_polyfit(t, y0[members]),
                la=_robust_polyfit(t, logA[members]),
                F=[_robust_polyfit(t, F[members, k]) for k in range(F.shape[1])],
                tmin=t.min(), tmax=t.max(), ymean=y0[members].mean())


def _residual(u, tj, tmin_arr, y0, logA, F, fscale):
    dy = abs(y0[u] - np.polyval(tj["y"], tmin_arr[u]))
    df = np.sqrt(sum(((F[u, k] - np.polyval(tj["F"][k], tmin_arr[u])) / fscale[k]) ** 2
                     for k in range(F.shape[1])))
    dla = abs(logA[u] - np.polyval(tj["la"], tmin_arr[u]))
    return dy, df, dla


def refine_bundles(frag, bundles, chunk, *, K=4, quantile=0.95, lat=0.25,
                   ext_min=0.0, chunk_min=12.0, max_iters=8):
    """Iteratively refine bundles with the trajectory model.

    frag    : per-fragment dict with 'template', 'y0', 'A', 't_mid'(seconds).
    bundles : list of index lists (link_session output).
    chunk   : per-fragment chunk id (same indexing as frag).
    Returns (new_bundles, info) where info has attached/evicted counts and the
    conflict count before/after.  ext_min (minutes): how far an attach may reach BEYOND a
    bundle's member time span.  Default 0 = interpolation only -- internal dropouts are
    still bridged (their chunk lies between members), but tracks are not extended past
    their endpoints.  Raise it (e.g. one chunk) to allow extrapolation-based extension;
    on g5 that stays in-family by leave-one-out residual but is, by nature, prediction
    beyond the data."""
    bundles = [list(b) for b in bundles]
    U = len(frag["y0"])
    y0 = np.asarray(frag["y0"], float)
    logA = np.log(np.clip(np.asarray(frag["A"], float), 1, None))
    ch = np.asarray(chunk)
    tmin_arr = np.asarray(frag["t_mid"], float) / 60.0           # minutes
    F, fscale = feature_coords(frag["template"], K)

    def traj(mem): return _trajectory(mem, tmin_arr, y0, logA, F)
    def resid(u, tj): return _residual(u, tj, tmin_arr, y0, logA, F, fscale)
    def is_multi(b): return len(set(ch[b])) >= 2

    # calibrate depth + feature tolerances from current multi-chunk members
    ry, rf = [], []
    for b in bundles:
        if not is_multi(b):
            continue
        tj = traj(b)
        for u in b:
            dy, df, _ = resid(u, tj); ry.append(dy); rf.append(df)
    if not ry:
        return bundles, dict(attached=0, evicted=0, conflicts_before=0, conflicts_after=0)
    yt = float(np.quantile(ry, quantile)); ft = float(np.quantile(rf, quantile))
    conf0 = sum(len(b) != len(set(ch[b])) for b in bundles)

    bid = np.full(U, -1)
    for i, b in enumerate(bundles):
        for u in b: bid[u] = i
    B = [list(b) for b in bundles]
    n_att = n_evict = 0
    for _ in range(max_iters):
        changed = 0
        TJ = {i: traj(B[i]) for i in range(len(B)) if is_multi(B[i])}
        # (a) conflict resolution
        for bi in list(TJ):
            bc = {}
            for u in B[bi]: bc.setdefault(int(ch[u]), []).append(u)
            for c in [c for c, us in bc.items() if len(us) > 1]:
                other = [m for m in B[bi] if ch[m] != c]
                tj = traj(other) if len(other) >= 2 else TJ[bi]
                sc = {u: (lambda r: r[0] / yt + r[1] / ft)(resid(u, tj)) for u in bc[c]}
                keep = min(sc, key=sc.get)
                for u in bc[c]:
                    if u != keep:
                        B[bi].remove(u); bid[u] = -1; n_evict += 1; changed += 1
            TJ[bi] = traj(B[bi]) if is_multi(B[bi]) else None
        TJ = {bi: tj for bi, tj in TJ.items() if tj is not None and B[bi]}
        # (b) attach orphans / small-bundle members to the best-fitting bundle
        blist = list(TJ)
        bym = np.array([TJ[bi]["ymean"] for bi in blist])
        bmn = np.array([TJ[bi]["tmin"] for bi in blist])
        bmx = np.array([TJ[bi]["tmax"] for bi in blist])
        for u in range(U):
            if bid[u] != -1 and is_multi(B[bid[u]]):
                continue
            cand = np.flatnonzero((np.abs(bym - y0[u]) < 3 * yt)
                                  & (tmin_arr[u] >= bmn - ext_min)
                                  & (tmin_arr[u] <= bmx + ext_min))
            best, bs = None, 2.0
            for j in cand:
                bi = blist[j]
                if any(ch[m] == ch[u] for m in B[bi]):       # no new same-chunk conflict
                    continue
                dy, df, dla = resid(u, TJ[bi])
                if dy <= yt and df <= ft and dla <= lat and dy / yt + df / ft < bs:
                    bs = dy / yt + df / ft; best = bi
            if best is not None:
                if bid[u] != -1: B[bid[u]].remove(u)
                B[best].append(u); bid[u] = best; n_att += 1; changed += 1
        if changed == 0:
            break
    B = [b for b in B if b]
    info = dict(attached=n_att, evicted=n_evict, conflicts_before=conf0,
                conflicts_after=sum(len(b) != len(set(ch[b])) for b in B),
                depth_tol=yt, feat_tol=ft)
    return B, info
