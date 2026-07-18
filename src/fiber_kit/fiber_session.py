#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════════════════════
#  fiber_session.py  —  full-session fiber clusterer (validated-Python reference)
#
#  Per chunk (all on the validated fiber_lib/fiber_tracer primitives):
#    off-spike .fil whitener (spkD space) -> realign -> mask -> whiten
#    -> COARSE in-band mean-shift ridge seeding -> dedup -> substantial centers
#       -> reassign -> run_from_seeds   (fat drift-stable fibers, for linking)
#    -> FINE refinement WITHIN the chunk: BIC-GMM (the Python stand-in for KK's
#       CEM split) on each coarse fiber -> sub-units.  Units inside a coarse fiber
#       are blobs separated in directions ORTHOGONAL to the ridge, which the
#       ridge-tracking fiber algorithm cannot split robustly (it leaves them whole
#       or shatters them: 14 vs GMM's 42 clean units on g5); CEM/GMM adapts per
#       fiber.  --fine-method fiber|none available but underperforms.
#  Refinement stays per-chunk so each fiber keeps its OWN geometry at each point
#  in time -> drift is visible across chunks.
#
#  Cross-chunk: OVERLAP-ANCHOR linking on the fine fibers.  A spike in chunk c's
#  overlap is the SAME physical spike in chunk c+1's overlap, so two fibers that
#  claim the same overlap spikes are the same unit (mutual-majority, drift-free).
#
#  Outputs:
#    <base>.clu.<elec>                          int32 nClusters header + ids (0=noise)
#    <base>.fibers.<method>.<elec>              npz: per (chunk,fiber) geometry,
#         keys: gid chunk tmin coarse nspk radius refrac depth (M,);
#               template (M,nsamp,nch); grid (M,n_grid); dir (M,n_grid,p);
#               + meta (elec, channels, sr, mask, n_grid, p, method, chunk/overlap min)
#         "fiber geometry over time" = rows sharing a gid, ordered by chunk.
#
#  Usage:
#    python3 fiber_session.py <FileBase> <ElecNo> \
#        --channels 32,33,34,35,36,37,38,39 --ntotal 96 --nsamp 32 --nchan 8 \
#        --sr 32552 --chunk-min 12 --overlap-min 4 \
#        --min-group 200 --fine-kappa 40 --fine-dedup-deg 5 --fine-min-group 40 \
#        --method stderiv [--no-fine] [--no-link]
# ════════════════════════════════════════════════════════════════════════════
import argparse, time
import os
import numpy as np
from collections import defaultdict, Counter
try:
    from . import fiber_lib as fl
except ImportError:
    import fiber_lib as fl
try:
    from . import fiber_tracer as ft
except ImportError:
    import fiber_tracer as ft
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
try:
    from .klustakwik import klustakwik as _rkk
except ImportError:
    from klustakwik import klustakwik as _rkk
try:
    from . import session_yaml as sy
except ImportError:
    import session_yaml as sy
try:
    from . import neuro_io as nio
except ImportError:
    import neuro_io as nio
try:
    from . import fiber_pca as _fpca
except ImportError:
    import fiber_pca as _fpca
try:
    from . import backend as _bk
except ImportError:
    import backend as _bk
try:
    from . import fiber_ccg as cg
except ImportError:
    import fiber_ccg as cg
try:
    import diptest as _diptest
    _HAVE_DIP = True
except Exception:
    _HAVE_DIP = False

P_DIM = len(fl.MASK_FULL) * 8     # default masked feature dim (recomputed per nchan below)


def cluster_chunk(waves, W, nmean, min_group=100, kappa=20.0, dr_frac=0.15,
                  n_seeds=800, n_support=20000, dedup_deg=8.0, dedup_radf=0.12,
                  mask=fl.MASK_FULL):
    """waves (n,nsamp,nch) spkD + chunk whitener -> per-spike label (0-based, -1=none)."""
    N = len(waves)
    if N < 2 * min_group:
        return np.full(N, -1, int)
    X = (fl.realign(waves)[:, mask, :].reshape(N, -1) - nmean) @ W
    r = np.linalg.norm(X, axis=1); d = X / (r[:, None] + 1e-12)
    nsup = min(N, n_support); supi = np.arange(nsup) * N // nsup
    dsup, rsup = d[supi], r[supi]
    S = min(N, n_seeds); sdi = np.arange(S) * N // S
    ds, rs = d[sdi].copy(), r[sdi].copy()
    rsort = np.sort(r); dr = dr_frac * (rsort[int(0.99 * (N - 1))] - rsort[int(0.01 * (N - 1))])
    for _ in range(15):
        cos = ds @ dsup.T
        w = np.where(np.abs(rsup[None, :] - rs[:, None]) < dr, np.exp(kappa * (cos - 1)), 0.0)
        sw = w.sum(1); sw[sw < 1e-9] = 1e-9
        ds = w @ dsup; ds /= np.linalg.norm(ds, axis=1, keepdims=True) + 1e-12
        rs = (w * rsup[None, :]).sum(1) / sw
    order = np.argsort(-rs); cd = []; cr = []; cth = np.cos(np.deg2rad(dedup_deg))
    for i in order:
        if all(not (ds[i] @ c > cth and abs(rs[i] - q) / q < dedup_radf) for c, q in zip(cd, cr)):
            cd.append(ds[i]); cr.append(rs[i])
    cd = np.array(cd); M = len(cd)
    lab = np.argmax(d @ cd.T, 1); sizes = np.bincount(lab, minlength=M)
    keep = np.flatnonzero(sizes >= min_group)
    if len(keep) == 0:
        return np.full(N, -1, int)
    lab2 = np.argmax(d @ cd[keep].T, 1)
    groups = {int(k): np.flatnonzero(lab2 == k) for k in range(len(keep))}
    out = ft.run_from_seeds(waves, groups, W, nmean, mask=mask)
    keys = {k: i for i, k in enumerate(out['keys'])}
    return np.array([keys.get(h, -1) if h is not None else -1 for h in out['hard']], int)


def fiber_geom(wsub, res_sub, W, nmean, mask, sr, n_grid=40, chunk_t0=None, chunk_t1=None):
    """Geometry + per-chunk quality / firing / drift statistics of one fiber.
    Every stat is per-chunk, so rows sharing a gid across the session form time
    series (depth(t), rate(t), nn_dist(t), within-chunk drift, ...) for curation."""
    p = len(mask) * wsub.shape[2]
    w_al = fl.realign(wsub); template = w_al.mean(0)
    Xg = (w_al[:, mask, :].reshape(len(w_al), -1) - nmean) @ W
    rr = np.linalg.norm(Xg, axis=1)
    grid, D = ft.trajectory(Xg)
    radii = np.linspace(grid[0], grid[-1], n_grid)
    dirs = np.array([ft.predict((grid, D), float(x)) for x in radii]) if grid[-1] > grid[0] \
        else np.repeat(D[:1], n_grid, 0)
    nch = template.shape[1]; ch = np.arange(nch)
    ptp_t = np.maximum(template.max(0) - template.min(0), 0.0)
    depth = float((ptp_t * ch).sum() / (ptp_t.sum() + 1e-9))
    wav = np.cumsum(template, axis=0); dom = int(np.argmax(wav.max(0) - wav.min(0))); sdom = wav[:, dom]
    tr = int(np.argmin(sdom)); pk = (tr + int(np.argmax(sdom[tr:]))) if tr < len(sdom) - 1 else tr
    width_ms = float((pk - tr) / sr * 1000.0)

    # ── firing / timing (from res) ──
    n = int(len(wsub)); o = np.argsort(res_sub.astype(float)); t = res_sub.astype(float)[o]
    isi_ms = (np.diff(t) / sr * 1000.0) if n > 1 else np.array([])
    refr = float((isi_ms < 2.0).mean() * 100) if n > 10 else float('nan')
    burst = float((isi_ms < 6.0).mean()) if n > 10 else float('nan')
    isi_cv = float(isi_ms.std() / (isi_ms.mean() + 1e-9)) if n > 10 else float('nan')
    ct0 = float(chunk_t0) if chunk_t0 is not None else float(t.min())
    ct1 = float(chunk_t1) if chunk_t1 is not None else float(t.max())
    dur_s = max((ct1 - ct0) / sr, 1e-9); rate = float(n / dur_s)
    presence = float((np.histogram(t, np.linspace(ct0, ct1, 21))[0] > 0).mean())
    rpv = int((isi_ms < 2.0).sum()); t_r = 0.002          # Hill refractory false-positive fraction
    Q = (rpv * dur_s / (2.0 * n * n * t_r)) if n > 1 else float('nan')
    hill = float(0.5 * (1.0 - np.sqrt(1.0 - 4.0 * Q))) if (Q == Q and Q <= 0.25) else float('nan')

    # ── compactness: whitened residual to own trajectory ──
    def _pv(rx):
        rc = np.clip(rx, grid[0], grid[-1]); j = np.clip(np.searchsorted(grid, rc) - 1, 0, len(grid) - 2)
        wf = ((rc - grid[j]) / (grid[j + 1] - grid[j] + 1e-12))[:, None]; v = D[j] + wf * (D[j + 1] - D[j])
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    if grid[-1] > grid[0] and n >= 20:
        resid = np.linalg.norm(Xg - rr[:, None] * _pv(rr), axis=1)
        resid_med = float(np.median(resid)); resid_mad = float(1.4826 * np.median(np.abs(resid - resid_med)))
    else:
        resid_med = resid_mad = float('nan')

    # ── within-chunk drift (slopes vs time; first/second-half direction change) ──
    tmin = (t - t.min()) / sr / 60.0                       # minutes, time-ordered
    ptp_s = np.maximum(w_al.max(1) - w_al.min(1), 0.0)
    depth_s = (ptp_s * ch).sum(1) / (ptp_s.sum(1) + 1e-9)
    if n >= 20 and tmin[-1] > tmin[0]:
        radius_slope = float(np.polyfit(tmin, rr[o], 1)[0])      # whitened-radius units / min
        depth_slope = float(np.polyfit(tmin, depth_s[o], 1)[0])  # channels / min
    else:
        radius_slope = depth_slope = float('nan')
    dir_drift = float('nan')
    if n >= 80:
        h = n // 2; a, b = o[:h], o[h:]
        dd = _profile_dir_dist(ft.trajectory(Xg[a]), np.percentile(rr[a], [15, 85]),
                               ft.trajectory(Xg[b]), np.percentile(rr[b], [15, 85]))
        if dd is not None: dir_drift = dd

    try:
        from . import fiber_adapt as _fa
    except ImportError:
        import fiber_adapt as _fa
    _z, _ai = _fa.adaptation_residual(wsub, res_sub, W, nmean, mask, sr)
    crv = ft.channel_residual_profile(wsub, W, nmean, mask) if n >= 20 else None
    return dict(n=n, radius=float(rr.mean()), rate=rate, presence=presence,
                refrac=refr, burst=burst, isi_cv=isi_cv, hill_fp=hill,
                resid_med=resid_med, resid_mad=resid_mad,
                chan_resid_var_mean=float(crv['mean']) if crv else float('nan'),
                chan_resid_var_max=float(crv['max']) if crv else float('nan'),
                depth=depth, width_ms=width_ms,
                radius_slope=radius_slope, depth_slope=depth_slope, dir_drift=dir_drift,
                adapt_corr=float(_ai['corr']), adapt_tau=float(_ai['tau']), adapt_snr=float(_ai['snr']),
                adapt_meanabsz=float(np.mean(np.abs(_z))), adapt_fracz3=float(np.mean(np.abs(_z) > 3)),
                nn_dist=float('nan'), nn_local=-1,        # filled by NN post-pass
                lratio=float('nan'), iso_dist=float('nan'),  # filled if quality_metrics
                rrange=np.array([np.percentile(rr, 15), np.percentile(rr, 85)], np.float32),
                template=template.astype(np.float32),
                grid=radii.astype(np.float32),
                dir=dirs.astype(np.float32).reshape(n_grid, p))


def _bic_gmm(F, max_sub=8, reg=1e-3):
    """BIC-selected full-covariance GMM on precomputed features F (n,d) -> labels 0..k-1."""
    N = len(F)
    if N < 60:
        return np.zeros(N, int)
    best = None
    for k in range(1, max_sub + 1):
        if k * 3 > N:
            break
        g = GaussianMixture(k, covariance_type='full', reg_covar=reg, random_state=0, n_init=2).fit(F)
        b = g.bic(F)
        if best is None or b < best[0]:
            best = (b, k, g)
    return best[2].predict(F)


def gmm_split(wf, pca_k=6, max_sub=8, mask=fl.MASK_FULL, reg=1e-3, basis=None):
    """BIC-selected Gaussian mixture on PCA of a coarse fiber's realigned waveforms
    (the Python stand-in for KK's CEM split).  Returns sub-labels 0..k-1.

    SHAPE features: if a global ndm_pca `basis` is given, the realigned waveforms are
    projected onto it (shared basis across chunks/runs) instead of a per-call local SVD;
    falls back to the local SVD when no basis is given or its channels mismatch."""
    N = len(wf)
    if N < 60: return np.zeros(N, int)
    al = fl.realign(wf)
    F = _fpca.cluster_features(al, basis, realign=False, dims=pca_k) if basis is not None else None
    if F is None:
        w = al[:, mask, :].reshape(N, -1); w = w - w.mean(0)
        U, S, Vt = np.linalg.svd(w, full_matrices=False); F = U[:, :pca_k] * S[:pca_k]
    return _bic_gmm(F, max_sub=max_sub, reg=reg)


def _energy_band_split(wcf, mask, band_w=0.45, overlap=0.2, pca_k=6, max_sub=8,
                       min_band=60, confound_thr=0.4, min_span=0.6, low_assign=0.0, reg=1e-3):
    """Energy-band-partitioned split of ONE coarse fiber (validated on g5).

    A coarse fiber's PC1 of waveform variation is typically energy/drift (it
    confounds the shape split, burying co-energy sub-units below a drift axis).
    Partitioning by overlapping log10-energy bands freezes that axis -- within a
    band the energy PC has ~no variance, so the BIC-GMM splits on SHAPE.

    Design (each choice validated):
      * features = align + PCA computed GLOBALLY on the whole fiber and REUSED per
        band; per-band recomputation is noisier (small-N) and loses ~0.07 ARI.
        The band restriction alone neutralises the energy PC (constant in a band).
      * clustering = BIC-GMM restricted to each band.
      * linking = OVERLAP-ANCHOR (shared spikes in the band overlap prove identity,
        gap-independent); the graph/A* linkage chains on this gapless cloud.

    Returns a list of index arrays into wcf, or None when the fiber is NOT
    energy-confounded (PC1 R^2 vs log-energy < confound_thr) or spans < min_span
    decades -- the caller then falls back to the configured whole-fiber split.
    Energy is raw (un-whitened) masked-window ||x||^2.

    low_assign : fraction of the energy range (from the bottom) treated as
    ASSIGNMENT-ONLY.  Near the noise floor the direction d=X/||X|| is noise-
    dominated, so a band's own shape split there is unreliable (measured: lowest-
    band within-band recovery collapses while the middle bands are clean).  In
    that floor no mixture is fit; those spikes are assigned to the units the bands
    above establish (overlap-anchor continuity + nearest-feature fallback).  0.0
    (default) splits every band independently."""
    N = len(wcf)
    if N < 2 * min_band:
        return None
    le = np.log10((wcf[:, mask, :].astype(np.float64) ** 2).sum(axis=(1, 2)) + 1e-12)
    if float(np.ptp(le)) < min_span:
        return None
    al = fl.realign(wcf)[:, mask, :].reshape(N, -1); al = al - al.mean(0)
    U, S, _ = np.linalg.svd(al, full_matrices=False); F = U[:, :pca_k] * S[:pca_k]
    yh = np.polyval(np.polyfit(le, F[:, 0], 1), le)        # confound gate: is PC1 ~ energy?
    r2 = 1.0 - ((F[:, 0] - yh) ** 2).sum() / (((F[:, 0] - F[:, 0].mean()) ** 2).sum() + 1e-12)
    if r2 < confound_thr:
        return None
    emin, emax = float(le.min()), float(le.max())
    lo_cut = emin + max(0.0, low_assign) * (emax - emin)   # below lo_cut: assignment-only (noise-floor angle)
    edges = list(np.arange(emin, emax, band_w)) + [emax]; nb = len(edges) - 1
    ext_idx, ext_lab = [], []
    core_band = np.full(N, -1, int); core_local = np.full(N, -1, int)
    for b in range(nb):
        lo, hi = edges[b], edges[b + 1]
        if hi <= lo_cut:                                   # band entirely inside the floor: no independent split
            ext_idx.append(np.array([], int)); ext_lab.append(np.array([], int)); continue
        clo = max(lo, lo_cut)                              # never train/core on floor spikes
        ei = np.where((le >= clo - overlap) & (le < hi + overlap) & (le >= lo_cut))[0]
        if len(ei) < min_band:
            ext_idx.append(np.array([], int)); ext_lab.append(np.array([], int)); continue
        sub = _bic_gmm(F[ei], max_sub=max_sub, reg=reg)
        ext_idx.append(ei); ext_lab.append(sub)            # ei = LOCAL indices; overlap spike = same local id in adjacent bands
        core = (le >= clo) & ((le < hi) if b < nb - 1 else (le <= hi))
        for j, gi in enumerate(ei):
            if core[gi]:
                core_band[gi] = b; core_local[gi] = int(sub[j])
    gid, _ = link_chunks(ext_idx, ext_lab, min_anchor=8, frac=0.5)
    glob = np.array([gid.get((core_band[i], core_local[i]), -1) for i in range(N)])
    valid = np.unique(glob[glob >= 0])
    if len(valid) <= 1:
        return None                                        # nothing gained over whole-fiber
    remap = {int(g): i for i, g in enumerate(valid)}
    lab = np.array([remap.get(int(x), -1) for x in glob])
    un = np.where(lab < 0)[0]                             # assignment-only (le<lo_cut) + too-sparse-band spikes -> nearest unit
    if len(un):
        cents = np.array([F[lab == i].mean(0) for i in range(len(valid))])
        for i in un:
            lab[i] = int(np.argmin(((cents - F[i]) ** 2).sum(1)))
    return [np.where(lab == i)[0] for i in range(len(valid))]


def _dipsplit_rec(F, idx, min_size=40, alpha=0.01, depth=0, maxd=4):
    """Recursive DipSplit: 2-means -> project on the centroid axis -> Hartigan dip
    test -> split iff p<alpha and both halves substantial.  Catches bimodal clusters
    BIC left merged.  F: (n,dim) low-dim features; idx: indices into F."""
    n = len(idx)
    if not _HAVE_DIP or n < 2 * min_size or depth > maxd: return [idx]
    km = KMeans(2, n_init=4, random_state=0).fit(F[idx])
    a = idx[km.labels_ == 0]; b = idx[km.labels_ == 1]
    if len(a) < min_size or len(b) < min_size: return [idx]
    dr = km.cluster_centers_[1] - km.cluster_centers_[0]; dr /= np.linalg.norm(dr) + 1e-9
    _, p = _diptest.diptest(np.ascontiguousarray(F[idx] @ dr))
    if p < alpha:
        return (_dipsplit_rec(F, a, min_size, alpha, depth + 1, maxd) +
                _dipsplit_rec(F, b, min_size, alpha, depth + 1, maxd))
    return [idx]


def _aligned_pca(waves, mask, k, basis=None):
    """Realign a (sub)cluster to its OWN median by iterated circular cross-correlation
    (fiber_lib.align_xcorr, the channel-summed sub-sample aligner) and return SHAPE features.
    If a global ndm_pca `basis` is given the median-aligned waveforms are projected onto it
    (shared basis across chunks/nodes); else the top-k scores of a per-call local SVD.  The
    integer dominant-channel fl.realign locks a sub-cluster onto the PARENT's peak; re-aligning
    to this node's own median before featurizing lets a deeper bisection be measured on correct
    alignment.  The xcorr realignment itself is unchanged -- only the feature projection moves to
    the global basis."""
    al = fl.align_xcorr(waves, ref="median", iters=6, maxlag=6)
    if basis is not None:
        F = _fpca.cluster_features(al, basis, realign=False, dims=k)
        if F is not None:
            return F
    w = al[:, mask, :].reshape(len(waves), -1)
    w = w - w.mean(0)
    U, S, _ = np.linalg.svd(w, full_matrices=False)
    return U[:, :k] * S[:k]


def _dipsplit_realign(waves, mask, dim, min_size=40, alpha=0.01, depth=0, maxd=4, basis=None):
    """Recursive DipSplit that REALIGNS EACH NODE to its own median before deciding the split:
    the 2-means centroid axis and dip test are recomputed from this sub-cluster's median-aligned
    SHAPE features (_aligned_pca: global ndm_pca basis when given, else local SVD), so every
    bisection is judged on its own alignment instead of the parent's (the per-step realign).
    Returns a list of index arrays into `waves`."""
    n = len(waves)
    if not _HAVE_DIP or n < 2 * min_size or depth > maxd:
        return [np.arange(n)]
    F = _aligned_pca(waves, mask, dim, basis=basis)         # realign THIS node + featurize
    km = KMeans(2, n_init=4, random_state=0).fit(F)
    a = np.flatnonzero(km.labels_ == 0); b = np.flatnonzero(km.labels_ == 1)
    if len(a) < min_size or len(b) < min_size:
        return [np.arange(n)]
    dr = km.cluster_centers_[1] - km.cluster_centers_[0]; dr /= np.linalg.norm(dr) + 1e-9
    _, p = _diptest.diptest(np.ascontiguousarray(F @ dr))
    if p >= alpha:
        return [np.arange(n)]
    out = []
    for loc in (a, b):
        for piece in _dipsplit_realign(waves[loc], mask, dim, min_size, alpha, depth + 1, maxd, basis=basis):
            out.append(loc[piece])
    return out


def _rkk_realign(waves, mask, dims, max_clusters, min_size, iters=2, delete=True, basis=None):
    """rkk (CEM) interleaved with per-cluster realignment -- the per-step realign analog for the
    flat KK split.  rkk assigns all spikes in one EM run, so there is no recursive node; instead
    iterate {cluster -> realign EACH cluster to its own median -> re-featurize -> re-cluster}, so
    the final CEM runs on consistently per-cluster-aligned features (a minority sub-unit locked
    onto the group's dominant peak by the integer fl.realign is otherwise smeared).  SHAPE features
    project onto the global ndm_pca basis when given, else a per-call local SVD.  Stops early
    when the cluster count is unchanged.  Returns per-spike sub-labels."""
    F = _aligned_pca(waves, mask, dims, basis=basis)       # whole-group median align + features (seed)
    lab = _rkk(F, max_clusters=max_clusters, min_size=min_size, seed=42, delete=delete)
    for _ in range(max(0, iters)):
        Wal = np.array(waves, dtype=float)
        for c in np.unique(lab):                           # realign each cluster to its OWN median
            idx = np.flatnonzero(lab == c)
            if len(idx) >= 8:
                Wal[idx] = fl.align_xcorr(waves[idx], ref="median", iters=6, maxlag=6)
        F = _fpca.cluster_features(Wal, basis, realign=False, dims=dims) if basis is not None else None
        if F is None:
            w = Wal[:, mask, :].reshape(len(waves), -1); w = w - w.mean(0)
            U, S, _ = np.linalg.svd(w, full_matrices=False); F = U[:, :dims] * S[:dims]
        new = _rkk(F, max_clusters=max_clusters, min_size=min_size, seed=42, delete=delete)
        stop = len(np.unique(new)) == len(np.unique(lab))
        lab = new
        if stop:
            break
    return lab


def _amp_spread(waves, mask):
    """Peak amplitude and number of signal channels of the median-aligned template — the
    low-amplitude / broad-noise gate for the nudge split."""
    T = np.median(fl.align_xcorr(waves, ref="median", iters=6, maxlag=6), 0)
    ptp = T.max(0) - T.min(0)
    amp = float(ptp.max())
    nch = int((ptp > 0.25 * amp).sum())
    return amp, nch


def _nudge_split(waves, mask, dim, min_size, alpha, max_nudge=3):
    """Offset-overlay split for low-amplitude clusters.  Two neurons of similar shape a few
    samples apart are MERGED by median realignment — it collapses the offset, so the per-node
    realign dipsplit returns them as one (validated: ARI 0.00).  Split on each spike's alignment
    LAG to the cluster median instead — the 'nudge' each spike wants: bimodal lags = two
    temporally-offset sub-units (ARI 0.98).  Self-gating: clean clusters have unimodal lags and
    are returned whole (0% spurious splits in test).  Each offset sub-cluster is then realigned to
    its own median and dip-refined (reusing _dipsplit_realign)."""
    n = len(waves)
    if not _HAVE_DIP or n < 2 * min_size:
        return [np.arange(n)]
    _, sh = fl.align_xcorr(waves, ref="median", iters=6, maxlag=max_nudge, return_shifts=True)
    parts = _dipsplit_rec(np.asarray(sh, float).reshape(-1, 1), np.arange(n), min_size, alpha)
    if len(parts) == 1:                                  # unimodal lags -> no offset overlay
        return [np.arange(n)]
    out = []
    for p in parts:                                      # refine each offset sub-unit on its own alignment
        for piece in _dipsplit_realign(waves[p], mask, dim, min_size, alpha):
            out.append(p[piece])
    return out


def _variance_split(waves, W, nmean, mask, n_grid, peak, margin, min_n, dims,
                    depth=0, max_depth=4):
    """Variance-driven auto-split: recursively bisect a fiber WHILE its per-channel
    residual-variance profile is peaked (channel-localized contamination) AND each
    bisection lowers the mean per-channel residual variance by >= margin.

    The measure is the stopping criterion, so it finds the right number of shape
    sub-units (no rkk over-fragmentation) and never splits on energy (an
    energy-only difference leaves the trajectory residual flat). Each split is on
    the trajectory residual WEIGHTED toward the high-variance channels, so the
    bisection looks where the contamination actually is. Returns a list of index
    arrays into `waves`."""
    n = len(waves)
    if n < 2 * min_n or depth >= max_depth:
        return [np.arange(n)]
    prof = ft.channel_residual_profile(waves, W, nmean, mask, n_grid=n_grid)
    vc = prof['per_channel']; med = float(np.median(vc)) + 1e-12
    if vc.max() / med < peak:                         # flat profile -> no shape contamination
        return [np.arange(n)]
    wch = np.sqrt(np.maximum(vc - med, 0.0))          # weight discriminative channels (excess over floor)
    if not np.any(wch > 0):
        return [np.arange(n)]
    F = (prof['residual'] * wch[None, None, :]).reshape(n, -1); F = F - F.mean(0)
    U, S, _ = np.linalg.svd(F, full_matrices=False); Fr = U[:, :dims] * S[:dims]
    km = KMeans(2, n_init=5, random_state=0).fit_predict(Fr)
    if np.bincount(km).min() < min_n:
        return [np.arange(n)]
    _, _, red = ft.split_meanvar(waves, km, W, nmean, mask, n_grid=n_grid, min_n=min_n)
    if red < margin:                                  # bisection doesn't reduce the measure -> stop
        return [np.arange(n)]
    out = []
    for k in (0, 1):
        idx = np.flatnonzero(km == k)
        for piece in _variance_split(waves[idx], W, nmean, mask, n_grid, peak, margin,
                                     min_n, dims, depth + 1, max_depth):
            out.append(idx[piece])
    return out


try:
    from . import fiber_cfiber as fcf
except ImportError:
    import fiber_cfiber as fcf


def _cfiber_edge_filter(edges, fine, waves, mask, q=0.90, modes=(2, 3, 4, -1, -2, -3)):
    """Veto candidate fragment-merge edges whose affine-invariant cfiber SHAPE disagrees.
    cfiber AUC on well-populated g5 units is ~0.998, so a shape mismatch beyond the within-
    fiber split-half null is strong evidence of two cells.  The veto threshold is CALIBRATED
    per chunk from that null (quantile q), so it adapts to the chunk's noise rather than a
    fixed constant.  Edges where either fiber is too small to estimate a stable shape are
    LEFT ALONE (the gate only vetoes when it is confident).  Returns the filtered edges."""
    if not edges:
        return edges
    theta = fcf.channel_angles(waves.shape[2])
    mi = np.asarray(mask); win = slice(int(mi.min()), int(mi.max()) + 1)
    rng = np.random.default_rng(0)
    def shp(idx):
        if len(idx) < 6:
            return None
        t = fl.realign(waves[idx]).mean(0)
        z = fcf.complex_loop(t[None], theta, win)[0]
        s, _, _, _ = fcf.shape_descriptor(z[None], modes)
        return s[0]
    nodes = sorted({u for e in edges for u in e})
    S = {}; nulls = []
    for u in nodes:
        ix = np.flatnonzero(fine == u); S[u] = shp(ix)
        if len(ix) >= 12:
            pp = rng.permutation(len(ix)); h = len(pp) // 2
            a = shp(ix[pp[:h]]); b = shp(ix[pp[h:]])
            if a is not None and b is not None:
                nulls.append(float(np.linalg.norm(a - b)))
    thr = float(np.quantile(nulls, q)) if nulls else np.inf
    return [(i, j) for (i, j) in edges
            if S.get(i) is None or S.get(j) is None or float(np.linalg.norm(S[i] - S[j])) <= thr]


def _gauss1d(x, sig):
    """Gaussian smooth along axis 0 (samples); numpy only (no scipy dep)."""
    r = max(1, int(3 * sig)); k = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2); k /= k.sum()
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, x)


def _rms_peak_window(median, sigma=1.0, half=8):
    """+-half-sample window centred on the smoothed-RMS-energy peak.  The smoothing ONLY locates the
    peak; callers compute the residual on the RAW window."""
    ms = _gauss1d(median, sigma); pk = int(np.argmax(np.sqrt((ms ** 2).mean(1))))
    return max(0, pk - half), min(median.shape[0], pk + half + 1)


def _shape_residual(waves, sigma=1.0, half=8):
    """Amplitude-scaled max residual from the cluster MEDIAN over the +-half window at the RMS peak.
    GT-free within-cluster tightness; rises when distinct cells are welded, robust to over-splitting."""
    m = np.median(waves, 0); lo, hi = _rms_peak_window(m, sigma, half)
    return float((waves[:, lo:hi, :] - m[lo:hi, :]).var(0).max() / (np.ptp(m) ** 2 + 1e-9))


def _episode_position(t_ms, win_ms=90.0):
    """Where a spike sits inside its firing episode: (spikes after) - (spikes before), counted in a
    +-win_ms window and excluding the spike itself.  Returned in the CALLER's spike order.

    Deliberately ANTI-CAUSAL.  A causal state -- the preceding ISI, or fiber_adapt's EWMA a[i] --
    cannot tell the START of a burst from its END, and that distinction is precisely what makes a
    feature order spikes in time.  Measured on g5: removing the axis this variable defines cancels
    69% of the split-induced CCG asymmetry on average and never increases it (worst case +16%),
    where fiber_adapt's EWMA at its best tau manages 56% and makes two of eight cells WORSE."""
    t = np.asarray(t_ms, float); o = np.argsort(t); ts = t[o]; ar = np.arange(len(ts))
    nb = ar - np.searchsorted(ts, ts - win_ms, side="left")
    na = np.searchsorted(ts, ts + win_ms, side="right") - ar - 1
    y = np.empty(len(t)); y[o] = (na - nb).astype(float)
    return y


def _detrend_axis(F, y):
    """Remove from F the single direction along which it covaries with y (rank-1 projection)."""
    y = np.asarray(y, float) - float(np.mean(y))
    c = (F - F.mean(0)).T @ y / max(len(F), 1)
    nrm = float(np.linalg.norm(c))
    if nrm <= 0:
        return F
    c = c / nrm
    return F - np.outer(F @ c, c)


def _em_swap(waves, topk=3, max_iter=40, min_reduction=0.20, min_n=10,
             episode=None, detrend_min_n=100):
    """Hard-EM spike swap in the TARGET-CHANNEL RESIDUAL space (not PCA).  After primary alignment,
    feature_i = (spike_i - combined group median) on the top-k highest-variance channels over the
    RMS-peak window.  2-means (farthest-point init, MEDIAN centroids) reassigns spikes to the sub-
    cluster whose residual-centroid they best match -- descending the within-group target-channel
    variance; the E-step is the swap.  Split kept only if it cuts that variance by >= min_reduction
    with both parts >= min_n.  Returns per-spike sub-labels (all 0 if not split)."""
    n = len(waves)
    if n < 2 * min_n:
        return np.zeros(n, int)
    M = np.median(waves, 0); lo, hi = _rms_peak_window(M)
    R = waves[:, lo:hi, :] - M[lo:hi, :]
    tgt = np.argsort(R.var(0).max(0))[::-1][:topk]
    F = R[:, :, tgt].reshape(n, -1)
    if episode is not None and n >= detrend_min_n:
        # Strip the episode-position axis BEFORE the split, so the 2-means cannot cut the cluster
        # along it.  That axis is a within-cell temporal gradient, not sub-structure: splitting on it
        # yields two "units" with a clean refractory gap and a strongly asymmetric CCG -- which reads
        # as a monosynaptic connection.  It holds ~1.5% of within-cluster variance for interneurons
        # and up to ~9% for bursty pyramidal cells (g5: asymmetry +0.53 on clu 3144 before removal).
        F = _detrend_axis(F, episode)
    var0 = F.var(0).sum()
    if var0 <= 0:
        return np.zeros(n, int)
    i1 = int(((F - F.mean(0)) ** 2).sum(1).argmax()); i2 = int(((F - F[i1]) ** 2).sum(1).argmax())
    cent = np.stack([F[i1], F[i2]]); lab = np.zeros(n, int)
    for _ in range(max_iter):
        nl = np.stack([((F - cent[k]) ** 2).sum(1) for k in range(2)]).argmin(0)
        if (nl == lab).all():
            break
        lab = nl
        cent = np.stack([np.median(F[lab == k], 0) if (lab == k).any() else cent[k] for k in range(2)])
    if (lab == 0).sum() < min_n or (lab == 1).sum() < min_n:
        return np.zeros(n, int)
    varS = sum(F[lab == k].var(0).sum() * (lab == k).sum() for k in (0, 1)) / n
    return lab if (1 - varS / var0) >= min_reduction else np.zeros(n, int)


def _rebuild_geoms(fine, waves, res_abs, W, nmean, mask, sr, n_grid, ct0, ct1, src_fine=None, src_geoms=None):
    """Relabel `fine` contiguous over its non-negative units and rebuild one geom per unit; noise
    (< 0) labels are preserved.  Used after the re-split so geoms match fine before the merge.
    Attaches the per-fiber metadata (`coarse`, `radius_incl`, `n_rejected`, `n_adapt_rejected`,
    `n_merged`) that fiber_geom does not set but _apply_edges + downstream read; coarse/radius_incl
    are carried from each fiber's majority parent in (src_fine, src_geoms) when given."""
    units = np.unique(fine[fine >= 0]); newfine = np.full(len(fine), -1, int); geoms = []
    for ni, u in enumerate(units):
        sidx = np.flatnonzero(fine == u)
        g = fiber_geom(waves[sidx], res_abs[sidx], W, nmean, mask, sr, n_grid, chunk_t0=ct0, chunk_t1=ct1)
        par = None
        if src_fine is not None and src_geoms is not None:
            pl = src_fine[sidx]; pl = pl[pl >= 0]
            if pl.size:
                par = src_geoms[int(np.bincount(pl).argmax())]
        g['coarse'] = int(par['coarse']) if par is not None else int(ni)
        g['radius_incl'] = par['radius_incl'] if par is not None else float('nan')
        g['n_rejected'] = 0; g['n_adapt_rejected'] = 0; g['n_merged'] = 1
        geoms.append(g); newfine[sidx] = ni
    newfine[fine < 0] = fine[fine < 0]
    return newfine, geoms


def cluster_chunk_fine(waves, res_abs, W, nmean, coarse_mg, mask, sr, method="gmm",
                       fine_kappa=40.0, fine_dedup=5.0, fine_mg=40, pca_k=6, max_sub=8, basis=None,
                       n_grid=40, incl_k=3.0, incl_assign=False, no_noise=False, shed=None, cone_channel_k=0.0, split_var_margin=0.0,
                       energy_band=False, eband_width=0.45, eband_overlap=0.2, eband_confound=0.4, eband_min_span=0.6, eband_min_band=60, eband_low_assign=0.0,
                       var_split=0.0, var_split_depth=4,
                       dipsplit=True, dip_dim=4, dip_alpha=0.01, dip_min=40, dip_realign=True,
                       nudge_split=True, nudge_max=3, nudge_amp_pct=40.0, nudge_min_channels=4, nudge_alpha=0.01,
                       rkk_dims=6, rkk_max=50, rkk_realign=True, rkk_realign_iters=2, rkk_delete=True, merge_corr=0.0, merge_method="template", sliding_nwin=14, cfiber_gate=False, cfiber_q=0.90,
                       profile_thr=None, profile_floor_pct=90.0, profile_min_n=120,
                       resplit_passes=0, resplit_residual_thr=0.08, resplit_topch=3, resplit_min_reduction=0.20, resplit_min_n=10, resplit_merge_corr=0.99,
                       resplit_detrend_episode=False, resplit_detrend_win=90.0, resplit_detrend_min_n=100,
                       refrac_ms=0.0, refrac_thr=0.3, refrac_min_exp=5.0, refrac_censor_ms=0.0,
                       emit_candidates=False, candidates_out=None,
                       deadapt=False, deadapt_min_corr=0.2,
                       adapt_clean=False, adapt_z=3.0, adapt_isi_ms=10.0,
                       adapt_clean_corr=0.4, adapt_clean_snr=0.5, adapt_taumax=0.5,
                       collision_flag=False, collision_gain=0.09, collision_shift=8,
                       quality_metrics=False, quality_dims=10):
    """Coarse fibers (for linking), refined WITHIN the chunk into sub-units so each
    fiber keeps its own geometry per chunk.  Split = method ('gmm' BIC mixture |
    'fiber' mean-shift | 'none'), then an optional DipSplit pass (dip test on lowered
    dims) catches bimodal clusters the BIC penalty left merged.  Each final cluster is
    re-seeded as a fiber: own trajectory + per-fiber inclusion radius rebuilt on it.
    deadapt=True: de-adapt each coarse fiber's amplitudes (EWMA-τ) before splitting,
    so RS fibers don't get carved into energy bands (the 'before' placement)."""
    coarse = cluster_chunk(waves, W, nmean, min_group=coarse_mg)
    ct0 = float(res_abs.min()); ct1 = float(res_abs.max())   # chunk time bounds for rate/presence
    fine = np.full(len(waves), -1, int); geoms = []; nid = 0
    for cf in np.unique(coarse[coarse >= 0]):
        cidx = np.flatnonzero(coarse == cf); wcf = waves[cidx]
        if deadapt:
            try:
                from . import fiber_adapt as _fa
            except ImportError:
                import fiber_adapt as _fa
            wsplit, _ = _fa.deadapt(wcf, res_abs[cidx], W, nmean, mask, sr, min_corr=deadapt_min_corr)
        else:
            wsplit = wcf
        groups = None
        if energy_band:                                # confound-gated energy-band split; None => not confounded, fall through
            groups = _energy_band_split(wsplit, mask, eband_width, eband_overlap, pca_k, max_sub,
                                        eband_min_band, eband_confound, eband_min_span, eband_low_assign)
        if groups is not None:
            pass
        elif method == "none":
            groups = [np.arange(len(cidx))]
        elif method == "fiber":
            sub = cluster_chunk(wsplit, W, nmean, min_group=fine_mg, kappa=fine_kappa, dedup_deg=fine_dedup)
            groups = ([np.arange(len(cidx))] if (sub < 0).all()
                      else [np.flatnonzero(sub == s) for s in np.unique(sub[sub >= 0])])
        elif method == "rkk":
            if rkk_realign:                                # per-cluster realign EM loop
                sub = _rkk_realign(wsplit, mask, rkk_dims, rkk_max, fine_mg, rkk_realign_iters, delete=rkk_delete, basis=basis)
            else:                                          # legacy: one parent realign, fixed features
                Fc = _fpca.cluster_features(fl.realign(wsplit), basis, realign=False, dims=rkk_dims) if basis is not None else None
                if Fc is None:
                    wc = fl.realign(wsplit)[:, mask, :].reshape(len(cidx), -1); wc = wc - wc.mean(0)
                    Uc, Sc, _ = np.linalg.svd(wc, full_matrices=False); Fc = Uc[:, :rkk_dims] * Sc[:rkk_dims]
                sub = _rkk(Fc, max_clusters=rkk_max, min_size=fine_mg, seed=42, delete=rkk_delete)
            groups = [np.flatnonzero(sub == s) for s in np.unique(sub)]
        else:
            sub = gmm_split(wsplit, pca_k=pca_k, max_sub=max_sub, mask=mask, basis=basis)
            groups = [np.flatnonzero(sub == s) for s in np.unique(sub)]
        if dipsplit and _HAVE_DIP:
            newg = []
            for grp in groups:                       # PCA each GROUP (within-unit variance)
                if len(grp) < 2 * dip_min:
                    newg.append(grp); continue
                if dip_realign:                      # realign EACH node to its own median (per step)
                    pieces = _dipsplit_realign(wcf[grp], mask, dip_dim, dip_min, dip_alpha, basis=basis)
                else:                                # legacy: one parent realign, fixed features
                    Fg = _fpca.cluster_features(fl.realign(wcf[grp]), basis, realign=False, dims=dip_dim) if basis is not None else None
                    if Fg is None:
                        wg = fl.realign(wcf[grp])[:, mask, :].reshape(len(grp), -1); wg = wg - wg.mean(0)
                        Ug, Sg, _ = np.linalg.svd(wg, full_matrices=False); Fg = Ug[:, :dip_dim] * Sg[:dip_dim]
                    pieces = _dipsplit_rec(Fg, np.arange(len(grp)), dip_min, dip_alpha)
                newg += [grp[piece] for piece in pieces]
            groups = newg
        if split_var_margin > 0 and len(groups) > 1:
            # accept the split only if it lowers the spike-weighted mean per-channel
            # RESIDUAL variance by >= margin (real shape sub-units do; energy splits don't)
            sub = np.full(len(cidx), -1, int)
            for gi, grp in enumerate(groups):
                sub[grp] = gi
            _, _, red = ft.split_meanvar(wcf, sub, W, nmean, mask, n_grid=n_grid, min_n=fine_mg)
            if red < split_var_margin:
                groups = [np.arange(len(cidx))]   # reject: insufficient variance reduction
        if var_split > 0:
            # auto-split fibers whose per-channel residual profile is peaked,
            # using the per-channel residual variance itself as the stop criterion
            vmargin = split_var_margin if split_var_margin > 0 else 0.05
            newg = []
            for grp in groups:
                pieces = (_variance_split(wcf[grp], W, nmean, mask, n_grid, var_split,
                                          vmargin, fine_mg, rkk_dims, max_depth=var_split_depth)
                          if len(grp) >= 2 * fine_mg else [np.arange(len(grp))])
                newg += [grp[pc] for pc in pieces]
            groups = newg
        if nudge_split:                                  # gated: split temporally-offset overlaid units
            gate = [(grp,) + (_amp_spread(wcf[grp], mask) if len(grp) >= 2 * fine_mg else (np.inf, 0))
                    for grp in groups]
            finite = [a for _, a, _ in gate if np.isfinite(a)]
            amp_thr = float(np.percentile(finite, nudge_amp_pct)) if finite else 0.0
            newg = []
            for grp, amp, nch in gate:                   # low amplitude + many channels = the condition
                if amp <= amp_thr and nch >= nudge_min_channels and len(grp) >= 2 * fine_mg:
                    newg += [grp[pc] for pc in _nudge_split(wcf[grp], mask, dip_dim, fine_mg,
                                                            nudge_alpha, nudge_max)]
                else:
                    newg.append(grp)
            groups = newg
        for grp in groups:
            g0 = len(grp)
            if g0 < fine_mg:
                if shed is not None: shed['small_group'] = shed.get('small_group', 0) + g0
                continue
            sidx = cidx[grp]; rad = float('nan'); rej = 0; rej_incl = 0; incl_rej = None
            if incl_k > 0 and len(sidx) >= 20:
                w_al = fl.realign(waves[sidx])
                Xg = (w_al[:, mask, :].reshape(len(sidx), -1) - nmean) @ W
                grid, D = ft.trajectory(Xg); rr = np.linalg.norm(Xg, axis=1)
                resid = np.linalg.norm(Xg - rr[:, None] * ft.predict_many((grid, D), rr), axis=1)
                med = float(np.median(resid)); mad = 1.4826 * float(np.median(np.abs(resid - med)))
                rad = med + incl_k * mad; keep = resid <= rad; rej = int((~keep).sum()); rej_incl = rej
                if incl_assign:
                    incl_rej = sidx[~keep]            # remember the inclusion tail (good spikes) to keep in the sort
                sidx = sidx[keep]
                if len(sidx) < fine_mg:
                    if shed is not None: shed['small_core'] = shed.get('small_core', 0) + g0
                    continue
            if cone_channel_k > 0 and len(sidx) >= 40:
                # tighten the cone PER CHANNEL: drop spikes that are residual outliers
                # on the discriminative (high-residual-variance) channels — peels
                # channel-localized contaminants the global norm averages away.
                prof = ft.channel_residual_profile(waves[sidx], W, nmean, mask, n_grid=n_grid)
                disc = prof['per_channel'] >= np.percentile(prof['per_channel'], 70)
                if disc.any():
                    ed = prof['per_spike_channel'][:, disc]
                    cmed = np.median(ed, 0); cmad = 1.4826 * np.median(np.abs(ed - cmed), 0) + 1e-9
                    keep2 = ((ed - cmed) / cmad).max(1) <= cone_channel_k
                    rcone = int((~keep2).sum()); rej += rcone; sidx = sidx[keep2]
                    if shed is not None: shed['cone'] = shed.get('cone', 0) + rcone
                    if len(sidx) < fine_mg:
                        if shed is not None: shed['small_core'] = shed.get('small_core', 0) + g0
                        continue
            arej = 0
            if adapt_clean and len(sidx) >= 40:
                try:
                    from . import fiber_adapt as _fa
                except ImportError:
                    import fiber_adapt as _fa
                z, ai = _fa.adaptation_residual(waves[sidx], res_abs[sidx], W, nmean, mask, sr)
                # only act where a fast adaptation law is real (gate on corr/snr/tau)
                if (abs(ai['corr']) >= adapt_clean_corr and ai['snr'] >= adapt_clean_snr
                        and ai['tau'] <= adapt_taumax):
                    t = res_abs[sidx].astype(float) / sr; o = np.argsort(t)
                    isi = np.full(len(sidx), 1e9); isi[o[1:]] = np.diff(t[o]) * 1000.0
                    akeep = ~((isi < adapt_isi_ms) & (z > adapt_z))   # high energy at short ISI = impossible
                    arej = int((~akeep).sum()); sidx = sidx[akeep]
                    if shed is not None: shed['adapt'] = shed.get('adapt', 0) + arej
                    if len(sidx) < fine_mg:
                        if shed is not None: shed['small_core'] = shed.get('small_core', 0) + g0
                        continue
            fine[sidx] = nid
            if shed is not None and rej_incl:
                _k = 'incl_tail_kept' if incl_assign else 'incl_tail_dropped'
                shed[_k] = shed.get(_k, 0) + rej_incl
            if incl_rej is not None and len(incl_rej):
                fine[incl_rej] = nid                  # inclusion-tail kept in the sort; geometry below stays on the pure core
            g = fiber_geom(waves[sidx], res_abs[sidx], W, nmean, mask, sr, n_grid, chunk_t0=ct0, chunk_t1=ct1)
            g['coarse'] = int(cf); g['radius_incl'] = rad; g['n_rejected'] = rej
            g['n_adapt_rejected'] = arej
            geoms.append(g); nid += 1
    # ── refinement: consolidate fragments of one unit. 'template' merges by
    #    mean-template correlation (amplitude-weighted, fast). 'sliding' compares
    #    direction profiles in a sliding window over radius, weighted by per-window
    #    angular concentration -> energy-resolved, robust to low-energy noise;
    #    use merge_corr ~0.90 for sliding, ~0.95 for template. ──
    # ── Block A: consolidate fragments by template / sliding-direction correlation.
    #    'template' = mean-template correlation (fast); 'sliding' = direction profile
    #    in a sliding radius window (energy-resolved).  merge_corr ~0.95 / ~0.90. ──
    if resplit_passes == 0 and merge_corr and merge_method in ("template", "sliding") and len(geoms) > 1:
        Kg = len(geoms)
        if merge_method == "sliding":
            Xs = [(fl.realign(waves[np.flatnonzero(fine == u)])[:, mask, :].reshape(-1, len(mask) * waves.shape[2]) - nmean) @ W
                  for u in range(Kg)]
            edges = _sliding_pairs(Xs, n_win=sliding_nwin, min_cos=merge_corr)
        else:
            T = np.array([g['template'].ravel() for g in geoms]); T = T - T.mean(1, keepdims=True)
            T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-12); C = T @ T.T
            edges = list(zip(*np.where(np.triu(C, 1) > merge_corr)))
        if cfiber_gate:
            edges = _cfiber_edge_filter(edges, fine, waves, mask, q=cfiber_q)
        fine, geoms = _apply_edges(fine, geoms, edges, waves, res_abs, W, nmean, mask, sr, n_grid, ct0, ct1)
    # ── Block B: same-neuron grouping by energy-resolved DIRECTION PROFILE d(r).
    #    Direction is the validated same-neuron signal (AUC ~0.98 same-fiber-halves
    #    vs distinct fibers); curvature/tangent add noise and are NOT used.  Threshold
    #    defaults to the within-fiber (same-neuron) floor.  emit_candidates=True writes
    #    proposals WITHOUT merging (curation); merge_method='profile' applies them.
    #    Runs AFTER Block A, so it can review/merge already-consolidated fibers. ──
    if resplit_passes == 0 and (merge_method == "profile" or emit_candidates) and len(geoms) > 1:
        Kg = len(geoms)
        Xs = [(fl.realign(waves[np.flatnonzero(fine == u)])[:, mask, :].reshape(-1, len(mask) * waves.shape[2]) - nmean) @ W
              for u in range(Kg)]
        trj = [ft.trajectory(X) for X in Xs]
        rng_p = [np.percentile(np.linalg.norm(X, axis=1), [15, 85]) for X in Xs]
        sz = [len(X) for X in Xs]
        thr_p = profile_thr if profile_thr is not None else _same_fiber_floor(Xs, profile_floor_pct, profile_min_n)
        cand = []
        for ai in range(Kg):
            if sz[ai] < profile_min_n:            # unreliable trajectory -> not a merge anchor
                continue
            for bi in range(ai + 1, Kg):
                if sz[bi] < profile_min_n:
                    continue
                d = _profile_dir_dist(trj[ai], rng_p[ai], trj[bi], rng_p[bi])
                if d is not None and d < thr_p:
                    cand.append((ai, bi, float(d)))
        cand.sort(key=lambda z: z[2])
        if candidates_out is not None:
            candidates_out.extend((ai, bi, d, thr_p) for ai, bi, d in cand)
        if merge_method == "profile" and not emit_candidates:
            edges = [(ai, bi) for ai, bi, _ in cand]
            if refrac_ms and refrac_ms > 0 and edges:
                # curation-INDEPENDENT veto: a profile-merge is PERMANENT (relink/defrag
                # only ever merge), so block one whose two in-chunk trains coincide at
                # chance level (two neurons, no refractory dip).  Power-aware: abstains
                # at low rate, so it only ever removes a false merge.
                refr = cg.refrac_samples(refrac_ms, sr)
                cens = cg.refrac_samples(refrac_censor_ms, sr)
                dur = float(res_abs.max() - res_abs.min()) if len(res_abs) else 0.0

                def _two_cells(ai, bi):
                    ta = np.sort(res_abs[np.flatnonzero(fine == ai)].astype(np.int64))
                    tb = np.sort(res_abs[np.flatnonzero(fine == bi)].astype(np.int64))
                    g = cg.refractory_gate(ta, tb, dur, refr, thr=refrac_thr,
                                           min_exp=refrac_min_exp, censor=cens)
                    return g["verdict"] == "veto"

                edges = [(ai, bi) for ai, bi in edges if not _two_cells(ai, bi)]
            fine, geoms = _apply_edges(fine, geoms, edges,
                                       waves, res_abs, W, nmean, mask, sr, n_grid, ct0, ct1)
    # ── nearest-neighbour isolation in the validated direction-profile space:
    #    per fiber, the closest other fiber and that distance (free; reuses the
    #    stored trajectory + radius range).  nn_dist is one isolation number AND
    #    the top merge candidate; small nn_dist on a clean fiber = over-split. ──
    if len(geoms) > 1:
        K = len(geoms)
        for i in range(K):
            best = (float('inf'), -1)
            for j in range(K):
                if j == i:
                    continue
                d = _profile_dir_dist((geoms[i]['grid'], geoms[i]['dir']), geoms[i]['rrange'],
                                      (geoms[j]['grid'], geoms[j]['dir']), geoms[j]['rrange'])
                if d is not None and d < best[0]:
                    best = (d, j)
            geoms[i]['nn_dist'] = float(best[0]) if best[1] >= 0 else float('nan')
            geoms[i]['nn_local'] = int(best[1])
    # ── optional L-ratio / isolation distance (Schmitzer-Torbert) in PCA-reduced
    #    whitened space.  O(N*K); behind a flag.  Field-standard quality metrics. ──
    if quality_metrics and len(geoms) > 1:
        from scipy.stats import chi2
        Xall = (fl.realign(waves)[:, mask, :].reshape(len(waves), -1) - nmean) @ W
        Xc = Xall - Xall.mean(0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        F = Xc @ Vt[:quality_dims].T; df = F.shape[1]
        for u in range(len(geoms)):
            mem = np.flatnonzero(fine == u)
            others = np.flatnonzero(fine >= 0); others = others[fine[others] != u]
            if len(mem) < df + 2 or len(others) == 0:
                continue
            mu = F[mem].mean(0); C = np.cov(F[mem].T) + 1e-6 * np.eye(df)
            try:
                Ci = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                continue
            dif = F[others] - mu; d2 = np.einsum('ij,jk,ik->i', dif, Ci, dif)
            geoms[u]['lratio'] = float(np.sum(1.0 - chi2.cdf(d2, df)) / len(mem))
            ds = np.sort(d2); geoms[u]['iso_dist'] = float(ds[min(len(mem), len(ds)) - 1])
    # ── Block C (opt-in): iterative residual-gated re-split (em_swap, target-channel residual) +
    #    correlation merge, to convergence.  klustakwik/gmm over-split PLUS this cleanup were validated
    #    to converge and drop within-cluster residual; replaces Block A/B when resplit_passes > 0. ──
    if resplit_passes > 0:
        for _rp in range(resplit_passes):
            src_fine = fine.copy(); src_geoms = geoms       # parents, for metadata carry after re-split
            nid = (int(fine.max()) + 1) if (fine >= 0).any() else 0
            nsplit = 0
            for u in np.unique(fine[fine >= 0]):
                loc = np.flatnonzero(fine == u)
                if loc.size < 2 * resplit_min_n:
                    continue
                w = fl.realign(waves[loc])                                  # primary alignment
                if _shape_residual(w) <= resplit_residual_thr:             # residual gate: skip tight fibers
                    continue
                sub = _em_swap(w, topk=resplit_topch, min_reduction=resplit_min_reduction, min_n=resplit_min_n,
                               episode=(_episode_position(res_abs[loc] / float(sr) * 1000.0, resplit_detrend_win)
                                        if resplit_detrend_episode else None),
                               detrend_min_n=resplit_detrend_min_n)
                if np.unique(sub).size >= 2:
                    for sv in np.unique(sub)[1:]:
                        fine[loc[sub == sv]] = nid; nid += 1
                    nsplit += 1
            if nsplit:
                fine, geoms = _rebuild_geoms(fine, waves, res_abs, W, nmean, mask, sr, n_grid, ct0, ct1, src_fine=src_fine, src_geoms=src_geoms)
            nmerge = 0
            if len(geoms) > 1:
                Xs = [(fl.realign(waves[np.flatnonzero(fine == u)])[:, mask, :].reshape(-1, len(mask) * waves.shape[2]) - nmean) @ W
                      for u in range(len(geoms))]
                edges = _sliding_pairs(Xs, n_win=sliding_nwin, min_cos=resplit_merge_corr)
                nmerge = len(edges)
                fine, geoms = _apply_edges(fine, geoms, edges, waves, res_abs, W, nmean, mask, sr, n_grid, ct0, ct1)
            if nsplit == 0 and nmerge == 0:
                break
    # ── collision flag: route recoverable collisions OUT of the noise cluster.
    #    The inclusion radius already sent collisions to noise; the two-template
    #    matching-pursuit gain (fiber_collision) separates recoverable collisions
    #    (gain > collision_gain) from junk.  They go to ONE dedicated collision
    #    cluster (heterogeneous by construction) and are NOT dual-assigned to a
    #    fiber pair (decomposition to a specific pair is unreliable on this data). ──
    if collision_flag and len(geoms) > 1:
        try:
            from . import fiber_collision as fcol
        except ImportError:
            import fiber_collision as fcol
        noise = np.flatnonzero(fine < 0)
        T = fcol.build_templates(waves, fine, mask=mask) if len(noise) >= 20 else {}
        if len(T) > 1:
            dic = fcol.whiten_atoms(T, W, nmean, mask=mask,
                                    shifts=range(-collision_shift, collision_shift + 1))
            gvec = fcol.decompose_batch(waves[noise], dic, W, nmean, mask)['gain']
            gains = {int(noise[j]): float(gvec[j]) for j in range(len(noise))}
            coll = [i for i, gn in gains.items() if gn > collision_gain]
            if len(coll) >= 20:
                try:
                    g = fiber_geom(waves[coll], res_abs[coll], W, nmean, mask, sr, n_grid, chunk_t0=ct0, chunk_t1=ct1)
                    g['coarse'] = -1; g['radius_incl'] = float('nan')
                    g['n_rejected'] = 0; g['n_merged'] = 0
                    g['collision'] = True; g['n_collision'] = len(coll)
                    g['collision_gain_median'] = float(np.median([gains[i] for i in coll]))
                    fine[np.array(coll)] = len(geoms); geoms.append(g)
                except Exception:
                    pass                                   # leave as noise if geom build fails
    # ── no-noise: sweep every REMAINING noise spike into ONE undefined fiber (a real cluster, not the
    #    noise cluster).  Heterogeneous by construction (rejected inclusion tails / coarse rejects / junk);
    #    kept in the sort so it is cleaned in subsequent steps rather than dropped.  Off by default. ──
    if no_noise:
        noise = np.flatnonzero(fine < 0)
        if noise.size >= 8:
            try:
                g = fiber_geom(waves[noise], res_abs[noise], W, nmean, mask, sr, n_grid, chunk_t0=ct0, chunk_t1=ct1)
                g['coarse'] = -1; g['radius_incl'] = float('nan'); g['n_rejected'] = 0; g['n_merged'] = 0
                g['undefined'] = True; g['n_undefined'] = int(noise.size)
                fine[noise] = len(geoms); geoms.append(g)
            except Exception:
                pass                                       # geom build failed: leave as noise
    return fine, geoms


def _apply_edges(fine, geoms, edges, waves, res_abs, W, nmean, mask, sr, n_grid, ct0=None, ct1=None):
    """Union-find merge of fiber `edges`, rebuilding fine labels + per-unit geometry."""
    if not edges:
        return fine, geoms
    Kg = len(geoms); par = list(range(Kg))

    def _find(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for a, b in edges:
        ra, rb = _find(int(a)), _find(int(b))
        if ra != rb: par[rb] = ra
    roots = sorted({_find(x) for x in range(Kg)})
    newfine = np.full(len(waves), -1, int); newgeoms = []
    for ni, r in enumerate(roots):
        members = [k for k in range(Kg) if _find(k) == r]
        sidx = np.flatnonzero(np.isin(fine, members))
        newfine[sidx] = ni
        g = fiber_geom(waves[sidx], res_abs[sidx], W, nmean, mask, sr, n_grid, chunk_t0=ct0, chunk_t1=ct1)
        g['coarse'] = geoms[r]['coarse']; g['radius_incl'] = geoms[r]['radius_incl']
        g['n_rejected'] = sum(geoms[k]['n_rejected'] for k in members)
        g['n_merged'] = len(members); newgeoms.append(g)
    return newfine, newgeoms


def _profile_dir_dist(t1, rr1, t2, rr2, n=7):
    """Mean (1 - cos) between two fibers' direction profiles d(r), evaluated on
    their OVERLAPPING energy range (interpolation, not extrapolation).  Returns
    None if the energy ranges don't overlap.  Direction (0th order) is the
    validated same-neuron signal (AUC ~0.98 same-fiber-halves vs different
    fibers); tangent/curvature were tested and add noise, so are NOT used."""
    lo = max(rr1[0], rr2[0]); hi = min(rr1[1], rr2[1])
    if hi - lo < 1e-6:
        return None
    rg = np.linspace(lo, hi, n)

    def prof(t):
        D = np.array([ft.predict(t, float(r)) for r in rg])
        return D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)
    D1, D2 = prof(t1), prof(t2)
    return float(np.mean(1.0 - np.sum(D1 * D2, axis=1)))


def _same_fiber_floor(Xs, pct=90.0, min_n=120, default=0.18, seed=0):
    """Calibrate the profile-merge threshold from the within-fiber (same-neuron)
    distance: split each large fiber's spikes in half and measure the direction-
    profile distance between the halves.  The pct-percentile of that distribution
    is the distance two genuinely-same-neuron fragments sit within, so it is the
    principled merge threshold.  Falls back to `default` if too few fibers are
    large enough to split reliably."""
    rng = np.random.default_rng(seed); ds = []
    for X in Xs:
        if len(X) < min_n:
            continue
        idx = np.arange(len(X)); rng.shuffle(idx); h = len(idx) // 2
        a, b = idx[:h], idx[h:]
        ra, rb = np.linalg.norm(X[a], axis=1), np.linalg.norm(X[b], axis=1)
        d = _profile_dir_dist(ft.trajectory(X[a]), np.percentile(ra, [15, 85]),
                              ft.trajectory(X[b]), np.percentile(rb, [15, 85]))
        if d is not None:
            ds.append(d)
    return float(np.percentile(ds, pct)) if len(ds) >= 3 else default


def _sliding_pairs(Xs, n_win=14, min_spikes=15, min_shared=2, min_cos=0.90):
    """Sliding window over radius: per-fragment mean direction in each radius
    window, compared window-by-window weighted by spike count x angular
    concentration (down-weights noisy low-energy windows). Returns (i,j) pairs
    whose weighted mean cosine over shared windows exceeds min_cos."""
    K = len(Xs); p = Xs[0].shape[1]
    rs = [np.linalg.norm(X, axis=1) for X in Xs]; ds = [X / (r[:, None] + 1e-12) for X, r in zip(Xs, rs)]
    allr = np.concatenate(rs); edges = np.linspace(np.percentile(allr, 1), np.percentile(allr, 99), n_win + 1)
    prof = np.zeros((K, n_win, p)); cnt = np.zeros((K, n_win)); conc = np.zeros((K, n_win))
    for k in range(K):
        wi = np.clip(np.searchsorted(edges, rs[k]) - 1, 0, n_win - 1)
        for w in range(n_win):
            m = wi == w
            if int(m.sum()) >= min_spikes:
                mr = ds[k][m].mean(0); R = np.linalg.norm(mr)
                prof[k, w] = mr / (R + 1e-12); cnt[k, w] = int(m.sum()); conc[k, w] = R
    pairs = []
    for i in range(K):
        for j in range(i + 1, K):
            sh = (cnt[i] > 0) & (cnt[j] > 0)
            if sh.sum() < min_shared: continue
            wts = np.minimum(cnt[i], cnt[j])[sh] * conc[i][sh] * conc[j][sh]
            if wts.sum() < 1e-9: continue
            coss = (prof[i][sh] * prof[j][sh]).sum(1)
            if (wts * coss).sum() / wts.sum() > min_cos: pairs.append((i, j))
    return pairs


def geoms_from_labels(waves, res_abs, lab, W, nmean, mask, sr, n_grid=40):
    geoms = []
    for f in np.unique(lab[lab >= 0]):
        idx = np.flatnonzero(lab == f)
        g = fiber_geom(waves[idx], res_abs[idx], W, nmean, mask, sr, n_grid)
        g['coarse'] = int(f); g['radius_incl'] = float('nan'); g['n_rejected'] = 0
        geoms.append(g)
    return geoms


def link_chunks(ext_idx, ext_lab, min_anchor=8, frac=0.5):
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[rb] = ra
    nC = len(ext_idx)
    for c in range(nC):
        for l in {int(x) for x in ext_lab[c] if x >= 0}: find((c, l))
    for c in range(nC - 1):
        A = {int(g): int(l) for g, l in zip(ext_idx[c],     ext_lab[c])     if l >= 0}
        B = {int(g): int(l) for g, l in zip(ext_idx[c + 1], ext_lab[c + 1]) if l >= 0}
        shared = set(A) & set(B)
        if not shared: continue
        ab = defaultdict(Counter); ba = defaultdict(Counter)
        for s in shared: ab[A[s]][B[s]] += 1; ba[B[s]][A[s]] += 1
        for f, row in ab.items():
            g, cnt = row.most_common(1)[0]
            if cnt < min_anchor or cnt < frac * sum(row.values()): continue
            f2, cnt2 = ba[g].most_common(1)[0]
            if f2 != f or cnt2 < frac * sum(ba[g].values()): continue
            union((c, f), (c + 1, g))
    roots = {}; gid = {}
    for c in range(nC):
        for l in {int(x) for x in ext_lab[c] if x >= 0}:
            r = find((c, l)); roots.setdefault(r, len(roots)); gid[(c, l)] = roots[r]
    return gid, len(roots)


def link_continuity(gid, nglob, depth, sig, *, depth_gate=14.0, sig_thr=0.6,
                    max_gap=2, use_sig=True):
    """Drift-predicted, signature-gated continuity fallback that runs AFTER the
    overlap-anchor backbone (`link_chunks`) to recover fibers too sparse to share
    enough overlap spikes.  Coherent drift is estimated from the multi-chunk
    globals the backbone already linked (per-chunk median depth step); a global
    that *ends* is bridged to one that *begins* within `max_gap` chunks only if
    the earlier track's drift-predicted depth matches the later track's start
    (within `depth_gate` per chunk of gap) AND their signatures agree
    (cosine >= `sig_thr`).  The signature gate blocks identity swaps when a
    different unit appears on a vanished unit's drift path; with `use_sig=False`
    (ablation) those swaps are wrongly merged.  Bridges may REFUSE, so genuine
    discontinuities are preserved.

    gid/nglob: output of `link_chunks`.  depth: {(chunk, localid): float} drift
    coordinate (e.g. energy-weighted channel centroid).  sig: {(chunk, localid):
    vector} drift-robust template signature.  Returns (gid', nglob')."""
    from collections import defaultdict
    members = defaultdict(list)
    for (c, l), g in gid.items():
        members[g].append((c, l))
    step = defaultdict(list)                                   # coherent drift from linked multi-chunk globals
    for g, ms in members.items():
        byc = {}
        for (c, l) in ms:
            byc.setdefault(c, depth[(c, l)])
        cs = sorted(byc)
        for a, b in zip(cs[:-1], cs[1:]):
            if b == a + 1:
                step[b].append(byc[b] - byc[a])
    dstep = {c: float(np.median(v)) for c, v in step.items()}
    med = float(np.median([np.median(v) for v in step.values()])) if step else 0.0
    ends, starts = {}, {}
    for g, ms in members.items():
        ms = sorted(ms)
        ends[g] = (ms[-1][0], depth[ms[-1]], np.asarray(sig[ms[-1]], float))
        starts[g] = (ms[0][0], depth[ms[0]], np.asarray(sig[ms[0]], float))
    parent = list(range(nglob))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    nbridge = 0
    for gb in sorted(starts, key=lambda g: starts[g][0]):
        cb, zb, sb = starts[gb]; best = None
        for ge in ends:
            if find(ge) == find(gb):
                continue
            ce, ze, se = ends[ge]; gap = cb - ce
            if not (1 <= gap <= max_gap):
                continue
            pred = ze + sum(dstep.get(k, med) for k in range(ce + 1, cb + 1))
            scos = float(se @ sb / (np.linalg.norm(se) * np.linalg.norm(sb) + 1e-12))
            if abs(pred - zb) <= depth_gate * gap and (scos >= sig_thr or not use_sig):
                score = abs(pred - zb) - 5.0 * scos
                if best is None or score < best[0]:
                    best = (score, ge)
        if best is not None:
            parent[find(gb)] = find(best[1]); nbridge += 1
    roots = {}; newg = {}
    for (c, l), g in gid.items():
        r = find(g); roots.setdefault(r, len(roots)); newg[(c, l)] = roots[r]
    return newg, len(roots)


def read_res(base, elec):
    return nio.read_res(base, elec, prefer=nio.prefer_canonical())

def open_spkD(base, elec, nsamp, nch):
    return nio.open_spkD(base, elec, nsamp, nch)

def fil_chunk_whitener(filmm, gch, s0, s1, spike_abs, nsamp, mask):
    # memmap path: reads only sampled baseline windows, never the whole span.
    return fl.chunk_whitener_mm(filmm, gch, s0, s1, spike_abs, mask=mask)


# ── chunk-level parallelism ──────────────────────────────────────────────────
# Each chunk is independent (its own whitener, clustering, geometry), so chunks
# are the natural coarse-grain parallel axis — the same flatten-the-independent-
# axis strategy kiloklustakwik uses for its CEM runs.  Workers reopen the .spkD
# and .fil memmaps from disk (cheap; OS page cache is shared) so no memmap or
# big array is pickled across the process boundary; only per-chunk spike indices
# cross it.  Results are independent of worker count and of completion order
# (cluster_chunk_fine is seeded and per-chunk), so jobs>1 is identical to serial.
_CTX = {}


def _init_chunk_worker(cfg):
    """Pool initializer: stash the static config and open the memmaps once per
    worker process.  Also runs (with a fresh dict) for the serial jobs==1 path."""
    _CTX.clear(); _CTX.update(cfg)
    if cfg.get("gpu"):
        _bk.use_gpu(True)
    _CTX["spk"], _ = nio.open_spk(cfg["base"], cfg["elec"], cfg["nsamp"], cfg["nchan"],
                                  prefer=nio.prefer_derived())
    _CTX["filmm"] = nio.open_signal(cfg["fil"], cfg["ntotal"])


def _process_chunk(task):
    """Cluster one chunk.  task = (c, ext, res_e); returns (c, ext, lab, geoms,
    cand, shed).  Reads everything else (memmaps + params) from _CTX."""
    c, ext, res_e = task
    ctx = _CTX; kw = ctx["cf"]
    waves = np.asarray(ctx["spk"][ext], dtype=float)
    s0 = int(res_e.min()) - ctx["nsamp"]; s1 = int(res_e.max()) + ctx["nsamp"] + 1
    W, nmean, _ = fil_chunk_whitener(ctx["filmm"], ctx["gch"], s0, s1, res_e, ctx["nsamp"], ctx["mask"])
    cand = []
    sd = {}
    lab, geoms = cluster_chunk_fine(waves, res_e, W, nmean, ctx["min_group"], ctx["mask"], ctx["sr"],
                                    candidates_out=cand, shed=sd, **kw)
    sd["_unsorted_ext"] = int((lab < 0).sum())
    return c, ext, lab, geoms, cand, sd


def add_core_arguments(ap):
    """Register every fiber-session clusterer argument on `ap` (after sy.add_session_args).
    Shared by main() and the stochastic harness so both expose an identical clusterer."""
    ap.add_argument("--chunk-min", "--chunk-minutes", type=float, default=12.0); ap.add_argument("--overlap-min", type=float, default=4.0)
    ap.add_argument("--min-group", type=int, default=200, help="COARSE min spikes/fiber (for linking)")
    ap.add_argument("--fine-method", choices=["gmm","rkk","fiber","none"], default="gmm")
    ap.add_argument("--rkk-dims", type=int, default=6); ap.add_argument("--rkk-max", type=int, default=50)
    ap.add_argument("--rkk-realign", dest="rkk_realign", action="store_true", default=True,
                    help="interleave rkk (CEM) with per-cluster realignment (per-step; default on)")
    ap.add_argument("--no-rkk-realign", dest="rkk_realign", action="store_false",
                    help="legacy: one parent realign + fixed features for the rkk split")
    ap.add_argument("--rkk-realign-iters", type=int, default=2,
                    help="cluster<->realign passes in the rkk realign loop")
    ap.add_argument("--rkk-delete", dest="rkk_delete", action="store_true", default=True,
                    help="rkk (CEM) culls sub-min-group sub-clusters during the per-fiber fine split (default on)")
    ap.add_argument("--no-rkk-delete", dest="rkk_delete", action="store_false",
                    help="keep small non-singular rkk sub-clusters -- session should OVER-cluster, leaving the cull "
                         "to refine; use to stop session shedding fragments into the residual/artifact bin")
    ap.add_argument("--merge-corr", type=float, default=0.0, help="consolidate fibers above this (0=off; 0.95 template / 0.90 sliding)")
    ap.add_argument("--resplit-passes", type=int, default=0,
                    help="iterative residual-gated re-split (em_swap on target-channel residual) + correlation merge; "
                         "0=off.  Replaces the Block-A/B consolidation when >0.")
    ap.add_argument("--resplit-residual-thr", type=float, default=0.08,
                    help="re-split only fibers whose amplitude-scaled max residual (+-8 @ RMS peak) exceeds this "
                         "(~0.08 for stderiv, ~0.15 for standard waveforms)")
    ap.add_argument("--resplit-topch", type=int, default=3, help="channels fed to em_swap (top residual-variance)")
    ap.add_argument("--resplit-min-reduction", type=float, default=0.20,
                    help="keep an em_swap split only if it cuts target-channel variance by >= this")
    ap.add_argument("--resplit-merge-corr", type=float, default=0.99, help="correlation merge threshold inside the loop")
    ap.add_argument("--resplit-detrend-episode", action="store_true",
                    help="before each em_swap, strip the episode-position axis (the direction covarying with "
                         "spikes-after minus spikes-before in a +-90 ms window) from the residual, so the split "
                         "cannot cut a cell along its own temporal gradient and manufacture an asymmetric CCG")
    ap.add_argument("--resplit-detrend-win", type=float, default=90.0,
                    help="half-window (ms) for the episode-position count")
    ap.add_argument("--resplit-detrend-min-n", type=int, default=100,
                    help="skip the detrend below this many spikes -- the axis is a covariance estimate and is "
                         "unreliable on small groups")
    ap.add_argument("--cfiber-gate", action="store_true", help="veto Block-A fragment merges whose affine-invariant cfiber shape disagrees beyond the per-chunk within-fiber null (precision gate; threshold self-calibrated at --cfiber-q)")
    ap.add_argument("--cfiber-q", type=float, default=0.90, help="quantile of the within-fiber split-half cfiber null used as the --cfiber-gate veto threshold")
    ap.add_argument("--merge-method", choices=["template","sliding","profile"], default="template")
    ap.add_argument("--sliding-nwin", type=int, default=14)
    ap.add_argument("--profile-thr", type=float, default=None,
                    help="profile-merge direction-distance threshold; default = auto same-neuron floor")
    ap.add_argument("--profile-floor-pct", type=float, default=90.0,
                    help="percentile of within-fiber-half distances used as the auto threshold")
    ap.add_argument("--profile-min-n", type=int, default=120,
                    help="min spikes/fiber to be eligible for a profile merge (trajectory reliability)")
    ap.add_argument("--emit-merge-candidates", action="store_true",
                    help="write proposed same-neuron merges to <base>.merge_candidates.<elec>.tsv WITHOUT merging (curation)")
    ap.add_argument("--refrac-ms", type=float, default=0.0,
                    help="DEFAULT OFF. >0 gates each within-chunk profile-merge through a refractory "
                         "cross-correlogram veto: a merge whose two trains coincide at chance level "
                         "(two neurons, no dip) is blocked. Profile merges are permanent (relink/defrag "
                         "only ever merge), so this stops irreversible over-merges at the source. "
                         "Power-aware: ABSTAINS at low rate, only ever removes a false merge.")
    ap.add_argument("--refrac-thr", type=float, default=0.3, help="coincidence ratio above which the pair is 'two neurons' (default 0.3)")
    ap.add_argument("--refrac-min-exp", type=float, default=5.0, help="min expected coincidences for the refractory test to be powered (default 5)")
    ap.add_argument("--refrac-censor-ms", type=float, default=0.0, help="censor window (ms) dropping duplicate detections of one spike (default 0)")
    ap.add_argument("--deadapt", action="store_true", help="de-adapt (EWMA-tau) RS coarse fibers before splitting")
    ap.add_argument("--deadapt-min-corr", type=float, default=0.2)
    ap.add_argument("--adapt-clean", action="store_true", help="reject high-energy-at-short-ISI spikes on real fast adapters")
    ap.add_argument("--adapt-z", type=float, default=3.0); ap.add_argument("--adapt-isi-ms", type=float, default=10.0)
    ap.add_argument("--adapt-clean-corr", type=float, default=0.4); ap.add_argument("--adapt-clean-snr", type=float, default=0.5)
    ap.add_argument("--adapt-taumax", type=float, default=0.5)
    ap.add_argument("--collision-flag", action="store_true", help="route recoverable collisions from noise to a dedicated collision cluster")
    ap.add_argument("--collision-gain", type=float, default=0.09); ap.add_argument("--collision-shift", type=int, default=8)
    ap.add_argument("--quality-metrics", action="store_true", help="also compute L-ratio + isolation distance (O(N*K))")
    ap.add_argument("--quality-dims", type=int, default=10, help="PCA dims for L-ratio/isolation Mahalanobis")
    ap.add_argument("--pca-k", type=int, default=6); ap.add_argument("--max-sub", type=int, default=8)
    ap.add_argument("--inclusion-k", type=float, default=3.0, help="per-fiber radius = median+k*MAD of residuals; 0 disables")
    ap.add_argument("--no-noise", dest="no_noise", action="store_true", default=False,
                    help="sweep every remaining noise spike (below the inclusion radius / rejected / collision junk) into a "
                         "single UNDEFINED FIBER (one real cluster, not the noise cluster) rather than dropping it. For clean "
                         "stderiv data; the undefined fiber is cleaned/re-split in later steps.")
    ap.add_argument("--incl-assign-rejected", dest="incl_assign", action="store_true", default=False,
                    help="assign spikes beyond the per-fiber inclusion radius to that fiber (kept in the sort) instead "
                         "of dropping them to the unsorted bin.  Geometry/templates still use the pure core; this only "
                         "rescues the good high-amplitude tail spikes the radius would otherwise discard.")
    ap.add_argument("--energy-band", action="store_true", help="energy-band split: partition each ENERGY-CONFOUNDED coarse fiber into overlapping log10-energy bands, BIC-GMM per band (global features), relink by overlap-anchor; surfaces shape sub-units the drift axis masks")
    ap.add_argument("--eband-width", type=float, default=0.45, help="energy-band width in decades (default 0.45)")
    ap.add_argument("--eband-overlap", type=float, default=0.2, help="energy-band overlap in decades for overlap-anchor linking (default 0.2)")
    ap.add_argument("--eband-confound", type=float, default=0.4, help="only band a fiber when PC1 R^2 vs log-energy >= this (default 0.4)")
    ap.add_argument("--eband-min-span", type=float, default=0.6, help="only band a fiber spanning >= this many decades (default 0.6)")
    ap.add_argument("--eband-min-band", type=int, default=60, help="min spikes per band to cluster (default 60)")
    ap.add_argument("--eband-low-assign", type=float, default=0.0, help="fraction of the energy range (from the bottom) made ASSIGNMENT-ONLY: in that low-SNR floor the direction is noise, so its spikes are assigned to units from the bands above instead of independently split (default 0.0 = split every band)")
    ap.add_argument("--cone-channel-k", type=float, default=0.0,
                    help="tighten the cone per channel: drop spikes that are residual outliers (>k MAD) "
                         "on the discriminative channels; 0 disables")
    ap.add_argument("--split-var-margin", type=float, default=0.0,
                    help="accept a within-fiber split only if it lowers the mean per-channel residual "
                         "variance by >= this fraction (e.g. 0.1); 0 accepts all splits")
    ap.add_argument("--var-split", type=float, default=0.0,
                    help="auto-split fibers whose per-channel residual profile is peaked: trigger when "
                         "max/median channel residual variance >= this ratio (e.g. 2.0); 0 disables. "
                         "Bisects on the high-variance channels, accepting only variance-reducing splits.")
    ap.add_argument("--var-split-depth", type=int, default=4,
                    help="max recursion depth for --var-split (max 2^depth sub-units per fiber)")
    ap.add_argument("--dipsplit", dest="dipsplit", action="store_true", default=True)
    ap.add_argument("--no-dipsplit", dest="dipsplit", action="store_false")
    ap.add_argument("--dip-dim", type=int, default=4); ap.add_argument("--dip-alpha", type=float, default=0.01)
    ap.add_argument("--dip-min", type=int, default=40)
    ap.add_argument("--dip-realign", dest="dip_realign", action="store_true", default=True,
                    help="realign each dipsplit node to its own median before splitting "
                         "(per-step alignment; default on)")
    ap.add_argument("--no-dip-realign", dest="dip_realign", action="store_false",
                    help="legacy: one parent realign + fixed features for the whole dipsplit recursion")
    ap.add_argument("--nudge-split", dest="nudge_split", action="store_true", default=True,
                    help="for low-amp clusters, split temporally-offset overlaid units by alignment "
                         "lag (similar-shape neurons a few samples apart that median realign merges); default on")
    ap.add_argument("--no-nudge-split", dest="nudge_split", action="store_false")
    ap.add_argument("--nudge-max", type=int, default=3, help="max +/- sample lag tested for offset overlays")
    ap.add_argument("--nudge-amp-pct", type=float, default=40.0,
                    help="only clusters below this template-amplitude percentile are nudge-split")
    ap.add_argument("--nudge-min-channels", type=int, default=4,
                    help="min signal channels for the broad-noise condition")
    ap.add_argument("--nudge-alpha", type=float, default=0.01, help="dip-test p for the lag-bimodality split")
    ap.add_argument("--fine-kappa", type=float, default=40.0)
    ap.add_argument("--fine-dedup-deg", type=float, default=5.0)
    ap.add_argument("--fine-min-group", type=int, default=40)
    ap.add_argument("--no-fine", action="store_true", help="coarse fibers only, no within-chunk refinement")
    ap.add_argument("--min-anchor", type=int, default=8)
    ap.add_argument("--no-link", action="store_true")
    ap.add_argument("--n-grid", type=int, default=40)
    ap.add_argument("--method", default="stderiv", help="extraction method tag in the .fibers filename")
    ap.add_argument("--no-cluster-basis", action="store_true",
                    help="ignore the global .pca basis for the fine-split shape features and use a "
                         "per-call local SVD (legacy behaviour)")
    ap.add_argument("--clu-stage", dest="clu_stage", default="fiber_session",
                    help="post-group stage tag for the clu: <base>.clu.<method>.<elec>[.<stage>] "
                         "(default 'fiber_session'); pass --clu-stage '' for an untagged .clu")
    ap.add_argument("--emit-hierarchy", dest="emit_hierarchy", action=argparse.BooleanOptionalAction, default=True,
                    help="emit the .clu/.clc/.clp microfiber triple (atoms = pre-link fine fragments, "
                         "fibers = linked global ids) via FiberHierarchy, instead of a flat .clu only. "
                         "--no-emit-hierarchy writes just the flat .clu (legacy). Ignored with --out.")
    ap.add_argument("--gpu", action="store_true", help="run the realign/whiten kernels on GPU (CuPy; needs the [gpu] extra)")
    ap.add_argument("--jobs", "-j", type=int, default=1,
                    help="parallel worker processes over chunks (default 1 = serial; chunks are independent)")
    ap.add_argument("--feature-align", dest="feature_align", choices=["xcorr", "centroid"], default=None,
                    help="feature-building alignment: xcorr (default) or centroid (pure, no refine -- "
                         "adds the trough-position-vs-asymmetry structure to the clustering/linking "
                         "features).  Does NOT touch committing alignment or fiber-realign.  Overrides "
                         "the FIBER_ALIGN env var.")
    ap.add_argument("--subsample", dest="subsample", action=argparse.BooleanOptionalAction, default=None,
                    help="enable (--subsample) or disable (--no-subsample) realign's per-spike "
                         "sub-sample (parabolic) refine in the feature build; default leaves the "
                         "FIBER_SUBSAMPLE env var / lever untouched (off).  Reaches pool workers.")
    ap.add_argument("--out", default=None)
    return ap


def build_cf(a, meth, cluster_basis):
    """Assemble the cluster_chunk_fine keyword dict from parsed args.  Extracted from main()
    so the stochastic harness configures the clusterer identically to the production path."""
    return dict(method=meth, fine_kappa=a.fine_kappa, fine_dedup=a.fine_dedup_deg,
        fine_mg=a.fine_min_group, pca_k=a.pca_k, max_sub=a.max_sub, n_grid=a.n_grid, basis=cluster_basis,
        incl_k=a.inclusion_k, incl_assign=a.incl_assign, no_noise=a.no_noise, cone_channel_k=a.cone_channel_k,
        energy_band=a.energy_band, eband_width=a.eband_width, eband_overlap=a.eband_overlap,
        eband_confound=a.eband_confound, eband_min_span=a.eband_min_span, eband_min_band=a.eband_min_band,
        eband_low_assign=a.eband_low_assign,
        split_var_margin=a.split_var_margin, var_split=a.var_split,
        var_split_depth=a.var_split_depth, dipsplit=a.dipsplit,
        dip_dim=a.dip_dim, dip_alpha=a.dip_alpha, dip_min=a.dip_min, dip_realign=a.dip_realign,
        nudge_split=a.nudge_split, nudge_max=a.nudge_max, nudge_amp_pct=a.nudge_amp_pct, nudge_min_channels=a.nudge_min_channels, nudge_alpha=a.nudge_alpha,
        rkk_dims=a.rkk_dims, rkk_max=a.rkk_max, rkk_realign=a.rkk_realign, rkk_realign_iters=a.rkk_realign_iters, rkk_delete=a.rkk_delete,
        merge_corr=a.merge_corr, merge_method=a.merge_method, sliding_nwin=a.sliding_nwin,
        resplit_passes=a.resplit_passes, resplit_residual_thr=a.resplit_residual_thr, resplit_topch=a.resplit_topch, resplit_min_reduction=a.resplit_min_reduction, resplit_merge_corr=a.resplit_merge_corr,
        resplit_detrend_episode=a.resplit_detrend_episode, resplit_detrend_win=a.resplit_detrend_win, resplit_detrend_min_n=a.resplit_detrend_min_n,
        cfiber_gate=a.cfiber_gate, cfiber_q=a.cfiber_q,
        profile_thr=a.profile_thr, profile_floor_pct=a.profile_floor_pct,
        profile_min_n=a.profile_min_n, emit_candidates=a.emit_merge_candidates,
        refrac_ms=a.refrac_ms, refrac_thr=a.refrac_thr,
        refrac_min_exp=a.refrac_min_exp, refrac_censor_ms=a.refrac_censor_ms,
        deadapt=a.deadapt, deadapt_min_corr=a.deadapt_min_corr,
        adapt_clean=a.adapt_clean, adapt_z=a.adapt_z, adapt_isi_ms=a.adapt_isi_ms,
        adapt_clean_corr=a.adapt_clean_corr, adapt_clean_snr=a.adapt_clean_snr,
        adapt_taumax=a.adapt_taumax, collision_flag=a.collision_flag,
        collision_gain=a.collision_gain, collision_shift=a.collision_shift,
        quality_metrics=a.quality_metrics, quality_dims=a.quality_dims)


def main():
    ap = argparse.ArgumentParser(
        description="Cluster a session group into fibers. Reads <session>.yaml "
                    "(or <session>/<session>.yaml) for channels/sr/nChannels; "
                    "CLI flags override the YAML.")
    sy.add_session_args(ap)
    add_core_arguments(ap)
    a = ap.parse_args()
    P = "\u25b8 fiber-session"; IND = " " * (len(P) + 3)
    def log(m=""):  print(f"{P} \u00b7 {m}" if m else P)
    def det(k, v, w=13):  print(f"{IND}{k:<{w}} {v}")
    if a.feature_align:
        os.environ["FIBER_ALIGN"] = a.feature_align   # reach forked/spawned pool workers
        fl.set_feature_align(a.feature_align)           # this (parent) process
    if a.subsample is not None:
        os.environ["FIBER_SUBSAMPLE"] = "1" if a.subsample else "0"   # reach forked/spawned pool workers
        fl.set_realign_subsample(a.subsample)                          # this (parent) process
    _gpu_line = None
    if a.gpu:
        _on = _bk.use_gpu(True)
        _gpu_line = _bk.backend_name() + ("" if _on else "  (unavailable -> CPU)")
    cfg = sy.resolve_session_params(a.session, a.group, channels=a.channels, ntotal=a.ntotal,
                                    nchan=a.nchan, nsamp=a.nsamp, sr=a.sr)
    a.base = cfg["base"]; a.elec = a.group
    a.ntotal = cfg["ntotal"]; a.nchan = cfg["nchan"]; a.nsamp = cfg["nsamp"]; a.sr = cfg["sr"]
    gch = np.array(cfg["channels"], int)
    assert len(gch) == a.nchan, f"--channels has {len(gch)} entries, nchan={a.nchan}"
    mask = fl.build_masks(cfg["nsamp"], cfg["peak"]).full; p = len(mask) * a.nchan

    t0 = time.time()
    res = read_res(a.base, a.elec); nspk = len(res)
    spk, spkpath = open_spkD(a.base, a.elec, a.nsamp, a.nchan)
    assert spk.shape[0] == nspk, f".res {nspk} vs {spkpath} {spk.shape[0]}"
    filmm = nio.open_signal(f"{a.base}.fil", a.ntotal)
    log(f"group {a.elec} \u00b7 {a.method} \u00b7 {nspk:,} spikes")
    det("source", spkpath)
    det("recording", f"{filmm.shape[0]:,} samples \u00d7 {a.ntotal} ch")
    det("feature-align", fl.get_feature_align())
    det("realign", "sub-sample" if fl.realign_subsample() else "whole-sample")
    if _gpu_line: det("GPU", _gpu_line)
    if a.dipsplit and not _HAVE_DIP:
        det("note", "diptest not installed \u2014 --dipsplit skipped")

    chunk_s = a.chunk_min * 60.0 * a.sr; ov_s = a.overlap_min * 60.0 * a.sr
    t_min, t_max = int(res.min()), int(res.max())
    nchunks = int(np.ceil((t_max - t_min) / chunk_s))
    ext_idx = [np.array([], int)] * nchunks; ext_lab = [np.array([], int)] * nchunks
    chunk_geoms = [[] for _ in range(nchunks)]; chunk_tmin = [0.0] * nchunks
    chunk_candidates = [[] for _ in range(nchunks)]

    # ── build per-chunk tasks (small chunks skipped here, serially & cheaply) ──
    meth = "none" if a.no_fine else a.fine_method
    # SHAPE features for the fine GMM split: the GLOBAL ndm_pca basis (shared across chunks),
    # so a basis change (nFeatures, varimax) propagates; None -> per-call local SVD fallback.
    cluster_basis = None if a.no_cluster_basis else _fpca.read_cluster_basis(a.base, a.elec, a.method)
    if cluster_basis is not None:
        log(f"fine-split shape features: global basis '{a.method}' "
            f"({cluster_basis['evec'].shape[0]}ch x {cluster_basis['evec'].shape[1]}comp)")
    cf = build_cf(a, meth, cluster_basis)
    cfg = dict(base=a.base, elec=a.elec, fil=f"{a.base}.fil", ntotal=a.ntotal,
               nsamp=a.nsamp, nchan=a.nchan, sr=a.sr, min_group=a.min_group,
               gch=gch, mask=mask, cf=cf, gpu=a.gpu)

    tasks = []; ncore_of = {}
    for c in range(nchunks):
        lo_s = t_min + c * chunk_s; hi_s = t_min + (c + 1) * chunk_s
        chunk_tmin[c] = (lo_s - t_min) / a.sr / 60.0
        ext = np.flatnonzero((res >= lo_s - ov_s) & (res < hi_s + ov_s))
        ncore = int(((res[ext] >= lo_s) & (res[ext] < hi_s)).sum()); ncore_of[c] = ncore
        if len(ext) < 2 * a.min_group:
            print(f"{IND}chunk {c+1:>3}/{nchunks}   {ncore:>7,} core   \u2192  skipped (too few)"); continue
        tasks.append((c, ext, res[ext]))

    shed_total = {}

    def _store(result):
        c, ext, lab, geoms, cand, sd = result
        ext_idx[c] = ext; ext_lab[c] = lab; chunk_geoms[c] = geoms; chunk_candidates[c] = cand
        for k, v in sd.items():
            shed_total[k] = shed_total.get(k, 0) + int(v)
        print(f"{IND}chunk {c+1:>3}/{nchunks}   {ncore_of[c]:>7,} core   \u2192  {len(geoms):>4} fibers")

    jobs = max(1, int(a.jobs))
    if jobs == 1 or len(tasks) <= 1:
        log(f"clustering {len(tasks)} chunks")
        _init_chunk_worker(cfg)                       # serial: identical to the former inline loop
        for task in tasks:
            _store(_process_chunk(task))
    else:
        from concurrent.futures import ProcessPoolExecutor
        nworkers = min(jobs, len(tasks))
        log(f"clustering {len(tasks)} chunks on {nworkers} processes")
        with ProcessPoolExecutor(max_workers=nworkers,
                                 initializer=_init_chunk_worker, initargs=(cfg,)) as ex:
            for result in ex.map(_process_chunk, tasks):
                _store(result)

    # ── measurement: where genuine spikes leave the sort (-1), by cause ──
    if shed_total:
        sg = shed_total.get('small_group', 0); sc = shed_total.get('small_core', 0)
        itk = shed_total.get('incl_tail_kept', 0); itd = shed_total.get('incl_tail_dropped', 0)
        co = shed_total.get('cone', 0); ad = shed_total.get('adapt', 0)
        uns = shed_total.get('_unsorted_ext', 0)
        coarse = max(0, uns - (sg + sc + itd + co + ad))
        rows = [("small-group skip", sg), ("small-core skip", sc)]
        if itd: rows.append(("inclusion tail dropped", itd))
        if co:  rows.append(("cone", co))
        if ad:  rows.append(("adapt", ad))
        rows.append(("coarse-unassigned", coarse))
        lw = max(len(n) for n, _ in rows + [("total", 0)])
        nw = max(len(f"{v:,}") for _, v in rows + [("", uns)])
        if itk:
            log(f"inclusion tail kept in sort: {itk:,}")
        log("unsorted spikes by cause")
        for name, val in rows:
            det(name, f"{val:>{nw},}", lw)
        print(f"{IND}{'─' * (lw + 1 + nw)}")
        det("total", f"{uns:>{nw},}", lw)

    if a.no_link:
        gid = {}; n = 0
        for c in range(nchunks):
            for l in sorted({int(x) for x in ext_lab[c] if x >= 0}): gid[(c, l)] = n; n += 1
        nglob = n; mode = "chunk-disjoint"
    else:
        gid, nglob = link_chunks(ext_idx, ext_lab, min_anchor=a.min_anchor)
        mode = f"overlap-anchor linked (min_anchor={a.min_anchor})"
    log(f"{nglob:,} global fibers across {nchunks} chunks")
    det("linking", mode)

    # ── .clu/.clc/.clp : the microfiber hierarchy triple ──
    # ATOM (.clc) = the pre-link fine fragment (chunk, local label); FIBER (.clu) = the
    # linked global id; child->parent map (.clp) = atom -> fiber.  The flat .clu is the
    # derived per-spike fiber layer; .clc/.clp preserve the over-split atoms for curation.
    labels = np.full(nspk, -1, int)
    child = np.zeros(nspk, np.int64)                      # per-spike atom id (.clc); 0 = noise
    parent = {}; atom_of = {}; next_atom = 1              # (chunk,local)->atom ; atom->fiber
    for c in range(nchunks):
        if len(ext_idx[c]) == 0: continue
        lo_s = t_min + c * chunk_s; hi_s = t_min + (c + 1) * chunk_s
        emap = {int(g): int(l) for g, l in zip(ext_idx[c], ext_lab[c])}
        for g in np.flatnonzero((res >= lo_s) & (res < hi_s)):
            l = emap.get(int(g), -1)
            if l >= 0:
                labels[g] = gid[(c, l)]
                aid = atom_of.get((c, l))
                if aid is None:
                    aid = next_atom; next_atom += 1; atom_of[(c, l)] = aid
                    parent[aid] = gid[(c, l)] + 1          # fiber id (+1; matches the .clu convention)
                child[g] = aid
    clu = np.where(labels >= 0, labels + 1, 0).astype(np.int32)
    if a.out:
        clu_out = a.out
        nio.write_clu_file(clu_out, clu)
    elif a.emit_hierarchy:                                # <base>.{clu,clc,clp}.<variant>.<elec>[.<stage>]
        from .fiber_refiberize import FiberHierarchy
        paths = FiberHierarchy(child, parent).save(a.base, a.elec, variant=a.method,
                                                   tag=a.clu_stage, backup=False)
        clu_out = paths["clu"]
        det("hierarchy", f"{len(parent)} atoms -> {len(set(parent.values()))} fibers (.clu/.clc/.clp)")
    else:                                                 # <base>.clu.<variant>.<elec>[.<stage>] (flat, legacy)
        clu_out = nio.write_clu(a.base, a.elec, clu, variant=a.method, tag=a.clu_stage)

    # ── .fibers.<method>.<elec> : per (chunk,fiber) geometry, tagged with gid ──
    rows = []
    for c in range(nchunks):
        for l, g in enumerate(chunk_geoms[c]):
            g2 = dict(g); g2['gid'] = gid.get((c, l), -1); g2['chunk'] = c; g2['tmin'] = chunk_tmin[c]
            g2['nn_gid'] = gid.get((c, g.get('nn_local', -1)), -1)
            rows.append(g2)
    M = len(rows)
    def col(k, dt): return np.array([r[k] for r in rows], dt) if M else np.zeros(0, dt)
    fib_out = nio.fibers_path(a.base, a.method, a.elec)
    arrs = dict(
        gid=col('gid', int), chunk=col('chunk', int), tmin=col('tmin', np.float32),
        coarse=col('coarse', int), nspk=col('n', int), radius=col('radius', np.float32),
        refrac=col('refrac', np.float32), depth=col('depth', np.float32),
        width_ms=col('width_ms', np.float32), radius_incl=col('radius_incl', np.float32),
        n_rejected=col('n_rejected', int),
        # firing / cell-type
        rate=col('rate', np.float32), presence=col('presence', np.float32),
        burst=col('burst', np.float32), isi_cv=col('isi_cv', np.float32), hill_fp=col('hill_fp', np.float32),
        # isolation / compactness
        resid_med=col('resid_med', np.float32), resid_mad=col('resid_mad', np.float32),
        chan_resid_var_mean=col('chan_resid_var_mean', np.float32),
        chan_resid_var_max=col('chan_resid_var_max', np.float32),
        nn_dist=col('nn_dist', np.float32), nn_gid=col('nn_gid', int),
        lratio=col('lratio', np.float32), iso_dist=col('iso_dist', np.float32),
        # within-chunk drift
        radius_slope=col('radius_slope', np.float32), depth_slope=col('depth_slope', np.float32),
        dir_drift=col('dir_drift', np.float32),
        # adaptation fingerprint
        adapt_corr=col('adapt_corr', np.float32), adapt_tau=col('adapt_tau', np.float32),
        adapt_snr=col('adapt_snr', np.float32), adapt_meanabsz=col('adapt_meanabsz', np.float32),
        adapt_fracz3=col('adapt_fracz3', np.float32),
        template=np.stack([r['template'] for r in rows]) if M else np.zeros((0, a.nsamp, a.nchan), np.float32),
        grid=np.stack([r['grid'] for r in rows]) if M else np.zeros((0, a.n_grid), np.float32),
        dir=np.stack([r['dir'] for r in rows]) if M else np.zeros((0, a.n_grid, p), np.float32),
        meta_elec=a.elec, meta_channels=gch, meta_sr=a.sr, meta_mask=np.asarray(mask),
        meta_n_grid=a.n_grid, meta_p=p, meta_nsamp=a.nsamp, meta_nchan=a.nchan,
        meta_method=a.method, meta_chunk_min=a.chunk_min, meta_overlap_min=a.overlap_min)
    with open(fib_out, "wb") as f:
        np.savez_compressed(f, **arrs)
    log("wrote")
    det("clu", clu_out)
    det("fibers", f"{fib_out}   ({M:,} instances \u00b7 {nglob:,} global)")

    # ── .merge_candidates.<elec>.tsv : proposed same-neuron merges (curation) ──
    if a.emit_merge_candidates or a.merge_method == "profile":
        cand_out = f"{a.base}.merge_candidates.{a.elec}.tsv"
        ncand = 0
        with open(cand_out, "w") as f:
            f.write("chunk\tgid_a\tgid_b\tlocal_a\tlocal_b\tprofile_dist\tthreshold\n")
            for c in range(nchunks):
                for (ai, bi, d, thr) in chunk_candidates[c]:
                    ga = gid.get((c, ai), -1); gb = gid.get((c, bi), -1)
                    f.write(f"{c}\t{ga}\t{gb}\t{ai}\t{bi}\t{d:.4f}\t{thr:.4f}\n"); ncand += 1
        note = "review-only (fibers NOT merged)" if a.emit_merge_candidates else "already applied"
        det("candidates", f"{cand_out}   ({ncand:,} pairs \u00b7 {note})")
    log(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
