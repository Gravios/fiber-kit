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
    from . import backend as _bk
except ImportError:
    import backend as _bk
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


def gmm_split(wf, pca_k=6, max_sub=8, mask=fl.MASK_FULL, reg=1e-3):
    """BIC-selected Gaussian mixture on PCA of a coarse fiber's realigned waveforms
    (the Python stand-in for KK's CEM split).  Returns sub-labels 0..k-1."""
    N = len(wf)
    if N < 60: return np.zeros(N, int)
    w = fl.realign(wf)[:, mask, :].reshape(N, -1); w = w - w.mean(0)
    U, S, Vt = np.linalg.svd(w, full_matrices=False); F = U[:, :pca_k] * S[:pca_k]
    best = None
    for k in range(1, max_sub + 1):
        if k * 3 > N: break
        g = GaussianMixture(k, covariance_type='full', reg_covar=reg, random_state=0, n_init=2).fit(F)
        b = g.bic(F)
        if best is None or b < best[0]: best = (b, k, g)
    return best[2].predict(F)


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


def _aligned_pca(waves, mask, k):
    """Realign a (sub)cluster to its OWN median by iterated circular cross-correlation
    (fiber_lib.align_xcorr, the channel-summed sub-sample aligner) and return the top-k PCA scores
    of the masked, mean-subtracted waveforms.  The integer dominant-channel fl.realign locks a
    sub-cluster onto the PARENT's peak; re-aligning to this node's own median before featurizing
    lets a deeper bisection be measured on correct alignment."""
    w = fl.align_xcorr(waves, ref="median", iters=6, maxlag=6)[:, mask, :].reshape(len(waves), -1)
    w = w - w.mean(0)
    U, S, _ = np.linalg.svd(w, full_matrices=False)
    return U[:, :k] * S[:k]


def _dipsplit_realign(waves, mask, dim, min_size=40, alpha=0.01, depth=0, maxd=4):
    """Recursive DipSplit that REALIGNS EACH NODE to its own median before deciding the split:
    the 2-means centroid axis and dip test are recomputed from this sub-cluster's median-aligned
    PCA (_aligned_pca), so every bisection is judged on its own alignment instead of the parent's
    (the per-step realign).  Returns a list of index arrays into `waves`."""
    n = len(waves)
    if not _HAVE_DIP or n < 2 * min_size or depth > maxd:
        return [np.arange(n)]
    F = _aligned_pca(waves, mask, dim)                      # realign THIS node + featurize
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
        for piece in _dipsplit_realign(waves[loc], mask, dim, min_size, alpha, depth + 1, maxd):
            out.append(loc[piece])
    return out


def _rkk_realign(waves, mask, dims, max_clusters, min_size, iters=2):
    """rkk (CEM) interleaved with per-cluster realignment -- the per-step realign analog for the
    flat KK split.  rkk assigns all spikes in one EM run, so there is no recursive node; instead
    iterate {cluster -> realign EACH cluster to its own median -> re-featurize -> re-cluster}, so
    the final CEM runs on consistently per-cluster-aligned features (a minority sub-unit locked
    onto the group's dominant peak by the integer fl.realign is otherwise smeared).  Stops early
    when the cluster count is unchanged.  Returns per-spike sub-labels."""
    F = _aligned_pca(waves, mask, dims)                    # whole-group median align + PCA (seed)
    lab = _rkk(F, max_clusters=max_clusters, min_size=min_size, seed=42)
    for _ in range(max(0, iters)):
        Wal = np.array(waves, dtype=float)
        for c in np.unique(lab):                           # realign each cluster to its OWN median
            idx = np.flatnonzero(lab == c)
            if len(idx) >= 8:
                Wal[idx] = fl.align_xcorr(waves[idx], ref="median", iters=6, maxlag=6)
        w = Wal[:, mask, :].reshape(len(waves), -1); w = w - w.mean(0)
        U, S, _ = np.linalg.svd(w, full_matrices=False); F = U[:, :dims] * S[:dims]
        new = _rkk(F, max_clusters=max_clusters, min_size=min_size, seed=42)
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


def cluster_chunk_fine(waves, res_abs, W, nmean, coarse_mg, mask, sr, method="gmm",
                       fine_kappa=40.0, fine_dedup=5.0, fine_mg=40, pca_k=6, max_sub=8,
                       n_grid=40, incl_k=3.0, cone_channel_k=0.0, split_var_margin=0.0,
                       var_split=0.0, var_split_depth=4,
                       dipsplit=True, dip_dim=4, dip_alpha=0.01, dip_min=40, dip_realign=True,
                       nudge_split=True, nudge_max=3, nudge_amp_pct=40.0, nudge_min_channels=4, nudge_alpha=0.01,
                       rkk_dims=6, rkk_max=50, rkk_realign=True, rkk_realign_iters=2, merge_corr=0.0, merge_method="template", sliding_nwin=14,
                       profile_thr=None, profile_floor_pct=90.0, profile_min_n=120,
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
        if method == "none":
            groups = [np.arange(len(cidx))]
        elif method == "fiber":
            sub = cluster_chunk(wsplit, W, nmean, min_group=fine_mg, kappa=fine_kappa, dedup_deg=fine_dedup)
            groups = ([np.arange(len(cidx))] if (sub < 0).all()
                      else [np.flatnonzero(sub == s) for s in np.unique(sub[sub >= 0])])
        elif method == "rkk":
            if rkk_realign:                                # per-cluster realign EM loop
                sub = _rkk_realign(wsplit, mask, rkk_dims, rkk_max, fine_mg, rkk_realign_iters)
            else:                                          # legacy: one parent realign, fixed features
                wc = fl.realign(wsplit)[:, mask, :].reshape(len(cidx), -1); wc = wc - wc.mean(0)
                Uc, Sc, _ = np.linalg.svd(wc, full_matrices=False); Fc = Uc[:, :rkk_dims] * Sc[:rkk_dims]
                sub = _rkk(Fc, max_clusters=rkk_max, min_size=fine_mg, seed=42)
            groups = [np.flatnonzero(sub == s) for s in np.unique(sub)]
        else:
            sub = gmm_split(wsplit, pca_k=pca_k, max_sub=max_sub, mask=mask)
            groups = [np.flatnonzero(sub == s) for s in np.unique(sub)]
        if dipsplit and _HAVE_DIP:
            newg = []
            for grp in groups:                       # PCA each GROUP (within-unit variance)
                if len(grp) < 2 * dip_min:
                    newg.append(grp); continue
                if dip_realign:                      # realign EACH node to its own median (per step)
                    pieces = _dipsplit_realign(wcf[grp], mask, dip_dim, dip_min, dip_alpha)
                else:                                # legacy: one parent realign, fixed features
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
            if len(grp) < fine_mg: continue
            sidx = cidx[grp]; rad = float('nan'); rej = 0
            if incl_k > 0 and len(sidx) >= 20:
                w_al = fl.realign(waves[sidx])
                Xg = (w_al[:, mask, :].reshape(len(sidx), -1) - nmean) @ W
                grid, D = ft.trajectory(Xg); rr = np.linalg.norm(Xg, axis=1)
                resid = np.linalg.norm(Xg - rr[:, None] * ft.predict_many((grid, D), rr), axis=1)
                med = float(np.median(resid)); mad = 1.4826 * float(np.median(np.abs(resid - med)))
                rad = med + incl_k * mad; keep = resid <= rad; rej = int((~keep).sum())
                sidx = sidx[keep]
                if len(sidx) < fine_mg: continue
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
                    rej += int((~keep2).sum()); sidx = sidx[keep2]
                    if len(sidx) < fine_mg: continue
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
                    if len(sidx) < fine_mg: continue
            fine[sidx] = nid
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
    if merge_corr and merge_method in ("template", "sliding") and len(geoms) > 1:
        Kg = len(geoms)
        if merge_method == "sliding":
            Xs = [(fl.realign(waves[np.flatnonzero(fine == u)])[:, mask, :].reshape(-1, len(mask) * waves.shape[2]) - nmean) @ W
                  for u in range(Kg)]
            edges = _sliding_pairs(Xs, n_win=sliding_nwin, min_cos=merge_corr)
        else:
            T = np.array([g['template'].ravel() for g in geoms]); T = T - T.mean(1, keepdims=True)
            T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-12); C = T @ T.T
            edges = list(zip(*np.where(np.triu(C, 1) > merge_corr)))
        fine, geoms = _apply_edges(fine, geoms, edges, waves, res_abs, W, nmean, mask, sr, n_grid, ct0, ct1)
    # ── Block B: same-neuron grouping by energy-resolved DIRECTION PROFILE d(r).
    #    Direction is the validated same-neuron signal (AUC ~0.98 same-fiber-halves
    #    vs distinct fibers); curvature/tangent add noise and are NOT used.  Threshold
    #    defaults to the within-fiber (same-neuron) floor.  emit_candidates=True writes
    #    proposals WITHOUT merging (curation); merge_method='profile' applies them.
    #    Runs AFTER Block A, so it can review/merge already-consolidated fibers. ──
    if (merge_method == "profile" or emit_candidates) and len(geoms) > 1:
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
            fine, geoms = _apply_edges(fine, geoms, [(ai, bi) for ai, bi, _ in cand],
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
    cand, ncore).  Reads everything else (memmaps + params) from _CTX."""
    c, ext, res_e = task
    ctx = _CTX; kw = ctx["cf"]
    waves = np.asarray(ctx["spk"][ext], dtype=float)
    s0 = int(res_e.min()) - ctx["nsamp"]; s1 = int(res_e.max()) + ctx["nsamp"] + 1
    W, nmean, _ = fil_chunk_whitener(ctx["filmm"], ctx["gch"], s0, s1, res_e, ctx["nsamp"], ctx["mask"])
    cand = []
    lab, geoms = cluster_chunk_fine(waves, res_e, W, nmean, ctx["min_group"], ctx["mask"], ctx["sr"],
                                    candidates_out=cand, **kw)
    return c, ext, lab, geoms, cand


def main():
    ap = argparse.ArgumentParser(
        description="Cluster a session group into fibers. Reads <session>.yaml "
                    "(or <session>/<session>.yaml) for channels/sr/nChannels; "
                    "CLI flags override the YAML.")
    sy.add_session_args(ap)
    ap.add_argument("--chunk-min", type=float, default=12.0); ap.add_argument("--overlap-min", type=float, default=4.0)
    ap.add_argument("--min-group", type=int, default=200, help="COARSE min spikes/fiber (for linking)")
    ap.add_argument("--fine-method", choices=["gmm","rkk","fiber","none"], default="gmm")
    ap.add_argument("--rkk-dims", type=int, default=6); ap.add_argument("--rkk-max", type=int, default=50)
    ap.add_argument("--rkk-realign", dest="rkk_realign", action="store_true", default=True,
                    help="interleave rkk (CEM) with per-cluster realignment (per-step; default on)")
    ap.add_argument("--no-rkk-realign", dest="rkk_realign", action="store_false",
                    help="legacy: one parent realign + fixed features for the rkk split")
    ap.add_argument("--rkk-realign-iters", type=int, default=2,
                    help="cluster<->realign passes in the rkk realign loop")
    ap.add_argument("--merge-corr", type=float, default=0.0, help="consolidate fibers above this (0=off; 0.95 template / 0.90 sliding)")
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
    ap.add_argument("--clu-stage", dest="clu_stage", default="",
                    help="post-group stage tag for the clu: <base>.clu.<method>.<elec>[.<stage>] "
                         "(default none); e.g. --clu-stage session")
    ap.add_argument("--gpu", action="store_true", help="run the realign/whiten kernels on GPU (CuPy; needs the [gpu] extra)")
    ap.add_argument("--jobs", "-j", type=int, default=1,
                    help="parallel worker processes over chunks (default 1 = serial; chunks are independent)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.gpu:
        on = _bk.use_gpu(True)
        print(f"[fiber_session] GPU requested: backend = {_bk.backend_name()}"
              + ("" if on else " (CuPy/CUDA unavailable -> CPU)"))
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
    print(f"loaded {nspk} spikes ({spkpath}); .fil {filmm.shape[0]} samples x {a.ntotal} ch")
    if a.dipsplit and not _HAVE_DIP:
        print("  [note] --dipsplit requested but the 'diptest' package is not installed; skipping (pip install diptest)")

    chunk_s = a.chunk_min * 60.0 * a.sr; ov_s = a.overlap_min * 60.0 * a.sr
    t_min, t_max = int(res.min()), int(res.max())
    nchunks = int(np.ceil((t_max - t_min) / chunk_s))
    ext_idx = [np.array([], int)] * nchunks; ext_lab = [np.array([], int)] * nchunks
    chunk_geoms = [[] for _ in range(nchunks)]; chunk_tmin = [0.0] * nchunks
    chunk_candidates = [[] for _ in range(nchunks)]

    # ── build per-chunk tasks (small chunks skipped here, serially & cheaply) ──
    meth = "none" if a.no_fine else a.fine_method
    cf = dict(method=meth, fine_kappa=a.fine_kappa, fine_dedup=a.fine_dedup_deg,
              fine_mg=a.fine_min_group, pca_k=a.pca_k, max_sub=a.max_sub, n_grid=a.n_grid,
              incl_k=a.inclusion_k, cone_channel_k=a.cone_channel_k,
              split_var_margin=a.split_var_margin, var_split=a.var_split,
              var_split_depth=a.var_split_depth, dipsplit=a.dipsplit,
              dip_dim=a.dip_dim, dip_alpha=a.dip_alpha, dip_min=a.dip_min, dip_realign=a.dip_realign,
              nudge_split=a.nudge_split, nudge_max=a.nudge_max, nudge_amp_pct=a.nudge_amp_pct, nudge_min_channels=a.nudge_min_channels, nudge_alpha=a.nudge_alpha,
              rkk_dims=a.rkk_dims, rkk_max=a.rkk_max, rkk_realign=a.rkk_realign, rkk_realign_iters=a.rkk_realign_iters,
              merge_corr=a.merge_corr, merge_method=a.merge_method, sliding_nwin=a.sliding_nwin,
              profile_thr=a.profile_thr, profile_floor_pct=a.profile_floor_pct,
              profile_min_n=a.profile_min_n, emit_candidates=a.emit_merge_candidates,
              deadapt=a.deadapt, deadapt_min_corr=a.deadapt_min_corr,
              adapt_clean=a.adapt_clean, adapt_z=a.adapt_z, adapt_isi_ms=a.adapt_isi_ms,
              adapt_clean_corr=a.adapt_clean_corr, adapt_clean_snr=a.adapt_clean_snr,
              adapt_taumax=a.adapt_taumax, collision_flag=a.collision_flag,
              collision_gain=a.collision_gain, collision_shift=a.collision_shift,
              quality_metrics=a.quality_metrics, quality_dims=a.quality_dims)
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
            print(f"[fiber_session] chunk {c+1}/{nchunks}: {ncore} core ({len(ext)} ext) -> 0 fibers (small)"); continue
        tasks.append((c, ext, res[ext]))

    def _store(result):
        c, ext, lab, geoms, cand = result
        ext_idx[c] = ext; ext_lab[c] = lab; chunk_geoms[c] = geoms; chunk_candidates[c] = cand
        print(f"[fiber_session] chunk {c+1}/{nchunks}: {ncore_of[c]} core ({len(ext)} ext) -> {len(geoms)} fibers")

    jobs = max(1, int(a.jobs))
    if jobs == 1 or len(tasks) <= 1:
        _init_chunk_worker(cfg)                       # serial: identical to the former inline loop
        for task in tasks:
            _store(_process_chunk(task))
    else:
        from concurrent.futures import ProcessPoolExecutor
        nworkers = min(jobs, len(tasks))
        print(f"[fiber_session] clustering {len(tasks)} chunks on {nworkers} processes")
        with ProcessPoolExecutor(max_workers=nworkers,
                                 initializer=_init_chunk_worker, initargs=(cfg,)) as ex:
            for result in ex.map(_process_chunk, tasks):
                _store(result)

    if a.no_link:
        gid = {}; n = 0
        for c in range(nchunks):
            for l in sorted({int(x) for x in ext_lab[c] if x >= 0}): gid[(c, l)] = n; n += 1
        nglob = n; mode = "chunk-disjoint"
    else:
        gid, nglob = link_chunks(ext_idx, ext_lab, min_anchor=a.min_anchor)
        mode = f"overlap-anchor linked (min_anchor={a.min_anchor})"
    print(f"[fiber_session] {nglob} global fibers across {nchunks} chunks  ({mode})")

    # ── .clu : core spikes -> global id (+1; 0=noise) ──
    labels = np.full(nspk, -1, int)
    for c in range(nchunks):
        if len(ext_idx[c]) == 0: continue
        lo_s = t_min + c * chunk_s; hi_s = t_min + (c + 1) * chunk_s
        emap = {int(g): int(l) for g, l in zip(ext_idx[c], ext_lab[c])}
        for g in np.flatnonzero((res >= lo_s) & (res < hi_s)):
            l = emap.get(int(g), -1)
            if l >= 0: labels[g] = gid[(c, l)]
    clu = np.where(labels >= 0, labels + 1, 0).astype(np.int32)
    if a.out:
        clu_out = a.out
        nio.write_clu_file(clu_out, clu)
    else:                                                 # <base>.clu.<variant>.<elec>[.<stage>]
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
    print(f"wrote {clu_out}")
    print(f"wrote {fib_out}  ({M} fiber-instances, {nglob} global; geometry over time = rows sharing gid)")

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
        print(f"wrote {cand_out}  ({ncand} candidate pairs; gid columns valid in emit mode) [{note}]")
    print(f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
