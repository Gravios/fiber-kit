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
import shutil
import numpy as np
from collections import defaultdict, Counter
try:
    from . import fiber_lib as fl
except ImportError:
    import fiber_lib as fl
try:
    from . import fiber_link_core as flc
except ImportError:
    import fiber_link_core as flc
try:
    from . import fiber_tracer as ft
except ImportError:
    import fiber_tracer as ft
try:
    from . import fiber_geometry as fg
except ImportError:
    import fiber_geometry as fg
try:
    from . import config as cfgmod
except ImportError:
    import config as cfgmod
try:
    from . import fiber_chunks as _fch
except ImportError:
    import fiber_chunks as _fch
try:
    from . import fiber_consolidate as fcons
except ImportError:
    import fiber_consolidate as fcons
from sklearn.mixture import GaussianMixture

# ── splitting primitives now live in fiber_split ─────────────────────────────
# Re-exported under their original names so every call site keeps working: this
# module's own body, and fiber_refine, which reaches them as fs._dipsplit_rec and
# friends.  fiber_refine should import fiber_split directly (it now does); these
# aliases exist so the move is invisible to anything else that reached in.
try:
    from . import fiber_split as _fsplit
except ImportError:
    import fiber_split as _fsplit
_dipsplit_rec       = _fsplit._dipsplit_rec
_aligned_pca        = _fsplit._aligned_pca
_dipsplit_realign   = _fsplit._dipsplit_realign
_amp_spread         = _fsplit._amp_spread
_nudge_split        = _fsplit._nudge_split
_variance_split     = _fsplit._variance_split
_cfiber_edge_filter = _fsplit._cfiber_edge_filter

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

# ── moved out; re-exported so nothing that reached through this module breaks ──
# read_res/open_spkD were pure passthroughs to neuro_io (read_res pinned
# prefer_canonical(), which is already nio.read_res's default) and their callers
# now call nio directly.  fil_chunk_whitener adapts fiber_lib.chunk_whitener_mm
# and lives there now.  These names remain so a caller outside the tree still
# resolves; new code should use neuro_io / fiber_lib.
read_res           = nio.read_res
open_spkD          = nio.open_spkD
fil_chunk_whitener = fl.fil_chunk_whitener
try:
    from . import fiber_pca as _fpca
except ImportError:
    import fiber_pca as _fpca
try:
    from . import fiber_stderiv as fst
except ImportError:
    import fiber_stderiv as fst
try:
    from . import backend as _bk
except ImportError:
    import backend as _bk
try:
    from . import fiber_ccg as cg
except ImportError:
    import fiber_ccg as cg
# diptest and the availability flag are owned by fiber_split, which is where the
# functions using them now live.  Re-exported so this module's body and anything
# that reached fs._HAVE_DIP keep working, and so there is one definition rather
# than two that can disagree about whether the optional dependency is present.
_diptest  = _fsplit._diptest
_HAVE_DIP = _fsplit._HAVE_DIP

P_DIM = len(fl.MASK_FULL) * 8     # default masked feature dim (recomputed per nchan below)


def cluster_chunk(waves, W, nmean, min_group=100, kappa=20.0, dr_frac=0.15,
                  n_seeds=800, n_support=20000, dedup_deg=8.0, dedup_radf=0.12,
                  mask=fl.MASK_FULL, seed_density=0.0):
    """waves (n,nsamp,nch) spkD + chunk whitener -> per-spike label (0-based, -1=none)."""
    N = len(waves)
    if N < 2 * min_group:
        return np.full(N, -1, int)
    X = (fl.realign(waves)[:, mask, :].reshape(N, -1) - nmean) @ W
    r = np.linalg.norm(X, axis=1); d = X / (r[:, None] + 1e-12)
    nsup = min(N, n_support); supi = np.arange(nsup) * N // nsup
    dsup, rsup = d[supi], r[supi]
    S = min(N, n_seeds)
    rsort = np.sort(r); dr = dr_frac * (rsort[int(0.99 * (N - 1))] - rsort[int(0.01 * (N - 1))])
    if seed_density > 0.0:
        # DENSITY-PREFERENTIAL seeding: draw seeds toward CONCENTRATED modes of the waveform space
        # instead of a uniform stride over spikes.  Uniform seeding spends the seed budget in proportion
        # to how many spikes sit somewhere, so most seeds land in the diffuse background between modes
        # and wander without converging; the isolated modes -- which is where reproducible fibers live,
        # independent of firing rate (on g5 the density<->rate correlation is only ~0.2) -- get too few.
        # Local density uses the SAME support set and the SAME band+cone geometry the mean-shift itself
        # uses (radius within dr, angular exp(kappa*(cos-1))).  seed_density in (0,1] is the sampling
        # exponent: p(seed=i) proportional to density_i**seed_density.  Seeds are drawn WITHOUT
        # replacement so a single dense mode cannot take the whole budget.
        dens = np.empty(N)
        for i in range(0, N, 2048):                       # block the N x nsup pass to bound memory
            dc = d[i:i + 2048] @ dsup.T
            bd = np.abs(rsup[None, :] - r[i:i + 2048, None]) < dr
            dens[i:i + 2048] = np.where(bd, np.exp(kappa * (dc - 1.0)), 0.0).sum(1)
        w_seed = dens ** float(seed_density)
        tot = w_seed.sum()
        if tot > 0:
            # Seed the RNG from the data itself (spike count + a coarse feature digest), NOT a constant:
            # each resampling draw sees a different spike subset, so this varies per draw and keeps the
            # ensemble's draw-to-draw diversity, while staying deterministic for a fixed input.
            seed = (N * 2654435761 + int(abs(X[:, 0]).sum() * 1e3)) & 0x7FFFFFFF
            rng = np.random.default_rng(seed)
            sdi = np.sort(rng.choice(N, size=S, replace=False, p=w_seed / tot))
        else:
            sdi = np.arange(S) * N // S
    else:
        sdi = np.arange(S) * N // S
    ds, rs = d[sdi].copy(), r[sdi].copy()
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


# Set per worker process by _init_chunk_worker from cfg["fast_stats"].  When true,
# fiber_geom skips the curation-only quality metrics (adaptation residual, per-channel
# residual profile, within-chunk drift slopes) and leaves them NaN.  Nothing in
# clustering, consensus or the clu/clc/clp triplet reads them -- they exist for the
# diagnostic npz and for hand curation -- so an ensemble run that only wants the
# clusters can skip several passes over every fiber's waveforms.
_FAST_STATS = False


def fiber_geom(wsub, res_sub, W, nmean, mask, sr, n_grid=40, chunk_t0=None, chunk_t1=None,
               stats=None):
    """Geometry + per-chunk quality / firing / drift statistics of one fiber.
    Every stat is per-chunk, so rows sharing a gid across the session form time
    series (depth(t), rate(t), nn_dist(t), within-chunk drift, ...) for curation."""
    if stats is None:
        stats = not _FAST_STATS
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
    if n >= 80 and stats:
        h = n // 2; a, b = o[:h], o[h:]
        dd = _profile_dir_dist(ft.trajectory(Xg[a]), np.percentile(rr[a], [15, 85]),
                               ft.trajectory(Xg[b]), np.percentile(rr[b], [15, 85]))
        if dd is not None: dir_drift = dd

    if stats:
        try:
            from . import fiber_adapt as _fa
        except ImportError:
            import fiber_adapt as _fa
        _z, _ai = _fa.adaptation_residual(wsub, res_sub, W, nmean, mask, sr)
        crv = ft.channel_residual_profile(wsub, W, nmean, mask) if n >= 20 else None
    else:
        _z = np.array([np.nan], np.float32)
        _ai = {'corr': float('nan'), 'tau': float('nan'), 'snr': float('nan')}
        crv = None
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
    gid, _, _ = link_chunks(ext_idx, ext_lab, min_anchor=8, frac=0.5)
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








# rkk parity knobs.  klustakwik.py is now in parity with kiloklustakwik's classic branch, whose
# default penalty is AIC (PenaltyMix=0); the pre-parity rkk used BIC.  FK_SESSION_RKK_PENALTY_MIX=1
# restores BIC if a session needs the old cluster counts.
_RKK_PENALTY_MIX = cfgmod.knob_value("FK_SESSION_RKK_PENALTY_MIX", float, 0.0)   # env > $FK_CONFIG > default (the
_RKK_SPLITS      = cfgmod.knob_value("FK_SESSION_RKK_SPLITS", int, 1)            # yaml was ignored on a direct call)


def _rkk_labels(F, max_clusters, min_size, seed=42, delete=True):
    """rkk, mapped onto fiber-session's label convention.

    klustakwik() now returns the CLASSIC labelling: 0 is the uniform NOISE cluster (the reject
    option, KK.cpp:295-300) and real clusters run 1..K.  fiber-session's convention is 0-based real
    labels with -1 = none, so noise maps to -1 and the rest shift down.  Without this the reject
    bucket would be emitted as a spurious fragment -- the exact inversion of why the noise cluster
    was added.

    min_clusters is pinned to max_clusters so a single starting K is used.  The classic driver
    sweeps every K in [MinClusters, MaxClusters] keeping the best score, which is O(K) full CEMs;
    that is affordable in OpenMP C++ and not here, and it would also silently change what
    `max_clusters` has always meant at these call sites.

    min_size is applied HERE rather than inside the CEM.  Classic KlustaKwik's only size rule is
    nClassMembers <= nDims (KK.cpp:189), far below the stage's fine min-group, so a cluster under
    min_size is moved to the RESERVE instead of being kept -- which is what these callers have
    always assumed min_size did.  A deliberate deviation, in the caller, leaving the parity
    implementation itself untouched.
    """
    lab = _rkk(F, max_clusters=max_clusters, min_clusters=max_clusters, seed=seed,
               penalty_mix=_RKK_PENALTY_MIX, splits=bool(_RKK_SPLITS), noise=True)
    out = np.asarray(lab, int) - 1                     # 0 (noise) -> -1 ; 1..K -> 0..K-1
    if min_size and int(min_size) > 1:
        for c in np.unique(out[out >= 0]):
            m = out == c
            if int(m.sum()) < int(min_size):
                out[m] = -1
    uniq = [c for c in np.unique(out) if c >= 0]        # renumber survivors densely
    remap = {c: i for i, c in enumerate(uniq)}
    return np.array([remap.get(int(v), -1) for v in out], dtype=int)


def _rkk_realign(waves, mask, dims, max_clusters, min_size, iters=2, delete=True, basis=None):
    """rkk (CEM) interleaved with per-cluster realignment -- the per-step realign analog for the
    flat KK split.  rkk assigns all spikes in one EM run, so there is no recursive node; instead
    iterate {cluster -> realign EACH cluster to its own median -> re-featurize -> re-cluster}, so
    the final CEM runs on consistently per-cluster-aligned features (a minority sub-unit locked
    onto the group's dominant peak by the integer fl.realign is otherwise smeared).  SHAPE features
    project onto the global ndm_pca basis when given, else a per-call local SVD.  Stops early
    when the cluster count is unchanged.  Returns per-spike sub-labels."""
    F = _aligned_pca(waves, mask, dims, basis=basis)       # whole-group median align + features (seed)
    lab = _rkk_labels(F, max_clusters, min_size, seed=42, delete=delete)
    for _ in range(max(0, iters)):
        Wal = np.array(waves, dtype=float)
        for c in np.unique(lab):                           # realign each cluster to its OWN median
            if c < 0:                                      # -1 = reserve, not a cluster
                continue
            idx = np.flatnonzero(lab == c)
            if len(idx) >= 8:
                Wal[idx] = fl.align_xcorr(waves[idx], ref="median", iters=6, maxlag=6)
        F = _fpca.cluster_features(Wal, basis, realign=False, dims=dims) if basis is not None else None
        if F is None:
            w = Wal[:, mask, :].reshape(len(waves), -1); w = w - w.mean(0)
            U, S, _ = np.linalg.svd(w, full_matrices=False); F = U[:, :dims] * S[:dims]
        new = _rkk_labels(F, max_clusters, min_size, seed=42, delete=delete)
        stop = len(np.unique(new)) == len(np.unique(lab))
        lab = new
        if stop:
            break
    return lab











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


def _median_resid_ratio(w, mask):
    """Median-subtracted residual variance / signal for a group of aligned waveforms.  A tight fiber
    scores low; a WELDED fiber (two shapes averaged) scores high because its spikes sit far from the
    pooled median.  Validated on g5: corr(#sub-modes, this) = +0.88, welded fibers ~1.7x clean ones,
    and a threshold of ~0.35 separates them with no false positives.  Used to target reseeding at the
    heterogeneous fibers only, rather than re-splitting everything."""
    wal = fl.realign(w)[:, mask, :].reshape(len(w), -1)
    med = np.median(wal, 0)
    sig = float(np.linalg.norm(med)) + 1e-9
    resid = np.median(np.linalg.norm(wal - med[None, :], axis=1))
    return float(resid / sig)


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


def _offset_edge_filter(edges, fine, waves_std, off_thr, amp_frac=0.3):
    """Drop Block-A merge edges whose two fibers disagree in INTER-CHANNEL TIMING.

    The edges came from amplitude-FOOTPRINT correlation (mean-template / sliding cosine);
    two co-located cells can match in footprint yet differ in per-channel spike timing --
    the drift-robust, position-encoding half of identity that footprint cosine is blind to.
    Timing is read from RAW/STANDARD templates (waves_std), never stderiv: stderiv's spatial
    common-mode removal (C*raw_c - sum_j raw_j) mixes channels, so per-channel timing on the
    stderiv waveform is not the cell's true timing.  Uses the validated fiber_geometry pair --
    interchannel_offsets(method='xcorr') + offset_distance -- and keeps an edge only when the
    RMS inter-channel offset difference is <= off_thr samples (xcorr LAG scale, ~0.2-0.3, not
    the trough ~1.0).  A fiber whose timing can't be measured (too few spikes / one channel)
    is never vetoed on that basis (offset_distance -> inf only blocks; None passes)."""
    if not edges or waves_std is None or off_thr <= 0:
        return edges
    off = {}
    for u in {v for e in edges for v in e}:
        w = waves_std[np.flatnonzero(fine == u)]
        if len(w) < 2:
            off[u] = None
            continue
        T = fl.realign(np.asarray(w, float)).mean(0)          # RAW mean template (nsamp, nchan)
        off[u] = fg.interchannel_offsets(T, amp_frac=amp_frac, method="xcorr")
    keep = []
    for a, b in edges:
        oa, ob = off.get(a), off.get(b)
        if oa is None or ob is None or fg.offset_distance(oa, ob) <= off_thr:
            keep.append((a, b))
    return keep


def cluster_chunk_fine(waves, res_abs, W, nmean, coarse_mg, mask, sr, method="gmm", coarse_dr=0.15, coarse_seed_density=0.0,
                       fine_kappa=40.0, fine_dedup=5.0, fine_mg=40, pca_k=6, max_sub=8, basis=None,
                       n_grid=40, incl_k=3.0, incl_assign=False, no_noise=False, keep_small=False, keep_small_min=5,
                       assign_unsorted=False, shed=None, cone_channel_k=0.0, split_var_margin=0.0,
                       energy_band=False, eband_width=0.45, eband_overlap=0.2, eband_confound=0.4, eband_min_span=0.6, eband_min_band=60, eband_low_assign=0.0,
                       var_split=0.0, var_split_depth=4,
                       dipsplit=True, dip_dim=4, dip_alpha=0.01, dip_min=40, dip_realign=True,
                       nudge_split=True, nudge_max=3, nudge_amp_pct=40.0, nudge_min_channels=4, nudge_alpha=0.01,
                       rkk_dims=6, rkk_max=50, rkk_realign=True, rkk_realign_iters=2, rkk_delete=True, merge_corr=0.0, merge_method="template", sliding_nwin=14, cfiber_gate=False, cfiber_q=0.90, off_thr=0.0, off_amp_frac=0.3, waves_std=None,
                       profile_thr=None, profile_floor_pct=90.0, profile_min_n=120,
                       resplit_passes=0, resplit_residual_thr=0.08, resplit_topch=3, resplit_min_reduction=0.20, resplit_min_n=10, resplit_merge_corr=0.99,
                       resplit_detrend_episode=False, resplit_detrend_win=90.0, resplit_detrend_min_n=100,
                       reseed_residual_thr=0.0, reseed_min_reduction=0.05,
                       refit_iters=0,
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
    # coarse_dr sets the radial (whitened-amplitude) band half-width for spike->seed association in
    # the coarse pass: wider -> each coarse fiber sweeps up a broader amplitude range -> fewer, fatter
    # coarse groups; narrower -> tighter bands, so amplitude-distinct cells stay separate coarse fibers
    # for the fine splitter to resolve.  Only the COARSE call is exposed; the per-fiber fine re-cluster
    # keeps the default band (it already operates within one coarse fiber).
    coarse = cluster_chunk(waves, W, nmean, min_group=coarse_mg, dr_frac=coarse_dr, seed_density=coarse_seed_density,
                           mask=mask)
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
            sub = cluster_chunk(wsplit, W, nmean, min_group=fine_mg, kappa=fine_kappa, dedup_deg=fine_dedup, mask=mask)
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
                sub = _rkk_labels(Fc, rkk_max, fine_mg, seed=42, delete=rkk_delete)
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
        if reseed_residual_thr > 0 and method not in ("none",):
            # RESIDUAL-GATED reseed: a fiber whose median-subtracted residual/signal exceeds the threshold
            # is heterogeneous -- typically two co-located cells of different shape averaged into one
            # (the welding the consensus matcher also reveals).  Re-run the fine splitter on JUST that
            # fiber's spikes; keep the re-split only if it actually LOWERS the residual (a real sub-unit
            # does; noise does not).  Targeted by the residual detector, so clean fibers are untouched.
            newg = []
            for grp in groups:
                if len(grp) < 2 * fine_mg or _median_resid_ratio(wcf[grp], mask) < reseed_residual_thr:
                    newg.append(grp); continue
                sub = cluster_chunk(wcf[grp], W, nmean, min_group=fine_mg, kappa=fine_kappa,
                                    dedup_deg=fine_dedup, seed_density=coarse_seed_density, mask=mask)
                if (sub < 0).all() or len(np.unique(sub[sub >= 0])) < 2:
                    newg.append(grp); continue
                pieces = [np.flatnonzero(sub == u) for u in np.unique(sub[sub >= 0])]
                before = _median_resid_ratio(wcf[grp], mask)
                after = max(_median_resid_ratio(wcf[grp[pc]], mask) for pc in pieces if len(pc) >= fine_mg)
                if after < before - reseed_min_reduction:
                    newg += [grp[pc] for pc in pieces]           # accept: reseed lowered residual
                else:
                    newg.append(grp)                             # reject: no improvement
            groups = newg
        def _keep_whole(gidx, why, cf=cf):
            """keep_small: a group the size gates would shed becomes a fiber of its own (all its
            spikes, no inclusion gate) instead of noise.  Returns True if kept."""
            nonlocal nid
            if not keep_small or len(gidx) < keep_small_min:
                return False
            try:
                g = fiber_geom(waves[gidx], res_abs[gidx], W, nmean, mask, sr, n_grid, chunk_t0=ct0, chunk_t1=ct1)
            except Exception:                                    # too small / degenerate: left to assign_unsorted
                return False
            fine[gidx] = nid
            g['coarse'] = int(cf); g['radius_incl'] = float('nan'); g['n_rejected'] = 0
            g['n_adapt_rejected'] = 0; g['small'] = why
            geoms.append(g); nid += 1
            if shed is not None: shed['small_kept'] = shed.get('small_kept', 0) + len(gidx)
            return True

        for grp in groups:
            g0 = len(grp)
            if g0 < fine_mg:
                if not _keep_whole(cidx[grp], 'group') and shed is not None:
                    shed['small_group'] = shed.get('small_group', 0) + g0
                continue
            sidx = cidx[grp]; rad = float('nan'); rej = 0; rej_incl = 0; incl_rej = None
            if incl_k > 0 and len(sidx) >= 20:
                # REALIGN -> REFIT -> RE-JUDGE.  A true member at a bad sub-sample offset lands far from
                # the fiber trajectory and is rejected as an outlier (an alignment artifact, not a wrong
                # cell), and its offset blurs the template judging everyone else.  Iterating {align this
                # fiber's spikes to their own median -> rebuild the trajectory -> re-apply the inclusion
                # gate} sharpens the template and re-judges from the FULL member set each pass, so an
                # early drop on a blurred template can be undone once it sharpens.  refit_iters=0 = the
                # original single pass.
                base = sidx.copy()
                keep_mask = np.ones(len(base), bool)
                for _it in range(max(1, refit_iters + 1)):
                    cur = base[keep_mask]
                    if len(cur) < fine_mg:
                        break
                    w_al = fl.realign(waves[cur])
                    Xg = (w_al[:, mask, :].reshape(len(cur), -1) - nmean) @ W
                    grid, D = ft.trajectory(Xg)
                    w_all = fl.realign(waves[base])
                    Xall = (w_all[:, mask, :].reshape(len(base), -1) - nmean) @ W
                    rr_all = np.linalg.norm(Xall, axis=1)
                    resid_all = np.linalg.norm(Xall - rr_all[:, None] * ft.predict_many((grid, D), rr_all), axis=1)
                    rc = resid_all[keep_mask] if keep_mask.any() else resid_all
                    med = float(np.median(rc)); mad = 1.4826 * float(np.median(np.abs(rc - med)))
                    rad = med + incl_k * mad
                    new_mask = resid_all <= rad
                    if new_mask.sum() < fine_mg:
                        break
                    if np.array_equal(new_mask, keep_mask) and _it > 0:
                        keep_mask = new_mask; break
                    keep_mask = new_mask
                keep = keep_mask; rej = int((~keep).sum()); rej_incl = rej
                if incl_assign:
                    incl_rej = base[~keep]            # remember the inclusion tail (good spikes) to keep in the sort
                sidx = base[keep]
                if len(sidx) < fine_mg:
                    if not _keep_whole(cidx[grp], 'core') and shed is not None:
                        shed['small_core'] = shed.get('small_core', 0) + g0
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
                        if not _keep_whole(cidx[grp], 'core') and shed is not None:
                            shed['small_core'] = shed.get('small_core', 0) + g0
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
                        if not _keep_whole(cidx[grp], 'core') and shed is not None:
                            shed['small_core'] = shed.get('small_core', 0) + g0
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
    vetoes = dict(cfiber_gate=cfiber_gate, cfiber_q=cfiber_q, off_thr=off_thr, off_amp_frac=off_amp_frac,
                  waves_std=waves_std, refrac_ms=refrac_ms, refrac_thr=refrac_thr, refrac_min_exp=refrac_min_exp,
                  refrac_censor_ms=refrac_censor_ms)
    if resplit_passes == 0 and merge_corr and merge_method in ("template", "sliding") and len(geoms) > 1:
        Kg = len(geoms)
        if merge_method == "sliding":
            Xs = [(fl.realign(waves[np.flatnonzero(fine == u)])[:, mask, :].reshape(-1, len(mask) * waves.shape[2]) - nmean) @ W
                  for u in range(Kg)]
            scores = _sliding_pairs(Xs, n_win=sliding_nwin, min_cos=merge_corr, return_scores=True)
        else:
            T = np.array([g['template'].ravel() for g in geoms]); T = T - T.mean(1, keepdims=True)
            T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-12); C = T @ T.T
            scores = {(int(i), int(j)): float(C[i, j]) for i, j in zip(*np.where(np.triu(C, 1) > merge_corr))}
        edges = _merge_vetoes(list(scores), fine, waves, mask, res_abs, sr, **vetoes)
        edges = _complete_link(edges, scores)
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
            # a profile-merge is PERMANENT (relink/defrag only ever merge): every veto applies
            # (the refractory one is power-aware, so it only ever removes a false merge)
            scores = {(ai, bi): -d for ai, bi, d in cand}
            edges = _merge_vetoes([(ai, bi) for ai, bi, _ in cand], fine, waves, mask, res_abs, sr, **vetoes)
            edges = _complete_link(edges, scores)
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
                # the same vetoes as Blocks A/B (until 0.56 this merge had none), and complete linkage:
                # A~B and B~C no longer weld A to C when A and C are not themselves a pair
                scores = _sliding_pairs(Xs, n_win=sliding_nwin, min_cos=resplit_merge_corr, return_scores=True)
                edges = _merge_vetoes(list(scores), fine, waves, mask, res_abs, sr, **vetoes)
                edges = _complete_link(edges, scores)
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
    # ── assign-unsorted: every spike still unsorted (coarse-unassigned, groups too small to keep,
    #    inclusion tails when not kept) is placed, in the whitened, realigned feature space the chunk was
    #    clustered in (amplitude counts): a spike within its nearest fiber's own spread (median +
    #    incl_k * MAD of that fiber's distances to its centre -- the inclusion rule) joins it; the spikes outside every fiber's spread are
    #    GMM-split among themselves first (with keep_small), each group >= keep_small_min becoming a fiber
    #    of its own -- a sparse cell that never made a fiber is not forced into a neighbour -- and only
    #    the remainder goes to the nearest fiber.  Collision / undefined fibers never receive.  Artifacts are removed BEFORE clustering (fiber-flag-artifacts
    #    -> --exclude-clu), so nothing real is left for the noise cluster.  A chunk with no fiber at
    #    all becomes one fiber.  Off by default. ──
    if assign_unsorted:
        un = np.flatnonzero(fine < 0)
        tgt = [l for l, g in enumerate(geoms) if not (g.get('collision') or g.get('undefined'))]
        n_asg = 0
        if un.size and tgt:
            Xall = (fl.realign(waves)[:, mask, :].reshape(len(waves), -1) - nmean) @ W
            cen = np.stack([np.median(Xall[fine == l], 0) for l in tgt])
            def _radius(dl):                                # the fiber's own inclusion rule: median + k*MAD
                med = float(np.median(dl)); mad = 1.4826 * float(np.median(np.abs(dl - med)))
                return med + (incl_k if incl_k > 0 else 3.0) * mad
            rad = np.array([_radius(np.linalg.norm(Xall[fine == l] - cen[q], axis=1)) for q, l in enumerate(tgt)])
            near = np.empty(un.size, int); dist = np.empty(un.size)
            for b0 in range(0, un.size, 4096):
                xb = Xall[un[b0:b0 + 4096]]
                d2 = np.maximum((xb * xb).sum(1)[:, None] - 2.0 * xb @ cen.T + (cen * cen).sum(1)[None, :], 0.0)
                q = np.argmin(d2, 1); near[b0:b0 + 4096] = q; dist[b0:b0 + 4096] = np.sqrt(d2[np.arange(len(q)), q])
            inside = dist <= rad[near]                     # within the nearest fiber's own spread: its spike
            fine[un[inside]] = np.asarray(tgt)[near[inside]]; n_asg += int(inside.sum())
            far = un[~inside]; far_near = near[~inside]
            # spikes outside every fiber's spread: their own groups first (a sparse cell that never
            # made a fiber), only the rest to the nearest fiber
            if keep_small and far.size >= 2 * keep_small_min:
                try:
                    sub = gmm_split(waves[far], pca_k=pca_k, max_sub=max_sub, mask=mask, basis=basis)
                except Exception:
                    sub = np.zeros(far.size, int)
                for u in np.unique(sub):
                    m = sub == u
                    if m.sum() >= keep_small_min:
                        try:
                            g = fiber_geom(waves[far[m]], res_abs[far[m]], W, nmean, mask, sr, n_grid,
                                           chunk_t0=ct0, chunk_t1=ct1)
                        except Exception:
                            continue
                        g['coarse'] = -1; g['radius_incl'] = float('nan'); g['n_rejected'] = 0
                        g['n_merged'] = 0; g['small'] = 'unsorted'
                        fine[far[m]] = len(geoms); geoms.append(g)
                        if shed is not None: shed['small_kept'] = shed.get('small_kept', 0) + int(m.sum())
                rest = fine[far] < 0
            else:
                rest = np.ones(far.size, bool)
            fine[far[rest]] = np.asarray(tgt)[far_near[rest]]; n_asg += int(rest.sum())
        elif un.size and not geoms:
            try:
                g = fiber_geom(waves[un], res_abs[un], W, nmean, mask, sr, n_grid, chunk_t0=ct0, chunk_t1=ct1)
                g['coarse'] = -1; g['radius_incl'] = float('nan'); g['n_rejected'] = 0; g['n_merged'] = 0
                fine[un] = len(geoms); geoms.append(g); n_asg += int(un.size)
            except Exception:
                pass
        if shed is not None and n_asg: shed['assigned'] = shed.get('assigned', 0) + n_asg
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


def _chunk_hi(c, t_min, chunk_s, nchunks):
    """The exclusive end of chunk c's core.  The last chunk has no end: nchunks = ceil(span / chunk), so when
    the span is an exact multiple of the chunk length the last spike sits ON t_min + nchunks*chunk_s and a
    `res < hi` test left it out of every chunk (it was written to cluster 0)."""
    return t_min + (c + 1) * chunk_s if c < nchunks - 1 else np.inf


def _merge_vetoes(edges, fine, waves, mask, res_abs, sr, *, cfiber_gate=False, cfiber_q=0.90, off_thr=0.0,
                  off_amp_frac=0.3, waves_std=None, refrac_ms=0.0, refrac_thr=0.3, refrac_min_exp=5.0,
                  refrac_censor_ms=0.0):
    """Every veto a within-chunk fragment merge must pass, in one place (Blocks A, B and C):
      * cfiber SHAPE (cfiber_gate; calibrated per chunk, abstains on small fragments);
      * inter-channel TIMING (off_thr > 0, needs the raw templates waves_std);
      * REFRACTORY (refrac_ms > 0): two trains that coincide at chance level are two cells
        (power-aware: abstains at low rate, so it only ever removes a false merge).
    Returns the edges that survive."""
    edges = [(int(a), int(b)) for a, b in edges]
    if not edges:
        return edges
    if cfiber_gate:
        edges = [(int(a), int(b)) for a, b in _cfiber_edge_filter(edges, fine, waves, mask, q=cfiber_q)]
    if edges and off_thr > 0 and waves_std is not None:
        edges = [(int(a), int(b)) for a, b in _offset_edge_filter(edges, fine, waves_std, off_thr,
                                                                  amp_frac=off_amp_frac)]
    if edges and refrac_ms and refrac_ms > 0:
        refr = cg.refrac_samples(refrac_ms, sr)
        cens = cg.refrac_samples(refrac_censor_ms, sr)
        dur = float(res_abs.max() - res_abs.min()) if len(res_abs) else 0.0

        def _two_cells(ai, bi):
            ta = np.sort(res_abs[np.flatnonzero(fine == ai)].astype(np.int64))
            tb = np.sort(res_abs[np.flatnonzero(fine == bi)].astype(np.int64))
            g = cg.refractory_gate(ta, tb, dur, refr, thr=refrac_thr, min_exp=refrac_min_exp, censor=cens)
            return g["verdict"] == "veto"
        edges = [(a, b) for a, b in edges if not _two_cells(a, b)]
    return edges


def _complete_link(edges, scores=None):
    """The merges `edges` allow under COMPLETE linkage: edges are taken best-first (by `scores`, higher =
    more similar) and two groups join only when EVERY cross pair is itself an edge.  Union-find on the raw
    edges chained A~B, B~C into A+B+C even when A and C fail the threshold or a veto -- a false merge,
    which the pipeline must never prefer to an over-split.  Returns the edges to union (one per join)."""
    E = {(min(int(a), int(b)), max(int(a), int(b))) for a, b in edges}
    if not E:
        return []
    sc = {(min(a, b), max(a, b)): v for (a, b), v in (scores or {}).items()}
    order = sorted(E, key=lambda e: (-sc.get(e, 0.0), e))
    group = {}

    def members(x):
        return group.get(x, frozenset((x,)))
    out = []
    for a, b in order:
        A, B = members(a), members(b)
        if b in A:
            continue
        if all((min(x, y), max(x, y)) in E for x in A for y in B):
            M = A | B
            for x in M:
                group[x] = M
            out.append((a, b))
    return out


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


def _sliding_pairs(Xs, n_win=14, min_spikes=15, min_shared=2, min_cos=0.90, return_scores=False):
    """Sliding window over radius: per-fragment mean direction in each radius
    window, compared window-by-window weighted by spike count x angular
    concentration (down-weights noisy low-energy windows). Returns (i,j) pairs
    whose weighted mean cosine over shared windows exceeds min_cos -- or, with
    return_scores, {(i, j): cosine} for those pairs."""
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
            sc = float((wts * coss).sum() / wts.sum())
            if sc > min_cos: pairs.append((i, j, sc))
    if return_scores:
        return {(i, j): sc for i, j, sc in pairs}
    return [(i, j) for i, j, _ in pairs]


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
    anchor_links = [[] for _ in range(max(nC - 1, 0))]   # per adjacent pair: the accepted (f,g) matches (exact shared-spike correspondences)
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
            union((c, f), (c + 1, g)); anchor_links[c].append((int(f), int(g)))
    roots = {}; gid = {}
    for c in range(nC):
        for l in {int(x) for x in ext_lab[c] if x >= 0}:
            r = find((c, l)); roots.setdefault(r, len(roots)); gid[(c, l)] = roots[r]
    return gid, len(roots), anchor_links


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






# ── cluster and atom ids ─────────────────────────────────────────────────────
# 0 and 1 are Klusters' artefact and noise bins, and every later stage keeps them as such:
# intrachunk reserve=(0, 1), consensus parks its quarantine in 1, xcorr-merge and backbone-link read
# units as clu > 1, anchor-link holds atoms 0/1 out of the link and folds flat-label-1 spikes into
# noise.  So the first fiber AND the first atom are 2.  (Until 0.51 global fiber 0 was written as
# cluster 1 and the first atom as 1: two real fragments of every sort fell into the noise bin
# downstream -- 308 + 242 spikes on sirotaA-jg-000005-20120316 group 6.)
FIBER0 = 2


def _fiber_id(gid, key):
    """The pre-save cluster id of local fiber `key` = (chunk, local label): gid + FIBER0, or 0 when the
    label is not a linked fiber (noise / unlabelled)."""
    g = gid.get(key)
    return int(g) + FIBER0 if g is not None and g >= 0 else 0


def _final_ids(pre, post):
    """{pre-save cluster id: id in the written .clu}.  The hierarchy save compacts ids (0 and 1 kept),
    so a fiber's .clu id is not gid + FIBER0 once any fiber vanished: read it back from what was written
    instead of assuming it."""
    pre = np.asarray(pre, np.int64); post = np.asarray(post, np.int64)
    if pre.size == 0:
        return {}
    pairs = np.unique(np.stack([pre, post], 1), axis=0)
    return {int(a): int(b) for a, b in pairs}


# ── chunk-level parallelism ──────────────────────────────────────────────────
# Each chunk is independent (its own whitener, clustering, geometry), so chunks
# are the natural coarse-grain parallel axis — the same flatten-the-independent-
# axis strategy kiloklustakwik uses for its CEM runs.  Workers reopen the .spkD
# and .fil memmaps from disk (cheap; OS page cache is shared) so no memmap or
# big array is pickled across the process boundary; only per-chunk spike indices
# cross it.  Results are independent of worker count and of completion order
# (cluster_chunk_fine is seeded and per-chunk), so jobs>1 is identical to serial.
_CTX = {}


def _limit_worker_threads():
    """Pin BLAS/OpenMP to one thread per worker process.

    Each worker is already one unit of process parallelism, but NumPy's BLAS
    defaults its thread pool to the core count -- so N workers on an N-core box
    ask for N*N threads, and every matmul pays an N-way fork/join barrier on
    arrays far too small to need one.  On a 64-core machine with 64 draws that is
    4096 threads contending for 64 cores.  Set before the first BLAS call: most
    backends read these at load time, so this must run in the initializer, not
    later.  An explicit environment setting by the caller wins.
    """
    import os
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, "1")
    try:                     # already-loaded backends need telling directly
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        pass


def _init_pool_worker(cfg):
    """Pool initializer for a POOL WORKER: pin BLAS to one thread, then set up
    the usual context.  Kept separate from _init_chunk_worker because the parent
    also calls that one, and the parent runs the consensus/template matching that
    genuinely wants threads."""
    _limit_worker_threads()
    _init_chunk_worker(cfg)


def _init_chunk_worker(cfg):
    """Pool initializer: stash the static config and open the memmaps once per
    worker process.  Also runs (with a fresh dict) for the serial jobs==1 path."""
    global _FAST_STATS
    _FAST_STATS = bool(cfg.get("fast_stats", False))
    _CTX.clear(); _CTX.update(cfg)
    if cfg.get("gpu"):
        _bk.use_gpu(True)
    _CTX["spk"], spkpath = nio.open_spkD(cfg["base"], cfg["elec"], cfg["nsamp"], cfg["nchan"],
                                         prefer=cfg.get("method"))
    # The .fil whitener is used when no other source is: no --no-whiten, no .nst whitener, and
    # --whiten-source fil or auto.  Its noise model is built in the space of the .spk this worker
    # reads (fil_chunk_whitener); the nst / spk sources need no transform (and an opaque variant
    # such as fiber-adlp's has none).
    _use_fil = (not cfg.get("no_whiten") and cfg.get("nst_W") is None
                and cfg.get("whiten_source", "fil") in ("fil", "auto"))
    _CTX["space_tf"] = fst.space_transform(
        nio.spk_variant(spkpath, cfg["base"], cfg["elec"]), cfg.get("sdiff_pairs")) if _use_fil else None
    _CTX["spk_std"] = None
    if float(cfg.get("cf", {}).get("off_thr", 0.0)) > 0:      # inter-channel-timing gate needs RAW templates
        try:
            _CTX["spk_std"], _ = nio.open_spk(cfg["base"], cfg["elec"], cfg["nsamp"], cfg["nchan"],
                                              prefer=nio.prefer_standard())
        except FileNotFoundError:
            _CTX["spk_std"] = None                            # no .spk.standard -> gate silently inert
    # the .fil is opened only when something reads it: the .fil whitener or deconvolution
    _CTX["filmm"] = nio.open_signal(cfg["fil"], cfg["ntotal"]) if (_use_fil or cfg.get("deconv")) else None


def _process_chunk(task):
    """Cluster one chunk.  task = (c, ext, res_e); returns (c, ext, lab, geoms,
    cand, shed).  Reads everything else (memmaps + params) from _CTX."""
    c, ext, res_e = task
    ctx = _CTX; kw = ctx["cf"]
    waves = np.asarray(ctx["spk"][ext], dtype=float)
    waves_std = np.asarray(ctx["spk_std"][ext], dtype=float) if ctx.get("spk_std") is not None else None
    s0 = int(res_e.min()) - ctx["nsamp"]; s1 = int(res_e.max()) + ctx["nsamp"] + 1
    if ctx.get("no_whiten"):
        # Whitening OFF: cluster in the raw (mask-selected, mean-centred) feature space.  Every
        # downstream use is `(X - nmean) @ W`, so an identity W makes them all no-ops while keeping the
        # mean-centring the ridge/mean-shift assumes.  The radial coordinate r = |X| is then the raw
        # feature norm (~amplitude in data units) rather than the whitened Mahalanobis radius, and the
        # angular metric is raw cosine -- i.e. no baseline-covariance normalisation across channels.
        # Size W from the actual feature matrix (mask is an INDEX array, so len(mask)*nch, not a sum).
        Xraw = fl.realign(waves)[:, ctx["mask"], :].reshape(len(waves), -1)
        W = np.eye(Xraw.shape[1])
        nmean = Xraw.mean(0)
    elif ctx.get("nst_W") is not None:
        W, nmean = ctx["nst_W"], ctx["nst_mean"]
    elif ctx.get("whiten_source") == "spk":
        W, nmean, _ = fl.spk_chunk_whitener(waves, res_e, ctx["mask"], ctx["peak"])
    else:
        W, nmean, _ = fil_chunk_whitener(ctx["filmm"], ctx["gch"], s0, s1, res_e, ctx["nsamp"], ctx["mask"],
                                         transform=ctx["space_tf"])
    cand = []
    sd = {}
    lab, geoms = cluster_chunk_fine(waves, res_e, W, nmean, ctx["min_group"], ctx["mask"], ctx["sr"],
                                    candidates_out=cand, shed=sd, waves_std=waves_std, **kw)
    sd["_unsorted_ext"] = int((lab < 0).sum())
    dcv = _deconv_chunk(ctx, res_e, lab, geoms, s0, s1) if ctx.get("deconv") else None
    return c, ext, lab, geoms, cand, sd, dcv


def _deconv_chunk(ctx, res_e, lab, geoms, s0, s1):
    """fiber_deconv on one chunk's detections against its own fibers.  Adds
    deconv_agree / deconv_partner_frac to each geom; returns a dict of
    per-detection arrays (ext order) plus the rescue mask, or {'error': msg}
    (a failed chunk is reported and skipped, never fatal to the sort)."""
    try:
        from . import fiber_deconv as fd
    except ImportError:
        import fiber_deconv as fd
    dc = ctx["deconv"]
    try:
        excl = {l for l, g in enumerate(geoms) if g.get("collision") or g.get("undefined")}
        read_raw = fd.memmap_reader(ctx["filmm"], ctx["gch"])
        r = fd.deconvolve(read_raw, res_e, lab, ctx["nsamp"], ctx["peak"], ctx["sr"],
                          s0=s0, s1=s1, R=dc["phases"], order=dc["order"],
                          refrac_ms=dc["refrac_ms"], null_q=dc["null_q"], thr_floor=dc["floor"],
                          min_n=dc["min_n"], exclude=excl, equiv_cos=dc["equiv_cos"])
    except Exception as e:                                   # noqa: BLE001 -- report, keep the sort
        return {"error": f"{type(e).__name__}: {e}"}
    lab = np.asarray(lab)
    in_excl = np.isin(lab, list(excl)) if excl else np.zeros(len(lab), bool)
    # Agreement up to template EQUIVALENCE: fragments of one cell share a template, and which of them
    # the pursuit names is not a disagreement with the clustering.
    agree_sp = np.array([int(p_) in r.equiv.get(int(l), ()) for l, p_ in zip(lab, r.primary)])
    good = set()
    for l, g in enumerate(geoms):
        own = (lab == l) & (r.primary >= 0)
        ag = float(np.mean(agree_sp[own])) if own.any() and l in r.equiv else float("nan")
        g["deconv_agree"] = ag
        g["deconv_partner_frac"] = float(np.mean(r.partner[lab == l] >= 0)) if (lab == l).any() else float("nan")
        g["deconv_n_equiv"] = int(len(r.equiv.get(l, ())))
        if np.isfinite(ag) and ag >= dc["fiber_agree"] and own.sum() >= 20 and l not in excl:
            good.add(l)
    # Calibrate the margin on the fibers where pursuit and clustering corroborate each other.  On a
    # contaminated fiber the disagreements are the CLUSTERING's errors (deconv_agree tracks purity), so
    # they would only push the margin to infinity; and rescues go only into corroborated fibers.
    in_good = np.isin(lab, list(good)) if good else np.zeros(len(lab), bool)
    clean = in_good & (r.primary >= 0)
    m_thr = fd.calibrate_margin(r.margin[clean], agree_sp[clean],
                                target=dc["agree"], floor=dc["margin_min"])
    prim_good = np.isin(r.primary, list(good)) if good else np.zeros(len(lab), bool)
    # A partner within the anchoring tolerance means the detection IS two coincident spikes: either
    # label is right, so neither is written (the manifest lists both).
    coincident = (r.partner >= 0) & (np.abs(r.partner_dt) <= 4)
    rescue = ((lab < 0) | in_excl) & prim_good & (r.margin >= m_thr) & ~coincident
    return dict(primary=r.primary, margin=r.margin, dt=r.dt, amp=r.amp, gain=r.gain,
                partner=r.partner, partner_dt=r.partner_dt, rescue=rescue, m_thr=m_thr, thr=r.thr,
                placements=r.placements)


def add_core_arguments(ap):
    """Register every fiber-session clusterer argument on `ap` (after sy.add_session_args).
    Shared by main() and the stochastic harness so both expose an identical clusterer."""
    _gcfg = cfgmod.load_global_config()
    def _fk_default(name, fb):
        """CLI default = FK_* env, else the global fiber-kit.yaml, else fallback (config.knob_value)."""
        return cfgmod.knob_value(name, float, fb, _gcfg)
    ap.add_argument("--chunk-min", "--chunk-minutes", type=float, default=_fk_default("FK_SESSION_CHUNK_MIN", 12.0)); ap.add_argument("--overlap-min", type=float, default=_fk_default("FK_SESSION_OVERLAP_MIN", 4.0))
    ap.add_argument("--min-group", type=int, default=200, help="COARSE min spikes/fiber (for linking)")
    ap.add_argument("--fine-algo", "--fine-method", dest="fine_method", choices=["gmm","rkk","fiber","none"], default="rkk",
                    help="per-fiber fine clusterer: rkk (default, masked-EM KlustaKwik with per-cluster "
                         "realign + BIC model selection, robust to unequal covariances), gmm (Gaussian-mixture "
                         "split), fiber (fiber-kit's native cluster_chunk), none (no fine split -- keep the "
                         "coarse cluster whole)")
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
    ap.add_argument("--refit-iters", type=int, default=0,
                    help="iterate {realign each fiber's spikes to their own median -> refit template -> re-apply the "
                         "inclusion gate} this many extra times.  Recovers true members rejected for a bad sub-sample "
                         "offset and tightens the template; re-judges from the full member set each pass.  0 = single "
                         "pass (original).  Marginal on clean fibers; helps contamination-heavy ones.")
    ap.add_argument("--reseed-residual-thr", type=float, default=0.0,
                    help="residual-gated reseed: after the fine split, any fiber whose median-subtracted "
                         "residual/signal exceeds this is re-split (a welded pair of co-located different-shape "
                         "cells scores high; ~0.35 separates welded from clean on g5).  0 = off.  The re-split is "
                         "kept only if it lowers the residual by --reseed-min-reduction.")
    ap.add_argument("--reseed-min-reduction", type=float, default=0.05,
                    help="keep a residual-gated reseed only if it lowers the median-subtracted residual/signal "
                         "by at least this much (a real sub-unit does; noise does not)")
    ap.add_argument("--seed-density", type=float, default=0.0,
                    help="density-preferential coarse seeding: draw ridge seeds toward concentrated modes of the "
                         "waveform space (p proportional to local_density**this) instead of a uniform stride.  0 = "
                         "uniform (current); 1 = fully density-weighted; small values a gentle tilt.  Concentrates "
                         "the seed budget on isolated, reproducible modes (isolation, not firing rate).")
    ap.add_argument("--coarse-dr", type=float, default=0.15,
                    help="radial band half-width (fraction of the 1-99%% whitened-radius span) for spike->seed "
                         "association in the COARSE pass.  Lower -> tighter amplitude bands -> more, smaller coarse "
                         "fibers (amplitude-distinct cells stay separate for the fine splitter); higher -> fewer, "
                         "fatter coarse groups.  Default 0.15.")
    ap.add_argument("--whiten-source", dest="whiten_source", choices=["auto", "nst", "fil", "spk"],
                    default=cfgmod.knob_value("FK_SESSION_WHITEN_SOURCE", str, "auto", _gcfg,
                                              ("auto", "nst", "fil", "spk")),
                    help="FK_SESSION_WHITEN_SOURCE.  Where the noise whitener comes from.  nst = the "
                         ".nst.<variant>.<group> ndm_extractspikes writes beside the .spk: noise measured in the "
                         ".spk's own transform on spike-free windows, outliers rejected (one whitener for the "
                         "session).  fil = per chunk, off-spike .fil baseline taken into the .spk's waveform "
                         "space (fiber_stderiv.space_transform; needs the .fil, and the sdiffPairs for a _C "
                         "variant).  spk = per chunk, the snippets' own pre-peak samples (stationary "
                         "block-Toeplitz covariance): needs neither file and follows any variant, including ones "
                         "no transform describes (fiber-adlp's per-spike filtered snippets).  auto (default) = "
                         "nst when the .spk's .nst exists, else fil.  --deconv needs the .fil either way.")
    ap.add_argument("--no-whiten", dest="no_whiten", action="store_true",
                    help="cluster in the RAW mask-selected (mean-centred) feature space instead of the .fil "
                         "baseline-whitened space: identity whitener, so the radial coordinate is the raw feature "
                         "norm and the angular metric is raw cosine (no per-channel covariance normalisation)")
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
    ap.add_argument("--off-thr", dest="off_thr", type=float, default=_fk_default("FK_SESSION_OFF_THR", 0.0),
                    help="inter-channel-TIMING co-veto on Block-A fragment merges: refuse a footprint-correlation "
                         "merge whose two fibers' RAW-template inter-channel offsets disagree by more than this "
                         "(xcorr LAG samples, ~0.2-0.3; NOT the trough ~1.0). Reads .spk.standard for raw templates "
                         "(stderiv timing is channel-mixed); inert if that file is absent. FK_SESSION_OFF_THR. 0 = OFF (default)")
    ap.add_argument("--off-amp-frac", dest="off_amp_frac", type=float, default=_fk_default("FK_SESSION_OFF_AMP_FRAC", 0.3),
                    help="channels below this fraction of the dominant peak-to-peak carry no reliable timing for --off-thr. FK_SESSION_OFF_AMP_FRAC")
    ap.add_argument("--merge-algo", "--merge-method", dest="merge_method", choices=["template","sliding","profile"], default="template")
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
    ap.add_argument("--deconv", dest="deconv", action=argparse.BooleanOptionalAction,
                    default=bool(_fk_default("FK_SESSION_DECONV", 0.0)),
                    help="detection-anchored sub-sample template deconvolution per chunk (fiber_deconv): "
                         "per-fiber deconv_agree contamination QC in .fibers and a .deconv.tsv manifest of "
                         "labels, collision partners and undetected spikes.  Changes no label unless "
                         "--deconv-assign")
    ap.add_argument("--deconv-assign", dest="deconv_assign", action=argparse.BooleanOptionalAction,
                    default=bool(_fk_default("FK_SESSION_DECONV_ASSIGN", 0.0)),
                    help="with --deconv: move confidently attributed noise / collision-cluster spikes into "
                         "their fiber as a parked child atom.  Off by default: on synthetic sessions with "
                         "look-alike cells its precision fell to 70-90 percent (docs/methods.md)")
    ap.add_argument("--deconv-phases", type=int, default=4, help="sub-sample phases per template (1 = whole samples)")
    ap.add_argument("--deconv-order", type=int, default=4, help="order of the VAR noise-whitening filter")
    ap.add_argument("--deconv-refrac-ms", type=float, default=1.0,
                    help="a unit is not placed twice within this many ms")
    ap.add_argument("--deconv-floor", type=float, default=16.0,
                    help="lowest pursuit gain (2 x log-likelihood) accepted, whatever the noise null says")
    ap.add_argument("--deconv-null-q", type=float, default=0.995,
                    help="quantile of the off-spike first-step gain used as the stopping threshold")
    ap.add_argument("--deconv-agree", type=float, default=0.999,
                    help="margin calibration: agreement with the clustering required among clean spikes above it")
    ap.add_argument("--deconv-margin-min", type=float, default=10.0,
                    help="floor on the calibrated label margin (2 x log-likelihood ratio, best vs other unit)")
    ap.add_argument("--deconv-min-n", type=int, default=30, help="min spikes for a fiber to get a template")
    ap.add_argument("--deconv-fiber-agree", type=float, default=0.95,
                    help="a fiber whose spikes the pursuit labels with an equivalent template at least this often is "
                         "corroborated: the margin is calibrated on such fibers and only they receive rescues")
    ap.add_argument("--deconv-equiv-cos", type=float, default=0.93,
                    help="fibers whose templates reach this whitened cosine (noise-corrected, best over +/-4 "
                         "samples) are equivalent: the over-split fragments of a cell.  Agreement, margin and "
                         "collision partners are judged up to equivalence.  >1 = identity only")
    ap.add_argument("--quality-metrics", action="store_true", help="also compute L-ratio + isolation distance (O(N*K))")
    ap.add_argument("--quality-dims", type=int, default=10, help="PCA dims for L-ratio/isolation Mahalanobis")
    ap.add_argument("--pca-k", type=int, default=6); ap.add_argument("--max-sub", type=int, default=8)
    ap.add_argument("--inclusion-k", type=float, default=3.0, help="per-fiber radius = median+k*MAD of residuals; 0 disables")
    ap.add_argument("--no-noise", dest="no_noise", action="store_true", default=False,
                    help="sweep every remaining noise spike (below the inclusion radius / rejected / collision junk) into a "
                         "single UNDEFINED FIBER (one real cluster, not the noise cluster) rather than dropping it. For clean "
                         "stderiv data; the undefined fiber is cleaned/re-split in later steps.")
    ap.add_argument("--keep-small-groups", dest="keep_small", action="store_true", default=False,
                    help="a fine group below --fine-min-group (or whose inclusion core falls below it) becomes a fiber "
                         "of its own -- all its spikes -- instead of going to the noise cluster.  Over-split is "
                         "recoverable downstream; a dropped unit is not.  Groups below --keep-small-min are left "
                         "to --assign-unsorted")
    ap.add_argument("--keep-small-min", dest="keep_small_min", type=int, default=5,
                    help="smallest group --keep-small-groups keeps as its own fiber (default 5)")
    ap.add_argument("--assign-unsorted", dest="assign_unsorted", action="store_true", default=False,
                    help="place every spike still unsorted at the end of a chunk (coarse-unassigned, groups too "
                         "small to keep): within its nearest fiber's own spread -> that fiber; outside every fiber's "
                         "spread -> GMM groups of their own (with --keep-small-groups), the remainder -> nearest fiber "
                         "(whitened, realigned features: shape and amplitude).  With fiber-flag-artifacts -> --exclude-clu, only flagged artifacts reach "
                         "cluster 0")
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
    ap.add_argument("--out-variant", default=None,
                    help="ALSO write .clu/.clc/.clp under this variant token, in addition to the "
                         "--method-named copies.  READS are unaffected (the .pca basis and .spk still "
                         "resolve under --method), so an alias like stderiv_C5_D34 can name a feature "
                         "space without needing its own basis or waveform file.")
    ap.add_argument("--emit-pca", action="store_true",
                    help="with --emit-fet: also write <base>.pca.<out-variant or method>.<elec>, the basis "
                         "the .fet is the projection of, so Klusters can realign/reproject in that space")
    ap.add_argument("--emit-fet", action="store_true",
                    help="write <base>.fet.<out-variant or method>.<elec> -- the FULL-WIDTH session-wide "
                         "projection on the global basis (the feature space the fine split used, before "
                         "its per-chunk reduction), so Klusters can display it.")
    ap.add_argument("--feat-lag", type=int,
                    default=int(os.environ.get("FK_SESSION_FEAT_LAG") or
                                (cfgmod.load_global_config() or {}).get("FK_SESSION_FEAT_LAG") or 0),
                    help="FK_SESSION_FEAT_LAG. >0 = fine-split shape features become PC1 sampled at "
                         "-N/0/+N samples plus PC2 (4 per channel) instead of PC1,PC2,PC3.  0 = classic.")
    ap.add_argument("--feat-lag-pc2", type=int, default=1,
                    help="with --feat-lag: 1 (default) keep PC2 as a 4th column per channel, 0 = lags only")
    ap.add_argument("--exclude-clu", default=None,
                    help="per-spike exclusion mask in .clu form (nonzero = drop), aligned to the .res -- "
                         "e.g. fiber-flag-artifacts' output.  Excluded spikes never enter a chunk, so they "
                         "cannot shape a cluster; they are emitted as cluster 0.")
    ap.add_argument("--in-stage", dest="in_stage", default=None,
                    help="input stage: the clu <base>.clu.<in-method>.<elec>.<in-stage> that --exclude-units "
                         "names clusters of ('' = the stage-less <base>.clu.<in-method>.<elec>).  It must be "
                         "aligned to the .res this run reads (same spike count); a stage with its own .res "
                         "(e.g. decollided) is refused")
    ap.add_argument("--in-method", dest="in_method", default=None,
                    help="variant token of the input-stage clu (default: --method), e.g. the "
                         "--out-variant alias a curated clu was saved under")
    ap.add_argument("--exclude-units", dest="exclude_units", default=None,
                    help="clusters of the input-stage clu whose spikes are excluded from the sort: "
                         "comma-separated ids and ranges, e.g. 3,7,12-15.  Excluded spikes never enter a "
                         "chunk; see --exclude-as for what the output holds for them")
    ap.add_argument("--exclude-as", dest="exclude_as", choices=["carry", "noise"], default="carry",
                    help="carry (default): each excluded unit is written back as one cluster of its own "
                         "(one atom), after the new fibers; ids 0 and 1 of the input go to 0.  "
                         "noise: excluded spikes are cluster 0, like --exclude-clu")
    ap.add_argument("--no-cluster-basis", action="store_true",
                    help="ignore the global .pca basis for the fine-split shape features and use a "
                         "per-call local SVD (legacy behaviour)")
    ap.add_argument("--clu-stage", dest="clu_stage", default="fiber_session",
                    help="post-group stage tag for the clu: <base>.clu.<method>.<elec>[.<stage>] "
                         "(default 'fiber_session'); pass --clu-stage '' for an untagged .clu")
    ap.add_argument("--skip-extant", dest="skip_extant", type=int,
                    default=int(os.environ.get("FK_SESSION_SKIP_EXTANT") or
                                (cfgmod.load_global_config() or {}).get("FK_SESSION_SKIP_EXTANT") or 0),
                    help="FK_SESSION_SKIP_EXTANT.  1 = if the destination .clu this run would write "
                         "already exists (--out, else <base>.clu.<out-variant or method>.<elec>"
                         "[.<clu-stage>]), log it and exit before any compute -- a plan re-run "
                         "resumes past its fiber-session step instead of regenerating hours of "
                         "sort over the top of an extant (possibly hand-curated) one.  Delete "
                         "the file, or set 0 (the default), to rebuild.")
    ap.add_argument("--emit-hierarchy", dest="emit_hierarchy", action=argparse.BooleanOptionalAction, default=True,
                    help="emit the .clu/.clc/.clp microfiber triple (atoms = pre-link fine fragments, "
                         "fibers = linked global ids) via FiberHierarchy, instead of a flat .clu only. "
                         "--no-emit-hierarchy writes just the flat .clu (legacy). Ignored with --out.")
    ap.add_argument("--gpu", action="store_true", help="run the realign/whiten kernels on GPU (CuPy; needs the [gpu] extra)")
    ap.add_argument("--jobs", "-j", type=int, default=1,
                    help="parallel worker processes over chunks (default 1 = serial; chunks are independent)")
    ap.add_argument("--feature-align", dest="feature_align", choices=["xcorr", "centroid", "off"], default=None,
                    help="feature-building alignment: xcorr (default), centroid (pure, no refine -- "
                         "adds the trough-position-vs-asymmetry structure to the clustering/linking "
                         "features), or off (skip the sub-sample align entirely; features are built from "
                         "the windows exactly as extracted).  Does NOT touch committing alignment or "
                         "fiber-realign.  Overrides the FIBER_ALIGN env var.")
    ap.add_argument("--subsample", dest="subsample", action=argparse.BooleanOptionalAction, default=None,
                    help="enable (--subsample) or disable (--no-subsample) realign's per-spike "
                         "sub-sample (parabolic) refine in the feature build; default leaves the "
                         "FIBER_SUBSAMPLE env var / lever untouched (off).  Reaches pool workers.")
    ap.add_argument("--out", default=None)
    fcons.add_consolidate_args(ap, _gcfg, stage="SESSION")   # per-spike strip+knn cleanup of the
                                            # finished labels (default off; FK_SESSION_CONS_MODE scopes the mode)
    return ap


def build_cf(a, meth, cluster_basis):
    """Assemble the cluster_chunk_fine keyword dict from parsed args.  Extracted from main()
    so the stochastic harness configures the clusterer identically to the production path."""
    return dict(method=meth, coarse_dr=a.coarse_dr, coarse_seed_density=a.seed_density, reseed_residual_thr=a.reseed_residual_thr, reseed_min_reduction=a.reseed_min_reduction, refit_iters=a.refit_iters, fine_kappa=a.fine_kappa, fine_dedup=a.fine_dedup_deg,
        fine_mg=a.fine_min_group, pca_k=a.pca_k, max_sub=a.max_sub, n_grid=a.n_grid, basis=cluster_basis,
        incl_k=a.inclusion_k, incl_assign=a.incl_assign, no_noise=a.no_noise, cone_channel_k=a.cone_channel_k,
        keep_small=a.keep_small, keep_small_min=a.keep_small_min, assign_unsorted=a.assign_unsorted,
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
        off_thr=a.off_thr, off_amp_frac=a.off_amp_frac,
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


def _park_consolidated(clu, new_clu, child, parent, next_atom, park_min=0):
    """Fold a consolidation pass's per-spike moves into the atom hierarchy.

    Per-spike moves cannot be expressed in the atom->fiber map, so every moved
    (source atom -> destination fiber) bucket becomes its own NEW child atom
    under the destination fiber — the same parking fiber-anchor-link uses —
    keeping each move a Klusters-inspectable, revertible atom.  `child` is
    patched in place; `parent` (atom -> 1-based fiber id) gains the new atoms.
    Returns (clu, next_atom) with the moves applied.

    park_min (FK_CONS_PARK_MIN): with an iterated strip, most (src -> dst)
    buckets hold one or two spikes, and per-bucket parking explodes the atom
    count (a lab session: 2,368 atoms for 996 fibers, median bucket size 1 —
    a wall of micro-fragments in Klusters' tree).  Buckets BELOW this many
    spikes collapse into ONE shared sweep atom per DESTINATION fiber instead:
    still inspectable and revertible at the destination, without a tree entry
    per stray spike.  0 (default) keeps the historical per-bucket parking.
    The flat clu is identical either way — only the atom structure changes."""
    new_clu = np.asarray(new_clu)
    m = new_clu != np.asarray(clu)
    if m.any():
        key = np.stack([np.asarray(child)[m], new_clu[m]], 1)
        uk, inv = np.unique(key, axis=0, return_inverse=True)
        if park_min and park_min > 1:
            small = np.bincount(inv) < park_min
            if small.any():
                uk2 = uk.copy(); uk2[small, 0] = -1        # per-dest sweep slot
                uk, inv2 = np.unique(uk2, axis=0, return_inverse=True)
                inv = inv2[inv]
        child[m] = next_atom + inv
        for j, fid in enumerate(uk[:, 1]):
            parent[int(next_atom + j)] = int(fid)
        next_atom += len(uk)
    return new_clu.astype(np.int32), next_atom


def parse_unit_list(text):
    """'3,7,12-15' -> [3, 7, 12, 13, 14, 15] (sorted, unique).  Whitespace is
    ignored; an empty string is an empty list.  Raises ValueError on anything
    that is not a non-negative integer or an ascending a-b range."""
    out = set()
    for tok in str(text).replace(" ", "").split(","):
        if not tok:
            continue
        lo, sep, hi = tok.partition("-")
        if not lo.isdigit() or (sep and not hi.isdigit()):
            raise ValueError(f"--exclude-units: '{tok}' is not an id or an a-b range")
        lo = int(lo); hi = int(hi) if sep else lo
        if hi < lo:
            raise ValueError(f"--exclude-units: range '{tok}' runs backwards")
        out.update(range(lo, hi + 1))
    return sorted(out)


def load_input_stage(base, elec, method, stage, nspk, units):
    """Read the input-stage clu and resolve `units` against it.

    Returns (in_clu, in_path, excluded_mask, present, missing).  Refuses
    (SystemExit) when the clu does not have one label per spike of the .res
    this run reads -- a stage that carries its own .res (de-collision adds
    spikes) would otherwise exclude the wrong spikes by index."""
    path = nio.session_path(base, "clu", elec, variant=method, tag=stage)
    try:
        _, in_clu = nio.read_clu_at(base, elec, variant=method, tag=stage)
    except FileNotFoundError as e:
        raise SystemExit(f"--in-stage: {e}") from None
    in_clu = np.asarray(in_clu, np.int64)
    if in_clu.size != nspk:
        raise SystemExit(f"--in-stage: {os.path.basename(path)} has {in_clu.size:,} labels but the .res has "
                         f"{nspk:,} spikes -- the stage carries its own .res (e.g. decollided) or belongs to "
                         f"another detection; fiber-session reads the shared .res and cannot follow it")
    present = [u for u in units if np.any(in_clu == u)]
    missing = [u for u in units if u not in present]
    excl = np.isin(in_clu, present) if present else np.zeros(nspk, bool)
    return in_clu, path, excl, present, missing


def _carry_units(clu, child, parent, next_atom, in_clu, units):
    """Write each excluded unit back as one cluster of its own: a new atom per
    unit, parented to a fresh fiber id after every existing one.  Units 0 and 1
    (Klusters' artefact and noise) stay in cluster 0.  Returns (clu, next_atom,
    {unit: provisional fiber id}); the hierarchy save renumbers, so final ids
    are read back from the written clu."""
    clu = np.asarray(clu).copy()
    nxt = _fch.next_free_id(np.concatenate([np.asarray(clu, np.int64).ravel(),
                                            np.asarray(list(parent.values()) or [0], np.int64)]))
    prov = {}
    for u in units:
        m = in_clu == u
        if u in (0, 1) or _fch.is_reserved(u) or not m.any():
            continue
        clu[m] = nxt
        child[m] = next_atom
        parent[next_atom] = nxt
        prov[u] = nxt
        nxt += 1; next_atom += 1
    return clu.astype(np.int32), next_atom, prov


def _apply_deconv(a, res, clu, child, parent, next_atom, chunk_deconv, ext_idx, gid,
                  t_min, chunk_s, nchunks, log, det):
    """Fold per-chunk fiber_deconv results into the sort and collect the manifest rows.

    Only CORE spikes of each chunk are taken (overlap spikes belong to the
    neighbour), mirroring how labels are assigned.  Rescued spikes move from
    noise / the collision fiber to the deconvolved fiber's global id and are
    parked as one new child atom per (source atom, fiber) -- reviewable and
    revertible in Klusters.  The manifest lists every core detection the
    pursuit labelled, every collision partner, and every placement that no
    detection anchors (an undetected spike: advisory, never in the .clu).
    Returns (clu, next_atom, rows); the rows carry PRE-SAVE cluster ids and the
    detection times, and _write_deconv_manifest writes them once the sort is
    saved (final ids) and committed (final times)."""
    new_clu = clu.copy()
    n_err = n_resc = n_part = n_un = 0
    mthr = []
    rows = []
    for c in sorted(chunk_deconv):
        d = chunk_deconv[c]
        if "error" in d:
            n_err += 1
            det("deconv", f"chunk {c + 1}: skipped ({d['error']})")
            continue
        lo_s = t_min + c * chunk_s; hi_s = _chunk_hi(c, t_min, chunk_s, nchunks)
        ext = ext_idx[c]
        mthr.append(d["m_thr"])
        for j in np.flatnonzero((res[ext] >= lo_s) & (res[ext] < hi_s)):
            g = int(ext[j])
            pg = _fiber_id(gid, (c, int(d["primary"][j]))) if d["primary"][j] >= 0 else 0
            qg = _fiber_id(gid, (c, int(d["partner"][j]))) if d["partner"][j] >= 0 else 0
            resc = bool(d["rescue"][j]) and pg > 0
            if resc:
                new_clu[g] = pg; n_resc += 1
            n_part += qg > 0
            if pg or qg:
                rows.append((g, int(res[g]), c, int(clu[g]), pg, d["dt"][j], d["amp"][j], d["gain"][j],
                             d["margin"][j], int(resc), qg, d["partner_dt"][j]))
        pl = d["placements"]
        for row in pl[pl["anchor"] < 0]:
            if lo_s <= row["t"] < hi_s:
                ug = _fiber_id(gid, (c, int(row["unit"])))
                rows.append((-1, int(round(row["t"])), c, 0, ug, row["t"] - round(row["t"]), row["amp"],
                             row["gain"], np.nan, 0, 0, np.nan))
                n_un += 1
    if a.deconv_assign and n_resc:
        clu, next_atom = _park_consolidated(clu, new_clu, child, parent, next_atom)
    fin = [m for m in mthr if np.isfinite(m)]
    det("deconv", f"{n_resc:,} spikes {'parked into' if a.deconv_assign else 'attributable to'} fibers"
                  f" \u00b7 {n_part:,} collision partners \u00b7 {n_un:,} unanchored placements"
                  + (f" \u00b7 margin {np.median(fin):.1f} (median)" if fin else " \u00b7 margin: none calibrated")
                  + (f" \u00b7 {n_err} chunk(s) failed" if n_err else ""))
    return clu, next_atom, rows


def _write_deconv_manifest(path, rows, final_of, res_now):
    """Write the deconvolution manifest in the terms of the WRITTEN sort: cluster ids as they are in
    the saved .clu (final_of: pre-save id -> written id; 0 when the fiber is gone), `res` the time in
    the .res as it is now -- the committed one after the end-of-run realign -- and `res_detect` the
    time the pursuit saw.  An unanchored placement (spike -1) has no .res row and keeps its own time.
    Until 0.51 the manifest held pre-renumber ids and pre-commit times, so neither matched the .clu
    or the .res beside it."""
    def fid(v):
        return final_of.get(int(v), 0) if int(v) > 0 else 0
    with open(path, "w") as f:
        f.write("spike\tres\tchunk\tclu\tdeconv_clu\tdt\tamp\tgain\tmargin\trescued"
                "\tpartner_clu\tpartner_dt\tres_detect\n")
        for (g, t_det, c, cl, pg, dt, amp, gain, margin, resc, qg, pdt) in rows:
            t_now = int(res_now[g]) if g >= 0 else int(t_det)
            r_ = (g, t_now, c, fid(cl), fid(pg), dt, amp, gain, margin, resc, fid(qg), pdt, int(t_det))
            f.write("\t".join(f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)
                              for v in r_) + "\n")
    return path


def _knob(name, fallback, choices=None):
    """An FK_* knob: exported env var > $FK_CONFIG yaml > fallback (the pipeline's precedence), checked
    against `choices` when given (an off/on spelling maps onto "off")."""
    return cfgmod.knob_value(name, str, fallback, None, choices)


def add_realign_arguments(ap):
    """The end-of-run commit: fiber-realign's in-place commit on the sort this run just wrote."""
    ap.add_argument("--realign", choices=["auto", "reextract", "shift-spk", "off"],
                    default=_knob("FK_SESSION_REALIGN", "auto", ("auto", "reextract", "shift-spk", "off")),
                    help="FK_SESSION_REALIGN.  At the end of the run, align every spike to its fiber's template "
                         "(fiber-realign's Klusters per-spike offsets, on the .clu this run wrote), commit the "
                         "shifted spike times to the shared .res (+ <base>.offsets.<group>.npy), and rebuild EVERY "
                         "waveform variant's .spk on disk at the shifted times (standard, stderiv*, ...) and its .fet "
                         "(from its .pca; _D alias .fet included) -- what `fiber-realign --reextract --refeaturize "
                         "--variants all` does as a separate step.  reextract = re-read the .fil; shift-spk = "
                         "circularly roll the existing .spk (no .fil); auto (default) = reextract when the .fil "
                         "exists, else shift-spk; off = leave .res/.spk/.fet as extracted.")
    ap.add_argument("--realign-algo", dest="realign_algo", choices=["klusters", "template", "centroid"],
                    default=_knob("FK_REALIGN_ALGO", "klusters", ("klusters", "template", "centroid")),
                    help="FK_REALIGN_ALGO.  fiber-realign --align-algo for the end-of-run commit; its other knobs "
                         "(FK_REALIGN_MAX_SHIFT / _ITERS / _MIN_SCORE / _RECENTER_RMS / _RMIN) apply as they do there")
    ap.add_argument("--realign-fil", dest="realign_fil", default=None,
                    help="filtered signal the end-of-run re-extraction reads (default <base>.fil)")


def main():
    ap = argparse.ArgumentParser(
        description="Cluster a session group into fibers. Reads <session>.yaml "
                    "(or <session>/<session>.yaml) for channels/sr/nChannels; "
                    "CLI flags override the YAML.")
    sy.add_session_args(ap, sdiff=True)
    add_core_arguments(ap)
    add_realign_arguments(ap)
    _fch.add_argument(ap, cfgmod.load_global_config(), length=False)    # --chunks (FK_CHUNKS); counts in --chunk-min
    cfgmod.check_choice_defaults(ap)
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
                                    nchan=a.nchan, nsamp=a.nsamp, sr=a.sr, sdiff_pairs=a.sdiff_pairs)
    a.base = cfg["base"]; a.elec = a.group
    sdiff = cfg.get("sdiff_pairs")
    yaml_peak = cfg["peak"]                    # (cfg is rebound to the chunk-worker dict below)
    a.ntotal = cfg["ntotal"]; a.nchan = cfg["nchan"]; a.nsamp = cfg["nsamp"]; a.sr = cfg["sr"]
    gch = np.array(cfg["channels"], int)
    assert len(gch) == a.nchan, f"--channels has {len(gch)} entries, nchan={a.nchan}"
    _masks = fl.build_masks(cfg["nsamp"], cfg["peak"])
    mask = _masks.full; p = len(mask) * a.nchan
    peak = int(_masks.offset)                 # the YAML peak, or build_masks' inference without one

    # ── skip-if-extant: never rebuild (and silently overwrite) an existing sort ──
    #   FK_SESSION_SKIP_EXTANT=1: when the destination .clu this invocation would write
    #   already exists -- a previous plan run's output, possibly hand-curated in Klusters
    #   since -- report it and stop BEFORE any compute (and before the working-.spk
    #   resolve, so a session whose products are all under a feature-space alias still
    #   skips cleanly).  The check is the EXACT path as invoked, so plan re-runs resume
    #   past their fiber-session step; delete the file or set the knob to 0 to rebuild.
    if a.skip_extant:
        _dst = a.out or nio.session_path(a.base, "clu", a.elec,
                                         variant=(a.out_variant or a.method), tag=a.clu_stage)
        if os.path.exists(_dst):
            log(f"group {a.elec} · {a.method}")
            det("skip-extant", f"{os.path.basename(_dst)} exists -- nothing regenerated "
                               f"(delete it or set FK_SESSION_SKIP_EXTANT=0 to rebuild)")
            return

    t0 = time.time()
    res = read_res(a.base, a.elec); nspk = len(res)
    keep = np.ones(nspk, bool)
    if a.exclude_clu:
        _, _ex = nio.read_clu_file(a.exclude_clu, n_spikes=nspk)
        _ex = np.asarray(_ex)
        if _ex.size != nspk:
            raise SystemExit(f"--exclude-clu has {_ex.size} labels, .res has {nspk}")
        keep = (_ex == 0) | _fch.is_reserved(_ex)       # a reserved id = a chunk the mask did not evaluate
    # ── input stage + unit exclusions ──
    in_clu = None; excl_units = []
    if a.exclude_units is not None and a.in_stage is None:
        raise SystemExit("--exclude-units names clusters of an input clu: give --in-stage "
                         "(--in-stage '' for the stage-less <base>.clu.<method>.<elec>)")
    if a.in_stage is not None:
        in_method = a.in_method or a.method
        try:
            units = parse_unit_list(a.exclude_units or "")
        except ValueError as e:
            raise SystemExit(str(e)) from None
        in_clu, in_path, excl_mask, excl_units, missing = load_input_stage(
            a.base, a.elec, in_method, a.in_stage, nspk, units)
        outs = [a.out] if a.out else [nio.session_path(a.base, "clu", a.elec, variant=v, tag=a.clu_stage)
                                      for v in (a.method, a.out_variant) if v]
        for o in outs:                                  # never write the sort over the clu it excludes from
            if os.path.abspath(o) == os.path.abspath(in_path):
                raise SystemExit(f"--in-stage: the output {os.path.basename(o)} would overwrite the input "
                                 f"clu -- choose another --clu-stage")
        keep &= ~excl_mask
    spk, spkpath = open_spkD(a.base, a.elec, a.nsamp, a.nchan, prefer=a.method)
    nio.check_spk_count(nspk, spk, spkpath, a.nsamp, a.nchan)
    space = nio.spk_variant(spkpath, a.base, a.elec)
    # ── noise statistics of the .spk actually opened (its own variant token) ──
    nst_W = nst_mean = None; nst_note = ""
    if not a.no_whiten and a.whiten_source in ("nst", "auto"):
        _nstp = nio.session_path(a.base, "nst", a.elec, variant=space) if space else ""
        if _nstp and os.path.exists(_nstp):
            _st = nio.read_nst(_nstp)
            if (_st["nsamp"], _st["nchan"]) != (a.nsamp, a.nchan):
                raise SystemExit(f"{_nstp}: {_st['nsamp']}x{_st['nchan']} window, the .spk is "
                                 f"{a.nsamp}x{a.nchan}")
            nst_W, nst_mean, _nk = fl.nst_whitener(_st, mask)
            nst_note = f"{os.path.basename(_nstp)} ({_nk:,} windows)"
        elif a.whiten_source == "nst":
            raise SystemExit(f"--whiten-source nst: no {_nstp or '.nst for ' + os.path.basename(spkpath)} "
                             "(ndm_extractspikes writes it beside the .spk; noiseWindows > 0)")
    use_fil = (not a.no_whiten) and nst_W is None and a.whiten_source in ("fil", "auto")
    if use_fil:                                # fail here, not inside a chunk worker
        try:
            fst.space_transform(space, sdiff)
        except ValueError as e:
            raise SystemExit(f"fiber-session: no .fil noise model for {os.path.basename(spkpath)}: {e} -- give "
                             f"the group's sdiffPairs in the session YAML or --sdiff-pairs, or use "
                             f"--whiten-source spk (or nst), or run --no-whiten") from None
    if a.deconv and not use_fil:
        det("note", "--deconv reads the .fil: it is opened for deconvolution only")
    filmm = nio.open_signal(f"{a.base}.fil", a.ntotal) if (use_fil or a.deconv) else None
    log(f"group {a.elec} \u00b7 {a.method} \u00b7 {nspk:,} spikes")
    if a.exclude_clu:
        det("exclude", f"{int((~keep).sum()):,} spikes ({100 * (~keep).mean():.4f}%) from "
                       f"{os.path.basename(a.exclude_clu)} -- emitted as cluster 0")
    if in_clu is not None:
        det("input", f"{os.path.basename(in_path)}  ({len(np.unique(in_clu)):,} clusters)")
        if a.exclude_units is not None:
            n_ex = int(np.isin(in_clu, excl_units).sum()) if excl_units else 0
            det("exclude", f"{len(excl_units)} unit(s), {n_ex:,} spikes ({100 * n_ex / max(nspk, 1):.2f}%) -- "
                           + ("each carried back as its own cluster" if a.exclude_as == "carry"
                              else "emitted as cluster 0"))
            if missing:
                det("", f"not in the input clu (ignored): {','.join(map(str, missing))}")
        else:
            det("exclude", "no --exclude-units: nothing excluded")
    det("source", spkpath)
    det("noise model", "off (--no-whiten)" if a.no_whiten else
        (f"noise statistics {nst_note}" if nst_W is not None else
         (f"snippet pre-peak, block-Toeplitz ({space or 'standard'})" if a.whiten_source == "spk"
          else fst.space_label(space, sdiff))))
    if filmm is not None:
        det("recording", f"{filmm.shape[0]:,} samples \u00d7 {a.ntotal} ch")
    det("feature-align", fl.get_feature_align())
    det("realign", "sub-sample" if fl.realign_subsample() else "whole-sample")
    if _gpu_line: det("GPU", _gpu_line)
    if a.dipsplit and not _HAVE_DIP:
        det("note", "diptest not installed \u2014 --dipsplit skipped")

    chunk_s = a.chunk_min * 60.0 * a.sr; ov_s = a.overlap_min * 60.0 * a.sr
    t_min, t_max = int(res.min()), int(res.max())
    nchunks = max(1, int(np.ceil((t_max - t_min) / chunk_s)))    # >=1 even for a degenerate single-timestamp session
    ext_idx = [np.array([], int) for _ in range(nchunks)]; ext_lab = [np.array([], int) for _ in range(nchunks)]
    chunk_geoms = [[] for _ in range(nchunks)]; chunk_tmin = [0.0] * nchunks
    chunk_candidates = [[] for _ in range(nchunks)]
    chunk_deconv = {}

    # ── build per-chunk tasks (small chunks skipped here, serially & cheaply) ──
    meth = "none" if a.no_fine else a.fine_method
    # SHAPE features for the fine GMM split: the GLOBAL ndm_pca basis (shared across chunks),
    # so a basis change (nFeatures, varimax) propagates; None -> per-call local SVD fallback.
    cluster_basis = None if a.no_cluster_basis else _fpca.read_cluster_basis(a.base, a.elec, a.method)
    if cluster_basis is not None and a.feat_lag > 0:
        # Carried ON THE BASIS rather than threaded through cluster_chunk_fine: the lag is a property
        # of the feature space this basis defines, and every downstream builder (_rkk_realign,
        # _dipsplit_realign, gmm_split) already receives the basis, so no signature changes and no
        # call site can be missed.
        cluster_basis["_lag"] = int(a.feat_lag)
        cluster_basis["_lag_pc2"] = bool(a.feat_lag_pc2)
    if cluster_basis is not None:
        if a.feat_lag > 0:
            per = 3 + (1 if a.feat_lag_pc2 else 0)
            log(f"fine-split shape features: global basis '{cluster_basis.get('_variant') or a.method}' "
                f"({cluster_basis['evec'].shape[0]}ch x {per}) "
                f"= PC1 at -{a.feat_lag}/0/+{a.feat_lag} samples"
                + (" + PC2" if a.feat_lag_pc2 else ""))
        else:
            log(f"fine-split shape features: global basis '{cluster_basis.get('_variant') or a.method}' "
                f"({cluster_basis['evec'].shape[0]}ch x {cluster_basis['evec'].shape[1]}comp)")
    cf = build_cf(a, meth, cluster_basis)
    if float(cf.get("off_thr", 0.0)) > 0:                   # the timing veto reads the RAW .spk.standard
        try:
            _sstd, _sr_ = nio.open_spk(a.base, a.elec, a.nsamp, a.nchan, prefer=nio.prefer_standard())
        except FileNotFoundError:
            det("timing veto", f"OFF: --off-thr {cf['off_thr']:g} is set but there is no .spk.standard to time "
                               "the templates on")
        else:
            if _sstd.shape[0] != nspk:
                raise SystemExit(f"{_sr_.path}: {_sstd.shape[0]:,} spikes, the .res has {nspk:,} -- the timing "
                                 "veto would compare the wrong spikes' templates")
            del _sstd
    cfg = dict(base=a.base, elec=a.elec, fil=f"{a.base}.fil", ntotal=a.ntotal,
               nsamp=a.nsamp, nchan=a.nchan, sr=a.sr, min_group=a.min_group,
               gch=gch, mask=mask, cf=cf, gpu=a.gpu, no_whiten=getattr(a, "no_whiten", False),
               whiten_source=getattr(a, "whiten_source", "fil"), nst_W=nst_W, nst_mean=nst_mean,
               method=a.method, peak=peak, sdiff_pairs=sdiff,
               deconv=(dict(phases=a.deconv_phases, order=a.deconv_order, refrac_ms=a.deconv_refrac_ms,
                            floor=a.deconv_floor, null_q=a.deconv_null_q, agree=a.deconv_agree,
                            margin_min=a.deconv_margin_min, min_n=a.deconv_min_n,
                            equiv_cos=a.deconv_equiv_cos, fiber_agree=a.deconv_fiber_agree)
                       if a.deconv else None))

    chunks_sel = _fch.parse(a.chunks)
    if chunks_sel is not None:
        beyond = [k for k in chunks_sel if k >= nchunks]
        det("chunks", _fch.describe(chunks_sel, a.chunk_min) + (f"; {beyond} beyond the last chunk ({nchunks - 1})"
                                                               if beyond else ""))
        if nchunks > _fch.RESERVED_N:
            raise SystemExit(f"--chunks: the session has {nchunks} chunks but only {_fch.RESERVED_N} reserved "
                             "clusters exist; use a longer --chunk-min")
    tasks = []; ncore_of = {}
    for c in range(nchunks):
        lo_s = t_min + c * chunk_s; hi_s = _chunk_hi(c, t_min, chunk_s, nchunks)
        chunk_tmin[c] = (lo_s - t_min) / a.sr / 60.0
        if chunks_sel is not None and c not in chunks_sel:   # not computed: its spikes go to a reserved cluster
            ncore_of[c] = int(((res >= lo_s) & (res < hi_s)).sum())
            continue
        ext = np.flatnonzero((res >= lo_s - ov_s) & (res < hi_s + ov_s) & keep)   # excluded spikes never enter a chunk
        ncore = int(((res[ext] >= lo_s) & (res[ext] < hi_s)).sum()); ncore_of[c] = ncore
        if len(ext) < 2 * a.min_group:
            print(f"{IND}chunk {c+1:>3}/{nchunks}   {ncore:>7,} core   \u2192  skipped (too few)"); continue
        tasks.append((c, ext, res[ext]))

    shed_total = {}

    def _store(result):
        c, ext, lab, geoms, cand, sd, dcv = result
        ext_idx[c] = ext; ext_lab[c] = lab; chunk_geoms[c] = geoms; chunk_candidates[c] = cand
        if dcv is not None:
            chunk_deconv[c] = dcv
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
                                 initializer=_init_pool_worker, initargs=(cfg,)) as ex:
            for result in ex.map(_process_chunk, tasks):
                _store(result)

    # ── measurement: where genuine spikes leave the sort (-1), by cause ──
    if shed_total:
        sg = shed_total.get('small_group', 0); sc = shed_total.get('small_core', 0)
        itk = shed_total.get('incl_tail_kept', 0); itd = shed_total.get('incl_tail_dropped', 0)
        co = shed_total.get('cone', 0); ad = shed_total.get('adapt', 0)
        uns = shed_total.get('_unsorted_ext', 0)
        kept = shed_total.get('small_kept', 0); asg = shed_total.get('assigned', 0)
        coarse = max(0, uns - (sg + sc + itd + co + ad) + asg)
        if kept:
            log(f"small groups kept as their own fibers: {kept:,} spikes")
        if asg:
            log(f"unsorted spikes assigned to their nearest fiber: {asg:,} (counted below by cause; total left: {uns:,})")
        rows = [("small-group skip", sg), ("small-core skip", sc)]
        if itd: rows.append(("inclusion tail dropped", itd))
        if co:  rows.append(("cone", co))
        if ad:  rows.append(("adapt", ad))
        rows.append(("coarse-unassigned", coarse))
        lw = max(len(n) for n, _ in rows + [("left unsorted", 0)])
        nw = max(len(f"{v:,}") for _, v in rows + [("", uns + asg)])
        if itk:
            log(f"inclusion tail kept in sort: {itk:,}")
        log("unsorted spikes by cause" + (" (before --assign-unsorted)" if asg else ""))
        for name, val in rows:
            det(name, f"{val:>{nw},}", lw)
        print(f"{IND}{'─' * (lw + 1 + nw)}")
        det("total", f"{uns + asg:>{nw},}", lw)
        if asg:
            det("left unsorted", f"{uns:>{nw},}", lw)

    # anchor_links ALWAYS holds one (possibly empty) list per adjacent chunk pair -- the
    # shape link_chunks returns.  The no-link branch used to leave it [] outright, which
    # the drift-fit block survived (any([]) is False) but the drift_anchor_pairs
    # comprehension below did not: anchor_links[c] raised IndexError AFTER the sort was
    # finished and the .clu/.clc/.clp were already written.
    anchor_links = [[] for _ in range(max(nchunks - 1, 0))]
    if a.no_link:
        gid = {}; n = 0
        for c in range(nchunks):
            for l in sorted({int(x) for x in ext_lab[c] if x >= 0}): gid[(c, l)] = n; n += 1
        nglob = n; mode = "chunk-disjoint"
    else:
        gid, nglob, anchor_links = link_chunks(ext_idx, ext_lab, min_anchor=a.min_anchor)
        mode = f"overlap-anchor linked (min_anchor={a.min_anchor})"
    log(f"{nglob:,} global fibers across {nchunks} chunks")
    det("linking", mode)

    # ── .clu/.clc/.clp : the microfiber hierarchy triple ──
    # ATOM (.clc) = the pre-link fine fragment (chunk, local label); FIBER (.clu) = the
    # linked global id; child->parent map (.clp) = atom -> fiber.  The flat .clu is the
    # derived per-spike fiber layer; .clc/.clp preserve the over-split atoms for curation.
    labels = np.full(nspk, -1, int)
    child = np.zeros(nspk, np.int64)                      # per-spike atom id (.clc); 0 = noise
    parent = {}; atom_of = {}; next_atom = FIBER0         # (chunk,local)->atom ; atom->fiber
    for c in range(nchunks):
        if len(ext_idx[c]) == 0: continue
        lo_s = t_min + c * chunk_s; hi_s = _chunk_hi(c, t_min, chunk_s, nchunks)
        emap = {int(g): int(l) for g, l in zip(ext_idx[c], ext_lab[c])}
        for g in np.flatnonzero((res >= lo_s) & (res < hi_s)):
            l = emap.get(int(g), -1)
            if l >= 0:
                labels[g] = gid[(c, l)]
                aid = atom_of.get((c, l))
                if aid is None:
                    aid = next_atom; next_atom += 1; atom_of[(c, l)] = aid
                    parent[aid] = gid[(c, l)] + FIBER0     # fiber id: 0/1 are the reserve bins
                child[g] = aid
    clu = np.where(labels >= 0, labels + FIBER0, 0).astype(np.int32)

    # ── deconvolution: park confidently attributed noise / collision spikes, write the manifest ──
    deconv_rows = None
    if a.deconv:
        clu, next_atom, deconv_rows = _apply_deconv(a, res, clu, child, parent, next_atom, chunk_deconv,
                                                    ext_idx, gid, t_min, chunk_s, nchunks, log, det)

    # ── optional per-spike consolidation (strip / knn) of the finished labels ──
    #   Session fibers are chunk-local by construction (linking only NAMES them across
    #   chunks), so the pass runs per chunk on core spikes, exactly as in the other host
    #   stages.  Per-spike moves cannot be expressed in the atom->fiber map, so each
    #   (source atom -> destination fiber) bucket becomes its own NEW child atom — the
    #   same parking fiber-anchor-link uses — keeping every move Klusters-inspectable
    #   and revertible.  The .fibers geometry stays pre-consolidation (diagnostic).
    if a.cons_mode != "off":
        cons_mode, strip_kw, knn_kw = fcons.kwargs_from_args(a)
        chid = np.minimum(((res - t_min) / chunk_s).astype(np.int64), nchunks - 1)
        chid = np.where(clu > 0, chid, -1)             # cluster 0 never joins a chunk
        new_clu, _ctot = fcons.consolidate(
            lambda ix: spk[np.asarray(ix)], clu.astype(np.int64), chid,
            mode=cons_mode, exclude=(0,), strip_kw=strip_kw, knn_kw=knn_kw,
            log=log, tag="consolidate")
        clu, next_atom = _park_consolidated(clu, new_clu, child, parent, next_atom,
                                            park_min=getattr(a, "cons_park_min", 0))

    # ── spikes still in cluster 0 that the artifact mask did not put there -- unplaced by the sort (a chunk
    #    too small to sort, a fiber whose geometry failed, a rejection --assign-unsorted did not cover) and
    #    not rescued by deconvolution -- are NOT noise: only fiber-flag-artifacts' derivative test sends a
    #    spike to cluster 0.  They go to one "unsorted" fiber per chunk -- a real cluster, one atom, visible
    #    and curatable in Klusters.  Excluded spikes and the chunks a chunk run did not compute are not
    #    touched here. ──
    _ci = np.minimum(((res - t_min) // chunk_s).astype(np.int64), nchunks - 1)
    _computed = np.ones(nspk, bool) if chunks_sel is None else np.isin(_ci, np.asarray(chunks_sel, np.int64))
    _left = (np.asarray(clu) == 0) & keep & _computed
    if _left.any():
        clu = np.asarray(clu, np.int64).copy()
        next_fiber = max(_fch.next_free_id(clu), FIBER0 + nglob)
        _ch = np.unique(_ci[_left])
        for k in _ch:
            rows = _left & (_ci == k)
            aid = next_atom; next_atom += 1
            parent[aid] = next_fiber
            child[rows] = aid; clu[rows] = next_fiber; next_fiber += 1
        det("unsorted", f"{int(_left.sum()):,} spikes the sort could not place -> {len(_ch)} unsorted fiber(s), "
                        "one per chunk (not cluster 0)")

    carried = {}
    if in_clu is not None and excl_units and a.exclude_as == "carry":
        clu, next_atom, carried = _carry_units(clu, child, parent, next_atom, in_clu, excl_units)

    # ── chunks not computed (--chunks / FK_CHUNKS): every core spike of such a chunk goes to that chunk's
    #    reserved cluster (fiber_chunks.RESERVED_LO + chunk), through ONE ordinary atom per chunk whose
    #    parent is the reserved id -- so the .clc/.clp stay small and Klusters shows the left-out chunks ──
    if chunks_sel is not None:
        ci = np.minimum(((res - t_min) // chunk_s).astype(np.int64), nchunks - 1)
        off = ~np.isin(ci, np.asarray(chunks_sel, np.int64))
        clu = np.asarray(clu, np.int64).copy()
        for k in np.unique(ci[off]):
            rows = off & (ci == k)
            aid = next_atom; next_atom += 1
            parent[aid] = int(_fch.reserved_id(int(k)))
            child[rows] = aid; clu[rows] = parent[aid]
        det("reserved", f"{int(off.sum()):,} spikes of {len(np.unique(ci[off]))} chunk(s) not computed -> clusters "
                        f"{_fch.RESERVED_LO}+chunk")

    if a.out:
        clu_out = a.out
        nio.write_clu_file(clu_out, clu)
        own_paths = {"clu": clu_out}
    elif a.emit_hierarchy:                                # <base>.{clu,clc,clp}.<variant>.<elec>[.<stage>]
        from .fiber_refiberize import FiberHierarchy
        paths = FiberHierarchy(child, parent).save(a.base, a.elec, variant=a.method,
                                                   tag=a.clu_stage, backup=False)
        clu_out = paths["clu"]
        own_paths = dict(paths)
        det("hierarchy", f"{len(parent)} atoms -> {len(set(parent.values()))} fibers (.clu/.clc/.clp)")
    else:                                                 # <base>.clu.<variant>.<elec>[.<stage>] (flat, legacy)
        clu_out = nio.write_clu(a.base, a.elec, clu, variant=a.method, tag=a.clu_stage)
        own_paths = {"clu": clu_out}
    # final ids: the hierarchy save renumbers, so map each pre-save id to what was written
    _, _final = nio.read_clu_file(clu_out)
    _final = np.asarray(_final)
    final_of = _final_ids(clu, _final)
    deconv_man = None
    if deconv_rows is not None:                           # in the written ids; times rewritten after a commit
        deconv_man = nio.session_path(a.base, "deconv", a.elec, variant=a.method, tag=a.clu_stage) + ".tsv"
        _write_deconv_manifest(deconv_man, deconv_rows, final_of, res)
        det("deconv", deconv_man)
    if carried:
        det("carried", ", ".join(f"{u}->{int(_final[np.flatnonzero(in_clu == u)[0]])}" for u in sorted(carried))
                       + "  (input unit -> output cluster)")

    # ── optional ALIAS copies under a second variant token ──
    #   Klusters resolves .fet/.clu/.spk by the SAME token, so viewing a clustering in a differently
    #   named feature space needs the clu under that name too.  Written as a copy rather than a move:
    #   the --method-named artifacts stay canonical and keep their real chain of custody.
    alias_paths = {}
    if a.out_variant and not a.out:
        if a.emit_hierarchy:
            from .fiber_refiberize import FiberHierarchy
            alias_paths = dict(FiberHierarchy(child, parent).save(a.base, a.elec, variant=a.out_variant,
                                                                  tag=a.clu_stage, backup=False))
            det("alias", f"{os.path.basename(alias_paths['clu'])} (+ .clc/.clp)")
        else:
            alias_paths = {"clu": nio.write_clu(a.base, a.elec, clu, variant=a.out_variant, tag=a.clu_stage)}
            det("alias", os.path.basename(alias_paths["clu"]))
    # ── optional .fet for Klusters ──
    #   A closure so the end-of-run realign can call it on the COMMITTED waveforms and spike times
    #   (the .fet's last column is the .res time, and its features project the committed windows).
    def _emit_fet(spk_src, res_src):
        fvar = a.out_variant or a.method
        if cluster_basis is None:
            log("--emit-fet: no global .pca basis resolved; skipped")
        else:
            rs, d2u = int(cluster_basis["recShift"]), int(cluster_basis["data2use"])
            lg = int(cluster_basis.get("_lag", 0) or 0)
            cols = []
            for s0 in range(0, len(res_src), 20000):
                e0 = min(len(res_src), s0 + 20000)
                w = fl.realign(np.asarray(spk_src[s0:e0], np.float64))
                nb = cluster_basis["evec"].shape[0]
                if w.shape[2] == nb + 1:
                    w = w[:, :, :nb]
                if lg > 0 and rs - lg >= 0 and rs + d2u + lg <= w.shape[1]:
                    cols.append(_fpca.lag_project(lambda sh: w[:, rs + sh:rs + sh + d2u, :],
                                                  cluster_basis, lg,
                                                  keep_pc2=bool(cluster_basis.get("_lag_pc2", True))))
                else:
                    cols.append(_fpca.project(_fpca.extract_windows(w, rs, d2u), cluster_basis))
            F = np.vstack(cols)
            # half-away-from-zero, matching ns3's llround (see the refeaturize rounding fix)
            Fi = np.sign(F) * np.floor(np.abs(F) + 0.5)
            # Klusters requires the TIMESTAMP as the last column (data.cpp: "timestamp is last
            # column"; nFeat = nbDimensions - 1).  Without it Klusters reads the final FEATURE as a
            # time, and is left one feature short of what nFeatures implies -- which is how a .fet
            # of the wrong width corrupts its heap instead of being rejected.
            Fi = np.column_stack([Fi, np.asarray(res_src, np.int64)])
            fp_out = nio.write_fet(a.base, a.elec, Fi, variant=fvar)
            nfeat_ch = F.shape[1] // a.nchan
            det("fet", f"{os.path.basename(fp_out)}  ({F.shape[1]} features + 1 timestamp "
                       f"= {Fi.shape[1]} columns x {Fi.shape[0]} spikes)")
            det("", f"set nFeatures={nfeat_ch} for group {a.elec} in the session yaml, else Klusters "
                    f"sizes on nChannels x nFeatures and disagrees with this file")
            if a.emit_pca:
                wb = _fpca.lag_basis(cluster_basis, lg,
                                     keep_pc2=bool(cluster_basis.get("_lag_pc2", True))) if lg > 0 \
                     else cluster_basis
                pca_out = nio.session_path(a.base, "pca", a.elec, variant=fvar)
                _fpca.write_pcad(pca_out, wb["means"], wb["evec"], int(wb["recShift"]),
                                 centered=int(wb["centered"]),
                                 method=int(cluster_basis.get("method", 0) or 0))
                det("pca", f"{os.path.basename(pca_out)}  ({wb['evec'].shape[1]} comp x "
                           f"{wb['data2use']} samples, recShift {wb['recShift']})")

    if a.emit_fet and a.realign == "off":
        _emit_fet(spk, res)

    # ── drift-transform fit from the overlap-anchors (DIAGNOSTIC; NOT used for linking) ──
    #   The anchors are exact shared-spike correspondences, so a per-adjacent-chunk-pair rigid
    #   Procrustes fit of the fiber-template constellation recovers the collective drift
    #   (rotation+translation) a consumer (e.g. fiber-backbone-link) can apply to drift-correct.
    #   RAW/standard templates ONLY -- stderiv breaks the amplitude-distance law and mutes drift.
    KDRIFT = 6; nP = max(nchunks - 1, 0)
    drift_basis = np.zeros((0, a.nsamp * a.nchan), np.float32); drift_mean = np.zeros(a.nsamp * a.nchan, np.float32)
    drift_R = (np.stack([np.eye(KDRIFT, dtype=np.float32) for _ in range(nP)]) if nP else np.zeros((0, KDRIFT, KDRIFT), np.float32))
    drift_t = np.zeros((nP, KDRIFT), np.float32); drift_resid = np.full(nP, np.nan, np.float32); drift_nanchor = np.zeros(nP, int)
    try:
        _spk_std, _ = nio.open_spk(a.base, a.elec, a.nsamp, a.nchan, prefer=nio.prefer_standard())
    except FileNotFoundError:
        _spk_std = None                                          # no .spk.standard -> emit identity transforms
    if _spk_std is not None and nP and any(anchor_links):
        _rng = np.random.default_rng(0)
        keys = sorted({(c, f) for c in range(nP) for f, g in anchor_links[c]}
                      | {(c + 1, g) for c in range(nP) for f, g in anchor_links[c]})
        rt = {}
        for (c, f) in keys:
            lo_c = t_min + c * chunk_s; hi_c = _chunk_hi(c, t_min, chunk_s, nchunks)
            gi = ext_idx[c][ext_lab[c] == f]
            gi = gi[(res[gi] >= lo_c) & (res[gi] < hi_c)]        # CORE spikes only: adjacent-chunk templates then
            if gi.size == 0: continue                            #   share NO spikes, so the fit sees the true drift
            gi = np.sort(_rng.choice(gi, 400, replace=False) if gi.size > 400 else gi)
            rt[(c, f)] = fl.realign(np.asarray(_spk_std[gi], float)).mean(0).ravel()   # RAW mean template
        if len(rt) >= KDRIFT + 2:
            X = np.array(list(rt.values())); mu = X.mean(0); drift_mean = mu.astype(np.float32)
            basis = np.linalg.svd(X - mu, full_matrices=False)[2][:KDRIFT]
            drift_basis = basis.astype(np.float32)
            posK = {k: (v - mu) @ basis.T for k, v in rt.items()}
            R, t, resid, na = flc.fit_drift_transforms(posK, anchor_links, KDRIFT)
            drift_R = R.astype(np.float32); drift_t = t.astype(np.float32)
            drift_resid = resid.astype(np.float32); drift_nanchor = na
        _med = float(np.nanmedian(drift_resid)) if np.isfinite(drift_resid).any() else float('nan')
        det("drift-fit", f"{int(np.sum(drift_nanchor > 0))}/{nP} pairs, median resid {_med:.2f}")

    # Exact overlap-anchor correspondences as ATOM pairs (c, atom_c, atom_c+1). These are res-based
    # (same physical spikes), so they SURVIVE fiber-realign's reextraction -- letting a consumer
    # re-fit the drift transform in ITS OWN post-realign feature space. The basis/R/t emitted above
    # are fit on the PRE-realign templates and are kept only as a fiber-session-space diagnostic.
    drift_anchor_pairs = np.array(
        [(c, atom_of[(c, f)], atom_of[(c + 1, g)]) for c in range(nP) for f, g in anchor_links[c]
         if (c, f) in atom_of and (c + 1, g) in atom_of], int).reshape(-1, 3)

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
        # the row's cluster in the written .clu (0 = none): NOT gid + 1 -- fibers start at FIBER0 and the
        # hierarchy save compacts ids
        clu_id=np.array([final_of.get(int(r['gid']) + FIBER0, 0) if int(r['gid']) >= 0 else 0 for r in rows],
                        int),
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
        # deconvolution QC (NaN when --deconv was off or the chunk's pursuit failed)
        deconv_agree=np.array([r.get('deconv_agree', np.nan) for r in rows], np.float32),
        deconv_partner_frac=np.array([r.get('deconv_partner_frac', np.nan) for r in rows], np.float32),
        deconv_n_equiv=np.array([r.get('deconv_n_equiv', 0) for r in rows], int),
        template=np.stack([r['template'] for r in rows]) if M else np.zeros((0, a.nsamp, a.nchan), np.float32),
        grid=np.stack([r['grid'] for r in rows]) if M else np.zeros((0, a.n_grid), np.float32),
        dir=np.stack([r['dir'] for r in rows]) if M else np.zeros((0, a.n_grid, p), np.float32),
        meta_elec=a.elec, meta_channels=gch, meta_sr=a.sr, meta_mask=np.asarray(mask),
        meta_n_grid=a.n_grid, meta_p=p, meta_nsamp=a.nsamp, meta_nchan=a.nchan,
        meta_method=a.method, meta_chunk_min=a.chunk_min, meta_overlap_min=a.overlap_min,
        meta_n_spikes=nspk,                               # the .res this sort indexes: a changed spike set shows
        drift_pca_basis=drift_basis, drift_pca_mean=drift_mean, drift_R=drift_R, drift_t=drift_t,
        drift_resid=drift_resid, drift_nanchor=drift_nanchor, meta_drift_k=KDRIFT,
        drift_anchor_pairs=drift_anchor_pairs)
    with nio.atomic_write(fib_out, "wb") as f:
        np.savez_compressed(f, **arrs)
    log("wrote")
    det("clu", clu_out)
    det("fibers", f"{fib_out}   ({M:,} instances \u00b7 {nglob:,} global)")

    # ── .merge_candidates.<elec>.tsv : proposed same-neuron merges (curation) ──
    if a.emit_merge_candidates or a.merge_method == "profile":
        cand_out = f"{a.base}.merge_candidates.{a.elec}.tsv"
        ncand = 0
        with open(cand_out, "w") as f:
            f.write("chunk\tgid_a\tgid_b\tlocal_a\tlocal_b\tprofile_dist\tthreshold\tclu_a\tclu_b\n")
            for c in range(nchunks):
                for (ai, bi, d, thr) in chunk_candidates[c]:
                    ga = gid.get((c, ai), -1); gb = gid.get((c, bi), -1)
                    ca = final_of.get(_fiber_id(gid, (c, ai)), 0); cb = final_of.get(_fiber_id(gid, (c, bi)), 0)
                    f.write(f"{c}\t{ga}\t{gb}\t{ai}\t{bi}\t{d:.4f}\t{thr:.4f}\t{ca}\t{cb}\n"); ncand += 1
        note = "review-only (fibers NOT merged)" if a.emit_merge_candidates else "already applied"
        det("candidates", f"{cand_out}   ({ncand:,} pairs \u00b7 {note})")

    # ── end-of-run realign: shifted .res, every variant's .spk and .fet ──
    #   Last, because everything above (consolidation, .fet, the drift fit's raw templates, the
    #   .fibers geometry) is defined on the windows as extracted.  The commit is fiber-realign's own
    #   main() on the .clu just written -- the same code, flags and output as the plan step
    #   `fiber-realign --reextract --refeaturize --variants all` -- so the two cannot drift apart.
    if a.realign != "off":
        commit_peak, how = _commit_peak(yaml_peak, spk, clu,
                                        locate=lambda: _locate_windows(a, gch, sdiff, res, space))
        if how != "yaml":
            det("peak", f"{commit_peak} (1-based) {how} -- the YAML has no peakSampleIndex for this group")
        _commit_realign(a, clu_out, space, sdiff, commit_peak, log, det)
        for kind, dst in alias_paths.items():             # the alias is a copy: it carries the parking too
            src = own_paths.get(kind)
            if src and os.path.exists(src) and os.path.abspath(src) != os.path.abspath(dst):
                shutil.copyfile(src, dst)
        if deconv_man is not None:                        # the manifest's res follows the committed .res
            _write_deconv_manifest(deconv_man, deconv_rows, final_of, read_res(a.base, a.elec))
        if a.emit_fet:                                    # the requested .fet, from the committed set
            _spk_c, _ = open_spkD(a.base, a.elec, a.nsamp, a.nchan, prefer=space or a.method)
            _emit_fet(_spk_c, read_res(a.base, a.elec))
    log(f"done in {time.time()-t0:.0f}s")


def _commit_peak(yaml_peak, spk, labels, locate=None, n_units=30, per_unit=200, seed=0):
    """The 1-based peakSampleIndex the end-of-run commit cuts its windows at (fiber-realign: the spike
    time at peak - 1, also its klusters target).  Returns (peak, how).

    The YAML's when the group has one: that is the window neurosuite-3 cut and Klusters re-cuts.
    Without one there is no declared geometry, and where a spike peaks in its window differs between
    data sets, so it is MEASURED.  First, where the windows on disk hold the spike time (`locate`:
    fiber_realign.window_index against the .fil) -- the geometry they were cut with, which the commit
    then keeps whatever the waveforms look like.  Without a .fil that reproduces them, the median over
    the largest units of each unit's template peak (argmax of the channel-summed |mean|, the klusters
    template peak): the extraction's index wherever the detector timestamps the peak, and the sample
    a --shift-spk commit (which keeps the windows' geometry) holds every template on, so it re-centres
    nothing systematically.  Before 0.51 this passed build_masks' 15/32 guess, read as 1-based: every
    spike of a group whose spikes peak elsewhere moved by the difference."""
    if yaml_peak is not None:
        return int(yaml_peak), "yaml"
    wi = locate() if locate is not None else None
    if wi is not None:
        return wi.index + 1, (f"measured: the .spk.{wi.variant} windows hold the spike time at index "
                              f"{wi.index} ({wi.frac:.0%} of {wi.n} re-cut identically from the .fil)")
    lab = np.asarray(labels)
    ids, cnt = np.unique(lab[lab > 1], return_counts=True)
    rng = np.random.default_rng(seed)
    peaks = []
    for k in np.argsort(-cnt, kind="stable")[:n_units]:
        idx = np.flatnonzero(lab == ids[k])
        if idx.size < 20:
            continue
        if idx.size > per_unit:
            idx = np.sort(rng.choice(idx, per_unit, replace=False))
        peaks.append(int(np.argmax(np.abs(np.asarray(spk[idx], np.float64).mean(0)).sum(1))))
    if peaks:
        return int(np.median(peaks)) + 1, f"measured on the template peaks of the {len(peaks)} largest units"
    idx = np.sort(rng.choice(len(spk), min(len(spk), 2000), replace=False))
    return (int(np.argmax(np.abs(np.asarray(spk[idx], np.float64)).mean(0).sum(1))) + 1,
            "measured on the whole group's template peak (no unit of 20 spikes)")


def _locate_windows(a, gch, sdiff, res, space):
    """fiber_realign.window_index of the .spk the commit aligns (this run's own variant, and only that:
    another variant may have been cut differently) against the .fil the commit reads, or None."""
    fil = a.realign_fil or f"{a.base}.fil"
    if not a.ntotal or not os.path.exists(fil):
        return None
    try:
        from . import fiber_realign as frl
    except ImportError:
        import fiber_realign as frl
    return frl.window_index(nio.open_signal(fil, a.ntotal),
                            {"channels": [int(c) for c in gch], "sdiff_pairs": sdiff}, a.elec, res,
                            a.nsamp, a.nchan, a.base, tokens=(space or a.method,))


def _commit_realign(a, clu_path, space, sdiff, peak, log, det):
    """Run fiber-realign's in-place commit on `clu_path` (see --realign).  `sdiff` is the group's
    sdiffPairs as this run resolved it; `peak` the group's 1-based peakSampleIndex -- the YAML's, else
    the one _commit_peak measured (a session sorted without a YAML would otherwise stop fiber-realign
    cold).  Returns the mode used."""
    try:
        from . import fiber_realign as frl
    except ImportError:
        import fiber_realign as frl
    fil = a.realign_fil or f"{a.base}.fil"
    mode = a.realign
    if mode == "auto":
        mode = "reextract" if os.path.exists(fil) else "shift-spk"
    if mode == "reextract" and not os.path.exists(fil):
        raise SystemExit(f"--realign reextract: no filtered signal {fil} (use --realign shift-spk, or "
                         f"--realign-fil <path>)")
    # --park-duplicates: the sort is this run's own, so two detections the alignment stacks on one sample
    # go straight to cluster 0 (one child atom under 0) instead of being counted twice in their unit.
    argv = [str(a.session), str(a.group), "--clu", clu_path, "--align-algo", a.realign_algo,
            "--variants", "all", "--variant", space or a.method, "--no-emit-clu", "--refeaturize",
            "--park-duplicates"]
    argv += ["--reextract", "--fil", fil] if mode == "reextract" else ["--shift-spk"]
    for flag, val in (("--channels", a.channels), ("--ntotal", a.ntotal), ("--nchan", a.nchan),
                      ("--nsamp", a.nsamp), ("--sr", a.sr), ("--sdiff-pairs", sdiff),
                      ("--peak", peak)):
        if val not in (None, ""):
            argv += [flag, ",".join(map(str, val)) if isinstance(val, (list, tuple)) else str(val)]
    log(f"realign: {mode} on {os.path.basename(clu_path)}")
    det("commit", "shifted .res + every variant's .spk/.fet  (fiber-realign " + " ".join(argv[2:]) + ")")
    try:
        frl.main(argv)
    except (Exception, SystemExit) as e:                 # the sort is on disk: say how to finish it
        if isinstance(e, SystemExit) and e.code in (None, 0):
            raise
        raise SystemExit(f"fiber-session: the sort is written ({os.path.basename(clu_path)}) but the "
                         f"end-of-run realign failed: {e}\n  finish it with:  fiber-realign "
                         + " ".join(argv) + "\n  (or rerun with --realign off)") from None
    return mode


if __name__ == "__main__":
    main()
