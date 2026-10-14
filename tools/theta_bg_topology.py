#!/usr/bin/env python3
"""Theta-cycle background structure and place-like topology over a FULL session.

Two steps (run `extract` once, iterate on `analyze`):

  extract   streams the session once and writes <out>.npz:
              * theta cycles from the LFP (5-11 Hz, trough to trough) with theta/delta ratio,
                duration and amplitude per cycle (all cycles; the theta selection is made in analyze)
              * BACKGROUND per cycle x theta quarter x channel: mean-square of the .fil over samples at
                least --guard-ms from any detected spike of that channel's group (spike-free activity)
              * optional UNIT counts per cycle x quarter (from a clu stage of every group)

  analyze   per theta cycle: the PHASE x CHANNEL pattern of the high-frequency background (each cycle's
            overall level removed), slow drift removed (--detrend-s running mean), smoothed over a few
            consecutive cycles, PCA.  Against a NULL that circularly shifts every feature independently
            (keeps each feature's statistics, destroys co-activation):
              * a 2-D LATENT MAP of the cycles (UMAP, else Isomap) -- no position needed
              * continuity of consecutive cycles on the map (a trajectory moves in small steps)
              * place-like FIELDS of the sorted units on that map: Skaggs information vs a 60-s
                circular-shift null, and odd/even-run stability.  With --features bg the map is built
                from the background alone, so a unit with a stable field is predicted by it
              * intrinsic dimension (TwoNN), revisit structure, persistent homology (gudhi/ripser)
              * optional --pos (.whl) decoding for validation when tracking exists

Example (neurosuite session layout, group channels from <base>.yaml):

  python3 tools/theta_bg_topology.py extract sirotaA-jg-000005-20120316 --groups all \\
      --clu-method stderiv_C5_D34 --clu-stage exp_final --out jg05_theta
  python3 tools/theta_bg_topology.py analyze jg05_theta.npz --features bg --pos sirotaA-jg-000005-20120316.whl

The background deliberately excludes spike times, so it measures sub-threshold / unsorted population
activity; `--features units` gives the classical population-vector control on the same cycles.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))


def _log(*a):
    print(*a, file=sys.stderr, flush=True)


# ════════════════════════════════════════════════════════════════════════════
#  extract
# ════════════════════════════════════════════════════════════════════════════
def _parse_list(s, n_max=None):
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def lfp_trace(path, nchan, chans, sr, target_sr=1250.0, start_s=0.0, chunk_s=60.0):
    """Median over `chans` of an int16 interleaved file, returned at ~target_sr.  A wide-band source
    (sr > 2.5 kHz) is low-passed and decimated chunk by chunk."""
    from scipy.signal import butter, sosfiltfilt
    mm = np.memmap(path, np.int16, "r").reshape(-1, nchan)
    ds = max(1, int(round(sr / target_sr)))
    fs = sr / ds
    sos = butter(4, 0.4 * fs, "low", fs=sr, output="sos") if ds > 1 else None
    n = mm.shape[0]; step = int(chunk_s * sr); pad = int(sr)
    out = []
    for a in range(0, n, step):
        lo, hi = max(0, a - pad), min(n, a + step + pad)
        x = np.median(np.asarray(mm[lo:hi][:, chans], np.float32), 1)
        if sos is not None:
            x = sosfiltfilt(sos, x.astype(np.float64))
        x = x[a - lo:a - lo + min(step, n - a)]
        out.append(x[::ds] if ds > 1 else x)
    return np.concatenate(out).astype(np.float32), fs, ds


def theta_cycles(x, fs, band=(5.0, 11.0), delta=(2.0, 4.0)):
    """Troughs of the theta phase -> cycles; per cycle: start/end (lfp samples), theta/delta ratio
    (1-s smoothed amplitudes), duration, theta amplitude.  Also returns the phase."""
    from scipy.signal import butter, sosfiltfilt, hilbert
    th = sosfiltfilt(butter(3, band, "bp", fs=fs, output="sos"), x.astype(np.float64))
    de = sosfiltfilt(butter(3, delta, "bp", fs=fs, output="sos"), x.astype(np.float64))
    zt = hilbert(th); ph = np.angle(zt)
    k = np.ones(int(fs)) / fs
    at = np.convolve(np.abs(zt), k, "same"); ad = np.convolve(np.abs(hilbert(de)), k, "same")
    tr = np.flatnonzero((ph[:-1] > 0) & (ph[1:] <= 0) & (np.abs(ph[:-1] - ph[1:]) > np.pi)) + 1
    a, b = tr[:-1], tr[1:]
    ratio = np.array([at[i:j].mean() / max(ad[i:j].mean(), 1e-9) for i, j in zip(a, b)])
    amp = np.array([np.abs(zt[i:j]).mean() for i, j in zip(a, b)])
    return dict(start=a, end=b, ratio=ratio, dur=(b - a) / fs, amp=amp), ph


def _res_for_group(nio, base, g, variant):
    if variant:
        return np.asarray(nio.read_res_at(base, g, variant=variant), np.int64)
    return np.asarray(nio.read_res(base, g), np.int64)


def cmd_extract(a):
    from fiber_kit import neuro_io as nio, session_yaml as sy
    yp = sy.find_session_yaml(a.base)
    if yp is None:
        raise SystemExit(f"no session yaml for {a.base}")
    info0 = sy.load_session(a.base, 1)
    base = info0["base"]; sr = float(info0["sr"]); ntot = int(info0["ntotal"])
    import yaml
    with open(yp) as fh:
        doc = yaml.safe_load(fh)
    ngroups = len(((doc.get("spikeDetection") or {}).get("channelGroups")) or [])
    groups = list(range(1, ngroups + 1)) if a.groups == "all" else _parse_list(a.groups)
    ginfo = {g: sy.load_session(a.base, g) for g in groups}
    dead = set(_parse_list(a.dead_channels)) if a.dead_channels else set()

    # ── LFP / theta ──
    lfp_path = a.lfp or f"{base}.lfp"
    lfp_sr = a.lfp_sr or float((doc.get("fieldPotentials") or {}).get("lfpSamplingRate") or 1250.0)
    lfp_n = a.lfp_nchan or ntot
    tch = (_parse_list(a.theta_channels) if a.theta_channels
           else [c for g in groups for c in ginfo[g]["channels"] if c not in dead])
    _log(f"[extract] theta from {os.path.basename(lfp_path)} ({lfp_n} ch @ {lfp_sr:g} Hz), median of {len(tch)} ch")
    x, fs, _ = lfp_trace(lfp_path, lfp_n, tch, lfp_sr)
    cyc, ph = theta_cycles(x, fs, band=(a.theta_lo, a.theta_hi))
    ncyc = len(cyc["start"])
    _log(f"[extract] {ncyc} cycles over {x.size / fs / 60:.1f} min (median {1e3 * np.median(cyc['dur']):.0f} ms)")
    # phase-bin per LFP sample (-1 outside cycles).  --phase-mode phase: NB equal bins of the Hilbert
    # theta phase, starting at the trough (bin k = [-pi + k*2pi/NB, -pi + (k+1)*2pi/NB)) -- theta is
    # asymmetric, so these are NOT equal in duration; time: NB equal-duration bins trough to trough.
    NB = int(a.phase_bins)
    qbin = np.full(x.size, -1, np.int64)
    for c in range(ncyc):
        s, e = cyc["start"][c], cyc["end"][c]
        if a.phase_mode == "phase":
            q = np.floor(((ph[s:e] + np.pi) % (2 * np.pi)) / (2 * np.pi / NB)).astype(np.int64)
        else:
            q = ((np.arange(s, e) - s) * NB) // max(e - s, 1)
        qbin[s:e] = c * NB + np.clip(q, 0, NB - 1)
    lfp_t0 = a.lfp_start_s

    # ── background from the .fil ──
    fil_path = a.fil or f"{base}.fil"
    F = np.memmap(fil_path, np.int16, "r").reshape(-1, ntot)
    sig0 = int(a.signal_start)                         # absolute sample of the .fil's first row
    guard = int(round(a.guard_ms * 1e-3 * sr))
    allch = sorted({c for g in groups for c in ginfo[g]["channels"] if c not in dead})
    col = {c: i for i, c in enumerate(allch)}
    E = np.zeros((ncyc * NB, len(allch)), np.float64); Ncnt = np.zeros((ncyc * NB, len(allch)), np.float64)
    res = {g: np.sort(_res_for_group(nio, base, g, a.res_variant)) for g in groups}
    step = int(a.chunk_s * sr)
    for s0 in range(0, F.shape[0], step):
        s1 = min(F.shape[0], s0 + step)
        abs0 = sig0 + s0
        li = np.floor(((np.arange(s0, s1) + sig0) / sr - lfp_t0) * fs).astype(np.int64)
        okl = (li >= 0) & (li < x.size)
        b = np.full(s1 - s0, -1, np.int64); b[okl] = qbin[li[okl]]
        if (b >= 0).sum() == 0:
            continue
        X = np.asarray(F[s0:s1], np.float32)
        for g in groups:
            r = res[g]; r = r[(r >= abs0 - guard) & (r < abs0 + (s1 - s0) + guard)] - abs0
            m = np.ones(s1 - s0, bool)
            if r.size:
                lo = np.clip(r - guard, 0, s1 - s0); hi = np.clip(r + guard + 1, 0, s1 - s0)
                d = np.zeros(s1 - s0 + 1, np.int64); np.add.at(d, lo, 1); np.add.at(d, hi, -1)
                m = np.cumsum(d)[:-1] == 0
            sel = m & (b >= 0); bb = b[sel]
            if bb.size == 0:
                continue
            for c in ginfo[g]["channels"]:
                if c in dead:
                    continue
                v = X[sel, c]
                E[:, col[c]] += np.bincount(bb, weights=(v.astype(np.float64)) ** 2, minlength=ncyc * NB)
                Ncnt[:, col[c]] += np.bincount(bb, minlength=ncyc * NB)
        _log(f"[extract] .fil {s1 / sr / 60:6.1f} / {F.shape[0] / sr / 60:.1f} min")
    with np.errstate(invalid="ignore", divide="ignore"):
        bg = (E / Ncnt).reshape(ncyc, NB, len(allch)).astype(np.float32)
    nsamp = Ncnt.reshape(ncyc, NB, len(allch)).min(2).astype(np.int32)

    # ── optional unit counts ──
    U = None; unames = []
    if a.clu_stage:
        cols = []; tried = []
        for g in groups:
            cp = nio.session_path(base, "clu", g, variant=a.clu_method, tag=a.clu_stage)
            if not os.path.exists(cp):
                # the method token is easy to mistype (stderiv_C4_D34 vs _C5_): take the stage's one .clu
                import glob as _g
                alt = sorted(_g.glob(f"{base}.clu.*.{g}.{a.clu_stage}"))
                tried.append(cp)
                if len(alt) == 1:
                    _log(f"[extract] group {g}: {os.path.basename(cp)} not found, using {os.path.basename(alt[0])}")
                    cp = alt[0]
                else:
                    _log(f"[extract] group {g}: no {os.path.basename(cp)}"
                         + (f" (candidates: {', '.join(os.path.basename(x) for x in alt)})" if alt else "") + "; skipped")
                    continue
            try:
                _, clu = nio.read_clu_file(cp, n_spikes=res[g].size)
            except (OSError, ValueError) as e:
                _log(f"[extract] group {g}: {os.path.basename(cp)} unreadable ({e}); skipped"); continue
            rg = _res_for_group(nio, base, g, a.res_variant)       # unsorted order matches the clu
            li = np.floor((rg / sr - lfp_t0) * fs).astype(np.int64)
            ok = (li >= 0) & (li < x.size); bi = np.full(rg.size, -1, np.int64); bi[ok] = qbin[li[ok]]
            clu = np.asarray(clu)
            for u in np.unique(clu[clu > 1]):
                m = (clu == u) & (bi >= 0)
                if (clu == u).sum() < a.min_unit_spikes:
                    continue
                cols.append(np.bincount(bi[m], minlength=ncyc * NB).reshape(ncyc, NB)); unames.append(f"{g}:{u}")
        if cols:
            U = np.stack(cols, 2).astype(np.uint16)
            _log(f"[extract] {len(unames)} units counted per cycle x phase bin")
        else:
            raise SystemExit(f"[extract] --clu-stage {a.clu_stage}: no group had a readable .clu (tried e.g. "
                             f"{', '.join(os.path.basename(x) for x in tried[:3])}); check --clu-method")
    out = a.out if a.out.endswith(".npz") else a.out + ".npz"
    np.savez_compressed(out, t=(cyc["start"] / fs + lfp_t0), dur=cyc["dur"], ratio=cyc["ratio"], amp=cyc["amp"],
                        bg=bg, nsamp=nsamp, channels=np.array(allch), units=(U if U is not None else np.zeros((ncyc, NB, 0))),
                        unit_names=np.array(unames), meta=json.dumps(dict(base=base, groups=groups, sr=sr, lfp_fs=fs,
                                                                          guard_ms=a.guard_ms, theta=[a.theta_lo, a.theta_hi],
                                                                          phase_bins=NB, phase_mode=a.phase_mode)))
    _log(f"[extract] wrote {out}: {ncyc} cycles x {NB} phase bins x {len(allch)} ch" + (f" + {len(unames)} units" if unames else ""))


# ════════════════════════════════════════════════════════════════════════════
#  analyze
# ════════════════════════════════════════════════════════════════════════════
def twonn(X, rng, n=4000):
    """Facco et al. TwoNN intrinsic dimension (MLE on mu = r2/r1)."""
    from sklearn.neighbors import NearestNeighbors
    if len(X) > n:
        X = X[rng.choice(len(X), n, replace=False)]
    d, _ = NearestNeighbors(n_neighbors=3).fit(X).kneighbors(X)
    mu = d[:, 2] / np.maximum(d[:, 1], 1e-12)
    mu = mu[np.isfinite(mu) & (mu > 1)]
    mu = np.sort(mu)[: int(0.9 * mu.size)]               # discard the top 10% (Facco's recommendation)
    return float(mu.size / np.sum(np.log(mu)))


def revisit(X, t, k, excl_s, rng, n=4000):
    """median distance to the k nearest neighbours OUTSIDE +-excl_s of the point's own time, divided
    by the median distance between random pairs (low = distant-in-time cycles revisit the same states),
    and the median time separation of those neighbours."""
    from sklearn.neighbors import NearestNeighbors
    q = rng.choice(len(X), min(n, len(X)), replace=False)
    nn = NearestNeighbors(n_neighbors=min(len(X) - 1, k + 400)).fit(X)
    d, j = nn.kneighbors(X[q])
    dd, dt = [], []
    for row in range(len(q)):
        far = np.abs(t[j[row]] - t[q[row]]) > excl_s
        if far.sum() >= k:
            dd.append(np.median(d[row][far][:k])); dt.extend(np.abs(t[j[row]][far][:k] - t[q[row]]).tolist())
    i1, i2 = rng.integers(0, len(X), 20000), rng.integers(0, len(X), 20000)
    base = np.median(np.linalg.norm(X[i1] - X[i2], axis=1))
    dt = np.array(dt)
    span = float(t.max() - t.min())
    return (float(np.median(dd) / base) if dd else np.nan, float(np.median(dt)) if dt.size else np.nan,
            float((dt > 0.25 * span).mean()) if dt.size else np.nan)


def maxmin(X, m, rng):
    idx = [int(rng.integers(len(X)))]; d = np.linalg.norm(X - X[idx[0]], axis=1)
    for _ in range(min(m, len(X)) - 1):
        i = int(np.argmax(d)); idx.append(i); d = np.minimum(d, np.linalg.norm(X - X[i], axis=1))
    return np.array(idx)


def homology(L, maxdim=2):
    """top persistence bars per dimension (finite ones), with ripser or gudhi; None if neither."""
    try:
        from ripser import ripser
        dg = ripser(L, maxdim=maxdim)["dgms"]
    except ImportError:
        try:
            import gudhi
        except ImportError:
            return None
        from scipy.spatial.distance import pdist
        thr = float(np.quantile(pdist(L), 0.5))
        st = gudhi.RipsComplex(points=L, max_edge_length=thr, sparse=0.5).create_simplex_tree(max_dimension=maxdim + 1)
        st.compute_persistence()
        dg = [np.array(st.persistence_intervals_in_dimension(k)) if len(st.persistence_intervals_in_dimension(k))
              else np.zeros((0, 2)) for k in range(maxdim + 1)]
    out = {}
    for k, D in enumerate(dg):
        D = np.asarray(D)
        if D.size == 0:
            out[f"H{k}"] = []; continue
        life = D[:, 1] - D[:, 0]; life = life[np.isfinite(life)]
        out[f"H{k}"] = np.sort(life)[::-1][:5].round(4).tolist()
    return out


def read_pos(path, sr):
    P = np.loadtxt(path)
    P = P[:, :2] if P.ndim == 2 else P.reshape(-1, 2)
    P = P.astype(float); P[(P < 0).any(1)] = np.nan
    return P, np.arange(len(P)) / sr


def features(Z, kind, remove_level):
    """bg: log mean-square per (phase bin, channel); remove_level drops each cycle's overall level so the
    phase x channel PATTERN (the within-cycle sequence across the shank) is what remains."""
    bg = np.log(Z["bg"].astype(np.float64) + 1e-9)                # cycles x bins x ch
    blocks = []
    if kind in ("bg", "both"):
        B = bg.reshape(len(bg), -1)
        if remove_level:
            B = B - B.mean(1, keepdims=True)
        blocks.append(B)
    if kind in ("units", "both"):
        Uc = Z["units"].astype(np.float64)
        if Uc.shape[2] == 0:
            raise SystemExit("no unit counts in the npz (run extract with --clu-stage)")
        blocks.append(np.log1p(Uc.reshape(len(Uc), -1)))
    return np.hstack(blocks)


def detrend_time(F, t, win_s):
    """subtract each feature's centred running mean over +-win_s/2 (in session time): removes the slow
    drift (electrode, state) that otherwise orders the map by time."""
    if not win_s or win_s <= 0:
        return F
    lo = np.searchsorted(t, t - win_s / 2); hi = np.searchsorted(t, t + win_s / 2, side="right")
    C = np.vstack([np.zeros((1, F.shape[1])), np.cumsum(F, 0)])
    return F - (C[hi] - C[lo]) / (hi - lo)[:, None]


def smooth_runs(F, run, w):
    if w <= 1:
        return F
    out = np.empty_like(F); k = np.ones(w) / w
    for r in np.unique(run):
        m = np.flatnonzero(run == r)
        for j in range(F.shape[1]):
            v = F[m, j]
            out[m, j] = np.convolve(np.pad(v, (w // 2, w - 1 - w // 2), mode="edge"), k, "valid")
    return out


def pipeline(F, a, rng, nk=None):
    """z-score + PCA; nk = dims kept (by --pca-var unless given: the null keeps the REAL data's count,
    so dimension and homology are compared in spaces of the same size)."""
    from sklearn.decomposition import PCA
    Fz = (F - F.mean(0)) / (F.std(0) + 1e-9)
    p = PCA(n_components=min(a.pca_max, Fz.shape[1], len(Fz) - 1), random_state=0).fit(Fz)
    if nk is None:
        nk = int(min(np.searchsorted(np.cumsum(p.explained_variance_ratio_), a.pca_var) + 1, p.n_components_))
    X = p.transform(Fz)[:, :nk]
    return X / np.median(np.linalg.norm(X - np.median(X, 0), axis=1)), nk     # unit median radius


def latent_map(X, a, rng):
    """2-D map of the cycles: UMAP when installed, else Isomap fitted on a subsample and applied to all.
    Returns (raw map, rank-normalised copy in [0,1]^2, method).  The raw map keeps the shape (holes,
    loops, clusters) for viewing and for step lengths; the ranked copy has uniform marginals so a grid
    over it is evenly occupied for the unit fields."""
    try:
        import umap
        Y = umap.UMAP(n_neighbors=a.map_k, min_dist=0.3, n_components=2, random_state=a.seed).fit_transform(X)
        how = "umap"
    except ImportError:
        from sklearn.manifold import Isomap
        sub = rng.choice(len(X), min(a.embed_n, len(X)), replace=False)
        iso = Isomap(n_neighbors=a.map_k, n_components=2).fit(X[sub]); Y = iso.transform(X); how = "isomap"
    from scipy.stats import rankdata
    Yr = np.c_[rankdata(Y[:, 0]) / len(Y), rankdata(Y[:, 1]) / len(Y)]
    return Y, Yr, how


def continuity(Y, run, rng):
    """median step between consecutive cycles of a run / median distance of random cycle pairs:
    a path that moves smoothly over a map (like a trajectory through space) is well below 1."""
    i = np.flatnonzero(run[:-1] == run[1:])
    step = np.median(np.linalg.norm(Y[i + 1] - Y[i], axis=1))
    i1, i2 = rng.integers(0, len(Y), 20000), rng.integers(0, len(Y), 20000)
    return float(step / np.median(np.linalg.norm(Y[i1] - Y[i2], axis=1)))


def map_fields(Y, counts, run, t, a, rng):
    """'Place fields' of units on the LATENT map: per unit, rate per map bin (spikes / cycle), Skaggs
    spatial information (bits/spike), against a null that circularly shifts the unit's counts by >= 60 s
    (the map is kept), and split-half stability (odd vs even runs).  The map is built from the
    background only when --features bg, so a unit with a field is predicted by the background pattern."""
    G = a.grid
    bx = np.minimum((Y[:, 0] * G).astype(int), G - 1); by = np.minimum((Y[:, 1] * G).astype(int), G - 1)
    cell = bx * G + by
    occ = np.bincount(cell, minlength=G * G).astype(float)
    p = occ / occ.sum()
    minsh = max(1, int(np.searchsorted(t - t[0], 60.0)))
    def si(c):
        lam = np.bincount(cell, weights=c, minlength=G * G) / np.maximum(occ, 1)
        lb = (p * lam).sum()
        if lb <= 0:
            return 0.0, lam
        r = lam / lb; m = r > 0
        return float((p[m] * r[m] * np.log2(r[m])).sum()), lam
    odd = (run % 2) == 1
    out = []
    for j in range(counts.shape[1]):
        c = counts[:, j].astype(float)
        if c.sum() < a.min_field_spikes:
            continue
        s_real, lam = si(c)
        null = np.array([si(np.roll(c, int(rng.integers(minsh, len(c) - minsh))))[0] for _ in range(a.n_shuffle)])
        _, l1 = si(np.where(odd, c, 0)); _, l2 = si(np.where(~odd, c, 0))
        ok = occ >= 5
        stab = float(np.corrcoef(l1[ok], l2[ok])[0, 1]) if ok.sum() > 3 else np.nan
        out.append(dict(j=j, si=s_real, z=float((s_real - null.mean()) / (null.std() + 1e-12)), stab=stab,
                        rate=lam.reshape(G, G)))
    return out


def cmd_analyze(a):
    rng = np.random.default_rng(a.seed)
    Z = np.load(a.npz, allow_pickle=False)
    t, ratio, dur = Z["t"], Z["ratio"], Z["dur"]
    nb = Z["bg"].shape[1]
    ok = np.isfinite(Z["bg"]).all((1, 2)) & (Z["nsamp"].min(1) >= a.min_samples)
    thr = np.quantile(ratio[ok], a.theta_q)
    sel = ok & (ratio >= thr) & (dur >= 0.09) & (dur <= 0.20)
    idx = np.flatnonzero(sel)
    run = np.r_[0, np.cumsum(np.diff(idx) != 1)]
    _log(f"[analyze] {sel.sum()} theta cycles (ratio >= q{a.theta_q:g}) of {len(t)}; {run.max() + 1} runs; "
         f"{(t[idx][-1] - t[idx][0]) / 60:.1f} min span; {nb} phase bins x {Z['bg'].shape[2]} ch")
    tt = t[idx]
    F = features(Z, a.features, a.remove_level)[idx]
    F = detrend_time(F, tt, a.detrend_s)
    F = smooth_runs(F, run, a.smooth)
    X, nk = pipeline(F, a, rng)
    Fn = F.copy(); n = len(Fn); minsh = max(1, int(np.searchsorted(tt - tt[0], 60.0)))
    for j in range(Fn.shape[1]):
        Fn[:, j] = np.roll(Fn[:, j], int(rng.integers(minsh, max(minsh + 1, n - minsh))))
    Xn, _ = pipeline(Fn, a, rng, nk=nk)
    res = dict(n_cycles=int(n), phase_bins=int(nb), features=a.features, remove_level=bool(a.remove_level),
               detrend_s=a.detrend_s, smooth=a.smooth, pca_dims=nk)
    res["twonn"] = dict(real=round(twonn(X, rng), 2), null=round(twonn(Xn, rng), 2))
    rv, rdt, rfar = revisit(X, tt, a.k, a.excl_s, rng); nv, ndt, nfar = revisit(Xn, tt, a.k, a.excl_s, rng)
    res["revisit_ratio"] = dict(real=round(rv, 3), null=round(nv, 3), median_dt_s_real=round(rdt, 1),
                                median_dt_s_null=round(ndt, 1), frac_beyond_quarter_session_real=round(rfar, 3),
                                frac_beyond_quarter_session_null=round(nfar, 3))
    L = maxmin(X, a.landmarks, rng); Ln = maxmin(Xn, a.landmarks, rng)
    h, hn = homology(X[L]), homology(Xn[Ln])
    res["homology"] = dict(real=h, null=hn) if h is not None else "ripser/gudhi not installed"
    Y, Yr, how = latent_map(X, a, rng); Yn, _, _ = latent_map(Xn, a, rng)
    res["map"] = dict(method=how, continuity_real=round(continuity(Y, run, rng), 3),
                      continuity_null=round(continuity(Yn, run, rng), 3))
    # time is NOT what the map should encode: share of map variance explained by session time
    from scipy.stats import spearmanr
    res["map"]["time_spearman_xy"] = [round(float(spearmanr(tt, Y[:, k])[0]), 3) for k in (0, 1)]
    fields = []
    Uc = Z["units"]
    if Uc.shape[2]:
        counts = Uc[idx].sum(1)                                     # spikes per cycle
        fields = map_fields(Yr, counts, run, tt, a, rng)
        names = Z["unit_names"]
        zs = np.array([f["z"] for f in fields]); st = np.array([f["stab"] for f in fields])
        res["latent_fields"] = dict(units=len(fields), n_z_gt3=int((zs > 3).sum()),
                                    n_z_gt3_and_stable=int(((zs > 3) & (st > 0.5)).sum()),
                                    median_si_bits=round(float(np.median([f["si"] for f in fields])), 3) if fields else None,
                                    top=[dict(unit=str(names[f["j"]]), si=round(f["si"], 3), z=round(f["z"], 1),
                                              split_half_r=round(f["stab"], 2))
                                         for f in sorted(fields, key=lambda f: -f["z"])[:10]],
                                    note=("map built from units too (--features " + a.features + "): fields are partly "
                                          "circular" if a.features != "bg" else "map from background only; units independent"))
    pos_ok = False
    if a.pos:
        P, pt = read_pos(a.pos, a.pos_sr)
        pi = np.clip(np.searchsorted(pt, tt + 0.5 * dur[idx]), 0, len(P) - 1)
        Pc = P[pi]; good = np.isfinite(Pc).all(1)
        from sklearn.neighbors import KNeighborsRegressor
        blocks = np.array_split(np.flatnonzero(good), 10)
        def decode(XX):
            err = []
            for b in blocks:
                tr = np.setdiff1d(np.flatnonzero(good), b)
                m = KNeighborsRegressor(n_neighbors=a.k).fit(XX[tr], Pc[tr])
                err.append(np.linalg.norm(m.predict(XX[b]) - Pc[b], axis=1))
            return float(np.median(np.concatenate(err)))
        res["position"] = dict(cycles_with_pos=int(good.sum()), decode_median_err=dict(real=round(decode(X), 2), null=round(decode(Xn), 2)),
                               chance_err=round(float(np.median(np.linalg.norm(Pc[good] - np.nanmedian(Pc[good], 0), axis=1))), 2))
        pos_ok = True
    out = a.out or os.path.splitext(a.npz)[0] + f"_{a.features}{'_pattern' if (a.remove_level and a.features != 'units') else ''}"
    with open(out + ".json", "w") as fh:
        json.dump(res, fh, indent=1)
    np.savez_compressed(out + "_map.npz", xy=Y, xy_rank=Yr, t=tt, cycle=idx, X=X.astype(np.float32),
                        pos=(Pc if pos_ok else np.zeros((0, 2))))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        top = sorted(fields, key=lambda f: -f["z"])[:6]
        fig = plt.figure(figsize=(16, 7.5))
        ax = fig.add_subplot(2, 5, 1); s0 = ax.scatter(Y[:, 0], Y[:, 1], c=tt / 60, s=1, cmap="viridis")
        plt.colorbar(s0, ax=ax, label="min"); ax.set_title(f"latent map ({how}); colour: time", fontsize=9)
        ax = fig.add_subplot(2, 5, 2)
        G = a.grid; bx = np.minimum((Yr[:, 0] * G).astype(int), G - 1); by = np.minimum((Yr[:, 1] * G).astype(int), G - 1)
        ax.imshow(np.bincount(bx * G + by, minlength=G * G).reshape(G, G).T, origin="lower", cmap="Greys")
        ax.set_title("occupancy (rank-normalised map)", fontsize=9)
        ax = fig.add_subplot(2, 5, 3)
        r0 = run[np.argmax(np.bincount(run))]; m = run == r0
        ax.scatter(Y[:, 0], Y[:, 1], s=1, color="0.85")
        ax.plot(Y[m, 0], Y[m, 1], "-", color="0.3", lw=0.8); ax.scatter(Y[m, 0], Y[m, 1], c=np.arange(m.sum()), s=8, cmap="plasma")
        ax.set_title(f"longest theta run ({m.sum()} cycles)", fontsize=9)
        ax = fig.add_subplot(2, 5, 4); ax.scatter(Yn[:, 0], Yn[:, 1], c=tt / 60, s=1, cmap="viridis")
        ax.set_title("null map (feature-shifted)", fontsize=9)
        for k, f in enumerate(top):
            ax = fig.add_subplot(2, 5, 6 + k if k < 5 else 5)
            ax.imshow(f["rate"].T, origin="lower", cmap="magma")
            ax.set_title(f"{Z['unit_names'][f['j']]}  SI {f['si']:.2f}  z {f['z']:.1f}  r {f['stab']:.2f}", fontsize=8)
        for ax in fig.axes:
            ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout(); fig.savefig(out + ".png", dpi=120)
    except Exception as e:                       # plotting is optional
        _log(f"[analyze] no figure ({e})")
    print(json.dumps(res, indent=1))
    _log(f"[analyze] wrote {out}.json, {out}_map.npz, {out}.png")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("extract", help="stream the session once -> per-cycle background (+ unit counts) npz")
    e.add_argument("base", help="session base name (the <base>.yaml must exist)")
    e.add_argument("--groups", default="all", help="spike groups, e.g. 'all' or '1-8,10'")
    e.add_argument("--fil", default=None, help="filtered signal (default <base>.fil)")
    e.add_argument("--signal-start", type=int, default=0, help="absolute sample of the .fil's first row")
    e.add_argument("--lfp", default=None, help="LFP source (default <base>.lfp); a wide-band file is decimated")
    e.add_argument("--lfp-sr", type=float, default=None, help="its sampling rate (default yaml lfpSamplingRate)")
    e.add_argument("--lfp-nchan", type=int, default=None, help="its channel count (default yaml nChannels)")
    e.add_argument("--lfp-start-s", type=float, default=0.0, help="session time (s) of the LFP's first sample")
    e.add_argument("--theta-channels", default=None, help="LFP channels for the theta median (default: all group channels)")
    e.add_argument("--dead-channels", default=None, help="channels to ignore everywhere")
    e.add_argument("--theta-lo", type=float, default=5.0); e.add_argument("--theta-hi", type=float, default=11.0)
    e.add_argument("--res-variant", default="", help="pin the .res variant (default: the shared .res of each group)")
    e.add_argument("--guard-ms", type=float, default=1.0, help="samples within this of a detected spike are excluded")
    e.add_argument("--clu-method", default="stderiv_C5_D34"); e.add_argument("--clu-stage", default="",
                   help="clu stage for unit counts (empty = background only)")
    e.add_argument("--min-unit-spikes", type=int, default=200)
    e.add_argument("--phase-bins", type=int, default=8, help="theta phase bins per cycle, starting at the trough")
    e.add_argument("--phase-mode", choices=["phase", "time"], default="phase",
                   help="phase = equal bins of the Hilbert theta phase (default); time = equal-duration bins")
    e.add_argument("--chunk-s", type=float, default=30.0, help="seconds of .fil per read (30 s x 96 ch = 375 MB float32)")
    e.add_argument("--out", required=True)
    n = sp.add_parser("analyze", help="topology of the per-cycle background vs a feature-shift null")
    n.add_argument("npz")
    n.add_argument("--features", choices=["bg", "units", "both"], default="bg")
    n.add_argument("--remove-level", type=int, default=1, help="1 = drop each cycle's overall background level")
    n.add_argument("--theta-q", type=float, default=0.5, help="theta/delta ratio quantile for a theta cycle")
    n.add_argument("--min-samples", type=int, default=50, help="spike-free samples needed per channel x quarter")
    n.add_argument("--smooth", type=int, default=3, help="moving average over this many consecutive cycles")
    n.add_argument("--pca-var", type=float, default=0.9); n.add_argument("--pca-max", type=int, default=30)
    n.add_argument("--k", type=int, default=10); n.add_argument("--excl-s", type=float, default=30.0,
                   help="neighbours within +-this many seconds do not count as revisits")
    n.add_argument("--landmarks", type=int, default=300); n.add_argument("--embed-n", type=int, default=5000)
    n.add_argument("--detrend-s", type=float, default=60.0,
                   help="subtract each feature's running mean over this window (s); 0 = keep slow drift")
    n.add_argument("--map-k", type=int, default=30, help="neighbours for the 2-D latent map (UMAP / Isomap)")
    n.add_argument("--grid", type=int, default=12, help="bins per axis of the latent map for unit fields")
    n.add_argument("--min-field-spikes", type=int, default=300)
    n.add_argument("--n-shuffle", type=int, default=100, help="circular shifts per unit for the field null")
    n.add_argument("--pos", default=None, help=".whl/.pos text file: x y per row (-1 = missing)")
    n.add_argument("--pos-sr", type=float, default=39.0625)
    n.add_argument("--seed", type=int, default=0); n.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    (cmd_extract if a.cmd == "extract" else cmd_analyze)(a)


if __name__ == "__main__":
    main()
