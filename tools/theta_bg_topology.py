#!/usr/bin/env python3
"""Theta-cycle background structure and place-like topology over a FULL session.

Two steps (run `extract` once, iterate on `analyze`):

  extract   streams the session once and writes <out>.npz:
              * theta cycles from the LFP (5-11 Hz, trough to trough) with theta/delta ratio,
                duration and amplitude per cycle (all cycles; the theta selection is made in analyze)
              * BACKGROUND per cycle x theta quarter x channel: mean-square of the .fil over samples at
                least --guard-ms from any detected spike of that channel's group (spike-free activity)
              * optional UNIT counts per cycle x quarter (from a clu stage of every group)

  analyze   per theta cycle: features (background pattern, unit counts, or both), smoothed over a few
            consecutive cycles, z-scored, PCA.  Then, against a NULL that circularly shifts every
            feature independently (destroying co-activation, keeping each feature's own statistics):
              * intrinsic dimension (TwoNN) -- a 2-D place map should sit near 2
              * revisit structure: nearest neighbours AFTER excluding +-T s of the cycle's own time; a
                map the animal revisits puts distant-in-time cycles close in feature space
              * persistent homology (ripser or gudhi, if installed) on max-min landmarks: an open arena
                is contractible (one H0 class, no long H1), a linear/circular track gives one long H1
              * Isomap 2-D embedding coloured by time (and by position when given)
              * with --pos (.whl: x y per row at --pos-sr, -1 = missing): blocked cross-validated kNN
                position decoding vs the null, and feature-distance vs position-distance (Spearman)

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
    cyc, _ = theta_cycles(x, fs, band=(a.theta_lo, a.theta_hi))
    ncyc = len(cyc["start"])
    _log(f"[extract] {ncyc} cycles over {x.size / fs / 60:.1f} min (median {1e3 * np.median(cyc['dur']):.0f} ms)")
    # quarter-bin per LFP sample (-1 outside cycles)
    qbin = np.full(x.size, -1, np.int64)
    for c in range(ncyc):
        s, e = cyc["start"][c], cyc["end"][c]
        q = np.minimum(((np.arange(s, e) - s) * 4) // max(e - s, 1), 3)
        qbin[s:e] = c * 4 + q
    lfp_t0 = a.lfp_start_s

    # ── background from the .fil ──
    fil_path = a.fil or f"{base}.fil"
    F = np.memmap(fil_path, np.int16, "r").reshape(-1, ntot)
    sig0 = int(a.signal_start)                         # absolute sample of the .fil's first row
    guard = int(round(a.guard_ms * 1e-3 * sr))
    allch = sorted({c for g in groups for c in ginfo[g]["channels"] if c not in dead})
    col = {c: i for i, c in enumerate(allch)}
    E = np.zeros((ncyc * 4, len(allch)), np.float64); Ncnt = np.zeros((ncyc * 4, len(allch)), np.float64)
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
                E[:, col[c]] += np.bincount(bb, weights=(v.astype(np.float64)) ** 2, minlength=ncyc * 4)
                Ncnt[:, col[c]] += np.bincount(bb, minlength=ncyc * 4)
        _log(f"[extract] .fil {s1 / sr / 60:6.1f} / {F.shape[0] / sr / 60:.1f} min")
    with np.errstate(invalid="ignore", divide="ignore"):
        bg = (E / Ncnt).reshape(ncyc, 4, len(allch)).astype(np.float32)
    nsamp = Ncnt.reshape(ncyc, 4, len(allch)).min(2).astype(np.int32)

    # ── optional unit counts ──
    U = None; unames = []
    if a.clu_stage:
        cols = []
        for g in groups:
            try:
                _, clu = nio.read_clu_file(nio.session_path(base, "clu", g, variant=a.clu_method, tag=a.clu_stage),
                                           n_spikes=res[g].size)
            except (FileNotFoundError, OSError) as e:
                _log(f"[extract] group {g}: no clu ({e}); units skipped"); continue
            rg = _res_for_group(nio, base, g, a.res_variant)       # unsorted order matches the clu
            li = np.floor((rg / sr - lfp_t0) * fs).astype(np.int64)
            ok = (li >= 0) & (li < x.size); bi = np.full(rg.size, -1, np.int64); bi[ok] = qbin[li[ok]]
            clu = np.asarray(clu)
            for u in np.unique(clu[clu > 1]):
                m = (clu == u) & (bi >= 0)
                if (clu == u).sum() < a.min_unit_spikes:
                    continue
                cols.append(np.bincount(bi[m], minlength=ncyc * 4).reshape(ncyc, 4)); unames.append(f"{g}:{u}")
        if cols:
            U = np.stack(cols, 2).astype(np.uint16)
            _log(f"[extract] {len(unames)} units counted per cycle quarter")
    out = a.out if a.out.endswith(".npz") else a.out + ".npz"
    np.savez_compressed(out, t=(cyc["start"] / fs + lfp_t0), dur=cyc["dur"], ratio=cyc["ratio"], amp=cyc["amp"],
                        bg=bg, nsamp=nsamp, channels=np.array(allch), units=(U if U is not None else np.zeros((ncyc, 4, 0))),
                        unit_names=np.array(unames), meta=json.dumps(dict(base=base, groups=groups, sr=sr, lfp_fs=fs,
                                                                          guard_ms=a.guard_ms, theta=[a.theta_lo, a.theta_hi])))
    _log(f"[extract] wrote {out}: {ncyc} cycles x 4 quarters x {len(allch)} ch" + (f" + {len(unames)} units" if unames else ""))


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
    bg = np.log(Z["bg"].astype(np.float64) + 1e-9)                # cycles x 4 x ch
    blocks = []
    if kind in ("bg", "both"):
        B = bg.reshape(len(bg), -1)
        if remove_level:                                             # pattern: drop each cycle's overall level
            B = B - B.mean(1, keepdims=True)
        blocks.append(B)
    if kind in ("units", "both"):
        Uc = Z["units"].astype(np.float64)
        if Uc.shape[2] == 0:
            raise SystemExit("no unit counts in the npz (run extract with --clu-stage)")
        blocks.append(np.log1p(Uc.reshape(len(Uc), -1)))
    return np.hstack(blocks)


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


def cmd_analyze(a):
    rng = np.random.default_rng(a.seed)
    Z = np.load(a.npz, allow_pickle=False)
    t, ratio, dur = Z["t"], Z["ratio"], Z["dur"]
    ok = np.isfinite(Z["bg"]).all((1, 2)) & (Z["nsamp"].min(1) >= a.min_samples)
    thr = np.quantile(ratio[ok], a.theta_q)
    sel = ok & (ratio >= thr) & (dur >= 0.09) & (dur <= 0.20)
    idx = np.flatnonzero(sel)
    run = np.r_[0, np.cumsum(np.diff(idx) != 1)]
    _log(f"[analyze] {sel.sum()} theta cycles (ratio >= q{a.theta_q:g}) of {len(t)}; {run.max() + 1} runs; "
         f"{(t[idx][-1] - t[idx][0]) / 60:.1f} min span")
    F = features(Z, a.features, a.remove_level)[idx]
    F = smooth_runs(F, run, a.smooth)
    tt = t[idx]
    X, nk = pipeline(F, a, rng)
    # null: independent circular shift of every feature (>= 60 s), same pipeline
    Fn = F.copy(); n = len(Fn); minsh = max(1, int(np.searchsorted(tt - tt[0], 60.0)))
    for j in range(Fn.shape[1]):
        Fn[:, j] = np.roll(Fn[:, j], int(rng.integers(minsh, max(minsh + 1, n - minsh))))
    Xn, nkn = pipeline(Fn, a, rng, nk=nk)
    res = dict(n_cycles=int(n), features=a.features, remove_level=bool(a.remove_level), smooth=a.smooth,
               pca_dims=nk, pca_dims_null=nkn)
    res["twonn"] = dict(real=round(twonn(X, rng), 2), null=round(twonn(Xn, rng), 2))
    rv, rdt, rfar = revisit(X, tt, a.k, a.excl_s, rng); nv, ndt, nfar = revisit(Xn, tt, a.k, a.excl_s, rng)
    # A slowly drifting state also gives revisits, but at SHORT lags; a map the animal re-enters gives
    # neighbours spread over the whole session -> frac_beyond_quarter_session.
    res["revisit_ratio"] = dict(real=round(rv, 3), null=round(nv, 3), median_dt_s_real=round(rdt, 1),
                                median_dt_s_null=round(ndt, 1), frac_beyond_quarter_session_real=round(rfar, 3),
                                frac_beyond_quarter_session_null=round(nfar, 3))
    L = maxmin(X, a.landmarks, rng); Ln = maxmin(Xn, a.landmarks, rng)
    h, hn = homology(X[L]), homology(Xn[Ln])
    res["homology"] = dict(real=h, null=hn) if h is not None else "ripser/gudhi not installed"
    # embedding
    from sklearn.manifold import Isomap
    sub = rng.choice(n, min(a.embed_n, n), replace=False); sub.sort()
    Y = Isomap(n_neighbors=a.k, n_components=2).fit_transform(X[sub])
    pos_ok = False
    if a.pos:
        P, pt = read_pos(a.pos, a.pos_sr)
        pi = np.clip(np.searchsorted(pt, tt + 0.5 * dur[idx]), 0, len(P) - 1)
        Pc = P[pi]; good = np.isfinite(Pc).all(1)
        from sklearn.neighbors import KNeighborsRegressor
        from scipy.stats import spearmanr
        blocks = np.array_split(np.flatnonzero(good), 10)
        def decode(XX):
            err = []
            for b in blocks:
                tr = np.setdiff1d(np.flatnonzero(good), b)
                m = KNeighborsRegressor(n_neighbors=a.k).fit(XX[tr], Pc[tr])
                err.append(np.linalg.norm(m.predict(XX[b]) - Pc[b], axis=1))
            return float(np.median(np.concatenate(err)))
        gi = np.flatnonzero(good)
        i1, i2 = rng.choice(gi, 20000), rng.choice(gi, 20000)
        far = np.abs(tt[i1] - tt[i2]) > a.excl_s
        def mantel(XX):
            return float(spearmanr(np.linalg.norm(XX[i1[far]] - XX[i2[far]], axis=1),
                                   np.linalg.norm(Pc[i1[far]] - Pc[i2[far]], axis=1))[0])
        res["position"] = dict(cycles_with_pos=int(good.sum()), decode_median_err=dict(real=round(decode(X), 2), null=round(decode(Xn), 2)),
                               chance_err=round(float(np.median(np.linalg.norm(Pc[gi] - np.nanmedian(Pc[gi], 0), axis=1))), 2),
                               feature_vs_place_distance_spearman=dict(real=round(mantel(X), 3), null=round(mantel(Xn), 3)))
        pos_ok = True
    out = a.out or os.path.splitext(a.npz)[0] + f"_{a.features}{'_pattern' if (a.remove_level and a.features != 'units') else ''}"
    with open(out + ".json", "w") as fh:
        json.dump(res, fh, indent=1)
    np.savez_compressed(out + "_embed.npz", Y=Y, t=tt[sub], X=X[sub], pos=(Pc[sub] if pos_ok else np.zeros((0, 2))))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        ncol = 3 if pos_ok else 1
        fig, ax = plt.subplots(1, ncol, figsize=(5 * ncol, 4.6), squeeze=False); ax = ax[0]
        s0 = ax[0].scatter(Y[:, 0], Y[:, 1], c=tt[sub] / 60, s=3, cmap="viridis"); plt.colorbar(s0, ax=ax[0], label="min")
        ax[0].set_title("Isomap of theta cycles (colour: time)", fontsize=9)
        if pos_ok:
            for k, lab in ((0, "x"), (1, "y")):
                v = Pc[sub][:, k]; s = ax[k + 1].scatter(Y[:, 0], Y[:, 1], c=v, s=3, cmap="twilight")
                plt.colorbar(s, ax=ax[k + 1], label=f"position {lab}"); ax[k + 1].set_title(f"colour: position {lab}", fontsize=9)
        for x_ in ax:
            x_.set_xticks([]); x_.set_yticks([])
        fig.tight_layout(); fig.savefig(out + ".png", dpi=120)
    except Exception as e:                       # plotting is optional
        _log(f"[analyze] no figure ({e})")
    print(json.dumps(res, indent=1))
    _log(f"[analyze] wrote {out}.json, {out}_embed.npz, {out}.png")


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
    n.add_argument("--pos", default=None, help=".whl/.pos text file: x y per row (-1 = missing)")
    n.add_argument("--pos-sr", type=float, default=39.0625)
    n.add_argument("--seed", type=int, default=0); n.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    (cmd_extract if a.cmd == "extract" else cmd_analyze)(a)


if __name__ == "__main__":
    main()
