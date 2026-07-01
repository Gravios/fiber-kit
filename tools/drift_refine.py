#!/usr/bin/env python3
"""drift_refine.py -- iteratively co-estimate a single coherent probe drift and the high-SNR chains,
until every chain agrees on that drift.

The tissue drifts rigidly, so every cell's footprint-centroid depth is  y_cell(t) = baseline_cell +
Delta(t)  for ONE shared drift Delta(t).  Starting from an initial peel (piece_interneurons.chase_from),
this alternates:

  M-step  fit the rigid model: amplitude-weighted Delta(chunk) shared across chains + a per-chain
          baseline depth (alternating weighted means; Delta gauged mean-zero).
  E-step  PREDICT each chain's centroid at every chunk (baseline + Delta), i.e. where its low-variance
          identity channels should sit under the drift, and RE-ASSIGN every fragment to the chain that
          best matches on BOTH primary-channel waveform cosine AND drift-predicted depth.

Iterating tightens the chains onto the shared drift: fragments that a waveform-only chase mis-linked
across the co-located crowd get pulled back to the chain whose drift-predicted depth they actually sit
at.  It stops when the assignments stop changing -- "all chains agree on the drift given their
amplitude/distance" (amplitude^2 weighting = the point-source amplitude~1/distance, so closer/louder
fragments pin the drift more).

Usage:
    python3 tools/drift_refine.py <session> <group> [--snr-thr 8] [--iters 10] \
        [--gap-min 60] [--cos-thr 0.92] [--pos-tol 8] [--cos-tol 0.08] [--celltype int|pyr] \
        [--tsv drift_refine.tsv] [--out drift_refine.png]
"""
import argparse
import os
import sys
import numpy as np

try:
    from fiber_kit import (fiber_lib as fl, session_yaml as sy, neuro_io as nio,
                           fiber_localize as loc, fiber_geometry as fg)
except ImportError:
    import fiber_lib as fl, session_yaml as sy, neuro_io as nio, fiber_localize as loc, fiber_geometry as fg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import piece_interneurons as pi                      # chase_from, _pcos


def build_pool(spk, res, ids, pos, *, sr, min_n, snr_thr, cap, chunk_min, celltype=None, seed=0):
    """High-SNR fragment dicts: clu, tmid, chunk, y (energy-weighted depth centroid), amp, dom, snr, t."""
    rng = np.random.default_rng(seed)
    tmin = (res - res.min()) / sr / 60.0
    uniq, cnt = np.unique(ids, return_counts=True)
    pool = []
    for u, c in zip(uniq, cnt):
        if u < 2 or c < min_n:
            continue
        idx = np.flatnonzero(ids == u)
        s = idx if len(idx) <= cap else rng.choice(idx, cap, replace=False)
        w = np.asarray(spk[np.sort(s)], float)
        t = np.median(fl.align_xcorr(w, ref="median", iters=4), axis=0)
        amp = np.ptp(t, axis=0); dom = int(np.argmax(amp))
        pk = int(np.argmin(t[:, dom])); resid = w[:, pk, dom] - np.median(w[:, pk, dom])
        noise = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-9
        snr = float(amp[dom] / noise)
        if snr < snr_thr:
            continue
        if celltype and fg.classify_celltype(t, sr) != celltype:
            continue
        ew = amp.astype(float) ** 2; ew /= ew.sum()
        tm = float(np.median(tmin[idx]))
        pool.append(dict(clu=int(u), tmid=tm, chunk=int(tm // chunk_min), y=float(ew @ pos[:, 1]),
                         amp=amp, dom=dom, snr=snr, t=t))
    return pool


def peel_init(pool, P, *, gap_min, cos_thr, amp_ratio, prim_frac):
    """Initial partition: seed highest-SNR unassigned, chase, remove, repeat."""
    avail = {f["clu"] for f in pool}; chains = []
    while avail:
        seed = max(avail, key=lambda u: P[u]["snr"])
        sub = sorted((P[u] for u in avail), key=lambda f: f["tmid"])
        si = next(k for k, f in enumerate(sub) if f["clu"] == seed)
        order = pi.chase_from(sub, si, gap_min=gap_min, cos_thr=cos_thr, amp_ratio=amp_ratio, prim_frac=prim_frac)
        ch = [sub[i]["clu"] for i in order]
        if len(ch) >= 2:
            chains.append(ch); avail -= set(ch)
        else:
            avail.discard(seed)
    return chains


def fit_rigid_drift(chains, P, iters=40):
    """Fit y = baseline[chain] + Delta[chunk] by alternating amplitude^2-weighted means; Delta gauged
    mean-zero.  Returns (baselines list, Delta dict)."""
    chunks = sorted({P[u]["chunk"] for c in chains for u in c})
    base = [0.0] * len(chains); delta = {k: 0.0 for k in chunks}
    for _ in range(iters):
        for k in chunks:
            num = den = 0.0
            for ci, c in enumerate(chains):
                for u in c:
                    f = P[u]
                    if f["chunk"] == k:
                        wt = f["amp"][f["dom"]] ** 2; num += wt * (f["y"] - base[ci]); den += wt
            if den:
                delta[k] = num / den
        m = np.mean(list(delta.values()))
        for k in delta:
            delta[k] -= m
        for ci, c in enumerate(chains):
            num = den = 0.0
            for u in c:
                f = P[u]; wt = f["amp"][f["dom"]] ** 2; num += wt * (f["y"] - delta[f["chunk"]]); den += wt
            base[ci] = num / den if den else 0.0
    return base, delta


def drift_rms(chains, P, base, delta):
    r = [P[u]["y"] - (base[ci] + delta[P[u]["chunk"]]) for ci, c in enumerate(chains) for u in c]
    return float(np.sqrt(np.mean(np.square(r)))) if r else float("nan")


def refine(pool, P, *, iters, gap_min, cos_thr, amp_ratio, prim_frac, pos_tol, cos_tol):
    """EM loop: fit drift, then re-assign each fragment by waveform cosine + drift-predicted depth."""
    chains = peel_init(pool, P, gap_min=gap_min, cos_thr=cos_thr, amp_ratio=amp_ratio, prim_frac=prim_frac)
    hist = []; prev = None
    for it in range(iters):
        base, delta = fit_rigid_drift(chains, P)
        rms = drift_rms(chains, P, base, delta)
        span = max(delta.values()) - min(delta.values())
        hist.append((len(chains), rms, span))
        T = [np.median(np.array([P[u]["t"] for u in c]), axis=0) for c in chains]
        new = [[] for _ in chains]
        for f in pool:
            best, bs = -1, 1e18
            for ci, c in enumerate(chains):
                if not any(abs(P[u]["chunk"] - f["chunk"]) <= 1 for u in c):
                    continue
                pred = base[ci] + delta.get(f["chunk"], 0.0)
                score = (1 - pi._pcos(f, {"t": T[ci], "amp": np.ptp(T[ci], 0)}, prim_frac)) / cos_tol \
                    + abs(f["y"] - pred) / pos_tol
                if score < bs:
                    bs, best = score, ci
            new[best if best >= 0 else 0].append(f["clu"])
        chains = [c for c in new if len(c) >= 2]
        key = tuple(sorted(tuple(sorted(c)) for c in chains))
        if key == prev:
            hist.append((len(chains), rms, span)); break
        prev = key
    base, delta = fit_rigid_drift(chains, P)
    return chains, base, delta, hist


def main():
    ap = argparse.ArgumentParser(prog="drift_refine", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sy.add_session_args(ap)
    ap.add_argument("--variant", default="stderiv"); ap.add_argument("--stage", default="fiber_session")
    ap.add_argument("--snr-thr", type=float, default=8.0); ap.add_argument("--min-n", type=int, default=200)
    ap.add_argument("--cap", type=int, default=800); ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--celltype", choices=["int", "pyr", ""], default="",
                    help="restrict the pool to a cell class (int/pyr); default '' = ALL high-SNR clusters")
    ap.add_argument("--gap-min", type=float, default=60.0); ap.add_argument("--cos-thr", type=float, default=0.92)
    ap.add_argument("--amp-ratio", type=float, default=2.2); ap.add_argument("--prim-frac", type=float, default=0.3)
    ap.add_argument("--pos-tol", type=float, default=8.0, help="depth tolerance (um) in the re-assignment score")
    ap.add_argument("--cos-tol", type=float, default=0.08, help="waveform tolerance in the re-assignment score")
    ap.add_argument("--tsv", default=None); ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = sy.resolve_session_params(a.session, a.group, channels=a.channels, ntotal=a.ntotal,
                                    nchan=a.nchan, nsamp=a.nsamp, sr=a.sr)
    base_p = cfg["base"]; elec = a.group; nsamp = cfg["nsamp"]; nchan = cfg["nchan"]; sr = cfg["sr"]
    pos = np.asarray(loc.load_geometry(cfg["probe"], cfg["channels"]), float)
    chunk_min = 12.0
    res = nio.read_res(base_p, elec)
    spk, _ = nio.open_spk_raw(base_p, elec, nsamp, nchan)
    _, ids = nio.read_clu_at(base_p, elec, variant=a.variant, tag=a.stage, n_spikes=len(res))

    pool = build_pool(spk, res, ids, pos, sr=sr, min_n=a.min_n, snr_thr=a.snr_thr, cap=a.cap, chunk_min=chunk_min, celltype=a.celltype or None)
    if len(pool) < 4:
        raise SystemExit(f"[drift_refine] pool has {len(pool)} clusters (lower --snr-thr / --min-n)")
    P = {f["clu"]: f for f in pool}
    print(f"[drift_refine] {os.path.basename(base_p)} elec {elec}: {len(pool)} high-SNR clusters SNR>={a.snr_thr:g}")
    chains, cbase, delta, hist = refine(pool, P, iters=a.iters, gap_min=a.gap_min, cos_thr=a.cos_thr,
                                        amp_ratio=a.amp_ratio, prim_frac=a.prim_frac,
                                        pos_tol=a.pos_tol, cos_tol=a.cos_tol)
    for it, (nc, rms, span) in enumerate(hist):
        print(f"  iter {it}: {nc} chains, drift residual RMS {rms:.2f} um, drift span {span:.1f} um")
    per = [np.sqrt(np.mean([(P[u]["y"] - (cbase[ci] + delta[P[u]["chunk"]])) ** 2 for u in c]))
           for ci, c in enumerate(chains)]
    dk = sorted(delta)
    print(f"  CONVERGED: {len(chains)} chains, coherent drift span {max(delta.values())-min(delta.values()):.1f} um, "
          f"per-chain residual median {np.median(per):.2f} um / worst {max(per):.2f} um "
          f"({'all agree' if max(per) < 4 else 'some disagreement'})")

    if a.tsv:
        with open(a.tsv, "w") as fh:
            fh.write("# coherent drift Delta(t)\nt_min\tdelta_um\n")
            for k in dk:
                fh.write(f"{k*chunk_min:.1f}\t{delta[k]:.2f}\n")
            fh.write("# chains: baseline depth + residual\nchain\tn\tbaseline_um\tresidual_um\tmembers\n")
            for ci, c in enumerate(chains):
                fh.write(f"{ci}\t{len(c)}\t{cbase[ci]:.2f}\t{per[ci]:.2f}\t{','.join(map(str, sorted(c)))}\n")
        print(f"  wrote {a.tsv}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (axc, axd, axr) = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    axc.plot([h[1] for h in hist], "-o"); axc.set_xlabel("iteration"); axc.set_ylabel("drift residual RMS (um)")
    axc.set_title("convergence"); axc.grid(alpha=0.3)
    axd.plot([k * chunk_min for k in dk], [delta[k] for k in dk], "-o", color="#264653")
    axd.set_xlabel("time (min)"); axd.set_ylabel("coherent drift Delta (um)"); axd.set_title("shared drift")
    axd.grid(alpha=0.3)
    for ci, c in enumerate(chains):
        cc = sorted((P[u] for u in c), key=lambda f: f["tmid"])
        axr.plot([f["tmid"] for f in cc], [f["y"] - cbase[ci] for f in cc], "-", alpha=0.4, lw=0.8)
    axr.plot([k * chunk_min for k in dk], [delta[k] for k in dk], "k-", lw=2.5, label="coherent")
    axr.set_xlabel("time (min)"); axr.set_ylabel("y - baseline (um)"); axr.set_title("chains vs shared drift")
    axr.legend(fontsize=7); axr.grid(alpha=0.3)
    out = a.out or f"{base_p}.drift_refine.{elec}.png"
    fig.savefig(out, dpi=120); print(f"  wrote {out}")


if __name__ == "__main__":
    main()
