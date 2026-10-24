#!/usr/bin/env python3
"""fiber-rdr -- per-channel contamination from the RMS of the residual's temporal derivative.

For each cluster the raw (standard) windows are aligned as Klusters' realign aligns a cluster
(fiber_realign.klusters_offsets, threshold 0.10), a median template is taken per time chunk, and per spike
and channel

    D(s, c) = sqrt( mean_t [ r(t+1, c) - r(t, c) ]^2 ),   r = aligned spike - template

over the samples that are real after the shift (no circular wrap), divided by the noise's derivative RMS
sigma_d(c).  Pure noise sits at 1.

A clean cell's D still grows with the channel's amplitude A (in noise SD): integer-alignment jitter and
the cell's own gain spread scale with the template.  Noise and mismatch add in variance, so over clean
channels

    median D^2 = D0^2 + k A^2      (g6 0316: 1.105 + 0.00153 A^2; g7 0312: 1.009 + 0.00173 A^2)

and a cluster's score on channel c is the observed excess over the noise floor against the line's:

    ratio(c) = sqrt(max(med D^2 - D0^2, 0)) / (sqrt(k) A),   on channels with A >= --min-amp (4 SD)

The cluster's score is the maximum over those channels.  A clean cluster sits near 1.  The line is fitted
on every cluster's channels with a 3-MAD clip (no curation needed), or fixed with --line D0SQ,K.

Validated (docs/stages.md, fiber-rdr): g6 0316 exp_final, ACG-labelled units -- dirty 1.77-2.57, clean <= 1.65 with
per-chunk templates; g7 0312 minutes 168-204, fiber-session clusters against the old curation -- AUC
0.87 for clusters < 90% pure (pure median 1.04, mixed 2.51; at 1.7 it flags 73% of the mixed and 17% of
the pure ones).

Noise: sigma(c) and sigma_d(c) come from the group's .nst (exact, from its covariance) when present,
else from the first 8 samples of a random sample of windows (robust MAD).  This is a QC report: it
writes a table, it moves no spike.
"""
import argparse
import os
import sys

import numpy as np

try:
    from . import neuro_io as nio, session_yaml as sy, fiber_realign as fr
except ImportError:                                              # script / direct execution
    import neuro_io as nio, session_yaml as sy, fiber_realign as fr

_LP = "▸ fiber-rdr"


def _mad(x):
    x = np.asarray(x, float).ravel()
    return 1.4826 * float(np.median(np.abs(x - np.median(x))))


def noise_sigma_d(cov, nsamp, nchan):
    """Exact RMS of the first difference of the noise per channel, from a .nst covariance in the .spk row
    layout (element t*nchan + c): sigma_d^2(c) = mean_t [S(t+1,t+1) + S(t,t) - 2 S(t,t+1)]."""
    cov = np.asarray(cov, float)
    out = np.zeros(nchan)
    for c in range(nchan):
        ix = np.arange(nsamp) * nchan + c
        S = cov[np.ix_(ix, ix)]
        out[c] = np.sqrt(np.mean(np.diag(S)[1:] + np.diag(S)[:-1] - 2 * np.diag(S, 1)))
    return out


def noise_from_windows(spk, *, n=60000, head=8, rng=None):
    """(sigma, sigma_d) per channel from the first `head` samples of up to n random windows (robust MAD).
    The fallback when the group has no .nst; the window's lead-in is mostly background."""
    rng = np.random.default_rng(0) if rng is None else rng
    N = spk.shape[0]
    sel = np.sort(rng.choice(N, min(n, N), replace=False))
    B = np.asarray(spk[sel])[:, :head, :].astype(float)
    C = B.shape[2]
    return (np.array([_mad(B[:, :, c]) for c in range(C)]),
            np.array([_mad(np.diff(B[:, :, c], axis=1)) for c in range(C)]))


def residual_derivative(W, ioff, groups=None):
    """D (n, C) in ADU for windows W (n, T, C) aligned by integer offsets ioff (roll_spikes convention),
    with one median template per value of `groups` (None: one template).  Returns (D, templates dict)."""
    W = np.asarray(W)
    ioff = np.asarray(ioff, int)
    n, T, C = W.shape
    A = fr.roll_spikes(W, ioff).astype(np.float64)
    g = np.zeros(n, int) if groups is None else np.asarray(groups)
    R = np.empty_like(A)
    tm = {}
    for k in np.unique(g):
        m = g == k
        tm[int(k)] = np.median(A[m], 0)
        R[m] = A[m] - tm[int(k)]
    t = np.arange(T)[None, :]
    real = (t + ioff[:, None] >= 0) & (t + ioff[:, None] < T)
    ok = real[:, 1:] & real[:, :-1]                               # both ends of the difference are real
    dR = np.diff(R, axis=1)
    D = np.sqrt((dR ** 2 * ok[:, :, None]).sum(1) / np.maximum(ok.sum(1), 1)[:, None])
    return D, tm


def cluster_profile(spk, idx, *, peak, sigma, sigma_d, chunk=None, cap=4000, min_score=0.10, rng=None):
    """Per-channel (amplitude in noise SD, median D in noise-derivative units) of one cluster.
    `chunk` (len(spk),) gives each spike's time chunk: one template per chunk (drift removed); the
    amplitude is the spike-weighted mean of the chunk templates' peak-to-peak.  `peak` is the YAML's
    1-based peakSampleIndex."""
    rng = np.random.default_rng(0) if rng is None else rng
    idx = np.asarray(idx)
    if idx.size > cap:
        idx = np.sort(rng.choice(idx, cap, replace=False))
    W = np.asarray(spk[idx])
    _, ioff = fr.klusters_offsets(W, np.full(len(W), 2), int(peak) - 1, min_n=1, min_score=min_score)
    g = None if chunk is None else np.asarray(chunk)[idx]
    D, tm = residual_derivative(W, ioff, g)
    gg = np.zeros(len(W), int) if g is None else g
    amp = sum((gg == k).sum() * (T.max(0) - T.min(0)) for k, T in tm.items()) / len(W)
    return amp / np.asarray(sigma), np.median(D, 0) / np.asarray(sigma_d), int(idx.size)


def fit_clean_line(A, M, *, clip=3.0, iters=3):
    """Robust fit of median D^2 = d0 + k A^2 over channels (rows of A/M flattened): least squares, then
    drop points beyond `clip` MADs of the residual and refit.  Contaminated channels sit above the line
    and are clipped.  Returns (d0, k)."""
    x = np.asarray(A, float).ravel() ** 2
    y = np.asarray(M, float).ravel() ** 2
    keep = np.isfinite(x) & np.isfinite(y)
    k, d0 = np.polyfit(x[keep], y[keep], 1)
    for _ in range(iters):
        r = y - (d0 + k * x)
        keep = np.isfinite(r) & (np.abs(r - np.median(r[keep])) <= clip * _mad(r[keep]) + 1e-12)
        k, d0 = np.polyfit(x[keep], y[keep], 1)
    return float(d0), float(max(k, 1e-12))


def excess_ratio(A, M, d0, k, *, min_amp=4.0):
    """Observed / expected excess per channel (NaN below min_amp)."""
    amp = np.asarray(A, float)
    ex = np.sqrt(np.clip(np.asarray(M, float) ** 2 - d0, 0, None))
    return np.where(amp >= min_amp, ex / (np.sqrt(k) * np.maximum(amp, 1e-12)), np.nan)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="fiber-rdr",
        description="Per-cluster contamination score from the RMS of the residual's temporal derivative, "
                    "against the clean amplitude line (a QC table; moves no spike).")
    ap.add_argument("session"); ap.add_argument("elec", type=int)
    ap.add_argument("--clu-method", default="stderiv")
    ap.add_argument("--clu-stage", default="", help="post-fiber stage tag of the .clu to score")
    ap.add_argument("--in-clu", default=None, help="explicit .clu path (overrides --clu-method/--clu-stage)")
    ap.add_argument("--spk-method", default="standard", help="waveform axis (raw windows; default standard)")
    ap.add_argument("--nst", default=None, help=".nst path (default <base>.nst.<spk-method>.<elec> if present)")
    ap.add_argument("--chunk-min", type=float, default=12.0, help="one template per chunk of this length (0: one per cluster)")
    ap.add_argument("--min-n", type=int, default=40, help="clusters with fewer spikes are skipped")
    ap.add_argument("--cap", type=int, default=4000, help="spikes sampled per cluster")
    ap.add_argument("--min-amp", type=float, default=4.0, help="channels below this amplitude (noise SD) are not scored")
    ap.add_argument("--align-min-score", type=float, default=0.10, help="Klusters realign threshold (Klusters' own: 0.70)")
    ap.add_argument("--thr", type=float, default=1.7, help="flag clusters whose ratio reaches this")
    ap.add_argument("--line", default=None, help="fixed clean line D0SQ,K instead of fitting it")
    ap.add_argument("--out", default=None, help="output TSV (default <base>.rdr.<clu-method>.<elec>[.<stage>].tsv)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(a.seed)

    cfg = sy.resolve_session_params(a.session, a.elec)
    base, SR, NS, NC, PK = cfg["base"], cfg["sr"], cfg["nsamp"], cfg["nchan"], cfg["peak"]
    res = nio.read_res(base, a.elec)
    if a.in_clu:
        _, clu = nio.read_clu_file(a.in_clu, n_spikes=res.size)
    else:
        _, clu = nio.read_clu_at(base, a.elec, variant=a.clu_method, tag=a.clu_stage, n_spikes=res.size)
    spk = nio.open_spk_file(nio.session_path(base, "spk", a.elec, variant=a.spk_method), NS, NC)
    if not (res.size == clu.size == spk.shape[0]):
        raise SystemExit(f"{_LP} · res/clu/spk length mismatch ({res.size}/{clu.size}/{spk.shape[0]})")

    nst_path = a.nst or nio.session_path(base, "nst", a.elec, variant=a.spk_method)
    if os.path.exists(nst_path):
        st = nio.read_nst(nst_path)
        if (st["nsamp"], st["nchan"]) != (NS, NC):
            raise SystemExit(f"{_LP} · {nst_path}: {st['nsamp']}x{st['nchan']} windows, the .spk has {NS}x{NC}")
        sigma, sigma_d, src = np.asarray(st["sigma"], float), noise_sigma_d(st["cov"], NS, NC), os.path.basename(nst_path)
    else:
        sigma, sigma_d = noise_from_windows(spk, rng=rng)
        src = "window lead-in (no .nst)"
    print(f"{_LP} · {base} elec {a.elec} | noise from {src}: sigma_d {np.round(sigma_d, 1).tolist()}")

    chunk = ((res - res.min()) // (a.chunk_min * 60.0 * SR)).astype(int) if a.chunk_min > 0 else None
    ids, cnt = np.unique(clu[clu > 1], return_counts=True)
    order = np.argsort(clu, kind="stable"); bounds = np.searchsorted(clu[order], ids)
    rows = []
    for u, c0, n in zip(ids, bounds, cnt):
        if n < a.min_n:
            continue
        A, M, ns = cluster_profile(spk, order[c0:c0 + n], peak=PK, sigma=sigma, sigma_d=sigma_d, chunk=chunk,
                                   cap=a.cap, min_score=a.align_min_score, rng=rng)
        rows.append((int(u), int(n), A, M))
    if not rows:
        raise SystemExit(f"{_LP} · no cluster with >= {a.min_n} spikes")
    AA = np.array([r[2] for r in rows]); MM = np.array([r[3] for r in rows])
    if a.line:
        d0, k = (float(v) for v in a.line.split(","))
        how = "fixed"
    else:
        d0, k = fit_clean_line(AA, MM)
        how = f"fitted on {AA.size} channels of {len(rows)} clusters"
    print(f"{_LP} · clean line ({how}): median D^2 = {d0:.3f} + {k:.5f} A^2 (excess {np.sqrt(k):.4f} per SD)")
    Q = excess_ratio(AA, MM, d0, k, min_amp=a.min_amp)
    out = a.out or (nio.session_path(base, "rdr", a.elec, variant=a.clu_method, tag=a.clu_stage) + ".tsv")
    flagged = []
    with open(out, "w") as fh:
        fh.write("cluster\tn\tmax_amp_sd\tratio\tratio_ch\tflag\t"
                 + "\t".join(f"amp{c}" for c in range(NC)) + "\t" + "\t".join(f"rdr{c}" for c in range(NC)) + "\n")
        for (u, n, A, M), q in zip(rows, Q):
            has = np.isfinite(q).any()
            r = float(np.nanmax(q)) if has else float("nan"); rc = int(np.nanargmax(q)) if has else -1
            fl = int(has and r >= a.thr)
            if fl:
                flagged.append((u, r, rc))
            fh.write(f"{u}\t{n}\t{A.max():.2f}\t{r:.3f}\t{rc}\t{fl}\t" + "\t".join(f"{v:.2f}" for v in A) + "\t"
                     + "\t".join(f"{v:.4f}" for v in M) + "\n")
    scored = int(np.isfinite(Q).any(1).sum())
    print(f"{_LP} · {len(rows)} clusters, {scored} with a channel >= {a.min_amp:g} SD; "
          f"{len(flagged)} at ratio >= {a.thr:g} -> {out}")
    for u, r, rc in sorted(flagged, key=lambda x: -x[1])[:40]:
        print(f"    cluster {u:>5}  ratio {r:5.2f}  (ch {rc})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
