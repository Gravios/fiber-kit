#!/usr/bin/env python3
"""chain_consensus.py -- systematically chain the HIGH-SNR clusters of a group, peeling each chain out
of the pool after it is found, then repeat the whole peel over many trials and track how consistently
each cluster lands in the same chain.

One trial (a peel): seed from the highest-SNR unassigned cluster, chase it across chunks with
piece_interneurons.chase_from (primary-channel cosine, drift-following), record the chain, REMOVE its
members from the pool, and repeat until the pool is empty.  This partitions the high-SNR clusters into
chains greedily, strongest cell first.

Across trials the per-cluster template is re-drawn from a fresh spike subsample (a bootstrap), so links
that sit near the cosine gate flip between trials while well-separated cells recur.  Tracking the
co-membership (fraction of trials two clusters share a chain) turns the greedy, seed-order-dependent
chaining into a stability map: high co-membership = a robust cell, singletons = ambiguous fragments.

Usage:
    python3 tools/chain_consensus.py <session> <group> [--snr-thr 8] [--trials 15] \
        [--variant stderiv] [--stage fiber_session] [--spk standard|stderiv] \
        [--gap-min 60] [--cos-thr 0.92] [--min-n 200] [--cap 500] [--comemb-thr 0.5] \
        [--tsv memberships.tsv] [--out consensus.png]
"""
import argparse
import os
import numpy as np

try:
    from fiber_kit import fiber_lib as fl, session_yaml as sy, neuro_io as nio
except ImportError:
    import fiber_lib as fl, session_yaml as sy, neuro_io as nio

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import piece_interneurons as pi                      # reuse chase_from + _pcos


def cluster_snr(spk, idx, sr, cap, rng):
    """Aligned template + dominant-channel SNR (dom p2p / robust single-spike residual noise)."""
    s = idx if len(idx) <= cap else rng.choice(idx, cap, replace=False)
    w = np.asarray(spk[np.sort(s)], float)
    t = np.median(fl.align_xcorr(w, ref="median", iters=4), axis=0)
    amp = np.ptp(t, axis=0); dom = int(np.argmax(amp))
    pk = int(np.argmin(t[:, dom]))
    resid = w[:, pk, dom] - np.median(w[:, pk, dom])
    noise = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-9
    return t, amp, dom, float(amp[dom] / noise)


def build_pool(spk, res, ids, *, sr, min_n, snr_thr, cap, seed=0):
    """High-SNR cluster pool: {clu -> spike idx}, {clu -> snr}, and each cluster's time centroid."""
    rng = np.random.default_rng(seed)
    tmin = (res - res.min()) / sr / 60.0
    uniq, cnt = np.unique(ids, return_counts=True)
    pool, snr, idxmap, tc = [], {}, {}, {}
    for u, c in zip(uniq, cnt):
        if u < 2 or c < min_n:
            continue
        idx = np.flatnonzero(ids == u)
        _, _, _, s = cluster_snr(spk, idx, sr, cap, rng)
        if s >= snr_thr:
            pool.append(int(u)); snr[int(u)] = s; idxmap[int(u)] = idx; tc[int(u)] = float(np.median(tmin[idx]))
    return pool, snr, idxmap, tc


def trial_templates(pool, idxmap, tc, spk, sr, cap, rng_seed):
    """Per-cluster fragment dicts (chase_from format) with templates drawn from a fresh subsample."""
    rng = np.random.default_rng(rng_seed)
    F = {}
    for u in pool:
        t, amp, dom, _ = cluster_snr(spk, idxmap[u], sr, cap, rng)
        F[u] = dict(clu=u, n=len(idxmap[u]), t=t, amp=amp, dom=dom, tmid=tc[u])
    return F


def peel(F, pool, snr, *, gap_min, cos_thr, amp_ratio, prim_frac):
    """Greedily peel chains: seed = highest-SNR unassigned, chase, remove, repeat."""
    avail = set(pool); chains = []
    while avail:
        seed = max(avail, key=lambda u: snr[u])
        sub = sorted((F[u] for u in avail), key=lambda f: f["tmid"])
        si = next(k for k, f in enumerate(sub) if f["clu"] == seed)
        order = pi.chase_from(sub, si, gap_min=gap_min, cos_thr=cos_thr, amp_ratio=amp_ratio, prim_frac=prim_frac)
        ch = [sub[i]["clu"] for i in order]
        if len(ch) >= 2:
            chains.append(ch); avail -= set(ch)
        else:
            avail.discard(seed)
    return chains


def consensus(memb, thr):
    """Co-membership matrix C[i,j] = fraction of trials clusters i,j share a chain, and the
    connected components of (C >= thr) as consensus groups."""
    nt, n = memb.shape
    C = np.zeros((n, n))
    for tr in range(nt):
        m = memb[tr]
        for i in range(n):
            if m[i] < 0:
                continue
            C[i] += (m == m[i]) & (m >= 0)
    C /= nt
    np.fill_diagonal(C, 1.0)
    # connected components of the thresholded co-membership graph (no scipy dependency)
    A = C >= thr
    comp = -np.ones(n, dtype=int); k = 0
    for s in range(n):
        if comp[s] >= 0:
            continue
        stack = [s]; comp[s] = k
        while stack:
            u = stack.pop()
            for v in np.flatnonzero(A[u]):
                if comp[v] < 0:
                    comp[v] = k; stack.append(v)
        k += 1
    return C, comp


def report_figure(C, comp, pool, snr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = np.lexsort((-np.array([snr[u] for u in pool]), comp))   # group, then SNR
    Cs = C[np.ix_(order, order)]
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(12, 5.5), gridspec_kw={"width_ratios": [4, 1]},
                                  constrained_layout=True)
    im = ax.imshow(Cs, cmap="magma", vmin=0, vmax=1)
    ax.set_title("co-membership across trials (clusters ordered by consensus group)")
    ax.set_xlabel("cluster"); ax.set_ylabel("cluster")
    fig.colorbar(im, ax=ax, fraction=0.046, label="fraction of trials in the same chain")
    stab = C.copy(); np.fill_diagonal(stab, np.nan)
    per = np.array([np.nanmax(stab[i]) for i in range(len(pool))])[order]
    axb.barh(np.arange(len(pool)), per, color="#2a9d8f"); axb.set_ylim(-0.5, len(pool) - 0.5)
    axb.invert_yaxis(); axb.set_xlim(0, 1); axb.set_title("peak co-membership\n(per cluster)")
    axb.set_yticks([])
    return fig, order


def main():
    ap = argparse.ArgumentParser(prog="chain_consensus", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sy.add_session_args(ap)
    ap.add_argument("--variant", default="stderiv"); ap.add_argument("--stage", default="fiber_session")
    ap.add_argument("--spk", choices=["standard", "stderiv"], default="standard",
                    help="waveform space for the templates (default standard = raw)")
    ap.add_argument("--snr-thr", type=float, default=8.0, help="dom-channel SNR floor for the pool (default 8)")
    ap.add_argument("--min-n", type=int, default=200); ap.add_argument("--cap", type=int, default=500)
    ap.add_argument("--trials", type=int, default=15, help="peel repeats with re-drawn templates (default 15)")
    ap.add_argument("--gap-min", type=float, default=60.0); ap.add_argument("--cos-thr", type=float, default=0.92)
    ap.add_argument("--amp-ratio", type=float, default=2.2); ap.add_argument("--prim-frac", type=float, default=0.3)
    ap.add_argument("--comemb-thr", type=float, default=0.5, help="co-membership for a consensus group (default 0.5)")
    ap.add_argument("--tsv", default=None, help="write per-cluster membership (clu, snr, group, stability) to TSV")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = sy.resolve_session_params(a.session, a.group, channels=a.channels, ntotal=a.ntotal,
                                    nchan=a.nchan, nsamp=a.nsamp, sr=a.sr)
    base = cfg["base"]; elec = a.group; nsamp = cfg["nsamp"]; nchan = cfg["nchan"]; sr = cfg["sr"]
    res = nio.read_res(base, elec)
    if a.spk == "stderiv":
        spk, _ = nio.open_spk(base, elec, nsamp, nchan, prefer=nio.prefer_derived())
    else:
        spk, _ = nio.open_spk_raw(base, elec, nsamp, nchan)
    _, ids = nio.read_clu_at(base, elec, variant=a.variant, tag=a.stage, n_spikes=len(res))

    pool, snr, idxmap, tc = build_pool(spk, res, ids, sr=sr, min_n=a.min_n, snr_thr=a.snr_thr, cap=a.cap)
    if len(pool) < 2:
        raise SystemExit(f"[consensus] high-SNR pool has {len(pool)} clusters (raise --min-n / lower --snr-thr)")
    print(f"[consensus] {os.path.basename(base)} elec {elec}: {len(pool)} clusters SNR>={a.snr_thr:g}, "
          f"{a.trials} trials ({a.spk} templates)")
    pidx = {u: i for i, u in enumerate(pool)}
    memb = np.full((a.trials, len(pool)), -1, dtype=int)
    nch = []
    for tr in range(a.trials):
        F = trial_templates(pool, idxmap, tc, spk, sr, a.cap, rng_seed=1000 + tr)
        chains = peel(F, pool, snr, gap_min=a.gap_min, cos_thr=a.cos_thr, amp_ratio=a.amp_ratio, prim_frac=a.prim_frac)
        nch.append(len(chains))
        for ci, c in enumerate(chains):
            for u in c:
                memb[tr, pidx[u]] = ci
    C, comp = consensus(memb, a.comemb_thr)

    from collections import Counter
    sizes = Counter(comp)
    groups = sorted((g for g, s in sizes.items() if s >= 2), key=lambda g: -sizes[g])
    stab = C.copy(); np.fill_diagonal(stab, np.nan)
    peakco = np.array([np.nanmax(stab[i]) for i in range(len(pool))])
    print(f"  chains/trial: {nch} (mean {np.mean(nch):.1f})")
    print(f"  consensus groups (co-membership >= {a.comemb_thr:g}, size >= 2): {len(groups)}; "
          f"singleton/ambiguous clusters: {sum(1 for g in comp if sizes[g] < 2)}/{len(pool)}")
    print(f"  {'grp':>4} {'size':>4} {'meanCo':>6} {'SNRrange':>12}  members")
    for g in groups:
        mem = sorted(pool[i] for i in range(len(pool)) if comp[i] == g)
        si = [pidx[u] for u in mem]; wc = C[np.ix_(si, si)][np.triu_indices(len(si), 1)]
        srng = f"{min(snr[u] for u in mem):.1f}-{max(snr[u] for u in mem):.1f}"
        print(f"  {g:>4} {len(mem):>4} {wc.mean():>6.2f} {srng:>12}  {mem[:12]}{'...' if len(mem) > 12 else ''}")

    if a.tsv:
        with open(a.tsv, "w") as fh:
            fh.write("clu\tsnr\tn\tconsensus_group\tgroup_size\tpeak_comembership\ttrials_chained\n")
            for i, u in enumerate(pool):
                tc_n = int((memb[:, i] >= 0).sum())
                gid = int(comp[i]); fh.write(f"{u}\t{snr[u]:.2f}\t{len(idxmap[u])}\t"
                                             f"{gid if sizes[gid] >= 2 else -1}\t{sizes[gid]}\t{peakco[i]:.3f}\t{tc_n}\n")
        print(f"  wrote {a.tsv}")

    fig, _ = report_figure(C, comp, pool, snr)
    out = a.out or f"{base}.consensus.{elec}.png"
    fig.savefig(out, dpi=120); print(f"  wrote {out}")


if __name__ == "__main__":
    main()
