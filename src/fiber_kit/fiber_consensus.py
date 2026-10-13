#!/usr/bin/env python3
"""fiber-consensus: resampled-t-SNE consensus over a neighbourhood of similar clusters, and a
QUARANTINE of the spikes whose assignment does not hold up.

The curator's t-SNE lasso works on one embedding.  An embedding is one random draw: which spikes
land next to which depends on the subsample, the initialisation and the perplexity, so a boundary
drawn on one map is partly a boundary of that map.  This stage repeats the embedding many times.

  1. NEIGHBOURHOOD   the units to study together -- given (--units, the clusters a curator would
                     lasso together) or found as average-linkage groups on envelope d'.
  2. RUNS            R t-SNE embeddings (openTSNE), each of a random fraction of the spikes, each
                     from a fresh random initialisation, in the stage's own feature space (.fet).
  3. CONSENSUS       co-neighbour probability P_ij = (runs where j is among i's k nearest 2-D
                     neighbours) / (runs containing both); Louvain communities on that graph.
  4. CELLS           communities are joined into cells by the envelope merge (fiber_envelope:
                     d' with merge inflation as a cannot-link) -- a single cell's spike cloud is a
                     noise ball that t-SNE cuts into several stable-looking islands.
  5. QUARANTINE      a spike is quarantined when
                       * it switches cell: in >= `switch_thr` of the runs containing it, the majority
                         of its 2-D neighbours belongs to ANOTHER cell, or
                       * the envelope flags it as a near collision (time-localised residual bump).
                     Quarantined spikes go to their own cluster, with a per-spike table of reasons
                     (switch rate, collision flag, explaining partner/shift) for review.

What was measured and NOT used, on four g6 neighbourhoods (36-47 min, GT = exp_final): the
across-run variance of 2-D distances to a spike's full-dimensional neighbours.  It is mostly a
density measure -- inside a single cell's noise ball every neighbour is as good as any other, so
core spikes are the UNSTABLE ones and outliers form small stable islands -- and it ran at AUC
0.25-0.57 against misassigned spikes.  The two criteria above ran at 51-79 % (collision flag) and
20-86 % (cell switching) precision against base rates of 8-42 %.
"""
import argparse
import sys
from collections import defaultdict

import numpy as np

try:
    from . import fiber_envelope as fe
    from . import neuro_io as nio
except ImportError:                       # script use
    import fiber_envelope as fe
    import neuro_io as nio


def _need(mod):
    try:
        return __import__(mod)
    except ImportError as e:                      # pragma: no cover - environment dependent
        raise SystemExit(f"fiber-consensus needs '{mod}' (pip install fiber-kit[consensus])") from e


def tsne_runs(X, runs=40, frac=0.7, k=15, perplexity=30.0, seed=0, n_jobs=1, log=None):
    """R subsampled t-SNE embeddings of X (n, d).  Returns a list of (subset indices (m,),
    2-D kNN of each subset spike as GLOBAL indices (m, k))."""
    _need("openTSNE")
    from openTSNE import TSNE
    from sklearn.neighbors import NearestNeighbors
    X = np.asarray(X, float)
    n = len(X); rng = np.random.default_rng(seed); out = []
    m = max(k + 2, int(round(frac * n)))
    for r in range(int(runs)):
        s = np.sort(rng.choice(n, min(m, n), replace=False))
        Y = np.asarray(TSNE(perplexity=min(perplexity, (len(s) - 1) / 3.0), n_jobs=n_jobs,
                            random_state=seed * 1000 + r, initialization="random",
                            verbose=False).fit(X[s]))
        _, ii = NearestNeighbors(n_neighbors=k + 1).fit(Y).kneighbors(Y)
        out.append((s, s[ii[:, 1:]]))
        if log and (r + 1) % 10 == 0:
            log(f"  t-SNE run {r + 1}/{runs}")
    return out


def coneighbour_communities(runs, n, p_min=0.1, resolution=1.0, seed=0):
    """Louvain communities on the co-neighbour probability graph.  Returns labels (n,) >= 0,
    largest community first."""
    nx = _need("networkx")
    C = defaultdict(int)
    inc = np.zeros((len(runs), n), bool)
    for r, (s, gl) in enumerate(runs):
        inc[r, s] = True
        for a, row in zip(s, gl):
            for b in row:
                C[(int(a), int(b))] += 1
    G = nx.Graph(); G.add_nodes_from(range(n))
    if C:
        pairs = np.array(list(C.keys())); cnt = np.array(list(C.values()), float)
        both = (inc[:, pairs[:, 0]] & inc[:, pairs[:, 1]]).sum(0)
        p = cnt / np.maximum(both, 1)
        for (a, b), pp in zip(pairs, p):
            if pp >= p_min:
                w = max(pp, G[a][b]["weight"]) if G.has_edge(a, b) else pp
                G.add_edge(int(a), int(b), weight=float(w))
    comms = nx.community.louvain_communities(G, weight="weight", resolution=resolution, seed=seed)
    lab = np.zeros(n, int)
    for k, cm in enumerate(sorted(comms, key=len, reverse=True)):
        lab[list(cm)] = k
    return lab


def switch_rate(runs, labels):
    """Per spike: fraction of the runs containing it in which the majority label of its 2-D
    neighbours differs from its own."""
    labels = np.asarray(labels)
    off = labels.min()
    sw = np.zeros(len(labels)); cnt = np.zeros(len(labels))
    for s, gl in runs:
        nb = labels[gl] - off
        maj = np.array([np.bincount(row).argmax() for row in nb]) + off
        sw[s] += maj != labels[s]; cnt[s] += 1
    return sw / np.maximum(cnt, 1)


def neighbourhoods(envs, theta=12.0, min_units=2):
    """Average-linkage groups of enveloped units on d' < theta -> list of unit lists."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    units = sorted(envs); n = len(units)
    if n < 2:
        return []
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = fe.separation(envs[units[i]], envs[units[j]])[0]
    g = fcluster(linkage(squareform(D, checks=False), "average"), float(theta), "distance")
    groups = defaultdict(list)
    for u, k in zip(units, g):
        groups[int(k)].append(int(u))
    return [m for m in groups.values() if len(m) >= min_units]


def consensus_neighbourhood(trace, res, clu, units, features, *, transform=None, sr=32552.0,
                            pre=21, post=21, margin=32, maxlag=4, runs=40, frac=0.7, k=15,
                            perplexity=30.0, theta_cell=7.0, switch_thr=0.3, seed=0, n_jobs=1,
                            log=print):
    """Consensus + cells + quarantine for ONE neighbourhood (`units`).

    features: (nSpikes, d) for ALL spikes (the .fet rows); only the neighbourhood's rows are used,
    z-scored per column.  Returns dict(idx, community, cell (unit-like labels), switch, flagged,
    explained, partner, shift, quarantine (bool), reason (list of str))."""
    clu = np.asarray(clu)
    idx = np.flatnonzero(np.isin(clu, list(units)))
    X = np.asarray(features, float)[idx]
    X = (X - X.mean(0)) / X.std(0).clip(1e-9)
    rr = tsne_runs(X, runs=runs, frac=frac, k=k, perplexity=perplexity, seed=seed, n_jobs=n_jobs, log=log)
    com = coneighbour_communities(rr, idx.size, seed=seed)
    lab = np.zeros(len(clu), int); lab[idx] = com + 2
    envs, table, noise = fe.run(trace, res, lab, pre=pre, post=post, margin=margin, maxlag=maxlag,
                                transform=transform, min_n=20, sr=sr, log=lambda s: None)
    new, _ = fe.merge_units(trace, res, lab, envs, noise, theta=theta_cell, pre=pre, post=post,
                            margin=margin, maxlag=maxlag, transform=transform, sr=sr, log=lambda s: None)
    cell = new[idx]
    sw = switch_rate(rr, cell)
    pos = {int(i): j for j, i in enumerate(table["idx"])}
    take = lambda key, default: np.array([table[key][pos[i]] if i in pos else default for i in idx])  # noqa: E731
    flagged = take("flagged", 0).astype(bool); explained = take("explained", 0).astype(bool)
    partner = take("partner", -1).astype(int); shift = take("shift", 0).astype(int)
    q = (sw >= switch_thr) | flagged
    reason = [",".join(r for r, v in (("switch", s_ >= switch_thr), ("collision", f_)) if v)
              for s_, f_ in zip(sw, flagged)]
    log(f"neighbourhood {sorted(units)}: {idx.size} spikes -> {np.unique(com).size} communities -> "
        f"{np.unique(cell).size} cells; quarantined {q.sum()} ({q.mean():.1%}: {(sw >= switch_thr).sum()} "
        f"switching, {flagged.sum()} collision-flagged)")
    return dict(idx=idx, community=com, cell=cell, switch=sw, flagged=flagged, explained=explained,
                partner=partner, shift=shift, quarantine=q, reason=reason)


def write_quarantine_table(path, results, res_abs, clu, out_labels):
    with open(path, "w") as f:
        f.write("spike\ttime\tunit\tcell\tout\tswitch\tcollision\texplained\tpartner\tshift\treason\n")
        for r in results:
            for j, i in enumerate(r["idx"]):
                if not r["quarantine"][j]:
                    continue
                f.write(f"{i}\t{int(res_abs[i])}\t{int(clu[i])}\t{int(r['cell'][j])}\t{int(out_labels[i])}\t"
                        f"{r['switch'][j]:.3f}\t{int(r['flagged'][j])}\t{int(r['explained'][j])}\t"
                        f"{int(r['partner'][j])}\t{int(r['shift'][j])}\t{r['reason'][j]}\n")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Resampled-t-SNE consensus over similar clusters + quarantine of unstable / colliding spikes.")
    ap.add_argument("base"); ap.add_argument("elec", type=int)
    ap.add_argument("--signal", required=True, help="continuous filtered int16 file (e.g. BASE.fil)")
    ap.add_argument("--total-channels", type=int, help="channels in the signal file (default: session yaml)")
    ap.add_argument("--channels", help="group channels, e.g. 40-47 (default: session yaml)")
    ap.add_argument("--signal-start", type=int, default=0, help="absolute sample of the signal's first row")
    ap.add_argument("--space", default=None, help="waveform space for the envelope (default: --clu-method's)")
    ap.add_argument("--sdiff-pairs", default=None, help="sdiffPairs for a _C space (default: session yaml)")
    ap.add_argument("--res-variant", default="stderiv", help="variant of the spike-time .res")
    nio.add_clu_args(ap, method_default="stderiv_C5_D34", stage_default="",
                     stage_dest="clu_stage", stage_alias=None)
    ap.add_argument("--fet-variant", default=None, help="variant of the .fet to embed (default: --clu-method)")
    ap.add_argument("--units", default=None,
                    help="comma-separated units to study together (repeat with ';' for several groups); "
                         "default: every envelope d' neighbourhood")
    ap.add_argument("--neighbourhood-theta", type=float, default=12.0, help="d' cut for automatic neighbourhoods")
    ap.add_argument("--runs", type=int, default=40); ap.add_argument("--frac", type=float, default=0.7)
    ap.add_argument("--k", type=int, default=15, help="2-D neighbours per spike")
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--theta-cell", type=float, default=7.0, help="envelope d' cut joining communities into cells")
    ap.add_argument("--switch-thr", type=float, default=0.3, help="cell-switch rate that quarantines a spike")
    ap.add_argument("--sr", type=float, default=None)
    ap.add_argument("--jobs", type=int, default=1, help="openTSNE threads")
    ap.add_argument("--out-tag", default=None, help="stage tag of the output .clu (default: <clu-stage>.consensus)")
    a = ap.parse_args(argv)
    try:
        from . import session_yaml as sy
    except ImportError:
        import session_yaml as sy
    info = sy.load_session(a.base, a.elec) if sy.find_session_yaml(a.base) is not None else {}
    if a.channels:
        ch = a.channels
        chans = (list(range(int(ch.split("-")[0]), int(ch.split("-")[1]) + 1))
                 if "-" in ch and "," not in ch else [int(x) for x in ch.split(",")])
    else:
        chans = info.get("channels")
    ntot = a.total_channels or info.get("ntotal")
    if not chans or not ntot:
        ap.error("--channels/--total-channels not given and no session yaml to read them from")
    if a.space is None:
        sp = nio.parse_variant_token(a.clu_method)
        a.space = sp.family if sp.kind is None else f"{sp.family}_{sp.kind}{sp.order}"
    transform = fe.space_transform(a.space, a.sdiff_pairs or info.get("sdiff_pairs"))
    sr = a.sr or info.get("sr") or 32552.0
    sig = nio.open_signal(a.signal, ntot)

    class _View:
        shape = (sig.shape[0], len(chans))
        def __getitem__(self, rows):
            return np.asarray(sig[rows][:, chans], np.float32)
    view = _View()
    log = lambda s: print(s, file=sys.stderr, flush=True)  # noqa: E731
    res_abs = nio.read_res_at(a.base, a.elec, variant=a.res_variant)
    _, clu = nio.resolve_clu(a, a.base, a.elec, n_spikes=res_abs.size, stage_dest="clu_stage")
    fet = nio.read_fet_at(a.base, a.elec, variant=a.fet_variant or a.clu_method)
    if fet.values.shape[0] != res_abs.size:
        raise SystemExit(f".fet has {fet.values.shape[0]} rows, .res {res_abs.size}")
    rel = res_abs - a.signal_start
    inwin = (rel >= 0) & (rel < sig.shape[0])
    lab = np.where(inwin, clu, 0); rel0 = np.where(inwin, rel, 0)
    if a.units:
        groups = [[int(x) for x in g.split(",") if x.strip()] for g in a.units.split(";")]
    else:
        envs, _, _ = fe.run(view, rel0, lab, transform=transform, pre=21, post=21, min_n=30, sr=sr,
                            partners=False, log=lambda s: None)
        groups = neighbourhoods(envs, a.neighbourhood_theta)
    out = clu.copy(); nxt = int(clu.max()) + 1; results = []
    for units in groups:
        r = consensus_neighbourhood(view, rel0, lab, units, fet.values, transform=transform, sr=sr,
                                    runs=a.runs, frac=a.frac, k=a.k, perplexity=a.perplexity,
                                    theta_cell=a.theta_cell, switch_thr=a.switch_thr, n_jobs=a.jobs, log=log)
        cmap = {}
        for c in np.unique(r["cell"]):
            cmap[int(c)] = nxt; nxt += 1
        out[r["idx"]] = [cmap[int(c)] for c in r["cell"]]
        if r["quarantine"].any():                 # one quarantine cluster per neighbourhood
            out[r["idx"][r["quarantine"]]] = nxt; nxt += 1
        results.append(r)
    tag = a.out_tag or (f"{a.clu_stage}.consensus" if a.clu_stage else "consensus")
    p_clu = nio.write_clu(a.base, a.elec, out, variant=a.clu_method, tag=tag)
    p_q = write_quarantine_table(f"{a.base}.qrn.{a.elec}.{tag}", results, res_abs, clu, out)
    nq = int(sum(r["quarantine"].sum() for r in results))
    print(f"{len(groups)} neighbourhood(s); {nq} spikes quarantined; wrote {p_clu} and {p_q}")


if __name__ == "__main__":
    main()
