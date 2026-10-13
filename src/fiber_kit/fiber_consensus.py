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
  5. QUARANTINE      with the envelope refitted ON THE CELLS, a spike is quarantined when
                       * collision: the envelope flags a near collision (time-localised residual
                         bump), or
                       * perimeter: its average 2-D radial rank inside its cell is >= 0.8 AND its
                         residual energy or bump score is >= 2 robust z above the cell's -- the rim
                         of a cell's cloud is where its collisions and foreign spikes sit, but also
                         its legitimate amplitude extremes, hence the waveform condition;
                       * (optional) switch: in >= 30 % of its runs the majority of its 2-D
                         neighbours belongs to another cell.
  6. CATEGORIES      the quarantine is sorted by t-SNE watershed basins: a few fresh embeddings of
                     the whole neighbourhood, a KDE on a grid, watershed basins from the density
                     maxima, a vote over the runs.  A quarantined spike is
                       * island      in a basin made mostly (>= 50 %) of quarantined spikes -- a
                                     collision / noise group of its own;
                       * own-rim     in a basin owned by its own cell (majority of the basin's kept
                                     spikes) -- an extreme of that cell;
                       * other-cell  in a basin owned by another cell -- a reassignment candidate
                                     (the destination is recorded, not applied).
                     Quarantined spikes go to TWO clusters per neighbourhood (island; own-rim +
                     other-cell), with a per-spike table of reasons and categories for review.
                     --ws-runs 0 skips the categories and writes one quarantine cluster.

Measured on four g6 neighbourhoods (36-47 min, GT = exp_final), against base rates of 8-42 %
misassigned: collision | perimeter quarantines 6.7-7.8 % of spikes at 0.40-0.85 precision and lifts
cell purity by 0.02-0.05; the perimeter rule adds 24-47 spikes per neighbourhood the collision flag
misses.  Cell switching runs at 0.13-0.22 precision where cells are close, so it is opt-in.  Of the
quarantine, 20-38 % falls in islands and 66-98 % of island spikes are foreign or noise; own-rim
spikes in a clean cell are mostly the cell's own (52-73 %).  A second round (quarantine held out,
cells rebuilt, quarantined again) moved kept-spike purity by -0.002..+0.004 -- not worth it.  NOT
used: the across-run VARIANCE of 2-D distances -- a density measure (inside one cell's noise ball
every neighbour is as good as any other, so core spikes are the unstable ones), AUC 0.25-0.57.
"""
import argparse
import os
import sys
from collections import Counter, defaultdict

import numpy as np

try:
    from . import fiber_envelope as fe
    from . import neuro_io as nio
except ImportError:                       # script use
    import fiber_envelope as fe
    import neuro_io as nio
try:
    from . import config as cfgmod
except ImportError:                       # script use
    import config as cfgmod


def _need(mod):
    try:
        return __import__(mod)
    except ImportError as e:                      # pragma: no cover - environment dependent
        raise SystemExit(f"fiber-consensus needs '{mod}' (pip install fiber-kit[consensus])") from e


def _tsne_fit(X, perplexity, seed, n_jobs=1, exaggeration=None):
    from openTSNE import TSNE
    kw = {} if exaggeration is None else dict(exaggeration=exaggeration)
    return np.asarray(TSNE(perplexity=min(perplexity, (len(X) - 1) / 3.0), n_jobs=n_jobs, random_state=seed,
                           initialization="random", verbose=False, **kw).fit(X))


def _one_run(X, s, perplexity, seed, k, n_jobs):
    """One subsampled run (a top-level function so a worker process can run it)."""
    from sklearn.neighbors import NearestNeighbors
    Y = _tsne_fit(X[s], perplexity, seed, n_jobs)
    _, ii = NearestNeighbors(n_neighbors=k + 1).fit(Y).kneighbors(Y)
    return s, s[ii[:, 1:]], Y.astype(np.float32)


def _pool(workers):
    """Process pool for independent t-SNE fits, or None for serial.  workers: 1 = serial, 0 = every
    core.  spawn, not fork: openTSNE / BLAS threads in the parent make a forked child unsafe."""
    import multiprocessing as mp
    import os as _os
    from concurrent.futures import ProcessPoolExecutor
    w = (_os.cpu_count() or 1) if not workers else int(workers)
    if w <= 1:
        return None
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        _os.environ.setdefault(v, "1")             # inherited by the spawned workers: no oversubscription
    return ProcessPoolExecutor(w, mp_context=mp.get_context("spawn"))


def tsne_runs(X, runs=40, frac=0.7, k=15, perplexity=30.0, seed=0, n_jobs=1, log=None, workers=1):
    """R subsampled t-SNE embeddings of X (n, d).  Returns a list of (subset indices (m,),
    2-D kNN of each subset spike as GLOBAL indices (m, k), 2-D coordinates (m, 2)).
    workers > 1 (0 = every core) runs the embeddings in parallel processes; the subsets and seeds are
    drawn up front, so the result does not depend on the worker count."""
    _need("openTSNE")
    X = np.asarray(X, float)
    n = len(X); rng = np.random.default_rng(seed)
    m = max(k + 2, int(round(frac * n)))
    subsets = [np.sort(rng.choice(n, min(m, n), replace=False)) for _ in range(int(runs))]
    args = [(X, s, perplexity, seed * 1000 + r, k, n_jobs) for r, s in enumerate(subsets)]
    pool = _pool(workers)
    if pool is None:
        out = []
        for r, a_ in enumerate(args):
            out.append(_one_run(*a_))
            if log and (r + 1) % 10 == 0:
                log(f"  t-SNE run {r + 1}/{runs}")
        return out
    with pool:
        futs = [pool.submit(_one_run, *a_) for a_ in args]
        out = []
        for r, f in enumerate(futs):
            out.append(f.result())
            if log and (r + 1) % 10 == 0:
                log(f"  t-SNE run {r + 1}/{runs}")
    return out


def coneighbour_communities(runs, n, p_min=0.1, resolution=1.0, seed=0):
    """Louvain communities on the co-neighbour probability graph.  Returns labels (n,) >= 0,
    largest community first."""
    nx = _need("networkx")
    C = defaultdict(int)
    inc = np.zeros((len(runs), n), bool)
    for r, (s, gl, *_) in enumerate(runs):
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
    for s, gl, *_ in runs:
        nb = labels[gl] - off
        maj = np.array([np.bincount(row).argmax() for row in nb]) + off
        sw[s] += maj != labels[s]; cnt[s] += 1
    return sw / np.maximum(cnt, 1)


def perimeter_rank(runs, labels, min_n=50):
    """Per spike: its radial rank inside its own label's 2-D cloud (distance from the cloud's median,
    0 = centre, 1 = outermost), averaged over the runs that contained it.  Labels with fewer than
    `min_n` spikes in a run are skipped (NaN if never ranked).  Unlike the run-to-run VARIANCE of
    2-D distances (a density measure), the average position is reproducible, and on g6 the outer
    ring of a cell is where its collisions and foreign spikes sit."""
    labels = np.asarray(labels)
    acc = np.zeros(len(labels)); cnt = np.zeros(len(labels))
    for s, _, Y in runs:
        ls = labels[s]
        for c in np.unique(ls):
            m = np.flatnonzero(ls == c)
            if m.size < min_n:
                continue
            d = np.linalg.norm(Y[m] - np.median(Y[m], 0), axis=1)
            acc[s[m]] += (np.argsort(np.argsort(d)) + 0.5) / m.size
            cnt[s[m]] += 1
    out = np.full(len(labels), np.nan)
    ok = cnt > 0
    out[ok] = acc[ok] / cnt[ok]
    return out


def neighbourhoods(envs, theta=12.0, min_units=2, max_spikes=None, step=0.8):
    """Average-linkage groups of enveloped units on d' < theta -> list of unit lists.

    max_spikes: a group holding more spikes is re-cut on its own linkage at theta*step, theta*step^2,
    ... until every piece fits (a single unit larger than the cap stays a group of one and is
    dropped by min_units).  Off by default: the large group d' 12 forms is a set of genuinely
    confusable cells, and splitting it costs completeness (g6, 11.7 k spikes: uncapped 0.81 vs 0.58
    with a 4000 cap at equal purity).  Use it only to bound run time."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    units = sorted(envs); n = len(units)
    if n < 2:
        return []
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = fe.separation(envs[units[i]], envs[units[j]])[0]
    size = np.array([envs[u].n for u in units])

    def cut(ii, th):
        if ii.size < 2:
            return [ii]
        g = fcluster(linkage(squareform(D[np.ix_(ii, ii)], checks=False), "average"), float(th), "distance")
        out = []
        for k in np.unique(g):
            jj = ii[g == k]
            if max_spikes and size[jj].sum() > max_spikes and jj.size > 1 and th > 0.5:
                out += cut(jj, th * step)
            else:
                out.append(jj)
        return out

    groups = cut(np.arange(n), theta)
    return [[int(units[i]) for i in jj] for jj in groups if jj.size >= min_units]


def _robust_z(v, groups):
    v = np.asarray(v, float); z = np.full(v.size, np.nan)
    for g in np.unique(groups):
        m = groups == g
        med = np.nanmedian(v[m]); mad = 1.4826 * np.nanmedian(np.abs(v[m] - med))
        z[m] = (v[m] - med) / max(mad, 1e-12)
    return z


WS_CATEGORIES = ("own-rim", "other-cell", "island")


def basin_labels(Y, grid=256):
    """Watershed basin of every 2-D point: histogram on a grid x grid lattice, Gaussian KDE
    (sigma = grid/100), markers at the density maxima (min distance grid/40, >= 2 % of the peak),
    watershed of -density inside density > 0.2 % of the peak.  0 = outside every basin."""
    from scipy.ndimage import gaussian_filter
    _need("skimage")
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
    Y = np.asarray(Y, float)
    lo = Y.min(0); hi = Y.max(0)
    ij = np.clip(((Y - lo) / (hi - lo + 1e-9) * (grid - 1)).astype(int), 0, grid - 1)
    H = np.zeros((grid, grid)); np.add.at(H, (ij[:, 0], ij[:, 1]), 1)
    dens = gaussian_filter(H, sigma=grid / 100)
    pk = peak_local_max(dens, min_distance=max(grid // 40, 1), threshold_rel=0.02)
    mk = np.zeros(dens.shape, int); mk[tuple(pk.T)] = np.arange(1, len(pk) + 1)
    floor = dens.max() * 0.002
    B = watershed(-dens, mk, mask=dens > floor)
    return B[ij[:, 0], ij[:, 1]]


def categorise_basins(basin, cell, quarantine, island_frac=0.5):
    """One map's verdict for every quarantined spike: (code, destination) with code an index into
    WS_CATEGORIES.  A basin's owner is the cell holding most of its NON-quarantined spikes; a basin
    with >= island_frac quarantined spikes, no kept spikes, or label 0 is an island."""
    basin = np.asarray(basin); cell = np.asarray(cell); q = np.asarray(quarantine, bool)
    owner = {}; qfrac = {}
    for b in np.unique(basin):
        m = basin == b; kept = cell[m & ~q]
        qfrac[b] = q[m].mean()
        owner[b] = Counter(kept.tolist()).most_common(1)[0][0] if kept.size else None
    qi = np.flatnonzero(q)
    code = np.empty(qi.size, int); dest = np.full(qi.size, -1, int)
    for k, i in enumerate(qi):
        b = basin[i]
        if b == 0 or qfrac[b] >= island_frac or owner[b] is None:
            code[k] = 2
        elif owner[b] == cell[i]:
            code[k] = 0; dest[k] = cell[i]
        else:
            code[k] = 1; dest[k] = owner[b]
    return code, dest


def watershed_categories(X, cell, quarantine, *, runs=5, grid=256, island_frac=0.5, perplexity=30.0,
                         seed=0, n_jobs=1, maps=None, log=None, workers=1):
    """Watershed category of every quarantined spike, voted over `runs` full t-SNE maps of X (or the
    given `maps`, a list of (n, 2) arrays).  Returns dict(category: (n,) str, '' where not quarantined;
    dest: (n,) destination cell (own cell for own-rim, the basin owner for other-cell, -1 otherwise);
    agree: (n,) fraction of runs voting for the winning category, nan where not quarantined)."""
    cell = np.asarray(cell); q = np.asarray(quarantine, bool); n = cell.size
    if maps is None:
        _need("openTSNE")
        X = np.asarray(X, float)
        pool = _pool(workers)
        if pool is None:
            maps = [_tsne_fit(X, perplexity, seed + 500 + r, n_jobs) for r in range(runs)]
        else:
            with pool:
                maps = list(pool.map(_tsne_fit, *zip(*[(X, perplexity, seed + 500 + r, n_jobs)
                                                       for r in range(runs)])))
        if log:
            log(f"  {runs} watershed maps")
    qi = np.flatnonzero(q)
    codes = np.empty((len(maps), qi.size), int); dests = np.empty((len(maps), qi.size), int)
    for r, Y in enumerate(maps):
        codes[r], dests[r] = categorise_basins(basin_labels(Y, grid), cell, q, island_frac)
    category = np.full(n, "", dtype=object); dest = np.full(n, -1, int); agree = np.full(n, np.nan)
    for k, i in enumerate(qi):
        c, cnt = Counter(codes[:, k].tolist()).most_common(1)[0]
        category[i] = WS_CATEGORIES[c]; agree[i] = cnt / len(maps)
        dest[i] = Counter(dests[codes[:, k] == c, k].tolist()).most_common(1)[0][0]
    return dict(category=category, dest=dest, agree=agree)


QUARANTINE_RULES = ("switch", "collision", "perimeter")


def consensus_neighbourhood(trace, res, clu, units, features, *, transform=None, sr=32552.0,
                            pre=21, post=21, margin=32, maxlag=4, runs=40, frac=0.7, k=15,
                            perplexity=30.0, theta_cell=7.0, switch_thr=0.3, perim_thr=0.8,
                            dev_z=2.0, rules=("collision", "perimeter"), ws_runs=5, ws_grid=256,
                            ws_island=0.5, seed=0, n_jobs=1, workers=1, log=print):
    """Consensus + cells + quarantine for ONE neighbourhood (`units`).

    features: (nSpikes, d) for ALL spikes (the .fet rows); only the neighbourhood's rows are used,
    z-scored per column.  After the cells are formed the envelope is refitted ON THE CELLS, so the
    collision flag, bump score and residual energy are relative to the cell a spike ends up in.

    Quarantine criteria (`rules`, any of):
      switch     cell-switch rate >= switch_thr
      collision  the envelope's near-collision flag
      perimeter  average 2-D radial rank >= perim_thr AND the spike's residual energy or bump score
                 is >= dev_z robust z above its cell's (the outer ring alone also holds a cell's
                 legitimate amplitude extremes)
    With ws_runs > 0 the quarantine is then categorised by watershed_categories (island / own-rim /
    other-cell) on ws_runs fresh full-neighbourhood maps.
    Returns dict(idx, community, cell, switch, perimeter, resid_z, bump_z, flagged, explained, partner,
    shift, quarantine (bool), reason (list of str), criteria, category, ws_dest, ws_agree)."""
    clu = np.asarray(clu)
    idx = np.flatnonzero(np.isin(clu, list(units)))
    X = np.asarray(features, float)[idx]
    X = (X - X.mean(0)) / X.std(0).clip(1e-9)
    rr = tsne_runs(X, runs=runs, frac=frac, k=k, perplexity=perplexity, seed=seed, n_jobs=n_jobs, log=log,
                   workers=workers)
    com = coneighbour_communities(rr, idx.size, seed=seed)
    lab = np.zeros(len(clu), int); lab[idx] = com + 2
    envs, _, noise = fe.run(trace, res, lab, pre=pre, post=post, margin=margin, maxlag=maxlag,
                            transform=transform, min_n=20, sr=sr, partners=False, log=lambda s: None)
    new, _ = fe.merge_units(trace, res, lab, envs, noise, theta=theta_cell, pre=pre, post=post,
                            margin=margin, maxlag=maxlag, transform=transform, sr=sr, log=lambda s: None)
    cell = new[idx]
    # the envelope on the CELLS: collision flags, bump scores and residual energy relative to them
    lab_c = np.zeros(len(clu), int); lab_c[idx] = cell
    cenvs, table, _ = fe.run(trace, res, lab_c, pre=pre, post=post, margin=margin, maxlag=maxlag,
                             transform=transform, min_n=20, sr=sr, log=lambda s: None)
    resid = np.full(idx.size, np.nan); bump = np.full(idx.size, np.nan)
    where = {int(i): j for j, i in enumerate(idx)}
    for e in cenvs.values():
        jj = np.array([where[int(i)] for i in e.meta["idx"]])
        resid[jj] = e.resid_energy; bump[jj] = e.score
    resid_z = _robust_z(resid, cell); bump_z = _robust_z(bump, cell)
    sw = switch_rate(rr, cell)
    per = perimeter_rank(rr, cell)
    pos = {int(i): j for j, i in enumerate(table["idx"])}
    take = lambda key, default: np.array([table[key][pos[i]] if i in pos else default for i in idx])  # noqa: E731
    flagged = take("flagged", 0).astype(bool); explained = take("explained", 0).astype(bool)
    partner = take("partner", -1).astype(int); shift = take("shift", 0).astype(int)
    crit = dict(switch=sw >= switch_thr, collision=flagged,
                perimeter=(np.nan_to_num(per) >= perim_thr)
                & ((np.nan_to_num(resid_z) >= dev_z) | (np.nan_to_num(bump_z) >= dev_z)))
    q = np.zeros(idx.size, bool)
    for r_ in rules:
        q |= crit[r_]
    reason = [",".join(r_ for r_ in rules if crit[r_][j]) for j in range(idx.size)]
    log(f"neighbourhood {sorted(units)}: {idx.size} spikes -> {np.unique(com).size} communities -> "
        f"{np.unique(cell).size} cells; quarantined {q.sum()} ({q.mean():.1%}: "
        + ", ".join(f"{crit[r_].sum()} {r_}" for r_ in rules) + ")")
    if ws_runs > 0 and q.any():
        ws = watershed_categories(X, cell, q, runs=ws_runs, grid=ws_grid, island_frac=ws_island,
                                  perplexity=perplexity, seed=seed, n_jobs=n_jobs, workers=workers)
        cat = ws["category"]
        log("  categories: " + ", ".join(f"{(cat == c).sum()} {c}" for c in WS_CATEGORIES)
            + f"; median vote agreement {np.nanmedian(ws['agree']):.2f}")
    else:
        ws = dict(category=np.full(idx.size, "", dtype=object), dest=np.full(idx.size, -1, int),
                  agree=np.full(idx.size, np.nan))
    return dict(idx=idx, community=com, cell=cell, switch=sw, perimeter=per, resid_z=resid_z, bump_z=bump_z,
                flagged=flagged, explained=explained, partner=partner, shift=shift, quarantine=q,
                reason=reason, criteria=crit, category=ws["category"], ws_dest=ws["dest"], ws_agree=ws["agree"])


def cell_ids(cell, unit, quarantine, nxt, keep=True):
    """Output id of every consensus cell.  keep: largest cell first, each takes the input unit holding
    most of its non-quarantined spikes unless a larger cell already took it; the rest get fresh ids from
    `nxt`.  So a cell that IS an input unit keeps its id (and its cross-chunk link), fragments a cell
    absorbed hand it their ids' spikes, and a split-off cell is new.  Returns ({cell: id}, next free id)."""
    cell = np.asarray(cell); unit = np.asarray(unit); q = np.asarray(quarantine, bool)
    cs, n = np.unique(cell, return_counts=True)
    cmap = {}; taken = set()
    for c in cs[np.argsort(-n, kind="stable")]:
        u = None
        if keep:
            m = (cell == c) & ~q
            for v, _ in Counter(unit[m].tolist()).most_common():
                if v > 1 and v not in taken:
                    u = int(v); break
                if v > 1:
                    break                          # its majority unit is taken -> a new cell
        if u is None:
            u = nxt; nxt += 1
        cmap[int(c)] = u; taken.add(u)
    return cmap, nxt


def write_quarantine_table(path, results, res_abs, clu, out_labels, cell_ids=None):
    """Per quarantined spike: index, absolute time, input unit, consensus cell, output cluster,
    criteria values, reason, and the watershed category ('-' when not run), its vote agreement and
    destination.  `cell_ids` (one {cell: output id} map per result) translates cell and ws_dest to
    the output .clu ids; without it they are the neighbourhood-local cell labels."""
    with open(path, "w") as f:
        f.write("spike\ttime\tunit\tcell\tout\tswitch\tperimeter\tresid_z\tbump_z\tcollision\texplained\t"
                "partner\tshift\treason\tcategory\tws_agree\tws_dest\n")
        for n_, r in enumerate(results):
            cm = cell_ids[n_] if cell_ids is not None else {}
            cat = r.get("category"); dst = r.get("ws_dest"); agr = r.get("ws_agree")
            for j, i in enumerate(r["idx"]):
                if not r["quarantine"][j]:
                    continue
                c = int(r["cell"][j]); d = int(dst[j]) if dst is not None else -1
                f.write(f"{i}\t{int(res_abs[i])}\t{int(clu[i])}\t{cm.get(c, c)}\t{int(out_labels[i])}\t"
                        f"{r['switch'][j]:.3f}\t{r['perimeter'][j]:.3f}\t{r['resid_z'][j]:.2f}\t{r['bump_z'][j]:.2f}\t"
                        f"{int(r['flagged'][j])}\t{int(r['explained'][j])}\t"
                        f"{int(r['partner'][j])}\t{int(r['shift'][j])}\t{r['reason'][j]}\t"
                        f"{(cat[j] if cat is not None and cat[j] else '-')}\t"
                        f"{(agr[j] if agr is not None and np.isfinite(agr[j]) else float('nan')):.2f}\t"
                        f"{cm.get(d, d) if d >= 0 else -1}\n")
    return path


# ── knob resolution: CLI > FK_CONSENSUS_* env > global fiber-kit.yaml ($FK_CONFIG) > default ──
# (FK_CONSENSUS_*, not FK_CONS_*: that prefix belongs to fiber-consolidate.)
_KNOBS = {
    "FK_CONSENSUS_NEIGHBOURHOOD_THETA": ("neighbourhood_theta", float, 12.0,
                                         "envelope d' cut for automatic neighbourhoods (ignored with --units)"),
    "FK_CONSENSUS_MAX_SPIKES": ("max_spikes", int, 0,
                                "re-cut a neighbourhood holding more spikes at a lower d' until it fits (0 = no cap; "
                                "a cap costs completeness: it splits confusable cells apart)"),
    "FK_CONSENSUS_RUNS": ("runs", int, 40, "resampled t-SNE runs"),
    "FK_CONSENSUS_FRAC": ("frac", float, 0.7, "fraction of the neighbourhood's spikes in each run"),
    "FK_CONSENSUS_K": ("k", int, 15, "2-D neighbours per spike"),
    "FK_CONSENSUS_PERPLEXITY": ("perplexity", float, 30.0, "t-SNE perplexity (runs and watershed maps)"),
    "FK_CONSENSUS_THETA_CELL": ("theta_cell", float, 7.0, "envelope d' cut joining communities into cells"),
    "FK_CONSENSUS_SWITCH_THR": ("switch_thr", float, 0.3, "cell-switch rate that quarantines a spike"),
    "FK_CONSENSUS_PERIM_THR": ("perim_thr", float, 0.8,
                               "average 2-D radial rank within the cell (0 centre, 1 rim) for the perimeter rule"),
    "FK_CONSENSUS_DEV_Z": ("dev_z", float, 2.0,
                           "robust z of residual energy or bump score (within the cell) for the perimeter rule"),
    "FK_CONSENSUS_RULES": ("rules", str, "collision,perimeter",
                           "quarantine criteria, any of: switch, collision, perimeter (switch is off by default: "
                           "0.13-0.22 precision where a neighbourhood's cells are close)"),
    "FK_CONSENSUS_WS_RUNS": ("ws_runs", int, 5,
                             "full t-SNE maps voting the watershed category; 0 = no categories, one quarantine "
                             "cluster per neighbourhood"),
    "FK_CONSENSUS_WS_GRID": ("ws_grid", int, 256, "KDE grid side for the watershed basins"),
    "FK_CONSENSUS_WS_ISLAND": ("ws_island", float, 0.5, "quarantined fraction that makes a basin an island"),
    "FK_CONSENSUS_WINDOW": ("window", str, "",
                            "restrict to spikes in START,END minutes of the signal ('' = the whole signal)"),
    "FK_CONSENSUS_CHUNK_MIN": ("chunk_min", float, None,
                               "process the sort chunk by chunk, this many minutes each (unset = FK_SESSION_CHUNK_MIN "
                               "or 12, fiber-session's chunking; 0 = the whole window at once)"),
    "FK_CONSENSUS_KEEP_IDS": ("keep_ids", int, 1,
                              "1 = a cell keeps the id of the input unit holding most of its kept spikes (so a "
                              "linked sort stays linked); 0 = every cell gets a fresh id"),
    "FK_CONSENSUS_MIN_SPIKES": ("min_spikes", int, 300,
                                "skip a neighbourhood with fewer spikes in the chunk (t-SNE at perplexity 30 needs "
                                "a few hundred)"),
    "FK_CONSENSUS_WORKERS": ("workers", int, 0,
                             "parallel processes for the independent t-SNE runs and watershed maps (0 = every "
                             "core, 1 = serial); results do not depend on it"),
    "FK_CONSENSUS_JOBS": ("jobs", int, 1, "openTSNE threads inside each run (keep 1 when workers > 1)"),
    "FK_CONSENSUS_SEED": ("seed", int, 0, "seed for subsampling, initialisation and Louvain"),
}


def _knob_default(name, typ, fallback, gcfg):
    """FK_CONSENSUS_* env if set, else the global fiber-kit.yaml, else the built-in default."""
    v = os.environ.get(name, "")
    if v in (None, ""):
        v = gcfg.get(name)
    if v in (None, ""):
        return fallback
    return typ(v)


def main(argv=None):
    gcfg = cfgmod.load_global_config()
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
    ap.add_argument("--sr", type=float, default=None)
    for name, (dest, typ, fb, hlp) in _KNOBS.items():
        d = _knob_default(name, typ, fb, gcfg)
        ap.add_argument("--" + dest.replace("_", "-"), dest=dest, type=typ, default=d,
                        help=f"{hlp} [{name}; default {d}]")
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
    if a.window:
        w0, w1 = (float(x) * 60 * sr for x in a.window.split(","))
        inwin &= (rel >= w0) & (rel < w1)
    rel0 = np.where(inwin, rel, 0)
    rules = tuple(x.strip() for x in a.rules.split(",") if x.strip())
    bad = [x for x in rules if x not in QUARANTINE_RULES]
    if bad:
        ap.error(f"unknown --rules {bad}; choose from {QUARANTINE_RULES}")
    chunk_min = a.chunk_min
    if chunk_min is None:                     # the session's chunking, as fiber-session / backbone-link read it
        chunk_min = float(os.environ.get("FK_SESSION_CHUNK_MIN") or gcfg.get("FK_SESSION_CHUNK_MIN") or 12.0)
    L = int(round(chunk_min * 60 * sr)) if chunk_min > 0 else 0
    chunk_of = res_abs // L if L else np.zeros(res_abs.size, np.int64)
    log(f"[consensus] knobs (CLI > FK_CONSENSUS_* env > $FK_CONFIG > default): runs={a.runs} frac={a.frac} "
        f"k={a.k} perplexity={a.perplexity} theta-cell={a.theta_cell} rules={','.join(rules)} "
        f"perim-thr={a.perim_thr} dev-z={a.dev_z} switch-thr={a.switch_thr} ws-runs={a.ws_runs} "
        f"ws-grid={a.ws_grid} ws-island={a.ws_island} window={a.window or 'all'} "
        f"chunk-min={chunk_min:g} keep-ids={a.keep_ids}")
    given = ([[int(x) for x in g.split(",") if x.strip()] for g in a.units.split(";")] if a.units else None)
    out = clu.copy(); nxt = int(clu.max()) + 1; results = []; cmaps = []; qclu = []; ngroups = 0
    for ch in np.unique(chunk_of[inwin]):
        lab = np.where(inwin & (chunk_of == ch), clu, 0)
        if given is not None:
            groups = given
        else:
            envs, _, _ = fe.run(view, rel0, lab, transform=transform, pre=21, post=21, min_n=30, sr=sr,
                                partners=False, log=lambda s: None)
            groups = neighbourhoods(envs, a.neighbourhood_theta, max_spikes=a.max_spikes)
        if L:
            log(f"chunk {int(ch)} ({int(ch) * chunk_min:g}-{(int(ch) + 1) * chunk_min:g} min): "
                f"{len(groups)} neighbourhood(s)")
        for units in groups:
            n_in = int(np.isin(lab, list(units)).sum())
            if n_in < a.min_spikes:
                log(f"  skip {sorted(units)}: {n_in} spikes < --min-spikes {a.min_spikes}")
                continue
            r = consensus_neighbourhood(view, rel0, lab, units, fet.values, transform=transform, sr=sr,
                                        runs=a.runs, frac=a.frac, k=a.k, perplexity=a.perplexity,
                                        theta_cell=a.theta_cell, switch_thr=a.switch_thr, perim_thr=a.perim_thr,
                                        dev_z=a.dev_z, rules=rules, ws_runs=a.ws_runs, ws_grid=a.ws_grid,
                                        ws_island=a.ws_island, seed=a.seed, n_jobs=a.jobs, workers=a.workers,
                                        log=log)
            cmap, nxt = cell_ids(r["cell"], clu[r["idx"]], r["quarantine"], nxt, keep=bool(a.keep_ids))
            out[r["idx"]] = [cmap[int(c)] for c in r["cell"]]
            q = r["quarantine"]; isl = q & (r["category"] == "island")
            # quarantine clusters per neighbourhood: island (collision / noise groups) and rim (own-rim +
            # other-cell: extremes of a cell, or reassignment candidates) -- or one cluster without categories
            for part, name in ((isl, "island"), (q & ~isl, "rim" if a.ws_runs > 0 else "quarantine")):
                if part.any():
                    out[r["idx"][part]] = nxt; qclu.append(f"{nxt}={name}:{int(part.sum())}"); nxt += 1
            results.append(r); cmaps.append(cmap); ngroups += 1
    tag = a.out_tag or (f"{a.clu_stage}.consensus" if a.clu_stage else "consensus")
    p_clu = nio.write_clu(a.base, a.elec, out, variant=a.clu_method, tag=tag)
    p_q = write_quarantine_table(f"{a.base}.qrn.{a.elec}.{tag}", results, res_abs, clu, out, cell_ids=cmaps)
    nq = int(sum(r["quarantine"].sum() for r in results))
    print(f"{ngroups} neighbourhood(s); {nq} spikes quarantined (clusters {' '.join(qclu) or 'none'}); "
          f"wrote {p_clu} and {p_q}")


if __name__ == "__main__":
    main()
