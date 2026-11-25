#!/usr/bin/env python3
"""fiber-polish -- clean and consolidate an over-split sort, per spike and per cluster.

Four operations on a finished .clu, each built on the gain-free residual of fiber-consolidate's
resid_pass (0.67.31), run in this order:

  LOOP (FK_POLISH_LOOPS times)
    strip     every cluster of at least STRIP_MIN spikes parks its own residual outliers
              (resid_pass: on- and off-channel terms, a robust tail each, flagged sets unioned).
    pool      the parked spikes and every cluster below POOL_MAX spikes are pooled, and a
              FULL-dimensional rkk (every column of the session's feature space, not the
              top-`rkk-dims` reduction the fine split uses) re-clusters the pool.  A pool cluster
              below POOL_MIN goes to the run's residue cluster -- a real cluster, never 0 or 1.
              Cluster 0 (artifacts; a chunk fiber-pipeline shielded) is never touched.
  ROUNDS (FK_POLISH_ROUNDS times)
    islands   each cluster is cut into leaves of at most LEAF spikes by median bisection along
              its first two principal components; each leaf gets a resampled t-SNE consensus
              (fiber-consensus's own runs and co-neighbour communities).  Every community but
              the leaf's largest is an ISLAND.  An island moves to another cluster only when its
              median waveform fits that cluster's median better than its own parent's by
              ISLAND_MARGIN (gain-only residual; the parent's median taken without the island).
    merge     the reverse residual strip.  acc(B|A) is the share of B's spikes that A's strip
              would NOT park; a pair merges when min(acc(A|B), acc(B|A)) >= MERGE_THR.  Greedy,
              best pair first, the union RE-TEMPLATED and re-scored before the next (no
              union-find chaining), with the tails taken from the CLEANER cluster of the pair
              (the smaller threshold per term) so a contaminated cluster's inflated tails cannot
              accept everything.

MEASURED (2026-11-18, against the curators' labels; `good` = share of curator-kept spikes in a
>=50-spike cluster whose dominant curated label is their own, `complete` = share in the single
largest cluster of their own unit):

  sort                                clusters   purity    good   complete
  0312 g6 chunk 19, standard   sort        178    63.6%   62.7%     33.8%
                               polish       80    73.8%   78.7%     42.6%
  0312 g6 chunk 19, hybrid     sort        110    64.1%   63.5%     71.8%
                               polish       63    72.8%   77.9%     56.4%
  0312 g6 t 24-36, hybrid      sort        110    68.1%   67.8%     51.3%
                               polish       71    70.7%   73.7%     55.5%
  0316 g6 (5 unlinked chunks)  sort        750    73.4%   59.6%      8.3%
                               polish      322    71.9%   77.1%     25.2%

What that table does and does not say:
  * the island pass is what raises purity and `good`, and it COSTS completeness (4-15 points a
    pass); the merge gives part of it back.  On a sort that starts complete (the hybrid) the
    chain is a trade, not a gain on every axis.
  * on a chunk-disjoint sort the merge is the cross-chunk linker: 354 unions on 0316 halved the
    cluster count for -0.8 purity.  At MERGE_THR 0.7 it still leaves 322 clusters for 33 units.
  * the loop cost 6.6 points of purity on 0316 and none on the single-chunk 0312 windows.  The
    pool is session-wide; whether that is the cause was not tested.
  * the curated labels are themselves contaminated, so none of these is a clean verdict.
  * MERGE_THR does not transfer from clean pairs: 0.3 separated 29 clean fibers at AUC 0.99 and
    collapsed a whole sort to 9 clusters.  0.7 was clean on the hybrid sorts (49 right / 1 wrong
    unions under the earlier union-find rule) and is the default; nothing below it is.
Every default here is the value those runs used.  None has been tuned.

OPTIONS FROM KLUSTERS' TOOLS (0.71.6; all off by default).  fiber_corecut's core cut beside or in
place of the residual strip (CORE_CUT 1 / 2), and its quality value as a check on a union (MERGE_Q:
not worse than the worse part by more than a tolerance; MERGE_Q_MAX: not above a bound).  MEASURED
2026-11-23 on sirotaA-jg-000005-20120312 group 8 from the fiber_session sort, against the curated
cur_pl, the ISLAND pass off (2 loops, 2 merge rounds; the residue cluster counted as noise; purity
within 10-minute bins, held = curated-unit spikes in a cluster):

  variant                         clusters   purity   >=90% pure    held   right
  as before                            495    83.8%        35%     85.8%   74.5%
  CORE_CUT 1 (beside the strip)        525    85.9%        40%     59.6%   52.5%
  CORE_CUT 2 (in its place)            632    82.3%        26%     67.4%   58.1%
  MERGE_Q_MAX 0.2                      612    84.0%        30%     85.8%   74.7%

  * the core cut parks far more than the strip (308,000 spikes over the two loops against 176,000)
    and buys two points of purity for a quarter of the spikes; alone it is worse than the strip.
  * of the unions that could be judged (both parts with a curated unit: 242 of 927) 46 were wrong.
    "Worse than the worse part" does not tell them from the right ones (AUC 0.48): a wrong union
    mostly involves a part that is already poor.  The union's quality value itself does better
    (AUC 0.74; median 0.18 wrong, -0.02 right), and a bound of 0.2 would have refused 20 of the 46.
    Run for real, the bound refused 1,347 pairs and the merge made other unions instead: 33 wrong
    unions in place of 46, carrying 22,400 spikes in place of 24,000, and the purity moved by 0.2.
  None of the three is a gain on this group; they are options to measure elsewhere, not defaults.
  Not measured: any of them with the island pass on (a full polish), and MERGE_Q run for real.

The strip, the island fit and the merge all score the RAW (.spk.standard) windows at their stored
alignment -- the space the residual was validated in, and a self-derived realign spends the timing
offset that marks contamination (0.67.32).

REALIGN ON THE TOP CHANNELS (0.71.9; off by default).  REALIGN_TOPCH N > 0: before every loop and
every round each cluster of at least REALIGN_MIN spikes is realigned to its own median, with only
the N channels on which that median is largest voting -- Klusters' "Realign top-ch" -- and the
shift applied to every channel, of the raw windows and of the method's.  Each spike is moved at
most REALIGN_MAXLAG samples from where it is stored, each realign starts from the stored windows
(the passes do not accumulate), and the cluster's median shift is removed, so a cluster stays
where it was.  Nothing is written back: the .res and the .spk are not touched, only what this
run's passes read.  MEASURED 2026-11-25 on sirotaA-jg-000005-20120312 group 8, two 18-minute
chunks (162-198 min, 121,352 spikes) of a fiber-session sort (noise model from the .nst,
deconvolution off: no .fil), islands off, against the curation in progress (fc_unmix_h):

  realign            clusters   purity   spikes in >=90%-pure   spikes in two-cell   ARI
  off                     148    71.4%            36%                 14.1%         0.682
  top 1                   151    72.4%            62%                 12.5%         0.756
  top 2                   161    72.6%            62%                 11.2%         0.757
  all channels            156    71.2%            62%                 15.7%         0.755

  * realigning at all is the large step (ARI 0.68 -> 0.76, the strip parks a quarter fewer
    spikes); between the channel counts the difference is small, with 2 ahead on purity and on
    the spikes in clusters that hold two curated cells.
  * the same on a second sort of the chunks (fiber-session's own aligner at top 2): off 0.691,
    top 2 0.750.
  * with the island pass ON (the full polish), same sort: purity 74.3% with and without; ARI
    0.456 off, 0.723 at top 2; curated cells found (a >=80%-pure cluster holding half the cell)
    11 off, 14 at top 2; spikes in two-cell clusters 5.8% off, 7.4% at top 2.  The island pass
    moves less than half as many spikes on realigned clusters (15,326 + 9,238 against 35,395 +
    22,389 over the two rounds): much of what it carries off at the stored alignment is timing.
  * the curation is unfinished and holds two clusters of over 170,000 spikes, so the ARI and the
    >=90%-pure column mostly say how well the large clusters are kept together.  Not measured: a
    whole session, a session with its .fil.
  That is one group, two chunks: an option to measure, not yet a default.  The feature space of the pool rkk and of the island
embedding is the session's: [method block | raw block] when the method .spk and both .pca bases
are present (the hybrid), else the standard basis, else a local SVD.

Writes <base>.clu.<method>.<group>.<out-tag>, and beside it <base>.polish.<method>.<group>.<out-tag>.tsv
with the id of the residue cluster, which `fiber-clean --junk residue` reads (0.71.10) so that a plan can
run the two stages without the id being copied by hand.  The .clc/.clp hierarchy is NOT rebuilt -- islands
and the pool cut across atoms -- so the output is a flat clu.
"""
import argparse
import heapq
import os
import sys
import time

import numpy as np

try:
    from . import config as cfgmod
    from . import fiber_chunks as _fch
    from . import fiber_clean as fcl
    from . import fiber_consolidate as fcn
    from . import fiber_corecut as fcc
    from . import fiber_pca as fp
    from . import neuro_io as nio
    from . import session_yaml as sy
except ImportError:                                    # run as a script from src/fiber_kit
    import config as cfgmod
    import fiber_chunks as _fch
    import fiber_clean as fcl
    import fiber_consolidate as fcn
    import fiber_corecut as fcc
    import fiber_pca as fp
    import neuro_io as nio
    import session_yaml as sy

_KNOBS = {
    "FK_POLISH_LOOPS": ("loops", int, 2, "strip -> pool -> rkk rounds (0 = skip; rounds 3-5 moved `good` <= 0.5 pt)"),
    "FK_POLISH_STRIP_K_ON": ("strip_k_on", float, 3.0, "strip: robust-tail multiplier of the on-channel residual (0 = term off)"),
    "FK_POLISH_STRIP_K_OFF": ("strip_k_off", float, 3.0, "strip: robust-tail multiplier of the off-channel residual (0 = term off)"),
    "FK_POLISH_STRIP_MIN": ("strip_min", int, 100, "strip: smallest cluster that is stripped"),
    "FK_POLISH_CORE_CUT": ("core_cut", int, 0, "strip: 0 = the residual strip alone, 1 = Klusters' core cut as well "
                                                "(a spike either of them parks is parked), 2 = the core cut in its place"),
    "FK_POLISH_CC_MIN": ("cc_min", int, 200, "strip: smallest cluster the core cut runs on"),
    "FK_POLISH_CC_MAX": ("cc_max", int, 40000, "strip: most spikes of a cluster the core cut orders (above it an even subset)"),
    "FK_POLISH_POOL_MAX": ("pool_max", int, 50, "pool: clusters below this many spikes join the pool whole"),
    "FK_POLISH_POOL_MIN": ("pool_min", int, 50, "pool: smallest cluster the pool rkk may create (smaller -> the residue cluster)"),
    "FK_POLISH_POOL_K": ("pool_k", int, 50, "pool: starting cluster count of the full-dimensional rkk"),
    "FK_POLISH_POOL_CAP": ("pool_cap", int, 30000, "pool: most spikes the rkk is run on (the rest go to the residue cluster)"),
    "FK_POLISH_ROUNDS": ("rounds", int, 2, "island -> merge rounds (0 = skip both)"),
    "FK_POLISH_ISLAND": ("island", int, 1, "1 = run the island pass (needs the [consensus] extra), 0 = merge only"),
    "FK_POLISH_LEAF": ("leaf", int, 300, "islands: most spikes per t-SNE leaf (the 200-300 regime)"),
    "FK_POLISH_ISLAND_RUNS": ("island_runs", int, 8, "islands: resampled t-SNE runs per leaf"),
    "FK_POLISH_ISLAND_MIN": ("island_min", int, 10, "islands: smallest community treated as an island"),
    "FK_POLISH_ISLAND_MIN_CLU": ("island_min_clu", int, 60, "islands: smallest cluster (and leaf) that is embedded"),
    "FK_POLISH_ISLAND_MARGIN": ("island_margin", float, 0.5, "islands: move only if residual to the target < MARGIN x residual to the parent"),
    "FK_POLISH_MERGE": ("merge", int, 1, "1 = run the reverse-strip merge, 0 = islands only"),
    "FK_POLISH_MERGE_THR": ("merge_thr", float, 0.7, "merge: min share of each cluster that survives the other's strip (0.3 collapses a sort)"),
    "FK_POLISH_MERGE_MIN": ("merge_min", int, 50, "merge: smallest cluster that takes part"),
    "FK_POLISH_MERGE_Q": ("merge_q", float, -1.0, "merge: quality guard -- a union is refused when its quality value (Klusters' "
                                                   "quality dot, on the raw windows) exceeds the worse of its two parts' by more "
                                                   "than this (0 = no worse than the worse part; -1 = no guard)"),
    "FK_POLISH_MERGE_Q_MAX": ("merge_q_max", float, 0.0, "merge: quality bound -- a union whose quality value exceeds this is refused "
                                                         "(0 = no bound; 0.2 moved purity by 0.2 pt on 0312 g8: the "
                                                         "module header)"),
    "FK_POLISH_REALIGN_TOPCH": ("realign_topch", int, 0, "realign: > 0 = before every loop and every round each cluster is "
                                                         "realigned to its own median, only its N largest channels voting "
                                                         "(Klusters' Realign top-ch; N >= the channel count = all of them), and "
                                                         "every pass reads the windows at that alignment.  0 = the stored "
                                                         "alignment, as before"),
    "FK_POLISH_REALIGN_MAXLAG": ("realign_maxlag", int, 3, "realign: furthest a spike is moved from its stored position (samples)"),
    "FK_POLISH_REALIGN_MIN": ("realign_min", int, 20, "realign: smallest cluster that is realigned (smaller ones stay as stored)"),
    "FK_POLISH_COS": ("cos", float, 0.5, "template cosine a pair needs to be a candidate (islands and merge)"),
    "FK_POLISH_TPL_CAP": ("tpl_cap", int, 1000, "spikes per cluster used for a median template and its tails"),
    "FK_POLISH_HYBRID_PCS": ("hybrid_pcs", int, 3, "features: standard-basis components per channel in the raw block of the hybrid space"),
    "FK_POLISH_HYBRID_SCALE": ("hybrid_scale", float, 2.0, "features: weight of the raw block (the hybrid-search winner's 2.0)"),
    "FK_POLISH_SEED": ("seed", int, 0, "random seed (subsamples, t-SNE runs, rkk)"),
}


def _knob_default(name, typ, fallback, gcfg):
    return cfgmod.knob_value(name, typ, fallback, gcfg)


# ── shared pieces ────────────────────────────────────────────────────────────
def _real(labels):
    """Mask of spikes in a real, non-reserved cluster (not 0/1, not a chunk's reserved id)."""
    labels = np.asarray(labels)
    return (labels > 1) & ~_fch.is_reserved(labels)


def _sub(rows, cap, rng):
    return np.sort(rng.choice(rows, cap, replace=False)) if rows.size > cap else rows


def gain_residual(TI, TX):
    """Gain-only residual of template TI against TX, on + off channel, in the strip's units
    (1 = as large as TX).  inf when TI carries no TX-shaped signal.  The difference of medians of
    two fragments of one cell is almost purely a gain term (0.046 of template energy against 0.46
    for different cells, 0312 g6); also fitting a SHIFT made the hardest pairs worse."""
    h = fcn._resid_taper(TX, 0)
    Ec = (h[:, None] * TX ** 2).sum(0)
    if not Ec.max() > 0:
        return np.inf
    u = Ec / Ec.max()
    Eon = float((u * Ec).sum())
    g = float((h[:, None] * u[None, :] * TI * TX).sum() / Eon)
    if g < 0.25:
        return np.inf
    return float((h[:, None] * (TI - g * TX) ** 2).sum() / (g * g * Eon))


def kd_leaves(P, leaf):
    """Index sets of at most `leaf` points: median bisection of P (n, 2) alternating its columns."""
    out, stack = [], [(np.arange(len(P)), 0)]
    while stack:
        ix, d = stack.pop()
        if ix.size <= leaf:
            out.append(ix)
            continue
        o = ix[np.argsort(P[ix, d % 2], kind="stable")]
        stack += [(o[:o.size // 2], d + 1), (o[o.size // 2:], d + 1)]
    return out


# ── loop: strip -> pool -> full-dimensional rkk ──────────────────────────────
def core_cut_pass(labels, raw, sr, *, cc_min=200, cc_max=40000, skip=(), noise_id=1):
    """Klusters' core cut (fiber_corecut.analyse, "Split Off Spurious Spikes") on every real cluster
    of at least cc_min spikes: the spikes past the cut go to `noise_id`.  Returns (labels, spikes
    parked, clusters cut)."""
    lab = np.asarray(labels).astype(np.int64).copy()
    ids, cnt = np.unique(lab[_real(lab)], return_counts=True)
    parked = cut = 0
    for u, c in zip(ids, cnt):
        if c < cc_min or int(u) in skip:
            continue
        j = np.flatnonzero(lab == u)
        r = fcc.analyse(raw(j), sr, min_spikes=cc_min, max_ordered=cc_max)
        if r["ok"] and not r["keep"].all():
            lab[j[~r["keep"]]] = noise_id
            parked += int((~r["keep"]).sum()); cut += 1
    return lab, parked, cut


def strip_pool_pass(labels, raw, feats, *, k_on=3.0, k_off=3.0, strip_min=100, pool_max=50, pool_min=50,
                    pool_k=50, pool_cap=30000, residue=None, rng=None, seed=42, rkk=None, log=None,
                    core_cut=0, cc_min=200, cc_max=40000, sr=0.0):
    """One loop round.  `raw(ix)` -> (m, S, C) raw windows; `feats(ix)` -> (m, d) full features.
    core_cut 1 / 2 (needs `sr`, the sampling rate): the core cut beside / in place of the residual
    strip.  Both are taken on the clusters as they came in, so with 1 a spike either parks is
    parked.  Returns (labels, stats, residue id)."""
    rng = rng or np.random.default_rng(0)
    lab = np.asarray(labels).astype(np.int64).copy()
    idx = np.flatnonzero(_real(lab) | (lab == 1))
    excl = tuple([0] + ([int(residue)] if residue is not None else []))
    cc_lab, cc_parked, cc_cut = None, 0, 0
    if int(core_cut) and sr > 0:
        cc_lab, cc_parked, cc_cut = core_cut_pass(lab, raw, sr, cc_min=cc_min, cc_max=cc_max,
                                                  skip=() if residue is None else (int(residue),))
    if int(core_cut) == 2 and cc_lab is not None:
        lab, tot = cc_lab, dict(resid=0, resid_clusters=0)
    else:
        lab, tot = fcn.resid_pass(raw, lab, idx, k_on=k_on, k_off=k_off, minspk=strip_min, exclude=excl,
                                  rng=rng)
        if cc_lab is not None:
            lab[(cc_lab == 1) & _real(lab)] = 1               # also parked by the core cut
    ids, cnt = np.unique(lab[_real(lab)], return_counts=True)
    small = ids[(cnt < pool_max) & (ids != (residue if residue is not None else -1))]
    pool = np.flatnonzero((lab == 1) | np.isin(lab, small) |
                          ((lab == residue) if residue is not None else False))
    stats = dict(stripped=int(tot["resid"]), pool=int(pool.size), born=0, residue=0)
    if not pool.size:
        return lab, stats, residue
    if residue is None:
        residue = _fch.next_free_id(lab)
    lab[pool] = residue
    use = _sub(pool, int(pool_cap), rng)
    if use.size >= max(2 * pool_min, 20):
        if rkk is None:
            try:
                from .fiber_session import _rkk_labels as rkk
            except ImportError:
                from fiber_session import _rkk_labels as rkk
        sub = rkk(np.asarray(feats(use), float), max_clusters=int(pool_k), min_size=int(pool_min), seed=seed)
        for c in np.unique(sub[sub >= 0]):
            lab[use[sub == c]] = _fch.next_free_id(lab)
            stats["born"] += 1
    stats["residue"] = int((lab == residue).sum())
    stats["core_cut"] = int(cc_parked)
    if log:
        cc = f"; core cut parked {cc_parked:,} from {cc_cut} clusters" if cc_lab is not None else ""
        log(f"  strip parked {stats['stripped']:,} from {tot['resid_clusters']} clusters{cc}; pool {stats['pool']:,} "
            f"-> {stats['born']} clusters, {stats['residue']:,} in residue cluster {residue}")
    return lab, stats, residue


# ── per-cluster realignment on the largest channels ──────────────────────────────
def shift_windows(W, total):
    """(m, S, C) windows moved by the per-spike sub-sample shifts `total` (Fourier; the convention of
    fiber_lib.align_xcorr: the result is roll(W, -total)).  A zero shift returns the windows as given."""
    W = np.asarray(W, float); total = np.asarray(total, float)
    if not total.size or not np.any(total):
        return W
    f = np.fft.fftfreq(W.shape[1])
    return np.fft.ifft(np.fft.fft(W, axis=1) * np.exp(2j * np.pi * f[None, :, None] * total[:, None, None]), axis=1).real


def cluster_shifts(W, topch=2, maxlag=3, iters=2, cap=2000, rng=None, batch=20000):
    """Per-spike shifts that align one cluster's windows W (m, S, C), as stored, to the cluster's
    own median -- Klusters' "Realign top-ch": the cross-correlation sums over only the `topch`
    channels on which the median has the largest peak |amplitude| (all of them when topch is 0
    or >= C), and the shift found is for every channel.  Each shift is the lag within +-maxlag of
    the stored position that maximises the raw correlation, refined by a 3-point parabola.  The
    median is taken on at most `cap` spikes and re-taken after each pass; the cluster's median
    shift is removed at the end, so the cluster as a whole stays where it was stored.
    Returns (shifts (m,), the voting channels)."""
    W = np.asarray(W); m, S, C = W.shape
    rng = rng or np.random.default_rng(0)
    total = np.zeros(m)
    if m < 3:
        return total, np.arange(C)
    sub = np.sort(rng.choice(m, cap, replace=False)) if m > cap else np.arange(m)
    lag = ((np.arange(S) + S // 2) % S) - S // 2
    inwin = np.abs(lag) <= int(maxlag)
    top = np.arange(C)
    for _ in range(max(1, int(iters))):
        T = np.median(shift_windows(np.asarray(W[sub], float), total[sub]), axis=0)
        if 0 < int(topch) < C:
            top = np.sort(np.argsort(-np.abs(T).max(0), kind="stable")[:int(topch)])
        Ft = np.conj(np.fft.fft(T[:, top], axis=0))[None]
        for a in range(0, m, int(batch)):
            w = np.asarray(W[a:a + int(batch)], float)[:, :, top]
            xc = np.fft.ifft(np.fft.fft(w, axis=1) * Ft, axis=1).real.sum(2)
            k = np.where(inwin[None, :], xc, -np.inf).argmax(1); d = lag[k].astype(float)
            r = np.arange(len(w)); y0 = xc[r, (k - 1) % S]; y1 = xc[r, k]; y2 = xc[r, (k + 1) % S]
            den = y0 - 2 * y1 + y2; ok = np.abs(den) > 1e-9
            d += np.where(ok, 0.5 * (y0 - y2) / np.where(ok, den, 1.0), 0.0).clip(-0.5, 0.5)
            total[a:a + len(w)] = d
    return total - np.median(total), top


def realign_clusters(labels, windows, shift, *, topch=2, maxlag=3, min_n=20, skip=(), rng=None, log=None):
    """Fill `shift` (N,) in place: every real cluster of at least `min_n` spikes gets cluster_shifts
    of its STORED windows (`windows(ix)` -> (m, S, C), unshifted); every other spike gets 0.  Each
    call starts from the stored windows, so repeated calls do not accumulate.  Returns the number
    of clusters realigned."""
    lab = np.asarray(labels); shift[:] = 0
    ids, cnt = np.unique(lab[_real(lab)], return_counts=True); done = 0; moved = []
    for u, c in zip(ids, cnt):
        if c < int(min_n) or int(u) in skip:
            continue
        ix = np.flatnonzero(lab == u)
        s, _ = cluster_shifts(windows(ix), topch=topch, maxlag=maxlag, rng=rng)
        shift[ix] = s; done += 1; moved.append(np.abs(s))
    if log:
        mv = np.concatenate(moved) if moved else np.zeros(1)
        log(f"  realign (top {int(topch)} channels, +-{int(maxlag)}): {done} clusters, {mv.size:,} spikes; "
            f"|shift| median {np.median(mv):.2f}, 90th pct {np.quantile(mv, .9):.2f} samples")
    return done


# ── islands ──────────────────────────────────────────────────────────────────
def island_pass(labels, raw, F, *, leaf=300, runs=8, island_min=10, min_clu=60, margin=0.5, cos=0.5,
                tpl_cap=1000, skip=(), rng=None, log=None):
    """Per-cluster small-leaf t-SNE consensus; reassign an island only where its median fits another
    cluster better than its parent.  `F` (n, d) full features of every spike.  Returns (labels, stats)."""
    try:
        from . import fiber_consensus as fcs
    except ImportError:
        import fiber_consensus as fcs
    rng = rng or np.random.default_rng(0)
    lab = np.asarray(labels).astype(np.int64)
    ids, cnt = np.unique(lab[_real(lab)], return_counts=True)
    clus = [int(f) for f, c in zip(ids, cnt) if c >= min_clu and int(f) not in skip]
    tmpl = {f: np.median(raw(_sub(np.flatnonzero(lab == f), tpl_cap, rng)), 0) for f in clus}
    tn = {f: tmpl[f].ravel() / max(np.linalg.norm(tmpl[f]), 1e-12) for f in clus}
    new = lab.copy()
    st = dict(leaves=0, islands=0, moved=0, spikes=0)
    for ci, f in enumerate(clus):
        m = np.flatnonzero(lab == f)
        Z = F[m] - F[m].mean(0)
        U, Sv, _ = np.linalg.svd(Z, full_matrices=False)
        for loc in kd_leaves(U[:, :2] * Sv[:2], int(leaf)):
            if loc.size < min_clu:
                continue
            st["leaves"] += 1
            Zc = F[m[loc]]
            Zc = (Zc - Zc.mean(0)) / Zc.std(0).clip(1e-9)
            sub = max(17, int(round(0.7 * loc.size)))
            rr = [fcs._one_run(Zc, np.sort(rng.choice(loc.size, sub, replace=False)),
                               min(30.0, (sub - 2) / 3.0), 1000 * ci + r, 15, 1, 120, 300)
                  for r in range(int(runs))]
            com = fcs.coneighbour_communities(rr, loc.size, seed=ci)
            sizes = np.bincount(com)
            core = int(sizes.argmax())
            for k in np.flatnonzero(sizes >= island_min):
                if k == core:
                    continue                                   # the leaf's core stays
                isl = m[loc[com == k]]
                st["islands"] += 1
                TI = np.median(raw(isl), 0)
                rest = np.setdiff1d(m, isl)
                if rest.size < 8:
                    continue
                dP = gain_residual(TI, np.median(raw(_sub(rest, tpl_cap, rng)), 0))
                ti = TI.ravel() / max(np.linalg.norm(TI), 1e-12)
                best, dB = None, np.inf
                for x in clus:
                    if x != f and float(ti @ tn[x]) >= cos:
                        d = gain_residual(TI, tmpl[x])
                        if d < dB:
                            best, dB = x, d
                if best is not None and dB < margin * dP:
                    new[isl] = best
                    st["moved"] += 1
                    st["spikes"] += int(isl.size)
    if log:
        log(f"  islands: {len(clus)} clusters -> {st['leaves']} leaves -> {st['islands']} islands, "
            f"{st['moved']} moved ({st['spikes']:,} spikes)")
    return new, st


# ── merge: the reverse residual strip ────────────────────────────────────────
def _entry(W, k):
    T = np.median(W, 0)
    t = fcn.resid_terms(W, T)
    if t is None or t["ok"].sum() < 10:
        return None
    ok = t["ok"]
    return dict(W=W, T=T, tn=T.ravel() / max(np.linalg.norm(T), 1e-12),
                thr=(fcn._resid_tail(t["onr"][ok], k), fcn._resid_tail(t["offr"][ok], k)), q=None)


def _quality(e):
    """The entry's quality value (fiber_corecut.residual_variance of its sampled raw windows), cached."""
    if e["q"] is None:
        e["q"] = fcc.residual_variance(e["W"])
    return e["q"]


def pair_acceptance(A, B):
    """min over both directions of the share of one cluster's spikes the other's strip keeps, under
    the CLEANER cluster's tails (the smaller threshold per term)."""
    to, tf = min(A["thr"][0], B["thr"][0]), min(A["thr"][1], B["thr"][1])
    out = []
    for X, Y in ((A, B), (B, A)):
        t = fcn.resid_terms(Y["W"], X["T"])
        out.append(0.0 if t is None else float((t["ok"] & (t["onr"] <= to) & (t["offr"] <= tf)).mean()))
    return min(out)


def merge_pass(labels, raw, *, thr=0.7, cos=0.5, min_n=50, tpl_cap=1000, k=3.0, skip=(), rng=None, log=None,
               merge_q=-1.0, merge_q_max=0.0, stats=None):
    """Greedy reverse-strip merge with re-templating.  merge_q >= 0: the quality guard -- the union's
    quality value (of a sample of its raw windows) may exceed the worse of its two parts' by at most
    merge_q, else the pair is refused (and scored again only when one side has changed).
    merge_q_max > 0: the quality bound -- a union whose quality value exceeds it is refused,
    whatever its parts'.  `stats`
    (a dict) receives unions, refused and, with either on, `pairs`: one (a, b, quality of a, of
    b, of the union, refused) per pair that reached the guard, in order.  Returns (labels, unions)."""
    rng = rng or np.random.default_rng(0)
    lab = np.asarray(labels).astype(np.int64).copy()
    ids, cnt = np.unique(lab[_real(lab)], return_counts=True)
    E = {}
    for f, c in zip(ids, cnt):
        if c >= min_n and int(f) not in skip:
            e = _entry(raw(_sub(np.flatnonzero(lab == f), tpl_cap, rng)), k)
            if e is not None:
                E[int(f)] = e
    ver = {f: 0 for f in E}
    heap, keys = [], list(E)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if float(E[a]["tn"] @ E[b]["tn"]) > cos:
                s = pair_acceptance(E[a], E[b])
                if s >= thr:
                    heap.append((-s, a, b, 0, 0))
    heapq.heapify(heap)
    unions = refused = 0
    pairs = []
    while heap:
        _, a, b, va, vb = heapq.heappop(heap)
        if a not in E or b not in E or ver[a] != va or ver[b] != vb:
            continue                                           # stale: one side changed since it was scored
        e = _entry(raw(_sub(np.flatnonzero((lab == a) | (lab == b)), tpl_cap, rng)), k)   # the union, re-templated
        if (merge_q >= 0 or merge_q_max > 0) and e is not None:
            qa, qb, qu = _quality(E[a]), _quality(E[b]), _quality(e)
            no = bool(merge_q >= 0 and np.isfinite(qa) and np.isfinite(qb) and np.isfinite(qu) and qu > max(qa, qb) + merge_q)
            no = no or bool(merge_q_max > 0 and np.isfinite(qu) and qu > merge_q_max)
            pairs.append((int(a), int(b), float(qa), float(qb), float(qu), no))
            if no:
                refused += 1                                   # the union is too poor a cluster
                continue
        lab[lab == b] = a
        del E[b]
        unions += 1
        if e is None:
            del E[a]
            continue
        E[a] = e
        ver[a] += 1
        for x in list(E):
            if x != a and float(e["tn"] @ E[x]["tn"]) > cos:
                s = pair_acceptance(e, E[x])
                if s >= thr:
                    heapq.heappush(heap, (-s, a, x, ver[a], ver[x]))
    if stats is not None:
        stats.update(unions=unions, refused=refused, pairs=pairs)
    if log:
        log(f"  merge: {len(keys)} clusters, {unions} unions at acceptance >= {thr:g}"
            + (f"; {refused} refused by the quality guard"
               + (f" (worse than the worse part by > {merge_q:g})" if merge_q >= 0 else "")
               + (f" (quality value > {merge_q_max:g})" if merge_q_max > 0 else "")
               if (merge_q >= 0 or merge_q_max > 0) else ""))
    return lab, unions


# ── driver ───────────────────────────────────────────────────────────────────
def polish(labels, raw, feats, F=None, *, loops=2, rounds=2, island=1, merge=1, strip_k_on=3.0,
           strip_k_off=3.0, strip_min=100, pool_max=50, pool_min=50, pool_k=50, pool_cap=30000, leaf=300,
           island_runs=8, island_min=10, island_min_clu=60, island_margin=0.5, merge_thr=0.7, merge_min=50,
           cos=0.5, tpl_cap=1000, seed=0, rkk=None, log=None, on_stage=None,
           core_cut=0, cc_min=200, cc_max=40000, merge_q=-1.0, merge_q_max=0.0, sr=0.0, merge_stats=None,
           realign=None, refeat=None):
    """Run the chain.  `realign(labels, skip)` (optional) is called before every loop and every round:
    it re-times each cluster's spikes, and `raw` / `feats` must then return the windows and features at
    that timing (the CLI's do); `refeat()` returns the full feature matrix after it, for the islands.
    Without `realign` nothing is re-timed.  `sr` (the sampling rate) is needed for core_cut 1 / 2.  `merge_stats` (a
    list) receives each merge round's stats dict, with the labels it started from as `before`.  `F` is the full feature matrix (needed for islands; feats(arange) if None).
    `on_stage(tag, labels)` is called after every stage (the CLI's scoring hook)."""
    rng = np.random.default_rng(int(seed))
    lab = np.asarray(labels).astype(np.int64).copy()   # cluster 0 (artifacts, a shielded chunk) is never touched
    residue = None
    for it in range(1, int(loops) + 1):
        t0 = time.time()
        if log:
            log(f"[polish] loop {it}/{loops}")
        if realign is not None:
            realign(lab, () if residue is None else (int(residue),))
        lab, _, residue = strip_pool_pass(lab, raw, feats, k_on=strip_k_on, k_off=strip_k_off,
                                          strip_min=strip_min, pool_max=pool_max, pool_min=pool_min,
                                          pool_k=pool_k, pool_cap=pool_cap, residue=residue, rng=rng,
                                          seed=42 + it + int(seed), rkk=rkk, log=log,
                                          core_cut=core_cut, cc_min=cc_min, cc_max=cc_max, sr=sr)
        if on_stage:
            on_stage(f"loop {it} ({time.time() - t0:.0f}s)", lab)
    skip = () if residue is None else (int(residue),)
    for it in range(1, int(rounds) + 1):
        if realign is not None:
            realign(lab, skip)
            if island and refeat is not None:
                F = refeat()
        if island:
            if F is None:
                F = np.asarray(feats(np.arange(lab.size)), float)
            t0 = time.time()
            if log:
                log(f"[polish] islands {it}/{rounds}")
            lab, _ = island_pass(lab, raw, F, leaf=leaf, runs=island_runs, island_min=island_min,
                                 min_clu=island_min_clu, margin=island_margin, cos=cos, tpl_cap=tpl_cap,
                                 skip=skip, rng=rng, log=log)
            if on_stage:
                on_stage(f"islands {it} ({time.time() - t0:.0f}s)", lab)
        if merge:
            t0 = time.time()
            if log:
                log(f"[polish] merge {it}/{rounds}")
            st = dict(before=lab.copy()) if merge_stats is not None else None
            lab, _ = merge_pass(lab, raw, thr=merge_thr, cos=cos, min_n=merge_min, tpl_cap=tpl_cap,
                                skip=skip, rng=rng, log=log, merge_q=merge_q, merge_q_max=merge_q_max, stats=st)
            if merge_stats is not None:
                merge_stats.append(st)
            if on_stage:
                on_stage(f"merge {it} ({time.time() - t0:.0f}s)", lab)
    if residue is not None:
        lab[lab == 1] = residue                               # nothing this stage parked stays in the noise cluster
    return lab, residue


def score_against(labels, truth, min_n=50):
    """(clusters, purity, good, complete) against a curated labelling -- see the module docstring."""
    lab, cur = np.asarray(labels), np.asarray(truth)
    kept = cur > 1
    ids, cnt = np.unique(lab[_real(lab)], return_counts=True)
    big = ids[cnt >= min_n]
    tot = dom = good = 0
    best = {}
    for f in big:
        c = np.bincount(cur[lab == f])
        tot += int(c.sum())
        dom += int(c.max())
        if int(c.argmax()) > 1:
            good += int(c.max())
        for u in np.flatnonzero(c):
            if u > 1:
                best[int(u)] = max(best.get(int(u), 0), int(c[u]))
    nk = max(int(kept.sum()), 1)
    return len(big), dom / max(tot, 1), good / nk, sum(best.values()) / nk


def build_features(base, elec, method, spk_m, raw, nchan, hybrid_pcs=3, hybrid_scale=2.0, log=None):
    """feats(ix) for the session's feature space: hybrid, else the standard basis, else None."""
    std = fp.read_cluster_basis(base, elec, "standard")
    mb = fp.read_cluster_basis(base, elec, method) if method else None
    if spk_m is not None and mb is not None and std is not None \
            and nio.parse_variant_token(method).family != "standard":
        try:
            from . import fiber_lib as fl
        except ImportError:
            import fiber_lib as fl
        n = raw.shape[0]
        mb["_hyb"] = fp.hybrid_basis(std, pcs=hybrid_pcs, scale=hybrid_scale, nraw=nchan)
        gs = np.sort(np.random.default_rng(0).choice(n, min(n, 20000), replace=False))
        fl.set_align_channels(nchan)
        mb["_hyb"]["gain"] = fp.fit_hybrid_gain(np.asarray(spk_m[gs], float), np.asarray(raw[gs], float), mb)
        if log:
            log(f"[polish] features: hybrid [{method} | standard x{hybrid_pcs}], scale {hybrid_scale:g}, "
                f"gain {mb['_hyb']['gain']:.4g}")
        return lambda ix: fp.cluster_features(fp.hybrid_stack(spk_m[np.asarray(ix)], raw[np.asarray(ix)]),
                                              mb, realign=False, dims=None)
    if std is not None:
        if log:
            log("[polish] features: standard basis (no method .spk / .pca pair for the hybrid)")
        return lambda ix: fp.cluster_features(np.asarray(raw[np.asarray(ix)], float), std, realign=False, dims=None)
    return None


def main(argv=None):
    gcfg = cfgmod.load_global_config()
    ap = argparse.ArgumentParser(description="Polish an over-split sort: residual strip + pooled full-dimensional "
                                             "rkk, small-leaf t-SNE islands, reverse-strip merge.")
    ap.add_argument("base"); ap.add_argument("elec", type=int)
    nio.add_clu_args(ap, method_default="stderiv", stage_default="fiber_session", stage_dest="clu_stage",
                     stage_alias=None)
    ap.add_argument("--res-variant", default="", help="pin the .res variant ('' = the shared .res)")
    ap.add_argument("--gt-clu", default="", help="curated .clu path: score every stage against it (nothing is fitted to it)")
    for name, (dest, typ, fb, hlp) in _KNOBS.items():
        d = _knob_default(name, typ, fb, gcfg)
        ap.add_argument("--" + dest.replace("_", "-"), dest=dest, type=typ, default=d,
                        help=f"{hlp} [{name}; default {d}]")
    ap.add_argument("--out-tag", default=None, help="output stage tag (default <clu-stage>_polish)")
    cfgmod.check_choice_defaults(ap)
    a = ap.parse_args(argv)
    log = lambda s: print(s, file=sys.stderr, flush=True)  # noqa: E731

    cfg = sy.resolve_session_params(a.base, a.elec, verbose=False)
    base = cfg["base"]; NS, NC = int(cfg["nsamp"]), int(cfg["nchan"])
    res = nio.read_res_at(base, a.elec, variant=a.res_variant) if a.res_variant else nio.read_res(base, a.elec)
    n = len(res)
    _, clu = nio.resolve_clu(a, base, a.elec, n_spikes=n, stage_dest="clu_stage")
    clu = np.asarray(clu, np.int64)
    try:
        rawmm, rs = nio.open_spk(base, a.elec, NS, NC, prefer=nio.prefer_standard())
    except FileNotFoundError:
        raise SystemExit("fiber-polish needs the RAW waveforms (.spk.standard.<group>): the residual strip, "
                         "the island fit and the merge are all scored in the raw space") from None
    if rawmm.shape[0] != n:
        raise SystemExit(f"{os.path.basename(rs.path)} holds {rawmm.shape[0]:,} spikes, the .res {n:,}")
    spk_m = None
    if nio.parse_variant_token(a.clu_method).family != "standard":
        r = nio.resolve(base, "spk", a.elec, a.clu_method)
        if r.found and r.variant == a.clu_method:
            spk_m = nio.open_spk_file(r.path, NS, NC)
            if spk_m.shape[0] != n:
                raise SystemExit(f"{os.path.basename(r.path)} holds {spk_m.shape[0]:,} spikes, the .res {n:,}")
    shift = np.zeros(n, np.float32) if a.realign_topch > 0 else None       # per-spike re-timing, 0 = as stored

    class _Timed:
        """A waveform file read at the current re-timing: [ix] -> float windows moved by shift[ix]."""
        def __init__(self, mm):
            self.mm = mm; self.shape = mm.shape

        def __getitem__(self, ix):
            ix = np.asarray(ix)
            return shift_windows(self.mm[ix], shift[ix])

    stored = rawmm
    if shift is not None:
        rawmm = _Timed(rawmm)
        spk_m = _Timed(spk_m) if spk_m is not None else None
    feats = build_features(base, a.elec, a.clu_method, spk_m, rawmm, NC, a.hybrid_pcs, a.hybrid_scale, log=log)
    if feats is None:
        log("[polish] features: no .pca basis -- local SVD of the raw windows (24 components)")
        sm = np.sort(np.random.default_rng(0).choice(n, min(n, 20000), replace=False))
        w = np.asarray(rawmm[sm], float).reshape(sm.size, -1)
        mu = w.mean(0)
        Vt = np.linalg.svd(w - mu, full_matrices=False)[2][:24]
        feats = lambda ix: (np.asarray(rawmm[np.asarray(ix)], float).reshape(len(ix), -1) - mu) @ Vt.T  # noqa: E731
    raw = lambda ix: np.asarray(rawmm[np.asarray(ix)], float)   # noqa: E731

    truth = None
    if a.gt_clu:
        _, truth = nio.read_clu_file(a.gt_clu, n_spikes=n)
        truth = np.asarray(truth)

    def on_stage(tag, lab):
        ids, cnt = np.unique(lab[_real(lab)], return_counts=True)
        line = f"[polish] {tag:<20} clusters {len(ids):>5} (>=50: {int((cnt >= 50).sum()):>4})"
        if truth is not None:
            nb, p, g, c = score_against(lab, truth)
            line += f"   purity {p:6.1%}  good {g:6.1%}  complete {c:6.1%}"
        log(line)

    log(f"[polish] {os.path.basename(base)} group {a.elec}: {n:,} spikes, stage "
        f"'{a.clu_stage}', raw {NS}x{NC}")
    on_stage("input", clu)
    F = None
    all_feats = lambda: np.vstack([np.asarray(feats(np.arange(i, min(i + 20000, n))), float)   # noqa: E731
                                   for i in range(0, n, 20000)])
    if a.rounds > 0 and a.island and shift is None:
        F = all_feats()
    kw = {dest: getattr(a, dest) for dest, *_ in _KNOBS.values() if not dest.startswith(("hybrid_", "realign_"))}
    if shift is not None:
        rrng = np.random.default_rng(int(a.seed) + 7)
        kw["realign"] = lambda lab, skip: realign_clusters(                                    # noqa: E731
            lab, lambda ix: stored[np.asarray(ix)], shift, topch=a.realign_topch, maxlag=a.realign_maxlag,
            min_n=a.realign_min, skip=skip, rng=rrng, log=log)
        kw["refeat"] = all_feats
    if a.core_cut and not float(cfg.get("sr") or 0) > 0:
        raise SystemExit("fiber-polish --core-cut needs the sampling rate (the session .yaml)")
    new, residue = polish(clu, raw, feats, F, log=log, on_stage=on_stage, sr=float(cfg.get("sr") or 0), **kw)
    tag = a.out_tag or (f"{a.clu_stage}_polish" if a.clu_stage else "polish")
    path = nio.write_clu(base, a.elec, new, variant=a.clu_method, tag=tag)
    # the residue id, for `fiber-clean --junk residue` on this stage
    fcl.write_residue_record(base, a.elec, a.clu_method, tag, residue,
                             0 if residue is None else int((new == residue).sum()), source=a.clu_stage or "")
    log(f"[polish] wrote {os.path.basename(path)}"
        + (f"  (residue cluster {residue}: {int((new == residue).sum()):,} spikes)" if residue is not None else "")
        + "  -- flat .clu; the .clc/.clp hierarchy is not rebuilt")


if __name__ == "__main__":
    main()
