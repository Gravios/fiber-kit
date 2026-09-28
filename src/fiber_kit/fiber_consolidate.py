#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════════════════════
#  fiber_consolidate.py — per-spike cluster consolidation for a finished
#  labelling: the Klusters waveform TEMPLATE STRIP (the same metric, ported)
#  followed by the fiber-refine KNN-PEEL (reused, not reimplemented).
#
#  The intrachunk and defrag stages operate on whole FRAGMENTS: an atom that
#  carries 5% of a neighbour's spikes carries them into whichever unit it
#  joins, and no fragment-level gate can take them back out.  Klusters grew
#  two per-spike tools for exactly this — the KNN split and the template
#  strip — which the curator runs by hand after linking.  This module is the
#  automated form of that pass, run PER CHUNK over the final labels:
#
#  STRIP — each eligible cluster's median template (≤ tpl_cap evenly-strided
#    spikes, stored alignment, exactly as Klusters samples it) claims spikes
#    of OTHER clusters whose kernel-weighted normalised residual D is at or
#    below max_dist and whose matched gain g sits in [gmin, gmax], plus the
#    optional per-channel uniformity gate — the Klusters metric verbatim
#    (klustersdoc_strip.cpp; kernel w=|T|, D = sqrt(Σw·(x−T)²/Σw)/denom with
#    denom = sqrt(Σw·T²/Σw), g = Σw·x·T / Σw·T²).  Where Klusters parks the
#    matches in a new cluster for the curator, consolidation REASSIGNS them
#    to the template's cluster, and adds the safeguard that interactive
#    review provided: a spike moves only when the claiming template beats
#    the spike's OWN cluster's template by a margin (D_target ≤ margin ·
#    D_own), so a template can never strip-mine a look-alike's core.  When
#    several templates claim a spike the smallest D wins.  Every spike is
#    scored against every template in one two-matmul pass per block — a
#    contaminant is by definition UNLIKE its own cluster's template, so no
#    donor-level similarity prefilter can be sound.
#
#  KNN — fiber_refine._knn_apply, the library form of the Klusters KNN
#    split: per-spike K-NN majority vote in a PCA feature space against the
#    pool of other clusters' spikes; a peeled bucket folds into the winner
#    when its median waveform matches (amplitude-sensitive xcorr ≥
#    fold_thr), else it becomes a NEW cluster.
#
#  Scope is PER CHUNK in both callers: intrachunk's units are chunk-local by
#  contract, and anchor-link's cells drift across chunks, so a cell is best
#  represented by its own within-chunk spikes rather than a drift-smeared
#  session template.  Labels in `exclude` (the reserve bins) neither donate
#  nor receive, and their spikes never move.
#
#  Knobs read FK_CONS_* (CLI > FK_CONS_* env > global yaml where the caller
#  passes one > default).
# ════════════════════════════════════════════════════════════════════════════
import os
import numpy as np

try:
    from . import fiber_lib as fl
    from .fiber_refine import _knn_apply
except ImportError:                                   # script / flat-layout fallback
    import fiber_lib as fl
    from fiber_refine import _knn_apply


# ── knob resolution: default <- global yaml (FK_CONS_*) <- FK_* env <- CLI ──
_KNOBS = {
    "FK_CONS_MODE": ("cons_mode", str, "off"),
    "FK_CONS_MAX_DIST": ("cons_max_dist", float, 0.5),
    "FK_CONS_GMIN": ("cons_gmin", float, 0.0),
    "FK_CONS_GMAX": ("cons_gmax", float, 10.0),
    "FK_CONS_CHAN": ("cons_chan", float, 0.0),
    "FK_CONS_MARGIN": ("cons_margin", float, 0.85),
    "FK_CONS_TPL_CAP": ("cons_tpl_cap", int, 1024),
    "FK_CONS_MIN_TPL": ("cons_min_tpl", int, 8),
    "FK_CONS_KNN_K": ("cons_knn_k", int, 20),
    "FK_CONS_KNN_THR": ("cons_knn_thr", float, 0.3),
    "FK_CONS_KNN_MINREF": ("cons_knn_minref", int, 50),
    "FK_CONS_KNN_MINNEW": ("cons_knn_minnew", int, 30),
    "FK_CONS_KNN_DIMS": ("cons_knn_dims", int, 16),
    "FK_CONS_FOLD_THR": ("cons_fold_thr", float, 0.9),
    "FK_CONS_SCORR": ("cons_scorr", float, 1.0),
    "FK_CONS_OFF_THR": ("cons_off_thr", float, 0.0),
}


def _knob_default(name, typ, fallback, gcfg):
    for src in (os.environ.get(name), (gcfg or {}).get(name)):
        if src is None or str(src).strip() == "":
            continue
        try:
            return typ(src)
        except (TypeError, ValueError):
            pass
    return fallback


def add_consolidate_args(ap, gcfg=None):
    """Attach the FK_CONS_* knob group to a stage's parser (same resolution
    order as the stage's own knobs: CLI > env > global yaml > default)."""
    g = ap.add_argument_group("consolidation (per-spike strip + knn cleanup of the final labels)")
    for name, (dest, typ, fb) in _KNOBS.items():
        d = _knob_default(name, typ, fb, gcfg)
        if dest == "cons_mode":
            g.add_argument("--cons-mode", dest=dest, choices=("off", "strip", "knn", "both"),
                           default=d, help=f"{name}: which consolidation passes run (default {d})")
        else:
            g.add_argument("--" + dest.replace("_", "-"), dest=dest, type=typ, default=d,
                           help=f"{name} (default {d})")


def kwargs_from_args(a):
    """(mode, strip_kw, knn_kw) from a parsed namespace add_consolidate_args filled."""
    strip_kw = dict(max_dist=a.cons_max_dist, gmin=a.cons_gmin, gmax=a.cons_gmax,
                    chan_uniform=a.cons_chan, margin=a.cons_margin,
                    tpl_cap=a.cons_tpl_cap, min_tpl=a.cons_min_tpl)
    knn_kw = dict(k=a.cons_knn_k, thr=a.cons_knn_thr, minref=a.cons_knn_minref,
                  minnew=a.cons_knn_minnew, dims=a.cons_knn_dims, fold_thr=a.cons_fold_thr,
                  scorr=a.cons_scorr, off_thr=(a.cons_off_thr if a.cons_off_thr > 0 else None))
    return a.cons_mode, strip_kw, knn_kw


# ── the Klusters strip metric, vectorised ────────────────────────────────────
def _template(get_waves, idx, tpl_cap, rng):
    """Median template over ≤ tpl_cap evenly-strided spikes (stored alignment —
    the strip scores unaligned records, exactly as Klusters does)."""
    idx = np.asarray(idx)
    step = max(1, idx.size // int(tpl_cap))
    w = np.asarray(get_waves(idx[::step][: int(tpl_cap)]), float)
    return np.median(w, 0)


def _tpl_terms(T):
    """Kernel terms of one template: (flatT, w, wT, W, E, denom) plus the
    per-channel (Wc, Ec, denomc, gate) arrays for the uniformity gate."""
    flat = T.ravel()
    w = np.abs(flat)
    W = float(w.sum())
    E = float((w * flat * flat).sum())
    if W <= 0.0 or E <= 0.0:
        return None                                    # flat template: cannot strip with it
    wch = np.abs(T)                                    # (nsamp, nch)
    Wc = wch.sum(0)
    Ec = (wch * T * T).sum(0)
    gate = (Ec >= 0.05 * float(Ec.max())) & (Wc > 0)
    denomc = np.sqrt(np.where(gate, Ec / np.maximum(Wc, 1e-30), 1.0))
    return dict(flat=flat, w=w, wT=w * flat, W=W, E=E, denom=float(np.sqrt(E / W)),
                Wc=Wc, Ec=Ec, denomc=denomc, gate=gate, shape=T.shape)


def _score_block(X, t):
    """D and g of a (n, P) block of flattened spikes against template terms `t`.
    q = Σw(x−T)² expanded as x²·w − 2·x·wT + E, so one pass, two matmuls."""
    A = (X * X) @ t["w"]
    B = X @ t["wT"]
    q = np.maximum(A - 2.0 * B + t["E"], 0.0)
    D = np.sqrt(q / t["W"]) / t["denom"]
    g = B / t["E"]
    return D, g


def _chan_worst(X, t):
    """Per-channel worst normalised distance over the gated channels (the
    uniformity gate), for a (n, nsamp*nch) block."""
    nsamp, nch = t["shape"]
    Xc = X.reshape(len(X), nsamp, nch)
    T = t["flat"].reshape(nsamp, nch)
    wch = np.abs(T)
    worst = np.zeros(len(X))
    for c in np.flatnonzero(t["gate"]):
        d = Xc[:, :, c] - T[:, c]
        qc = (d * d) @ wch[:, c]
        worst = np.maximum(worst, np.sqrt(qc / max(t["Wc"][c], 1e-30)) / t["denomc"][c])
    return worst


def strip_pass(get_waves, labels, idx, *, max_dist=0.5, gmin=0.0, gmax=10.0,
               chan_uniform=0.0, margin=0.85, tpl_cap=1024,
               min_tpl=8, exclude=(0,), block=20000, rng=None, log=None):
    """One template-strip pass over the spikes `idx` (absolute indices; one
    chunk).  Returns (labels, n_moved): labels is a full-length copy with the
    moved spikes reassigned to the claiming template's cluster."""
    rng = rng or np.random.default_rng(0)
    labels = np.asarray(labels).copy()
    lab = labels[idx]
    excl = {int(e) for e in exclude}
    clusters = [int(c) for c in np.unique(lab) if int(c) not in excl]
    if len(clusters) < 2:
        return labels, 0

    terms = {}
    for c in clusters:
        ci = idx[lab == c]
        if ci.size < min_tpl:
            continue
        t = _tpl_terms(_template(get_waves, ci, tpl_cap, rng))
        if t is not None:
            terms[c] = t
    if len(terms) < 1:
        return labels, 0

    # Every spike against EVERY template in one two-matmul pass per block:
    # A2 = x²·w and B = x·(wT) per template give q = A2 − 2B + E, so the whole
    # (block × templates) distance matrix costs the same as one donor loop —
    # and the spike's own column IS the margin denominator D_own.
    tids = sorted(terms)
    col = {c: i for i, c in enumerate(tids)}
    Wmat = np.stack([terms[c]["w"] for c in tids])              # (C, P)
    WTm = np.stack([terms[c]["wT"] for c in tids])              # (C, P)
    Ev = np.array([terms[c]["E"] for c in tids])
    Wv = np.array([terms[c]["W"] for c in tids])
    Dv = np.array([terms[c]["denom"] for c in tids])

    best_d = np.full(idx.size, np.inf)
    best_t = np.full(idx.size, -1, int)
    own_col = np.array([col.get(int(c), -1) for c in lab])
    for s in range(0, idx.size, block):
        r = np.arange(s, min(s + block, idx.size))
        X = np.asarray(get_waves(idx[r]), float).reshape(r.size, -1)
        A2 = (X * X) @ Wmat.T
        B = X @ WTm.T
        q = np.maximum(A2 - 2.0 * B + Ev[None, :], 0.0)
        D = np.sqrt(q / Wv[None, :]) / Dv[None, :]
        g = B / Ev[None, :]
        oc = own_col[r]
        has_own = oc >= 0
        d_own = np.full(r.size, np.inf)
        d_own[has_own] = D[np.flatnonzero(has_own), oc[has_own]]
        ok = (max_dist >= D) & (g >= gmin) & (g <= gmax) & (margin * d_own[:, None] >= D)
        ok[np.flatnonzero(has_own), oc[has_own]] = False        # own column never claims
        if chan_uniform > 0.0 and ok.any():
            for ci in range(len(tids)):
                cand = np.flatnonzero(ok[:, ci])
                if cand.size:
                    ok[cand, ci] &= _chan_worst(X[cand], terms[tids[ci]]) <= chan_uniform
        Dm = np.where(ok, D, np.inf)
        bi = Dm.argmin(1)
        bd = Dm[np.arange(r.size), bi]
        win = bd < best_d[r]
        best_d[r[win]] = bd[win]
        best_t[r[win]] = np.array(tids)[bi[win]]
    moved = best_t >= 0
    labels[idx[moved]] = best_t[moved]
    if log and moved.any():
        log(f"strip: {int(moved.sum())} spike(s) reassigned across "
            f"{len(np.unique(best_t[moved]))} template(s)")
    return labels, int(moved.sum())


# ── the fiber-refine knn-peel, on a chunk subset with arbitrary label ids ───
def knn_pass(get_waves, labels, idx, *, k=20, thr=0.3, minref=50, minnew=30,
             fold_thr=0.9, scorr=1.0, off_thr=None, dims=16, exclude=(0,),
             next_id=None, pca_cap=50000, rng=None, log=None):
    """One knn-peel pass (fiber_refine._knn_apply) over the spikes `idx`.
    Returns (labels, n_moved, n_new, next_id): peeled buckets that fold move
    to their winner; buckets distinct from both sides become NEW clusters,
    numbered from `next_id` (default max(labels)+1)."""
    from sklearn.decomposition import PCA
    rng = rng or np.random.default_rng(0)
    labels = np.asarray(labels).copy()
    if next_id is None:
        next_id = int(labels.max()) + 1
    excl = {int(e) for e in exclude}
    inc = np.flatnonzero(~np.isin(labels[idx], list(excl)))
    if inc.size <= max(2 * k, minref):
        return labels, 0, 0, next_id
    sub = idx[inc]
    lab_orig = labels[sub]
    ids = np.unique(lab_orig)
    dense = {int(c): i for i, c in enumerate(ids)}
    lab = np.array([dense[int(c)] for c in lab_orig], int)

    waves = np.asarray(get_waves(sub), np.float32)
    rw = fl.realign(waves)
    X = rw.reshape(len(rw), -1)
    X = X - X.mean(0)
    fit = X if len(X) <= pca_cap else X[rng.choice(len(X), pca_cap, replace=False)]
    p = PCA(n_components=min(dims, X.shape[1]), random_state=0).fit(fit)
    F = p.transform(X)

    new, fo, ke = _knn_apply(lab, F, rw, None, None, k, thr, minref, minnew,
                             fold_thr, scorr=scorr, off_thr=off_thr)
    inv = {i: int(c) for c, i in dense.items()}
    out = np.empty(len(new), int)
    fresh = {}
    for i, v in enumerate(new):
        v = int(v)
        if v in inv:
            out[i] = inv[v]
        else:                                          # a kept-distinct bucket: new cluster
            if v not in fresh:
                fresh[v] = next_id
                next_id += 1
            out[i] = fresh[v]
    n_moved = int((out != lab_orig).sum())
    labels[sub] = out
    if log and (fo or ke):
        log(f"knn: folded {fo} bucket(s), kept {ke} as new cluster(s) "
            f"({n_moved} spike(s) relabelled)")
    return labels, n_moved, len(fresh), next_id


# ── the per-chunk driver both stages call ────────────────────────────────────
def consolidate(get_waves, labels, chunk_ids, *, mode="both", exclude=(0,),
                strip_kw=None, knn_kw=None, rng=None, log=print, tag="consolidate"):
    """Run the requested passes per chunk over `labels`.  `chunk_ids` assigns
    every spike a chunk (< 0 = never touched).  Returns (labels, stats)."""
    if mode == "off":
        return np.asarray(labels), dict(strip=0, knn=0, new=0)
    rng = rng or np.random.default_rng(0)
    labels = np.asarray(labels).copy()
    chunk_ids = np.asarray(chunk_ids)
    next_id = int(labels.max()) + 1
    tot = dict(strip=0, knn=0, new=0)
    for ch in np.unique(chunk_ids[chunk_ids >= 0]):
        idx = np.flatnonzero(chunk_ids == ch)
        if mode in ("strip", "both"):
            labels, n = strip_pass(get_waves, labels, idx, exclude=exclude,
                                   rng=rng, **(strip_kw or {}))
            tot["strip"] += n
        if mode in ("knn", "both"):
            labels, n, k_new, next_id = knn_pass(get_waves, labels, idx, exclude=exclude,
                                                 next_id=next_id, rng=rng, **(knn_kw or {}))
            tot["knn"] += n
            tot["new"] += k_new
    if log:
        log(f"[{tag}] mode={mode}: strip moved {tot['strip']}, knn relabelled {tot['knn']} "
            f"({tot['new']} new cluster(s))")
    return labels, tot
