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
#    spikes) claims spikes
#    of OTHER clusters whose kernel-weighted normalised residual D is at or
#    below max_dist and whose matched gain g sits in [gmin, gmax], plus the
#    optional per-channel uniformity gate — the Klusters metric verbatim
#    (klustersdoc_strip.cpp; kernel w=|T|, D = sqrt(Σw·(x−T)²/Σw)/denom with
#    denom = sqrt(Σw·T²/Σw), g = Σw·x·T / Σw·T²).  Where Klusters parks the
#    matches in a new cluster for the curator, consolidation REASSIGNS them
#    to the template's cluster, and adds the safeguard that interactive
#    review provided, with thresholds CALIBRATED FROM THE SESSION rather
#    than absolute (the anchor-link philosophy: a fixed floor tuned
#    elsewhere lands mid-distribution on new data and gates nothing — or
#    everything).  Measured on the reference session, D to a cell's OWN
#    template has median 0.54: on an over-split sort, half of a low-SNR
#    cell's own spikes sit past any fixed "misfit" floor, so absolute
#    thresholds conflate noise level with misfit and reshuffled a third of
#    the session between near-siblings.  Calibrated, both sides are
#    quantiles of each cluster's own-spike distance distribution:
#      * strippable  — D_own ≥ that cluster's own_q quantile (default 0.90):
#        the spike fits its home worse than 90% of its peers, a ≤10%
#        per-cell churn budget BY CONSTRUCTION;
#      * claimable   — D_target ≤ tgt_scale × the TARGET's tgt_q quantile
#        (defaults 1.25 × 0.90, capped at max_dist): the spike fits the
#        target as well as the target's own spikes fit it, with a small
#        out-of-sample allowance (the template is built FROM the target's
#        spikes, so its in-sample quantile understates where fresh members
#        of the same population land);
#    plus the margin (D_target ≤ margin · D_own): the claim must beat home
#    decisively, so a template cannot strip-mine a look-alike.  The last piece is the TWIN GATE: a pair of
#    clusters whose templates are SHAPE-identical at some scale (normalized
#    best-lag correlation ≥ twin_thr, fiber-refine's _ncorr — "energy
#    levels of one neuron score high") is ineligible to trade at all.  In
#    an over-split sort such pairs are one neuron's energy levels awaiting
#    a MERGE; the amplitude-sensitive D happily reshuffles a ladder cell's
#    low-energy spikes onto their amplitude-matched sibling (measured:
#    tens of thousands of moves at pair cosine ≥ 0.95), which is churn,
#    not decontamination — the same judgement the knn's scorr gate encodes
#    per bucket, applied per pair.  Cross-shape claims, the actual
#    contamination, are untouched.  When several templates claim a spike
#    the smallest D wins.  Every spike is
#    scored against every template in one two-matmul pass per block — a
#    contaminant is by definition UNLIKE its own cluster's template, so no
#    donor-level similarity prefilter can be sound.
#
#  KNN — fiber_refine._knn_apply, the library form of the Klusters KNN
#    split: per-spike K-NN majority vote in a PCA feature space against the
#    pool of other clusters' spikes; a peeled bucket folds into the winner
#    when its median waveform matches (amplitude-sensitive xcorr ≥
#    fold_thr), else it becomes a NEW cluster.  scorr defaults to 0.93 —
#    fiber-refine's own operating value for the shape-distinctness gate
#    ("same shape as its own cluster → not a contaminant, keep"), which is
#    the knn's twin guard; 1.0 disables it, and did, in the first draft.
#
#  Scope is PER CHUNK in both callers: intrachunk's units are chunk-local by
#  contract, and anchor-link's cells drift across chunks, so a cell is best
#  represented by its own within-chunk spikes rather than a drift-smeared
#  session template.  Labels in `exclude` (the reserve bins) neither donate
#  nor receive, and their spikes never move.
#
#  ALIGNMENT — the driver realigns each chunk's spikes ONCE to a common
#  reference (fl.realign, the same call fiber-refine's feature build makes on
#  its own mixed pools) and both passes template and score in that frame.
#  Klusters scores stored records because its curation axis is realigned on
#  disk; the pipeline's standard .spk is only loosely stored-aligned —
#  anchor-link realigns before templating for exactly this reason — and at
#  stored alignment the per-spike jitter inflates D_own past any sane floor
#  (measured on the reference session: 29% of spikes moved on jitter alone;
#  aligned, the pass is surgical).
#
#  Knobs read FK_CONS_* (CLI > FK_CONS_* env > global yaml where the caller
#  passes one > default).
# ════════════════════════════════════════════════════════════════════════════
import os
import numpy as np

try:
    from . import fiber_lib as fl
    from .fiber_refine import _knn_apply, _ncorr
except ImportError:                                   # script / flat-layout fallback
    import fiber_lib as fl
    from fiber_refine import _knn_apply, _ncorr


# ── knob resolution: default <- global yaml (FK_CONS_*) <- FK_* env <- CLI ──
_KNOBS = {
    "FK_CONS_MODE": ("cons_mode", str, "off"),
    "FK_CONS_MAX_DIST": ("cons_max_dist", float, 0.5),
    "FK_CONS_GMIN": ("cons_gmin", float, 0.0),
    "FK_CONS_GMAX": ("cons_gmax", float, 10.0),
    "FK_CONS_CHAN": ("cons_chan", float, 0.0),
    "FK_CONS_MARGIN": ("cons_margin", float, 0.85),
    "FK_CONS_OWN_Q": ("cons_own_q", float, 0.90),
    "FK_CONS_TGT_Q": ("cons_tgt_q", float, 0.90),
    "FK_CONS_TGT_SCALE": ("cons_tgt_scale", float, 1.25),
    "FK_CONS_TWIN": ("cons_twin", float, 0.93),
    "FK_CONS_TPL_CAP": ("cons_tpl_cap", int, 1024),
    "FK_CONS_MIN_TPL": ("cons_min_tpl", int, 8),
    "FK_CONS_ITERS": ("cons_iters", int, 1),
    "FK_CONS_CAP_AMP": ("cons_cap_amp", float, 0.0),
    "FK_CONS_PURE_IQR": ("cons_pure_iqr", float, 0.0),
    "FK_CONS_AGG_IQR": ("cons_agg_iqr", float, 0.0),
    "FK_CONS_AGG_EVERY": ("cons_agg_every", int, 3),
    "FK_CONS_AGG_MINNEW": ("cons_agg_minnew", int, 30),
    "FK_CONS_AGG_VAC": ("cons_agg_vac", int, 0),
    "FK_CONS_TPL_ALIGN": ("cons_tpl_align", int, 0),
    "FK_CONS_PARK_MIN": ("cons_park_min", int, 0),
    "FK_CONS_KNN_K": ("cons_knn_k", int, 20),
    "FK_CONS_KNN_THR": ("cons_knn_thr", float, 0.3),
    "FK_CONS_KNN_MINREF": ("cons_knn_minref", int, 50),
    "FK_CONS_KNN_MINNEW": ("cons_knn_minnew", int, 30),
    "FK_CONS_KNN_DIMS": ("cons_knn_dims", int, 16),
    "FK_CONS_FOLD_THR": ("cons_fold_thr", float, 0.9),
    "FK_CONS_SCORR": ("cons_scorr", float, 0.93),
    "FK_CONS_OFF_THR": ("cons_off_thr", float, 0.0),
}


def _knob_default(name, typ, fallback, gcfg):
    for src in (os.environ.get(name), (gcfg or {}).get(name)):
        if isinstance(src, bool):
            # YAML parses a bare off/on as a boolean -- read it as the mode it
            # spelled (off -> "off", on -> "both") rather than str(False).
            return ("both" if src else "off") if typ is str else typ(src)
        if src is None or str(src).strip() == "":
            continue
        try:
            return typ(src)
        except (TypeError, ValueError):
            pass
    return fallback


def add_consolidate_args(ap, gcfg=None, stage=None):
    """Attach the FK_CONS_* knob group to a stage's parser (same resolution
    order as the stage's own knobs: CLI > env > global yaml > default).

    `stage` (e.g. "SESSION", "INTRA", "ALINK") additionally lets a
    stage-scoped FK_<stage>_CONS_MODE override the shared FK_CONS_MODE
    default (scoped env > scoped yaml > shared env > shared yaml > off).
    The measured verdict is PER LAYER -- the strip purifies fiber-session's
    over-split output but contradicts a curated-grade linked sort -- so a
    plan must be able to enable the pass in one host stage without enabling
    it in the others.  The threshold knobs stay shared: they describe the
    metric, not the host."""
    g = ap.add_argument_group("consolidation (per-spike strip + knn cleanup of the final labels)")
    for name, (dest, typ, fb) in _KNOBS.items():
        d = _knob_default(name, typ, fb, gcfg)
        if dest == "cons_mode":
            label = name
            if stage:
                label = f"FK_{stage}_CONS_MODE > {name}"
                d = _knob_default(f"FK_{stage}_CONS_MODE", typ, d, gcfg)
            g.add_argument("--cons-mode", dest=dest, choices=("off", "strip", "knn", "both"),
                           default=d, help=f"{label}: which consolidation passes run (default {d})")
        else:
            g.add_argument("--" + dest.replace("_", "-"), dest=dest, type=typ, default=d,
                           help=f"{name} (default {d})")


def kwargs_from_args(a):
    """(mode, strip_kw, knn_kw) from a parsed namespace add_consolidate_args filled."""
    strip_kw = dict(max_dist=a.cons_max_dist, gmin=a.cons_gmin, gmax=a.cons_gmax,
                    chan_uniform=a.cons_chan, margin=a.cons_margin,
                    own_q=a.cons_own_q, tgt_q=a.cons_tgt_q, tgt_scale=a.cons_tgt_scale,
                    twin_thr=(a.cons_twin if a.cons_twin > 0 else None),
                    tpl_cap=a.cons_tpl_cap, min_tpl=a.cons_min_tpl,
                    iters=a.cons_iters, cap_amp=a.cons_cap_amp,
                    pure_iqr=a.cons_pure_iqr, agg_iqr=a.cons_agg_iqr,
                    agg_every=a.cons_agg_every, agg_minnew=a.cons_agg_minnew,
                    agg_vac=a.cons_agg_vac, tpl_align=bool(a.cons_tpl_align))
    knn_kw = dict(k=a.cons_knn_k, thr=a.cons_knn_thr, minref=a.cons_knn_minref,
                  minnew=a.cons_knn_minnew, dims=a.cons_knn_dims, fold_thr=a.cons_fold_thr,
                  scorr=a.cons_scorr, off_thr=(a.cons_off_thr if a.cons_off_thr > 0 else None))
    return a.cons_mode, strip_kw, knn_kw


# ── the Klusters strip metric, vectorised ────────────────────────────────────
def _template(get_waves, idx, tpl_cap, rng, tpl_align=False):
    """Median template over ≤ tpl_cap evenly-strided spikes.

    tpl_align=False scores stored alignment, exactly as Klusters does (in the
    driver that means the shared per-chunk mixture frame).  tpl_align=True
    additionally SELF-ALIGNS the subset to its own median first — the
    curator's realign-the-cluster-then-strip step: a cluster whose spikes
    carry a consistent lag against the chunk mixture gets a sharper template
    in its own frame.  The systematic frame offset this introduces cancels in
    every calibrated quantity (d_own, floor, radius and margin all derive
    from distances to the SAME template), so only the sharpness gain acts."""
    idx = np.asarray(idx)
    step = max(1, idx.size // int(tpl_cap))
    w = np.asarray(get_waves(idx[::step][: int(tpl_cap)]), float)
    if tpl_align and len(w) > 2:
        w = fl.realign(w, iters=2)
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
               chan_uniform=0.0, margin=0.85, own_q=0.90, tgt_q=0.90,
               tgt_scale=1.25, twin_thr=0.93, tpl_cap=1024, min_tpl=8, exclude=(0,),
               cap_amp=0.0, pure_iqr=0.0, no_tpl=(), stats=None, tpl_align=False,
               block=20000, rng=None, log=None):
    """One template-strip pass over the spikes `idx` (absolute indices; one
    chunk).  Returns (labels, n_moved): labels is a full-length copy with the
    moved spikes reassigned to the claiming template's cluster.  Thresholds
    are per-cluster quantiles of each cluster's own-spike distances (see the
    module preamble); max_dist survives as the hard cap on any claim.

    cap_amp makes that cap AMPLITUDE-AWARE: D is normalized by each template's
    kernel RMS (denom), so with additive noise D_own scales as ~1/amplitude and
    one fixed cap is a different absolute-residual bound per cluster — tight on
    faint cells, loose on big ones.  cap_c = max_dist * (denom_med/denom_c)
    ** cap_amp refers every cluster's cap to the chunk's median template
    amplitude: 0 keeps the fixed cap, 1 caps a constant ABSOLUTE residual.

    pure_iqr (>0) is the curator's only-strip-with-pure-clusters rule: a
    cluster may CLAIM only while the IQR of its own spikes' matched gain g is
    at or below this — a mixture's or energy ladder's own-gain spread is wide
    where a single unit's is tight (the best GT-purity predictor measured,
    AUC 0.84), and an impure claimant is doubly wrong: its template is a
    mixture and its inflated own-distance quantiles hand it an oversized
    radius.  Gated clusters remain strippable FROM, and under iteration the
    score is recomputed every round, so a cluster earns claiming rights as it
    purifies.  0 = every templated cluster claims.

    no_tpl ids never template (so never claim, and carry no churn floor) but
    their spikes remain CLAIMABLE — the aggregate-pool contract.  exclude ids
    are fully inert both ways: they neither template nor lose spikes.
    `stats`, when a dict, is filled with per-templated-cluster diagnostics
    ("giqr": own-gain IQR) for the driver's dissolve decision."""
    rng = rng or np.random.default_rng(0)
    labels = np.asarray(labels).copy()
    lab = labels[idx]
    excl = {int(e) for e in exclude}
    ntpl = {int(e) for e in no_tpl}
    pooled = np.isin(lab, sorted(ntpl)) if ntpl else np.zeros(idx.size, bool)
    clusters = [int(c) for c in np.unique(lab)
                if int(c) not in excl and int(c) not in ntpl]
    if len(clusters) < 2 and not ntpl:
        return labels, 0

    terms = {}
    for c in clusters:
        ci = idx[lab == c]
        if ci.size < min_tpl:
            continue
        t = _tpl_terms(_template(get_waves, ci, tpl_cap, rng, tpl_align=tpl_align))
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
    # Twin gate: pair eligibility by SHAPE (normalized best-lag correlation of
    # the medians) — see the module preamble.  True = the pair may trade.
    C = len(tids)
    pair_ok = np.ones((C + 1, C), bool)                # row C = spikes with no own template
    if twin_thr is not None:
        meds = [terms[c]["flat"].reshape(terms[c]["shape"]) for c in tids]
        for i in range(C):
            for j in range(C):
                if i != j and _ncorr(meds[i], meds[j]) >= twin_thr:
                    pair_ok[i, j] = False
    Wmat = np.stack([terms[c]["w"] for c in tids])              # (C, P)
    WTm = np.stack([terms[c]["wT"] for c in tids])              # (C, P)
    Ev = np.array([terms[c]["E"] for c in tids])
    Wv = np.array([terms[c]["W"] for c in tids])
    Dv = np.array([terms[c]["denom"] for c in tids])

    best_d = np.full(idx.size, np.inf)
    best_t = np.full(idx.size, -1, int)
    own_col = np.array([col.get(int(c), -1) for c in lab])
    movable = ~np.isin(lab, sorted(excl))              # excluded ids never lose spikes

    # Pass 1 — every spike's distance to its OWN template, then the
    # per-cluster calibration quantiles (see the module preamble).
    d_own_all = np.full(idx.size, np.inf)
    g_own_all = np.full(idx.size, np.nan)
    for s in range(0, idx.size, block):
        r = np.arange(s, min(s + block, idx.size))
        oc = own_col[r]
        has_own = np.flatnonzero(oc >= 0)
        if has_own.size == 0:
            continue
        X = np.asarray(get_waves(idx[r[has_own]]), float).reshape(has_own.size, -1)
        cols = oc[has_own]
        A2 = np.einsum("ij,ij->i", X * X, Wmat[cols])
        Bo = np.einsum("ij,ij->i", X, WTm[cols])
        qo = np.maximum(A2 - 2.0 * Bo + Ev[cols], 0.0)
        d_own_all[r[has_own]] = np.sqrt(qo / Wv[cols]) / Dv[cols]
        g_own_all[r[has_own]] = Bo / Ev[cols]
    floor_c = np.full(C, np.inf)                       # own_q quantile per cluster
    radius_c = np.zeros(C)                             # tgt_q quantile per cluster, capped
    cap_c = np.full(C, float(max_dist))
    if cap_amp > 0.0 and C > 1:
        cap_c = float(max_dist) * (float(np.median(Dv)) / Dv) ** float(cap_amp)
    claim_ok = np.ones(C, bool)
    for c, i in col.items():
        dc = d_own_all[lab == c]
        dc = dc[np.isfinite(dc)]
        if dc.size:
            floor_c[i] = float(np.quantile(dc, own_q))
            radius_c[i] = min(tgt_scale * float(np.quantile(dc, tgt_q)), float(cap_c[i]))
        if pure_iqr > 0.0 or stats is not None:
            gc = g_own_all[lab == c]
            gc = gc[np.isfinite(gc)]
            iq = float(np.quantile(gc, 0.75) - np.quantile(gc, 0.25)) if gc.size else float("inf")
            if stats is not None:
                stats.setdefault("giqr", {})[int(c)] = iq
            if pure_iqr > 0.0:
                claim_ok[i] = bool(gc.size) and iq <= pure_iqr

    # Pass 2 — claims.
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
        d_own = d_own_all[r]
        own_floor = np.where(has_own, floor_c[np.maximum(oc, 0)], 0.0)
        ok = ((radius_c[None, :] >= D) & (g >= gmin) & (g <= gmax)
              & (margin * d_own[:, None] >= D) & (d_own >= own_floor)[:, None])
        ok &= claim_ok[None, :]                                 # only pure clusters claim
        ok &= pair_ok[np.where(has_own, oc, C)]                 # twin pairs never trade
        ok[~movable[r]] = False                                 # excluded spikes sit out
        pl = np.flatnonzero(pooled[r])
        if pl.size:
            # Pool spikes have no own template, so the own-distance margin
            # cannot protect them — replace it with a CONTRAST margin: the
            # best template must beat the runner-up by the same factor
            # (best D <= margin * second-best D over the eligible columns),
            # so a pool spike leaves only for an UNAMBIGUOUS home.  A single
            # eligible column passes (runner-up = inf).
            Dp = np.where(ok[pl], D[pl], np.inf)
            if Dp.shape[1] >= 2:
                two = np.partition(Dp, 1, axis=1)[:, :2]
                bad = two[:, 0] > margin * two[:, 1]
                ok[pl[bad]] = False
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


def _agg_recluster(get_waves, labels, idx, agg_id, next_id, minnew, rng,
                   promote_iqr=0.0, cap=6000, dims=10, log=None):
    """KlustaKwik over the chunk's aggregate pool: components with >= minnew
    member spikes become NEW clusters.  kk's noise bucket and sub-minnew
    components stay pooled.  Only a <= cap evenly-strided subsample is
    clustered — the recluster's job is to FIND structure, not to assign
    every spike.

    promote_iqr (>0; the driver passes the claim gate's pure_iqr when the
    VACUUM is on) additionally holds components to the claimants' purity
    bar — own-gain IQR against the component's own median template — and
    failed components stay pooled, where the vacuum keeps mining them.
    Measured on g6 pools: ungated promotion mints ~0.43-GT-purity mixture
    parents; gated, the few survivors run ~0.75.  Under QUARANTINE the
    driver leaves it 0: there the components ARE the deliverable —
    structured curation parents in place of dozens of dissolved husks — and
    a mixture component cannot strip anyway (the claim gate).  Returns
    (labels, next_id, new_ids)."""
    ai = idx[labels[idx] == agg_id]
    if ai.size < max(2 * minnew, 60):
        return labels, next_id, []
    sub = ai[:: max(1, ai.size // int(cap))][: int(cap)]
    W = np.asarray(get_waves(sub), float).reshape(sub.size, -1)
    Wc = W - W.mean(0)
    k = min(int(dims), Wc.shape[1], Wc.shape[0] - 1)
    F = Wc @ np.linalg.svd(Wc, full_matrices=False)[2][:k].T
    try:
        from .klustakwik import klustakwik as _kk
    except ImportError:
        from klustakwik import klustakwik as _kk
    kmax = int(np.clip(sub.size // (2 * max(minnew, 1)), 2, 20))
    lab_kk = _kk(F, max_clusters=kmax, min_clusters=2, splits=False,
                 verbose=False, seed=int(rng.integers(2 ** 31)))
    new_ids = []
    for c in np.unique(lab_kk):
        if c == 0:
            continue                                   # kk noise stays pooled
        m = lab_kk == c
        if int(m.sum()) < int(minnew):
            continue
        if promote_iqr > 0.0:
            t = _tpl_terms(np.median(np.asarray(get_waves(sub[m]), float), 0))
            if t is None:
                continue
            _, gg = _score_block(W[m], t)
            if float(np.quantile(gg, 0.75) - np.quantile(gg, 0.25)) > promote_iqr:
                continue                               # a mixture is not a template
        labels[sub[m]] = next_id
        new_ids.append(next_id)
        next_id += 1
    if log and new_ids:
        log(f"aggregate recluster: {len(new_ids)} component(s) from "
            f"{ai.size} pooled spike(s)")
    return labels, next_id, new_ids


# ── the fiber-refine knn-peel, on a chunk subset with arbitrary label ids ───
def knn_pass(get_waves, labels, idx, *, k=20, thr=0.3, minref=50, minnew=30,
             fold_thr=0.9, scorr=0.93, off_thr=None, dims=16, exclude=(0,),
             next_id=None, pca_cap=50000, realigned=False, rng=None, log=None):
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
    rw = waves if realigned else fl.realign(waves)
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
    strip_kw = dict(strip_kw or {})
    # The curator's strip is ITERATIVE: strip, re-template on the result,
    # adjust the distance, strip again.  iters automates that loop — every
    # round refits the templates AND the calibration quantiles on the updated
    # membership, so each cluster's floor and radius tighten as it purifies
    # (the per-cluster, amplitude-aware "adjusting the distance").  Rounds
    # stop early once a chunk trades nothing.
    strip_iters = max(1, int(strip_kw.pop("iters", 1)))
    # The AGGREGATE pool (agg_iqr > 0): after each round, clusters whose
    # remainder is hopeless — own-gain IQR above agg_iqr, a mixture no pure
    # claimant can finish pulling apart — dissolve into ONE per-chunk
    # aggregate cluster; every agg_every rounds (and once more after the
    # loop) KlustaKwik runs over a pool subsample and its components become
    # new parents: structured curation material in place of dozens of husks,
    # occasionally a recovered template.  Pool-born parents are never
    # re-dissolved (one way through the pool, no oscillation).  By default
    # the pool is a QUARANTINE — its spikes are not claimable — because the
    # vacuum measurably costs precision (agg_vac=1 turns it on: the pool
    # stops templating but its spikes become claimable, protected by a
    # runner-up CONTRAST margin in place of the own-distance margin they
    # no longer have).
    agg_iqr = float(strip_kw.pop("agg_iqr", 0.0))
    agg_every = max(1, int(strip_kw.pop("agg_every", 3)))
    agg_minnew = int(strip_kw.pop("agg_minnew", 30))
    agg_vac = int(strip_kw.pop("agg_vac", 0))
    tot["agg"] = 0
    tot["agg_new"] = 0
    for ch in np.unique(chunk_ids[chunk_ids >= 0]):
        idx = np.flatnonzero(chunk_ids == ch)
        # One realignment per chunk, shared by both passes (see ALIGNMENT above).
        aligned = fl.realign(np.asarray(get_waves(idx), float)).astype(np.float32)

        def aw(ix, _idx=idx, _al=aligned):
            return _al[np.searchsorted(_idx, np.asarray(ix))]

        if mode in ("strip", "both"):
            agg_id = -1
            born = set()
            since_recluster = 0
            for r in range(strip_iters):
                st = {} if agg_iqr > 0.0 else None
                agg_args = {}
                if agg_id > 0:
                    agg_args = (dict(no_tpl=(agg_id,)) if agg_vac
                                else dict(exclude=tuple(exclude) + (agg_id,)))
                labels, n = strip_pass(aw, labels, idx,
                                       **{**dict(exclude=exclude), **agg_args},
                                       stats=st, rng=rng, **strip_kw)
                tot["strip"] += n
                dissolved = reseeded = 0
                if agg_iqr > 0.0:
                    hopeless = [c for c, q in st.get("giqr", {}).items()
                                if q > agg_iqr and c != agg_id and c not in born]
                    if hopeless:
                        if agg_id < 0:
                            agg_id = next_id
                            next_id += 1
                        hm = np.isin(labels[idx], hopeless)
                        labels[idx[hm]] = agg_id
                        dissolved = int(hm.sum())
                        tot["agg"] += dissolved
                        since_recluster += 1
                    if agg_id > 0 and since_recluster and (r + 1) % agg_every == 0:
                        labels, next_id, k_new = _agg_recluster(
                            aw, labels, idx, agg_id, next_id, agg_minnew, rng,
                            promote_iqr=(float(strip_kw.get("pure_iqr", 0.0))
                                         if agg_vac else 0.0), log=log)
                        born.update(k_new)
                        tot["agg_new"] += len(k_new)
                        reseeded = len(k_new)
                        since_recluster = 0
                if n == 0 and dissolved == 0 and reseeded == 0:
                    break
            if agg_id > 0 and since_recluster:          # final reseed of a changed pool
                labels, next_id, k_new = _agg_recluster(
                    aw, labels, idx, agg_id, next_id, agg_minnew, rng,
                    promote_iqr=(float(strip_kw.get("pure_iqr", 0.0))
                                 if agg_vac else 0.0), log=log)
                born.update(k_new)
                tot["agg_new"] += len(k_new)
        if mode in ("knn", "both"):
            labels, n, k_new, next_id = knn_pass(aw, labels, idx, exclude=exclude,
                                                 next_id=next_id, realigned=True,
                                                 rng=rng, **(knn_kw or {}))
            tot["knn"] += n
            tot["new"] += k_new
    if log:
        agg_line = (f", dissolved {tot['agg']} into per-chunk aggregates "
                    f"({tot['agg_new']} reseeded template(s))" if tot.get("agg") else "")
        log(f"[{tag}] mode={mode}: strip moved {tot['strip']}, knn relabelled {tot['knn']} "
            f"({tot['new']} new cluster(s)){agg_line}")
    return labels, tot
