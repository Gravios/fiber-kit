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
import numpy as np

try:
    from . import config as cfgmod
except ImportError:                      # run as a script from src/fiber_kit
    import config as cfgmod

try:
    from . import fiber_chunks as _fch
except ImportError:
    import fiber_chunks as _fch
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
    "FK_CONS_SHED_K": ("cons_shed_k", float, 3.0),
    "FK_CONS_SHED_MIN": ("cons_shed_min", int, 40),
    "FK_CONS_KNN_K": ("cons_knn_k", int, 20),
    "FK_CONS_KNN_THR": ("cons_knn_thr", float, 0.3),
    "FK_CONS_KNN_MINREF": ("cons_knn_minref", int, 50),
    "FK_CONS_KNN_MINNEW": ("cons_knn_minnew", int, 30),
    "FK_CONS_KNN_DIMS": ("cons_knn_dims", int, 16),
    "FK_CONS_FOLD_THR": ("cons_fold_thr", float, 0.9),
    "FK_CONS_SCORR": ("cons_scorr", float, 0.93),
    "FK_CONS_OFF_THR": ("cons_off_thr", float, 0.0),
    # The gain-free residual strip (resid_pass).  Off unless a tail multiplier is set;
    # independent of --cons-mode, so it composes with strip/knn/shed or runs alone.
    "FK_CONS_RESID_K_ON": ("cons_resid_k_on", float, 0.0),
    "FK_CONS_RESID_K_OFF": ("cons_resid_k_off", float, 0.0),
    "FK_CONS_RESID_WIN": ("cons_resid_win", int, 0),
    "FK_CONS_RESID_MINGAIN": ("cons_resid_mingain", float, 0.25),
    "FK_CONS_RESID_MIN": ("cons_resid_min", int, 200),
    "FK_CONS_RESID_IQR_MAX": ("cons_resid_iqr_max", float, 0.0),
    "FK_CONS_RESID_IQR_MIN": ("cons_resid_iqr_min", float, 0.0),
    "FK_CONS_RESID_ALIGN": ("cons_resid_align", str, "chunk"),
}


def _knob_default(name, typ, fallback, gcfg):
    """FK_* env var, then the $FK_CONFIG yaml, then the default (config.knob_value: one precedence
    and one validation for every stage)."""
    return cfgmod.knob_value(name, typ, fallback, gcfg)


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
            if str(d).strip().lower() in ("on", "true", "yes", "1"):
                d = "both"                                 # "on" = every pass (as YAML's boolean on used to mean)
            d = cfgmod.coerce_knob(label, d, str, ("off", "strip", "knn", "both", "shed"))
            g.add_argument("--cons-mode", dest=dest, choices=("off", "strip", "knn", "both", "shed"),
                           default=d, help=f"{label}: which consolidation passes run (default {d})")
        elif dest == "cons_resid_align":
            # Validated like cons_mode: a frame outside the set stops the stage
            # with the knob's name rather than silently scoring the wrong one.
            d = cfgmod.coerce_knob(name, d, str, ("chunk", "cluster", "stored"))
            g.add_argument("--cons-resid-align", dest=dest,
                           choices=("chunk", "cluster", "stored"), default=d,
                           help=f"{name}: which alignment the residual strip scores in "
                                f"(default {d})")
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
                    agg_vac=a.cons_agg_vac, tpl_align=bool(a.cons_tpl_align),
                    shed_k=a.cons_shed_k, shed_min=a.cons_shed_min,
                    resid_k_on=a.cons_resid_k_on, resid_k_off=a.cons_resid_k_off,
                    resid_win=a.cons_resid_win, resid_mingain=a.cons_resid_mingain,
                    resid_min=a.cons_resid_min, resid_iqr_max=a.cons_resid_iqr_max,
                    resid_iqr_min=a.cons_resid_iqr_min,
                    resid_align=a.cons_resid_align)
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
        next_id = _fch.next_free_id(labels)                 # below the reserved block (fiber_chunks)
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
# ── the gain-free residual strip, on/off-channel split ───────────────────────
def _resid_taper(T, win):
    """Squared Hamming taper centred on the template's own peak sample.

    win > 0 is its width in samples, win == 0 means half the record, win < 0
    disables the taper.  SQUARED because the taper is an AMPLITUDE weight and
    every form it enters below (h x T, h T^2, h r^2) is quadratic -- storing
    h_amp^2 once is the same statistic as tapering x and T and then squaring,
    which is what the offline measurement did.  Channel-free by construction,
    unlike the |T| kernel the strip and shed use: the whole point is a weight
    that does NOT vanish off-template, since that is where a spatially
    displaced contaminant shows itself."""
    S = int(T.shape[0])
    if win < 0:
        return np.ones(S)
    width = int(win) if win > 0 else max(1, S // 2)
    tpk = int((T ** 2).sum(1).argmax())
    half = max(1, width // 2)
    d = np.arange(S) - tpk
    h = np.zeros(S)
    m = np.abs(d) <= half
    amp = 0.54 - 0.46 * np.cos(2.0 * np.pi * (d[m] + half) / (2.0 * half))
    h[m] = amp * amp
    return h


def resid_terms(W, T, win=0, mingain=0.25):
    """Per-spike on- and off-channel residual, with the gain divided out.

    The shed metric is a gain-blind distance under a |T| point kernel, so
    amplitude counts as distance and the off-template channels carry almost no
    weight.  This fits a per-spike gain on the template's own channels, takes
    r = x - a*T, and splits the residual's power by how much of the template
    each channel carries:

      E_c  = sum_t h T^2 ;  u_c = E_c / max_c E_c ;  Eon = sum_c u_c E_c
      a    = sum_c u_c sum_t h x T / Eon      (fitted ON-channel only, so an
             intruder's off-channel energy cannot inflate it and erase its own
             off-residual)
      q_c  = sum_t h (x - a T)^2
      onr  = sum_c u_c       q_c / (a^2 Eon)
      offr = sum_c (1 - u_c) q_c / (a^2 Eon)

    Both dimensionless and gain-free; 1 = residual as large as the template.
    u_c WEIGHTS rather than splits, so a flank channel at half the energy feeds
    both terms in proportion.  offr is normalised by the ON-channel energy on
    purpose -- it asks how much signal sits where this cell has none, relative
    to this cell's own size -- so it does not blow up on a quiet channel.

    Spikes whose gain falls below `mingain` carry no template-shaped signal and
    cannot be normalised by a^2; `ok` marks the rest.  That guard is
    load-bearing: on the validated unit a runs down through zero and the scores
    of near-zero-gain spikes reach 2e+06, which would set the MAD by themselves.

    Returns None when the template is flat on these channels."""
    h = _resid_taper(T, win)
    Ec = (h[:, None] * T ** 2).sum(0)
    if not Ec.max() > 0:
        return None
    u = Ec / Ec.max()
    Eon = float((u * Ec).sum())
    if not Eon > 0:
        return None
    nsp = int(W.shape[0])
    onr = np.empty(nsp)
    offr = np.empty(nsp)
    av = np.empty(nsp)
    for b0 in range(0, nsp, 20000):                 # bounded working set on a big cluster
        Wb = np.asarray(W[b0:b0 + 20000], float)
        a = (h[None, :, None] * Wb * T[None] * u[None, None, :]).sum((1, 2)) / Eon
        r = Wb - a[:, None, None] * T[None]
        q = (h[None, :, None] * r ** 2).sum(1)
        nrm = np.maximum(a ** 2 * Eon, 1e-300)
        onr[b0:b0 + len(a)] = (q * u[None, :]).sum(1) / nrm
        offr[b0:b0 + len(a)] = (q * (1.0 - u)[None, :]).sum(1) / nrm
        av[b0:b0 + len(a)] = a
    return dict(onr=onr, offr=offr, a=av, ok=av >= float(mingain), u=u, Eon=Eon,
                offmargin=float((1.0 - u).sum()))


def _resid_tail(v, k):
    """Robust one-sided upper tail, median + k*1.4826*MAD.  The terms are
    non-negative and right-skewed, so a symmetric sigma would misread them
    (the same cut shed uses).  A zero-MAD distribution has no tail: inf."""
    if not len(v):
        return np.inf
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    return med + k * 1.4826 * mad if mad > 0 else np.inf


def resid_pass(get_waves, labels, idx, *, k_on=3.0, k_off=3.0, win=0, mingain=0.25,
               minspk=200, iqr_max=0.0, iqr_min=0.0, align="chunk", raw_waves=None,
               tpl_cap=1024, exclude=(0,), rng=None, log=None):
    """Expel each cluster's gain-free residual outliers to the reserve (label 1).

    The interactive twin of Klusters' "Strip Residual Outliers", and a sibling
    of shed rather than a replacement: shed stays calibrated to FK_CONS_SHED_K,
    so the two can be run on the same clusters and compared.  Per cluster: its
    OWN median template, resid_terms, then a SEPARATE robust tail per term with
    the flagged sets UNIONED.  Separate because the terms are complementary --
    on the validated unit one sibling separates on shape (on-channel AUC 0.943)
    and another on position (off-channel 0.786 while amplitude alone is blind at
    0.522) -- and their SUM scores 0.869, below the on-channel term alone.

    THE IQR GATE, AND WHY ITS DEFAULT IS OFF.  `iqr_max` admits a cluster only
    while the IQR of its own matched gain (the same statistic `pure_iqr` gates
    the claim on) is at or below it; `iqr_min` admits it only above.  Measured
    over 22 clusters in two verified 12-min windows of jg05-20120312 group 6
    (t 24-36 and t 228-240 min), scored against the curator's own file:

      gate              clusters  parked  precision  recall  keepers lost
      none (all)              22   10148       0.49    0.29          9.5%
      giqr <= 0.25            12    6119       0.27    0.15          9.2%
      giqr <= 0.20            11    6083       0.27    0.15          9.2%

    The ceiling HALVES precision, and the reason is structural, not a bad
    threshold.  It rejects the clusters the strip is informative on -- 104 at
    57% contamination runs at precision 0.94 / recall 0.85, 2 at 78% at 0.99 --
    and admits the clusters the curator left clean, where there is nothing to
    remove and the strip parks 6-14% of good spikes for precision 0.00 (cluster
    4: 1,354 of 9,673 parked, precision 0.00, 14.0% of keepers lost).  Read as
    lift over each cluster's own base rate the strip is most useful in the
    middle band, 15-40% contamination (lift 1.6-3.0), and purely costly at 0%.

    So the gate that matches the measurement is a FLOOR, not a ceiling -- and
    `iqr_min` is how to set one.  Both default to 0 (off) because the clusters
    the curator left at 0.0% may be clean or may be untouched, and that
    confound decides how hard a floor is justified; the numbers are here so the
    choice is informed rather than inherited.

    Of six label-free candidates the gain IQR was the best predictor of the
    curator's contamination (Spearman rho 0.743; shape IQR 0.600, median onr
    0.589, IQR of onr 0.666, median offr 0.680, no-gain fraction 0.643), which
    is why the gate reuses it instead of adding a statistic.  It has one blind
    spot that no candidate covered: a near-even mixture of two cells of similar
    amplitude AND similar shape.  Cluster 3 at 53% contamination scores giqr
    0.188, inside the band of clusters at 0.0% -- and 53% is past the per-point
    median's breakdown point, where the template is no longer the dominant
    cell's.  A residual-covariance test was tried for that case and refuted
    (rho -0.361: a CLEAN unit's residual is dominated by alignment jitter, so
    structure does not mark a mixture).  The protection for it is a seeded
    template, not a gate.

    THE FRAME (`align`).  consolidate realigns every spike of a chunk ONCE, to
    the chunk MIXTURE's median, and hands that frame to every pass; `align`
    chooses what this pass scores in.  Measured on the same 22 clusters, where
    the stored alignment is the curator's own per-cluster committed realign:

      align     clusters  parked  precision  recall  keepers lost
      chunk           22   10504       0.43    0.27         11.0%
      cluster         22   10319       0.43    0.26         10.8%
      stored          22   10148       0.49    0.29          9.5%

    Realigning does not help and slightly hurts, for a reason that is the same
    one the template and the threshold already ran into: a contaminated
    cluster's own median is a POLLUTED alignment reference, so aligning to it
    pulls the contaminant into the cell's frame and shrinks the residual that
    would have exposed it.  Per cluster the cost lands exactly where the strip
    was working -- cluster 3 drops 0.80 -> 0.63 precision, 11 drops 0.61 ->
    0.31, 184 drops 0.27 -> 0.08.  A timing offset against the template is
    INFORMATION about contamination, and a self-derived realign spends it.

    CONFOUND, stated because it decides the default: "stored" in that table is
    not an unaligned session.  It is this session's curator-committed alignment,
    done per cluster AFTER splitting -- a good frame no fresh pipeline run has.
    So the table shows that a well-seeded alignment beats a self-derived one; it
    does NOT show that skipping the realign helps a raw session, which is not
    measured here.  The default therefore stays "chunk", matching shed and every
    other pass, and `align="stored"` is the lever for a session whose .spk
    already carries a curated realign (set FK_CONS_RESID_ALIGN=stored).
    `align="cluster"` realigns each cluster to its own median first, both
    template and scored spikes in one frame, and is the arm the table shows
    buying nothing.

    Parked spikes go to the reserve, so this is park-and-review, never
    deletion.  Returns (labels, stats)."""
    labels = np.asarray(labels).copy()
    idx = np.asarray(idx)
    rng = rng or np.random.default_rng(0)
    tot = dict(resid=0, resid_on=0, resid_off=0, resid_both=0, resid_nogain=0,
               resid_clusters=0, resid_gated=0)
    if k_on <= 0.0 and k_off <= 0.0:
        return labels, tot
    for c in np.unique(labels[idx]):
        if c <= 0 or c == 1 or c in exclude:
            continue
        ci = idx[labels[idx] == c]
        if len(ci) < int(minspk):
            continue
        if align == "stored" and raw_waves is not None:
            W = np.asarray(raw_waves(ci), float)        # bypass the chunk realign
        elif align == "cluster":
            W = np.asarray(fl.realign(np.asarray(get_waves(ci), float)), float)
        else:
            W = np.asarray(get_waves(ci), float)        # the chunk mixture frame
        # The template comes from the SAME array, so it and the spikes it scores
        # are always in one frame -- unlike strip's tpl_align, which sharpens the
        # template alone (harmless for its calibrated quantities, but here it
        # would inflate every residual it is subtracted from).
        step = max(1, len(W) // max(1, int(tpl_cap)))
        T = np.median(W[::step][: int(tpl_cap)], 0)
        t = resid_terms(W, T, win=win, mingain=mingain)
        if t is None:
            continue
        ok = t["ok"]
        if ok.sum() < 5:
            continue
        gq = np.quantile(t["a"][ok], (0.25, 0.75))
        giqr = float(gq[1] - gq[0])
        if (iqr_max > 0.0 and giqr > iqr_max) or (iqr_min > 0.0 and giqr < iqr_min):
            tot["resid_gated"] += 1
            if log:
                log(f"    [resid] cluster {int(c)}: giqr {giqr:.3f} outside the gate "
                    f"({iqr_min or '-'} .. {iqr_max or '-'}), left alone")
            continue
        thr_on = _resid_tail(t["onr"][ok], k_on) if k_on > 0 else np.inf
        thr_off = _resid_tail(t["offr"][ok], k_off) if k_off > 0 else np.inf
        f_on = ok & (t["onr"] > thr_on)
        f_off = ok & (t["offr"] > thr_off)
        bad = (~ok) | f_on | f_off
        tot["resid_clusters"] += 1
        if log:
            log(f"    [resid] cluster {int(c)}: {int(ok.sum())} scored, giqr {giqr:.3f}, "
                f"median on={np.median(t['onr'][ok]):.3g} off={np.median(t['offr'][ok]):.3g}, "
                f"tail on>{thr_on:.3g} off>{thr_off:.3g} -> {int(bad.sum())} parked")
        if not bad.any():
            continue
        tot["resid_nogain"] += int((~ok).sum())
        tot["resid_both"] += int((f_on & f_off).sum())
        tot["resid_on"] += int((f_on & ~f_off).sum())
        tot["resid_off"] += int((f_off & ~f_on).sum())
        tot["resid"] += int(bad.sum())
        labels[ci[bad]] = 1
    return labels, tot


def consolidate(get_waves, labels, chunk_ids, *, mode="both", exclude=(0,),
                strip_kw=None, knn_kw=None, rng=None, log=print, tag="consolidate"):
    """Run the requested passes per chunk over `labels`.  `chunk_ids` assigns
    every spike a chunk (< 0 = never touched).  Returns (labels, stats)."""
    strip_kw = dict(strip_kw or {})
    # resid_pass is independent of `mode` -- it composes with strip/knn/shed or
    # runs alone, so mode="off" with a resid tail set is a valid request and
    # must not short-circuit here.
    resid_kw = dict(k_on=float(strip_kw.pop("resid_k_on", 0.0)),
                    k_off=float(strip_kw.pop("resid_k_off", 0.0)),
                    win=int(strip_kw.pop("resid_win", 0)),
                    mingain=float(strip_kw.pop("resid_mingain", 0.25)),
                    minspk=int(strip_kw.pop("resid_min", 200)),
                    iqr_max=float(strip_kw.pop("resid_iqr_max", 0.0)),
                    iqr_min=float(strip_kw.pop("resid_iqr_min", 0.0)),
                    align=str(strip_kw.pop("resid_align", "chunk")))
    resid_on = resid_kw["k_on"] > 0.0 or resid_kw["k_off"] > 0.0
    # Per-cluster resid lines carry the sited tails, which is the only way to see
    # whether the thresholds are anywhere near the data; FK_CONS_RESID_QUIET=1
    # silences them once a config is settled.
    resid_log = not cfgmod.knob_value("FK_CONS_RESID_QUIET", int, 0, None)
    if mode == "off" and not resid_on:
        return np.asarray(labels), dict(strip=0, knn=0, new=0)
    rng = rng or np.random.default_rng(0)
    labels = np.asarray(labels).copy()
    chunk_ids = np.asarray(chunk_ids)
    next_id = _fch.next_free_id(labels)                     # below the reserved block (fiber_chunks)
    tot = dict(strip=0, knn=0, new=0)
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
    shed_k = float(strip_kw.pop("shed_k", 3.0))
    shed_min = int(strip_kw.pop("shed_min", 40))
    tot["shed"] = 0
    tot["agg"] = 0
    tot["agg_new"] = 0
    for k in ("resid", "resid_on", "resid_off", "resid_both", "resid_nogain",
              "resid_clusters", "resid_gated"):
        tot[k] = 0
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
        if mode == "shed":
            # ── the INVERSE of the strip: expel a cluster's own outliers ──
            #   The strip PULLS matching spikes in from other clusters; shed PUSHES a
            #   cluster's own worst-fitting spikes OUT, to the reserve (label 1), for a
            #   partially contaminated cluster whose junk no template claims.  Per
            #   cluster: median template (same _template as the strip), amplitude-
            #   weighted residual distance per spike, and a robust self-threshold
            #   (median + shed_k * 1.4826*MAD) -- self-relative, so the metric's scale
            #   cancels and no cross-cluster calibration is needed.  Shed spikes stay
            #   Klusters-inspectable in the reserve (the hosts park (src -> 1) buckets),
            #   so this is park-and-review, never deletion.  Measured on the lab
            #   session's linked chain: shed_k=3 bought +1.5pp purity for -1.1pp
            #   completeness (4,916 spikes reserved) -- a terminal-curation precision
            #   dial, deliberately NOT part of strip/both.
            for c in np.unique(labels[idx]):
                if c <= 0 or c == 1 or c in exclude:
                    continue
                ci = idx[labels[idx] == c]
                if len(ci) < shed_min:
                    continue
                T = _template(aw, ci, int(strip_kw.get("tpl_cap", 1024)), rng)
                w = np.abs(T)
                W = aw(ci)
                D = np.sqrt((w * (W - T) ** 2).sum((1, 2)) / (w.sum() + 1e-9))
                med = float(np.median(D))
                mad = float(np.median(np.abs(D - med))) + 1e-9
                thr = med + shed_k * 1.4826 * mad
                bad = ci[thr < D]
                if bad.size:
                    labels[bad] = 1
                    tot["shed"] += int(bad.size)

        if mode in ("knn", "both"):
            labels, n, k_new, next_id = knn_pass(aw, labels, idx, exclude=exclude,
                                                 next_id=next_id, realigned=True,
                                                 rng=rng, **(knn_kw or {}))
            tot["knn"] += n
            tot["new"] += k_new
        if resid_on:
            # LAST in the chunk, and independent of mode: the residual strip is a
            # terminal precision dial like shed, and it scores against each
            # cluster's own median, so it wants whatever membership the earlier
            # passes settled on rather than the one they started from.
            labels, rtot = resid_pass(aw, labels, idx, exclude=exclude, rng=rng,
                                      raw_waves=get_waves,
                                      tpl_cap=int(strip_kw.get("tpl_cap", 1024)),
                                      log=(log if resid_log else None), **resid_kw)
            for k, v in rtot.items():
                tot[k] = tot.get(k, 0) + v
    if log:
        agg_line = (f", dissolved {tot['agg']} into per-chunk aggregates "
                    f"({tot['agg_new']} reseeded template(s))" if tot.get("agg") else "")
        shed_line = f", shed {tot['shed']} own-outlier(s) to reserve" if tot.get("shed") else ""
        resid_line = (f", residual strip parked {tot['resid']} ({tot['resid_on']} on-channel, "
                      f"{tot['resid_off']} off-channel, {tot['resid_both']} both, "
                      f"{tot['resid_nogain']} no-gain) from {tot['resid_clusters']} cluster(s)"
                      + (f"; {tot['resid_gated']} left alone by the IQR gate"
                         if tot.get("resid_gated") else "")
                      if tot.get("resid") or tot.get("resid_gated") else "")
        log(f"[{tag}] mode={mode}: strip moved {tot['strip']}, knn relabelled {tot['knn']} "
            f"({tot['new']} new cluster(s)){agg_line}{shed_line}{resid_line}")
    return labels, tot
