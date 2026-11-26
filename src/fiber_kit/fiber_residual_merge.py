#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════════════════════
#  fiber_residual_merge.py -- link/merge clusters by the REVERSE RESIDUAL STRIP, as
#  the same CONFIDENCE-ORDERED AGGLOMERATION fiber-xcorr-merge runs: the most
#  confident pair at/above the threshold fuses first, the merged cluster is
#  re-templated, and only then is the next pair picked.  A merge is vetoed when the
#  refractory cross-correlogram is powered yet shows no dip.  A pure clu relabel;
#  .res/.spk untouched.  Same inputs, outputs, hierarchy carry and drift table as
#  fiber-xcorr-merge, so it takes that stage's place in a plan.
#
#  WHAT IS DIFFERENT IS THE SCORE.  fiber-xcorr-merge asks whether two MEDIAN
#  waveforms look alike (max roll-shift cosine).  Two cells at one site do; so does
#  a cell and its own drifted or amplitude-scaled copy, and the cosine cannot tell
#  those two cases apart -- which is why it needs a stack of vetoes behind it
#  (band overlap, inter-channel timing, gain IQR, residual-variance uniformity, a
#  knn split).  This stage asks the question the residual strip (fiber-consolidate
#  resid_pass, Klusters' Strip Residual Outliers) asks of a single spike, of a whole
#  cluster:
#
#      would cluster A's strip KEEP cluster B's spikes -- and B's keep A's?
#
#  Per spike x and template T, with the gain divided out (fiber_consolidate.resid_terms):
#      a    = on-channel least-squares gain of x on T
#      onr  = residual power on the channels T lives on    / (a^2 Eon)
#      offr = residual power on the channels T is quiet on / (a^2 Eon)
#  A cluster's own spikes site a robust tail per term (median + k 1.4826 MAD).  The
#  ACCEPTANCE of B by A is the share of B's spikes that land inside A's tails;
#  the pair's score is the MINIMUM of the two directions.  Amplitude is not
#  distance (a is fitted per spike), so a bursting cell's small spikes are accepted;
#  a different cell at the same site moves onr, one at a neighbouring site moves offr.
#
#  Three rules carried over from where the score was measured (fiber-polish,
#  claude/recipe-loop-and-reverse-merge-2026-11-18.md):
#    * the pair is judged under the CLEANER cluster's tails (the smaller threshold per
#      term).  A contaminated cluster's own tails are inflated and it accepts
#      everything; with each side's own tails a threshold of 0.3 collapsed a sort.
#    * greedy with RE-TEMPLATING, no union-find chaining: a merged cluster gets a new
#      median and new tails before it is offered another partner.
#    * ACC_THR well above the collapse.  0.3 separated clean pairs at AUC 0.99 and does not
#      transfer to a real sort: below ~0.5 a session sort collapses into a few blobs.
#
#  TUNED 2026-11-19 on g6 0316 (plans/11-session-residual.yaml; per-knob notes in section [I]
#  of fiber-kit-exp.yaml): ACC_THR 0.675, SHIFT 2, STRIP_K_ON 6.  Chosen as the highest
#  per-unit accuracy whose purity stays within a point of the input sort's, NOT by
#  fiber-tune's objective, which rises all the way into the collapse.
#
#  A THRESHOLD SWEEP COSTS ONE RUN.  The heap pops in descending acceptance and a lower
#  threshold only adds pairs below the others, so a run at ACC_THR t is exactly a run at any
#  lower threshold truncated at the first pair popped below t.  The .rsm table is that pop
#  log, in order, with the run's settings in its first line; --replay <table> reproduces the
#  run at any threshold at or above the table's without scoring anything.
#
#  Candidates are pre-filtered twice, both conservative: the template cosine must
#  exceed COS (0.5 -- unrelated shapes are never scored), and each cluster's MEDIAN
#  must itself pass the other's strip (if the median of B is outside A's tails, well
#  under half of B's spikes are inside them, so the pair cannot reach 0.7).
#
#  Templates and scores are built on the RAW .spk (--spk-method standard) as it is
#  STORED: run it after a fiber-realign, as the plans do.  A pair is compared at the lag
#  within +/- SHIFT (2) that maximises the template cosine -- fragments of one cell in
#  different chunks sit a sample or two apart.  --merge-align peak|klusters re-aligns each
#  cluster's sample first, as fiber-xcorr-merge does.
#
#  Knobs read FK_RSM_* (CLI/plan-param > FK_* env > global fiber-kit.yaml > default).
#  Writes <base>.rsm.<group>.<out-stage>.tsv: every pair that reached the threshold, in
#  the order it was popped, with its acceptance, cosine, lag, sizes and verdict.  Optional --gt-clu scores
#  purity + completeness like fiber-xcorr-merge.
# ════════════════════════════════════════════════════════════════════════════
import argparse
import heapq
import os
import numpy as np

try:
    from . import neuro_io as nio, fiber_lib as fl, session_yaml as sy, fiber_ccg as cg
    from . import fiber_consolidate as _fc
    from . import fiber_xcorr_merge as _xm
    from . import config as cfgmod
    from . import fiber_chunks as _fch
except ImportError:
    import neuro_io as nio, fiber_lib as fl, session_yaml as sy, fiber_ccg as cg
    import fiber_consolidate as _fc
    import fiber_xcorr_merge as _xm
    import config as cfgmod
    import fiber_chunks as _fch

_LP = "▸ fiber-residual-merge"

_KNOBS = {
    # min share of EACH cluster's spikes that survives the other's strip.  The merge threshold.
    "FK_RSM_ACC_THR": ("acc_thr", float, 0.675),
    # template cosine a pair needs to be scored at all (a pre-filter, not the decision)
    "FK_RSM_COS": ("cos", float, 0.5),
    # +/- samples: a pair is compared at the lag (within this) that maximises the template cosine.
    # 0 = the stored alignment.  2 since the 2026-11-19 tuning: fragments of one cell in different chunks
    # sit a sample or two apart, and at lag 0 that offset is on-channel residual.
    "FK_RSM_SHIFT": ("shift", int, 2),
    # robust-tail multipliers of the on- and off-channel residual (the strip's own k)
    "FK_RSM_STRIP_K_ON": ("strip_k_on", float, 6.0),
    "FK_RSM_STRIP_K_OFF": ("strip_k_off", float, 3.0),
    "FK_RSM_WIN": ("win", int, 0),                 # Hamming taper width; 0 = half the record
    "FK_RSM_MINGAIN": ("mingain", float, 0.25),    # a spike below this gain has no template-shaped signal
    "FK_RSM_REFRAC_MS": ("refrac_ms", float, 2.0),
    "FK_RSM_REFRAC_THR": ("refrac_thr", float, 0.3),
    "FK_RSM_REFRAC_MIN_EXP": ("refrac_min_exp", float, 5.0),
    "FK_RSM_MIN_N": ("min_n", int, 40),
    # spikes per cluster for the template, the tails and the acceptance share
    "FK_RSM_SPK_CAP": ("spk_cap", int, 1000),
    "FK_RSM_SAMPLE_KEYED": ("sample_keyed", int, 1),
    # stored = the .spk as it stands; peak / klusters = re-align each cluster's sample first
    "FK_RSM_MERGE_ALIGN": ("merge_align", str, "stored"),
    "FK_RSM_MERGE_ALIGN_MIN_SCORE": ("merge_align_min_score", float, 0.10),
    # the drift table, exactly as fiber-xcorr-merge emits it (FK_XCM_EMIT_DRIFT)
    "FK_RSM_EMIT_DRIFT": ("emit_drift", int, 1),
    "FK_RSM_DRIFT_K": ("drift_k", int, 6),
    "FK_RSM_DRIFT_CHUNK_MIN": ("drift_chunk_min", float, 12.0),
}


def _knob_default(name, typ, fb, gcfg):
    return cfgmod.knob_value(name, typ, fb, gcfg)


def _sample(spk, idx, *, cap, rng, keyed, align, ref_sample, align_min_score):
    """A cluster's waveform sample (<= cap spikes, float), aligned as asked."""
    sel = fl.subsample(idx, cap, rng, keyed=keyed)
    w = np.asarray(spk[sel])
    if align == "klusters":
        return _xm._klusters_align(w, ref_sample, align_min_score)
    if align == "peak":
        return fl.realign(np.asarray(w, float))
    return np.asarray(w, float)


def entry(W, *, k_on=3.0, k_off=3.0, win=0, mingain=0.25):
    """Everything a cluster brings to a comparison: its sample, its median, the unit template and the
    two tails its OWN spikes site.  None when the cluster cannot be judged (a flat template, or fewer
    than 10 spikes with a usable gain)."""
    T = np.median(W, 0)
    t = _fc.resid_terms(W, T, win=win, mingain=mingain)
    if t is None or int(t["ok"].sum()) < 10:
        return None
    ok = t["ok"]
    nrm = float(np.linalg.norm(T))
    if not nrm > 0:
        return None
    return dict(W=W, T=T, tn=T / nrm,
                thr=(_fc._resid_tail(t["onr"][ok], k_on) if k_on > 0 else np.inf,
                     _fc._resid_tail(t["offr"][ok], k_off) if k_off > 0 else np.inf))


def best_lag(A, B, shift):
    """(cosine, lag): the template cosine of A and B at the lag s in [-shift, shift] that maximises it,
    B rolled by s along time.  shift 0 -> the plain cosine at lag 0."""
    if shift <= 0:
        return float((A["tn"] * B["tn"]).sum()), 0
    best, lag = -2.0, 0
    for s in range(-shift, shift + 1):
        c = float((A["tn"] * np.roll(B["tn"], s, axis=0)).sum())
        if c > best:
            best, lag = c, s
    return best, lag


def _inside(t, to, tf):
    return t["ok"] & (t["onr"] <= to) & (t["offr"] <= tf)


def acceptance(A, B, lag=0, *, win=0, mingain=0.25, medians_only=False):
    """min over both directions of the share of one cluster's spikes the other's strip keeps, under the
    CLEANER cluster's tails (the smaller threshold per term).  `lag` rolls B onto A (and A onto B by
    -lag).  medians_only: test just the two MEDIANS -- the cheap pre-filter (1.0 if both pass, else 0.0)."""
    to, tf = min(A["thr"][0], B["thr"][0]), min(A["thr"][1], B["thr"][1])
    out = []
    for X, Y, s in ((A, B, lag), (B, A, -lag)):
        Wy = Y["T"][None] if medians_only else Y["W"]
        if s:
            Wy = np.roll(Wy, s, axis=1)
        t = _fc.resid_terms(Wy, X["T"], win=win, mingain=mingain)
        out.append(0.0 if t is None else float(_inside(t, to, tf).mean()))
    return min(out)


def agglomerate(spk, ids, idx0, times0, duration, *, acc_thr=0.675, cos=0.5, shift=2, k_on=6.0, k_off=3.0,
                win=0, mingain=0.25, refrac=0, refrac_thr=0.3, refrac_min_exp=5.0, censor=0, cap=1000,
                ref_sample=0, rng=None, keyed=True, align="stored", align_min_score=0.10, stats=None):
    """Confidence-ordered reverse-strip agglomeration with re-templating + refractory veto.
    Returns (mapping id -> group id, n_merge, n_veto) -- fiber_xcorr_merge.agglomerate's contract.

    `stats`, when a dict, receives `pairs` (every pair that reached acc_thr: a, b, acc, cos, lag, na, nb,
    verdict), `scored` (pairs whose spikes were scored), `prefiltered` (dropped on their medians) and
    `unjudged` (clusters with no usable template)."""
    rng = rng or np.random.default_rng(0)
    kw = dict(k_on=k_on, k_off=k_off, win=win, mingain=mingain)
    skw = dict(cap=cap, rng=rng, keyed=keyed, align=align, ref_sample=ref_sample, align_min_score=align_min_score)
    idx = {int(u): np.asarray(ix) for u, ix in zip(ids, idx0)}
    tt = {int(u): np.asarray(t) for u, t in zip(ids, times0)}
    E, unjudged = {}, 0
    for u in idx:
        e = entry(_sample(spk, idx[u], **skw), **kw)
        if e is None:
            unjudged += 1
        else:
            E[u] = e
    ver = dict.fromkeys(E, 0)
    heap, pairs, vetoed = [], [], set()
    counts = dict(scored=0, prefiltered=0)

    def offer(a, b):
        """Score the pair and queue it if it reaches the threshold."""
        c, lag = best_lag(E[a], E[b], shift)
        if not c > cos:
            return
        if acceptance(E[a], E[b], lag, win=win, mingain=mingain, medians_only=True) < 1.0:
            counts["prefiltered"] += 1                     # a median outside the other's tails: cannot reach thr
            return
        counts["scored"] += 1
        s = acceptance(E[a], E[b], lag, win=win, mingain=mingain)
        if s >= acc_thr:
            heapq.heappush(heap, (-s, a, b, ver[a], ver[b], c, lag))

    keys = sorted(E)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            offer(a, b)

    parent = {int(u): int(u) for u in idx}
    n_merge = n_veto = 0
    while heap:
        negs, a, b, va, vb, c, lag = heapq.heappop(heap)
        if a not in E or b not in E or ver[a] != va or ver[b] != vb:
            continue                                       # stale: one side changed since it was scored
        row = [a, b, -negs, c, lag, int(idx[a].size), int(idx[b].size)]
        if (a, b) in vetoed:
            continue
        if refrac > 0 and cg.refractory_gate(tt[a], tt[b], duration, refrac, thr=refrac_thr,
                                              min_exp=refrac_min_exp, censor=censor)["verdict"] == "veto":
            vetoed.add((a, b)); n_veto += 1
            pairs.append(row + ["refractory-veto"])
            continue
        # merge b into a, then RE-TEMPLATE the union before it is offered another partner
        idx[a] = np.concatenate([idx[a], idx[b]])
        tt[a] = np.sort(np.concatenate([tt[a], tt[b]]))
        parent[b] = a
        del E[b], idx[b], tt[b]
        n_merge += 1
        pairs.append(row + ["merge"])
        e = entry(_sample(spk, idx[a], **skw), **kw)
        if e is None:
            del E[a]                                       # the union cannot be judged: it keeps what it has
            continue
        E[a] = e
        ver[a] += 1
        for x in sorted(E):
            if x != a:
                lo, hi = (a, x) if a < x else (x, a)
                if (lo, hi) not in vetoed:
                    offer(lo, hi)

    def root(x):
        while parent[x] != x:
            x = parent[x]
        return x
    if stats is not None:
        stats.update(pairs=pairs, unjudged=unjudged, **counts)
    return {int(u): root(int(u)) for u in ids}, n_merge, n_veto


# The settings a pop log depends on: a replay is only the same run if these agree (ACC_THR aside).
_SETTINGS = ("acc_thr", "cos", "shift", "strip_k_on", "strip_k_off", "win", "mingain", "refrac_ms", "refrac_thr",
             "refrac_min_exp", "min_n", "spk_cap", "sample_keyed", "merge_align", "merge_align_min_score")


def table_lines(pairs, settings=None):
    """The <base>.rsm.<g>.<tag>.tsv lines: an optional `# key=value ...` line recording the run's settings,
    the header, then one row per pair IN THE ORDER IT WAS POPPED (which is what makes the file a replayable
    log, not just a report)."""
    out = []
    if settings:
        out.append("# fiber-residual-merge " + " ".join(f"{k}={settings[k]}" for k in _SETTINGS if k in settings))
    out.append("a\tb\tacceptance\tcos\tlag\tn_a\tn_b\tverdict")
    for a, b, s, c, lag, na, nb, v in pairs:
        out.append(f"{a}\t{b}\t{s:.6f}\t{c:.4f}\t{lag}\t{na}\t{nb}\t{v}")
    return out


def read_table(path):
    """(settings dict of strings, pairs) from a .rsm table."""
    settings, pairs = {}, []
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.rstrip("\n")
            if ln.startswith("#"):
                settings.update(kv.split("=", 1) for kv in ln.split()[2:] if "=" in kv)
            elif ln and not ln.startswith("a\t"):
                a, b, s, c, lag, na, nb, v = ln.split("\t")
                pairs.append([int(a), int(b), float(s), float(c), int(lag), int(na), int(nb), v])
    return settings, pairs


def replay(ids, pairs, acc_thr):
    """The result of a run at `acc_thr`, from the pop log of a run at a LOWER threshold and otherwise the
    same settings: the log truncated at the first pair popped below `acc_thr`.

    Exact, not an approximation.  The heap pops in descending acceptance and a lower threshold only ADDS
    pairs below the others, so until the best remaining pair falls under `acc_thr` the two runs pop the
    same pairs, veto the same pairs and re-template the same unions; at that point the run at `acc_thr`
    stops.  Returns (mapping id -> group id, n_merge, n_veto, the pairs used) -- agglomerate's contract."""
    parent = {int(u): int(u) for u in ids}
    used, n_merge, n_veto = [], 0, 0
    for row in pairs:
        a, b, s, v = row[0], row[1], row[2], row[-1]
        if s < acc_thr:
            break
        if a not in parent or b not in parent:
            raise ValueError(f"the table names cluster {a if a not in parent else b}, which is not among the "
                             "clusters of this input: it was written for another .clu or another --min-n")
        used.append(row)
        if v == "merge":
            parent[b] = a; n_merge += 1
        else:
            n_veto += 1

    def root(x):
        while parent[x] != x:
            x = parent[x]
        return x
    return {int(u): root(int(u)) for u in ids}, n_merge, n_veto, used


def main():
    gcfg = cfgmod.load_global_config()
    ap = argparse.ArgumentParser(prog="fiber-residual-merge",
                                 description="Confidence-ordered reverse-residual-strip merge (re-template after each merge).")
    ap.add_argument("session"); ap.add_argument("elec", type=int)
    ap.add_argument("--clu-method", default="stderiv")
    ap.add_argument("--clu-stage", "--variant", dest="clu_stage", default="backbone_linked",
                    help="input .clu stage tag (e.g. the linker's output)")
    ap.add_argument("--in-clu", default=None, help="explicit input .clu path (overrides --clu-method/--clu-stage)")
    ap.add_argument("--spk-method", "--spk-variant", dest="spk_variant", default="standard",
                    help="waveform axis for templates and residuals (RAW: the strip was measured there)")
    ap.add_argument("--out-stage", "--out-tag", dest="out_tag", default="resid_merged",
                    help="post-fiber stage tag of the output .clu")
    ap.add_argument("--refrac-censor-ms", type=float, default=0.0, help="detection censor window (ms)")
    ap.add_argument("--nsamp", type=int, default=None, help="override; default from <session>.yaml")
    ap.add_argument("--nchan", type=int, default=None, help="override; default from <session>.yaml")
    ap.add_argument("--ref-sample", type=int, default=None, help="override; default = peak from <session>.yaml")
    ap.add_argument("--gt-stage", "--gt-clu", dest="gt_clu", default=None,
                    help="post-fiber stage tag (or path) of the curated .clu to score purity+completeness")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--replay", default=None, metavar="TSV",
                    help="the .rsm table of an earlier run on the SAME input at a LOWER --acc-thr and otherwise "
                         "the same settings: this run is that one truncated at --acc-thr (exact), and scores "
                         "nothing.  One run at e.g. 0.4, then any threshold above it for free")
    for name, (dest, typ, fb) in _KNOBS.items():
        ap.add_argument("--" + dest.replace("_", "-"), dest=dest, type=typ, default=_knob_default(name, typ, fb, gcfg),
                        help=f"{name} (default {_knob_default(name, typ, fb, gcfg)})")
    _fch.add_argument(ap, gcfg)
    cfgmod.check_choice_defaults(ap)
    try:                                                 # --in / --out / --method: one spelling on every stage
        from . import stages as _stages
    except ImportError:
        import stages as _stages
    _stages.add_io_aliases(ap, "fiber-residual-merge")
    a = ap.parse_args()
    if a.merge_align not in ("stored", "peak", "klusters"):
        raise SystemExit(f"{_LP} · --merge-align must be stored, peak or klusters, not {a.merge_align!r}")
    if not 0.0 < a.acc_thr <= 1.0:
        raise SystemExit(f"{_LP} · --acc-thr must be in (0, 1], not {a.acc_thr}")
    rng = np.random.default_rng(a.seed)

    cfg = sy.resolve_session_params(a.session, a.elec)
    base = cfg["base"]; elec = a.elec; SR = cfg["sr"]
    NS = a.nsamp if a.nsamp else cfg["nsamp"]
    NC = a.nchan if a.nchan else cfg["nchan"]
    PK = a.ref_sample if a.ref_sample is not None else cfg["peak"]
    res = nio.read_res(base, elec)
    if a.in_clu:
        _, clu = nio.read_clu_file(a.in_clu, n_spikes=res.size)
    else:
        _, clu = nio.read_clu_at(base, elec, variant=a.clu_method, tag=a.clu_stage, n_spikes=res.size)
    spk = nio.open_spk_file(nio.session_path(base, "spk", elec, variant=a.spk_variant), NS, NC)
    assert res.size == clu.size == spk.shape[0], "res/clu/spk length mismatch"
    chunks = _fch.parse(a.chunks)
    clu = _fch.mark_inactive(clu, res, SR, chunks, a.chunk_length)   # other chunks -> reserved clusters
    if chunks is not None:
        print(f"{_LP} · {_fch.describe(chunks, a.chunk_length)}")
    duration = float(res.max()) if res.size else 1.0
    refrac = cg.refrac_samples(a.refrac_ms, SR); censor = cg.refrac_samples(a.refrac_censor_ms, SR)

    ids0, cnt = np.unique(clu[(clu > 1) & ~_fch.is_reserved(clu)], return_counts=True)   # reserved: never merged
    big = [int(u) for u, c in zip(ids0, cnt) if c >= a.min_n]
    print(f"{_LP} · {base} elec {elec} | {a.spk_variant} .spk | {len(big)}/{len(ids0)} clusters >= {a.min_n} spk"
          f" | acceptance>={a.acc_thr:g} (k {a.strip_k_on:g}/{a.strip_k_off:g}) cos>{a.cos:g} shift+/-{a.shift}"
          f" refrac={a.refrac_ms}ms re-template-after-merge ({a.merge_align})")
    if len(big) < 2:
        raise SystemExit(f"{_LP} · too few clusters to merge")

    idx0 = [np.flatnonzero(clu == u) for u in big]
    times0 = [np.sort(res[ix]) for ix in idx0]
    _st = {}
    settings = {k: getattr(a, k) for k in _SETTINGS}
    if a.replay:
        told, pairs = read_table(a.replay)
        if not told:
            raise SystemExit(f"{_LP} · --replay {a.replay}: the table records no settings (written before 0.67.35); "
                             "it cannot be checked against this run")
        if float(told["acc_thr"]) > a.acc_thr + 1e-12:
            raise SystemExit(f"{_LP} · --replay: the table was written at --acc-thr {told['acc_thr']} and holds no pair "
                             f"below it; it cannot give a run at {a.acc_thr:g}")
        diff = [f"{k} {told.get(k)} -> {settings[k]}" for k in _SETTINGS
                if k != "acc_thr" and str(settings[k]) != told.get(k)]
        if diff:
            raise SystemExit(f"{_LP} · --replay: the table was written with other settings ({'; '.join(diff)}); "
                             "a replay is exact only when everything but --acc-thr agrees")
        try:
            mapping, n_merge, n_veto, used = replay(big, pairs, a.acc_thr)
        except ValueError as e:
            raise SystemExit(f"{_LP} · --replay: {e}") from None
        _st.update(pairs=used, scored=0, prefiltered=0, unjudged=0)
        print(f"{_LP} · replayed {os.path.basename(a.replay)} (written at acceptance >= {told['acc_thr']}): "
              f"{len(used)} of its {len(pairs)} pairs are at or above {a.acc_thr:g}")
    else:
        mapping, n_merge, n_veto = agglomerate(
            spk, big, idx0, times0, duration, acc_thr=a.acc_thr, cos=a.cos, shift=a.shift, k_on=a.strip_k_on,
            k_off=a.strip_k_off, win=a.win, mingain=a.mingain, refrac=refrac, refrac_thr=a.refrac_thr,
            refrac_min_exp=a.refrac_min_exp, censor=censor, cap=a.spk_cap, ref_sample=PK, rng=rng,
            keyed=bool(a.sample_keyed), align=a.merge_align, align_min_score=a.merge_align_min_score, stats=_st)
    new = clu.copy().astype(np.int64)
    for u in big:
        if mapping[u] != u:
            new[clu == u] = mapping[u]
    _real = lambda x: x[(x > 1) & ~_fch.is_reserved(x)]   # noqa: E731
    n_in = len(np.unique(_real(clu))); n_out = len(np.unique(_real(new)))
    print(f"{_LP} · merged {n_in - n_out} clusters ({n_in} -> {n_out}); merges {n_merge}, refractory vetoes {n_veto}"
          f"; {_st['scored']:,} pairs scored, {_st['prefiltered']:,} dropped on their medians"
          + (f", {_st['unjudged']} clusters without a usable template" if _st["unjudged"] else ""))
    nio.write_clu(base, elec, new, variant=a.clu_method, tag=a.out_tag)
    print(f"{_LP} · wrote {os.path.basename(nio.session_path(base, 'clu', elec, variant=a.clu_method, tag=a.out_tag))}")

    log_path = f"{base}.rsm.{elec}.{a.out_tag}.tsv"
    with nio.atomic_write(log_path, "w") as fh:
        fh.write("\n".join(table_lines(_st["pairs"], settings)) + "\n")
    print(f"{_LP} · wrote {os.path.basename(log_path)}  ({len(_st['pairs'])} pairs at acceptance >= {a.acc_thr:g})")

    # the hierarchy and the drift table, exactly as fiber-xcorr-merge carries / emits them
    _xm.carry_hierarchy(base, elec, clu_method=a.clu_method, clu_stage=a.clu_stage, out_tag=a.out_tag,
                        mapping=mapping, new=new, chunks=chunks, lp=_LP)
    if a.emit_drift:
        _xm.emit_drift(base, elec, spk, res, SR, big, idx0, mapping, clu_method=a.clu_method, out_tag=a.out_tag,
                       spk_variant=a.spk_variant, drift_k=a.drift_k, drift_chunk_min=a.drift_chunk_min,
                       spk_cap=min(a.spk_cap, 300), rng=rng, keyed=bool(a.sample_keyed), lp=_LP)

    if a.gt_clu:
        try:
            _, gt = (nio.read_clu_file(a.gt_clu, n_spikes=res.size)
                     if (os.path.sep in a.gt_clu or a.gt_clu.endswith(".clu"))
                     else nio.read_clu_at(base, elec, variant=a.clu_method, tag=a.gt_clu, n_spikes=res.size))
        except ValueError as e:
            print(f"{_LP} · GT not scored: {e}")
            return
        keep = np.isin(clu, np.array(big))
        for tag, lab in (("before", clu.astype(np.int64)), ("after ", new)):
            pur, comp = _xm._score(lab, np.asarray(gt), keep)
            print(f"{_LP} · GT {tag} (INCOMPLETE): spike-weighted purity {pur:.3f} | per-parent completeness {comp:.3f}")


if __name__ == "__main__":
    main()
