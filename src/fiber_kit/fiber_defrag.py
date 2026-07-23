#  fiber_defrag.py — de-fragment an over-clustered sort by warp-gated MNN merge.
#
#  An auto sort over-splits each neuron into many drift/amplitude fragments
#  (g5 180-216 min: ~276 fragments for an expected ~47 neurons).  This pass
#  reunites them WITHOUT fusing distinct cells, using two gates:
#
#    shape : mean-template cosine >= cos_thr (baseline-subtracted, flattened).
#    width : the time-WARP needed to align the two templates, |alpha-1| <= smax.
#            Validated on g5 -- the warp stretch separates same-vs-different at
#            AUC 0.97 AT MATCHED COSINE: fragments of one neuron share a spike
#            width (alpha ~ 1) and merge; two distinct cells differ in width and
#            are held apart even when their cosine is high.  This is the lever a
#            fixed cosine threshold cannot provide (it recovered ~half the
#            over-merges a low cosine threshold makes on the curated block).
#
#  Agglomeration is MUTUAL-NEAREST-NEIGHBOUR with template recompute each round,
#  not transitive union-find, so a merge never chains a long drift sequence into
#  one blob (g5: union-find chained 140 fragments into one cluster; MNN caps the
#  largest merge at 20 and lands 276 -> 121).
#
#  Shape-based (drift-naive): templates are compared as given, so it reunites
#  fragments that are time-local (as over-splits typically are).  For long-range
#  cross-drift linking, the localize/drift/link stages still apply.  The energy
#  gate is wide by default (drift moves amplitude); the WIDTH gate is what guards
#  against over-merge.  Pair this with fiber-contam: defrag -> contamination QC
#  on the merged result -> split the flagged.

import argparse
import numpy as np

try:
    from . import (fiber_session as fs, neuro_io as nio, session_yaml as sy,
                   fiber_cfiber as cf, fiber_ccg as ccg, fiber_score as fsc)
except ImportError:                                              # script / direct execution
    import fiber_session as fs, neuro_io as nio, session_yaml as sy, fiber_cfiber as cf
    import fiber_ccg as ccg, fiber_score as fsc

COS_THR   = 0.92    # mean-template cosine at/above this is a merge candidate
SMAX      = 0.06    # |alpha-1| time-warp at/above this == width mismatch -> keep separate
AMP_GATE  = 1.40    # |delta log|F1|| above this == too far in energy to be one neuron (wide: drift moves A)
MIN_SPK   = 40      # fragments smaller than this are left untouched
SAMPLE    = 3000    # cap spikes used to estimate a fragment template
MAX_ROUND = 30


def _baseline(W):
    return W - W[:, :6, :].mean(1, keepdims=True)


def _trace_var(n, s, q):
    """Trace of the pooled covariance (sum of per-PC variances) from sufficient stats."""
    return float(np.sum(q / n - (s / n) ** 2))


def template_cosine(a, b):
    a = (a - a[:6].mean(0)).ravel()
    b = (b - b[:6].mean(0)).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def warp_stretch(A, B, peak):
    """Smallest |alpha-1| over a joint time-stretch + integer-shift search that
    best aligns B to A.  ~0 for two fragments of one neuron (same width); grows
    when the two differ in spike width (distinct cells)."""
    A = A - A[:6].mean(0)
    B = B - B[:6].mean(0)
    nsamp = A.shape[0]
    x = np.arange(float(nsamp))
    Af = A.ravel()
    nA = np.linalg.norm(Af) + 1e-12
    best = (-1.0, 1.0)
    for al in np.linspace(0.85, 1.18, 30):
        xs = peak + (x - peak) / al
        Bw = np.stack([np.interp(x, xs, B[:, c], left=0.0, right=0.0) for c in range(B.shape[1])], 1)
        for sh in range(-3, 4):
            bf = np.roll(Bw, sh, 0).ravel()
            c = float(Af @ bf / (nA * (np.linalg.norm(bf) + 1e-12)))
            if c > best[0]:
                best = (c, al)
    return abs(best[1] - 1.0)


def _energy_log(tm, theta, win):
    z = cf.complex_loop(tm[None], theta, win)
    _, scale, _, _ = cf.shape_descriptor(z)
    return float(np.log(max(float(scale[0]), 1.0)))


def defrag(templates, counts, peak, nchan, *, cos_thr=COS_THR, smax=SMAX,
           amp_gate=AMP_GATE, max_rounds=MAX_ROUND, vstats=None, var_allow=None,
           spike_times=None, duration=None, ccg_refrac=0, ccg_thr=0.3, ccg_min_exp=5.0, ccg_censor=0,
           verbose=True):
    """Mutual-nearest-neighbour merge with template recompute.

    templates : dict id -> (nsamp, nchan) mean stderiv template (baseline-subtracted)
    counts    : dict id -> spike count behind that template (for the weighted recompute)
    vstats    : optional dict id -> (n, s, q) PC-score sufficient stats; when given with
                var_allow, a merge is also rejected if the merged cluster's PC-variance
                exceeds var_allow (the curated envelope from fiber-calibrate).
    spike_times : optional dict id -> sorted sample times; with ccg_refrac>0 and duration, a proposed
                merge is VETOed when the refractory cross-correlogram is powered yet shows no dip
                (distinct neurons).  Power-aware: abstains where rates are too low to decide.
    Returns (root_of, members): root_of maps every input id to its surviving merged
    id; members maps each surviving id to the list of ids folded into it."""
    theta = cf.channel_angles(nchan)
    win = slice(max(0, peak - 12), peak + 14)
    T = {i: np.asarray(t, float) for i, t in templates.items()}
    C = dict(counts)
    E = {i: _energy_log(T[i], theta, win) for i in T}
    VS = {i: (vstats[i][0], np.asarray(vstats[i][1], float), np.asarray(vstats[i][2], float))
          for i in vstats} if (vstats is not None and var_allow is not None) else None
    use_ccg = spike_times is not None and duration and ccg_refrac > 0
    ST = {i: np.asarray(spike_times[i]) for i in spike_times} if use_ccg else None
    ccg_vetoes = 0
    members = {i: [i] for i in T}
    active = set(T)
    for rnd in range(max_rounds):
        al = sorted(active)
        # vectorised best-partner: cosine matrix over baseline-subtracted, L2-normalised templates
        M = np.stack([(T[i] - T[i][:6].mean(0)).ravel() for i in al])
        M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
        Cmat = M @ M.T
        Evec = np.array([E[i] for i in al])
        Cmat[np.abs(Evec[:, None] - Evec[None, :]) > amp_gate] = -1.0   # energy gate
        np.fill_diagonal(Cmat, -1.0)
        bidx = Cmat.argmax(1)
        bcos = Cmat[np.arange(len(al)), bidx]
        best = {al[i]: (al[int(bidx[i])], float(bcos[i])) for i in range(len(al))}
        used = set()
        did = False
        for a in al:
            if a in used:
                continue
            b, cs = best[a]
            if b is None or b in used or cs < cos_thr:
                continue
            if best[b][0] != a:                                  # mutual nearest neighbour only
                continue
            if warp_stretch(T[a], T[b], peak) > smax:            # width mismatch -> distinct cell
                continue
            if VS is not None:                                   # curated variance budget
                n2 = VS[a][0] + VS[b][0]; s2 = VS[a][1] + VS[b][1]; q2 = VS[a][2] + VS[b][2]
                if _trace_var(n2, s2, q2) > var_allow:           # merged cluster too spread for one neuron
                    continue
            if ST is not None:                                   # refractory cross-correlogram veto
                g = ccg.refractory_gate(ST[a], ST[b], duration, ccg_refrac,
                                        thr=ccg_thr, min_exp=ccg_min_exp, censor=ccg_censor)
                if g["verdict"] == "veto":                       # powered, no dip -> distinct neurons
                    ccg_vetoes += 1
                    continue
            na, nb = C[a], C[b]                                  # merge b into a, recompute template
            T[a] = (T[a] * na + T[b] * nb) / (na + nb)
            C[a] = na + nb
            E[a] = _energy_log(T[a], theta, win)
            members[a] += members[b]
            if VS is not None:
                VS[a] = (VS[a][0] + VS[b][0], VS[a][1] + VS[b][1], VS[a][2] + VS[b][2])
            if ST is not None:
                ST[a] = np.sort(np.concatenate([ST[a], ST[b]]))
            active.discard(b)
            del T[b], C[b], E[b], members[b]
            if VS is not None:
                del VS[b]
            if ST is not None:
                del ST[b]
            used.add(a); used.add(b)
            did = True
        if verbose:
            print(f"  round {rnd + 1}: {len(active)} clusters")
        if not did:
            break
    root_of = {m: r for r in active for m in members[r]}
    if use_ccg and verbose and ccg_vetoes:
        print(f"  refractory gate vetoed {ccg_vetoes} merge(s) (powered, no dip)")
    return root_of, members


def main():
    ap = argparse.ArgumentParser(
        description="De-fragment an over-clustered sort: reunite a neuron's drift/amplitude "
                    "fragments by mutual-nearest-neighbour template merging, gated by cosine AND "
                    "the time-warp (spike width) so distinct same-shape cells are held apart.")
    sy.add_session_args(ap)
    ap.add_argument("--clu-method", default="stderiv", help="feature space before the group (default stderiv)")
    ap.add_argument("--clu-stage", "--variant", dest="variant", default="refine",
                    help="fiber stage after the group (default refine; '' = none)")
    ap.add_argument("--in-clu", default=None, help="explicit .clu path (overrides --clu-method/--variant)")
    ap.add_argument("--cos-thr", type=float, default=COS_THR, help="template cosine merge candidate (default 0.92)")
    ap.add_argument("--warp-max", type=float, default=SMAX, help="|alpha-1| width gate; above this keep separate (default 0.06)")
    ap.add_argument("--amp-gate", type=float, default=AMP_GATE, help="|delta log|F1|| energy gate (default 1.4, wide)")
    ap.add_argument("--min-cluster", type=int, default=MIN_SPK, help="fragments smaller than this are left untouched")
    ap.add_argument("--var-budget", default=None,
                    help="path to a fiber-calibrate .npz; adds a curated PC-variance stopping gate "
                         "(merge rejected once a merged cluster would be more spread than a real unit)")
    ap.add_argument("--var-scale", type=float, default=1.0,
                    help="multiply the loaded variance allowance (dial the operating point: "
                         ">1 looser/more merging, <1 tighter; floor is conservative, ~1.5-2x reaches the baseline)")
    ap.add_argument("--out-stage", "--out-tag", dest="out_tag", default=None,
                    help="post-fiber stage tag for the merged result (default 'defrag', single token)")
    ap.add_argument("--ccg-refrac-ms", type=float, default=0.0,
                    help="refractory cross-correlogram veto window (ms); 0 disables. ~1.5 to enable. "
                         "Power-aware: abstains where firing rates are too low to show a dip (e.g. g5).")
    ap.add_argument("--ccg-thr", type=float, default=0.3, help="cross-CCG ratio above which a powered pair is vetoed")
    ap.add_argument("--ccg-min-exp", type=float, default=5.0, help="min expected coincidences for the veto to act")
    ap.add_argument("--ccg-censor-ms", type=float, default=0.3, help="duplicate censor band for the cross-CCG (ms)")
    ap.add_argument("--gt-clu", default=None, help="ground-truth .clu to score before/after the merge against")
    ap.add_argument("--gt-res", default=None, help=".res for the ground truth (timestamp alignment if it covers a window)")
    a = ap.parse_args()

    cfg = sy.resolve_session_params(a.session, a.group, channels=a.channels, ntotal=a.ntotal,
                                    nchan=a.nchan, nsamp=a.nsamp, sr=a.sr)
    base = cfg["base"]; elec = a.group
    nchan, nsamp, peak = cfg["nchan"], cfg["nsamp"], cfg["peak"]

    res = fs.read_res(base, elec)
    if a.in_clu:
        _, clu = nio.read_clu_file(a.in_clu, n_spikes=len(res))
    else:
        _, clu = nio.read_clu_at(base, elec, variant=a.clu_method, tag=a.variant, n_spikes=len(res))
    spk, spkpath = fs.open_spkD(base, elec, nsamp, nchan)
    assert spk.shape[0] == len(res) == len(clu), \
        f".res {len(res)} / .clu {len(clu)} / {spkpath} {spk.shape[0]} mismatch"

    rng = np.random.default_rng(0)
    var_basis = None; var_allow = None
    if a.var_budget:
        cal = np.load(a.var_budget)
        var_basis = (np.asarray(cal["mu"], float), np.asarray(cal["B"], float))
        var_allow = float(cal["allow"]) * a.var_scale
        print(f"variance budget: allow={var_allow:.3e} ({a.var_budget}, scale {a.var_scale})")
    ids = [int(c) for c in np.unique(clu) if c > 1 and int((clu == c).sum()) >= a.min_cluster]
    templates, counts = {}, {}
    vstats = {} if var_basis is not None else None
    for c in ids:
        idx = np.flatnonzero(clu == c)
        if idx.size > SAMPLE:
            idx = idx[rng.permutation(idx.size)[:SAMPLE]]
        W = _baseline(spk[idx].astype(float))
        templates[c] = W.mean(0)
        counts[c] = int(idx.size)
        if var_basis is not None:
            F = (W.reshape(len(idx), -1) - var_basis[0]) @ var_basis[1].T
            vstats[c] = (len(idx), F.sum(0), (F ** 2).sum(0))
    print(f"loaded {len(res)} spikes; {len(ids)} fragments >= {a.min_cluster} spk ({spkpath})")

    spike_times = duration = None
    ccg_refrac = ccg_censor = 0
    if a.ccg_refrac_ms > 0:
        sr = cfg["sr"]
        ccg_refrac = ccg.refrac_samples(a.ccg_refrac_ms, sr)
        ccg_censor = ccg.refrac_samples(a.ccg_censor_ms, sr)
        duration = float(res.max() - res.min())
        spike_times = {c: np.sort(res[clu == c]) for c in ids}
        print(f"refractory veto on: {a.ccg_refrac_ms} ms window, ratio>{a.ccg_thr}, "
              f"min C_exp {a.ccg_min_exp} (else abstain)")

    root_of, members = defrag(templates, counts, peak, nchan,
                              cos_thr=a.cos_thr, smax=a.warp_max, amp_gate=a.amp_gate,
                              vstats=vstats, var_allow=var_allow,
                              spike_times=spike_times, duration=duration, ccg_refrac=ccg_refrac,
                              ccg_thr=a.ccg_thr, ccg_min_exp=a.ccg_min_exp, ccg_censor=ccg_censor)
    n_merged = len(members)
    largest = max((len(m) for m in members.values()), default=0)
    print(f"defrag: {len(ids)} fragments -> {n_merged} clusters "
          f"(cos>={a.cos_thr}, warp<={a.warp_max}); largest merge {largest} fragments")

    # relabel: every fragment id maps to its surviving root; untouched ids (noise, small) keep their id
    new = clu.copy()
    for orig, root in root_of.items():
        if orig != root:
            new[clu == orig] = root
    tag = a.out_tag if a.out_tag else "defrag"
    nio.write_clu(base, elec, new.astype(np.int64),
                  n_clusters=int(len(np.unique(new[new > 0]))),
                  variant=a.clu_method, tag=tag)
    print(f"wrote .clu.{a.clu_method}.{elec}.{tag}  "
          f"(then run fiber-contam on it to flag any cells the merge over-reached on)")

    if a.gt_clu:                                                 # measure whether the merge improved agreement
        _, gt = nio.read_clu_file(a.gt_clu)
        if gt.size == len(res):
            c_before, c_after, gt_lab = clu, new, gt
        elif a.gt_res:
            gres = nio.read_res_file(a.gt_res)
            cb, gt_lab, _ = fsc.align_by_res(clu, res, gt, gres)
            ca, _, _ = fsc.align_by_res(new, res, gt, gres)
            c_before, c_after = cb, ca
        else:
            print("--gt-clu length differs from .res; pass --gt-res to align by timestamp"); return
        sb = fsc.score(c_before, gt_lab); sa = fsc.score(c_after, gt_lab)
        print("ground-truth score (before -> after defrag):")
        print("  ARI            %.4f -> %.4f" % (sb["ari"], sa["ari"]))
        print("  pairwise prec  %.4f -> %.4f" % (sb["pairwise_precision"], sa["pairwise_precision"]))
        print("  pairwise recall%.4f -> %.4f" % (sb["pairwise_recall"], sa["pairwise_recall"]))
        print("  GT units split %d -> %d   |  merged candidates %d -> %d"
              % (sb["n_gt_split"], sa["n_gt_split"], sb["n_cand_merged"], sa["n_cand_merged"]))


if __name__ == "__main__":
    main()
