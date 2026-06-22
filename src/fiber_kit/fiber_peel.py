"""fiber-peel: confidence-ordered footprint + refractory agglomeration.

Consolidates the over-split fragments a refine pass leaves behind into cells, by
greedy best-footprint-cosine merging gated by TWO complementary signals:

  * footprint cosine >= theta   -- the amplitude-PRESERVING mean-waveform cosine
    (NOT scale-invariant shape: the across-channel amplitude profile is the most
    stable identity of a non-adapting cell, and dropping it is what made pure
    shape clustering useless).
  * the refractory cross-correlogram does NOT veto (powered-only, ratio-based).

The two are complementary -- each covers the other's blind spot:
  - footprint-similar but INDEPENDENT cells (two real units at one site, cosine
    high) are blocked by the refractory veto;
  - the censoring artifact that FAKES a refractory dip (a dominant cell masks a
    minority's coincident spikes during detection, so the cross-CCG has a hole at
    zero lag that mimics same-cell) is blocked by the footprint floor, because a
    genuine same-cell pair sits at cosine ~1.0 while the censored pair does not.

Because the dominant, refractory-respecting unit -- typically a high-rate
interneuron carrying a large share of the spikes -- has fragments at cosine ~1.0
to one another, it agglomerates FIRST and is peeled out of the feature space
before the sparser pyramidal cells are resolved, de-cluttering their space.

Footprint cosine is stable WITHIN a drift-coherent window but drifts with the
cell, so this pass is a within-window consolidator: cross-window drift identity
is reunited downstream by fiber-link's overlap anchor.  Runs in-place on the clu
(a pure relabel; .res/.spk untouched), so it can be dropped between fiber-refine
and fiber-cpos without changing any stage's I/O wiring.
"""
import argparse
import numpy as np

try:
    from . import neuro_io as nio
except ImportError:
    import neuro_io as nio
try:
    from . import session_yaml as sy
except ImportError:
    import session_yaml as sy
try:
    from . import fiber_ccg as cg
except ImportError:
    import fiber_ccg as cg


def _unit(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-12)


def peel_agglomerate(footprints, times, counts, duration, sr, *,
                     foot_hi=0.97, foot_lo=0.90, anneal_steps=4,
                     refrac_ms=2.0, refrac_thr=0.3, refrac_min_exp=5.0,
                     refrac_censor_ms=0.0):
    """Greedy, confidence-ordered footprint+refractory agglomeration of fragments.

    footprints : list of 1-D float arrays (mean waveform per fragment; any common
                 layout, mean-subtraction is applied here).
    times      : list of 1-D int arrays of spike sample times per fragment.
    counts     : list of ints (spikes per fragment).
    duration   : recording span in samples (for the refractory expectation).
    Returns a list `lab` (len == n fragments) of 0-based merged-group ids.

    The anneal descends theta from foot_hi to foot_lo; at each level the
    highest-cosine non-vetoed pair merges first (so the most certain identities --
    the dominant clean unit's own fragments at cosine ~1.0 -- fuse earliest).
    """
    n = len(footprints)
    if n == 0:
        return []
    refr = cg.refrac_samples(refrac_ms, sr)
    cens = cg.refrac_samples(refrac_censor_ms, sr)
    foot = [(_unit(np.asarray(f, float) - np.asarray(f, float).mean())) for f in footprints]
    tim = [np.asarray(t, np.int64) for t in times]
    cnt = [int(c) for c in counts]
    parent = list(range(n))
    alive = list(range(n))

    def fcos(a, b):
        return float(foot[a] @ foot[b])

    for theta in np.linspace(foot_hi, foot_lo, max(1, anneal_steps)):
        blacklist = set()
        while True:
            best = None
            for ii in range(len(alive)):
                a = alive[ii]
                for jj in range(ii + 1, len(alive)):
                    b = alive[jj]
                    key = (a, b) if a < b else (b, a)
                    if key in blacklist:
                        continue
                    c = fcos(a, b)
                    if c < theta:
                        continue
                    if best is None or c > best[0]:
                        best = (c, ii, jj, a, b)
            if best is None:
                break
            _, ii, jj, a, b = best
            g = cg.refractory_gate(tim[a], tim[b], duration, refr,
                                   thr=refrac_thr, min_exp=refrac_min_exp, censor=cens)
            if g["verdict"] == "veto":            # footprint says merge, refractory says two cells
                blacklist.add((a, b) if a < b else (b, a))
                continue
            # accept: count-weighted footprint, merged train, union
            foot[a] = _unit(foot[a] * cnt[a] + foot[b] * cnt[b])
            tim[a] = np.sort(np.concatenate([tim[a], tim[b]]))
            cnt[a] += cnt[b]
            parent[b] = a
            alive.pop(jj)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    roots = {}
    lab = [0] * n
    for f in range(n):
        r = find(f)
        lab[f] = roots.setdefault(r, len(roots))
    return lab


def peel_clu(src, res, spk, sr, *, min_n=15, **gate):
    """Apply the agglomeration to a clu labelling. `src` 0/1 are reserved
    (0=noise/out-of-window, 1=unsorted); only ids > 1 are fragments.  Returns a
    new label array with the same reserve convention."""
    src = np.asarray(src)
    res = np.asarray(res, np.int64)
    duration = float(res.max() - res.min()) if len(res) else 0.0
    fids = [c for c in np.unique(src[src > 1]) if (src == c).sum() >= min_n]
    if len(fids) < 2:
        return src.copy(), 0
    foot, tim, cnt, idxs = [], [], [], []
    for c in fids:
        m = np.flatnonzero(src == c)
        foot.append(np.asarray(spk[m], float).mean(0).ravel())
        tim.append(np.sort(res[m]))
        cnt.append(len(m))
        idxs.append(m)
    lab = peel_agglomerate(foot, tim, cnt, duration, sr, **gate)
    out = src.copy()
    for f, group in enumerate(lab):
        out[idxs[f]] = group + 2                  # 0/1 reserved
    n_merged = len(fids) - len(set(lab))
    # keep the small / reserved spikes as they were
    out[src <= 1] = src[src <= 1]
    return out, n_merged


def main():
    ap = argparse.ArgumentParser(
        description="fiber-peel: consolidate over-split refine fragments by "
                    "footprint cosine gated by the refractory cross-CCG. In-place "
                    "relabel of the clu (pure relabel; .res/.spk untouched).")
    sy.add_session_args(ap)
    ap.add_argument("--cpos-method", default="stderiv")
    ap.add_argument("--cpos-stage", default="refine")
    ap.add_argument("--clu-method", default=None)
    ap.add_argument("--clu-stage", default=None)
    ap.add_argument("--out-stage", default=None,
                    help="output clu stage (default: same as input == in-place relabel)")
    ap.add_argument("--foot-hi", type=float, default=0.97,
                    help="anneal start: strict footprint-cosine for the first, most certain merges")
    ap.add_argument("--foot-lo", type=float, default=0.90,
                    help="anneal floor: loosest footprint-cosine accepted (best g5 result at 0.90)")
    ap.add_argument("--anneal-steps", type=int, default=4)
    ap.add_argument("--refrac-ms", type=float, default=2.0,
                    help="refractory half-window (ms) for the veto cross-CCG")
    ap.add_argument("--refrac-thr", type=float, default=0.3,
                    help="coincidence-ratio above which the pair is two cells (veto)")
    ap.add_argument("--refrac-min-exp", type=float, default=5.0,
                    help="min expected coincidences for the veto to be powered (else abstain -> merge allowed)")
    ap.add_argument("--refrac-censor-ms", type=float, default=0.0,
                    help="censor window (ms) dropping duplicate detections of one spike")
    ap.add_argument("--min-n", type=int, default=15, help="min spikes for a fragment to participate")
    a = ap.parse_args()

    cfg = sy.resolve_session_params(a.session, a.group, require=("ntotal", "sr"))
    base = cfg.base
    elec = a.group
    sr = float(cfg.sr)
    nsamp = int(cfg.nsamp)
    nch = int(cfg.nchan)
    clu_method = a.clu_method if a.clu_method is not None else a.cpos_method
    clu_stage = a.clu_stage if a.clu_stage is not None else a.cpos_stage
    out_stage = a.out_stage if a.out_stage is not None else clu_stage   # default in-place

    _, src = nio.read_clu_at(base, elec, variant=clu_method, tag=clu_stage)
    res = nio.read_res(base, elec)
    spk, _ = nio.open_spk(base, elec, nsamp, nch)

    out, n_merged = peel_clu(
        src, res, spk, sr, min_n=a.min_n,
        foot_hi=a.foot_hi, foot_lo=a.foot_lo, anneal_steps=a.anneal_steps,
        refrac_ms=a.refrac_ms, refrac_thr=a.refrac_thr,
        refrac_min_exp=a.refrac_min_exp, refrac_censor_ms=a.refrac_censor_ms)

    ncl = int(out.max()) + 1
    out_path = nio.session_path(base, "clu", elec, variant=clu_method, tag=out_stage)
    nio.write_clu_file(out_path, out, n_clusters=ncl)
    n_in = len(np.unique(src[src > 1]))
    n_out = len(np.unique(out[out > 1]))
    print(f"[peel] {n_in} fragments -> {n_out} units ({n_merged} merges) "
          f"foot[{a.foot_hi:.2f}->{a.foot_lo:.2f}] refrac {a.refrac_ms}ms thr {a.refrac_thr}")
    print(f"[peel] wrote {out_path}  ({ncl} clusters incl reserve)")


if __name__ == "__main__":
    main()
