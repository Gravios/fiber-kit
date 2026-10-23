"""fiber-xcorr-merge amplitude-aware gates: gain-IQR veto, per-channel residual-variance veto, knn split.

Synthetic standard .spk: unit A and unit A' share ONE footprint and shape and differ only in amplitude
(A' = 0.65 A), so the roll cosine -- DC removed, unit-normalised -- cannot tell them apart.  Each unit
is in two fragments; one A fragment also carries A' spikes as contamination.

  * no gate: every fragment of A and A' fuses into one cluster (the failure being fixed);
  * --rvu-rise: per-channel residual-variance non-uniformity vetoes A+A' and keeps both units;
  * --giqr-thr: the own-gain IQR vetoes A+A' as well;
  * --rvu-knn (with --rvu-thr, the absolute non-uniformity that catches a fragment that is ALREADY
    mixed, which a rise test cannot see): the waveform knn split moves the A' contamination out of the A fragment into an A'
    cluster, so both units come out pure;
  * all gates off: the output is identical to the call without the new arguments.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fiber_kit import fiber_xcorr_merge as xm   # noqa: E402

fails = 0
def check(ok, what):
    global fails
    print(("  ok:   " if ok else "  FAIL: ") + what); fails += not ok


def build(rng):
    T, C, PK = 32, 4, 16
    t = np.arange(T)
    shape = -np.exp(-0.5 * ((t - PK) / 1.4) ** 2) + 0.3 * np.exp(-0.5 * ((t - PK - 5) / 2.5) ** 2)
    A = shape[:, None] * np.array([1.0, 0.6, 0.3, 0.1])[None, :] * 1500
    plan = [(2, A, 300), (3, A, 250), (3, 0.65 * A, 80), (4, 0.65 * A, 300), (5, 0.65 * A, 300)]
    W, ids, unit = [], [], []
    for lab, tm, n in plan:
        for _ in range(n):
            W.append(tm + rng.normal(0, 60, (T, C))); ids.append(lab); unit.append(0 if tm is A else 1)
    spk = np.clip(np.rint(np.array(W)), -32768, 32767).astype(np.int16)
    res = np.sort(rng.choice(np.arange(10**8), len(ids), replace=False))
    perm = rng.permutation(len(ids))                       # interleave in time, as a recording would
    return spk[perm], np.array(ids)[perm], np.array(unit)[perm], res, PK


def run(spk, ids, res, PK, **kw):
    labs = [2, 3, 4, 5]
    idx0 = [np.flatnonzero(ids == u) for u in labs]; times0 = [np.sort(res[i]) for i in idx0]
    st = {}
    mp, nm, nv = xm.agglomerate(spk, labs, idx0, times0, float(res.max()), cos_thr=0.96, shift=4, refrac=0,
                                refrac_thr=0.3, refrac_min_exp=5.0, censor=0, cap=300, ref_sample=PK,
                                rng=np.random.default_rng(0), band_thr=0.5, res=res, min_n=15,
                                next_id=6, stats=st, **kw)
    out = ids.copy()
    if st.get("knn_moved", 0):
        for u, ix in st["members"].items():
            out[ix] = u
    else:
        for u in labs:
            out[ids == u] = mp[u]
    return out, st


def purity(out, unit):
    """(clusters holding both units with >= 10% of the minority, worst purity over clusters >= 30 spikes)"""
    mixed, worst = 0, 1.0
    for c in np.unique(out):
        u = unit[out == c]
        if u.size < 30:
            continue
        p = max(u.mean(), 1 - u.mean()); worst = min(worst, p)
        mixed += (1 - p) >= 0.10
    return mixed, worst


def main():
    spk, ids, unit, res, PK = build(np.random.default_rng(7))
    out0, _ = run(spk, ids, res, PK)
    m0, w0 = purity(out0, unit)
    check(np.unique(out0).size == 1, f"no gate: A and A' fuse into one cluster ({np.unique(out0).size} cluster(s), "
                                     f"worst purity {w0:.2f})")
    outr, str_ = run(spk, ids, res, PK, rvu_rise=0.05)
    check(str_["rvu_vetoes"] > 0 and np.unique(outr[unit == 0]).size < np.unique(outr).size,
          f"--rvu-rise 0.05: residual-variance veto keeps A' out of A ({str_['rvu_vetoes']} vetoes)")
    outg, stg = run(spk, ids, res, PK, giqr_thr=0.15)
    check(stg["gain_vetoes"] > 0 and np.unique(outg).size >= 2,
          f"--giqr-thr 0.15: gain-IQR veto keeps A and A' apart ({stg['gain_vetoes']} vetoes)")
    outk, stk = run(spk, ids, res, PK, rvu_thr=0.5, rvu_rise=0.05, rvu_knn=1)
    mk, wk = purity(outk, unit)
    check(stk["knn_splits"] >= 1 and stk["knn_moved"] > 0, f"--rvu-knn: knn split ran and moved spikes "
                                                           f"({stk['knn_splits']} splits, {stk['knn_moved']} moved)")
    check(mk == 0 and wk >= 0.95, f"--rvu-knn: every cluster pure (worst purity {wk:.3f}; without knn "
                                  f"{purity(outr, unit)[1]:.3f})")
    a_cl = np.unique(outk[unit == 0], return_counts=True); b_cl = np.unique(outk[unit == 1], return_counts=True)
    check(a_cl[1].max() >= 0.9 * (unit == 0).sum() and b_cl[1].max() >= 0.9 * (unit == 1).sum(),
          f"--rvu-knn: each unit ends up >= 90% in one cluster (A {a_cl[1].max()}/{(unit == 0).sum()}, "
          f"A' {b_cl[1].max()}/{(unit == 1).sum()})")
    outoff, _ = run(spk, ids, res, PK, giqr_thr=0.0, giqr_rise=0.0, rvu_thr=0.0, rvu_rise=0.0, rvu_knn=0)
    check(np.array_equal(outoff, out0), "all new gates off: identical to the call without them")
    print(f"\ntest_xcorr_merge_vetoes: {'FAIL' if fails else 'PASS'} ({fails} failed)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
