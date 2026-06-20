#  fiber_calibrate.py — emit a curated variance/energy envelope for fiber-defrag --var-budget.
#
#  Point this at a HAND-CURATED .clu for a group: it learns how spread-out a real
#  single unit is in feature space (PC-score variance) and in energy (log|F1|), and
#  writes a small .npz the defrag merge consumes as a stopping gate.  The merge then
#  stops folding fragments together once a merged cluster would be more spread than a
#  real neuron -- a data-driven budget in place of an arbitrary cosine cutoff.
#
#  Two allowance modes:
#    default      : within-unit PC-variance p95 (how big a single neuron gets).
#    --floor      : the merged-variance FLOOR of distinct curated pairs that survive
#                   the cosine+warp gate -- the tightest gap between genuinely
#                   confusable real units, i.e. "never merge across a gap that real
#                   distinct cells exhibit".  Safer (more conservative) and recommended.
#
#  NOTE the two distributions OVERLAP on hard data: a drifting single unit can be
#  more spread than a confusable distinct pair is separated, so no variance threshold
#  separates every case.  The budget REGULATES the merge (a calibrated stop); it does
#  not add separation power for the confusable class -- that still needs the warp gate
#  upstream and the CCG/refractory check downstream.  See fiber-contam.

import argparse
import itertools
import numpy as np

try:
    from . import (fiber_session as fs, neuro_io as nio, session_yaml as sy,
                   fiber_cfiber as cf)
    from .fiber_defrag import _baseline, template_cosine, warp_stretch, _energy_log
except ImportError:                                              # script / direct execution
    import fiber_session as fs, neuro_io as nio, session_yaml as sy, fiber_cfiber as cf
    from fiber_defrag import _baseline, template_cosine, warp_stretch, _energy_log

FIT_SAMPLE = 60000


def _trace_var(n, s, q):
    return float(np.sum(q / n - (s / n) ** 2))


def main():
    ap = argparse.ArgumentParser(
        description="Learn the variance/energy envelope of a curated group's single units "
                    "and write an .npz budget for `fiber-defrag --var-budget`.")
    sy.add_session_args(ap)
    ap.add_argument("--clu-method", default="stderiv", help="feature space before the group (default stderiv)")
    ap.add_argument("--variant", "--clu-stage", dest="variant", default="",
                    help="curated fiber stage tag after the group (default none)")
    ap.add_argument("--in-clu", default=None, help="explicit curated .clu path (overrides --clu-method/--variant)")
    ap.add_argument("--n-pc", type=int, default=10, help="number of PCs for the feature space (default 10)")
    ap.add_argument("--min-cluster", type=int, default=60, help="ignore curated units below this many spikes")
    ap.add_argument("--floor", action="store_true",
                    help="set allowance to the confusable-pair merged-variance floor (tighter, recommended)")
    ap.add_argument("--cos-thr", type=float, default=0.85, help="candidate cosine for the confusable-pair gate")
    ap.add_argument("--warp-max", type=float, default=0.06, help="width gate for the confusable-pair gate")
    ap.add_argument("--out", default=None, help="output .npz (default '<base>.calib.<group>.npz')")
    a = ap.parse_args()

    cfg = sy.resolve_session_params(a.session, a.group, channels=a.channels, ntotal=a.ntotal,
                                    nchan=a.nchan, nsamp=a.nsamp, sr=a.sr)
    base = cfg["base"]; elec = a.group
    nchan, nsamp, peak = cfg["nchan"], cfg["nsamp"], cfg["peak"]
    theta = cf.channel_angles(nchan); ewin = slice(8, min(nsamp, 34))

    res = fs.read_res(base, elec)
    if a.in_clu:
        _, clu = nio.read_clu_file(a.in_clu, n_spikes=len(res))
    else:
        _, clu = nio.read_clu_at(base, elec, variant=a.clu_method, tag=a.variant, n_spikes=len(res))
    spk, spkpath = fs.open_spkD(base, elec, nsamp, nchan)
    rng = np.random.default_rng(0)

    # PC basis fit on a random sample of the whole group (so all unit variances are comparable)
    samp = rng.permutation(len(spk))[:FIT_SAMPLE]
    X = _baseline(spk[samp].astype(float)).reshape(len(samp), -1)
    mu = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    B = Vt[:a.n_pc]

    ids = [int(c) for c in np.unique(clu) if c > 1 and int((clu == c).sum()) >= a.min_cluster]
    Wu = []; logE_std = []; NN = {}; SS = {}; QQ = {}; T = {}; EL = {}
    for c in ids:
        idx = np.flatnonzero(clu == c)
        W = _baseline(spk[idx].astype(float))
        F = (W.reshape(len(idx), -1) - mu) @ B.T
        NN[c], SS[c], QQ[c] = len(idx), F.sum(0), (F ** 2).sum(0)
        Wu.append(_trace_var(NN[c], SS[c], QQ[c]))
        T[c] = W.mean(0); EL[c] = _energy_log(T[c], theta, ewin)
        s = idx if len(idx) <= 400 else idx[rng.permutation(len(idx))[:400]]
        z = cf.complex_loop(_baseline(spk[s].astype(float)), theta, ewin)
        _, sc, _, _ = cf.shape_descriptor(z)
        logE_std.append(float(np.log(np.maximum(np.asarray(sc), 1.0)).std()))
    Wu = np.array(Wu); logE_std = np.array(logE_std)
    p95 = float(np.percentile(Wu, 95))
    amp_gate = float(2.0 * np.percentile(logE_std, 95))

    allow = p95
    floor = np.nan
    if a.floor:
        gated = []
        for x, y in itertools.combinations(ids, 2):
            if abs(EL[x] - EL[y]) > amp_gate:
                continue
            if template_cosine(T[x], T[y]) < a.cos_thr:
                continue
            if warp_stretch(T[x], T[y], peak) > a.warp_max:
                continue
            gated.append(_trace_var(NN[x] + NN[y], SS[x] + SS[y], QQ[x] + QQ[y]))
        if gated:
            floor = float(min(gated))
            allow = floor
        else:
            print("  --floor: no distinct curated pair survives the cos+warp gate; "
                  "falling back to within-unit p95")

    out = a.out if a.out else f"{base}.calib.{elec}.npz"
    np.savez(out, B=B, mu=mu, allow=allow, k=a.n_pc, amp_gate=amp_gate,
             p95=p95, floor=floor, n_units=len(ids))
    print(f"calibrated on {len(ids)} curated units ({spkpath})")
    print(f"  within-unit PC-variance: median {np.median(Wu):.3e}  p95 {p95:.3e}")
    print(f"  within-unit log|F1| jitter p95 -> amp_gate {amp_gate:.2f}")
    if a.floor and floor == floor:
        print(f"  confusable-pair floor:   {floor:.3e}  (allowance set here)")
    print(f"  wrote {out}  (allow={allow:.3e}, k={a.n_pc})  -> fiber-defrag --var-budget {out}")


if __name__ == "__main__":
    main()
