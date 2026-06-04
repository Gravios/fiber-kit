#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════
#  validate_merge_candidates.py  —  full-session evidence for profile-merge pairs
#
#  fiber_session.py --emit-merge-candidates proposes same-neuron fiber pairs from
#  the energy-resolved direction profile (the AUC ~0.98 signal).  This script
#  gathers the INDEPENDENT, full-session evidence the in-chunk geometry can't:
#
#    (1) Cross-correlogram, binned, so you SEE where coincidences fall.
#        - true same neuron: the [0,2] ms refractory bin stays empty AND the
#          combined train shows no new short-lag structure.
#        - NOTE within one shank this is partially confounded: near-coincident
#          spikes of *different* neurons are extracted as collisions and removed,
#          so the [0,2] ms bin reads ~0 for distinct pairs too.  The informative
#          window is [2,5] ms (just outside collision removal): a real single
#          neuron is still in relative refractory there; two neurons are not.
#    (2) Rate co-modulation across the session (30 s bins).  An over-split-by-
#        drift neuron hands its spikes between the two fibers as amplitude wanders
#        -> their binned rates are anti-correlated (one rises as the other falls).
#        Two distinct neurons co-modulated only by network state -> >= 0 correlation.
#
#  Reads <base>.res.<elec>, <base>.clu.<elec>, <base>.merge_candidates.<elec>.tsv.
#  Pair ids are gid (global) = clu value - 1.  Use the gid columns from a run made
#  with --emit-merge-candidates (review-only; gid columns are valid only then).
#
#  Usage:
#    python3 validate_merge_candidates.py <FileBase> <ElecNo> --sr 32552
# ═══════════════════════════════════════════════════════════════════════════
import argparse, numpy as np
try:
    from .fiber_session import read_res
except ImportError:
    from fiber_session import read_res


def read_clu(base, elec):
    a = np.fromfile(f"{base}.clu.{elec}", dtype=np.int32)
    return a[1:]                                   # drop nClusters header


def ccg_counts(tA, tB, sr, edges_ms):
    """Cross-CG mass in each [lo,hi) ms band (per ms), symmetric in |dt|."""
    tB = np.sort(tB); sa = np.searchsorted; out = []
    for lo, hi in edges_ms:
        lo_s, hi_s = lo * sr / 1000.0, hi * sr / 1000.0
        c = (sa(tB, tA + hi_s) - sa(tB, tA + lo_s)) + (sa(tB, tA - lo_s) - sa(tB, tA - hi_s))
        out.append(c.sum() / (2.0 * (hi - lo)))    # counts per ms
    return np.array(out)


def rate_corr(tA, tB, sr, bin_s=30.0):
    t0 = min(tA.min(), tB.min()); t1 = max(tA.max(), tB.max())
    bins = np.arange(t0, t1 + bin_s * sr, bin_s * sr)
    ra, _ = np.histogram(tA, bins); rb, _ = np.histogram(tB, bins)
    if ra.std() < 1e-9 or rb.std() < 1e-9 or len(ra) < 4:
        return float('nan')
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("elec", type=int)
    ap.add_argument("--sr", type=float, default=32552.0)
    ap.add_argument("--cand", default=None, help="candidates tsv (default <base>.merge_candidates.<elec>.tsv)")
    a = ap.parse_args()
    res = read_res(a.base, a.elec); clu = read_clu(a.base, a.elec)
    assert len(res) == len(clu), f".res {len(res)} vs .clu {len(clu)}"
    cand_path = a.cand or f"{a.base}.merge_candidates.{a.elec}.tsv"
    rows = [l.split("\t") for l in open(cand_path).read().splitlines()[1:] if l.strip()]
    bands = [(0, 2), (2, 5), (5, 10), (10, 20)]    # ms; (10,20)=baseline

    def times(gid): return res[clu == gid + 1].astype(float)

    print(f"loaded {len(res)} spikes, {clu.max()} clusters; {len(rows)} candidate pairs\n")
    print(f"{'pair(gid)':>13} {'n_a':>6} {'n_b':>6} {'pdist':>6} | "
          f"{'0-2':>6} {'2-5':>6} {'5-10':>6} {'base':>6} | {'rate_r':>6}  verdict")
    for r in rows:
        chunk, ga, gb, la, lb, pd, thr = r[0], int(r[1]), int(r[2]), r[3], r[4], float(r[5]), float(r[6])
        if ga < 0 or gb < 0:
            continue
        tA, tB = times(ga), times(gb)
        if len(tA) < 30 or len(tB) < 30:
            continue
        c = ccg_counts(tA, tB, a.sr, bands); base = c[-1] + 1e-9
        ref2 = c[0] / base; ref5 = c[1] / base    # normalized to baseline rate
        rc = rate_corr(tA, tB, a.sr)
        # same-neuron evidence: relative refractory survives in [2,5] ms AND rates
        # don't co-rise (anti-/un-correlated handoff).  Heuristic flag only.
        same = (ref5 < 0.5) and (not np.isnan(rc) and rc < 0.2)
        verdict = "SAME?" if same else ("distinct?" if ref5 > 0.8 else "review")
        print(f"{ga:>5}-{gb:<5}({chunk:>1}) {len(tA):>6} {len(tB):>6} {pd:>6.3f} | "
              f"{c[0]:>6.1f} {c[1]:>6.1f} {c[2]:>6.1f} {c[3]:>6.1f} | {rc:>6.2f}  {verdict}")
    print("\nbands are cross-CG counts per ms; refractory evidence = low [2,5] vs base.")
    print("rate_r<0: drift handoff (same neuron);  rate_r>0: co-active (likely distinct).")
    print("flags are heuristics — inspect the CCG shape for borderline pairs.")


if __name__ == "__main__":
    main()
