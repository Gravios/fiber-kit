#!/usr/bin/env python3
"""Refractory cross-correlogram: a temporal, shape-independent merge gate.

Two spikes from the *same* neuron cannot occur within its refractory period, so the cross-correlogram of
two over-split fragments of one neuron shows a dip at short lag, while two *distinct* neurons fire
independently and show no dip.  This is orthogonal to waveform shape and amplitude -- it is the one signal
that can separate two cells with identical templates, which no shape/variance statistic can (their
distributions overlap).

The gate compares the coincidences observed within +/- refrac to the number expected if the two trains were
independent.  Crucially it is POWER-AWARE: at low firing rates even independent neurons almost never
coincide within a refractory window, so the test has no power and the gate ABSTAINS rather than mislead.
It only VETOes a merge when there are enough expected coincidences to actually see a dip and none appears.

    verdict 'allow'   refractory dip present            -> same-neuron-consistent
    verdict 'veto'    powered but no dip (ratio > thr)  -> distinct neurons; block the merge
    verdict 'abstain' too few expected coincidences     -> defer to the other gates

All times are integer sample indices; sorted inputs are required for the searchsorted counts.
"""
import numpy as np


def refrac_samples(refrac_ms, sr):
    return int(round(float(refrac_ms) * float(sr) / 1000.0))


def cross_coincidences(t_a, t_b, refrac, censor=0):
    """Number of (a, b) spike pairs with censor < |t_a - t_b| <= refrac.  The censor band removes
    near-simultaneous duplicates (the same physical spike landing in both fragments), which would
    otherwise masquerade as a perfect dip.  t_a, t_b must be sorted."""
    t_a = np.asarray(t_a); t_b = np.asarray(t_b)
    lo = np.searchsorted(t_a, t_b - refrac, "left")
    hi = np.searchsorted(t_a, t_b + refrac, "right")
    total = int((hi - lo).sum())
    if censor > 0:
        lo2 = np.searchsorted(t_a, t_b - censor, "left")
        hi2 = np.searchsorted(t_a, t_b + censor, "right")
        total -= int((hi2 - lo2).sum())
    return total


def expected_coincidences(n_a, n_b, duration, refrac, censor=0):
    """Coincidences expected under independent (uniform) firing over the +/- refrac band minus censor."""
    width = 2.0 * (refrac - censor)
    return width * n_a * n_b / float(duration) if duration > 0 else 0.0


def refractory_ratio(t_a, t_b, duration, refrac, censor=0):
    """(c_obs, c_exp, ratio); ratio ~0 => strong dip (same neuron), ~1 => independent (distinct)."""
    c_obs = cross_coincidences(t_a, t_b, refrac, censor)
    c_exp = expected_coincidences(len(t_a), len(t_b), duration, refrac, censor)
    return c_obs, c_exp, (c_obs / c_exp if c_exp > 0 else float("nan"))


def refractory_gate(t_a, t_b, duration, refrac, thr=0.3, min_exp=5.0, censor=0):
    """Power-aware verdict for a proposed merge of two spike trains (see module docstring)."""
    c_obs, c_exp, r = refractory_ratio(t_a, t_b, duration, refrac, censor)
    if c_exp < min_exp:
        verdict = "abstain"
    elif r > thr:
        verdict = "veto"
    else:
        verdict = "allow"
    return dict(verdict=verdict, ratio=r, c_obs=c_obs, c_exp=c_exp, powered=c_exp >= min_exp)


def isi_violation_fraction(t, refrac, censor=0):
    """Fraction of consecutive ISIs in (censor, refrac] -- single-train refractory contamination."""
    t = np.asarray(t)
    if t.size < 2:
        return 0.0
    isi = np.diff(np.sort(t))
    viol = (isi > censor) & (isi <= refrac)
    return float(viol.sum()) / (t.size - 1)


# ───────────────────────────── standalone QC CLI ─────────────────────────────
def main():
    import argparse
    from . import neuro_io as nio
    from . import fiber_session as fs
    from . import session_yaml as sy
    ap = argparse.ArgumentParser(
        description="Refractory QC for a group's clustering: per-cluster ISI-violation fraction, and the "
                    "cluster pairs whose refractory cross-correlogram shows a dip (merge-consistent).")
    sy.add_session_args(ap)
    ap.add_argument("--clu-method", default="stderiv")
    ap.add_argument("--variant", "--clu-stage", dest="variant", default="refine")
    ap.add_argument("--in-clu", default=None, help="explicit .clu path")
    ap.add_argument("--refrac-ms", type=float, default=1.5, help="refractory window (ms, default 1.5)")
    ap.add_argument("--censor-ms", type=float, default=0.3, help="duplicate censor band (ms, default 0.3)")
    ap.add_argument("--thr", type=float, default=0.3, help="ratio at/below which a pair shows a dip")
    ap.add_argument("--min-exp", type=float, default=5.0, help="min expected coincidences to have power")
    ap.add_argument("--min-cluster", type=int, default=40, help="ignore clusters smaller than this")
    ap.add_argument("--top", type=int, default=15, help="how many merge-consistent pairs to list")
    a = ap.parse_args()

    cfg = sy.resolve_session_params(a.session, a.group, channels=a.channels, ntotal=a.ntotal,
                                    nchan=a.nchan, nsamp=a.nsamp, sr=a.sr)
    base = cfg["base"]; elec = a.group; sr = cfg["sr"]
    res = fs.read_res(base, elec)
    if a.in_clu:
        _, clu = nio.read_clu_file(a.in_clu, n_spikes=len(res))
    else:
        _, clu = nio.read_clu_at(base, elec, variant=a.clu_method, tag=a.variant, n_spikes=len(res))
    refrac = refrac_samples(a.refrac_ms, sr); censor = refrac_samples(a.censor_ms, sr)
    duration = float(res.max() - res.min())
    ids = [int(c) for c in np.unique(clu) if c > 1 and int((clu == c).sum()) >= a.min_cluster]
    times = {c: np.sort(res[clu == c]) for c in ids}

    print("per-cluster refractory contamination (ISI <= %.2f ms):" % a.refrac_ms)
    for c in ids:
        f = isi_violation_fraction(times[c], refrac, censor)
        flag = "  high" if f > 0.01 else ""
        print("  cluster %-6d n=%-7d ISI-viol %.4f%s" % (c, times[c].size, f, flag))

    pairs = []
    powered = 0
    for ii in range(len(ids)):
        for jj in range(ii + 1, len(ids)):
            g = refractory_gate(times[ids[ii]], times[ids[jj]], duration, refrac,
                                thr=a.thr, min_exp=a.min_exp, censor=censor)
            if g["powered"]:
                powered += 1
            if g["verdict"] == "allow":
                pairs.append((ids[ii], ids[jj], g["ratio"], g["c_exp"]))
    total = len(ids) * (len(ids) - 1) // 2
    print("\n%d/%d cluster pairs have refractory power (C_exp >= %.0f)" % (powered, total, a.min_exp))
    if powered == 0:
        print("  -> rates too low for the refractory test to discriminate on this group "
              "(it will ABSTAIN in the merge path, deferring to shape/warp).")
    pairs.sort(key=lambda p: p[2])
    for c1, c2, r, ce in pairs[:a.top]:
        print("  dip: cluster %-6d + %-6d  ratio %.3f  (C_exp %.1f) -> merge-consistent" % (c1, c2, r, ce))


if __name__ == "__main__":
    main()
