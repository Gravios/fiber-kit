"""The HYBRID fine-split feature space: stderiv (lag) features + standard PCs of the paired raw window.

Why: the derivative spaces lose broad, low-energy units -- on the 20120312 g6 analysis d'(C5)/d'(std)
falls 0.93 -> 0.61 -> 0.44 -> 0.26 as the spike widens 1x -> 1.5x -> 2x -> 3x, at every amplitude, and
the C5 operator is near-singular so the raw window cannot be recovered from the stderiv one.  So the
raw block has to be read from .spk.standard and ride along.

Unit level (fiber_lib / fiber_pca):
  * set_align_channels(n): the aligner's xcorr and centroid seed sum over the first n channels only,
    and the shift applies to all -- the raw block follows the stderiv alignment, never votes in it;
  * hybrid_features on a hybrid_stack'ed array = [method block | scale * gain * standard PCs of the
    raw block], with the method block identical to the non-hybrid features of the method array;
  * a hybrid basis on a plain (non-stacked) array still gives the method block alone.

End to end (the real CLI on a synthetic session): two units whose stderiv windows are the SAME
template and whose raw windows differ in width.  Without the hybrid the fine split sees one unit;
with FK_SESSION_FEAT_HYBRID=1 it sees two, the hybrid .fet is named <method>_D<lag><per+pcs>
(stderiv_C5_D36 for lag 3: 4 lag columns + 2 raw PCs per channel) and has nch x 6 + 1 columns.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fiber_kit import fiber_lib as fl, fiber_pca as fpca, neuro_io as nio   # noqa: E402

SR = 20000.0
NSAMP, NCH, NTOTAL = 32, 3, 4
SETS3 = "0-1+2,1-0+2,2-0+1"
PEAK = 12
N_PER = 400


def _templates():
    t = np.arange(NSAMP, dtype=float)
    narrow = -np.exp(-0.5 * ((t - PEAK) / 1.6) ** 2)
    broad = -np.exp(-0.5 * ((t - PEAK) / 4.0) ** 2)
    T = np.zeros((2, NSAMP, NCH))
    T[0, :, 0] = 3000 * narrow; T[0, :, 1] = 1500 * narrow
    T[1, :, 0] = 3000 * broad; T[1, :, 1] = 1500 * broad
    return T


def _build_session(td, rng):
    base = os.path.join(td, "mini")
    t0, t1 = int(2 * SR), int(58 * SR)
    res = np.sort(rng.integers(t0, t1, 2 * N_PER)).astype(np.int64)
    unit = rng.integers(0, 2, len(res))
    res.tofile(f"{base}.res.1")
    T = _templates()
    raw = np.zeros((len(res), NSAMP, NCH))
    for u in (0, 1):
        raw[unit == u] = T[u]
    raw += rng.normal(0, 60, raw.shape)
    raw.astype(np.int16).tofile(f"{base}.spk.standard.1")
    # the "stderiv" windows: ONE template for both units (the derivative space cannot tell them apart)
    sd = np.zeros_like(raw)
    sd[:] = 0.5 * (T[0] + T[1])
    sd += rng.normal(0, 60, sd.shape)
    sd.astype(np.int16).tofile(f"{base}.spk.stderiv_C5.1")
    # bases: 3 comps, window [4, 26) so a lag of 3 fits on both sides
    for tok, W in (("standard", raw), ("stderiv_C5", sd)):
        means, evec = fpca.fit_basis(fpca.extract_windows(W, 4, 22), nComp=3)
        fpca.write_pcad(f"{base}.pca.{tok}.1", means, evec, 4, centered=1,
                        method=int(fpca.Method.STANDARD) if tok == "standard" else int(fpca.Method.STDERIV_CUSTOM_CAR))
    nfil = int(60 * SR)
    rng.integers(-30, 30, nfil * NTOTAL).astype(np.int16).tofile(f"{base}.fil")
    return base, res, unit


def _unit_checks(check, rng):
    T = _templates()
    n = 200
    shifts = rng.integers(-3, 4, n)
    wm = np.zeros((n, NSAMP, NCH)); wr = np.zeros((n, NSAMP, NCH))
    for i in range(n):
        wm[i] = np.roll(T[0], shifts[i], axis=0); wr[i] = np.roll(T[1], shifts[i], axis=0)
    wm += rng.normal(0, 30, wm.shape); wr += rng.normal(0, 30, wr.shape)
    stack = fpca.hybrid_stack(wm, wr)
    check("hybrid_stack appends the raw block as trailing channels", stack.shape == (n, NSAMP, 2 * NCH)
          and np.array_equal(stack[:, :, NCH:], wr))
    # the aligner: method channels vote; one shift for both blocks
    fl.set_align_channels(NCH)
    try:
        al_m, sh_m = fl.align_xcorr(wm, ref="median", iters=6, maxlag=6, return_shifts=True)
        al_s, sh_s = fl.align_xcorr(stack, ref="median", iters=6, maxlag=6, return_shifts=True)
    finally:
        fl.set_align_channels(0)
    check("with the lever the stacked array's shifts are the method array's", np.allclose(sh_m, sh_s))
    check("the method block of the aligned stack is the aligned method array", np.allclose(al_s[:, :, :NCH], al_m))
    al_r = fl.align_xcorr(wr, ref="median", iters=6, maxlag=6, return_shifts=False)
    check("the raw block was shifted by the method shift, not its own",
          not np.allclose(al_s[:, :, NCH:], al_r, atol=1e-6) or np.allclose(sh_m, fl.align_xcorr(wr, return_shifts=True)[1]))
    # features
    means, evec = fpca.fit_basis(fpca.extract_windows(wm, 4, 22), nComp=3)
    bm = dict(means=means, evec=evec, recShift=4, data2use=22, centered=1, _lag=3, _lag_pc2=True)
    means, evec = fpca.fit_basis(fpca.extract_windows(wr, 4, 22), nComp=3)
    bs = dict(means=means, evec=evec, recShift=4, data2use=22, centered=1)
    bm["_hyb"] = fpca.hybrid_basis(bs, pcs=2, scale=0.5, gain=1.0, nraw=NCH)
    check("hybrid_basis cuts the standard basis to `pcs` components", bm["_hyb"]["basis"]["evec"].shape == (NCH, 2, 22))
    F = fpca.cluster_features(al_s, bm, realign=False)
    plain = dict(bm); plain.pop("_hyb")
    Fm = fpca.cluster_features(al_m, plain, realign=False)
    check("hybrid features are [method lag block | raw block]: width nch x (4 + 2)", F.shape == (n, NCH * 6))
    check("the method block equals the non-hybrid lag features", np.allclose(F[:, :NCH * 4], Fm))
    Fr = fpca.project(fpca.extract_windows(al_s[:, :, NCH:], 4, 22), bm["_hyb"]["basis"])
    check("the raw block is scale x gain x the standard projection", np.allclose(F[:, NCH * 4:], 0.5 * Fr))
    check("a hybrid basis on a plain method array gives the method block alone",
          np.allclose(fpca.cluster_features(al_m, bm, realign=False), Fm))
    g = fpca.fit_hybrid_gain(wm, wr, bm)
    check("fit_hybrid_gain is positive and finite", np.isfinite(g) and g > 0)


def main():
    ok = 0

    def check(name, cond):
        nonlocal ok
        assert cond, name
        ok += 1
        print(f"  ok  {name}")

    rng = np.random.default_rng(0)
    _unit_checks(check, rng)
    with tempfile.TemporaryDirectory() as td:
        base, res, unit = _build_session(td, rng)
        env = {k: v for k, v in os.environ.items() if not k.startswith("FK_") and not k.startswith("FIBER_")}
        env["PYTHONPATH"] = os.pathsep.join(
            [os.path.join(os.path.dirname(__file__), "..", "src"), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
        argv = [sys.executable, "-m", "fiber_kit.fiber_session", base, "1",
                "--channels", "0,1,2", "--ntotal", str(NTOTAL), "--nsamp", str(NSAMP),
                "--sr", str(int(SR)), "--method", "stderiv_C5", "--sdiff-pairs", SETS3,
                "--no-link", "--chunk-min", "1", "--overlap-min", "0", "--min-group", "50",
                "--fine-method", "rkk", "--feat-lag", "3", "--emit-fet", "--realign", "off", "--jobs", "1"]

        def run(extra, stage):
            r = subprocess.run(argv + ["--clu-stage", stage] + extra, capture_output=True, text=True, env=env, timeout=600)
            out = r.stdout + r.stderr
            if r.returncode != 0:
                print(out[-4000:])
            return r.returncode, out

        rc, out = run(["--feat-hybrid", "0"], "plain")
        check("the non-hybrid run exits 0", rc == 0)
        clu_p = np.fromfile(f"{base}.clu.stderiv_C5.1.plain", np.int32)[1:]
        n_plain = len([c for c in np.unique(clu_p) if c > 1])
        rc, out = run(["--feat-hybrid", "1", "--hybrid-pcs", "2", "--hybrid-scale", "1.0"], "hyb")
        check("the hybrid run exits 0", rc == 0)
        check("the banner names the hybrid block", "hybrid features: + standard basis 3ch x 2" in out)
        check("--emit-fet names the hybrid space <method>_D36 (4 lag + 2 raw per channel)",
              "stderiv_C5_D36" in out and os.path.exists(f"{base}.fet.stderiv_C5_D36.1"))
        nd = int(np.fromfile(f"{base}.fet.stderiv_C5_D36.1", np.int32, 1)[0])
        check("the hybrid .fet has nch x 6 features + the timestamp", nd == NCH * 6 + 1)
        fet = np.fromfile(f"{base}.fet.stderiv_C5_D36.1", np.int64, offset=4).reshape(-1, nd)
        check("its last column is the .res", np.array_equal(fet[:, -1], res))
        check("the raw block of the .fet separates the two units (the method block cannot)",
              _sep(fet[:, NCH * 4:-1], unit) > 3 * _sep(fet[:, :NCH * 4], unit))
        check("no .pca is written for the hybrid space", not os.path.exists(f"{base}.pca.stderiv_C5_D36.1"))
        clu_h = np.fromfile(f"{base}.clu.stderiv_C5_D36.1.hyb", np.int32)[1:]
        n_hyb = len([c for c in np.unique(clu_h) if c > 1])
        pp, ph = _purity(clu_p, unit), _purity(clu_h, unit)
        print(f"      fine clusters: plain {n_plain} (purity {pp:.2f}), hybrid {n_hyb} (purity {ph:.2f})")
        check("the derivative-only split cannot tell the two units apart (purity < 0.75)", pp < 0.75)
        check("the hybrid clu recovers the two units (purity >= 0.9)", ph >= 0.9)
    print(f"test_session_hybrid: {ok}/20 checks passed")


def _sep(F, unit):
    """Fisher separation of the two units in feature block F (d' with the pooled covariance)."""
    A, B = F[unit == 0].astype(float), F[unit == 1].astype(float)
    sc = F.std(0) + 1e-9; A, B = A / sc, B / sc
    d = A.mean(0) - B.mean(0)
    S = 0.5 * (np.cov(A.T) + np.cov(B.T)) + 1e-6 * np.eye(F.shape[1])
    return float(np.sqrt(d @ np.linalg.solve(S, d)))


def _purity(clu, unit):
    """Spike-weighted purity of the clusters (>1) against the two true units."""
    tot = pure = 0
    for c in np.unique(clu):
        if c <= 1:
            continue
        m = clu == c
        pure += max((unit[m] == 0).sum(), (unit[m] == 1).sum()); tot += m.sum()
    return pure / max(tot, 1)


if __name__ == "__main__":
    main()
