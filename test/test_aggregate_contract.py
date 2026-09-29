"""aggregate_units keeps the full signature contract across grouping passes.

The reported crash: fiber-intrachunk with gate='band' (the exp config) died on
pass 2 of group_intrachunk_iter with KeyError: 'sigma' -- aggregate_units'
collapsed dict dropped every key beyond template/position, so the band gate
(sigma), the kernel gates (feat), the dual off-thr gate (celltype) and the
ccg/refrac gates (times) all lost their inputs after the first pass.  The
aggregate now pools them: law-of-total-variance sigma, _merge_var variance,
unioned spike times, n-weighted-majority celltype, re-capped feature unions.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import fiber_kit.fiber_intrachunk as fic  # noqa: E402


def _toy_sig(M=6, ns=20, nch=2, rng=None):
    rng = rng or np.random.default_rng(0)
    base = np.zeros((ns, nch)); base[8:14, 0] = -np.hanning(6) * 10
    T = np.stack([base + 0.01 * rng.standard_normal((ns, nch)) for _ in range(M)]).astype(np.float32)
    return dict(
        ids=np.arange(2, 2 + M), template=T,
        sigma=np.full((M, ns, nch), 0.5, np.float32),
        offset=np.zeros((M, nch), np.float32),
        x0=np.zeros(M), y0=np.zeros(M), z0=np.zeros(M), A=np.full(M, 100.0),
        chunk=np.zeros(M, int), t_mid=np.linspace(1, 2, M),
        n=np.full(M, 100, int), var=np.full(M, 1.0),
        times=np.array([np.sort(rng.uniform(0, 60, 50)) for _ in range(M)], dtype=object),
        feat=np.array([rng.standard_normal((30, 4)) for _ in range(M)], dtype=object),
        celltype=np.array([1, 1, 0, 0, 1, 0]),
        shape_null=np.full(M, 0.3),
    )


def main():
    ok = 0

    def check(name, cond):
        nonlocal ok
        assert cond, name
        ok += 1
        print(f"  ok  {name}")

    sig = _toy_sig()
    lab = np.array([0, 0, 1, 1, 2, 2])          # three pairwise merges
    u = fic.aggregate_units(sig, lab)
    u["ids"] = u["unit"]

    need = ("sigma", "var", "times", "feat", "celltype", "shape_null")
    check("aggregate carries every gate input " + str(need),
          all(k in u for k in need))
    check("sigma keeps per-sample shape", u["sigma"].shape == (3, 20, 2))
    check("times are unioned (50+50 per pair)",
          all(len(u["times"][g]) == 100 for g in range(3)))
    check("feat unions are re-capped to 80 rows",
          all(np.asarray(u["feat"][g]).shape == (60, 4) for g in range(3)))
    check("celltype is the n-weighted majority", list(u["celltype"]) == [1, 0, 1])

    # pooled sigma follows the law of total variance: two members with sigma=0 and
    # templates apart by d (equal n) pool to elementwise |d|/2
    s2 = dict(sig)
    s2["sigma"] = np.zeros_like(sig["sigma"])
    s2["template"] = sig["template"].copy()
    s2["template"][1] = s2["template"][0] + 2.0          # d = 2 everywhere
    v = fic.aggregate_units(s2, lab)
    check("pooled sigma = |d|/2 for sigma-0 members two apart",
          np.allclose(np.asarray(v["sigma"][0]), 1.0, atol=1e-6))

    # the regression itself: iterated grouping with gate='band' survives pass >= 2
    lab2 = fic.group_intrachunk_iter(sig, max_iter=3, gate="band", band_thr=0.1,
                                     off_thr=10.0, depth_gate=100.0)
    check("group_intrachunk_iter(gate='band') completes multiple passes (no KeyError)",
          len(lab2) == len(sig["ids"]))

    # _collapse_sig pools sigma too (pre-merge path feeding the band gate)
    c = fic._collapse_sig(s2, lab)
    check("_collapse_sig sigma is pooled, not a representative",
          np.allclose(np.asarray(c["sigma"][0]), 1.0, atol=1e-6))

    print(f"test_aggregate_contract: {ok}/8 checks passed")


if __name__ == "__main__":
    main()
