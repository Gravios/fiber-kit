#!/usr/bin/env python3
"""`truth: core:` scores each curated unit on the spikes that fit its own template.  A truth unit A whose
label set holds 150 spikes of unit B's shape (a contaminated interneuron) is scored as a label set: a sort
that separates the two shapes perfectly gets A at 0.73 accuracy and a MERGE charged to B's cluster for the
150 spikes the truth calls A.  With the core, the 150 fall outside A's residual threshold, count for no unit,
and the same sort scores A at 1.0 with no merge; B is untouched; clean units lose nothing."""
import os, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from fiber_kit import fiber_tune as ft, neuro_io as nio   # noqa: E402

fails = 0


def check(ok, what):
    global fails
    print(("  ok:   " if ok else "  FAIL: ") + what); fails += not ok


T, C, SR = 32, 4, 20000.0
rng = np.random.default_rng(2)
t = np.arange(T) - 15
shapeA = (-900 * np.exp(-0.5 * (t / 1.5) ** 2))[:, None] * np.array([1.0, 0.6, 0.2, 0.1])[None, :]
shapeB = (-500 * np.exp(-0.5 * (t / 3.5) ** 2) + 150 * np.exp(-0.5 * ((t - 7) / 4) ** 2))[:, None] * np.array([0.2, 0.5, 1.0, 0.6])[None, :]
nA, nB, nX = 400, 400, 150                       # A, B, and B-shaped spikes the truth labels A
shapes = [shapeA] * nA + [shapeB] * nB + [shapeB] * nX
N = len(shapes)
res = np.sort(rng.choice(np.arange(1000, 2_000_000, 50), N, replace=False)).astype(np.int64)
jit = rng.integers(-1, 2, N)
spk = np.stack([np.roll(s, int(j), axis=0) + rng.normal(0, 25, (T, C)) for s, j in zip(shapes, jit)])
spk = np.clip(np.rint(spk), -32768, 32767).astype(np.int16)
truth_lab = np.array([2] * nA + [3] * nB + [2] * nX)          # the contamination: B-shaped spikes called A (2)
sort_lab = np.array([2] * nA + [3] * nB + [3] * nX)           # the perfect sort by shape
with tempfile.TemporaryDirectory() as d:
    base = os.path.join(d, "rat")
    nio.write_spk(base, 1, spk, variant="standard"); nio.write_res(base, 1, res, variant="standard")
    nio.write_clu_file(f"{base}.gt.clu", truth_lab); res.tofile(f"{base}.gt.res")
    tr = dict(clu=f"{base}.gt.clu", res=f"{base}.gt.res", tol=2)
    plain = ft.Truth(tr, SR)
    M0 = ft.score_labels(sort_lab, plain.matched(res), plain, plain.matched_all(res))
    lines = []
    core = ft.Truth(dict(tr, core=dict(k=3.0, min_n=50)), SR, spk_of=(base, 1, T, C), log=lines.append)
    M1 = ft.score_labels(sort_lab, core.matched(res), core, core.matched_all(res))
    rowA = next(r for r in core.core_rows if r["unit"] == 2); rowB = next(r for r in core.core_rows if r["unit"] == 3)
    print(f"    label set: all_acc {M0['all_acc']:.3f} merges {M0['merges']}   core: all_acc {M1['all_acc']:.3f} merges {M1['merges']}")
    print(f"    core rows: A {rowA}  B {rowB}")
    check(M0["all_acc"] < 0.9 and M0["merges"] >= 1, "as a label set the perfect sort is charged (A below 0.9, a merge)")
    check(rowA["core"] <= nA + 5 and rowA["core"] >= nA - 5, f"A's core is its {nA} own-shaped spikes ({rowA['core']})")
    check(rowB["core"] >= nB - 5, f"B's core keeps B ({rowB['core']} of {nB})")
    check(M1["all_acc"] > 0.98 and M1["merges"] == 0, "scored on the core the same sort is perfect and unmerged")
    check(core.id != plain.id and "core:k=3" in core.id, "the truth id carries the core (trials scored without it are rescored)")
    g1 = core.matched(res); ga = core.matched_all(res)
    check(int((g1[nA + nB:] == -1).sum()) >= nX - 5 and int((ga[nA + nB:] == -1).sum()) >= nX - 5,
          "the spikes outside A's core are matched to no truth at all (neither unit nor noise)")
    check(any("core unit" in ln for ln in lines), "the per-unit core is logged")
    try:
        ft.Truth(dict(tr, core=dict(k=3.0)), SR)
        check(False, "core without windows is refused")
    except SystemExit:
        check(True, "core without the session's windows is refused")
print(f"test_truth_core: {'PASS' if not fails else 'FAIL'} ({fails} failed)")
sys.exit(1 if fails else 0)
