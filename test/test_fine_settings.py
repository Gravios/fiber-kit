#!/usr/bin/env python3
"""test_fine_settings.py — cluster_chunk_fine takes its settings as one object.

The function had 96 parameters; 85 of them were settings, passed one by one through three layers.
They are the fields of fiber_session.FineSettings now.  Checked here:

  * a keyword the old signature took still works and gives the labels the settings object gives;
  * an unknown keyword is refused (it was before, by the signature), and so is a positional
    argument where the settings go (the 8th argument used to be `method`);
  * every keyword fiber-session, fiber-refine and raw_vs_stderiv pass to the function is a field
    or one of the four data arguments, read from their source;
  * the body reads no setting as a bare name any more (each is s.<name>).
"""
import ast
import dataclasses
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src", "fiber_kit")
sys.path.insert(0, os.path.join(ROOT, "src"))
from fiber_kit import fiber_session as fs, fiber_lib as fl      # noqa: E402

fails = 0
DATA = {"basis", "shed", "waves_std", "candidates_out"}


def check(ok, what):
    global fails
    fails += not ok
    print(("  ok:   " if ok else "  FAIL: ") + what)


def main():
    fields = {f.name for f in dataclasses.fields(fs.FineSettings)}
    check(len(fields) == 85, f"FineSettings has 85 fields ({len(fields)})")

    rng = np.random.default_rng(3)
    n, nsamp, nch = 600, 32, 3
    w = rng.normal(0, 30, (n, nsamp, nch))
    unit = rng.integers(0, 2, n)
    shape = np.zeros(nsamp)
    shape[12:20] = np.hanning(8)
    amp = np.array([[-2500.0, -600, -100], [-300.0, -900, -2400]])
    w += shape[None, :, None] * amp[unit][:, None, :] * (1 + 0.05 * rng.standard_normal(n))[:, None, None]
    res = np.sort(rng.integers(0, 600000, n))
    mask = fl.MASK_FULL
    X = w[:, mask, :].reshape(n, -1)
    nmean = X.mean(0)
    W = np.eye(X.shape[1]) / 30.0                              # the noise is white with sd 30
    by_kw, _ = fs.cluster_chunk_fine(w, res, W, nmean, 40, mask, 20000.0, method="none", dipsplit=False)
    by_obj, _ = fs.cluster_chunk_fine(w, res, W, nmean, 40, mask, 20000.0,
                                      fs.FineSettings(method="none", dipsplit=False))
    mixed, _ = fs.cluster_chunk_fine(w, res, W, nmean, 40, mask, 20000.0, fs.FineSettings(method="gmm"),
                                     method="none", dipsplit=False)
    check(np.array_equal(by_kw, by_obj) and np.array_equal(by_kw, mixed) and (by_kw >= 0).any(),
          "keywords, a settings object, and a keyword over a settings object give the same labels")
    for what, call in (("an unknown keyword", lambda: fs.cluster_chunk_fine(w, res, W, nmean, 40, mask, 20000.0, no_such=1)),
                       ("a positional method", lambda: fs.cluster_chunk_fine(w, res, W, nmean, 40, mask, 20000.0, "none"))):
        try:
            call()
            check(False, f"{what} is refused")
        except TypeError:
            check(True, f"{what} is refused")

    passed = set()
    for mod in ("fiber_session.py", "fiber_refine.py", "raw_vs_stderiv.py", "fiber_stochastic.py"):
        with open(os.path.join(SRC, mod)) as fh:
            tree = ast.parse(fh.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", getattr(node.func, "id", "")) == "cluster_chunk_fine":
                passed |= {k.arg for k in node.keywords if k.arg}
            # the keyword set fiber-session and fiber-stochastic hand to the chunk worker: build_cf()
            if isinstance(node, ast.FunctionDef) and node.name == "build_cf":
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Call) and getattr(sub.func, "id", "") == "dict":
                        passed |= {k.arg for k in sub.keywords if k.arg}
                    if isinstance(sub, ast.Dict):
                        passed |= {k.value for k in sub.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}
                    if (isinstance(sub, ast.Subscript) and isinstance(sub.ctx, ast.Store)
                            and isinstance(sub.slice, ast.Constant) and isinstance(sub.slice.value, str)):
                        passed.add(sub.slice.value)
    passed -= {"coarse_mg", "mask", "sr"}
    check(len(passed) >= 60, f"{len(passed)} keywords are passed to the function across the source")
    check(passed <= fields | DATA, f"each is a field or a data argument ({sorted(passed - fields - DATA)})")

    with open(os.path.join(SRC, "fiber_session.py")) as fh:
        tree = ast.parse(fh.read())
    fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "cluster_chunk_fine")
    bare = sorted({x.id for x in ast.walk(fn) if isinstance(x, ast.Name) and x.id in fields})
    check(not bare, f"the body reads no setting as a bare name ({bare[:5]})")
    check(len(fn.args.args) + len(fn.args.kwonlyargs) <= 12, "the function takes 12 named parameters or fewer")
    print(f"test_fine_settings: {'PASS' if not fails else 'FAIL'} ({fails} failed)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
