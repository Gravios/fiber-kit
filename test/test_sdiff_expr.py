#!/usr/bin/env python3
# test_sdiff_expr.py -- runs the shared expression-pattern vectors (sdiff_expr_vectors.tsv, an
# identical copy of the canonical table in neurosuite-3 libneurosuite-core/test/) against
# fiber_stderiv, and checks the order-6 transform against the older forms it generalises.  The
# C++ runner (sdiff_expr_test.cpp) checks the same table against channel_expr.hpp, so the two
# grammars cannot drift.
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

try:
    from fiber_kit import fiber_stderiv as fst
except ImportError:
    sys.path.insert(0, os.path.join(HERE, "..", "src", "fiber_kit"))
    import fiber_stderiv as fst

VECTORS = os.environ.get("SDIFF_EXPR_VECTORS", os.path.join(HERE, "sdiff_expr_vectors.tsv"))

fails = 0
ran = 0


def check(ok, what):
    global fails, ran
    ran += 1
    if not ok:
        print(f"FAIL: {what}")
        fails += 1


evals = rejects = 0
with open(VECTORS) as fh:
    for line in fh:
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        f = line.split("\t")
        if f[0] == "eval":
            evals += 1
            exprs = fst.parse_sdiff_exprs(f[1])
            x = [float(v) for v in f[2].split()]
            want = [float(v) for v in f[3].split()]
            check(len(exprs) == len(want), f"one output per element: {f[1]}")
            for k, (e, w) in enumerate(zip(exprs, want)):
                got = float(e.eval(lambda i, x=x: x[i]))
                check(abs(got - w) <= 1e-9 * max(1.0, abs(w)), f"{f[1]} output {k}: got {got} want {w}")
        elif f[0] == "reject":
            rejects += 1
            try:
                fst.parse_sdiff_exprs(f[1])
                check(False, f"rejected: {f[1]}")
            except ValueError as e:
                check(bool(str(e)), f"a reason is given for: {f[1]}")
        else:
            check(False, f"unknown vector line: {line}")
check(evals >= 10 and rejects >= 10, "the vector file has its cases")

# The form a spec selects.
check(fst.sdiff_spec_order("0-1,1-2,2-3") == 4 and fst.sdiff_spec_order("0-1+2,1-0,2-0") == 5
      and fst.sdiff_spec_order("c0-(c1+c2)/2,c1,c2") == 6, "the order a spec selects")
check(not fst.sdiff_spec_uses_sets("c0-(c1+c2)/2,c1,c2"), "an expression's '+' is not a reference set")

# Order 6 reproduces orders 4 and 5 when written as the same arithmetic, bit for bit.
rng = np.random.default_rng(3)
raw = rng.integers(-3000, 3000, size=(50, 33, 4)).astype(np.float64)
partner, _ = fst.parse_sdiff_pairs("0-1,1-2,2-3")
check(np.array_equal(fst.apply_stderiv_transform(raw, partner=partner),
                     fst.apply_stderiv_transform(raw, exprs=fst.parse_sdiff_exprs("c0-c1,c1-c2,c2-c3,c3-c2"))),
      "order 6 == order 4 written as differences")
sets = fst.parse_sdiff_sets("0-1+2,1-0+2,2-1+3,3-2")
check(np.array_equal(fst.apply_stderiv_transform(raw, sets=sets),
                     fst.apply_stderiv_transform(raw, exprs=fst.parse_sdiff_exprs("0-1+2,1-0+2,2-1+3,3-2"))),
      "an older-form spec given to parse_sdiff_exprs computes the same values")
# A session mixing forms runs _C6 for every group: an older-form group's pattern is read as the
# expressions it computes (fiber_envelope / fiber_realign), with the values of its own order.
try:
    from fiber_kit import fiber_envelope as fe
except ImportError:
    import fiber_envelope as fe
check(np.array_equal(fe.space_transform("stderiv_C6", "0-1,1-2,2-3")(raw),
                     fe.space_transform("stderiv_C4", "0-1,1-2,2-3")(raw)),
      "a _C6 space with an older-form group pattern: the same values as its own order")
try:
    fst.parse_sdiff_exprs("(" * 400 + "c0" + ")" * 400)
    check(False, "deep nesting is refused")
except ValueError:
    pass                                     # not RecursionError
try:
    fst.apply_stderiv_transform(raw, exprs=fst.parse_sdiff_exprs("c0-c1,c1-c0"))
    check(False, "two expressions for four channels are refused")
except ValueError:
    pass

print(f"sdiff_expr: {ran - fails}/{ran} checks passed ({evals} eval, {rejects} reject vectors)")
sys.exit(1 if fails else 0)
