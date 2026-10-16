#!/usr/bin/env python3
"""test_stderiv_vectors.py -- fiber_stderiv.apply_stderiv_transform against the shared
stderiv_vectors.tsv, the table neurosuite-3's core_test_stderiv_vectors runs against
stderiv_transform.hpp.  Byte-identical copies in both repositories (test_shared_contract).

Each row: the raw window (channel-major), the raw sample before it, and the int16
stderiv window the extractor writes.  fiber_stderiv takes the window with that sample
prepended (raw_ext) and the order as a map: orders 1 and 2 through plain_order_spec,
4 as partner, 5 as sets, 6 as expressions, 3 (and a map that does not fit) as neither.

Rows fiber_stderiv does not cover are said so, not silently passed:
  order 0          no transform (fiber-kit reads such a .spk as raw)
  1 or 2 on 1 ch   plain_order_spec refuses a group without a neighbour
  6 that does not fit the group: fiber_stderiv refuses it, as the extractor does
                   (core's all-pairs fallback is never reached by an engine)
Exits non-zero on failure.
"""
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

fails = 0
ran = skipped = refused = 0


def check(ok, what):
    global fails
    if not ok:
        fails += 1
        print("  FAIL: " + what)


def ints(text):
    return [int(v) for v in text.split()]


for line in open(os.path.join(HERE, "stderiv_vectors.tsv"), encoding="utf-8"):
    if not line.strip() or line.startswith("#"):
        continue
    f = line.rstrip("\n").split("\t")
    if len(f) != 8:
        check(False, f"8 columns: {line[:60]!r}")
        continue
    name, order, mp = f[0], int(f[1]), f[2]
    n_chan, n_samp = int(f[3]), int(f[4])
    prev, inp, want = ints(f[5]), ints(f[6]), ints(f[7])
    if order == 0 or (order in (1, 2) and n_chan == 1):
        skipped += 1
        continue
    win = np.array(inp, np.int64).reshape(n_chan, n_samp).T            # (nSamp, C)
    raw_ext = np.vstack([np.array(prev, np.int64)[None, :], win])[None]  # (1, nSamp+1, C)
    partner = sets = exprs = None
    if order in (1, 2):
        partner, sets = fst.plain_order_spec(order, n_chan)
    elif order == 4 and mp != "-":
        partner = ints(mp)
    elif order == 5 and mp != "-":
        sets = [[int(m) for m in s.split("+")] for s in mp.split()]
    elif order == 6 and mp != "-":
        exprs = fst.parse_sdiff_exprs(mp)
        try:
            fst.check_exprs_fit(exprs, n_chan)
        except ValueError:
            refused += 1
            try:
                fst.apply_stderiv_transform(raw_ext, exprs=exprs)
                check(False, f"{name}: expressions that do not fit the group are refused")
            except ValueError:
                pass
            continue
    got = fst.apply_stderiv_transform(raw_ext, partner=partner, sets=sets, exprs=exprs)[0]
    got = got.T.reshape(-1).tolist()                                      # channel-major
    ran += 1
    if got != want:
        k = next(i for i in range(len(want)) if got[i] != want[i])
        check(False, f"{name}: channel {k // n_samp} sample {k % n_samp}: got {got[k]}, want {want[k]}")

check(ran >= 25, f"the table's rows ran ({ran})")
print(f"test_stderiv_vectors: {ran} rows checked, {skipped} not covered by fiber_stderiv, "
      f"{refused} refused as the extractor refuses; {fails} failed")
sys.exit(1 if fails else 0)
