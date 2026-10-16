#!/usr/bin/env python3
"""test_sdiff_grammar.py -- fiber_stderiv's sdiffPairs parsers against the shared
sdiff_grammar_vectors.tsv, the table neurosuite-3's klustersshared_sdiff_pairs_test runs
against sdiff_pairs.h.  Byte-identical copies in both repositories (test_shared_contract).

Each row: a pattern, the spatial order it selects (sdiff_spec_order), and the partner
map (parse_sdiff_pairs) and reference sets (parse_sdiff_sets) it parses to, or ERR when
the C++ refuses it.  Exits non-zero on failure.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
try:
    from fiber_kit import fiber_stderiv as fst
except ImportError:
    sys.path.insert(0, os.path.join(HERE, "..", "src", "fiber_kit"))
    import fiber_stderiv as fst

fails = rows = 0


def check(ok, what):
    global fails
    if not ok:
        fails += 1
        print("  FAIL: " + what)


def pairs(spec):
    try:
        return " ".join(map(str, fst.parse_sdiff_pairs(spec)[0]))
    except ValueError:
        return "ERR"


def sets(spec):
    try:
        return " ".join("+".join(map(str, s)) for s in fst.parse_sdiff_sets(spec))
    except ValueError:
        return "ERR"


for line in open(os.path.join(HERE, "sdiff_grammar_vectors.tsv"), encoding="utf-8"):
    if not line.strip() or line.startswith("#"):
        continue
    f = line.rstrip("\n").split("\t")
    if len(f) != 4:
        check(False, f"4 columns: {line!r}")
        continue
    rows += 1
    spec, order, want_p, want_s = f
    check(str(fst.sdiff_spec_order(spec)) == order, f"order of {spec!r}")
    if want_p != "-":
        got = pairs(spec)
        check(got == want_p, f"pairs of {spec!r}: got {got!r}, want {want_p!r}")
    if want_s != "-":
        got = sets(spec)
        check(got == want_s, f"sets of {spec!r}: got {got!r}, want {want_s!r}")

check(rows >= 30, f"the table has its rows ({rows})")
print(f"test_sdiff_grammar: {rows} patterns, {fails} failed")
sys.exit(1 if fails else 0)
