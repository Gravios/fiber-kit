#!/usr/bin/env python3
# test_custody_token_conformance.py — runs the shared chain-of-custody vectors
# (custody_vectors.tsv, an identical copy of the canonical table in neurosuite-3
# libneurosuite-core/test/) against neuro_io's variant-token parser.
#
# neurosuite-3 keeps three implementations of the token grammar in step with this
# table — custody.hpp (C++), ndm_custody (bash) and ndm_resolve_io (Python).
# fiber-kit's parse_variant_token is a FOURTH implementation, in a different repo,
# and until this runner existed nothing tied it to the other three: a change to the
# grammar there would have gone unnoticed here, and resolve_input would quietly stop
# matching the tokens neurosuite-3 writes.
#
# fiber-kit mirrors only the token-grammar kinds. The other kinds in the table
# (classify / method_of / parse_anchor / resolve) describe naming policy fiber-kit
# does not implement — it has session_path, not a path classifier — so they are
# skipped, and the skipped kinds are REPORTED rather than silently ignored, so a kind
# added upstream shows up here as something to look at.
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

try:
    from fiber_kit import neuro_io as nio
except ImportError:
    sys.path.insert(0, os.path.join(HERE, "..", "src", "fiber_kit"))
    import neuro_io as nio

VECTORS = os.environ.get("CUSTODY_VECTORS",
                         os.path.join(HERE, "custody_vectors.tsv"))

MIRRORED = ("method_token", "is_stderiv")

fails = 0
ran = 0
skipped = {}


def check(ok, what):
    global fails, ran
    ran += 1
    if not ok:
        print(f"FAIL: {what}")
        fails += 1


with open(VECTORS) as fh:
    for line in fh:
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        f = line.split("\t")
        kind = f[0]

        if kind == "method_token":
            # <token> <family> <kind> <order>; kind/order empty when absent.
            token = f[1]
            want = (f[2], f[3] or None, int(f[4]) if f[4] else None)
            spec = nio.parse_variant_token(token)
            check((spec.family, spec.kind, spec.order) == want,
                  f"parse_variant_token({token!r}) == {want} (got {tuple(spec)})")
            # variant_family stays consistent with the full parse.
            check(nio.variant_family(token) == want[0],
                  f"variant_family({token!r}) == {want[0]!r}")

        elif kind == "is_stderiv":
            token, want = f[1], (f[2] == "1")
            got = nio.is_stderiv_variant(token)
            check(got == want, f"is_stderiv_variant({token!r}) == {want} (got {got})")

        else:
            skipped[kind] = skipped.get(kind, 0) + 1

print(f"custody token conformance (fiber-kit): {ran} checks, {fails} failed")
if skipped:
    detail = ", ".join(f"{k}×{n}" for k, n in sorted(skipped.items()))
    print(f"  not mirrored by fiber-kit, skipped: {detail}")
    unexpected = sorted(set(skipped) - {"classify", "method_of",
                                        "parse_anchor", "resolve"})
    if unexpected:
        print(f"  NOTE: unrecognised vector kind(s) {unexpected} — the canonical "
              f"table gained something; check whether fiber-kit should mirror it")
if fails == 0:
    print("ALL CUSTODY TOKEN CONFORMANCE TESTS PASS")
sys.exit(0 if fails == 0 else 1)
