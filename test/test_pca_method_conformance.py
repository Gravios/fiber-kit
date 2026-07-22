#!/usr/bin/env python3
# test_pca_method_conformance.py — runs the shared PCAE transform-method vectors
# (pca_method_vectors.tsv, an identical copy of the canonical table in
# neurosuite-3 libneurosuite-core/test/) against fiber_pca.  The C++ runner
# (pca_method_conformance_test.cpp) checks the same table against core::Method, so
# the cross-repo enum and its integer values cannot drift: a reorder in either repo
# breaks that repo's runner.
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

try:
    from fiber_kit import fiber_pca as fp
except ImportError:
    sys.path.insert(0, os.path.join(HERE, "..", "src", "fiber_kit"))
    import fiber_pca as fp

VECTORS = os.environ.get("PCA_METHOD_VECTORS",
                         os.path.join(HERE, "pca_method_vectors.tsv"))

fails = 0
ran = 0


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
        if kind == "format":
            magic = int(f[1], 0)
            version = int(f[2])
            check(fp.PCAE_MAGIC == magic, f"format magic {f[1]}")
            check(fp.PCAE_VERSION == version, f"format version {f[2]}")
        elif kind == "method":
            value = int(f[1])
            name = f[2]
            order = int(f[3])
            temporal = (f[4] == "1")
            tag = f[5]
            # The contract is value -> meaning (names differ by language convention:
            # core 'StderivAllPairs' vs fiber-kit 'STDERIV_ALLPAIRS').  (spatial_order,
            # has_temporal) is a bijection over the seven methods, so pinning it by
            # value catches any reorder — including a swap within one tag that
            # tag+temporal alone would miss.
            try:
                m = fp.Method(value)
            except ValueError:
                check(False, f"no Method member for value {value} ({name})")
                continue
            check(fp.spatial_order(m) == order, f"spatial_order({name}={value}) == {f[3]}")
            check(fp.has_temporal_diff(m) == temporal,
                  f"has_temporal_diff({name}={value}) == {f[4]}")
            check(fp.method_tag(m) == tag, f"method_tag({name}={value}) == {tag}")
        else:
            check(False, f"unknown vector kind '{kind}'")

# The enum has exactly the canonical members (count guards against an added/removed value).
check(len(list(fp.Method)) == 9, "Method has exactly 9 members")

print(f"pca method conformance (python): {ran} checks, {fails} failed")
if fails == 0:
    print("ALL PCA METHOD CONFORMANCE TESTS PASS")
sys.exit(0 if fails == 0 else 1)
