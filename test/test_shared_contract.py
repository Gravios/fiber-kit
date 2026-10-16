#!/usr/bin/env python3
"""
test_shared_contract.py — the cross-repo shared files must be byte-identical.

custody_vectors.tsv, pca_method_vectors.tsv, sdiff_expr_vectors.tsv,
stderiv_vectors.tsv and sdiff_grammar_vectors.tsv are
shipped in BOTH fiber-kit and neurosuite-3 on purpose: each is one table run by
every language mirror, so the implementations cannot drift apart.  That only
holds while the two copies are the same file, so changing a vector file is a
two-repo change: this test fails until both sides carry it.

Point NS3_ROOT at a neurosuite-3 checkout and the two copies are compared:

    NS3_ROOT=/path/to/neurosuite-3 python3 test/test_shared_contract.py

A file missing on either side is a failure.  Without NS3_ROOT there is nothing
to compare against: the hashes are listed and the test reports SKIPPED.
neurosuite-3 carries the mirror-image script (FK_ROOT), which its ctest runs as
core_shared_contract.
"""
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# file in this repo -> its path relative to the neurosuite-3 checkout
SHARED = {
    "custody_vectors.tsv":
        "src/libneurosuite-core/test/custody_vectors.tsv",
    "pca_method_vectors.tsv":
        "src/libneurosuite-core/test/pca_method_vectors.tsv",
    "sdiff_expr_vectors.tsv":
        "src/libneurosuite-core/test/sdiff_expr_vectors.tsv",
    "stderiv_vectors.tsv":
        "src/libneurosuite-core/test/stderiv_vectors.tsv",
    "sdiff_grammar_vectors.tsv":
        "src/libneurosuite-core/test/sdiff_grammar_vectors.tsv",
}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ns3 = os.environ.get("NS3_ROOT")
    bad = 0
    for name, rel in sorted(SHARED.items()):
        mine = os.path.join(HERE, name)
        if not os.path.exists(mine):
            print(f"  FAIL {name}: missing from this repo")
            bad += 1
            continue
        h = md5(mine)
        if ns3:
            theirs = os.path.join(ns3, rel)
            if not os.path.exists(theirs):
                print(f"  FAIL {name}: not at {theirs}")
                bad += 1
                continue
            t = md5(theirs)
            ok = h == t
            print(f"  {'ok  ' if ok else 'FAIL'} {name}: {h[:12]} vs {t[:12]} "
                  f"({'identical' if ok else 'DIVERGED'})")
            bad += not ok
        else:
            print(f"  {name}: {h}")

    if not ns3:
        print("\nNS3_ROOT not set, so this listed the hashes rather than comparing.\n"
              "Set it to a neurosuite-3 checkout to verify the two repos agree:\n"
              "    NS3_ROOT=/path/to/neurosuite-3 python3 test/test_shared_contract.py\n"
              "The paths on the other side are:")
        for name, rel in sorted(SHARED.items()):
            print(f"    {name:26s} -> {rel}")
        if not bad:
            print("\nSKIPPED: nothing compared (NS3_ROOT not set)")
            return 0

    print(f"\n{'FAIL' if bad else 'PASS'}: {bad} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
