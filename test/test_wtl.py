#!/usr/bin/env python3
# test_wtl.py — the manual template-linkage sidecar (.wtl) reader/writer in
# neuro_io, the Python mirror of neurosuite-3's neurofileio readWtl/writeWtl
# (claude/template-curation-plan.md phase 3).  A .wtl written here must read in
# Klusters and vice versa, so this pins the shared text contract on the Python
# side.  Section 6 reads a file written by neurosuite-3's writeWtl (version 2),
# byte for byte as the C++ writer produced it.
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

try:
    from fiber_kit import neuro_io as nio  # noqa: E402
except ImportError:
    sys.path.insert(0, os.path.join(HERE, "..", "src", "fiber_kit"))
    import neuro_io as nio  # noqa: E402

fails = 0
ran = 0


def check(ok, what):
    global fails, ran
    ran += 1
    print(("  ok:   " if ok else "  FAIL: ") + what)
    if not ok:
        fails += 1


def nodes_eq(a, b):
    """Field equality for a parsed node vs the dict it was built from (a/b as
    floats; spikes and ints exact)."""
    return (int(a["node"]) == int(b["node"]) and int(a["class"]) == int(b["class"])
            and a["kind"] == b["kind"] and int(a["parent"]) == int(b["parent"])
            and abs(float(a["a"]) - float(b["a"])) < 1e-6
            and abs(float(a["b"]) - float(b["b"])) < 1e-6
            and [int(x) for x in a["spikes"]] == [int(x) for x in b["spikes"]])


with tempfile.TemporaryDirectory() as d:
    base = os.path.join(d, "sess")
    elec = 1

    # ── 1. Round-trip a two-class forest ────────────────────────────────────
    forest = [
        {"node": 0, "class": 31, "kind": "drift-root",     "parent": -1,
         "a": 0.0, "b": 120.0, "spikes": [12, 37, 59, 1024, 2048, 40000]},
        {"node": 1, "class": 31, "kind": "adapt-leaf",     "parent": 0,
         "a": 1.0, "b": 2.0,   "spikes": [12, 59]},
        {"node": 2, "class": 31, "kind": "collision-leaf", "parent": 0,
         "a": 0.0, "b": 0.0,   "spikes": [88, 91, 37]},
        {"node": 3, "class": 40, "kind": "drift-root",     "parent": -1,
         "a": 0.0, "b": 240.0, "spikes": []},               # empty placeholder
    ]
    nio.write_wtl(base, elec, forest, tag="")
    r = nio.read_wtl(base, elec, tag="")
    check(r["version"] == 1, "version preserved")
    check(len(r["nodes"]) == 4, "all 4 nodes parsed")
    check(len(r["nodes"]) == 4 and all(nodes_eq(r["nodes"][i], forest[i]) for i in range(4)),
          "every node field + spike set round-trips exactly")
    check(len(r["nodes"]) == 4 and r["nodes"][3]["spikes"] == [], "empty spike set round-trips")
    check(len(r["nodes"]) == 4 and r["nodes"][0]["spikes"] == [12, 37, 59, 1024, 2048, 40000],
          "large non-contiguous spike set intact")

    # ── 2. Header / version rejections ──────────────────────────────────────
    def write_raw(name, body):
        p = os.path.join(d, name)
        with open(p, "w") as fh:
            fh.write(body)
        return p

    # A version neither side implements, at the method-less wtl path for elec 7.
    with open(nio.session_path(base, "wtl", 7, variant="", tag=""), "w") as fh:
        fh.write("wtl 3\nnNodes 0\n")
    check(read_ok := (not nio.read_wtl(base, 7)["nodes"]) and nio.read_wtl(base, 7)["version"] == 0,
          "unknown version rejected (version 0, no nodes)")

    with open(nio.session_path(base, "wtl", 8, variant="", tag=""), "w") as fh:
        fh.write("# comment first\nnode 0 1 drift-root -1 0 0 1 5\n")
    check(nio.read_wtl(base, 8)["version"] == 0, "missing 'wtl <ver>' header rejected")

    check(nio.read_wtl(base, 99)["version"] == 0, "missing file -> version 0")

    # ── 3. nNodes declared-count check ──────────────────────────────────────
    with open(nio.session_path(base, "wtl", 9, variant="", tag=""), "w") as fh:
        fh.write("wtl 1\nnNodes 2\nnode 0 1 drift-root -1 0 0 1 5\n")   # declares 2, has 1
    check(nio.read_wtl(base, 9)["version"] == 0, "nNodes mismatch rejected")

    # ── 4. Corrupt node line (no nNodes line, so the count check is off) ─────
    # short tail dropped; long tail kept, truncated to nSpikes.
    with open(nio.session_path(base, "wtl", 10, variant="", tag=""), "w") as fh:
        fh.write("wtl 1\n"
                 "node 0 7 drift-root -1 0 0 3 10 11\n"           # short -> dropped
                 "node 1 7 adapt-leaf 0 1 2 2 20 21 22 23\n")     # long  -> kept {20,21}
    rc = nio.read_wtl(base, 10)
    check(len(rc["nodes"]) == 1, "short-tail node dropped, long-tail node kept")
    check(len(rc["nodes"]) == 1 and rc["nodes"][0]["node"] == 1
          and rc["nodes"][0]["spikes"] == [20, 21], "long tail truncated to the declared nSpikes")

    # ── 5. Verbatim/unknown kind token preserved ────────────────────────────
    nio.write_wtl(base, 11, [{"node": 0, "class": 3, "kind": "my-custom-kind",
                              "parent": -1, "a": 0, "b": 0, "spikes": [99]}])
    rk = nio.read_wtl(base, 11)
    check(len(rk["nodes"]) == 1 and rk["nodes"][0]["kind"] == "my-custom-kind",
          "unknown kind token kept verbatim (extensible)")

    # ── 6. Version 2, as neurosuite-3 writes it ─────────────────────────────
    # Output of neurofileio::writeWtl for a three-node forest (two classes, a
    # count-0 placeholder leaf, an unknown kind); readWtl reads it back with
    # ok=1, version=2, 3 nodes.
    CPP_V2 = (
        "wtl 2\n"
        "nSamples 2\n"
        "nChannels 2\n"
        "nNodes 3\n"
        "# node class kind parent a b count\n"
        "# mean <N> ...  then  std <N> ...   (N = nSamples*nChannels, 0 = placeholder)\n"
        "node 0 31 drift-root -1 0 120.123456789 540\n"
        "mean 4 1.5 -2.25 0.1 1e-07\n"
        "std 4 0.5 0.25 3 12345.7\n"
        "node 1 31 adapt-leaf 0 1 2 0\n"
        "mean 0\n"
        "std 0\n"
        "node 2 40 my-custom-kind -1 0 240 3\n"
        "mean 4 4 5 6 7\n"
        "std 4 0 0 0 0\n")
    with open(nio.session_path(base, "wtl", 12, variant="", tag=""), "w") as fh:
        fh.write(CPP_V2)
    r2 = nio.read_wtl(base, 12)
    n2 = r2["nodes"]
    check(r2["version"] == 2 and r2["nSamples"] == 2 and r2["nChannels"] == 2,
          "C++-written v2: version and geometry read")
    check([(n["node"], n["class"], n["kind"], n["parent"], n["count"]) for n in n2]
          == [(0, 31, "drift-root", -1, 540), (1, 31, "adapt-leaf", 0, 0),
              (2, 40, "my-custom-kind", -1, 3)],
          "C++-written v2: every node, its class, kind, parent and count")
    check(len(n2) == 3 and n2[0]["b"] == 120.123456789,
          "C++-written v2: window edges at full precision")
    check(len(n2) == 3 and n2[0]["mean"] == [1.5, -2.25, 0.1, 1e-07]
          and n2[0]["std"] == [0.5, 0.25, 3.0, 12345.7], "C++-written v2: running mean/std")
    check(len(n2) == 3 and n2[1]["mean"] == [] and n2[1]["std"] == [],
          "C++-written v2: a count-0 placeholder has empty mean/std")
    check(len(n2) == 3 and "spikes" not in n2[0], "v2 nodes carry no spike list")

    # ── 7. v2 corruption, handled as readWtl does ────────────────────────────
    with open(nio.session_path(base, "wtl", 13, variant="", tag=""), "w") as fh:
        fh.write("wtl 2\n"
                 "mean 2 9 9\n"                       # before any node -> ignored
                 "node 0 5 drift-root -1 0 10 7\n"
                 "mean 3 1 2\n"                       # short tail -> dropped
                 "std 99999 1 2\n"                    # more than the line can hold -> dropped
                 "node x 5 drift-root -1 0 10 7\n"    # malformed fixed field -> skipped
                 "node 1 5 adapt-leaf 0 0 1 -4\n"     # negative count -> skipped
                 "someNewKey 1 2 3\n")                # unknown key -> ignored
    r3 = nio.read_wtl(base, 13)
    check(r3["version"] == 2 and len(r3["nodes"]) == 1 and r3["nodes"][0]["count"] == 7,
          "v2: malformed and negative-count nodes skipped, unknown key ignored")
    check(len(r3["nodes"]) == 1 and r3["nodes"][0]["mean"] == [] and r3["nodes"][0]["std"] == [],
          "v2: short, oversized and stray summary lines dropped")

    # v1 nodes report their spike count as `count`, as readWtl maps them.
    check(r["nodes"][0]["count"] == 6 and r["nodes"][3]["count"] == 0,
          "v1: count = nSpikes")

print(f"\ntest_wtl: {ran} checks, {fails} failed")
sys.exit(1 if fails else 0)
