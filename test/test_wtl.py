#!/usr/bin/env python3
# test_wtl.py — the manual template-linkage sidecar (.wtl) reader/writer in
# neuro_io, the Python mirror of neurosuite-3's neurofileio readWtl/writeWtl
# (claude/template-curation-plan.md phase 3).  A .wtl written here must read in
# Klusters and vice versa, so this pins the shared text contract on the Python
# side; the cross-repo interop check lives in the dev scratchpad.
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

    # A file whose header is "wtl 2" lands at the method-less wtl path for elec 7.
    with open(nio.session_path(base, "wtl", 7, variant="", tag=""), "w") as fh:
        fh.write("wtl 2\nnNodes 0\n")
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

print(f"\ntest_wtl: {ran} checks, {fails} failed")
sys.exit(1 if fails else 0)
