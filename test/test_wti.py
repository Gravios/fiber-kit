#!/usr/bin/env python3
# test_wti.py — the .wti template-index version-2 schema bump in neuro_io: a
# trailing `parent` lineage column (inserted before the src_clu_* columns) and the
# `collision` link kind, mirroring neurosuite-3's neurofileio v2.  Pins the
# Python side of the shared contract; the cross-repo interop check lives in the
# dev scratchpad.
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


def first_line(path):
    with open(path) as f:
        return f.readline().strip()


with tempfile.TemporaryDirectory() as d:
    base = os.path.join(d, "sess")

    # ── 1. v1 (no parent) stays v1, scv/sct preserved ───────────────────────
    v1_rows = [
        {"class": 31, "link": "drift", "bin": 0, "lo": 0.0,   "hi": 120.0, "nspk": 540,
         "src_clu_variant": "stderiv", "src_clu_tag": "refine"},
        {"class": 31, "link": "adapt", "bin": 0, "lo": 1.0,   "hi": 2.0,   "nspk": 410},
    ]
    nio.write_wti(base, 1, v1_rows, 42, 8, sr=32552.0, tag="")
    check(first_line(nio.session_path(base, "wti", 1, variant="", tag="")) == "wti 1",
          "no parent -> header stays 'wti 1'")
    r1 = nio.read_wti(base, 1, tag="")
    check(r1["version"] == 1 and len(r1["rows"]) == 2, "v1 reads back, 2 rows")
    check(all(rd["parent"] == -1 for rd in r1["rows"]), "v1 rows read with parent == -1")
    check(r1["rows"][0]["src_clu_variant"] == "stderiv" and r1["rows"][0]["src_clu_tag"] == "refine",
          "v1 src_clu_* columns preserved")

    # ── 2. v2: a parent set + a collision link ──────────────────────────────
    v2_rows = [
        {"class": 31, "link": "drift",     "bin": 0, "lo": 0.0, "hi": 120.0, "nspk": 540, "parent": -1},
        {"class": 31, "link": "adapt",     "bin": 0, "lo": 1.0, "hi": 2.0,   "nspk": 410, "parent": 0},
        {"class": 31, "link": "collision", "bin": 0, "lo": 0.0, "hi": 0.0,   "nspk": 12,  "parent": 0,
         "src_clu_variant": "stderiv", "src_clu_tag": "refine"},
    ]
    nio.write_wti(base, 2, v2_rows, 42, 8, sr=32552.0, tag="")
    check(first_line(nio.session_path(base, "wti", 2, variant="", tag="")) == "wti 2",
          "a parent present -> header becomes 'wti 2'")
    r2 = nio.read_wti(base, 2, tag="")
    check(r2["version"] == 2 and len(r2["rows"]) == 3, "v2 reads back, 3 rows")
    check([rd["parent"] for rd in r2["rows"]] == [-1, 0, 0], "v2 parent column round-trips")
    check(r2["rows"][2]["link"] == "collision", "collision link round-trips verbatim")
    check(r2["rows"][2]["src_clu_variant"] == "stderiv",
          "v2 src_clu_* still read (now at columns 9/10)")

    # read -> write -> read keeps v2 + parents (exercises the a/b fallback in write)
    nio.write_wti(base, 3, r2["rows"], 42, 8, sr=32552.0, tag="")
    r3 = nio.read_wti(base, 3, tag="")
    check(r3["version"] == 2 and [rd["parent"] for rd in r3["rows"]] == [-1, 0, 0],
          "read->write->read preserves v2 + parents")

    # ── 3. Hand-written v1 and v2 files read correctly ──────────────────────
    with open(nio.session_path(base, "wti", 4, variant="", tag=""), "w") as fh:
        fh.write("wti 1\nnSamples 4\nnChannels 2\nnRows 1\n"
                 "# row unit link bin a b nSpikes src_clu_variant src_clu_tag\n"
                 "row 0 7 drift 0 0 1 99 - -\n")
    h1 = nio.read_wti(base, 4, tag="")
    check(h1["version"] == 1 and h1["rows"][0]["parent"] == -1 and h1["rows"][0]["nSpikes"] == 99,
          "hand-written v1 reads (parent -1, '-' src -> '')")

    with open(nio.session_path(base, "wti", 5, variant="", tag=""), "w") as fh:
        fh.write("wti 2\nnRows 2\n# ...\n"
                 "row 0 7 drift 0 0 1 99 -1 - -\nrow 1 7 collision 0 0 0 4 0 - -\n")
    h2 = nio.read_wti(base, 5, tag="")
    check(h2["version"] == 2 and h2["rows"][1]["parent"] == 0 and h2["rows"][1]["link"] == "collision",
          "hand-written v2 reads parent + collision")

    # A C++-style v2 row with NO trailing src_clu columns (9 tokens) still reads.
    with open(nio.session_path(base, "wti", 6, variant="", tag=""), "w") as fh:
        fh.write("wti 2\nnRows 1\n# ...\nrow 0 7 drift 0 0 1 99 -1\n")
    h3 = nio.read_wti(base, 6, tag="")
    check(h3["rows"][0]["parent"] == -1 and h3["rows"][0]["src_clu_variant"] == "",
          "v2 row without src_clu columns (C++ writer shape) reads")

    # ── 4. Unknown version rejected ─────────────────────────────────────────
    with open(nio.session_path(base, "wti", 7, variant="", tag=""), "w") as fh:
        fh.write("wti 3\nnRows 0\n")
    check(nio.read_wti(base, 7, tag="")["version"] == 0, "version 3 rejected")

print(f"\ntest_wti: {ran} checks, {fails} failed")
sys.exit(1 if fails else 0)
