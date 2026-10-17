#!/usr/bin/env python3
# test_golden_formats.py — the template-library formats against reference files.
#
# test/golden/ is shared with neurosuite-3 (byte-identical copies, hashes in
# contract.yaml):
#
#   golden.*  written by neurosuite-3's C++ writers (core_test_golden_formats):
#             golden.wti.1 (v1), golden.wti.2 (v2, parents), golden.wtf.standard.1,
#             golden.wtl.1 (v2 summaries), golden.eap.1, golden.tcl.1
#   fk.*      written by the neuro_io writers here: fk.wti.1, fk.wti.2 (with the
#             src_clu_* columns) and fk.wtl.1 (v1, spike lists).  neurosuite-3's
#             test reads them with the C++ readers.
#
# Checked here:
#   - neuro_io reads every field of every golden.* file;
#   - write_wtf, write_eap and write_tcl reproduce the golden.* bytes (the same
#     format, so the same bytes);
#   - the neuro_io writers still produce the committed fk.* bytes.
#
#     python3 test/test_golden_formats.py            check
#     python3 test/test_golden_formats.py --write    regenerate fk.* (then update
#                                                    both contract.yaml files and
#                                                    neurosuite-3's copies)
#
# The content is spelled out once here and once in neurosuite-3's
# golden_formats_test.cpp; keep the two in step.
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "golden")
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from fiber_kit import neuro_io as nio   # noqa: E402

NS, NCH = 4, 2
A = nio.EAP_ABSENT

fails = 0
ran = 0


def check(ok, what):
    global fails, ran
    ran += 1
    print(("  ok:   " if ok else "  FAIL: ") + what)
    if not ok:
        fails += 1


def wti_rows(with_parents):
    rows = [dict(row=0, unit=7, link="drift", bin=0, a=0.0, b=120.5, nSpikes=540, parent=-1),
            dict(row=1, unit=7, link="drift", bin=1, a=120.5, b=241.25, nSpikes=0,
                 parent=0 if with_parents else -1),
            dict(row=2, unit=9, link="adapt", bin=0, a=1.5, b=2.75, nSpikes=33, parent=-1)]
    if with_parents:
        rows.append(dict(row=3, unit=7, link="collision", bin=0, a=0.0, b=120.5, nSpikes=12, parent=0))
    return rows


WTF = np.array([[0, 1, -1, 32767, -32768, 100, -100, 7],
                [-5, -4, -3, -2, -1, 0, 1, 2],
                [1000, -2000, 3000, -4000, 5000, -6000, 7000, -8000]],
               dtype=np.int16).reshape(3, NS, NCH)

WTL_NODES = [
    dict(node=0, cls=7, kind="drift-root", parent=-1, a=0.0, b=120.5, count=540,
         mean=[0.1, -1.5, 1e-3, 1234.5678, -0.0, 3.0, -250.25, 1e-7],
         std=[2.1, 3.0, 0.5, 10.75, 0.0, 1e-4, 7.0, 0.333333]),
    dict(node=1, cls=7, kind="adapt-leaf", parent=0, a=1.5, b=2.75, count=0, mean=[], std=[]),
    dict(node=2, cls=9, kind="drift-root", parent=-1, a=241.25, b=360.0, count=3,
         mean=[-1.0, -2.0, -3.0, -4.0, 5.0, 6.0, 7.0, 8.0],
         std=[1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]),
]

EAP = np.array([[0, A, A, A],
                [A, 2, A, A],
                [-3, A, 127, A],
                [A, A, A, -127],
                [1, -1, A, 0]], dtype=np.int8)
EAP_GROUP, EAP_FLAGS = 1, 3

TCL = [dict(col=0, status="active", label="CA1 pyr a", provenance_clu=23, provenance_stage="gt",
            created="2026-10-02", merged_into=-1),
       dict(col=1, status="free", label="", provenance_clu=-1, provenance_stage="", created="",
            merged_into=-1),
       dict(col=2, status="tomb", label="old", provenance_clu=4, provenance_stage="", created="",
            merged_into=-1),
       dict(col=3, status="merged", label="x", provenance_clu=-1, provenance_stage="", created="",
            merged_into=0),
       dict(col=4, status="active", label="", provenance_clu=0, provenance_stage="", created="",
            merged_into=-1)]

# What fiber-kit writes: the same rows plus its provenance columns; a v1 .wtl.
FK_SRC = dict(src_clu_variant="stderiv_C5", src_clu_tag="")
FK_WTL = [dict(node=0, **{"class": 7}, kind="drift-root", parent=-1, a=0.0, b=120.5, spikes=[3, 5, 8]),
          dict(node=1, **{"class": 7}, kind="adapt-leaf", parent=0, a=1.5, b=2.75, spikes=[]),
          dict(node=2, **{"class": 9}, kind="collision-leaf", parent=-1, a=241.25, b=360.0,
               spikes=[0, 1])]


def write_fk(d):
    base = os.path.join(d, "fk")
    for elec, wp in ((1, False), (2, True)):
        rows = [dict(r, lo=r["a"], hi=r["b"], nspk=r["nSpikes"], **FK_SRC) for r in wti_rows(wp)]
        for r in rows:
            r.pop("a"), r.pop("b"), r.pop("nSpikes"), r.pop("row")
            if not wp:
                r.pop("parent")
        nio.write_wti(base, elec, rows, NS, NCH, peak=1, sr=20000.0)
    nio.write_wtl(base, 1, FK_WTL)


def read_bytes(path):
    with open(path, "rb") as fh:
        return fh.read()


def check_wti(got, version, with_parents, name):
    check(got["version"] == version, f"{name}: version {got['version']}")
    check((got["nSamples"], got["nChannels"], got["peakSample"], got["sr"]) == (NS, NCH, 1, 20000.0),
          f"{name}: header")
    keys = ("row", "unit", "link", "bin", "a", "b", "nSpikes", "parent")
    check([{k: r[k] for k in keys} for r in got["rows"]] == wti_rows(with_parents),
          f"{name}: every row")


def main():
    if sys.argv[1:] == ["--write"]:
        write_fk(GOLDEN)
        print(f"wrote fk.* into {GOLDEN}")
        return 0

    g = os.path.join(GOLDEN, "golden")
    print("neuro_io reads the C++-written files:")
    check_wti(nio.read_wti(g, 1), 1, False, "golden.wti.1")
    check_wti(nio.read_wti(g, 2), 2, True, "golden.wti.2")
    w = nio.read_wtf(g, 1, NS, NCH, variant="standard")
    check(w.shape == WTF.shape and np.array_equal(w, WTF), "golden.wtf.standard.1: every sample")

    t = nio.read_wtl(g, 1)
    check((t["version"], t["nSamples"], t["nChannels"]) == (2, NS, NCH), "golden.wtl.1: header")
    check(len(t["nodes"]) == len(WTL_NODES), "golden.wtl.1: node count")
    for got, want in zip(t["nodes"], WTL_NODES):
        same = (got["node"], got["class"], got["kind"], got["parent"], got["a"], got["b"], got["count"]) \
            == (want["node"], want["cls"], want["kind"], want["parent"], want["a"], want["b"], want["count"])
        check(same, f"golden.wtl.1: node {want['node']} fields")
        # The file holds float32 values at round-trip precision.
        check(np.array_equal(np.float32(got["mean"]), np.float32(want["mean"]))
              and np.array_equal(np.float32(got["std"]), np.float32(want["std"])),
              f"golden.wtl.1: node {want['node']} mean/std as float32")

    e = nio.read_eap(g, 1)
    check(e["ok"] and (e["nSpikes"], e["nClasses"], e["group"], e["flags"]) == (5, 4, EAP_GROUP, EAP_FLAGS)
          and np.array_equal(e["cells"], EAP), "golden.eap.1: header and every cell")

    c = nio.read_tcl(g, 1)
    check(c["ok"] and c["version"] == 1 and c["nClasses"] == 5, "golden.tcl.1: header")
    keys = ("col", "status", "label", "provenance_clu", "provenance_stage", "created", "merged_into")
    check([{k: x[k] for k in keys} for x in c["entries"]] == TCL, "golden.tcl.1: every entry")

    print("\nThe neuro_io writers give the same bytes for the same format:")
    with tempfile.TemporaryDirectory() as d:
        b = os.path.join(d, "golden")
        nio.write_wtf(b, 1, WTF, variant="standard")
        nio.write_eap(b, 1, EAP, group=EAP_GROUP, flags=EAP_FLAGS)
        nio.write_tcl(b, 1, TCL, n_classes=5)
        for name in ("golden.wtf.standard.1", "golden.eap.1", "golden.tcl.1"):
            check(read_bytes(os.path.join(d, name)) == read_bytes(os.path.join(GOLDEN, name)),
                  f"{name}: neuro_io writes the C++ bytes")

        print("\nread_mti/read_mtf read the committed-model names (same schemas):")
        shutil.copy(os.path.join(GOLDEN, "golden.wti.2"), os.path.join(d, "m.mti.3.final"))
        shutil.copy(os.path.join(GOLDEN, "golden.wtf.standard.1"), os.path.join(d, "m.mtf.standard.3.final"))
        m = os.path.join(d, "m")
        check_wti(nio.read_mti(m, 3, tag="final"), 2, True, "m.mti.3.final")
        check(np.array_equal(nio.read_mtf(m, 3, NS, NCH, variant="standard", tag="final"), WTF),
              "m.mtf.standard.3.final: every sample")

        print("\nThe neuro_io writers still produce the committed fk.* files:")
        write_fk(d)
        for name in ("fk.wti.1", "fk.wti.2", "fk.wtl.1"):
            check(read_bytes(os.path.join(d, name)) == read_bytes(os.path.join(GOLDEN, name)),
                  f"{name}: unchanged")

    print(f"\ntest_golden_formats: {ran} checks, {fails} failed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
