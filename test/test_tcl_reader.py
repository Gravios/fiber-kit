#!/usr/bin/env python3
# test_tcl_reader.py -- neuro_io.read_tcl accepts and refuses what neurosuite-3's
# neurofileio::readTcl does.
#
# Each case is a .tcl text, whether readTcl accepts it, and the columns of the
# entries it returns.  The expected values were taken by running readTcl
# (libneurosuite-core at 12a9abf4) on these exact files; the rules are in
# neurosuite-3 doc/ndmanager-plugins/formats/tcl.md.  A malformed row, an
# unknown status, a bad integer, a merged: target outside the table or equal to
# its own column, or anything but one row per column 0..nClasses-1 fails the
# whole read, and entries[c] is column c whatever the order of the rows.
#
#     python3 test/test_tcl_reader.py
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from fiber_kit import neuro_io as nio   # noqa: E402

# (name, .tcl text, readTcl accepts it, columns of its entries)
CASES = [
    ('ok_basic', 'tcl 1\nnClasses 3\n# col\tstatus\n0\tactive\ta\t5\tfc_unmix\t2026\n1\ttomb\t-\t-\t-\t-\n2\tmerged:0\tb\t7\ts\t-\n', True, [0, 1, 2]),
    ('ok_no_nclasses', 'tcl 1\n0\tfree\n1\tactive\tx\t3\n', True, [0, 1]),
    ('ok_reordered', 'tcl 1\nnClasses 2\n1\tactive\tb\t2\n0\tactive\ta\t1\n', True, [0, 1]),
    ('ok_crlf', 'tcl 1\r\nnClasses 1\r\n0\tactive\ta\t1\r\n', True, [0]),
    ('ok_comment_blank', '# c\n\ntcl 1\n  \n0\tfree\n', True, [0]),
    ('bad_status', 'tcl 1\n0\tweird\n', False, []),
    ('bad_short_row', 'tcl 1\n0\tactive\n1\n', False, []),
    ('bad_col', 'tcl 1\nx\tactive\n', False, []),
    ('bad_dup_col', 'tcl 1\n0\tactive\n0\tfree\n', False, []),
    ('bad_missing_col', 'tcl 1\nnClasses 3\n0\tactive\n2\tfree\n', False, []),
    ('bad_count', 'tcl 1\nnClasses 3\n0\tactive\n1\tfree\n', False, []),
    ('bad_merged_self', 'tcl 1\n0\tmerged:0\n1\tfree\n', False, []),
    ('bad_merged_out', 'tcl 1\n0\tmerged:5\n1\tfree\n', False, []),
    ('bad_merged_num', 'tcl 1\n0\tmerged:x\n1\tfree\n', False, []),
    ('bad_prov', 'tcl 1\n0\tactive\ta\t7x\n', False, []),
    ('bad_version', 'tcl 2\n0\tfree\n', False, []),
    ('bad_header', 'tc 1\n0\tfree\n', False, []),
    ('bad_nclasses', 'tcl 1\nnClasses x\n0\tfree\n', False, []),
    ('bad_nclasses_extra', 'tcl 1\nnClasses 1 2\n0\tfree\n', False, []),
    ('ok_signed_col', 'tcl 1\n+0\tactive\n', True, [0]),
    ('empty', '', False, []),
]

fails = 0


def check(ok, what):
    global fails
    print(("  ok:   " if ok else "  FAIL: ") + what)
    fails += not ok


with tempfile.TemporaryDirectory() as td:
    for name, text, accepted, cols in CASES:
        base = os.path.join(td, name)
        with open(base + ".tcl.1", "w", newline="") as f:
            f.write(text)
        r = nio.read_tcl(base, 1)
        got = [e["col"] for e in r["entries"]]
        check(r["ok"] == accepted and got == cols and (not accepted or r["nClasses"] == len(cols)),
              f"{name}: {'accepted' if accepted else 'refused'}"
              + (f", entries by column {cols}" if accepted else "")
              + ("" if (r["ok"], got) == (accepted, cols) else f"  (got ok={r['ok']} cols={got})"))

    # the reordered table: entries[c] is column c, fields kept with their column
    r = nio.read_tcl(os.path.join(td, "ok_reordered"), 1)
    check([(e["col"], e["label"], e["provenance_clu"]) for e in r["entries"]] == [(0, "a", 1), (1, "b", 2)],
          "ok_reordered: each column keeps its own label and provenance")
    r = nio.read_tcl(os.path.join(td, "ok_basic"), 1)
    check(r["entries"][2]["status"] == "merged" and r["entries"][2]["merged_into"] == 0
          and r["entries"][1]["label"] == "" and r["entries"][1]["provenance_clu"] == -1,
          "ok_basic: merged:0 and the '-' fields read as neurofileio reads them")

    # write_tcl -> read_tcl round trip
    entries = [dict(col=0, status="active", label="u", provenance_clu=4, provenance_stage="fc_xcm",
                    created="", merged_into=-1),
               dict(col=1, status="merged", label="", provenance_clu=-1, provenance_stage="",
                    created="", merged_into=0)]
    nio.write_tcl(os.path.join(td, "rt"), 1, entries, n_classes=2)
    r = nio.read_tcl(os.path.join(td, "rt"), 1)
    check(r["ok"] and [(e["col"], e["status"], e["merged_into"], e["provenance_clu"]) for e in r["entries"]]
          == [(0, "active", -1, 4), (1, "merged", 0, -1)], "write_tcl output reads back whole")

    # write_eap_membership keeps a .tcl that is there but not valid (Klusters' rule):
    # it used to start from free slots and overwrite the registry.
    sys.path.insert(0, os.path.join(HERE, "..", "src", "fiber_kit"))
    import fiber_decollide as fd   # noqa: E402
    base = os.path.join(td, "keep")
    bad = "tcl 1\n0\tactive\tmy-unit\t5\ts\t-\n0\tactive\tdup\t6\ts\t-\n"
    with open(base + ".tcl.1", "w") as f:
        f.write(bad)
    manifest = [dict(source_idx=0, k1=5, tau1=0, k2=6, tau2=3)]
    em = fd.write_eap_membership(base, 1, parent_tag="", n_spikes=4, manifest=manifest)
    with open(base + ".tcl.1") as f:
        kept = f.read()
    check(em is None and kept == bad and not os.path.exists(base + ".eap.1"),
          "write_eap_membership: an invalid .tcl is kept and no .eap written")
    os.remove(base + ".tcl.1")
    em = fd.write_eap_membership(base, 1, parent_tag="", n_spikes=4, manifest=manifest)
    check(em is not None and nio.read_tcl(base, 1)["ok"],
          "write_eap_membership: with no .tcl it starts a fresh one")

print(f"\ntest_tcl_reader: {len(CASES) + 5} checks, {fails} failed")
sys.exit(1 if fails else 0)
