#!/usr/bin/env python3
# test_dedup_klusters_tables.py -- a dedup moves Klusters' residual tables of the group aside.
#
# .spkx, .spkr and .fetr (neurosuite-3 doc/ndmanager-plugins/formats/) name .spk rows; fiber-kit
# neither reads nor writes them.  After apply_spike_keep their row numbers name other spikes, so
# the group's are renamed <file>.stalebkp, with a line saying how Klusters rebuilds each; another
# group's are left alone, and the dedup no longer calls them files of an unknown type.
#
#     python3 test/test_dedup_klusters_tables.py
import contextlib
import io
import os
import struct
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from fiber_kit import neuro_io as nio   # noqa: E402

fails = 0


def check(ok, what):
    global fails
    print(("  ok:   " if ok else "  FAIL: ") + what)
    fails += not ok


def spkx_bytes(n_spikes, rows, ns, nc, group, margin=2):
    rows = np.asarray(rows, "<i8")
    return (b"SPX\x01" + struct.pack("<7I", rows.size, ns, nc, n_spikes, group, margin, 0)
            + rows.tobytes() + rows.tobytes() + np.full(rows.size, -1, "<i8").tobytes()
            + np.zeros((rows.size, ns + 2 * margin, nc), "<i2").tobytes())


with tempfile.TemporaryDirectory() as td:
    b = os.path.join(td, "s")
    N, NS, NC = 10, 4, 2
    for g in ("1", "2"):
        np.arange(N, dtype="<i8").tofile(f"{b}.res.{g}")
        np.r_[2, np.ones(N)].astype("<i4").tofile(f"{b}.clu.{g}")
        np.zeros((N, NS, NC), "<i2").tofile(f"{b}.spk.{g}")
    tables = [f"{b}.spkx.standard.1", f"{b}.spkx.stderiv_C5.1.fc_unmix_res",
              f"{b}.spkr.standard.1", f"{b}.fetr.stderiv_C5.1"]
    for t in tables:                      # contents only matter to Klusters; the sweep goes by name
        with open(t, "wb") as f:
            f.write(spkx_bytes(N, [3, 8], NS, NC, 1))
    other = f"{b}.spkx.standard.2"
    with open(other, "wb") as f:
        f.write(spkx_bytes(N, [3, 8], NS, NC, 2))

    keep = np.ones(N, bool)
    keep[[0, 1]] = False
    out = io.StringIO()                   # the dedup's notes go to stdout, its warnings to stderr
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        nio.apply_spike_keep(b, "1", keep, N, NS, NC)
    log = out.getvalue()

    check(np.fromfile(f"{b}.res.1", "<i8").size == 8, "the group's .res is deduped to 8 spikes")
    for t in tables:
        name = os.path.basename(t)
        check(not os.path.exists(t) and os.path.exists(t + ".stalebkp"), f"{name} -> {name}.stalebkp")
        check(f"{name} names rows of the old spike set" in log, f"{name}: the move is reported")
    check("Rebuild Residual Spikes' Extended Records" in log, ".spkx: the line names Klusters' rebuild")
    check(log.count("Subtract the partner from collision rows on") == 2,
          ".spkr / .fetr: the line names the Save preference they need")
    check(os.path.exists(other) and not os.path.exists(other + ".stalebkp"),
          "another group's .spkx is left alone")
    check("type the dedup does not know" not in log, "no 'unknown type' warning for them")

    # a second dedup of the same group: nothing left to move, the .stalebkp files are not touched
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        nio.apply_spike_keep(b, "1", np.ones(8, bool), 8, NS, NC)
    check(all(os.path.exists(t + ".stalebkp") for t in tables) and "stalebkp" not in out.getvalue(),
          "a second dedup leaves the .stalebkp files as they are")

print(f"\ntest_dedup_klusters_tables: {fails} failed")
sys.exit(1 if fails else 0)
