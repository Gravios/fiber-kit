"""check_spk_count: a res-vs-spk count mismatch is diagnosed as GEOMETRY.

The reported failure: the session yaml's group nSamples was edited 42 -> 52
while .spk.stderiv_C5_D34 stayed the 42-sample extraction, so the memmap
reshaped 138,319 spikes into 111,719 (plus a dropped partial) and the stage
died with a bare count assert.  The gate now does the divisibility diagnosis
and names the window the file was really extracted at.
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import fiber_kit.neuro_io as nio  # noqa: E402


def main():
    ok = 0

    def check(name, cond):
        nonlocal ok
        assert cond, name
        ok += 1
        print(f"  ok  {name}")

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "sess.spk.stderiv_C5_D34.6")
        nspk, true_ns, wrong_ns, nch = 200, 42, 52, 8
        np.zeros(nspk * true_ns * nch, np.int16).tofile(path)

        # matched geometry -> silent no-op
        mm = nio.open_spk_file(path, true_ns, nch)
        nio.check_spk_count(nspk, mm, path, true_ns, nch)
        check("matching count passes silently", True)

        # the reported case: yaml says 52, file was extracted at 42
        mm = nio.open_spk_file(path, wrong_ns, nch)
        try:
            nio.check_spk_count(nspk, mm, path, wrong_ns, nch)
            raised = ""
        except SystemExit as e:
            raised = str(e)
        check("mismatch raises SystemExit, not a bare assert", bool(raised))
        check("message names the file's true window (42) and the session's (52)",
              "42 samples" in raised and "nSamples 52" in raised)
        check("message says how to fix it",
              "nSamples/peakSampleIndex" in raised and "--nsamp 42" in raised)

        # non-divisible size -> the out-of-step reading, no bogus window claim
        with open(path, "ab") as fh:
            fh.write(b"\x00\x00" * 3)                      # 3 stray samples
        mm = nio.open_spk_file(path, wrong_ns, nch)
        try:
            nio.check_spk_count(nspk, mm, path, wrong_ns, nch)
            raised = ""
        except SystemExit as e:
            raised = str(e)
        check("non-divisible size falls back to the out-of-step message",
              "out of step" in raised and "samples per spike" not in raised)

    print(f"test_spk_geometry_check: {ok}/5 checks passed")


if __name__ == "__main__":
    main()
