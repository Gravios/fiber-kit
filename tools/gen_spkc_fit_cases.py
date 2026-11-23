#!/usr/bin/env python3
"""Write test/golden/spkc_fit_cases.bin: collision records with the fit both repositories must make.

The partner fit of a collision row is implemented twice -- fiber_kit.fiber_spkc here and
neurosuite/core/decollide_fit.hpp in neurosuite-3 (Klusters).  This file holds INPUTS and the
EXPECTED result of each case; both repositories' golden-format tests run their own fit on the
inputs and compare.  The expected values are fiber_spkc's at the time of writing; a change of the
fit on either side shows in the other's test.

Layout (little-endian):
  'S' 'F' 'C' 0x01, nCases u32, nSamples u32, nChannels u32
  per case:  tau i32 (the stored offset), record int16[nSamples*nChannels] (sample-major),
             own template float32[...], partner template float32[...],
             accepted u32, fitted tau i32, corrected record int16[...]

    python3 tools/gen_spkc_fit_cases.py            rewrite test/golden/spkc_fit_cases.bin
then copy it to neurosuite-3 (src/libneurosuite-core/test/golden/) and update both contract.yaml.
"""
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from fiber_kit import fiber_spkc as fs   # noqa: E402

NS, NC = 32, 4


def shape(peak, width, amps, sign=-1.0):
    t = np.arange(NS, dtype=np.float64)[:, None]
    g = np.exp(-0.5 * ((t - peak) / width) ** 2) - 0.45 * np.exp(-0.5 * ((t - peak - 2.2 * width) / (1.6 * width)) ** 2)
    return (sign * g * np.asarray(amps, np.float64)[None, :]).astype(np.float32)


def cases():
    rng = np.random.default_rng(20261123)
    own = shape(15, 1.6, [900, 520, 260, 90])
    partner = shape(15, 2.3, [150, 420, 800, 610])
    lookalike = shape(15, 1.7, [820, 560, 240, 110])
    out = []

    def add(stored, true_shift, amp, q=partner, own_amp=1.0, noise=12.0, present=True):
        w = own_amp * own.astype(np.float64)
        if present:
            w = w + amp * fs.shifted(q, true_shift).astype(np.float64)
        w = w + rng.normal(0, noise, w.shape)
        out.append((int(stored), np.clip(np.rint(w), -32768, 32767).astype(np.int16), own, q))

    for true, off in ((6, 0), (6, 2), (6, -3), (9.4, 1), (11.7, -2), (-7, 0), (-8.5, 3), (4, -4), (12, 4), (3.3, 0)):
        add(round(true) + off, true, 0.95)
    for amp in (0.35, 0.6, 1.4, 2.2):                      # across the accepted amplitudes
        add(8, 8.25, amp)
    for amp in (0.12, 3.4):                                # outside them: not this unit / not there
        add(8, 8, amp)
    for stored in (5, -9, 12):                             # the .eap names a partner the record does not hold
        add(stored, 0, 0, present=False)
    add(16, 10, 0.9)                                       # the partner lies outside the search
    add(2, 1.5, 0.9, q=lookalike)                          # a look-alike partner on top of the own spike
    add(7, 7, 0.9, q=lookalike)
    add(13, 13.5, 0.9, own_amp=0.7)                        # a smaller own spike
    add(10, 10, 0.9, noise=60.0)                           # a noisy record
    add(14, 14.2, 0.9)                                     # the partner's tail leaves the window
    add(-13, -13.4, 0.9)
    return out


def main():
    target = os.path.join(HERE, "..", "test", "golden", "spkc_fit_cases.bin")
    cs = cases()
    accepted = 0
    with open(target, "wb") as fh:
        fh.write(b"SFC\x01" + struct.pack("<III", len(cs), NS, NC))
        for stored, w, own, q in cs:
            f = fs.fit_partner(w.astype(np.float32), own, q, stored)
            c = w.astype(np.float32)
            if f["accepted"]:
                c = c - np.float32(f["amp"]) * fs.shifted(q, f["shift"])
            accepted += bool(f["accepted"])
            fh.write(struct.pack("<i", stored) + w.tobytes() + own.tobytes() + q.tobytes()
                     + struct.pack("<Ii", int(bool(f["accepted"])), int(f["tau"])) + fs.to_int16(c).tobytes())
    print(f"wrote {len(cs)} cases ({accepted} accepted) to {os.path.normpath(target)}")


if __name__ == "__main__":
    main()
