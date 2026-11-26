#!/usr/bin/env python3
"""session_identity.py — does a change to fiber_session leave its output byte-identical?

Builds a small synthetic session (seeded: five units on four channels, two chunks) and runs
fiber-session on it under several configurations that between them switch on most of the fine
pass's options.  Prints one line per configuration with a hash over every file the run wrote.
Run it on the tree before a change and on the tree after, and compare:

    python3 tools/audit/session_identity.py <checkout-before>/src > before.txt
    python3 tools/audit/session_identity.py <checkout-after>/src  > after.txt
    diff before.txt after.txt && echo identical

A synthetic session is not a recording: for a change that matters, also run fiber-session on a
real group before and after (same flags, two --clu-stage names) and `cmp` the .clu files.
"""
import hashlib
import os
import subprocess
import sys
import tempfile

import numpy as np

SR = 20000.0
NSAMP, NCH, NTOTAL = 32, 4, 5
SETS = "0-1+2+3,1-0+2+3,2-0+1+3,3-0+1+2"
PER_CHUNK = 1500

CONFIGS = {
    "rkk": ["--fine-algo", "rkk"],
    "gmm": ["--fine-algo", "gmm", "--var-split", "1.5"],
    "fiber": ["--fine-algo", "fiber", "--no-dipsplit"],
    "rkk-resplit": ["--fine-algo", "rkk", "--resplit-passes", "1", "--refit-iters", "1", "--merge-corr", "0.9",
                    "--merge-algo", "sliding", "--inclusion-k", "2.5", "--incl-assign-rejected"],
    "rkk-neigh": ["--fine-algo", "rkk", "--neigh-passes", "1", "--cfiber-gate", "--no-rkk-delete",
                  "--dip-dim", "6", "--dip-alpha", "0.02", "--dip-min", "30"],
    "none": ["--no-fine"],
}


def build(td):
    rng = np.random.default_rng(7)
    base = os.path.join(td, "mini")
    n = 2 * PER_CHUNK
    res = np.sort(rng.integers(int(2 * SR), int(58 * SR), n)).astype(np.int64)
    res.tofile(f"{base}.res.1")
    unit = rng.integers(0, 5, n)
    amp = np.array([[-3000, -900, -200, -100], [-2200, -1600, -300, -100], [-300, -2600, -1100, -200],
                    [-150, -700, -2400, -1300], [-100, -200, -900, -1800]], float)
    width = np.array([6, 8, 10, 7, 9])
    w = rng.integers(-60, 60, (n, NSAMP, NCH)).astype(np.float64)
    for u in range(5):
        m = unit == u
        k = int(width[u])
        shape = np.zeros(NSAMP)
        shape[16 - k // 2:16 - k // 2 + k] = np.hanning(k)
        gain = 1.0 + 0.08 * rng.standard_normal(int(m.sum()))
        w[m] += gain[:, None, None] * shape[None, :, None] * amp[u][None, None, :]
    np.clip(w, -32000, 32000).astype(np.int16).tofile(f"{base}.spk.stderiv_C5.1")
    rng.integers(-30, 30, int(60 * SR) * NTOTAL).astype(np.int16).tofile(f"{base}.fil")
    return base


def main():
    src = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "..", "src")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("FK_", "FIBER_"))}
    env["PYTHONPATH"] = src
    bad = 0
    with tempfile.TemporaryDirectory() as td:
        base = build(td)
        before = set(os.listdir(td))
        for name, extra in CONFIGS.items():
            argv = [sys.executable, "-m", "fiber_kit.fiber_session", base, "1", "--channels", "0,1,2,3",
                    "--ntotal", str(NTOTAL), "--nsamp", str(NSAMP), "--sr", str(int(SR)), "--method", "stderiv_C5",
                    "--sdiff-pairs", SETS, "--no-link", "--no-deconv", "--chunk-min", "0.5", "--overlap-min", "0",
                    "--min-group", "40", "--clu-stage", name.replace("-", "_")] + extra
            r = subprocess.run(argv, cwd=td, env=env, capture_output=True, text=True)
            if r.returncode != 0:
                bad += 1
                print(f"{name:12s} FAILED: {(r.stderr or r.stdout).strip().splitlines()[-1][:160]}")
                continue
            h = hashlib.sha256()
            files = sorted(f for f in os.listdir(td) if f not in before)
            nclu = "?"
            for f in files:
                with open(os.path.join(td, f), "rb") as fh:
                    data = fh.read()
                h.update(f.encode() + b"\0" + data)
                if ".clu." in f:
                    nclu = len(np.unique(np.frombuffer(data, np.int32)[1:] if len(data) % 4 == 0 else []))
            print(f"{name:12s} {len(files):2d} files  {nclu} clusters  {h.hexdigest()[:20]}")
            before |= set(files)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
