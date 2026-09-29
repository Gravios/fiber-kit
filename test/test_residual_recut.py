#!/usr/bin/env python3
# test_residual_recut.py — the quality-gated median-residual KlustaKwik denoise
# (fiber_intrachunk._residual_recut).
#
# Contract: per unit, kk on low-dimensional cropped median-residual features;
# the dominant component keeps the unit id, distinct components become new
# units, kk noise sheds to reserve; same-shape components (REGISTERED cosine
# >= 0.98, sub-sample) fold back, which keeps repeated rounds harmless.
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import numpy as np  # noqa: E402  (path shim above, as every test here does)

try:
    from fiber_kit.fiber_intrachunk import _residual_recut  # noqa: E402
except ImportError:
    sys.path.insert(0, os.path.join(HERE, "..", "src", "fiber_kit"))
    from fiber_intrachunk import _residual_recut  # noqa: E402

NSAMP, NCH = 32, 4
fails = 0
ran = 0


def check(ok, what):
    global fails, ran
    ran += 1
    print(("  ok:   " if ok else "  FAIL: ") + what)
    if not ok:
        fails += 1


def bump(center, width, chans, amps):
    t = np.zeros((NSAMP, NCH))
    x = np.arange(NSAMP)
    for c, a in zip(chans, amps):
        t[:, c] = a * np.exp(-0.5 * ((x - center) / width) ** 2)
    return t


TPL_A = bump(12, 2.5, (0, 1), (80.0, -55.0))
TPL_B = bump(18, 4.0, (2, 3), (-70.0, 40.0))
r = np.random.default_rng(3)
mixed = np.concatenate([TPL_A[None] + r.normal(0, 3, (300, NSAMP, NCH)),
                        TPL_B[None] + r.normal(0, 3, (150, NSAMP, NCH))])
clean = TPL_A[None] + r.normal(0, 3, (200, NSAMP, NCH))
W = np.concatenate([mixed, clean]).astype(np.float32)
U0 = np.concatenate([np.full(450, 5), np.full(200, 6)]).astype(np.int64)


def gw(ix):
    return W[np.asarray(ix)]


new, n_new, n_shed = _residual_recut(U0, gw, gate="all", peak=None)
check((new[:300] == 5).all(), "the dominant component keeps the unit id")
check(len(set(new[300:450].tolist())) == 1 and int(new[300]) > 6 and n_new == 1,
      "the contamination becomes exactly ONE new unit (same-shape siblings and "
      "phase splits fold back under the registered-cosine acceptance)")
check((new[450:] == 6).all(), "a clean unit is untouched")

off, n_off, _ = _residual_recut(U0, gw, gate="", peak=None)
check((off == U0).all() and n_off == 0, "gate '' is a no-op")

# giqr gate: the mixed unit (wide own-gain IQR) is recut; a clean unit whose
# IQR sits under the bar is skipped even though gate machinery runs.
gq, n_gq, _ = _residual_recut(U0, gw, gate="giqr", giqr_thr=0.4, peak=None)
check(n_gq == 1 and (gq[450:] == 6).all() and len(set(gq[300:450].tolist())) == 1,
      "gate 'giqr' recuts the impure unit and skips the clean one")

# below min_spk nothing is attempted
small = np.concatenate([mixed[:50]]).astype(np.float32)
us = np.full(50, 9, np.int64)
s2, ns2, _ = _residual_recut(us, lambda ix: small[np.asarray(ix)], gate="all",
                             min_spk=60, peak=None)
check((s2 == 9).all() and ns2 == 0, "units under min_spk are never touched")

# reserve and noise ids sit out entirely
ur = np.concatenate([np.full(450, 1), np.full(200, 0)]).astype(np.int64)
r2, nr2, sr2 = _residual_recut(ur, gw, gate="all", peak=None)
check((r2 == ur).all() and nr2 == 0 and sr2 == 0, "reserve/noise ids sit out")

print(f"\n{ran} checks, {fails} failed")
sys.exit(1 if fails else 0)
