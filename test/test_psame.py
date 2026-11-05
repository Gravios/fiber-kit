#!/usr/bin/env python3
"""fiber-psame-merge: history-conditioned cell models and the P(same) merge.

  1. ewma_state equals fiber_adapt.ewma (the reference recurrence) on a random train; the default
     10-tau horizon is within 1e-3 of it.
  2. A bursting cell (later burst spikes smaller AND differently shaped, the amplitude set by the cell's
     own recent firing; noise 15 against a 100 peak, SNR ~7 as on g6) cut in time into two clusters:
     one history-conditioned model explains the union (dbic > 0) and psame_merge joins them.
  3. A second cell with A's AVERAGE waveform but tonic firing (no bursts, no history-driven amplitude)
     is NOT joined to either half (dbic < 0) -- the negative control: roll-cosine alone ranks it as
     close as A's true other half, so the cell model is what separates them.
  Not covered, because it fails: a cell cut by a hard AMPLITUDE threshold.  Each half is then narrower
  than the cell along the noisy amplitude axis (selection), the union always fits worse, and dbic < 0
  (synthetic: -0.53 for the true pair, -0.06 for a tonic look-alike).  See the module docstring.
  4. A distinct cell stays alone; the input labels are not modified in place.
Exits nonzero on failure.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from fiber_kit import fiber_psame as ps, fiber_adapt as fa  # noqa: E402

fails = 0


def check(name, ok, detail=""):
    global fails
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail and not ok else ""))
    fails += 0 if ok else 1


rng = np.random.default_rng(3)
t = np.sort(rng.uniform(0, 50, 400))
check("ewma_state = fiber_adapt.ewma (full horizon)", np.allclose(ps.ewma_state(t, 0.3, horizon=60.0), fa.ewma(t, 0.3), atol=1e-9))
check("ewma_state, default 10-tau horizon, within 1e-3", np.allclose(ps.ewma_state(t, 0.3), fa.ewma(t, 0.3), atol=1e-3))

NS, NC = 32, 8
tt = np.arange(NS)


def template(c0, w):
    T = np.zeros((NS, NC))
    for c in range(NC):
        g = np.exp(-0.5 * ((c - c0) / 1.5) ** 2)
        T[:, c] = -g * np.exp(-0.5 * ((tt - 12) / w) ** 2) + 0.4 * g * np.exp(-0.5 * ((tt - 17) / (2 * w)) ** 2)
    return T * 100


TA = template(3.0, 1.2)
SA = template(3.6, 2.0) - template(3.0, 1.2)        # shape change per unit amplitude loss
TC = template(6.5, 1.0)
spk, times, lab = [], [], []
# cell A: bursts of 4 at 5 ms; the amplitude follows the cell's own recent firing (an EWMA adaptation
# state, tau 30 ms) and the shape follows S as the amplitude drops
tA = np.sort(np.concatenate([t0 + 0.005 * np.arange(4) for t0 in rng.uniform(0, 300, 300)]))
aA = np.clip(1.0 - 0.25 * fa.ewma(tA, 0.03), 0.3, 1.0)
for t_, a in zip(tA, aA):
    spk.append(a * TA + (1 - a) * SA + rng.normal(0, 15, (NS, NC))); times.append(t_); lab.append(0)
# cell B: A's AVERAGE waveform, but tonic (Poisson): no burst structure, no history-driven amplitude
TB = aA.mean() * TA + (1 - aA.mean()) * SA
for t0 in np.sort(rng.uniform(0, 300, 900)):
    spk.append(TB * rng.uniform(0.9, 1.1) + rng.normal(0, 15, (NS, NC))); times.append(t0); lab.append(1)
# cell C: elsewhere
for t0 in np.sort(rng.uniform(0, 300, 600)):
    spk.append(TC + rng.normal(0, 15, (NS, NC))); times.append(t0); lab.append(2)
o = np.argsort(times)
spk = np.asarray(spk, np.float32)[o]; times = np.asarray(times)[o]; lab = np.asarray(lab)[o]
amp = np.array([float((w * TA).sum() / (TA * TA).sum()) for w in spk])
clu = np.where(lab == 0, np.where(times < 150, 2, 3), np.where(lab == 1, 4, 5))   # A cut in time (drift-like)
check("setup: A's two fragments both populated", (clu == 2).sum() > 100 and (clu == 3).sum() > 100,
      f"{(clu == 2).sum()} / {(clu == 3).sum()}")
mu, V = ps.pca_basis(spk, n=12, nfit=len(spk))
M = {u: ps.CellModel(ps.project(spk, np.flatnonzero(clu == u), mu, V), times[clu == u]) for u in (2, 3, 4, 5)}
d_true = ps.dbic(M[2], M[3]); d_b3 = ps.dbic(M[4], M[3]); d_b2 = ps.dbic(M[2], M[4])
rc_true = ps.rollcos(spk[clu == 2].mean(0), spk[clu == 3].mean(0))
rc_b = ps.rollcos(spk[clu == 4].mean(0), spk[clu == 3].mean(0))
check("history model: A's fragments are one cell (dbic > 0)", d_true > 0, f"{d_true:.3f}")
check("negative control: B is not one of A's halves (dbic < 0)", d_b3 < 0 and d_b2 < 0, f"{d_b3:.3f}, {d_b2:.3f}")
check("roll-cosine alone cannot tell them apart (B as close to A's second half)", rc_b >= rc_true - 0.02,
      f"rc B-A2 {rc_b:.3f} vs A1-A2 {rc_true:.3f}")
check("A's halves score above B (dbic)", d_true > max(d_b3, d_b2) + 0.5, f"{d_true:.3f} vs {d_b3:.3f}, {d_b2:.3f}")
clu_in = clu.copy()
new, merges = ps.psame_merge(spk, times, clu, theta=0.5, cannot_link=0.3, knn=3, cos_floor=0.5, min_n=50,
                             w_rc=0.0, w_dbic=4.0, bias=0.0, log=lambda *_: None)
check("A's fragments merged", len(np.unique(new[lab == 0])) == 1, str(merges))
check("B stays apart from A", not set(np.unique(new[lab == 1])) & set(np.unique(new[lab == 0])))
check("C stays alone", len(np.unique(new[lab == 2])) == 1 and not set(np.unique(new[lab == 2])) & set(np.unique(new[lab != 2])))
check("input labels untouched", (clu == clu_in).all())
print("OK (0 failures)" if not fails else f"FAILED ({fails})")
sys.exit(1 if fails else 0)
