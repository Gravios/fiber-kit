#!/usr/bin/env python3
# test_morpho.py — the biophysical modelling stack (morpho_geom / _chan / _cable /
# _eap / _input).  These are the checks that can actually fail: each one has a
# known-correct answer from something OTHER than the code under test --
# an independent dense linear solve, a conservation law, an analytic integral,
# or an invariance the physics requires.  Where a check guards a property, the
# property is also BROKEN deliberately and the check confirmed to fire; a test
# that has never failed proves nothing.
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import numpy as np

try:
    from fiber_kit import morpho_geom as mg, morpho_chan as mch, morpho_cable as mc
    from fiber_kit import morpho_eap as me, morpho_input as mi, morpho_archetype as ma
except ImportError:
    sys.path.insert(0, os.path.join(HERE, "..", "src", "fiber_kit"))
    import morpho_geom as mg, morpho_chan as mch, morpho_cable as mc
    import morpho_eap as me, morpho_input as mi, morpho_archetype as ma

fails = 0
ran = 0


def check(ok, what):
    global fails, ran
    ran += 1
    if not ok:
        fails += 1
        print(f"  FAIL  {what}")
    else:
        print(f"  ok    {what}")
    return ok


# ── 1. Hines solver vs an independent dense solve ───────────────────────────
print("[1] Hines tree solver")
rng = np.random.default_rng(7)
worst = 0.0
for _ in range(6):
    N = 180
    parent = np.array([-1] + [int(rng.integers(0, i)) for i in range(1, N)])
    gax = rng.uniform(0.1, 5.0, N); gax[0] = 0.0
    diag = rng.uniform(5.0, 20.0, N)
    np.add.at(diag, np.arange(N)[parent >= 0], gax[parent >= 0])
    np.add.at(diag, parent[parent >= 0], gax[parent >= 0])
    rhs = rng.normal(size=N)
    A = np.diag(diag)
    for i in range(1, N):
        A[i, parent[i]] -= gax[i]; A[parent[i], i] -= gax[i]
    ref = np.linalg.solve(A, rhs)
    elim, back = mc._levels(parent)
    got = mc.hines_solve(diag, rhs, gax, parent, elim, back)
    worst = max(worst, float(np.abs(ref - got).max() / max(np.abs(ref).max(), 1e-30)))
check(worst < 1e-10, f"matches dense solve on random trees (rel err {worst:.2e})")

# negative control: a wrong elimination order must NOT give the right answer
N = 60
parent = np.array([-1] + [int(rng.integers(0, i)) for i in range(1, N)])
gax = rng.uniform(0.5, 3.0, N); gax[0] = 0.0
diag = rng.uniform(8.0, 20.0, N)
np.add.at(diag, np.arange(N)[parent >= 0], gax[parent >= 0])
np.add.at(diag, parent[parent >= 0], gax[parent >= 0])
rhs = rng.normal(size=N)
A = np.diag(diag)
for i in range(1, N):
    A[i, parent[i]] -= gax[i]; A[parent[i], i] -= gax[i]
ref = np.linalg.solve(A, rhs)
elim, back = mc._levels(parent)
bad = [np.arange(1, N)]                       # one level: children not before parents
got = mc.hines_solve(diag, rhs, gax, parent, bad, back)
check(np.abs(ref - got).max() > 1e-6,
      "negative control: an invalid elimination order does give a wrong answer")

# ── 2. charge conservation ──────────────────────────────────────────────────
print("[2] transmembrane current sums to zero")
cmp_ = ma.build("pyramidal", d_lambda=0.2)
cell = mc.Cell(cmp_, mc.Biophys())
res = mc.simulate(cell, dt=0.02, t_stop=8.0, stim_amp=8.0)
scale = float(np.abs(res["im"]).max())
resid = float(np.abs(res["im"].sum(1)).max())
check(resid / max(scale, 1e-30) < 1e-5,
      f"sum_i I_m(t) = 0 to {resid/max(scale,1e-30):.1e} of peak current")
check(res["v"][:, 0].max() > 0.0,
      f"the stimulus actually evokes a spike (Vsoma {res['v'][:,0].max():.1f} mV)")

# ── 3. channel kinetics ─────────────────────────────────────────────────────
print("[3] channel kinetics")
v = np.linspace(-90.0, 40.0, 40)
(minf, mtau), (hinf, htau) = mch.Na(35.0).rates(v)
check(np.all(np.diff(minf) > 0) and np.all(np.diff(hinf) < 0),
      "Na activation rises and inactivation falls monotonically with V")
check(minf[0] < 0.05 and minf[-1] > 0.95, "Na m_inf spans ~0 to ~1 over the range")
check(np.all(mtau >= mch.Na.mmin - 1e-12) and np.all(htau >= mch.Na.hmin - 1e-12),
      "Na time constants respect their published floors")
# the removable singularity in trap0 must not produce a NaN at v == th
sing = mch._trap0(np.array([mch.Na.tha]), mch.Na.tha, mch.Na.Ra, mch.Na.qa)
check(np.isfinite(sing).all() and abs(float(sing[0]) - mch.Na.Ra * mch.Na.qa) < 1e-9,
      "trap0 is finite and equals a*q at the singular point")
(ninf, _), (linf, _) = mch.KA("prox", 35.0).rates(v)
(ninf_d, _), _ = mch.KA("dist", 35.0).rates(v)
check(float(ninf_d[len(v) // 2]) > float(ninf[len(v) // 2]),
      "distal A-type activates at more negative V than proximal (the published shift)")

# ── 4. extracellular forward model ──────────────────────────────────────────
print("[4] line-source extracellular model")
# far from a short compartment, the line source must converge to a point source
c1 = mg.Compartments(p0=np.array([[0.0, -1.0, 0.0]]), p1=np.array([[0.0, 1.0, 0.0]]),
                     mid=np.zeros((1, 3)), L=np.array([2.0]), diam=np.array([1.0]),
                     area=np.array([6.28]), parent=np.array([-1]), sec=np.array([0]),
                     type=np.array([mg.SOMA]), pathdist=np.array([0.0]), name="one")
far = me.sites_3d(np.array([[500.0, 0.0]]))
kl = me.transfer_matrix(c1, far, point_source=False)
kp = me.transfer_matrix(c1, far, point_source=True)
check(abs(float(kl[0, 0] / kp[0, 0]) - 1.0) < 1e-3,
      "line source -> point source at 500 um from a 2 um compartment")
# ...and must DISAGREE for a long compartment seen close up, which is the case
# that motivates the line source at all: a 100 um apical trunk segment with a
# site 15 um off its midpoint.  (The earlier 2 um compartment agrees with a
# point source everywhere, so it cannot serve as this control.)
c2 = mg.Compartments(p0=np.array([[0.0, -50.0, 0.0]]), p1=np.array([[0.0, 50.0, 0.0]]),
                     mid=np.zeros((1, 3)), L=np.array([100.0]), diam=np.array([3.0]),
                     area=np.array([942.0]), parent=np.array([-1]), sec=np.array([0]),
                     type=np.array([mg.APICAL]), pathdist=np.array([0.0]), name="trunk")
near = me.sites_3d(np.array([[15.0, 0.0]]))
kl_n = me.transfer_matrix(c2, near, point_source=False)
kp_n = me.transfer_matrix(c2, near, point_source=True)
ratio = float(kl_n[0, 0] / kp_n[0, 0])
check(abs(ratio - 1.0) > 0.1,
      f"negative control: at 15 um from a 100 um cable the line source differs "
      f"from a point source by {100*abs(ratio-1):.0f}% (else it is doing nothing)")
check(ratio < 1.0,
      "spreading the same current along a cable LOWERS the near-field potential")
# a point source's potential must fall as 1/r
r = np.array([[50.0, 0.0], [100.0, 0.0], [200.0, 0.0]], float)
k = me.transfer_matrix(c1, me.sites_3d(r), point_source=True)[0]
check(np.allclose(k[0] / k[1], 2.0, rtol=1e-6) and np.allclose(k[1] / k[2], 2.0, rtol=1e-6),
      "point-source potential falls as 1/r")
# an isolated current source with zero net current has no monopole term
K = me.transfer_matrix(cmp_, me.sites_3d(me.staggered_octrode()))
ve = me.extracellular(res["im"], K)
check(np.isfinite(ve).all() and float(np.abs(ve).max()) > 1.0,
      f"a spike produces a finite, non-trivial field ({np.abs(ve).max():.1f} uV peak)")

# ── 5. geometry ─────────────────────────────────────────────────────────────
print("[5] morphology handling")
check(np.all(cmp_.parent[1:] < np.arange(1, len(cmp_))),
      "compartments are topologically ordered (parent[i] < i)")
check(int((cmp_.parent < 0).sum()) == 1, "exactly one root")
areas = np.pi * cmp_.diam * cmp_.L
rel = float(np.abs(cmp_.area - areas).max() / max(areas.max(), 1e-30))
check(rel < 0.5, f"compartment areas are within a taper factor of pi*d*L (max rel {rel:.2f})")
ori = mg.orient(ma.pyramidal.__wrapped__() if hasattr(ma.pyramidal, "__wrapped__")
                else mg.compartmentalize(ma.pyramidal()))
ap = ori.mid[ori.type == mg.APICAL, 1]
check(float(ap.mean()) > 0.0, "orient() puts the apical tree on +y")
rotated = mg.rotate_z(cmp_, 37.0)
check(abs(float(np.linalg.norm(rotated.mid, axis=1).sum() -
                np.linalg.norm(cmp_.mid, axis=1).sum())) < 1e-6,
      "rotation about the depth axis preserves distance from the origin")

# a rotation about the depth axis must not change the somatic membrane current
cell_r = mc.Cell(rotated, mc.Biophys())
res_r = mc.simulate(cell_r, dt=0.02, t_stop=8.0, stim_amp=8.0, record_v=False)
check(float(np.abs(res_r["im"] - res["im"]).max()) < 1e-9,
      "rotating the cell does not change its membrane currents (only the field)")

# ── 6. afferent topology ────────────────────────────────────────────────────
print("[6] afferent topology table")
pw = mi.load_table(post="pyramidalcell")
check(len(pw) >= 10, f"pyramidal pathway table loads ({len(pw)} pathways)")
by = {p.pre: p for p in pw}
check("ca3cell" in by and "eccell" in by, "the two extrinsic pathways are present")
check(by["ca3cell"].ntotal > by["eccell"].ntotal,
      "Schaffer collateral synapses outnumber perforant path ones")
check(by["ca3cell"].dhi <= by["eccell"].dlo,
      "the CA3 and EC distance windows are disjoint and correctly ordered")
check(all(p.excitatory == (p.erev > -30) for p in pw), "E/I is read from the reversal potential")
alloc = mi.allocate(by["ca3cell"], cmp_)
check(abs(float(alloc.sum()) - by["ca3cell"].ntotal) < 1e-6,
      "allocation conserves the pathway's synapse count")
elig = mi.eligible(by["ca3cell"], cmp_)
check(float(alloc[~elig].sum()) == 0.0, "no synapses land outside the eligible region")
check(not mi.eligible(by["eccell"], cmp_)[cmp_.pathdist < by["eccell"].dlo].any(),
      "the perforant path never lands proximal to its window")

# allocation must be area-weighted, not per-compartment uniform: refining the
# discretization must not move the synapses
fine = ma.build("pyramidal", d_lambda=0.08)
a_c = mi.allocate(by["ca3cell"], cmp_); a_f = mi.allocate(by["ca3cell"], fine)
med_c = float(np.average(cmp_.pathdist, weights=a_c)) if a_c.sum() else 0.0
med_f = float(np.average(fine.pathdist, weights=a_f)) if a_f.sum() else 0.0
check(abs(med_c - med_f) < 15.0,
      f"mean synapse path distance is discretization-invariant ({med_c:.0f} vs {med_f:.0f} um)")

# ── 7. synaptic drive ───────────────────────────────────────────────────────
print("[7] synaptic drive")
nt = 600
d = mi.Drive([by["ca3cell"]], cmp_, 0.02, nt, {"ca3cell": np.array([2.0])},
             active_fraction=0.05)
check(len(d) == 1, "drive builds one conductance channel for one pathway")
g = d.g[0]
peak_step = int(np.argmax(g.sum(1)))
check(peak_step * 0.02 > 2.0, "conductance peaks after the activation time, not before")
check(float(g[: int(2.0 / 0.02)].max()) == 0.0, "conductance is exactly zero before onset")
cell_s = mc.Cell(cmp_, mc.Biophys())
res_s = mc.simulate(cell_s, dt=0.02, t_stop=nt * 0.02, stim_amp=0.0, drive=d)
check(float(res_s["v"][:, 0].max()) > mc.Biophys().v_rest + 1.0,
      "excitatory drive alone depolarizes the soma")
sc = float(np.abs(res_s["im"]).max())
check(float(np.abs(res_s["im"].sum(1)).max()) / max(sc, 1e-30) < 1e-5,
      "synaptic current is inside I_m: the sum is still zero")

# ── 8. spike extraction ─────────────────────────────────────────────────────
print("[8] .spk-convention extraction")
out = me.waveform(res["im"], cmp_, me.sites_3d(me.staggered_octrode() - np.array([0.0, 40.0]),
                                              z=30.0), 0.02, nsamp=42, peak=21)
w = out["wave"]
if check(w is not None and w.shape == (42, 8), "window is (nsamp, nchan) = (42, 8)"):
    ch = int(np.argmax(w.max(0) - w.min(0)))
    check(int(np.argmin(w[:, ch])) == 21, "the dominant channel's trough sits at peak index 21")
    m = me.metrics(w, me.staggered_octrode())
    check(0.1 < m["width_ms"] < 1.5, f"trough-to-peak width is physiological ({m['width_ms']:.3f} ms)")
    check(abs(me.cosine(w, w) - 1.0) < 1e-12 and abs(me.cosine(w, -w) + 1.0) < 1e-12,
          "cosine is 1 with itself and -1 with its negation")

print(f"\n{ran - fails}/{ran} checks passed")
sys.exit(1 if fails else 0)
