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
(minf, mtau), (hinf, htau), (sinf, staus) = mch.Na(35.0).rates(v)
check(np.all(np.diff(minf) > 0) and np.all(np.diff(hinf) < 0),
      "Na activation rises and inactivation falls monotonically with V")
check(minf[0] < 0.05 and minf[-1] > 0.95, "Na m_inf spans ~0 to ~1 over the range")
check(np.all(mtau >= mch.Na.mmin - 1e-12) and np.all(htau >= mch.Na.hmin - 1e-12)
      and np.all(staus >= mch.Na.smax - 1e-9),
      "Na time constants respect their published floors")
check(np.allclose(sinf, 1.0), "at the published ar=1 the slow gate is identically 1")
(_, _), (_, _), (sinf_ar, _) = mch.Na(35.0, ar=0.5).rates(v)
check(sinf_ar[0] > 0.99 and abs(float(sinf_ar[-1]) - 0.5) < 1e-3
      and np.all(np.diff(sinf_ar) <= 1e-12),
      "with ar=0.5 the slow gate falls monotonically from 1 to the ar floor")
check(float(np.median(staus)) >= 10.0,
      f"slow inactivation is slow ({np.median(staus):.0f} ms median tau, vs "
      f"{np.median(htau):.2f} ms for h) — the two gates cover different timescales")
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

# ── 9. physiological merge envelope ─────────────────────────────────────────
print("[9] physiological merge envelope")
try:
    from fiber_kit import morpho_envelope as mv
except ImportError:
    import morpho_envelope as mv

check(mv.train_times("burst4_5") == [5.0, 10.0, 15.0, 20.0], "burst ISI pattern expands correctly")
check(len(mv.train_times("tonic_20_4")) == 4 and
      abs(mv.train_times("tonic_20_4")[1] - mv.train_times("tonic_20_4")[0] - 50.0) < 1e-9,
      "tonic rate maps to the right ISI")
check(len(mv.train_times("recover_4_300")) == 5 and
      mv.train_times("recover_4_300")[-1] - mv.train_times("recover_4_300")[-2] == 300.0,
      "the recovery pattern puts a long gap before its last spike")

# a train must show USE-dependent amplitude decrement -- the property the whole
# envelope rests on.  If a burst's spikes were identical there would be nothing
# for a gate to license, so this is the load-bearing check.
cell_t = mc.Cell(cmp_, mc.Biophys(na_ar=0.5))
tt = mv.train_times("burst4_4")
_, im_t, v_t = mv.simulate_train(cell_t, tt, dt=0.02, stim_amp=6.0)
sites_t = me.sites_3d(me.staggered_octrode() - np.array([0.0, 40.0]), z=30.0)
Wt, kept = mv.train_footprints(im_t, cmp_, sites_t, tt, 0.02, v=v_t, v_thresh=0.0)
if check(len(Wt) >= 3, f"a 4-spike burst yields at least 3 detected footprints ({len(Wt)})"):
    p2p_t = (Wt.max(1) - Wt.min(1)).max(1)
    check(p2p_t[0] > p2p_t[-1] * 1.05,
          f"amplitude decrements within the burst ({p2p_t[0]:.0f} -> {p2p_t[-1]:.0f} uV)")
    r_t, d_t = mv.pairwise(Wt)
    check(np.all(r_t >= 1.0) and len(r_t) == len(Wt) * (len(Wt) - 1) // 2,
          "pairwise returns one ratio >= 1 per spike pair")
    check(float(np.corrcoef(r_t, d_t)[0, 1]) > 0.0,
          "shape difference grows with amplitude ratio (the envelope's premise)")

# negative control: with the slow gate pinned off AND a saturating stimulus the
# decrement must largely vanish, or the effect above was not use-dependence.
cell_n = mc.Cell(cmp_, mc.Biophys(na_ar=1.0))
_, im_n, v_n = mv.simulate_train(cell_n, tt, dt=0.02, stim_amp=20.0)
Wn, _ = mv.train_footprints(im_n, cmp_, sites_t, tt, 0.02, v=v_n, v_thresh=0.0)
if len(Wn) >= 3 and len(Wt) >= 3:
    pn = (Wn.max(1) - Wn.min(1)).max(1)
    check((pn.max() / pn.min()) < (p2p_t.max() / p2p_t.min()),
          f"negative control: saturating drive with ar=1 decrements less "
          f"({pn.max()/pn.min():.2f} vs {p2p_t.max()/p2p_t.min():.2f})")

# detection threshold must BOUND the reachable amplitude ratio
W_all, _ = mv.train_footprints(im_t, cmp_, sites_t, tt, 0.02, v=v_t, detect_uv=0.0)
W_det, _ = mv.train_footprints(im_t, cmp_, sites_t, tt, 0.02, v=v_t, detect_uv=40.0)
if len(W_all) > len(W_det) >= 2:
    ra = (W_all.max(1) - W_all.min(1)).max(1)
    rd = (W_det.max(1) - W_det.min(1)).max(1)
    check((rd.max() / rd.min()) <= (ra.max() / ra.min()) + 1e-9,
          "raising the detection threshold cannot widen the reachable amplitude ratio")

# envelope construction and its two rejection modes
rng2 = np.random.default_rng(3)
rr = np.concatenate([np.full(400, 1.02), np.full(400, 1.5)])
dd = np.concatenate([abs(rng2.normal(0, 0.004, 400)), abs(rng2.normal(0, 0.05, 400))])
env = mv.build_envelope(rr, dd, q=0.99, nbin=6)
check(np.all(np.diff(env.cos_thr) >= -1e-12), "envelope thresholds are non-decreasing in ratio")
check(float(env.allowed(1.5)) > float(env.allowed(1.02)),
      "a wider amplitude ratio licenses more shape change")
base = np.zeros((42, 8)); base[21, 3] = -100.0; base[25, 3] = 30.0
ok_same, rho_s, d_s, _ = env.admissible(base, base * 1.01)
check(ok_same and abs(rho_s - 1.01) < 1e-6 and d_s < 1e-9,
      "a pure rescale of the same shape is admissible")
far = base.copy(); far[21, 3] = -100.0; far[21, 6] = -90.0
ok_shape = env.admissible(base, far)[0]
check(not ok_shape, "a footprint on a different channel is rejected on SHAPE")
ok_amp, rho_a, _, _ = env.admissible(base, base * 40.0)
check(not ok_amp and rho_a > env.ratio_max,
      "an unreachable amplitude ratio is rejected on RATIO, separately from shape")

import tempfile as _tf
with _tf.TemporaryDirectory() as td:
    pth = os.path.join(td, "env.npz")
    env.save(pth)
    e2 = mv.Envelope.load(pth)
    check(np.allclose(e2.cos_thr, env.cos_thr) and abs(e2.ratio_max - env.ratio_max) < 1e-12,
          "an envelope round-trips through save/load")

# ── 10. validation primitives ───────────────────────────────────────────────
print("[10] validation primitives")
try:
    from fiber_kit import morpho_validate as mvd
except ImportError:
    import morpho_validate as mvd


class _FakeSort:
    """Minimal stand-in with the attributes the measurements touch, so the
    statistics can be checked against constructed data whose answer is known.
    Building it from a real session would make these tests unrunnable anywhere
    the session is absent, which is everywhere except one machine."""

    def __init__(self, res, clu, spk):
        self.res, self.clu, self.spk = res, clu, spk

    sizes = mvd.Sort.sizes
    idx = mvd.Sort.idx
    template = mvd.Sort.template
    minutes = mvd.Sort.minutes


SR = 32552.0
rng3 = np.random.default_rng(11)
base_w = np.zeros((42, 8), np.float32); base_w[21, 3] = -400.0; base_w[26, 3] = 120.0
base_w[21, 2] = -180.0; base_w[21, 4] = -160.0

# unit A: 4000 spikes, Poisson at 5 Hz over 600 s, identical shape + noise
ta = np.sort(rng3.uniform(0, 600, 4000)) * SR
wa = base_w[None] + rng3.normal(0, 6.0, (4000, 42, 8)).astype(np.float32)
# unit B: a *different* shape, on another channel
bw = np.zeros((42, 8), np.float32); bw[21, 6] = -400.0; bw[26, 6] = 120.0
tb = np.sort(rng3.uniform(0, 600, 800)) * SR
wb = bw[None] + rng3.normal(0, 6.0, (800, 42, 8)).astype(np.float32)
# unit C: A's own spikes, always 3-6 ms after an A spike (a burst continuation)
pick = rng3.choice(len(ta), 600, replace=False)
tc = ta[pick] + rng3.uniform(0.003, 0.006, 600) * SR
wc = base_w[None] * 0.97 + rng3.normal(0, 6.0, (600, 42, 8)).astype(np.float32)

res = np.concatenate([ta, tb, tc]).astype(np.int64)
clu = np.concatenate([np.full(len(ta), 1), np.full(len(tb), 2), np.full(len(tc), 3)])
spk = np.concatenate([wa, wb, wc])
o = np.argsort(res)
fs = _FakeSort(res[o], clu[o], spk[o])

check(fs.sizes() == {1: 4000, 2: 800, 3: 600}, "cluster sizes are recovered")
d_ab = mvd.cos_dist(fs.template(1), fs.template(2))
d_ac = mvd.cos_dist(fs.template(1), fs.template(3))
check(d_ab > 0.9 and d_ac < 0.02,
      f"a different-channel unit is far ({d_ab:.3f}) and a rescaled copy is near ({d_ac:.4f})")

nf = mvd.split_half_noise(fs, 1)
check(0.0 < nf < 0.02, f"split-half noise of a stationary unit is small ({nf:.4f})")
nf_small = mvd.split_half_noise(fs, 3)
check(nf_small > nf,
      f"a smaller cluster has a larger noise floor ({nf_small:.4f} > {nf:.4f}) — "
      "the reason a raw distance cannot be read without it")

tb6 = mvd.time_budget(fs, 1, 6)
check(tb6.shape == (6, 6) and np.nanmax(tb6) < 0.03 and
      np.allclose(np.diag(tb6), 0.0, atol=1e-9),
      f"a non-drifting unit has a small time budget ({np.nanmax(tb6):.4f}) and zero diagonal")

lat_c, ch_c = mvd.latency_enrichment(fs, 1, 3, SR, 10.0)
lat_b, ch_b = mvd.latency_enrichment(fs, 1, 2, SR, 10.0)
check(lat_c > 0.9, f"a burst continuation is strongly enriched ({100*lat_c:.0f}%)")
check(abs(lat_b - ch_b) < 0.03,
      f"an independent unit sits at chance ({100*lat_b:.1f}% vs {100*ch_b:.1f}%)")
check(lat_c / max(ch_c, 1e-9) > 5.0, "enrichment is many-fold over chance")

r1 = mvd.refractory(fs, [1], SR, 2.0)
r13 = mvd.refractory(fs, [1, 3], SR, 2.0)
check(r13 > r1, f"merging a short-latency partner costs refractory ({100*r1:.2f}% -> "
                f"{100*r13:.2f}%)")

# recovery: build a unit whose amplitude genuinely DOES depend on preceding ISI,
# and confirm the measurement recovers the imposed law rather than inventing one.
n = 6000
gaps = np.concatenate([rng3.uniform(0.003, 0.010, n // 2),
                       rng3.uniform(0.3, 1.0, n - n // 2)])
rng3.shuffle(gaps)
tt = np.cumsum(gaps)
scale = np.ones(n)
scale[1:] = np.where(np.diff(tt) < 0.020, 0.60, 1.0)
ww = (base_w[None] * scale[:, None, None]).astype(np.float32) + \
    rng3.normal(0, 4.0, (n, 42, 8)).astype(np.float32)
fs2 = _FakeSort((tt * SR).astype(np.int64), np.ones(n, np.int64), ww)
rows, b0, niso = mvd.recovery_curve(fs2, [1], SR)
short = [r for r in rows if r[1] <= 12]
long_ = [r for r in rows if r[0] >= 200]
if check(short and long_, "recovery curve produced both short- and long-ISI bins"):
    check(abs(np.mean([r[3] for r in short]) - 0.60) < 0.05,
          f"the imposed 0.60 short-ISI ratio is recovered "
          f"({np.mean([r[3] for r in short]):.3f})")
    check(abs(long_[0][3] - 1.0) < 0.02, "the isolated bin normalises to 1.0")

# negative control: a unit with NO ISI dependence must come back flat, or the
# measurement is manufacturing the effect it is meant to detect.
fs3 = _FakeSort((tt * SR).astype(np.int64), np.ones(n, np.int64),
                (base_w[None] + rng3.normal(0, 4.0, (n, 42, 8))).astype(np.float32))
rows3, _, _ = mvd.recovery_curve(fs3, [1], SR)
spread = max(r[3] for r in rows3) - min(r[3] for r in rows3)
check(spread < 0.05,
      f"negative control: an ISI-independent unit gives a flat curve (spread {spread:.3f})")

# ── 11. session feature space ───────────────────────────────────────────────
print("[11] session feature space (PCAE basis, lagged projection)")
try:
    from fiber_kit import morpho_features as mf
except ImportError:
    import morpho_features as mf

import struct as _st
import tempfile as _tf2


def _write_pcae(path, nch, d2u, ncomp, rec, cen, method, evec, means):
    with open(path, "wb") as fh:
        fh.write(_st.pack("<9i", mf.PCAE_MAGIC, 2, nch, d2u, ncomp, rec,
                          1 if cen else 0, method, nch))
        for ch in range(nch):
            fh.write(np.asarray(means[ch], np.float64).tobytes())
        for ch in range(nch):
            fh.write(np.asarray(evec[ch], np.float64).tobytes())


rngf = np.random.default_rng(23)
NCH, D2U, NCOMP, REC = 4, 21, 4, 5
# build a basis with the session's own structure: one shape at three lags, plus
# a second shape at lag 0.  The loader must RECOVER that structure from the file.
pc1 = np.zeros(D2U); pc1[3:18] = np.sin(np.linspace(0, 2 * np.pi, 15))
pc2 = np.zeros(D2U); pc2[3:18] = np.cos(np.linspace(0, 2 * np.pi, 15))
ev = np.zeros((NCH, NCOMP, D2U))
for ch in range(NCH):
    ev[ch, 0] = pc1
    ev[ch, 1] = np.concatenate([np.zeros(3), pc1[:-3]])   # zero-padded, as the
    ev[ch, 2] = np.concatenate([np.zeros(6), pc1[:-6]])   # real basis stores it
    ev[ch, 3] = pc2
mn = rngf.normal(0, 1, (NCH, D2U))

with _tf2.TemporaryDirectory() as td:
    pth = os.path.join(td, "s.pca.1")
    _write_pcae(pth, NCH, D2U, NCOMP, REC, False, 8, ev, mn)
    b = mf.load_pca(pth)
    check(b.nch == NCH and b.data2use == D2U and b.ncomp == NCOMP and b.rec_shift == REC,
          "PCAE header round-trips")
    check(b.method == 8 and b.method_tag == "stderiv_C5" and b.temporal_diff,
          "method 8 is stderiv_C5 and applies the temporal difference")
    check(np.allclose(b.evec, ev) and np.allclose(b.means, mn),
          "block-wise body (all means, then all eigenvectors) is read correctly")

    ls = b.lag_structure()[0]
    check([r[1] for r in ls] == [0, 0, 0, 3] and [r[2] for r in ls] == [0, -3, -6, 0],
          f"the lag triple is RECOVERED from the basis, not assumed ({ls})")

    # negative control: an interleaved reader would mangle the body, so confirm
    # the loader is not silently accepting a wrong layout
    bad = os.path.join(td, "bad.pca.1")
    with open(pth, "rb") as fh:
        raw = fh.read()
    with open(bad, "wb") as fh:
        fh.write(raw[:28] + _st.pack("<i", 99) + raw[32:])   # method -> invalid
    try:
        mf.load_pca(bad); ok_rej = False
    except ValueError:
        ok_rej = True
    check(ok_rej, "an invalid method in the header is refused, not coerced")

    NS = 40
    w = rngf.normal(0, 30, (25, NS, NCH))
    F = mf.project(w, b)
    check(F.shape == (25, NCH * NCOMP), f"projection is channel-major {F.shape}")
    # channel-major order: dim 4*ch+k must depend ONLY on channel ch
    w2 = w.copy(); w2[:, :, 1] = 0.0
    F2 = mf.project(w2, b)
    touched = np.flatnonzero(np.abs(F - F2).max(0) > 1e-9)
    check(set(touched.tolist()) == {4, 5, 6, 7},
          f"zeroing channel 1 changes exactly dims 4-7 ({touched.tolist()})")

    # the lag triple must behave like a lag: shifting the WAVEFORM by 3 samples
    # should map component k's value onto component k-1's
    sh = np.roll(w, 3, axis=1)
    Fs = mf.project(sh, b)
    a0 = np.corrcoef(Fs[:, 1], F[:, 0])[0, 1]
    check(a0 > 0.99, f"a 3-sample waveform shift moves PC1@lag onto its neighbour "
                     f"(corr {a0:.3f}) — the triple really is a lag basis")

    # Shifting the extraction window IS, in the interior, the same operation as
    # reading a different lag.  Asserted the opposite on the first pass and the
    # test caught it; recorded here because it kills a tempting explanation for
    # why window-shift realignment perturbs this space.  Whatever the mechanism
    # is, it is not that the two operations differ algebraically.
    Fw = mf.project(w, b, shift=-3)
    agree = float(np.corrcoef(Fw[:, 1], F[:, 0])[0, 1])
    check(agree > 0.99,
          f"a window shift moves along the SAME lag axis as the filter shift "
          f"(corr {agree:.3f}) — they are not independent knobs")

    check(abs(mf.project(w, b)[0, 0] - float(w[0, REC:REC + D2U, 0] @ ev[0, 0])) < 1e-9,
          "a projected value equals the explicit dot product over [recShift, +data2use)")

    # session_transform: order is channel-difference then temporal difference
    sets = [[1, 2], [2, 3], [3, 0], [0, 1]]
    tw = mf.session_transform(w, sets, temporal_diff=True)
    check(tw.shape == w.shape, "session_transform preserves the sample count "
                               "(so peakSampleIndex still means what it says)")
    car = mf.session_transform(w, sets, temporal_diff=False)
    check(abs(float(car.sum(-1).mean())) < abs(float(w.sum(-1).mean())) + 1e-9,
          "the channel difference removes common mode")
    d = np.diff(car, axis=-2)
    check(np.allclose(tw[:, 1:, :], d), "the temporal step is x[t]-x[t-1], applied AFTER "
                                        "the channel difference")

# alignment: a known imposed shift must be recovered
base_t = np.zeros((40, 3)); base_t[20, 1] = -500.0; base_t[24, 1] = 150.0
imposed = rngf.integers(-2, 3, 300)
wav = np.stack([np.roll(base_t, int(s), axis=0) for s in imposed]).astype(float)
wav += rngf.normal(0, 5, wav.shape)
got, sc = mf.align_shifts(wav, base_t, max_shift=3)
hit = max(float((got == imposed).mean()), float((got == -imposed).mean()))
check(hit > 0.95, f"align_shifts recovers an imposed shift ({100*hit:.0f}%)")
check(float((mf.align_shifts(np.stack([base_t] * 50), base_t)[0] == 0).mean()) == 1.0,
      "negative control: identical spikes get zero shift")

# ── 12. cable templates (L/diam, no 3-D points) ─────────────────────────────
print("[12] cable-template morphologies")
_tpl = """
begintemplate TestCell
create soma, radT1, radM1, oriT1, oriT2, axon
proc topol() {
  connect radT1(0), soma(1)
  connect radM1(0), radT1(1)
  connect oriT1(0), soma(0)
  connect oriT2(0), soma(0)
  connect axon(0), soma(0)
}
proc geom() {
  soma  { L = 20  diam = 10 }
  radT1 { L = 100 diam = 4 }
  radM1 { L = 150 diam = 3 }
  oriT1 { L = 80  diam = 2 }
  oriT2 { L = 80  diam = 2 }
  axon  { L = 200 diam = 1 }
}
endtemplate TestCell
"""
with _tf.TemporaryDirectory() as td:
    tp = os.path.join(td, "t.hoc")
    open(tp, "w").write(_tpl)
    secs_t = mg.load_cable_template(tp)
    check(len(secs_t) == 6, f"all six sections are created ({len(secs_t)})")
    tot = sum(s.length() for s in secs_t)
    check(abs(tot - 630.0) < 1.0,
          f"total cable length matches the template's own geom() ({tot:.0f} vs 630)")
    ct = mg.orient(mg.compartmentalize(secs_t), axis=(0.0, 1.0, 0.0))
    check(int((ct.parent < 0).sum()) == 1, "the connect topology gives a single root")

    # the laminar name hints must actually place radiatum above and oriens below
    yr = ct.mid[[i for i in range(len(ct)) if secs_t[ct.sec[i]].name.startswith("rad")], 1]
    yo = ct.mid[[i for i in range(len(ct)) if secs_t[ct.sec[i]].name.startswith("ori")], 1]
    check(float(yr.mean()) > 0 > float(yo.mean()),
          f"rad* is placed apical (+{yr.mean():.0f}) and ori* basal ({yo.mean():.0f}) — "
          "the section names carry anatomy that L and diam do not")
    check(ct.type[ct.sec == 5][0] == mg.AXON, "the axon is typed from its name")

    # a template with 3-D points must NOT be routed here silently, and a
    # pt3d-free file must NOT be accepted by load_hoc
    try:
        mg.load_hoc(tp); ok_ref = False
    except ValueError:
        ok_ref = True
    check(ok_ref, "load_hoc still refuses a pt3d-free template rather than "
                  "returning a cell with no geometry")

    # orient()'s default axis search must NOT be used on these: a symmetric cell
    # gets stood upright.  Confirm the two paths actually differ, so the warning
    # in the docstring is about something real.
    sym = _tpl.replace("oriT1 { L = 80  diam = 2 }", "oriT1 { L = 250 diam = 3 }")
    sym = sym.replace("oriT2 { L = 80  diam = 2 }", "oriT2 { L = 250 diam = 3 }")
    sp = os.path.join(td, "sym.hoc"); open(sp, "w").write(sym)
    cs = mg.compartmentalize(mg.load_cable_template(sp))
    a_free = mg.orient(cs)
    a_fix = mg.orient(cs, axis=(0.0, 1.0, 0.0))
    check(float(np.abs(a_free.mid - a_fix.mid).max()) > 1.0,
          "orient()'s axis search rotates a laminar layout — hence axis=(0,1,0)")

# ── 13. dispersion ──────────────────────────────────────────────────────────
print("[13] feature-space dispersion")
rngd = np.random.default_rng(31)
D = 32
# three cells of very different amplitude but the SAME intrinsic spread -- the
# constant-radius claim.  If radius tracked amplitude the whole approach fails,
# so the test asserts independence rather than assuming it.
Fs, Is = [], []
for j, gain in enumerate([1.0, 2.0, 3.5, 6.0, 9.0]):
    mu = rngd.normal(0, 50, D) * gain
    Fs.append(mu + rngd.normal(0, 20, (900, D)))
    Is.append(np.full(900, j))
# a FRAGMENT: a compact sub-region of cell 0, tighter than the cell
frag_mu = Fs[0].mean(0) + rngd.normal(0, 10, D)
Fs.append(frag_mu + rngd.normal(0, 8, (300, D))); Is.append(np.full(300, 10))
# a CONTAMINATED cluster: two cells pooled
Fs.append(np.concatenate([Fs[0][:300], Fs[1][:300]])); Is.append(np.full(600, 11))
F = np.concatenate(Fs); I = np.concatenate(Is)

keys, rad, nn, en = mvd.dispersion_table(F, I, min_spikes=100)
base = float(np.median(rad[np.isin(keys, [0, 1, 2, 3, 4])]))
check(len(keys) == 7, f"dispersion_table returns every cluster above the floor ({len(keys)})")
CELLS = [0, 1, 2, 3, 4]
cells = rad[np.isin(keys, CELLS)]
check(cells.std() / cells.mean() < 0.10,
      f"five cells spanning 9x in amplitude share one radius (CoV {cells.std()/cells.mean():.3f})")
sl, r2 = mvd.constancy(rad[np.isin(keys, CELLS)], en[np.isin(keys, CELLS)])
check(abs(sl) < 0.15 and r2 < 0.5,
      f"radius is independent of template energy (slope {sl:+.3f}, R2 {r2:.2f})")

vf, qf = mvd.dispersion_verdict(float(rad[keys == 10][0]), base)
vc, qc = mvd.dispersion_verdict(float(rad[keys == 11][0]), base)
check(vf.startswith("under"), f"a compact sub-region reads as a fragment ({qf:.2f}x)")
check(vc.startswith("over"), f"two pooled cells read as contaminated ({qc:.2f}x)")
v1, q1 = mvd.dispersion_verdict(float(rad[keys == 1][0]), base)
check(v1 == "one cell", f"an actual cell reads as one cell ({q1:.2f}x)")

# negative control: dispersion must NOT separate two cells that differ only in
# LOCATION.  If it did, it would be a disguised centroid test and would break on
# exactly the co-located pairs it is meant to survive.
far = Fs[0] + 500.0
r_far, _ = mvd.feature_radius(np.concatenate([Fs[0], far]),
                              np.concatenate([np.zeros(900), np.ones(900)]), 1)
r_near, _ = mvd.feature_radius(Fs[0], np.zeros(900), 0)
check(abs(r_far / r_near - 1.0) < 0.10,
      "negative control: translating a cell does not change its radius")

check(np.isnan(mvd.feature_radius(F, I, 999)[0]),
      "an absent cluster gives NaN rather than a number")
check(mvd.dispersion_verdict(np.nan, base)[0] == "unknown",
      "a NaN radius is reported unknown, not classified")

# isi_share: a cell whose features genuinely depend on ISI must show it, and one
# that does not must come back near zero.
tt = np.cumsum(rngd.uniform(0.003, 0.4, 4000))
isi_ms = np.concatenate([[1e9], np.diff(tt) * 1e3])
Xd = rngd.normal(0, 20, (4000, D))
Xd[:, 0] += np.where(isi_ms < 16, 300.0, 0.0)
sh, rtot, risi = mvd.isi_share(Xd, tt)
check(sh > 0.05, f"an imposed ISI dependence is detected ({100*sh:.1f}% of variance)")
Xn = rngd.normal(0, 20, (4000, D))
shn, _, _ = mvd.isi_share(Xn, tt)
check(shn < 0.02,
      f"negative control: no ISI dependence gives ~0 ({100*shn:.2f}%) — the bin "
      "structure alone does not manufacture a share")

print(f"\n{ran - fails}/{ran} checks passed")
sys.exit(1 if fails else 0)
