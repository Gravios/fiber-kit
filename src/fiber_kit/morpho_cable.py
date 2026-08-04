#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════════════════════
#  morpho_cable.py — multi-compartment cable solver (Hines) + spike simulation.
#
#  Produces the one thing the extracellular model needs: the per-compartment
#  transmembrane current I_m(t).  Everything else (V_m traces, bAP attenuation)
#  falls out of the same integration.
#
#  Two design points worth stating, because both were chosen against the more
#  obvious alternative:
#
#  1. I_m is DEFINED as the axial current flowing into each compartment, i.e.
#     I_m = C dV/dt + I_ion - I_stim, evaluated on the solved V(t+dt).  Computed
#     that way it sums to zero across the cell to machine precision at every
#     step, by construction rather than by luck.  Charge conservation is the
#     only cheap global check on a compartmental model -- an EAP built from
#     currents that do not sum to zero has a spurious monopole term that decays
#     as 1/r instead of 1/r^2 and will quietly dominate the far field.  The
#     stimulus is counted as a membrane current (it enters from the bath, like
#     a synaptic current), so the sum stays exactly zero during the stimulus.
#
#  2. The Hines elimination is vectorized by TREE LEVEL, not run as a Python
#     loop over compartments.  A detailed reconstruction is ~1-3k compartments
#     but only ~60-120 levels deep, so level scheduling turns ~3M scalar steps
#     into ~200k array ops and makes a full spike simulation seconds instead of
#     minutes -- without approximating anything: the level order is a valid
#     elimination order for the exact same matrix.
#
#  Units are NEURON's: mV, ms, nA, nF, uS, um, S/cm^2, ohm-cm.
# ════════════════════════════════════════════════════════════════════════════
import numpy as np

try:
    from . import morpho_chan as mch
except ImportError:
    import morpho_chan as mch
try:
    from . import morpho_geom as mg
except ImportError:
    import morpho_geom as mg


# ── biophysical specification ───────────────────────────────────────────────
class Biophys:
    """Passive + active parameters.

    Defaults are the published CA1 operating point of Migliore, Ferrante &
    Ascoli (2005): Rm 28 kohm-cm^2, Ra 150 (50 in axon), Cm 1 uF/cm^2,
    gNa 0.025 S/cm^2 (x5 in axon), gKdr 0.01, A-type 0.03 at the soma rising as
    (1 + d/100 um) out to ka_dist_max, celsius 35.

    ka_scale is exposed as a single knob because it is the experimental
    dichotomy: CA1 cells with a strong dendritic A-current attenuate the bAP
    steeply, cells with less of it propagate nearly unattenuated (Golding, Kath
    & Spruston 2001, J Neurophysiol 86:2998).  Sweeping it is how this model
    represents a population of cells rather than one cell.
    """

    def __init__(self, Rm=28000.0, Ra=150.0, Ra_axon=50.0, cm=1.0, v_rest=-65.0,
                 gna=0.025, axon_na_mult=5.0, gkdr=0.01, gka=0.03, ka_scale=1.0,
                 ka_slope_per_100um=1.0, ka_dist_max=350.0, ka_prox_lim=100.0,
                 ena=55.0, ek=-90.0, celsius=35.0, soma_na_mult=1.0):
        self.__dict__.update(locals()); del self.__dict__["self"]

    def densities(self, cmp_):
        """Per-compartment conductance densities (S/cm^2) and axial resistivity."""
        n = len(cmp_)
        d = cmp_.pathdist
        gna = np.full(n, self.gna)
        gna[cmp_.type == mg.AXON] *= self.axon_na_mult
        gna[cmp_.type == mg.SOMA] *= self.soma_na_mult
        gkdr = np.full(n, self.gkdr)
        grad = 1.0 + self.ka_slope_per_100um * np.minimum(d, self.ka_dist_max) / 100.0
        gka = self.gka * self.ka_scale * grad
        prox = d <= self.ka_prox_lim
        gka_p = np.where(prox, gka, 0.0)
        gka_d = np.where(prox, 0.0, gka)
        gka_p[cmp_.type == mg.AXON] = self.gka * self.ka_scale
        gka_d[cmp_.type == mg.AXON] = 0.0
        Ra = np.where(cmp_.type == mg.AXON, self.Ra_axon, self.Ra)
        return dict(na=gna, kdr=gkdr, ka_prox=gka_p, ka_dist=gka_d, Ra=Ra)


# ── tree scheduling ─────────────────────────────────────────────────────────
def _levels(parent):
    """(elimination levels, back-substitution levels).

    Elimination height: leaves 0, a node one more than its deepest child, so all
    children of a node are eliminated strictly before it.  Back-substitution
    depth: root 0, child one more than parent.  Within either level no node is
    an ancestor of another, so each level is safe to apply as one array op.
    """
    n = len(parent)
    height = np.zeros(n, int)
    for i in range(n - 1, 0, -1):
        p = parent[i]
        if p >= 0:
            height[p] = max(height[p], height[i] + 1)
    depth = np.zeros(n, int)
    for i in range(1, n):
        p = parent[i]
        depth[i] = depth[p] + 1 if p >= 0 else 0
    elim = [np.flatnonzero((height == h) & (parent >= 0)) for h in range(height.max() + 1)]
    elim = [e for e in elim if e.size]
    back = [np.flatnonzero(depth == k) for k in range(depth.max() + 1)]
    back = [b for b in back if b.size]
    return elim, back


def axial_conductance(cmp_, Ra):
    """uS between each compartment and its parent (two half-cylinders in series).

    Ra is per-compartment (ohm-cm); the half-resistance of each is used, which
    is the standard NEURON coupling and the reason an axon with Ra=50 loads the
    soma differently from one with Ra=150.
    """
    L_cm = cmp_.L * 1e-4
    a_cm2 = np.pi * (cmp_.diam * 1e-4) ** 2 / 4.0
    r_half = np.where(a_cm2 > 0, Ra * (L_cm / 2.0) / np.maximum(a_cm2, 1e-30), np.inf)
    p = cmp_.parent
    g = np.zeros(len(cmp_))
    ok = p >= 0
    g[ok] = 1.0 / (r_half[ok] + r_half[p[ok]]) * 1e6      # S -> uS
    return g


def hines_solve(diag, rhs, gax, parent, elim, back):
    """Solve the symmetric tree system in place-free form; returns V (mV).

    diag: (N,) diagonal (uS), rhs: (N,) (nA), gax: (N,) coupling to parent (uS,
    gax[root] unused).  The off-diagonals are -gax on both sides, so a single
    factor per node suffices.  np.add.at is required, not fancy-index +=,
    because several children scatter into one parent within a level.
    """
    d = diag.copy(); b = rhs.copy()
    for lv in elim:
        p = parent[lv]
        f = gax[lv] / d[lv]
        np.add.at(d, p, -gax[lv] * f)
        np.add.at(b, p, f * b[lv])
    v = np.empty_like(b)
    root = back[0]
    v[root] = b[root] / d[root]
    for lv in back[1:]:
        v[lv] = (b[lv] + gax[lv] * v[parent[lv]]) / d[lv]
    return v


# ── simulation ──────────────────────────────────────────────────────────────
class Cell:
    """A compartmentalized morphology with channels, ready to integrate."""

    def __init__(self, cmp_, bio=None):
        self.c = cmp_
        self.bio = bio or Biophys()
        n = len(cmp_)
        dens = self.bio.densities(cmp_)
        area_cm2 = cmp_.area * 1e-8
        self.area_cm2 = area_cm2
        self.cap = self.bio.cm * area_cm2 * 1e3                 # nF
        self.gpas = (1.0 / self.bio.Rm) * area_cm2 * 1e6        # uS
        self.gax = axial_conductance(cmp_, dens["Ra"])
        self.parent = cmp_.parent
        self.elim, self.back = _levels(cmp_.parent)

        cel = self.bio.celsius
        self.chans = [
            (mch.Na(cel), dens["na"] * area_cm2 * 1e6, self.bio.ena),
            (mch.Kdr(cel), dens["kdr"] * area_cm2 * 1e6, self.bio.ek),
            (mch.KA("prox", cel), dens["ka_prox"] * area_cm2 * 1e6, self.bio.ek),
            (mch.KA("dist", cel), dens["ka_dist"] * area_cm2 * 1e6, self.bio.ek),
        ]
        self.chans = [c for c in self.chans if np.any(c[1] > 0)]
        self.n = n
        self.reset()

    def reset(self):
        v = np.full(self.n, self.bio.v_rest)
        self.v = v
        self.state = [[np.asarray(inf, float) * np.ones(self.n) for inf, _ in ch.rates(v)]
                      for ch, _, _ in self.chans]
        # e_pas chosen so the resting state is a true steady state (the published
        # models do the same at init): any residual ionic current at v_rest is
        # absorbed into the leak reversal rather than left to drift the cell.
        ion = np.zeros(self.n)
        for (ch, gbar, erev), st in zip(self.chans, self.state):
            ion += gbar * ch.g(st) * (v - erev)
        self.e_pas = v + ion / np.maximum(self.gpas, 1e-30)

    def _advance_gates(self, v, dt):
        for (ch, _, _), st in zip(self.chans, self.state):
            for k, (inf, tau) in enumerate(ch.rates(v)):
                st[k] = inf + (st[k] - inf) * np.exp(-dt / tau)

    def step(self, dt, istim=None, gsyn=None):
        """One backward-Euler step.  Returns (v_new, I_m) with I_m in nA.

        gsyn: [(g (ncomp,) uS, erev mV), ...].  Synaptic conductance is folded
        into the same implicit diagonal as the intrinsic channels rather than
        applied as an explicit current -- an explicit synapse of the size the
        perforant path delivers (thousands of synapses) is stiff enough to
        oscillate at dt=0.01 ms, which would look like dendritic ringing.

        The resulting synaptic current is part of I_m by construction: it enters
        through the membrane, so unlike the electrode stimulus it must NOT be
        subtracted out, or the extracellular field would lose the very
        dendritic sink this module exists to model."""
        v = self.v
        self._advance_gates(v, dt)
        gtot = self.gpas.copy()
        gerev = self.gpas * self.e_pas
        for (ch, gbar, erev), st in zip(self.chans, self.state):
            g = gbar * ch.g(st)
            gtot += g; gerev += g * erev
        if gsyn:
            for g, erev in gsyn:
                gtot = gtot + g; gerev = gerev + g * erev
        cdt = self.cap / dt
        diag = cdt + gtot
        rhs = cdt * v + gerev
        if istim is not None:
            rhs = rhs + istim
        np.add.at(diag, np.arange(self.n)[self.parent >= 0], self.gax[self.parent >= 0])
        np.add.at(diag, self.parent[self.parent >= 0], self.gax[self.parent >= 0])
        vn = hines_solve(diag, rhs, self.gax, self.parent, self.elim, self.back)
        im = self.cap * (vn - v) / dt + (gtot * vn - gerev)
        if istim is not None:
            im = im - istim
        self.v = vn
        return vn, im


def simulate(cell, dt=0.01, t_stop=15.0, stim_amp=2.0, stim_start=2.0, stim_dur=1.0,
             stim_comp=0, record_v=True, drive=None):
    """Integrate one spike.  Returns dict with t, im (nt, N), v (nt, N) or None.

    stim_amp is nA delivered to stim_comp (compartment 0 = soma by convention of
    morpho_geom's ordering).  A 1 ms suprathreshold pulse is used rather than a
    long current step so the spike is not riding on a depolarizing plateau that
    would contaminate the extracellular waveform's late phase.

    drive: an optional morpho_input.Drive supplying synaptic conductance, whose
    step count must match nt.
    """
    if drive is not None and drive.nt < int(round(t_stop / dt)):
        raise ValueError(f"drive covers {drive.nt} steps, simulation needs "
                         f"{int(round(t_stop / dt))}")
    nt = int(round(t_stop / dt))
    N = cell.n
    im = np.zeros((nt, N), np.float32)
    vv = np.zeros((nt, N), np.float32) if record_v else None
    istim = np.zeros(N)
    for k in range(nt):
        t = k * dt
        istim[:] = 0.0
        if stim_start <= t < stim_start + stim_dur:
            istim[stim_comp] = stim_amp
        v, i = cell.step(dt, istim, drive.at(k) if drive is not None else None)
        im[k] = i
        if record_v:
            vv[k] = v
    return dict(t=np.arange(nt) * dt, im=im, v=vv, dt=dt)


def find_spike(v, dt, thresh=-20.0, comp=0):
    """Index of the somatic spike peak, or None."""
    tr = v[:, comp]
    above = np.flatnonzero(tr > thresh)
    return int(tr.argmax()) if above.size else None


def bap_profile(res, cmp_, thresh=-20.0):
    """Back-propagation summary per compartment.

    amp: peak depolarization from rest (mV); tpeak: time of that peak (ms);
    dist: path distance from soma (um).  Amplitude is measured from the
    pre-stimulus baseline of each compartment, not from a global rest, so a
    dendrite sitting at a slightly different steady-state potential is not
    credited with extra bAP amplitude.
    """
    v = res["v"]
    if v is None:
        raise ValueError("simulate(record_v=True) is required for bap_profile")
    base = v[: max(1, int(1.0 / res["dt"]))].mean(0)
    k = int(v[:, 0].argmax())
    w = slice(max(0, k - int(2.0 / res["dt"])), k + int(6.0 / res["dt"]))
    seg = v[w]
    amp = seg.max(0) - base
    tpk = (np.argmax(seg, axis=0) + (w.start or 0)) * res["dt"]
    return dict(dist=cmp_.pathdist, amp=amp, tpeak=tpk - k * res["dt"],
                soma_peak=float(v[:, 0].max()), spike_index=k)
