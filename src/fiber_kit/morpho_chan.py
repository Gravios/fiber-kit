#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════════════════════
#  morpho_chan.py — voltage-gated channel kinetics, vectorized over compartments.
#
#  These are TRANSCRIPTIONS of the published NEURON mod files of the Migliore
#  CA1 model family (na3/nax, kdr, kap, kad; Migliore et al. 1999, J Comput
#  Neurosci 7:5-15; Migliore, Ferrante & Ascoli 2005, J Neurophysiol 94:4145),
#  not re-fits.  That matters for the question this module exists to answer:
#  the dendritic A-type gradient in kad is the dominant determinant of how far
#  a back-propagating spike gets, so inventing plausible-looking kinetics would
#  put the answer's main free parameter under our own control rather than under
#  the experimental literature's.
#
#  Every rate is evaluated as a plain array expression over all compartments at
#  once; per-compartment variation lives entirely in the conductance densities,
#  which is the only place the published models vary them too.
#
#  Units follow NEURON: mV, ms, S/cm^2, degC.
# ════════════════════════════════════════════════════════════════════════════
import numpy as np

FARADAY, GASCON = 9.648e4, 8.315          # as written in the mod files


def _zfac(celsius):
    """1e-3 * F / (R * T) -- the exponent prefactor shared by the Migliore
    Boltzmann rates (per mV per unit valence)."""
    return 1e-3 * FARADAY / (GASCON * (273.16 + celsius))


def _trap0(v, th, a, q):
    """a*(v-th)/(1-exp(-(v-th)/q)) with the removable singularity handled."""
    x = v - th
    small = np.abs(x) < 1e-6
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        out = a * x / (1.0 - np.exp(-x / q))
    return np.where(small, a * q, out)


class Na:
    """na3 / nax -- fast sodium, m^3 h s.

    All three gates, including the slow-inactivation state s.  An earlier
    version of this module dropped s on the grounds that sinf == 1 identically
    at the published ar = 1, which is true and was still the wrong call: s is
    the mechanism that makes Na availability USE-dependent over hundreds of ms,
    and use-dependent availability is exactly what sets the spike-amplitude
    decrement within a burst.  Removing a no-op at one operating point removed
    the ability to leave that operating point.  ar is now a parameter: ar = 1
    reproduces the published behaviour exactly (s stays at 1 and the extra state
    costs only arithmetic), ar < 1 lets a fraction 1 - ar of the channels
    slow-inactivate when depolarized.

    The two inactivation gates cover different timescales and both matter for
    within-neuron waveform variance: h recovers with a floor of 0.5 ms, so it
    sets the amplitude of the second spike of a doublet; s has taus >= 10 ms and
    growing, so it sets the decrement across a whole complex-spike burst and the
    recovery between bursts.
    """
    name = "na"
    tha, qa, Ra, Rb = -30.0, 7.2, 0.4, 0.124
    thi1, thi2, qd, qg, Rd, Rg = -45.0, -45.0, 1.5, 1.5, 0.03, 0.01
    thinf, qinf, mmin, hmin, q10 = -50.0, 4.0, 0.02, 0.5, 2.0
    vhalfs, a0s, zetas, gms, smax, vvh, vvs = -60.0, 0.0003, 12.0, 0.2, 10.0, -58.0, 2.0

    def __init__(self, celsius=35.0, sh=0.0, ar=1.0):
        self.qt = self.q10 ** ((celsius - 24.0) / 10.0); self.sh = sh
        self.ar = float(ar); self.zf = _zfac(celsius)

    def rates(self, v):
        sh = self.sh
        a = _trap0(v, self.tha + sh, self.Ra, self.qa)
        b = _trap0(-v, -self.tha - sh, self.Rb, self.qa)
        mtau = np.maximum(1.0 / (a + b) / self.qt, self.mmin)
        minf = a / (a + b)
        a = _trap0(v, self.thi1 + sh, self.Rd, self.qd)
        b = _trap0(-v, -self.thi2 - sh, self.Rg, self.qg)
        htau = np.maximum(1.0 / (a + b) / self.qt, self.hmin)
        hinf = 1.0 / (1.0 + np.exp((v - self.thinf - sh) / self.qinf))
        with np.errstate(over="ignore"):
            c = 1.0 / (1.0 + np.exp((v - self.vvh - sh) / self.vvs))
            alps = np.exp(self.zetas * self.zf * (v - self.vhalfs - sh))
            bets = np.exp(self.zetas * self.gms * self.zf * (v - self.vhalfs - sh))
        sinf = c + self.ar * (1.0 - c)
        taus = np.maximum(bets / (self.a0s * (1.0 + alps)), self.smax)
        return (minf, mtau), (hinf, htau), (sinf, taus)

    def g(self, st):
        m, h, s = st
        return m * m * m * h * s


class Kdr:
    """kdrca1 -- delayed rectifier, first order in n (NOT n^4)."""
    name = "kdr"
    vhalfn, a0n, zetan, gmn, nmax, q10 = 13.0, 0.02, -3.0, 0.7, 2.0, 1.0

    def __init__(self, celsius=35.0):
        self.qt = self.q10 ** ((celsius - 24.0) / 10.0); self.zf = _zfac(celsius)

    def rates(self, v):
        a = np.exp(self.zetan * self.zf * (v - self.vhalfn))
        b = np.exp(self.zetan * self.gmn * self.zf * (v - self.vhalfn))
        ninf = 1.0 / (1.0 + a)
        taun = np.maximum(b / (self.qt * self.a0n * (1.0 + a)), self.nmax)
        return ((ninf, taun),)

    def g(self, st):
        return st[0]


class KA:
    """kap / kad -- transient A-type, n*l.  variant='prox' or 'dist'.

    Proximal and distal differ only in activation half-point and slope; both are
    used in the same cell, kap within ~100 um of the soma and kad beyond, which
    is how the published model reproduces the measured A-current gradient
    (Hoffman et al. 1997, Nature 387:869).
    """
    name = "ka"
    vhalfl, zetal, gml, a0l, lmin, qtl = -56.0, 3.0, 1.0, 0.05, 2.0, 1.0
    pw, tq, qq, q10 = -1.0, -40.0, 5.0, 5.0
    _P = {"prox": dict(vhalfn=11.0, a0n=0.05, zetan=-1.5, gmn=0.55, nmin=0.1),
          "dist": dict(vhalfn=-1.0, a0n=0.10, zetan=-1.8, gmn=0.39, nmin=0.2)}

    def __init__(self, variant="prox", celsius=35.0):
        if variant not in self._P:
            raise ValueError(f"KA variant must be prox|dist, got {variant!r}")
        self.__dict__.update(self._P[variant])
        self.variant = variant; self.name = f"ka_{variant}"
        self.qt = self.q10 ** ((celsius - 24.0) / 10.0); self.zf = _zfac(celsius)

    def rates(self, v):
        zeta = self.zetan + self.pw / (1.0 + np.exp((v - self.tq) / self.qq))
        a = np.exp(zeta * self.zf * (v - self.vhalfn))
        b = np.exp(zeta * self.gmn * self.zf * (v - self.vhalfn))
        ninf = 1.0 / (1.0 + a)
        taun = np.maximum(b / (self.qt * self.a0n * (1.0 + a)), self.nmin)
        al = np.exp(self.zetal * self.zf * (v - self.vhalfl))
        linf = 1.0 / (1.0 + al)
        taul = np.maximum(0.26 * (v + 50.0) / self.qtl, self.lmin / self.qtl)
        return (ninf, taun), (linf, taul)

    def g(self, st):
        n, l = st
        return n * l
