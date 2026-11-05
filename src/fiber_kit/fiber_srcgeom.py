#!/usr/bin/env python3
"""fiber_srcgeom -- a physically constrained source model of a unit's footprint, and the
"one cell at one place" test for a merge.

Model
-----
A cell is a chain of M point current sources on one straight axis through the soma, at axial offsets
OFFSETS * scale, in a homogeneous medium:

    V_c(t) = sum_m I_m(t) / r_cm            (1 / 4 pi sigma absorbed in I)
    sum_m I_m(t) = 0                         (charge conservation, at every t)

Geometry theta = (x0, y0, z0 >= 0, polar, azimuth, scale) in probe micrometres (site 0 at the origin).
Given theta the currents are linear, so the best fit of a RAW template X (nsamp x nchan) is its
projection onto the channel subspace span(K B), B a basis orthogonal to the all-ones current, and the
misfit is the residual energy / template energy.  Only the geometry is searched (Nelder-Mead from a
coarse grid), which makes a fit ~0.2-0.5 s.

Why this model.  Recorded octrode templates have spatial rank ~3 at the noise level (g6 0316: residual
energy after rank 3 = 0.0020, split-half noise 0.0023), so a 6-source chain (5 free currents) reaches the
noise (median misfit / noise 1.05) while still being forced to come from ONE place and orientation.
The reconstruction library (fiber-morpho) stays 60x the noise; this is what replaced it.

Merge test (`shared_geometry_z`)
--------------------------------
Fit A and B separately, then jointly with ONE geometry and their own currents.  The cost of sharing,
in units of the two templates' noise, is

    z = (misfit_joint - misfit_A - misfit_B) / (noise_A + noise_B)

Fragments of one cell share a geometry (z ~ 0..few); two cells at different places do not.  Currents
are free per fragment, so amplitude and shape changes of one cell (bursts, adaptation, drift of the
current mix) cost nothing.  On g6 0316 the z separates same-cell from different-cell candidate pairs
at AUC 0.78-0.94 (four sorts) and adds to roll-cosine (held-out-sort AUC 0.878 -> 0.914).
As a veto inside fiber-xcorr-merge it changed nothing (cos 0.98: no vetoes; 0.95 / 0.92: 5 vetoes, ARI
-0.002 / -0.005), because the different cells a cosine merge looks at mostly share a place.

RESOLUTION IS COARSE.  Five free currents span a 5-dim subspace of an 8-channel octrode, so one
geometry absorbs nearby places: on a synthetic staggered octrode (test_srcgeom) a source moved 10 / 20 /
40 / 80 um in depth gives z 2.4 / 0.9 / 2.2 / 5.1, and 20 / 40 um laterally 0.1 / 0.3 (not resolved).
It separates cells tens of micrometres apart along the shank, not neighbours.

Positions are NOT the useful output: the soma of a 6-source chain is less constrained than fiber-cpos's
monopole (feeding it to fiber-link / fiber-intrachunk's position gates did not help).  The shared
subspace is.
"""
import numpy as np
from scipy.optimize import minimize

OFFSETS = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 4.0])    # basal ... soma ... apical, x scale
RMIN = 5.0                                              # um; no site closer to a source than this


def _ones_basis(m):
    return np.linalg.qr(np.c_[np.ones(m), np.eye(m)[:, :-1]])[0][:, 1:]


def sites3(xy):
    """(nchan, 3) site coordinates with site 0 at the origin (z = 0 is the shank plane)."""
    xy = np.asarray(xy, float)
    return np.c_[xy - xy[0], np.zeros(len(xy))]


def _axis(polar, az):
    return np.array([np.sin(polar) * np.cos(az), np.cos(polar), np.sin(polar) * np.sin(az)])


def source_positions(th, offsets=OFFSETS):
    x0, y0, z0, pol, az, sc = th
    return np.array([x0, y0, abs(z0)])[None] + (np.asarray(offsets) * sc)[:, None] * _axis(pol, az)[None]


def transfer(th, S, offsets=OFFSETS):
    P = source_positions(th, offsets)
    r = np.sqrt(((S[:, None, :] - P[None]) ** 2).sum(2))
    return 1.0 / np.maximum(r, RMIN)


def subspace(th, S, offsets=OFFSETS):
    Q, _ = np.linalg.qr(transfer(th, S, offsets) @ _ones_basis(len(offsets)))
    return Q


def misfit(th, Xs, S, offsets=OFFSETS):
    """Sum over templates of residual energy / energy outside span(K B)."""
    Q = subspace(th, S, offsets)
    return sum(float(((X - (X @ Q) @ Q.T) ** 2).sum() / max((X ** 2).sum(), 1e-30)) for X in Xs)


def _starts(S):
    ymin, ymax = S[:, 1].min(), S[:, 1].max(); xm = S[:, 0].mean()
    return [np.array([xm, y0, z0, pol, 0.0, sc])
            for y0 in np.linspace(ymin - 40, ymax + 40, 9) for z0 in (10.0, 25.0, 50.0)
            for pol in (0.0, np.pi, np.pi / 2) for sc in (20.0, 50.0)]


def fit(X, S, starts=None, n_keep=6, offsets=OFFSETS):
    """Geometry of one template (or of several sharing one geometry).  Returns (theta, misfit)."""
    Xs = list(X) if isinstance(X, (list, tuple)) else [X]
    Xs = [np.asarray(x, float) for x in Xs]

    def f(th):
        return misfit(th, Xs, S, offsets)
    starts = _starts(S) if starts is None else [np.asarray(s, float) for s in starts]
    vals = np.array([f(s) for s in starts])
    best = None
    for i in np.argsort(vals)[:n_keep]:
        r = minimize(f, starts[i], method="Nelder-Mead", options=dict(maxiter=3000, xatol=0.05, fatol=1e-7))
        if best is None or r.fun < best.fun:
            best = r
    th = best.x.copy(); th[2] = abs(th[2])
    return th, float(best.fun)


def template_noise(T, SD, n):
    """Noise energy share of a mean template from its per-sample SD over n spikes."""
    return float((np.asarray(SD, float) ** 2).sum() / max(n, 1) / max((np.asarray(T, float) ** 2).sum(), 1e-30))


def shared_geometry_z(TA, TB, S, nA, nB, fitA=None, fitB=None, offsets=OFFSETS):
    """Cost of explaining two raw templates by ONE source geometry, in units of their noise.
    nA / nB are template noise energy shares (template_noise); fitA / fitB optional cached (theta, misfit)."""
    fA = fitA or fit(TA, S, offsets=offsets)
    fB = fitB or fit(TB, S, offsets=offsets)
    _, mj = fit([TA, TB], S, starts=[fA[0], fB[0]], n_keep=2, offsets=offsets)
    return float((mj - fA[1] - fB[1]) / max(nA + nB, 1e-12)), fA, fB
