#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════════════════════
#  morpho_archetype.py — parametric cells, for sweeping morphological factors.
#
#  Reconstructions answer "how different are these particular cells"; they do
#  not answer "which morphological property drives the difference", because the
#  properties covary and n is small.  These archetypes exist for the second
#  question: one factor moves at a time, everything else held fixed.
#
#  They are NOT substitutes for reconstructions and should never be reported as
#  if they were cells.  Their dimensions are set from published rat hippocampal
#  measurements (see DEFAULTS below), but a smooth tapering cylinder tree has
#  none of the branch-point clutter that makes a real dendrite's extracellular
#  signature what it is.
# ════════════════════════════════════════════════════════════════════════════
import numpy as np

try:
    from . import morpho_geom as mg
except ImportError:
    import morpho_geom as mg


def _cable(name, type_, start, direction, length, d0, d1, nstep=6):
    """One tapering cable as a Section, from start along a unit direction."""
    s = mg.Section(name, type_)
    u = np.asarray(direction, float)
    u = u / max(np.linalg.norm(u), 1e-12)
    for k in range(nstep + 1):
        f = k / nstep
        p = np.asarray(start, float) + u * (length * f)
        s.points.append([p[0], p[1], p[2], d0 + (d1 - d0) * f])
    return s


def _soma(diam, length=None):
    L = float(length if length is not None else diam)
    s = mg.Section("soma[0]", mg.SOMA)
    s.points = [[0.0, -L / 2, 0.0, diam], [0.0, L / 2, 0.0, diam]]
    return s


def _attach_axon(secs, soma_i=0, ais_len=25.0, ais_diam=1.2, axon_len=200.0,
                 axon_diam=0.8, direction=(0.0, -1.0, 0.0)):
    """Axon initial segment + axon proper, hanging off the soma's 0-end.

    Included by default because the AIS is where the spike starts and it is a
    dense current sink a few tens of microns from the soma; leaving it out
    changes the early phase of the extracellular waveform, not just its
    amplitude.
    """
    start = np.asarray(secs[soma_i].points[0][:3], float)
    ais = _cable("axon[0]", mg.AXON, start, direction, ais_len, ais_diam, ais_diam, 3)
    ax = _cable("axon[1]", mg.AXON, np.asarray(ais.points[-1][:3]), direction,
                axon_len, axon_diam, axon_diam, 5)
    ais.parent = soma_i; ais.parent_x = 0.0
    ax.parent = len(secs); ax.parent_x = 1.0
    secs.extend([ais, ax])
    return secs


def pyramidal(soma_diam=20.0, apical_len=450.0, apical_diam=3.5, tuft_len=150.0,
              n_oblique=6, oblique_len=120.0, n_basal=6, basal_len=180.0,
              basal_diam=1.5, spread_deg=45.0, axon=True):
    """Pyramidal archetype: one tapering apical trunk with obliques and a tuft,
    plus a basal skirt.  Defaults are mid-range rat CA1 (soma ~20 um, trunk to
    ~450 um, tuft in SLM)."""
    secs = [_soma(soma_diam)]
    trunk = _cable("apical[0]", mg.APICAL, [0, soma_diam / 2, 0], [0, 1, 0],
                   apical_len, apical_diam, apical_diam * 0.35, 10)
    trunk.parent = 0; trunk.parent_x = 1.0
    secs.append(trunk)
    trunk_pts = np.asarray(trunk.points, float)
    for k in range(n_oblique):
        f = 0.25 + 0.6 * (k / max(n_oblique - 1, 1))
        base = trunk_pts[int(f * (len(trunk_pts) - 1))]
        ang = np.radians(spread_deg * (1 if k % 2 else -1))
        d = [np.sin(ang), np.cos(ang) * 0.35, 0.35 * (1 if k % 3 else -1)]
        s = _cable(f"apical[{k + 1}]", mg.APICAL, base[:3], d, oblique_len,
                   base[3] * 0.45, base[3] * 0.2, 4)
        s.parent = 1; s.parent_x = float(f); secs.append(s)
    tip = trunk_pts[-1]
    for k in range(2):
        ang = np.radians(35.0 * (1 if k else -1))
        s = _cable(f"apical[{n_oblique + 1 + k}]", mg.APICAL, tip[:3],
                   [np.sin(ang), np.cos(ang), 0.0], tuft_len, tip[3], tip[3] * 0.5, 5)
        s.parent = 1; s.parent_x = 1.0; secs.append(s)
    for k in range(n_basal):
        a = 2 * np.pi * k / n_basal
        d = [np.cos(a) * 0.8, -0.8, np.sin(a) * 0.8]
        s = _cable(f"basal[{k}]", mg.BASAL, [0, -soma_diam / 2, 0], d, basal_len,
                   basal_diam, basal_diam * 0.4, 5)
        s.parent = 0; s.parent_x = 0.0; secs.append(s)
    return _attach_axon(secs) if axon else secs


def multipolar(soma_diam=15.0, n_dend=6, dend_len=350.0, dend_diam=2.0,
               elongation=1.0, axon=True):
    """Aspiny/multipolar interneuron archetype: dendrites radiating from the
    soma.  elongation stretches the field along the laminar axis (1 = radial,
    >1 = bitufted/vertical, <1 = horizontal)."""
    secs = [_soma(soma_diam)]
    for k in range(n_dend):
        a = 2 * np.pi * k / n_dend + 0.3
        d = [np.cos(a), np.sin(a) * elongation, 0.4 * np.cos(2 * a)]
        s = _cable(f"dend[{k}]", mg.BASAL if np.sin(a) < 0 else mg.APICAL,
                   [0, 0, 0], d, dend_len, dend_diam, dend_diam * 0.35, 6)
        s.parent = 0; s.parent_x = 0.5; secs.append(s)
    return _attach_axon(secs) if axon else secs


def bipolar(soma_diam=12.0, dend_len=280.0, dend_diam=2.5, tilt_deg=20.0, axon=True):
    """Horizontally-oriented bipolar archetype (OLM-like): two opposed dendrites
    nearly parallel to the layer, so the cell's current dipole lies across the
    shank rather than along it."""
    secs = [_soma(soma_diam)]
    for k, sgn in enumerate((1, -1)):
        a = np.radians(tilt_deg) * sgn
        s = _cable(f"dend[{k}]", mg.BASAL, [0, 0, 0], [sgn * np.cos(a), np.sin(a), 0],
                   dend_len, dend_diam, dend_diam * 0.4, 6)
        s.parent = 0; s.parent_x = 0.5; secs.append(s)
    return _attach_axon(secs, direction=(0.0, 1.0, 0.0)) if axon else secs


def granule(soma_diam=11.0, n_dend=3, dend_len=300.0, dend_diam=1.8,
            spread_deg=30.0, axon=True):
    """Dentate granule archetype: a narrow cone of apical dendrites, no basals."""
    secs = [_soma(soma_diam)]
    for k in range(n_dend):
        a = np.radians(spread_deg) * (k - (n_dend - 1) / 2)
        s = _cable(f"apical[{k}]", mg.APICAL, [0, soma_diam / 2, 0],
                   [np.sin(a), np.cos(a), 0.0], dend_len, dend_diam, dend_diam * 0.35, 6)
        s.parent = 0; s.parent_x = 1.0; secs.append(s)
    return _attach_axon(secs) if axon else secs


FAMILY = dict(pyramidal=pyramidal, multipolar=multipolar, bipolar=bipolar,
              granule=granule)


def build(kind="pyramidal", d_lambda=0.1, **kw):
    """Archetype -> oriented Compartments."""
    if kind not in FAMILY:
        raise ValueError(f"unknown archetype {kind!r}; have {sorted(FAMILY)}")
    c = mg.compartmentalize(FAMILY[kind](**kw), d_lambda=d_lambda)
    c.name = kind
    return mg.orient(c)
