#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════════════════════
#  morpho_input.py — where hippocampal afferents land, and what that does to
#  the recorded waveform.
#
#  The connection to spike sorting is indirect but real.  Input topology does
#  not change a cell's spike TEMPLATE directly -- the extracellular waveform is
#  dominated by the somato-axonal sink, which the synapses do not touch.  What
#  it changes is the DENDRITIC contribution: a back-propagating spike into a
#  depolarized dendrite is larger and reaches further than one into a resting
#  dendrite, and the dendritic current is what the distal channels of a probe
#  actually see.  So pathway-specific dendritic depolarization produces
#  within-unit waveform variance that is
#
#     (a) state-dependent, not drift -- it tracks what the animal's inputs are
#         doing on a theta/behaviour timescale, not the electrode's position,
#     (b) concentrated on the channels FURTHEST from the peak channel, where
#         drift's signature is a coordinated shift of the whole footprint,
#
#  which is exactly the kind of nuisance direction a drift-correction method
#  can mistake for drift.  Quantifying it is the point of this module.
#
#  The topology itself is not invented: it comes from the Bezaire et al. (2016)
#  full-scale CA1 model's own connectivity and synapse datasets, distilled into
#  hippocampal_pathways.tsv (see that file's header for provenance and for how
#  to regenerate it from a checkout of the model).
# ════════════════════════════════════════════════════════════════════════════
import os
import numpy as np

try:
    from . import morpho_geom as mg
except ImportError:
    import morpho_geom as mg

TABLE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "hippocampal_pathways.tsv")

# NEURON SectionList name -> the SWC types it draws from.  "dendrite_list" is
# ALL dendrite in the source model (basal included), which is correct: Schaffer
# collaterals synapse in stratum oriens as well as radiatum.  The path-distance
# window, not the list name, is what does the laminar work.
REGION_TYPES = {
    "soma_list": (mg.SOMA,),
    "axon_list": (mg.AXON,),
    "basal_list": (mg.BASAL,),
    "apical_list": (mg.APICAL,),
    "dendrite_list": (mg.BASAL, mg.APICAL),
    "all": (mg.SOMA, mg.AXON, mg.BASAL, mg.APICAL),
}

# CA1 layer thicknesses (um), from the source model's LayerHeights default
# "4;100;50;200;100" -- four layers, basal to molecular.  Boundaries are given
# relative to the SOMA CENTRE, since that is the origin morpho_geom.orient()
# establishes and the only landmark a simulated and a recorded cell share.
LAYERS = [("so", -125.0, -25.0),      # stratum oriens
          ("sp", -25.0, 25.0),        # stratum pyramidale
          ("sr", 25.0, 225.0),        # stratum radiatum
          ("slm", 225.0, 325.0)]      # stratum lacunosum-moleculare


class Pathway:
    """One afferent projection onto one postsynaptic cell type."""

    __slots__ = ("post", "pre", "region", "dlo", "dhi", "nconv", "nsyn", "ntotal",
                 "w", "tau1", "tau2", "erev", "src")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    @property
    def excitatory(self):
        return self.erev > -30.0

    @property
    def gmax_total(self):
        """Total peak conductance of the projection (uS) if every synapse fired
        together -- the natural scale for comparing pathway strength, since
        neither synapse count nor per-synapse weight means anything alone."""
        return self.ntotal * self.w

    def __repr__(self):
        return (f"<{self.pre}->{self.post} {self.region}[{self.dlo:.0f},{self.dhi:.0f}] "
                f"n={self.ntotal:.0f} g={self.gmax_total*1e3:.1f} nS "
                f"{'E' if self.excitatory else 'I'}>")


def load_table(path=TABLE, post=None):
    """Read hippocampal_pathways.tsv.  post filters to one cell type."""
    rows, hdr = [], None
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            f = line.split("\t")
            if hdr is None:
                hdr = f
                continue
            d = dict(zip(hdr, f))
            for k in ("dlo", "dhi", "nconv", "nsyn", "ntotal", "w", "tau1", "tau2", "erev"):
                d[k] = float(d[k])
            rows.append(Pathway(**d))
    if post is not None:
        rows = [p for p in rows if p.post == post]
    return rows


def parse_bezaire(datasets_dir, num="101", conn="430", syn="120", post=None):
    """Regenerate the pathway table directly from a checkout of the source model.

    Kept because the shipped TSV is a derived artifact: anyone who wants a
    different connectivity dataset (the model ships hundreds) must be able to
    produce the table rather than edit the derived file, or the provenance line
    in its header becomes a lie.
    """
    import re
    d = datasets_dir
    counts = {}
    for ln in open(os.path.join(d, f"cellnumbers_{num}.dat")).read().splitlines()[1:]:
        f = ln.split()
        if len(f) >= 3:
            counts[f[0]] = float(f[2])
    cn = {}
    for ln in open(os.path.join(d, f"conndata_{conn}.dat")).read().splitlines()[1:]:
        f = ln.split()
        if len(f) >= 5:
            cn[(f[1], f[0])] = (float(f[2]), float(f[3]), float(f[4]))
    ext = {"ca3cell", "eccell", "ca3ripcell", "msgabacell"}
    out = []
    for ln in open(os.path.join(d, f"syndata_{syn}.dat")).read().splitlines()[1:]:
        f = ln.split()
        if len(f) < 6:
            continue
        po, pr, region = f[0], f[1], f[3]
        lo, hi = -1.0, 1e4
        for op, v in re.findall(r"distance\(x\)([<>])(-?\d+)", ln):
            lo, hi = (float(v), hi) if op == ">" else (lo, float(v))
        tail = [float(x) for x in f[4:] if re.match(r"^-?\d+\.\d+$", x)]
        t1, t2, er = (tail + [0.0, 0.0, 0.0])[:3]
        w, nc, ns = cn.get((po, pr), (0.0, 0.0, 0.0))
        if nc * ns <= 0:
            continue
        out.append(Pathway(post=po, pre=pr, region=region, dlo=lo, dhi=hi, nconv=nc,
                           nsyn=ns, ntotal=nc * ns, w=w, tau1=t1, tau2=t2, erev=er,
                           src="extrinsic" if pr in ext else "local"))
    return [p for p in out if post is None or p.post == post]


# ── allocation onto a morphology ────────────────────────────────────────────
def eligible(pathway, cmp_):
    """Boolean mask of compartments a pathway's synapses may land on."""
    types = REGION_TYPES.get(pathway.region, REGION_TYPES["all"])
    m = np.isin(cmp_.type, types)
    return m & (cmp_.pathdist > pathway.dlo) & (cmp_.pathdist < pathway.dhi)


def allocate(pathway, cmp_):
    """Synapse counts per compartment, area-weighted within the eligible set.

    Area-weighted rather than uniform-per-compartment because the d_lambda rule
    makes compartments unequal in size: uniform allocation would put the same
    number of synapses on a 2 um soma slab as on a 40 um distal cable, which
    would silently make the synapse density a function of the discretization.
    Returns a float array -- fractional synapses are meaningful here, since the
    quantity of interest is conductance density, not individual release sites.
    """
    m = eligible(pathway, cmp_)
    n = np.zeros(len(cmp_))
    if not m.any():
        return n
    w = cmp_.area * m
    n[m] = pathway.ntotal * w[m] / w.sum()
    return n


def layer_of(y, layers=LAYERS):
    """Layer name per y coordinate (um, soma centre = 0); None outside."""
    y = np.asarray(y, float)
    out = np.full(y.shape, "", dtype=object)
    for name, lo, hi in layers:
        out[(y >= lo) & (y < hi)] = name
    return out


def laminar_profile(pathways, cmp_, layers=LAYERS):
    """Synapses (and total peak conductance) per layer per pathway.

    This is the model's actual claim about input topology: it takes a pathway
    defined by PATH distance in the source model and reports it in the LAMINAR
    coordinate a probe measures.  The two do not coincide -- a synapse 150 um
    along a basal dendrite is in oriens, not radiatum -- and the discrepancy is
    the honest uncertainty in mapping a cable-distance rule onto real geometry.
    """
    lay = layer_of(cmp_.mid[:, 1], layers)
    names = [n for n, _, _ in layers] + ["outside"]
    prof = {}
    for p in pathways:
        n = allocate(p, cmp_)
        row = {k: 0.0 for k in names}
        for k in names:
            m = (lay == k) if k != "outside" else (lay == "")
            row[k] = float(n[m].sum())
        prof[(p.pre, p.region)] = dict(counts=row, total=float(n.sum()),
                                       gmax_nS=float(n.sum() * p.w * 1e3),
                                       excitatory=p.excitatory, src=p.src)
    return prof


# ── synaptic drive ──────────────────────────────────────────────────────────
def _dual_exp(t, tau1, tau2):
    """Normalized two-exponential waveform, peak 1.0 (NEURON's Exp2Syn)."""
    if tau1 >= tau2:
        tau1 = tau2 * 0.9
    tp = (tau1 * tau2) / (tau2 - tau1) * np.log(tau2 / tau1)
    factor = 1.0 / (-np.exp(-tp / tau1) + np.exp(-tp / tau2))
    g = np.where(t >= 0, np.exp(-t / tau2) - np.exp(-t / tau1), 0.0)
    return g * factor


class Drive:
    """Precomputed per-pathway synaptic conductance, g[k] of shape (nt, ncomp).

    Precomputed rather than event-driven because the transfer to the electrode
    is linear and time-invariant, so nothing is gained by streaming events, and
    a dense array lets the same drive be replayed against several biophysical
    variants without re-randomizing which compartments were hit.
    """

    def __init__(self, pathways, cmp_, dt, nt, activation, rng=None,
                 active_fraction=1.0, jitter_ms=0.0):
        rng = rng or np.random.default_rng(0)
        self.dt, self.nt = dt, nt
        self.g, self.erev, self.tag = [], [], []
        t = np.arange(nt) * dt
        for p in pathways:
            times = activation.get(p.pre)
            if times is None:
                continue
            n = allocate(p, cmp_) * float(active_fraction)
            if n.sum() <= 0:
                continue
            gc = np.zeros((nt, len(cmp_)), np.float32)
            for t0 in np.atleast_1d(times):
                shift = rng.normal(0.0, jitter_ms) if jitter_ms > 0 else 0.0
                gc += np.float32(_dual_exp(t - (t0 + shift), p.tau1, p.tau2)[:, None]
                                 * (n * p.w)[None, :])
            self.g.append(gc); self.erev.append(p.erev); self.tag.append(f"{p.pre}->{p.region}")

    def at(self, k):
        """[(g (ncomp,) uS, erev mV), ...] at step k."""
        return [(g[k], e) for g, e in zip(self.g, self.erev)]

    def __len__(self):
        return len(self.g)
