#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
#  fiber_merge.py — position + shape co-gated fragment merging.
#
#  A fiber fragment is merged with another iff they are CO-LOCATED in physical
#  space (monopole inverse solution: depth y0, distance z0, source amplitude A)
#  AND their median raw templates AGREE in shape.  Validated on real g5: interior
#  co-located fragments track shape identity (12-way over-split at y0=64um,
#  A~4.7e4 -> min pairwise template cosine 0.879), while OFF-PROBE / one-flank
#  units are degenerate (co-located only by extrapolation) and are correctly
#  rejected by the shape co-gate (their templates disagree, min-cos < 0).
#
#  Intra-chunk (no drift): use the full (y0, z0, A) + shape gate.
#  Inter-chunk (drift): A is the drift-invariant anchor; gate on A + shape and
#  drift-predicted y0 (pass already-drift-removed y0).  Edge units (one_flank, or
#  y0 outside the probe) drop the z0 term and rely on (y0, A, shape).
#
#  IMPORTANT: localize on RAW amplitudes (fiber_localize on .spk/.fil), never the
#  stderiv .spkD — the stderiv transform breaks the amplitude-distance law.
# ─────────────────────────────────────────────────────────────────────────────
import itertools
import numpy as np

try:
    from . import fiber_lib as fl
except ImportError:
    import fiber_lib as fl


def masked_cos(ta, tb, mask):
    """Cosine between two templates over the masked spike frame (shape identity)."""
    a = ta[mask].ravel(); b = tb[mask].ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def is_interior(pos_i, probe_y=None):
    """A fragment's lateral/distance localization is reliable only if its peak is
    not on a terminal channel (one_flank) and its depth lands within the probe."""
    if pos_i.get("one_flank"):
        return False
    if probe_y is not None and not (probe_y[0] <= pos_i["y0"] <= probe_y[1]):
        return False
    return True


def pair_gate(i, j, pos, templates, mask, *, dy_um, dlogA, dz_um, min_cos, probe_y):
    """True iff fragments i, j are co-located AND shape-consistent (mergeable)."""
    fi, fj = pos[i], pos[j]
    if abs(fi["y0"] - fj["y0"]) > dy_um:
        return False
    if abs(np.log(fi["A"]) - np.log(fj["A"])) > dlogA:           # source amplitude within ~exp(dlogA)
        return False
    if is_interior(fi, probe_y) and is_interior(fj, probe_y):
        if abs(fi["z0"] - fj["z0"]) > dz_um:                     # lateral only when reliable
            return False
    return masked_cos(templates[i], templates[j], mask) >= min_cos   # shape co-gate (vetoes degeneracies)


def position_shape_merge(pos, templates, mask=None, *, dy_um=6.0, dlogA=0.25,
                         dz_um=8.0, min_cos=0.85, probe_y=None, clique=False):
    """Group fragments by position+shape co-gate.

    pos:        {id: dict(y0, z0, A, one_flank)}   monopole localization per fragment
    templates:  {id: (nsamp, nchan) median raw template}
    clique=False: union by connected components (a fragment chains in if it gates
                  to ANY group member); clique=True requires it to gate to ALL
                  members (stricter, no shape chaining).
    Returns a list of groups (each a list of fragment ids)."""
    if mask is None:
        mask = fl.MASK_FULL
    ids = list(pos)
    kw = dict(dy_um=dy_um, dlogA=dlogA, dz_um=dz_um, min_cos=min_cos, probe_y=probe_y)
    parent = {i: i for i in ids}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    if not clique:
        for i, j in itertools.combinations(ids, 2):
            if pair_gate(i, j, pos, templates, mask, **kw):
                parent[find(i)] = find(j)
    else:
        groups = []                                              # greedy cliques
        for i in ids:
            placed = False
            for g in groups:
                if all(pair_gate(i, m, pos, templates, mask, **kw) for m in g):
                    g.append(i); placed = True; break
            if not placed:
                groups.append([i])
        return groups
    out = {}
    for i in ids:
        out.setdefault(find(i), []).append(i)
    return list(out.values())
