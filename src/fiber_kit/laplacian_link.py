#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════════════════════
#  laplacian_link.py — connect fibers fragmented BETWEEN energy levels by
#  testing curve continuity, using the Laplacian (curvature) of each fiber arc.
#
#  A fiber is one smooth curve d(r): unit direction vs energy r.  If a unit is
#  fragmented across energy (e.g. the inclusion radius cut its low/high tail, or
#  drift split its amplitude band), the pieces are adjacent ARCS of that curve.
#  We extrapolate each arc to the join radius with a local quadratic
#  (value + tangent + curvature; the curvature IS the discrete Laplacian) and
#  link arcs whose extrapolated directions agree.  Grouping = connected
#  components of the resulting graph = null space of its graph Laplacian.
#
#  ── CAVEAT (measured on g5) ────────────────────────────────────────────────
#  On low-drift, well-above-noise data this UNDERPERFORMS the template-corr
#  consolidation in fiber_session (merge_corr):
#    * RKK fragments there are co-energy (fully overlapping) -> no gap to bridge.
#    * Split a clean fiber at median energy and its halves' templates already
#      match at corr 0.984; curve continuity is noisier (0.920) and the
#      curvature term makes it worse (deg1 0.940 > deg2 0.920) -- 2nd-derivative
#      is noise-sensitive.  0/16 rescues over template-corr.
#  Use this when fragments are genuinely GAPPED in energy with a degraded
#  low-energy arc (template-corr fails, curve still continuous).  Run
#  energy_banding_report() first to see whether your data has that pattern.
# ════════════════════════════════════════════════════════════════════════════
import numpy as np


def _extrap(grid, d, r_star, end, deg=2, m=10):
    """Local degree-`deg` fit of direction-vs-radius at one end of an arc,
    evaluated at r_star.  deg=2 includes the curvature (Laplacian) term."""
    grid = np.asarray(grid, float); d = np.asarray(d, float)
    if end == 'hi': g = grid[-m:]; dd = d[-m:]; x0 = grid[-1]
    else:           g = grid[:m];  dd = d[:m];  x0 = grid[0]
    deg = min(deg, len(g) - 1)
    x = g - x0; V = np.vander(x, deg + 1, increasing=True)
    coef, *_ = np.linalg.lstsq(V, dd, rcond=None)
    p = np.vander(np.array([r_star - x0]), deg + 1, increasing=True)[0] @ coef
    return p / (np.linalg.norm(p) + 1e-12)


def laplacian_link(geoms, max_gap_frac=0.5, min_cos=0.93, deg=2, fit_pts=10):
    """geoms: list of fiber dicts with 'grid' (radii) and 'dir' (n_grid, p) unit
    directions (as produced by fiber_session.fiber_geom).  Links arcs that are
    each other's smooth continuation across the energy axis.  Returns
    (groups, A, eigvals): index groups, affinity matrix, graph-Laplacian spectrum
    (count of ~0 eigenvalues == number of fibers)."""
    K = len(geoms)
    grids = [np.asarray(g['grid'], float) for g in geoms]
    dirs = [np.asarray(g['dir'], float) for g in geoms]
    rlo = np.array([gr[0] for gr in grids]); rhi = np.array([gr[-1] for gr in grids])
    rmid = 0.5 * (rlo + rhi); span = np.maximum(rhi - rlo, 1e-9)
    A = np.zeros((K, K))
    for i in range(K):
        for j in range(i + 1, K):
            a, b = (i, j) if rmid[i] < rmid[j] else (j, i)     # a lower energy
            gap = rlo[b] - rhi[a]                               # >0 disjoint, <0 overlap
            if gap > max_gap_frac * 0.5 * (span[a] + span[b]): continue
            ma = min(fit_pts, len(grids[a])); mb = min(fit_pts, len(grids[b]))
            if ma < 2 or mb < 2: continue
            r_star = 0.5 * (rhi[a] + rlo[b])
            c = float(_extrap(grids[a], dirs[a], r_star, 'hi', deg, ma)
                      @ _extrap(grids[b], dirs[b], r_star, 'lo', deg, mb))
            if c > min_cos: A[i, j] = A[j, i] = c
    seen = np.zeros(K, bool); groups = []
    for s in range(K):
        if seen[s]: continue
        st = [s]; comp = []
        while st:
            u = st.pop()
            if seen[u]: continue
            seen[u] = True; comp.append(u)
            st += [v for v in range(K) if A[u, v] > 0 and not seen[v]]
        groups.append(comp)
    Abin = (A > 0).astype(float); L = np.diag(Abin.sum(1)) - Abin
    return groups, A, np.sort(np.linalg.eigvalsh(L))


def energy_banding_report(geoms):
    """Does the data actually have energy-banded fragmentation?  Reports, over
    all fiber pairs, how many are energy-ADJACENT (a real gap, where this linker
    applies) vs energy-OVERLAPPING (co-energy, where template/direction is the
    right tool).  Run this before trusting laplacian_link to add anything."""
    K = len(geoms)
    rlo = np.array([np.asarray(g['grid'], float)[0] for g in geoms])
    rhi = np.array([np.asarray(g['grid'], float)[-1] for g in geoms])
    span = np.maximum(rhi - rlo, 1e-9); rmid = 0.5 * (rlo + rhi)
    adj = ov = 0
    for i in range(K):
        for j in range(i + 1, K):
            a, b = (i, j) if rmid[i] < rmid[j] else (j, i)
            gap = rlo[b] - rhi[a]
            if gap > 0: adj += 1
            elif gap > -0.5 * min(span[a], span[b]): adj += 1   # small overlap ~ adjacent
            else: ov += 1
    print(f"{K} fibers, {adj+ov} pairs: {adj} energy-adjacent (gap), {ov} energy-overlapping")
    print("  -> laplacian_link helps only if a meaningful fraction is energy-adjacent")
    return adj, ov
