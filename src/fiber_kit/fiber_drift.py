# ════════════════════════════════════════════════════════════════════════════
#  fiber_drift.py — probe drift tracking from fiber depth trajectories.
#
#  Every unit tracked across chunks is a drift fiducial: its apparent depth on the
#  probe changes only because the probe/tissue moved.  Pooling the per-chunk depths
#  of all units on a probe and solving a decentralised registration recovers the
#  probe's drift over time — using ONLY the .fibers files (per-(chunk,fiber) depth
#  + spike count), no raw data.
#
#  Model (one shared drift curve per probe, per-unit depth offset):
#
#      depth_u(c)  =  base_u  +  D(c)  +  ε
#
#  D(c) is the probe drift at chunk c (gauge: D = 0 at the first chunk); base_u is
#  the unit's own depth.  Solved by an iteratively-reweighted median (robust to
#  mis-tracked units): D(c) = median_u[depth_u(c) − base_u]; base_u =
#  median_c[depth_u(c) − D(c)].  With several groups on one probe, all their units
#  feed one joint D(c) (rigid probe); each group also gets its own D_g(c) so the
#  spread across shanks reveals tilt / non-rigid motion.  After removing D(c), the
#  slope of the residual against a unit's base depth is the depth-gradient of drift
#  — the signature that triggers position-dependent (non-rigid) correction.
#
#  Depths come from .fibers in CHANNEL units; output is converted to µm via the
#  site pitch (default 20).  Because cross-chunk tracking gates on small per-step
#  depth change, this sees SLOW drift well; drift faster than the link gate breaks
#  tracks and is better estimated by raster registration — a documented limit.
# ════════════════════════════════════════════════════════════════════════════
import argparse
from collections import defaultdict

import numpy as np

try:
    from . import fiber_relink as frl
except ImportError:
    import fiber_relink as frl


def _load_depths(fibers_file, relink=True, min_nspk=60, min_span=3):
    """Return obs[unit]->{chunk:depth}, chunk->tmin(min), and the group id, using
    re-linked units (more fiducials) or the raw .fibers gid when relink=False.
    Only fibers with >= min_nspk spikes are used; units must span >= min_span chunks."""
    z = np.load(fibers_file, allow_pickle=True)
    gid, chunk, depth, nspk, tmin = z['gid'], z['chunk'], z['depth'], z['nspk'], z['tmin']
    group = int(z['meta_elec']) if 'meta_elec' in z.files else -1
    if relink:
        row2unit, _, _, _ = frl.relink(fibers_file, verbose=False)
        lab = row2unit
    else:
        lab = gid.astype(int)
    obs = defaultdict(dict)
    tmap = {}
    for r in range(len(gid)):
        if lab[r] < 0 or nspk[r] < min_nspk:
            continue
        c = int(chunk[r])
        obs[(group, int(lab[r]))][c] = float(depth[r])
        tmap[c] = float(tmin[r])
    obs = {u: tr for u, tr in obs.items() if len(tr) >= min_span}
    return obs, tmap, group


def decentralized_drift(obs, chunks, iters=60):
    """Solve depth_u(c)=base_u+D(c) by robust (median) alternation.
    Returns D (len(chunks), channel units, gauge D[0]=0), base dict, and arrays
    (residual, base_depth) over all observations for the non-rigid diagnostic."""
    ci = {c: i for i, c in enumerate(chunks)}
    base = {u: float(np.median(list(tr.values()))) for u, tr in obs.items()}
    D = np.zeros(len(chunks))
    for _ in range(iters):
        acc = defaultdict(list)
        for u, tr in obs.items():
            for c, d in tr.items():
                acc[c].append(d - base[u])
        Dn = np.array([np.median(acc[c]) if acc[c] else 0.0 for c in chunks])
        Dn -= Dn[0]
        for u, tr in obs.items():
            base[u] = float(np.median([tr[c] - Dn[ci[c]] for c in tr]))
        if np.max(np.abs(Dn - D)) < 1e-4:
            D = Dn; break
        D = Dn
    res, bd = [], []
    for u, tr in obs.items():
        for c, d in tr.items():
            res.append(d - base[u] - D[ci[c]]); bd.append(base[u])
    return D, base, np.array(res), np.array(bd)


def drift_curve(fibers_files, relink=True, min_nspk=60, min_span=3, pitch=20.0, verbose=True):
    """Estimate per-group and joint-probe drift from one or more .fibers files."""
    per_group = {}
    all_obs = {}
    tmap = {}
    for f in fibers_files:
        obs, tm, g = _load_depths(f, relink, min_nspk, min_span)
        if not obs:
            continue
        tmap.update(tm); all_obs.update(obs)
        chunks = sorted({c for tr in obs.values() for c in tr})
        D, _, res, bd = decentralized_drift(obs, chunks)
        per_group[g] = dict(chunks=chunks, D=D * pitch, n_units=len(obs),
                            resid_std=float(res.std() * pitch),
                            grad=float(np.polyfit(bd - bd.mean(), res, 1)[0] * pitch)
                            if len(res) > 8 else float('nan'))

    chunks = sorted({c for tr in all_obs.values() for c in tr})
    Dj, _, resj, bdj = decentralized_drift(all_obs, chunks)
    Dj = Dj * pitch
    grad = float(np.polyfit(bdj - bdj.mean(), resj, 1)[0] * pitch) if len(resj) > 8 else float('nan')

    rows = []
    ci = {c: i for i, c in enumerate(chunks)}
    for c in chunks:
        nu = sum(1 for tr in all_obs.values() if c in tr)
        row = dict(chunk=c, t_min=round(tmap.get(c, float('nan')), 1), n_units=nu,
                   drift_um=round(float(Dj[ci[c]]), 3))
        for g, pg in per_group.items():
            row[f"drift_g{g}_um"] = round(float(pg['D'][pg['chunks'].index(c)]), 3) \
                if c in pg['chunks'] else float('nan')
        rows.append(row)

    rng = Dj.max() - Dj.min()
    step = np.abs(np.diff(Dj))
    # cross-shank disagreement (non-rigid probe motion) where groups overlap
    shank_spread = float('nan')
    if len(per_group) > 1:
        common = set.intersection(*[set(pg['chunks']) for pg in per_group.values()])
        if common:
            sp = [np.std([pg['D'][pg['chunks'].index(c)] for pg in per_group.values()]) for c in common]
            shank_spread = float(np.mean(sp))
    summary = dict(n_groups=len(per_group), n_units=len(all_obs), n_chunks=len(chunks),
                   drift_range_um=round(float(rng), 2),
                   max_step_um=round(float(step.max()) if len(step) else 0.0, 2),
                   depth_gradient_um_per_ch=round(grad, 3),
                   resid_std_um=round(float(resj.std() * pitch), 2),
                   cross_shank_spread_um=round(shank_spread, 2) if shank_spread == shank_spread else None)
    if verbose:
        print(f"[drift] groups={summary['n_groups']} fiducial-units={summary['n_units']} "
              f"chunks={summary['n_chunks']}")
        print(f"[drift] probe drift range={summary['drift_range_um']}µm  max step={summary['max_step_um']}µm  "
              f"resid={summary['resid_std_um']}µm")
        nr = abs(grad) > 0.15 and abs(grad) > 0.3 * (rng / max(1, len(chunks)))
        print(f"[drift] depth-gradient={summary['depth_gradient_um_per_ch']}µm/ch "
              f"-> {'NON-RIGID (position-dependent) drift indicated' if abs(grad) > 0.15 else 'consistent with rigid drift'}")
        if summary['cross_shank_spread_um'] is not None:
            print(f"[drift] cross-shank spread={summary['cross_shank_spread_um']}µm "
                  f"({'shanks disagree -> tilt/bending' if summary['cross_shank_spread_um'] > 1.0 else 'shanks agree -> rigid'})")
    return dict(rows=rows, summary=summary, per_group=per_group, chunks=chunks, drift_um=Dj)


_COLS_BASE = ["chunk", "t_min", "n_units", "drift_um"]


def write_drift_table(result, path):
    extra = [k for k in result['rows'][0] if k not in _COLS_BASE]
    cols = _COLS_BASE + extra
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for d in result['rows']:
            f.write("\t".join(str(d.get(c, "")) for c in cols) + "\n")
    return path


def main():
    ap = argparse.ArgumentParser(
        description="Track probe drift over time from the fiber files of a probe's groups.")
    ap.add_argument("fibers", nargs="+", help="one or more <base>.fibers.<method>.<group> files (a probe's groups)")
    ap.add_argument("--no-relink", action="store_true", help="use raw .fibers gid instead of re-linked units")
    ap.add_argument("--min-nspk", type=int, default=60)
    ap.add_argument("--min-span", type=int, default=3)
    ap.add_argument("--pitch", type=float, default=20.0, help="site pitch in µm (depth is in channel units)")
    ap.add_argument("--out", default=None, help="drift table TSV")
    ap.add_argument("--npy", default=None, help="save the drift curve D(c) in µm as .npy")
    a = ap.parse_args()
    res = drift_curve(a.fibers, relink=not a.no_relink, min_nspk=a.min_nspk,
                      min_span=a.min_span, pitch=a.pitch)
    out = a.out or "fiber_drift.tsv"
    write_drift_table(res, out); print(f"[drift] wrote {out}")
    if a.npy:
        np.save(a.npy, res['drift_um']); print(f"[drift] wrote {a.npy}")


if __name__ == "__main__":
    main()
