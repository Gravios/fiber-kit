# ════════════════════════════════════════════════════════════════════════════
#  fiber_position.py — per-spike NORMALIZED POSITION ALONG THE FIBER MANIFOLD.
#
#  A fiber is a unit's whitened multichannel direction as a function of spike
#  energy, d(r): a curve that BENDS as amplitude changes.  Spike-frequency
#  adaptation and input dynamics move a spike ALONG this curve — an attenuated
#  (short-ISI) spike sits lower on it.  This module assigns each spike a scalar
#  s ∈ [0,1] giving where along its unit's manifold it sits (0 = low-energy /
#  most-adapted end, 1 = high-energy end), as fractional ARC LENGTH.
#
#  ── The manifold is the one ESTIMATED IN THE .fibers FILE (not re-estimated) ──
#  fiber-session already fit, per (chunk, fiber), the smooth direction curve
#  d(r) and stored it (`grid`, `dir`).  fiber-relink groups those rows into
#  units tracked across chunks.  Here we CONSOLIDATE a unit's stored per-chunk
#  curves into ONE manifold d̂(u) — direction as a function of normalized arc
#  length u ∈ [0,1] — by arc-length-normalizing each chunk's curve and averaging
#  (nspk-weighted).  There is a single manifold per unit; it is not rebuilt from
#  raw spikes per chunk.
#
#  ── Why position is read from DIRECTION, and why that is drift-independent ────
#  Drift is a slow geometric motion of the probe: over minutes it rescales a
#  unit's footprint, so absolute energy/radius WALKS WITH THE DRIFT and is not a
#  drift-independent coordinate.  But the manifold is parametrized by the
#  footprint SHAPE (unit direction), which is amplitude-invariant: a spike's
#  position is found as the arc length u whose manifold direction d̂(u) best
#  matches the spike's whitened direction.  Adaptation rotates the footprint
#  along the manifold (the curve bends with energy), so this shape coordinate
#  captures the adaptation / input state while being invariant to the slow
#  amplitude rescaling that drift imposes.  Raw energy is reported alongside as
#  a (drift-dependent) reference, but s itself is the direction/shape position.
#
#  Caveat: the stored `dir` of each chunk lives in that chunk's whitener basis;
#  consolidating across chunks assumes the noise covariance is similar between
#  chunks (the same assumption fiber-relink already relies on to compare
#  cross-chunk geometry).  Per-spike direction is noisy (the curve spans only a
#  modest rotation), so s is most informative AGGREGATED (per ISI bin, per time
#  window) — consistent with this pipeline treating geometry as a per-population
#  quantity rather than a per-spike one.
# ════════════════════════════════════════════════════════════════════════════
import argparse
from collections import defaultdict

import numpy as np

try:
    from . import fiber_lib as fl
except ImportError:
    import fiber_lib as fl
try:
    from . import fiber_relink as frl
except ImportError:
    import fiber_relink as frl
try:
    from . import neuro_io as nio
except ImportError:
    import neuro_io as nio
try:
    from . import session_yaml as sy
except ImportError:
    import session_yaml as sy


# ── curve geometry ───────────────────────────────────────────────────────────
def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def curve_arclength(grid, D):
    """Cumulative arc length along {grid[k]·D[k]} -> (G,), L[0]=0, normalized to
    [0,1].  This is the intrinsic position axis of one stored fiber curve."""
    P = np.asarray(grid)[:, None] * np.asarray(D)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    L = np.concatenate([[0.0], np.cumsum(seg)])
    return L / (L[-1] + 1e-12)


def resample_dir_to_u(grid, D, u_grid):
    """Resample a stored direction curve D(grid) onto the normalized-arc-length
    axis u_grid ∈ [0,1] -> (len(u_grid), p) unit directions."""
    L = curve_arclength(grid, D)                       # (G,) in [0,1]
    D = np.asarray(D, float)
    out = np.empty((len(u_grid), D.shape[1]))
    for k in range(D.shape[1]):
        out[:, k] = np.interp(u_grid, L, D[:, k])
    return _unit(out)


# ── consolidate the manifold(s) from the .fibers file ────────────────────────
def load_manifolds(fibers_file, relink=True, min_nspk=60, n_u=64, verbose=True):
    """Build ONE manifold per unit from the curves stored in a .fibers file.

    Returns {unit: dict(u, dhat (n_u,p), n_rows, n_spk, r_lo, r_hi)} and a meta
    dict (mask, nsamp, nchan, p, sr, channels).  Units are the re-linked tracks
    (default) or raw .fibers gids (relink=False)."""
    z = np.load(fibers_file, allow_pickle=True)
    gid = z['gid']; nspk = z['nspk']
    grid = z['dir'].shape[1] and z['grid']; dirs = z['dir']      # (M,G), (M,G,p)
    meta = dict(mask=np.asarray(z['meta_mask']) if 'meta_mask' in z.files else fl.MASK_FULL,
                nsamp=int(z['meta_nsamp']), nchan=int(z['meta_nchan']),
                p=int(z['meta_p']), sr=float(z['meta_sr']),
                channels=np.asarray(z['meta_channels']) if 'meta_channels' in z.files else None,
                elec=int(z['meta_elec']) if 'meta_elec' in z.files else -1)
    if relink:
        row2unit, _, _, _ = frl.relink(fibers_file, verbose=False)
        lab = np.asarray(row2unit)
    else:
        lab = gid.astype(int)

    u_grid = np.linspace(0.0, 1.0, n_u)
    acc = defaultdict(lambda: np.zeros((n_u, meta['p'])))
    wsum = defaultdict(float); nrow = defaultdict(int); nsp = defaultdict(int)
    rlo = defaultdict(lambda: np.inf); rhi = defaultdict(lambda: -np.inf)
    for r in range(len(gid)):
        u = int(lab[r])
        if u < 0 or nspk[r] < min_nspk:
            continue
        g = np.asarray(grid[r], float); D = np.asarray(dirs[r], float)
        if g[-1] <= g[0]:
            continue
        w = float(nspk[r])
        acc[u] += w * resample_dir_to_u(g, D, u_grid)
        wsum[u] += w; nrow[u] += 1; nsp[u] += int(nspk[r])
        rlo[u] = min(rlo[u], float(g[0])); rhi[u] = max(rhi[u], float(g[-1]))

    manifolds = {}
    for u in acc:
        if nrow[u] < 1:
            continue
        manifolds[u] = dict(u=u_grid, dhat=_unit(acc[u] / max(wsum[u], 1e-9)),
                            n_rows=nrow[u], n_spk=nsp[u], r_lo=rlo[u], r_hi=rhi[u])
    if verbose:
        print(f"[position] {fibers_file}: {len(manifolds)} unit manifolds "
              f"(relink={'on' if relink else 'off'}, min_nspk={min_nspk}, n_u={n_u})")
    return manifolds, meta


# ── per-spike position by direction along a manifold ─────────────────────────
def position_by_direction(Xdir, u_grid, dhat):
    """Spike unit-directions Xdir (n,p) -> (s, conf): s = arc length u whose
    manifold direction best matches the spike (parabolic-refined argmax of the
    cosine), conf = that cosine (1 = perfect shape match)."""
    cos = Xdir @ dhat.T                                # (n, n_u)
    j = cos.argmax(1); rows = np.arange(len(Xdir)); G = len(u_grid)
    jm = np.clip(j - 1, 0, G - 1); jp = np.clip(j + 1, 0, G - 1)
    y0 = cos[rows, jm]; y1 = cos[rows, j]; y2 = cos[rows, jp]
    denom = y0 - 2.0 * y1 + y2
    delta = np.where(np.abs(denom) > 1e-12, 0.5 * (y0 - y2) / denom, 0.0)
    edge = (j == 0) | (j == G - 1)
    f = j.astype(float) + np.where(edge, 0.0, np.clip(delta, -1.0, 1.0))
    s = np.interp(f, np.arange(G), u_grid)
    return np.clip(s, 0.0, 1.0), y1


def spike_positions(Xg, manifold):
    """Whitened features Xg (n,p) of one unit's spikes -> (s, conf, energy).
    Position is the manifold arc length matched by DIRECTION (drift-independent);
    energy = ‖Xg‖ is returned as a drift-DEPENDENT reference, not the position."""
    r = np.linalg.norm(Xg, axis=1)
    s, conf = position_by_direction(_unit(Xg), manifold['u'], manifold['dhat'])
    return s, conf, r


# ── session driver: per chunk compute features, project onto unit manifolds ──
def run(base, elec, fibers_file, nsamp, nchan, ntotal, gch, sr, clu_path=None,
        relink=True, min_nspk=60, n_u=64, chunk_min=20.0, min_n=20,
        mask=None, verbose=True):
    """Assign every sorted spike a drift-independent position along its unit's
    manifold (consolidated from `fibers_file`).  Per-spike whitened features are
    computed per chunk (realign-per-unit + the chunk whitener, as fiber-session
    does); the MANIFOLD is the stored/consolidated one, not re-estimated.
    Returns res-ordered arrays: res, unit, s, conf, energy, chunk."""
    try:
        from .fiber_session import read_res, open_spkD, fil_chunk_whitener
    except ImportError:
        from fiber_session import read_res, open_spkD, fil_chunk_whitener
    manifolds, meta = load_manifolds(fibers_file, relink=relink, min_nspk=min_nspk,
                                     n_u=n_u, verbose=verbose)
    mask = meta['mask'] if mask is None else mask
    res = read_res(base, elec)
    spk, _ = open_spkD(base, elec, nsamp, nchan)
    n = min(len(res), spk.shape[0]); res = res[:n]
    if clu_path:
        _, labels = nio.read_clu_file(clu_path)
    else:
        _, labels = nio.read_clu(base, elec)
    labels = np.asarray(labels[:n])
    filmm = nio.open_signal(f"{base}.fil", ntotal)
    gch = np.asarray(gch, int)

    s = np.full(n, np.nan); conf = np.full(n, np.nan)
    energy = np.full(n, np.nan); chunk_id = np.full(n, -1, int)
    chunk_s = chunk_min * 60.0 * sr
    t0, t1 = int(res.min()), int(res.max())
    nchunks = max(1, int(np.ceil((t1 - t0) / chunk_s)))
    for c in range(nchunks):
        lo = t0 + c * chunk_s; hi = t0 + (c + 1) * chunk_s
        sel = np.flatnonzero((res >= lo) & (res < hi))
        if len(sel) < min_n:
            continue
        res_e = res[sel]; waves = np.asarray(spk[sel], float); lab_e = labels[sel]
        s0 = int(res_e.min()) - nsamp; s1 = int(res_e.max()) + nsamp + 1
        W, nmean, _ = fil_chunk_whitener(filmm, gch, s0, s1, res_e, nsamp, mask)
        for u in np.unique(lab_e[lab_e > 0]):
            if u not in manifolds:
                continue
            idx = sel[lab_e == u]
            if len(idx) < min_n:
                continue
            Wal = fl.realign(waves[lab_e == u])            # per-fiber alignment
            Xg = (Wal[:, mask, :].reshape(len(idx), -1) - nmean) @ W
            su, cu, ru = spike_positions(Xg, manifolds[u])
            s[idx] = su; conf[idx] = cu; energy[idx] = ru; chunk_id[idx] = c
        if verbose:
            print(f"[position] chunk {c+1}/{nchunks}: {len(sel)} spikes, "
                  f"{int(np.isfinite(s[sel]).sum())} positioned")
    if verbose:
        ok = np.isfinite(s)
        print(f"[position] {int(ok.sum())}/{n} spikes positioned on {len(manifolds)} "
              f"unit manifolds (drift-independent shape position)")
    return dict(res=res, unit=labels, s=s, conf=conf, energy=energy, chunk=chunk_id)


def write_positions(out, base, elec):
    """Bare per-spike position as binary float32 (parallel to .res) + an
    analysis .npz with all columns (res, unit, s, conf, energy, chunk)."""
    pos_path = f"{base}.position.{elec}"
    np.asarray(out["s"], dtype="<f4").tofile(pos_path)
    npz_path = f"{base}.position.{elec}.npz"
    np.savez(npz_path, **out)
    return pos_path, npz_path


def main():
    ap = argparse.ArgumentParser(
        description="Per-spike drift-independent position along the fiber manifold (from a .fibers file).")
    ap.add_argument("base"); ap.add_argument("elec", type=int)
    ap.add_argument("--fibers", required=True, help="<base>.fibers.<method>.<elec> (the estimated manifold)")
    ap.add_argument("--nsamp", type=int, default=None, help="override: samples per spike (default from YAML)")
    ap.add_argument("--nchan", type=int, default=None, help="override: channels in this group (default from YAML)")
    ap.add_argument("--ntotal", type=int, default=None, help="override: total channels in the .fil (default from YAML)")
    ap.add_argument("--channels", default=None, help="comma-separated global channel ids (else from --session/.fibers)")
    ap.add_argument("--session", default=None, help="session .yaml for channel/sr lookup")
    ap.add_argument("--sr", type=float, default=None)
    ap.add_argument("--clu", default=None, help="cluster file (the re-linked .clu)")
    ap.add_argument("--no-relink", action="store_true", help="use raw .fibers gid units instead of re-linked tracks")
    ap.add_argument("--min-nspk", type=int, default=60, help="min spikes for a chunk-curve to enter a manifold")
    ap.add_argument("--n-u", type=int, default=64, help="manifold arc-length resolution")
    ap.add_argument("--chunk-min", type=float, default=20.0)
    ap.add_argument("--min-n", type=int, default=20)
    a = ap.parse_args()
    # channels / sr / nsamp / nchan / ntotal: prefer explicit, else session, else the .fibers meta
    gch = sr = None
    nsamp, nchan, ntotal = a.nsamp, a.nchan, a.ntotal
    if a.channels:
        gch = [int(c) for c in a.channels.split(",")]
    if a.sr:
        sr = a.sr
    if a.session and (gch is None or sr is None or nsamp is None or nchan is None or ntotal is None):
        prm = sy.resolve_session_params(a.session, a.elec)
        gch = gch or prm["channels"]; sr = sr or prm.get("sr")
        nsamp = nsamp if nsamp is not None else prm.get("nsamp")
        nchan = nchan if nchan is not None else prm.get("nchan")
        ntotal = ntotal if ntotal is not None else prm.get("ntotal")
    if gch is None or sr is None:
        z = np.load(a.fibers, allow_pickle=True)
        if gch is None and 'meta_channels' in z.files: gch = np.asarray(z['meta_channels'])
        if sr is None and 'meta_sr' in z.files: sr = float(z['meta_sr'])
    if gch is None or sr is None:
        raise SystemExit("[position] need --channels/--sr or --session (or a .fibers with meta)")
    if nsamp is None or nchan is None or ntotal is None:
        raise SystemExit("[position] need nSamples/nChannels/nTotal from <session>.yaml or --nsamp/--nchan/--ntotal")
    out = run(a.base, a.elec, a.fibers, nsamp, nchan, ntotal, gch, sr,
              clu_path=a.clu, relink=not a.no_relink, min_nspk=a.min_nspk,
              n_u=a.n_u, chunk_min=a.chunk_min, min_n=a.min_n)
    pos_path, npz_path = write_positions(out, a.base, a.elec)
    print(f"[position] wrote {pos_path} and {npz_path}")


if __name__ == "__main__":
    main()
