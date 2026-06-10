# ════════════════════════════════════════════════════════════════════════════
#  fiber_localize.py — physical localization of fibers from raw waveform geometry.
#
#  Fits a point-source field to each unit's per-channel RAW peak-to-peak amplitude
#  profile to recover its position relative to the probe:
#
#      a_c  =  A / d_c   +   B · ((y_c − y0) / d_c) / d_c²        (monopole + axial dipole)
#      d_c  =  √((x0 − x_c)² + (y0 − y_c)² + z0²)
#
#  The MONOPOLE term encodes distance through the spatial SPREAD of the footprint
#  (FWHM ≈ 3.46·z0): a steep falloff across channels ⇒ near, a broad one ⇒ far —
#  i.e. distance is read from the angular spread across electrodes, independent of
#  source strength A.  The DIPOLE term captures an ASYMMETRIC footprint (a source
#  off to one side, or soma/dendrite separation along the shank); it carries
#  distance information the symmetric monopole misses and rescues units a pure
#  monopole would pin to the boundary.
#
#  IMPORTANT — localize on RAW amplitudes (.spk / .fil), never on the stderiv .spkD
#  or whitened features: the stderiv transform removes common mode and reweights
#  channels, breaking the amplitude–distance law.
#
#  Honest limits on a linear / single-column probe (validated on the Buzsaki64L
#  octrode, group 5 of sirotaA-jg-000005-20120312):
#    - depth (y0) along the shank is well constrained for interior units;
#    - perpendicular distance (z0) is recoverable from the footprint spread ONLY
#      when BOTH flanks of the footprint are sampled — interior units localize to
#      ~1–3 µm (e.g. z≈24 µm, stable across loss choices), but EDGE units (peak on
#      a terminal channel, one flank only) stay degenerate and are flagged;
#    - per-spike amplitude noise is large, so distance is a per-UNIT (template)
#      quantity with a bootstrap CI, not a per-spike one.
#
#  ENERGY-STRATIFIED depth (the depth fit on low- vs high-energy spikes) reports
#  whether the effective source moves along the shank with spike amplitude: ~0 for
#  a compact source, non-zero for an axial (soma–dendrite) extent — the d(r)
#  curvature expressed as a length.  Lightweight CPU least-squares (no GPU needed).
# ════════════════════════════════════════════════════════════════════════════
import argparse
from collections import defaultdict

import numpy as np
import yaml
from scipy.optimize import least_squares

try:
    from . import neuro_io as nio
except ImportError:
    import neuro_io as nio
try:
    from . import session_yaml as sy
except ImportError:
    import session_yaml as sy


# ── probe geometry ───────────────────────────────────────────────────────────
def load_geometry(probe_paths, channels):
    """Read one or more NeuroSuite .probe YAML files, concatenate their
    sites.geometry [x,y] lists in order into a global table, and index it by the
    (global) channel ids of a spike group.  Returns xy (nchan, 2) in µm."""
    geo = []
    for p in (probe_paths if isinstance(probe_paths, (list, tuple)) else [probe_paths]):
        with open(p) as fh:
            doc = yaml.safe_load(fh)
        sites = (doc.get("probeFile", {}) or {}).get("sites", {}) or {}
        g = sites.get("geometry")
        if g is None:
            raise SystemExit(f"[localize] {p}: no probeFile.sites.geometry")
        geo.extend(g)
    geo = np.asarray(geo, float)
    ch = np.asarray(channels, int)
    if ch.max() >= len(geo):
        raise SystemExit(f"[localize] channel {ch.max()} exceeds geometry table "
                         f"({len(geo)} sites) — pass all probe files in channel order")
    return geo[ch]


# ── point-source forward model + fit ─────────────────────────────────────────
def _model(p, xy, dipole):
    x0, y0, z0, A = p[0], p[1], p[2], p[3]
    d = np.sqrt((x0 - xy[:, 0]) ** 2 + (y0 - xy[:, 1]) ** 2 + z0 ** 2)
    pred = A / d
    if dipole:
        pred = pred + p[4] * ((xy[:, 1] - y0) / d) / d ** 2
    return pred, d


def _fit(a, xy, dipole=True):
    """Fit the (monopole[+dipole]) field to an amplitude profile a (nchan,).
    Returns params, 1σ (analytic), and relative RMS residual."""
    a = np.maximum(np.asarray(a, float), 1e-3)
    w = a / a.sum()
    x0 = float((w * xy[:, 0]).sum()); y0 = float((w * xy[:, 1]).sum())
    p0 = [x0, y0, 25.0, a.max() * 30.0]
    lb = [xy[:, 0].min() - 60, xy[:, 1].min() - 40, 3.0, 1.0]
    ub = [xy[:, 0].max() + 60, xy[:, 1].max() + 40, 300.0, 1e9]
    if dipole:
        p0 += [0.0]; lb += [-1e11]; ub += [1e11]
    p0 = list(np.clip(np.asarray(p0, float), lb, ub))   # keep the initial guess feasible
                                                         # (bootstrap resamples can push p0[3] past ub)
    fscale = max(a.max() * 0.05, 1e-6)

    def resid(p):
        pred, _ = _model(p, xy, dipole)
        return a - pred

    s = least_squares(resid, p0, bounds=(lb, ub), loss="soft_l1",
                      f_scale=fscale, max_nfev=6000)
    pred, _ = _model(s.x, xy, dipole)
    rel = float(np.sqrt(np.mean((a - pred) ** 2)) / a.max())
    try:
        dof = max(1, len(a) - len(s.x))
        cov = np.linalg.inv(s.jac.T @ s.jac) * (2 * s.cost / dof)
        sig = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        sig = np.full(len(s.x), np.nan)
    return s.x, sig, rel


def _amp(wav):
    """Per-channel peak-to-peak of a stack of waveforms (n, nsamp, nchan)."""
    return wav.max(1) - wav.min(1)


def _edge_flag(a, xy):
    """One-flank (degenerate distance) if the peak channel is at a terminal depth
    of the group so only one side of the footprint is observed."""
    ypc = xy[int(np.argmax(a)), 1]
    return int(ypc <= xy[:, 1].min() + 1e-6 or ypc >= xy[:, 1].max() - 1e-6)


# ── per-unit localization ────────────────────────────────────────────────────
def localize_unit(wav, xy, dipole=True, nboot=200, min_terc=60, rng=None):
    """Localize one unit from its raw waveforms wav (nspk, nsamp, nchan).
    Distance is read from the footprint spread (bootstrap CI over spikes);
    energy-stratified depth gives the axial extent (soma–dendrite signature)."""
    rng = rng or np.random.default_rng(0)
    ptp = _amp(np.asarray(wav, np.float32))             # (nspk, nchan)
    a = np.median(ptp, 0)
    p, sig, rel = _fit(a, xy, dipole)
    x0, y0, z0 = float(p[0]), float(p[1]), abs(float(p[2]))
    A = float(p[3])                                  # monopole source amplitude (drift-invariant identity)
    B = float(p[4]) if dipole else 0.0
    pc = int(np.argmax(a))
    dist = float(np.sqrt((x0 - xy[pc, 0]) ** 2 + (y0 - xy[pc, 1]) ** 2 + z0 ** 2))
    # bootstrap distance + depth over spikes (the trustworthy CI)
    z_lo = z_hi = y_lo = y_hi = np.nan
    if nboot and len(wav) >= 8:
        zb = np.empty(nboot); yb = np.empty(nboot)
        for b in range(nboot):
            ab = np.median(ptp[rng.integers(0, len(wav), len(wav))], 0)
            pb, _, _ = _fit(ab, xy, dipole); zb[b] = abs(pb[2]); yb[b] = pb[1]
        z_lo, z_hi = np.percentile(zb, [16, 84]); y_lo, y_hi = np.percentile(yb, [16, 84])
    # energy-stratified depth → axial extent
    depth_shift = np.nan
    if len(wav) >= min_terc:
        e = ptp.max(1); o = np.argsort(e); t = len(o) // 3
        y_low = _fit(np.median(ptp[o[:t]], 0), xy, dipole)[0][1]
        y_high = _fit(np.median(ptp[o[2 * t:]], 0), xy, dipole)[0][1]
        depth_shift = float(y_high - y_low)
    return dict(x0=x0, y0=y0, z0=z0, A=A, dist=dist, dipoleB=B, resid=rel,
                sig_y=float(sig[1]), z_lo=float(z_lo), z_hi=float(z_hi),
                y_lo=float(y_lo), y_hi=float(y_hi), depth_shift=depth_shift,
                one_flank=_edge_flag(a, xy),
                at_bound=int(z0 < 4.0 or z0 > 299.0))   # distance pinned to the fit limit


# ── driver ───────────────────────────────────────────────────────────────────
def localize(base, elec, nsamp, nchan, xy, clu_path=None, dipole=True,
             min_n=50, nboot=200, max_spikes=2000, max_resid=0.10, verbose=True):
    """Localize every unit in <base>.spk.<elec> (RAW waveforms) using its
    (re-linked) .clu.  Returns one report row per unit."""
    spk, spk_path = nio.open_spk(base, elec, nsamp, nchan)
    if clu_path:
        _, labels = nio.read_clu_file(clu_path)
    else:
        _, labels = nio.read_clu(base, elec)
    n = min(len(labels), spk.shape[0]); labels = labels[:n]
    by = defaultdict(list)
    for i, l in enumerate(labels):
        if l > 0:
            by[int(l)].append(i)
    rng = np.random.default_rng(0)
    rows = []
    for u, idx in sorted(by.items()):
        idx = np.asarray(idx)
        if len(idx) > max_spikes:                       # subsample for template/bootstrap
            idx = rng.choice(idx, max_spikes, replace=False)
        wav = np.asarray(spk[np.sort(idx)], np.float32)
        d = localize_unit(wav, xy, dipole, nboot if len(idx) >= min_n else 0, rng=rng)
        d.update(unit=u, nspk=int((np.asarray(labels) == u).sum()),
                 low_n=int(len(idx) < min_n), high_resid=int(d["resid"] > max_resid))
        d["reliable"] = int(not (d["one_flank"] or d["at_bound"] or d["low_n"] or d["high_resid"]))
        rows.append(d)
    rows.sort(key=lambda r: -r["nspk"])
    if verbose:
        rel = sum(r["reliable"] for r in rows)
        print(f"[localize] {spk_path}: {len(rows)} units  reliable(both-flanks, n≥{min_n}, "
              f"resid≤{max_resid:.0%})={rel}  edge/one-flank={sum(r['one_flank'] for r in rows)}")
    return rows


_COLS = ["unit", "nspk", "x0", "y0", "z0", "dist", "z_lo", "z_hi", "y_lo", "y_hi",
         "dipoleB", "depth_shift", "resid", "one_flank", "at_bound", "low_n", "high_resid", "reliable"]


def write_localizations(rows, path):
    with open(path, "w") as f:
        f.write("\t".join(_COLS) + "\n")
        for d in rows:
            f.write("\t".join(
                (f"{d[c]:.3f}" if isinstance(d[c], float) else str(d[c])) for c in _COLS) + "\n")
    return path


def main():
    ap = argparse.ArgumentParser(
        description="Localize fibers (distance + depth + orientation) from raw waveform spread.")
    ap.add_argument("base"); ap.add_argument("elec", type=int)
    ap.add_argument("--nsamp", type=int, required=True)
    ap.add_argument("--nchan", type=int, required=True)
    ap.add_argument("--probe", nargs="+", required=True,
                    help="NeuroSuite .probe YAML(s), in global-channel order")
    ap.add_argument("--channels", default=None,
                    help="comma-separated global channel ids of this group (else read <session>.yaml)")
    ap.add_argument("--session", default=None, help="session for channel lookup if --channels omitted")
    ap.add_argument("--clu", default=None, help="cluster file (pass the re-linked .clu)")
    ap.add_argument("--no-dipole", action="store_true")
    ap.add_argument("--min-n", type=int, default=50)
    ap.add_argument("--nboot", type=int, default=200)
    ap.add_argument("--max-resid", type=float, default=0.10)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.channels:
        channels = [int(c) for c in a.channels.split(",")]
    elif a.session:
        channels = sy.resolve_session_params(a.session, a.elec)["channels"]
    else:
        raise SystemExit("[localize] need --channels or --session for the group's channel ids")
    xy = load_geometry(a.probe, channels)
    rows = localize(a.base, a.elec, a.nsamp, a.nchan, xy, a.clu,
                    dipole=not a.no_dipole, min_n=a.min_n, nboot=a.nboot, max_resid=a.max_resid)
    out = a.out or f"{a.base}.localize.{a.elec}.tsv"
    write_localizations(rows, out)
    print(f"[localize] wrote {out}")


if __name__ == "__main__":
    main()
