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
    from . import fiber_lib as fl
except ImportError:
    import fiber_lib as fl

try:
    from . import neuro_io as nio
except ImportError:
    import neuro_io as nio
try:
    from . import session_yaml as sy
except ImportError:
    import session_yaml as sy
try:
    from . import fiber_pca as fpca
except ImportError:
    import fiber_pca as fpca


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


def _fit_edge(a, xy, dipole=True):
    """Angular-constrained fit for one-flank units whose unconstrained fit runs the depth OFF the
    array end (the seen flank fixes the radial decay but not the off-end axial position, so with the
    dipole term z0 and A blow up together).  The one-flank source sits AT the terminal channel, so
    pin the lateral position to the peak channel, allow the depth to slide only within ±one pitch of
    that channel (toward / just past the end), and let the decay set the perpendicular distance r —
    the angular relation (a_pk/a_adj = √(1+(s/r)²)) constraining the last dimension.  Monopole only
    (the dipole DOF is what enabled the off-end runaway).  Validated on g5: bounded depth/distance,
    residual ~0.08, split-half stability on par with interior units."""
    a = np.maximum(np.asarray(a, float), 1e-3)
    ymin, ymax = xy[:, 1].min(), xy[:, 1].max()
    pitch = (ymax - ymin) / max(1, len(np.unique(xy[:, 1])) - 1)
    pk = int(np.argmax(a)); x0 = float(xy[pk, 0]); yp = float(xy[pk, 1])

    def resid(q):
        y0, r, A = q
        d = np.sqrt((x0 - xy[:, 0]) ** 2 + (y0 - xy[:, 1]) ** 2 + r ** 2)
        return a - A / d

    s = least_squares(resid, [yp, 25.0, a.max() * 30.0],
                      bounds=([yp - pitch, 3.0, 1.0], [yp + pitch, 300.0, 1e9]),
                      x_scale=[pitch, 10.0, a.max() * 30.0], loss="soft_l1",
                      f_scale=max(a.max() * 0.05, 1e-6), max_nfev=4000)
    y0, r, A = float(s.x[0]), float(s.x[1]), float(s.x[2])
    pred = A / np.sqrt((x0 - xy[:, 0]) ** 2 + (y0 - xy[:, 1]) ** 2 + r ** 2)
    rel = float(np.sqrt(np.mean((a - pred) ** 2)) / a.max())
    try:
        cov = np.linalg.inv(s.jac.T @ s.jac) * (2 * s.cost / max(1, len(a) - 3))
        sy, sr, sA = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        sy = sr = sA = np.nan
    p = [x0, y0, r, A] + ([0.0] if dipole else [])
    sig = [1.0, float(sy), float(sr), float(sA)] + ([0.0] if dipole else [])
    return np.asarray(p), np.asarray(sig), rel


def _edge_flag(a, xy):
    """One-flank (degenerate distance) if the peak channel is at a terminal depth
    of the group so only one side of the footprint is observed."""
    ypc = xy[int(np.argmax(a)), 1]
    return int(ypc <= xy[:, 1].min() + 1e-6 or ypc >= xy[:, 1].max() - 1e-6)


# ── per-unit localization ────────────────────────────────────────────────────
def load_pca_basis(base, elec):
    """Read the on-disk per-channel PCA eigenvectors that <base>.fet.standard.<elec> was
    projected with: <base>.pca.standard.<elec> (then legacy .pca.<elec>), via fiber_pca.read_pcad.
    Returns the basis dict (means, evec, recShift, data2use, centered) — the exact subspace the
    sort used, so no SVD is fit at all.  Raises FileNotFoundError if absent (caller may fall back
    to fit_amp_basis).  RAW/standard only — never the stderiv .pcaD."""
    r = nio.resolve_input(base, "pca", elec, nio.prefer_standard())
    if not r.found:
        raise FileNotFoundError(f"no .pca.standard basis for {base} elec {elec}")
    b = fpca.read_pcad(r.path)
    b["_pca"] = True                                    # tag for _profile dispatch
    b["_path"] = r.path
    return b


def _pca_profile(W, basis):
    """Per-channel amplitude profile from the .pca.standard basis: the FIRST principal component
    (PC1) score of each channel IS the amplitude — exactly the first feature of that channel's
    block in .fet.standard.N, so this equals reading the fet scores directly.  Returns |PC1 score|
    of the cluster-mean window per channel; mean-subtraction follows the file's `centered` flag
    (the real basis is centered=0).  No realign — the on-disk extraction already carries the sort's
    alignment.  Fixed basis -> stable at any spike count."""
    means, evec = basis["means"], basis["evec"]
    rec, d2, centered = int(basis["recShift"]), int(basis["data2use"]), int(basis["centered"])
    if W.shape[1] < rec + d2:
        raise ValueError(f"_pca_profile: waveform nsamp={W.shape[1]} < recShift+data2use={rec + d2} "
                         f"— the .pca.standard basis is for a different nSamples; use the matching one")
    win = np.asarray(W, np.float64)[:, rec:rec + d2, :].mean(0)   # (data2use, nch) cluster-mean window
    nch = min(evec.shape[0], win.shape[1])
    prof = np.empty(nch)
    for ch in range(nch):
        mu = means[ch] if centered else 0.0
        prof[ch] = abs(float(evec[ch][0] @ (win[:, ch] - mu)))    # PC1 score = the channel amplitude
    return prof


def fit_amp_basis(spk, by, rng, *, k=12, n_clusters=400, per=120, min_n=40):
    """Fit a GROUP-WIDE low-rank waveform basis once, over realigned spikes pooled across many
    clusters — the same object the raw .fet encodes.  Returns (mu, Vt) for use as the `basis`
    argument of _profile / localize_unit.

    Denoising the amplitude template by projecting a cluster's mean onto this fixed basis avoids
    a per-cluster SVD, whose rank-1 estimate is fit to noise on small/low-SNR clusters and
    produces an occasional badly-mislocalized unit (split-half SD_depth 90th-pct ~28 µm vs ~16
    µm here on g5).  The group basis keeps the median denoising win (SD_depth 0.78 µm) while
    cutting that tail to ptp-like (~16 µm), independent of a cluster's spike count.

    Must be a RAW-waveform basis (raw .spk / raw .fet / .pca) — never the stderiv .fetD, whose
    whitening breaks the amplitude–distance law the monopole inverse relies on."""
    cl = [c for c in by if len(by[c]) >= min_n]
    if not cl:
        return None
    cl = list(rng.permutation(np.asarray(cl)))
    pool = []
    for c in cl[:n_clusters]:
        idx = np.asarray(by[int(c)])
        if len(idx) > per:
            idx = rng.choice(idx, per, replace=False)
        Wr = fl.realign(np.asarray(spk[np.sort(idx)], np.float32)).reshape(len(idx), -1)
        pool.append(Wr)
    P = np.vstack(pool)
    mu = P.mean(0)
    Vt = np.linalg.svd(P - mu, full_matrices=False)[2][:k]
    return mu, Vt


def _profile(wav, method="pc1", basis=None):
    """Per-channel amplitude profile fed to the monopole inverse.

      'ptp'  : median over spikes of per-spike peak-to-peak.  Alignment-invariant, but max-min
               of ~nsamp noisy samples carries a ~4σ NOISE FLOOR on every channel, so far/low-
               signal channels are pinned near the floor and the footprint is flattened
               (measured spatial contrast ~2.9 on g5) — the decay the fit reads is degraded.
      'wave' : peak-to-peak of the median realigned waveform (noise averaged out).
      'pc1'  : peak-to-peak of a rank-1 denoised template in standard space — the sharpest
               footprint (contrast ~8.0) and the lowest median split-half SD.  Default.
               With `basis` (a GROUP-WIDE (mu, Vt) from fit_amp_basis / the raw .fet), the
               denoising direction comes from that fixed subspace — the correct, stable choice.
               Without `basis` it falls back to a PER-CLUSTER SVD, which is fit to noise on
               small/low-SNR clusters (heavy split-half tail); pass a basis whenever possible.

    For well-isolated, energy-band-stratified clusters the denoised profiles are both sharper
    and more precise; 'ptp' is retained for back-compat / very low spike counts."""
    W = np.asarray(wav, np.float32)
    if basis is not None and isinstance(basis, dict) and basis.get("_pca"):
        return _pca_profile(W, basis)                   # on-disk per-channel PCA basis (the .fet/.pca)
    if method == "ptp" or len(W) < 8:
        return np.median(_amp(W), 0)
    Wr = fl.realign(W)
    if method == "wave":
        mw = np.median(Wr, 0)
        return mw.max(0) - mw.min(0)
    X = Wr.reshape(len(Wr), -1)
    if basis is not None:                               # GROUP-WIDE basis (the .fet): denoise the
        mu, Vt = basis                                  # cluster mean in a fixed subspace — stable at any n
        m = X.mean(0)
        t = (mu + (m - mu) @ Vt.T @ Vt).reshape(Wr.shape[1:])
        return np.abs(t.max(0) - t.min(0))
    Vt = np.linalg.svd(X, full_matrices=False)[2]       # fallback: per-cluster rank-1 (unstable at low n)
    t = ((X @ Vt[0]).mean() * Vt[0]).reshape(Wr.shape[1:])
    return np.abs(t.max(0) - t.min(0))


def localize_unit(wav, xy, dipole=True, nboot=0, min_terc=60, amp_method="pc1", rng=None, basis=None):
    """Localize one unit from its raw waveforms wav (nspk, nsamp, nchan).
    Position is fit to a denoised per-channel amplitude profile (amp_method, default PC1; pass
    `basis` from fit_amp_basis so the denoising uses the group-wide subspace, not a per-cluster
    SVD); energy-stratified depth gives the axial extent (soma–dendrite signature).
    Depth/distance uncertainty defaults to the analytic Gaussian σ from the fit covariance
    (percentiles around the mean); nboot>0 re-enables the spike bootstrap (validated to match
    the analytic σ on well-isolated clusters, so it is redundant and off by default)."""
    rng = rng or np.random.default_rng(0)
    W = np.asarray(wav, np.float32)
    e = (W.max(1) - W.min(1)).max(1)                    # per-spike energy (tercile stratification)
    a = _profile(W, amp_method, basis)
    p, sig, rel = _fit(a, xy, dipole)
    ymin, ymax = xy[:, 1].min(), xy[:, 1].max()
    pitch = (ymax - ymin) / max(1, len(np.unique(xy[:, 1])) - 1)
    edge = bool(p[1] < ymin - pitch or p[1] > ymax + pitch)   # depth ran off the array end → one-flank degeneracy
    fitfn = _fit
    if edge:                                            # re-fit with the angular constraint (source at the
        fitfn = _fit_edge                               # terminal channel, decay sets the perpendicular distance)
        p, sig, rel = _fit_edge(a, xy, dipole)
    x0, y0, z0 = float(p[0]), float(p[1]), abs(float(p[2]))
    A = float(p[3])                                  # monopole source amplitude (drift-invariant identity)
    B = float(p[4]) if dipole else 0.0
    pc = int(np.argmax(a))
    dist = float(np.sqrt((x0 - xy[pc, 0]) ** 2 + (y0 - xy[pc, 1]) ** 2 + z0 ** 2))
    # analytic σ CI ("percentiles around the mean") — default, free from the fit covariance
    sy, sz = float(sig[1]), float(sig[2])
    y_lo, y_hi = y0 - sy, y0 + sy
    z_lo, z_hi = max(z0 - sz, 0.0), z0 + sz
    if nboot and not edge and len(W) >= 8:              # optional spike bootstrap (redundant; n/a for edge)
        ptp = _amp(W); zb = np.empty(nboot); yb = np.empty(nboot)
        for b in range(nboot):
            ab = np.median(ptp[rng.integers(0, len(W), len(W))], 0)
            pb, _, _ = _fit(ab, xy, dipole); zb[b] = abs(pb[2]); yb[b] = pb[1]
        z_lo, z_hi = np.percentile(zb, [16, 84]); y_lo, y_hi = np.percentile(yb, [16, 84])
    # energy-stratified depth → axial extent (denoised profile within each tercile)
    depth_shift = np.nan
    if len(W) >= min_terc:
        o = np.argsort(e); t = len(o) // 3
        y_low = fitfn(_profile(W[o[:t]], amp_method, basis), xy, dipole)[0][1]
        y_high = fitfn(_profile(W[o[2 * t:]], amp_method, basis), xy, dipole)[0][1]
        depth_shift = float(y_high - y_low)
    return dict(x0=x0, y0=y0, z0=z0, A=A, dist=dist, dipoleB=B, resid=rel,
                sig_y=sy, z_lo=float(z_lo), z_hi=float(z_hi),
                y_lo=float(y_lo), y_hi=float(y_hi), depth_shift=depth_shift,
                one_flank=int(_edge_flag(a, xy)), edge_fit=int(edge),
                at_bound=int(z0 < 4.0 or z0 > 299.0))   # distance pinned to the fit limit


# ── driver ───────────────────────────────────────────────────────────────────
def localize(base, elec, nsamp, nchan, xy, clu_path=None, dipole=True,
             min_n=50, nboot=0, max_spikes=2000, max_resid=0.10, amp_method="pc1",
             amp_basis="auto", verbose=True):
    """Localize every unit in <base>.spk.<elec> (RAW waveforms) using its
    (re-linked) .clu.  Returns one report row per unit.

    amp_basis : 'auto' (default) fits one group-wide raw-waveform basis (fit_amp_basis) and
                denoises every cluster's amplitude template by projecting onto it — stable at
                any spike count, unlike a per-cluster SVD.  None reverts to the per-cluster SVD
                (back-compat).  Only used when amp_method='pc1'."""
    spk, spk_path = nio.open_spk_raw(base, elec, nsamp, nchan)
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
    basis = None
    if amp_method == "pc1" and amp_basis not in (None, "none"):
        if amp_basis in ("pca", "auto"):                # prefer the on-disk .pca.standard eigenvectors
            try:
                basis = load_pca_basis(base, elec)
                if verbose:
                    print(f"[localize] amplitude basis: on-disk {basis['_path']} "
                          f"(per-channel PCA, nComp {basis['evec'].shape[1]})")
            except FileNotFoundError:
                if amp_basis == "pca":
                    raise
        if basis is None and amp_basis in ("auto", "fit"):   # fall back: fit one group-wide basis
            basis = fit_amp_basis(spk, by, rng)
            if verbose and basis is not None:
                print(f"[localize] amplitude basis: fit rank {basis[1].shape[0]} over "
                      f"{min(len(by), 400)} clusters (no .pca.standard found)")
    rows = []
    for u, idx in sorted(by.items()):
        idx = np.asarray(idx)
        if len(idx) > max_spikes:                       # subsample for template/bootstrap
            idx = rng.choice(idx, max_spikes, replace=False)
        wav = np.asarray(spk[np.sort(idx)], np.float32)
        d = localize_unit(wav, xy, dipole, nboot if len(idx) >= min_n else 0,
                          amp_method=amp_method, rng=rng, basis=basis)
        d.update(unit=u, nspk=int((np.asarray(labels) == u).sum()),
                 low_n=int(len(idx) < min_n), high_resid=int(d["resid"] > max_resid))
        d["reliable"] = int(not (d["at_bound"] or d["low_n"] or d["high_resid"]))  # edge units now angular-constrained
        rows.append(d)
    rows.sort(key=lambda r: -r["nspk"])
    if verbose:
        rel = sum(r["reliable"] for r in rows)
        print(f"[localize] {spk_path}: {len(rows)} units  reliable(both-flanks, n≥{min_n}, "
              f"resid≤{max_resid:.0%})={rel}  edge/one-flank={sum(r['one_flank'] for r in rows)}")
    return rows


_COLS = ["unit", "nspk", "x0", "y0", "z0", "dist", "z_lo", "z_hi", "y_lo", "y_hi",
         "dipoleB", "depth_shift", "resid", "one_flank", "edge_fit", "at_bound", "low_n", "high_resid", "reliable"]


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
    ap.add_argument("--probe", nargs="+", default=None,
                    help="NeuroSuite .probe YAML(s), in global-channel order "
                         "(default: the probe named in <session>.yaml)")
    ap.add_argument("--channels", default=None,
                    help="comma-separated global channel ids of this group (else read <session>.yaml)")
    ap.add_argument("--session", default=None, help="session for channel/probe lookup if --channels/--probe omitted")
    ap.add_argument("--clu", default=None, help="cluster file (pass the re-linked .clu)")
    ap.add_argument("--no-dipole", action="store_true")
    ap.add_argument("--min-n", type=int, default=50)
    ap.add_argument("--nboot", type=int, default=0,
                    help="spike bootstrap draws for depth/distance CIs; 0 (default) uses the analytic "
                         "Gaussian sigma (matches the bootstrap on isolated clusters, ~Nx cheaper).")
    ap.add_argument("--amp-method", choices=("pc1", "wave", "ptp"), default="pc1",
                    help="per-channel amplitude profile: pc1=rank-1 denoised template (default, sharpest "
                         "+ most precise); wave=median-waveform ptp; ptp=median per-spike ptp (legacy, "
                         "carries a ~4-sigma noise floor on far channels).")
    ap.add_argument("--max-resid", type=float, default=0.10)
    ap.add_argument("--amp-basis", choices=("auto", "pca", "fit", "none"), default="auto",
                    help="amplitude denoising basis: 'pca' = read .pca.standard.<elec> eigenvectors "
                         "(PC1 score per channel = the .fet amplitude); 'fit' = fit one group-wide "
                         "basis from .spk; 'auto' = pca if present else fit; 'none' = per-cluster SVD")
    ap.add_argument("--no-amp-basis", action="store_true",
                    help="alias for --amp-basis none (per-cluster SVD; unstable at low n)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    # resolve channels and/or probe from <session>.yaml when not given explicitly
    cfg = (sy.resolve_session_params(a.session or a.base, a.elec, require=())
           if (not a.channels or not a.probe) else None)
    if a.channels:
        channels = [int(c) for c in a.channels.split(",")]
    elif cfg and cfg.get("channels"):
        channels = cfg["channels"]
    else:
        raise SystemExit("[localize] need --channels or a <session>.yaml for the group's channel ids")
    probe = a.probe or (cfg.get("probe") if cfg else None)
    if not probe:
        raise SystemExit("[localize] no probe geometry: pass --probe or name it in "
                         "<session>.yaml (probeFile/probe/...)")
    xy = load_geometry(probe, channels)
    rows = localize(a.base, a.elec, a.nsamp, a.nchan, xy, a.clu,
                    dipole=not a.no_dipole, min_n=a.min_n, nboot=a.nboot, max_resid=a.max_resid,
                    amp_method=a.amp_method, amp_basis=("none" if a.no_amp_basis else a.amp_basis))
    out = a.out or f"{a.base}.localize.{a.elec}.tsv"
    write_localizations(rows, out)
    print(f"[localize] wrote {out}")


if __name__ == "__main__":
    main()
