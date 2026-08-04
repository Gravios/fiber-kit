#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════════════════════
#  morpho_eap.py — transmembrane current -> extracellular waveform on a probe.
#
#  Line-source approximation (Holt & Koch 1999, J Comput Neurosci 6:169) in a
#  homogeneous isotropic volume conductor.  Each compartment is a line of
#  uniform current density rather than a point: at the distances that matter
#  here (a Buzsaki-style site sits 10-60 um from the soma of a cell it records)
#  a 100 um apical trunk segment is NOT far-field, and a point-source collapse
#  of it misplaces the dendritic return current by tens of microns.
#
#  The transfer from currents to sites is a TIME-INDEPENDENT matrix K (ncomp,
#  nsite): V_e(t) = I_m(t) @ K.  So one simulation supports arbitrarily many
#  probe placements, rotations and depths for the cost of a matmul, which is
#  what makes the position-vs-morphology variance decomposition affordable --
#  and is exact, since the extracellular field is linear in the currents and
#  the (passive, non-invasive) electrode does not load the source.
#
#  Sign convention: I_m is outward-positive (a sink, i.e. Na influx, is
#  negative), so a somatic spike gives the familiar negative extracellular
#  deflection without any post-hoc sign flip.
# ════════════════════════════════════════════════════════════════════════════
import numpy as np

SIGMA_DEFAULT = 0.3          # S/m, rat cortex/hippocampus ~0.3 (Logothetis 2007)
K_UV = 1e3                   # I[nA], r[um], sigma[S/m] -> V[uV]:  k/(4 pi sigma r)


# ── probe geometry ──────────────────────────────────────────────────────────
def staggered_octrode(pitch_y=21.65, offset_x=12.5, n=8):
    """FALLBACK geometry only: an 8-site staggered column, +y = depth.

    Defaults put site k at y = 21.65*k um in two columns 25 um apart, i.e. the
    equilateral zig-zag a Buzsaki-style shank uses; site 3 then lands at 65 um,
    which is the spacing the reference session's group 5 shows.  This is a
    STAND-IN.  Real work must pass the session's .probe file: site layout is
    recorded there precisely because it cannot be reconstructed from channel
    index arithmetic, and a wrong layout biases every distance-dependent
    quantity computed downstream.
    """
    k = np.arange(n)
    return np.stack([np.where(k % 2, offset_x, -offset_x), k * pitch_y], 1).astype(float)


def load_probe(paths, channels):
    """Site xy (nchan, 2) in um for a spike group, from NeuroSuite .probe YAML.

    Delegates to fiber_localize.load_geometry so that a simulated footprint and
    a recorded one are indexed by the same channel order; a second, independent
    reader here would be exactly the duplicated-policy failure this codebase
    keeps hitting.
    """
    try:
        from . import fiber_localize as floc
    except ImportError:
        import fiber_localize as floc
    return floc.load_geometry(paths, channels)


def sites_3d(xy, z=0.0):
    xy = np.asarray(xy, float)
    return np.column_stack([xy[:, 0], xy[:, 1], np.full(len(xy), float(z))])


# ── forward model ───────────────────────────────────────────────────────────
def transfer_matrix(cmp_, sites, sigma=SIGMA_DEFAULT, point_source=False,
                    r_min=1.0):
    """(ncomp, nsite) uV per nA.  r_min (um) keeps a site from landing on a
    neurite: the potential there is unbounded in the model but bounded in
    reality, and a single compartment at r->0 would otherwise dominate the whole
    footprint."""
    p0 = np.asarray(cmp_.p0, float); p1 = np.asarray(cmp_.p1, float)
    r = np.asarray(sites, float)
    k = K_UV / (4.0 * np.pi * sigma)
    rad = np.maximum(cmp_.diam / 2.0, r_min)

    if point_source:
        d = np.linalg.norm(cmp_.mid[:, None, :] - r[None, :, :], axis=2)
        return k / np.maximum(d, rad[:, None])

    ds = np.linalg.norm(p1 - p0, axis=1)
    good = ds > 1e-9
    u = np.zeros_like(p1); u[good] = (p1[good] - p0[good]) / ds[good, None]
    w0 = r[None, :, :] - p0[:, None, :]
    w1 = r[None, :, :] - p1[:, None, :]
    t0 = np.einsum("nsk,nk->ns", w0, u)                    # along-axis, from p0
    h = np.einsum("nsk,nk->ns", w1, u)                     # along-axis, from p1
    l = h + ds[:, None]
    rho2 = np.maximum(np.einsum("nsk,nsk->ns", w0, w0) - t0 ** 2, 0.0)
    rho = np.maximum(np.sqrt(rho2), rad[:, None])
    num = np.sqrt(h ** 2 + rho ** 2) - h
    den = np.sqrt(l ** 2 + rho ** 2) - l
    with np.errstate(divide="ignore", invalid="ignore"):
        K = k / np.maximum(ds[:, None], 1e-9) * np.log(np.abs(num / den))
    # degenerate (zero-length) compartments fall back to a point source
    if (~good).any():
        d = np.linalg.norm(cmp_.mid[~good][:, None, :] - r[None, :, :], axis=2)
        K[~good] = k / np.maximum(d, rad[~good, None])
    return np.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0)


def extracellular(im, K):
    """(nt, nsite) uV from (nt, ncomp) nA and the transfer matrix."""
    return np.asarray(im, float) @ K


# ── acquisition chain ───────────────────────────────────────────────────────
def bandpass(x, dt_ms, lo=300.0, hi=6000.0, order=3):
    """Zero-phase Butterworth, applied along time (axis 0).

    Zero-phase, not causal: the pipeline this feeds compares waveform SHAPE
    across units, and a causal filter's phase distortion is common to every
    unit, so removing it costs nothing and keeps the simulated peak time equal
    to the true peak time.  Match the acquisition filter if absolute latency
    matters.
    """
    from scipy.signal import butter, filtfilt
    fs = 1e3 / dt_ms
    ny = fs / 2.0
    hi = min(hi, 0.45 * fs)
    if lo <= 0:
        b, a = butter(order, hi / ny, btype="low")
    else:
        b, a = butter(order, [lo / ny, hi / ny], btype="band")
    n = x.shape[0]
    pad = min(3 * max(len(a), len(b)), n - 1)
    return filtfilt(b, a, x, axis=0, padlen=pad)


def resample(x, dt_ms, sr_out):
    """Cubic-spline resample from the solver grid to the acquisition rate.

    The signal is already band-limited well below the output Nyquist by
    bandpass(), so interpolation is adequate and avoids the rational-ratio
    approximation an FIR resampler would need for 100 kHz -> 32552 Hz.
    """
    from scipy.interpolate import CubicSpline
    t = np.arange(x.shape[0]) * dt_ms
    t_out = np.arange(0.0, t[-1], 1e3 / sr_out)
    return CubicSpline(t, x, axis=0)(t_out), t_out


def extract_window(x, nsamp=42, peak=21, polarity="neg"):
    """Cut an nsamp window aligned like a NeuroSuite .spk: the extremum of the
    dominant channel sits at index `peak`.

    Returns (nsamp, nchan) or None if the window would run off either end --
    returning a zero-padded window instead would silently create a waveform
    whose energy and shape are artifacts of the cut.
    """
    amp = x.max(0) - x.min(0)
    ch = int(np.argmax(amp))
    k = int(np.argmin(x[:, ch]) if polarity == "neg" else np.argmax(x[:, ch]))
    a = k - peak
    b = a + nsamp
    if a < 0 or b > x.shape[0]:
        return None, ch
    return x[a:b], ch


def waveform(im, cmp_, sites, dt_ms, sr=32552.0, nsamp=42, peak=21,
             sigma=SIGMA_DEFAULT, lo=300.0, hi=6000.0, point_source=False):
    """Full chain: currents -> uV at sites -> band-pass -> resample -> window."""
    K = transfer_matrix(cmp_, sites, sigma=sigma, point_source=point_source)
    ve = extracellular(im, K)
    vf = bandpass(ve, dt_ms, lo, hi)
    vr, _ = resample(vf, dt_ms, sr)
    w, ch = extract_window(vr, nsamp, peak)
    return dict(raw=ve, filt=vf, resampled=vr, wave=w, peak_chan=ch, K=K)


# ── footprint descriptors ───────────────────────────────────────────────────
def metrics(w, xy, sr=32552.0):
    """Shape/spatial descriptors of one footprint w (nsamp, nchan).

    p2p: per-channel peak-to-peak (uV).  width_ms: trough-to-peak time on the
    dominant channel -- the standard narrow/broad axis used to separate putative
    interneurons from pyramidal cells.  decay_um: length constant of an
    exponential fit to p2p vs site distance from the peak site; reported only
    as a shape summary, NOT as a localization (the monopole fit in
    fiber_localize does that, and this quantity is deliberately not fed back
    into it).  spread_um: distance over which p2p stays above half of maximum.
    """
    w = np.asarray(w, float)
    p2p = w.max(0) - w.min(0)
    ch = int(np.argmax(p2p))
    tr = w[:, ch]
    i_min = int(np.argmin(tr))
    after = tr[i_min:]
    i_max = i_min + (int(np.argmax(after)) if after.size else 0)
    width = (i_max - i_min) / sr * 1e3
    xy = np.asarray(xy, float)
    d = np.linalg.norm(xy - xy[ch], axis=1)
    o = np.argsort(d)
    y = p2p[o]; x = d[o]
    m = y > 0
    decay = np.nan
    if m.sum() >= 3:
        A = np.polyfit(x[m], np.log(y[m]), 1)
        decay = float(-1.0 / A[0]) if A[0] < 0 else np.inf
    half = p2p >= 0.5 * p2p[ch]
    spread = float(d[half].max()) if half.any() else 0.0
    return dict(p2p=p2p, peak_chan=ch, amp=float(p2p[ch]), width_ms=float(width),
                decay_um=decay, spread_um=spread,
                trough_uV=float(tr.min()), peak_uV=float(tr.max()),
                asym=float((tr.max() + tr.min()) / max(abs(tr.min()), 1e-9)))


def normalize(w):
    """Unit-Frobenius footprint -- the space in which fiber-kit compares
    templates, so shape differences are not confounded with distance."""
    w = np.asarray(w, float)
    n = np.linalg.norm(w)
    return w / n if n > 0 else w


def cosine(a, b):
    a, b = normalize(a).ravel(), normalize(b).ravel()
    return float(a @ b)
