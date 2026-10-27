#!/usr/bin/env python3
"""fil_noise_rhythms.py -- how does a group's spike-band noise vary with theta and gamma?

Streams one group's channels of the .fil in --win-s windows (default 1 s) and keeps the SPIKE-FREE samples
(at least --guard-ms outside every detected spike's window, as fil_noise_states.py does), in the waveform space
the pipeline sorts in (--variant).  Reads the LFP (<base>.lfp, else <base>.eeg; --lfp to name it), takes the
median over --lfp-channels (default the group's) as the reference, and band-passes it (zero-phase Butterworth,
Hilbert) for delta (1-4 Hz), theta (--theta-band) and each --gamma-bands band.  Four questions:

  STATE      per window: the noise level (log10 robust SD; channel mean and per channel) against the theta index
             (log10 theta/delta power) and each band's log power.  Theta windows (theta index in its top
             1 - --theta-q) against non-theta windows (bottom 1 - --theta-q); correlations, raw and partial given
             the detection rate -- the noise follows the population, so the question is whether the rhythm adds
             anything to that.
  PHASE      noise power per phase bin of the rhythm: theta in the theta windows, each gamma band in the samples
             whose band envelope is in its top 1 - --gamma-q (bursts), and again in its EXCLUSIVE bursts (every
             other band below its own threshold).  A strong band leaks into its neighbour's bursts, and its phase
             with it, so a modulation in the bursts that vanishes in the exclusive bursts belongs to the
             neighbour.  Each window's power is divided by its own mean, per channel, so slow level changes drop
             out.  Modulation m from h(phase) ~ 1 + m cos(phase - phase0), the preferred phase phase0, max/min,
             per channel and for the channel mean; the null rotates each window's phase histogram by a random
             offset (--shuffles), which keeps everything but the phase alignment across windows.  The detection
             rate per phase bin (spikes over the time spent in the bin) is the comparison: sub-threshold spiking
             should follow the detected spiking, at a smaller depth.
  AMPLITUDE  noise power (window-normalised, so this is the fast co-variation within seconds, not the slow one)
             and the detection rate per decile of each gamma band's instantaneous envelope; the top/bottom decile
             ratio with a 95% block-bootstrap interval (blocks of 60 windows).  Then a joint linear regression of
             the noise power on all the bands' envelopes (z-scored over the session): each band's UNIQUE slope (%
             noise power per SD, the other bands held fixed) beside its marginal one.
  COUPLING   each gamma band's envelope by theta phase in the theta windows (the LFP's own theta-gamma coupling),
             and the correlation between the bands' instantaneous envelopes, for reading the PHASE answers.

Phase convention: 0 deg = peak of the band-passed reference LFP, +-180 deg = its trough.  The reference's
polarity depends on the layer the --lfp-channels sit in; a pyramidal-layer channel puts CA1 pyramidal firing
near the theta trough.

Windows left out: flat ones (a channel constant over most of the window: robust SD below 1 file unit),
dropouts (a channel's robust SD below --drop-frac of its session median: a hardware fault), --exclude-minutes
spans, and windows the LFP does not cover.  The report says how many.

Caveat: fast gamma (~90-150 Hz) at the same site carries the spikes' own low-frequency energy, so a noise-fast
gamma relation measured with the group's own channels as the LFP reference is partly the spiking counted twice.
Give --lfp-channels from another shank, or another layer, to break that.

Example (neurosuite session layout, run in the session directory):

  python3 tools/fil_noise_rhythms.py sirotaA-jg-000005-20120312 7 --exclude-minutes 35.45-49.25 --out g7_rhythms
  python3 tools/fil_noise_rhythms.py sirotaA-jg-000005-20120312 7 --lfp-channels 20 --theta-channels 23 \
      --minutes 160-220 --out g7_rhythms_ref

Writes <out>.npz (profiles, nulls, per-window features), <out>.png, and prints the report.  A full session
streams the .fil once (as fil_noise_states.py); --minutes restricts the span, --every N analyses one window in N.
"""
import argparse
import contextlib
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
import fil_noise_states as fns    # noqa: E402  -- free_mask, robust_sd, read_lfp

DEFAULT_GAMMA = "slow:30-50,mid:55-90,fast:90-150"


def _log(*a):
    print(*a, file=sys.stderr, flush=True)


def _list(s):
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if part:
            lo, _, hi = part.partition("-")
            out.extend(range(int(lo), int(hi or lo) + 1))
    return out


def parse_band(s):
    lo, _, hi = str(s).partition("-")
    return float(lo), float(hi)


def parse_bands(s):
    """'slow:30-50,mid:55-90' -> [('slow', 30.0, 50.0), ('mid', 55.0, 90.0)]; a band without a name is named by its range."""
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        name, _, rng_ = part.rpartition(":")
        lo, hi = parse_band(rng_)
        if not 0 < lo < hi:
            raise SystemExit(f"fil_noise_rhythms: band '{part}' is not lo-hi with 0 < lo < hi")
        out.append((name or f"{lo:g}-{hi:g}", lo, hi))
    return out


def parse_spans(s):
    """'35.45-49.25,100-101' -> [(35.45, 49.25), (100.0, 101.0)] (minutes)."""
    out = []
    for part in str(s or "").split(","):
        part = part.strip()
        if part:
            a, _, b = part.partition("-")
            out.append((float(a), float(b)))
    return out


def read_lfp_median(path, ntotal, chans):
    """Median over `chans` of an int16 interleaved LFP file, as float32 (one value per LFP sample)."""
    mm = np.memmap(path, np.int16, "r")
    n = mm.size // ntotal
    mm = mm[:n * ntotal].reshape(n, ntotal)
    x = np.empty(n, np.float32)
    step = 750000
    for s in range(0, n, step):
        x[s:s + step] = np.median(np.asarray(mm[s:s + step][:, chans], np.float32), 1)
    return x


_SOS = {}


def analytic(x, lo, hi, fs):
    """Zero-phase band-pass (Butterworth, order 3) and Hilbert: the band's analytic signal."""
    from scipy.signal import butter, hilbert, sosfiltfilt
    hi = min(hi, 0.45 * fs)
    key = (lo, hi, fs)
    if key not in _SOS:
        _SOS[key] = butter(3, (lo, hi), "band", fs=fs, output="sos")
    return hilbert(sosfiltfilt(_SOS[key], x))


def at(A, u):
    """A complex block signal at fractional indices u (linear interpolation of the analytic signal)."""
    i0 = np.clip(np.floor(u).astype(np.int64), 0, A.size - 2)
    f = u - i0
    return A[i0] * (1.0 - f) + A[i0 + 1] * f


def phase_bins(z, nb):
    return ((np.angle(z) + np.pi) / (2 * np.pi) * nb).astype(np.int64) % nb


def modulation(h, centers):
    """(m, phase0 in degrees, max/min) of a phase profile h: h ~ 1 + m cos(phase - phase0)."""
    h = np.asarray(h, np.float64)
    if h.size == 0 or not np.isfinite(h).all() or h.sum() <= 0:
        return np.nan, np.nan, np.nan
    z = (h * np.exp(1j * centers)).sum() / h.sum()
    return 2.0 * abs(z), float(np.degrees(np.angle(z))), (float(h.max() / h.min()) if h.min() > 0 else np.inf)


def pooled(S, N):
    """Pool per-window sums S (n, nb, C) and counts N (n, nb) into a mean profile (nb, C)."""
    return S.sum(0) / np.maximum(N.sum(0), 1)[:, None]


def rotation_null(S, N, centers, R, rng):
    """Null of the modulation: each window's phase histogram rotated by its own random offset.
    Returns (m of the channel mean (R,), m per channel (R, C))."""
    n, nb = N.shape
    C = S.shape[2]
    ms = np.full(R, np.nan); mc = np.full((R, C), np.nan)
    if n == 0:
        return ms, mc
    base = np.arange(nb)
    for r in range(R):
        idx = (base[None, :] + rng.integers(0, nb, n)[:, None]) % nb
        h = pooled(np.take_along_axis(S, idx[:, :, None], 1), np.take_along_axis(N, idx, 1))
        ms[r] = modulation(h.mean(1), centers)[0]
        mc[r] = [modulation(h[:, c], centers)[0] for c in range(C)]
    return ms, mc


def rate_profile(K, O, sr):
    """Detections per second per bin: spikes over the samples spent in the bin."""
    k, o = K.sum(0).astype(np.float64), O.sum(0).astype(np.float64)
    return np.where(o > 0, k / np.maximum(o, 1) * sr, np.nan)


def rate_null(K, O, centers, R, rng, sr):
    n, nb = K.shape
    out = np.full(R, np.nan)
    if n == 0:
        return out
    base = np.arange(nb)
    for r in range(R):
        idx = (base[None, :] + rng.integers(0, nb, n)[:, None]) % nb
        out[r] = modulation(rate_profile(np.take_along_axis(K, idx, 1), np.take_along_axis(O, idx, 1), sr), centers)[0]
    return out


def pvalue(m, null):
    null = null[np.isfinite(null)]
    return (1.0 + np.sum(null >= m)) / (1.0 + null.size) if null.size and np.isfinite(m) else np.nan


def partial_corr(x, y, z):
    """corr(x, y) and corr(x, y | z) (z regressed out of both, linearly)."""
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if m.sum() < 10:
        return np.nan, np.nan
    x, y, z = x[m], y[m], z[m]
    r = float(np.corrcoef(x, y)[0, 1])
    A = np.column_stack([np.ones_like(z), z])
    rx = x - A @ np.linalg.lstsq(A, x, rcond=None)[0]
    ry = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    return r, float(np.corrcoef(rx, ry)[0, 1])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("base", help="session base (the <base>.yaml must exist, or pass --channels/--ntotal)")
    ap.add_argument("group", type=int, help="spike group (1-based, as the session yaml numbers them)")
    ap.add_argument("--variant", default="standard",
                    help="waveform space of the noise: standard (the .fil) or a stderiv token, e.g. stderiv_C5")
    ap.add_argument("--fil", default=None, help="filtered signal (default <base>.fil)")
    ap.add_argument("--res", default=None, help="spike times of the group (default: its .res, any variant)")
    ap.add_argument("--channels", default=None, help="override the group's channels, e.g. 48-55")
    ap.add_argument("--ntotal", type=int, default=None, help="override the file's channel count")
    ap.add_argument("--lfp", default=None, help="LFP file (default <base>.lfp, else <base>.eeg)")
    ap.add_argument("--lfp-sr", type=float, default=None, help="LFP sampling rate (default the yaml's lfpSamplingRate, else 1250)")
    ap.add_argument("--lfp-start-s", type=float, default=0.0, help="session time (s) of the LFP's first sample")
    ap.add_argument("--lfp-channels", default=None,
                    help="channels whose median is the gamma (and default theta) reference (default the group's)")
    ap.add_argument("--theta-channels", default=None, help="channels for the theta reference (default --lfp-channels)")
    ap.add_argument("--theta-band", default="5-12", help="theta band, Hz (default 5-12)")
    ap.add_argument("--delta-band", default="1-4", help="delta band for the theta index, Hz (default 1-4)")
    ap.add_argument("--gamma-bands", default=DEFAULT_GAMMA, help=f"name:lo-hi,... Hz (default {DEFAULT_GAMMA})")
    ap.add_argument("--minutes", default=None, help="span to analyse, e.g. 160-220 (default: the whole file)")
    ap.add_argument("--exclude-minutes", default=None, help="spans to leave out, e.g. 35.45-49.25 (a hardware fault)")
    ap.add_argument("--win-s", type=float, default=1.0, help="window length in seconds (default 1)")
    ap.add_argument("--every", type=int, default=1, help="analyse one window in N (default 1 = all)")
    ap.add_argument("--guard-ms", type=float, default=1.0,
                    help="spike-free = at least this far outside a detected spike's window (default 1 ms)")
    ap.add_argument("--drop-frac", type=float, default=0.6,
                    help="a window is a dropout when a channel's robust SD is below this share of its median (default 0.6)")
    ap.add_argument("--theta-q", type=float, default=0.7,
                    help="theta windows: theta index at or above this quantile; non-theta: at or below 1 - it (default 0.7)")
    ap.add_argument("--gamma-q", type=float, default=0.75,
                    help="gamma phase: samples whose band envelope is at or above this quantile (default 0.75)")
    ap.add_argument("--theta-bins", type=int, default=18, help="theta phase bins (default 18)")
    ap.add_argument("--gamma-bins", type=int, default=12, help="gamma phase bins (default 12)")
    ap.add_argument("--env-bins", type=int, default=10, help="envelope quantile bins (default 10)")
    ap.add_argument("--shuffles", type=int, default=200, help="rotations for the phase null (default 200)")
    ap.add_argument("--block-s", type=float, default=60.0, help="LFP processing block, s (default 60)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="output prefix (<out>.npz, <out>.png)")
    a = ap.parse_args(argv)

    from fiber_kit import neuro_io as nio, session_yaml as sy, fiber_stderiv as fst

    rng = np.random.default_rng(a.seed)
    t_start = time.time()
    sp = sy.resolve_session_params(a.base, a.group, channels=(_list(a.channels) if a.channels else None),
                                   ntotal=a.ntotal, require=("channels", "ntotal"))
    gch = np.asarray(sp.channels, np.int64); C = gch.size; sr = float(sp.sr)
    nsamp, peak = int(sp.nsamp or 32), sp.peak
    tf = fst.space_transform(a.variant, sp.sdiff_pairs)
    filp = a.fil or (sp.base + ".fil")
    fil = nio.open_signal(filp, int(sp.ntotal))
    T = fil.shape[0]
    spikes = np.sort(np.asarray(nio.read_res_file(a.res) if a.res else nio.read_res(sp.base, a.group), np.int64))
    g = int(round(a.guard_ms * 1e-3 * sr))
    pk = int(peak if peak is not None else nsamp // 2)
    pre, post = pk + g, nsamp - pk + g
    W = int(round(a.win_s * sr))

    # ── LFP ──
    lfpp = a.lfp
    if lfpp is None:
        for ext in (".lfp", ".eeg"):
            if os.path.exists(sp.base + ext):
                lfpp = sp.base + ext
                break
    if lfpp is None or not os.path.exists(lfpp):
        raise SystemExit(f"fil_noise_rhythms: no LFP ({sp.base}.lfp or .eeg); name it with --lfp")
    lfp_sr = a.lfp_sr
    if lfp_sr is None and sp.yaml:
        try:
            import yaml as _yaml
            with open(sp.yaml) as fh:
                lfp_sr = (_yaml.safe_load(fh).get("fieldPotentials") or {}).get("lfpSamplingRate")
        except Exception:                                            # noqa: BLE001 -- fall back to 1250
            lfp_sr = None
    lfp_sr = float(lfp_sr or 1250.0)
    lch = np.asarray(_list(a.lfp_channels), np.int64) if a.lfp_channels else gch
    tch = np.asarray(_list(a.theta_channels), np.int64) if a.theta_channels else lch
    lg = read_lfp_median(lfpp, int(sp.ntotal), lch)
    lt = lg if np.array_equal(tch, lch) else read_lfp_median(lfpp, int(sp.ntotal), tch)
    nl = lg.size
    th_band, de_band = parse_band(a.theta_band), parse_band(a.delta_band)
    bands = parse_bands(a.gamma_bands)
    G = len(bands)
    bnames = [b[0] for b in bands]

    # ── windows ──
    s_lo, s_hi = 1, T
    if a.minutes:
        m0, _, m1 = a.minutes.partition("-")
        s_lo = max(1, int(float(m0) * 60 * sr)); s_hi = min(T, int(float(m1) * 60 * sr))
    starts = np.arange(s_lo, s_hi - W + 1, W)[:: max(1, a.every)]
    nw = starts.size
    if nw < 20:
        raise SystemExit(f"fil_noise_rhythms: only {nw} windows of {a.win_s:g} s in the span")
    tw = starts / sr
    excl = np.zeros(nw, bool)
    for e0, e1 in parse_spans(a.exclude_minutes):
        excl |= (tw + a.win_s > e0 * 60) & (tw < e1 * 60)
    scale = lfp_sr / sr
    u0w = (starts / sr - a.lfp_start_s) * lfp_sr                       # LFP index of each window's first sample
    u1w = u0w + (W - 1) * scale
    lfp_ok = (u0w >= 0) & (np.ceil(u1w) + 1 < nl)
    pad = int(round(4.0 * lfp_sr))
    contiguous = bool((np.diff(gch) == 1).all())
    nbt, nbg, ne = a.theta_bins, a.gamma_bins, a.env_bins
    ct = -np.pi + (np.arange(nbt) + 0.5) * 2 * np.pi / nbt
    cg = -np.pi + (np.arange(nbg) + 0.5) * 2 * np.pi / nbg
    _log(f"fil_noise_rhythms: {filp} ({T / sr / 60:.1f} min, {sp.ntotal} ch), group {a.group} ch {gch.tolist()}, "
         f"variant {a.variant or 'standard'}; LFP {lfpp} at {lfp_sr:g} Hz ({nl / lfp_sr / 60:.1f} min), reference "
         f"median of ch {lch.tolist()}" + (f", theta from ch {tch.tolist()}" if lt is not lg else "")
         + f"; {nw} windows of {a.win_s:g} s; {spikes.size} detections")

    # blocks of windows that share one band-passed LFP segment
    blk = np.floor((tw - tw[0]) / a.block_s).astype(np.int64)
    blocks = [np.flatnonzero(blk == b) for b in np.unique(blk)]

    def block_signals(wins, gamma=True):
        b0 = max(0, int(np.floor(u0w[wins[0]])) - pad)
        b1 = min(nl, int(np.ceil(u1w[wins[-1]])) + 2 + pad)
        segt = np.asarray(lt[b0:b1], np.float64)
        segg = segt if lt is lg else np.asarray(lg[b0:b1], np.float64)
        TH = analytic(segt, th_band[0], th_band[1], lfp_sr)
        DE = analytic(segt, de_band[0], de_band[1], lfp_sr)
        GB = [analytic(segg, lo, hi, lfp_sr) for _, lo, hi in bands] if gamma else []
        return b0, TH, DE, GB

    # ── pass 1: the LFP alone -- theta index, band power per window, envelope quantiles ──
    theta_idx = np.full(nw, np.nan)
    bandpow = np.full((nw, G), np.nan)
    env_smp = [[] for _ in range(G)]
    for wins in blocks:
        wins = wins[lfp_ok[wins]]
        if wins.size == 0:
            continue
        b0, TH, DE, GB = block_signals(wins)
        for wi in wins:
            j0 = int(np.ceil(u0w[wi])) - b0; j1 = int(np.floor(u1w[wi])) - b0 + 1
            pt = np.mean(np.abs(TH[j0:j1]) ** 2); pd = np.mean(np.abs(DE[j0:j1]) ** 2)
            if pd > 0 and pt > 0:
                theta_idx[wi] = np.log10(pt / pd)
            envs = [np.abs(GB[k][j0:j1]) for k in range(G)]
            pgs = [float(np.mean(e ** 2)) for e in envs]
            bandpow[wi] = [np.log10(v) if v > 0 else np.nan for v in pgs]
            if not excl[wi] and min(pgs) > 0:
                for k in range(G):
                    env_smp[k].append(envs[k][::4].astype(np.float32))
    thr = np.full(G, np.nan); edges = np.full((G, ne - 1), np.nan)
    mu_e = np.zeros(G); sd_e = np.ones(G); env_corr = np.full((G, G), np.nan)
    if env_smp[0]:
        E = np.stack([np.concatenate(env_smp[k]) for k in range(G)])     # (G, n) aligned samples
        for k in range(G):
            thr[k] = np.quantile(E[k], a.gamma_q)
            edges[k] = np.quantile(E[k], np.arange(1, ne) / ne)
            mu_e[k] = float(E[k].mean()); sd_e[k] = float(E[k].std()) or 1.0
        env_corr = np.corrcoef(E) if G > 1 else np.ones((1, 1))
        del E
    del env_smp
    _log(f"  pass 1 (LFP) done in {time.time() - t_start:.0f} s")

    # ── pass 2: the .fil against the LFP ──
    rms_ch = np.full((nw, C), np.nan)
    mua = np.full(nw, np.nan)
    flat = np.zeros(nw, bool)
    S_th = np.zeros((nw, nbt, C), np.float32); N_th = np.zeros((nw, nbt), np.int32)
    O_th = np.zeros((nw, nbt), np.int32); K_th = np.zeros((nw, nbt), np.int32)
    S_g = np.zeros((G, nw, nbg, C), np.float32); N_g = np.zeros((G, nw, nbg), np.int32)
    O_g = np.zeros((G, nw, nbg), np.int32); K_g = np.zeros((G, nw, nbg), np.int32)
    # the same in EXCLUSIVE bursts: this band's envelope high, every other band's below its threshold
    S_x = np.zeros((G, nw, nbg, C), np.float32); N_x = np.zeros((G, nw, nbg), np.int32)
    O_x = np.zeros((G, nw, nbg), np.int32); K_x = np.zeros((G, nw, nbg), np.int32)
    S_e = np.zeros((G, nw, ne, C), np.float32); N_e = np.zeros((G, nw, ne), np.int32)
    O_e = np.zeros((G, nw, ne), np.int32); K_e = np.zeros((G, nw, ne), np.int32)
    # sufficient statistics of the joint regression of the noise power on every band's standardised envelope
    XtX = np.zeros((nw, G + 1, G + 1)); XtY = np.zeros((nw, G + 1, C))
    E_c = np.zeros((G, nw, nbt), np.float32); N_c = np.zeros((nw, nbt), np.int32)
    ar = np.arange(C)
    done = 0
    for wins in blocks:
        wins = wins[lfp_ok[wins] & ~excl[wins]]
        if wins.size == 0:
            continue
        b0, TH, DE, GB = block_signals(wins)
        for wi in wins:
            s0 = int(starts[wi])
            if contiguous:
                raw = np.asarray(fil[s0 - 1:s0 + W, gch[0]:gch[-1] + 1], np.float32)
            else:
                raw = np.asarray(fil[s0 - 1:s0 + W][:, gch], np.float32)
            x = raw[1:] if tf is None else np.asarray(tf(raw[None]), np.float32)[0]
            i0, i1 = np.searchsorted(spikes, s0), np.searchsorted(spikes, s0 + W)
            mua[wi] = (i1 - i0) / a.win_s
            free = fns.free_mask(W, s0, spikes, pre, post)
            if free.sum() < 0.05 * W:
                continue
            xf = x[free]
            r = fns.robust_sd(xf)
            if (np.ptp(xf, axis=0) == 0).any() or (r < 1.0).any():
                flat[wi] = True                  # a channel constant over (most of) the window: the dither's 0.37
                continue
            rms_ch[wi] = r
            p = (xf - np.median(xf, 0)) ** 2
            np.minimum(p, (8.0 * r) ** 2, out=p)
            pm = p.mean(0); pm[pm <= 0] = 1.0
            pn = p / pm
            u = u0w[wi] + np.arange(W) * scale - b0
            usp = (spikes[i0:i1] - s0) * scale + u0w[wi] - b0
            # theta phase
            bt = phase_bins(at(TH, u), nbt); btf = bt[free]
            S_th[wi] = np.bincount((btf[:, None] * C + ar).ravel(), weights=pn.ravel(), minlength=nbt * C).reshape(nbt, C)
            N_th[wi] = np.bincount(btf, minlength=nbt)
            O_th[wi] = np.bincount(bt, minlength=nbt)
            K_th[wi] = np.bincount(phase_bins(at(TH, usp), nbt), minlength=nbt)
            # LFP-resolution theta phase for the coupling
            j0 = int(np.ceil(u0w[wi])) - b0; j1 = int(np.floor(u1w[wi])) - b0 + 1
            btl = phase_bins(TH[j0:j1], nbt)
            N_c[wi] = np.bincount(btl, minlength=nbt)
            Z = [at(GB[k], u) for k in range(G)]
            ENV = [np.abs(z) for z in Z]
            ZS = [at(GB[k], usp) for k in range(G)]
            ENVS = [np.abs(z) for z in ZS]
            HI = [ENV[k] >= thr[k] for k in range(G)]
            HIS = [ENVS[k] >= thr[k] for k in range(G)]
            X = np.empty((int(free.sum()), G + 1)); X[:, 0] = 1.0
            for k in range(G):
                X[:, k + 1] = (ENV[k][free] - mu_e[k]) / sd_e[k]
            XtX[wi] = X.T @ X
            XtY[wi] = X.T @ pn
            for k in range(G):
                z, env, zs = Z[k], ENV[k], ZS[k]
                bg = phase_bins(z, nbg); bgf = bg[free]; bgs = phase_bins(zs, nbg)
                others = np.zeros(W, bool); others_s = np.zeros(usp.size, bool)
                for j in range(G):
                    if j != k:
                        others |= HI[j]; others_s |= HIS[j]
                # gamma phase in the bursts, all and exclusive
                for hi, his, S_, N_, O_, K_ in ((HI[k], HIS[k], S_g, N_g, O_g, K_g),
                                                (HI[k] & ~others, HIS[k] & ~others_s, S_x, N_x, O_x, K_x)):
                    hf = hi[free]
                    if hf.any():
                        S_[k, wi] = np.bincount((bgf[hf][:, None] * C + ar).ravel(), weights=pn[hf].ravel(),
                                                minlength=nbg * C).reshape(nbg, C)
                        N_[k, wi] = np.bincount(bgf[hf], minlength=nbg)
                    O_[k, wi] = np.bincount(bg[hi], minlength=nbg)
                    K_[k, wi] = np.bincount(bgs[his], minlength=nbg)
                envs = ENVS[k]
                # envelope quantile bins
                d = np.searchsorted(edges[k], env); df = d[free]
                S_e[k, wi] = np.bincount((df[:, None] * C + ar).ravel(), weights=pn.ravel(), minlength=ne * C).reshape(ne, C)
                N_e[k, wi] = np.bincount(df, minlength=ne)
                O_e[k, wi] = np.bincount(d, minlength=ne)
                K_e[k, wi] = np.bincount(np.searchsorted(edges[k], envs), minlength=ne)
                # the LFP's theta-gamma coupling
                el = np.abs(GB[k][j0:j1])
                if el.mean() > 0:
                    E_c[k, wi] = np.bincount(btl, weights=el / el.mean(), minlength=nbt)
            done += 1
            if done % 600 == 0:
                _log(f"  {done} windows, {time.time() - t_start:.0f} s")

    # ── selection ──
    lev_ch = np.log10(rms_ch)
    lev = np.nanmean(lev_ch, 1)
    base_ok = np.isfinite(lev) & ~flat & ~excl & lfp_ok
    med_ch = np.nanmedian(rms_ch[base_ok], 0) if base_ok.any() else np.full(C, np.nan)
    dropout = base_ok & (rms_ch < a.drop_frac * med_ch).any(1)
    valid = base_ok & ~dropout & np.isfinite(theta_idx)
    if valid.sum() < 20:
        raise SystemExit(f"fil_noise_rhythms: only {valid.sum()} usable windows")
    qt_hi = np.quantile(theta_idx[valid], a.theta_q); qt_lo = np.quantile(theta_idx[valid], 1.0 - a.theta_q)
    thw = valid & (theta_idx >= qt_hi)
    ntw = valid & (theta_idx <= qt_lo)
    lmua = np.log10(mua + 1.0)
    print(f"\n== {os.path.basename(sp.base)} group {a.group}, {a.variant or 'standard'}: {nw} windows of {a.win_s:g} s; "
          f"used {valid.sum()} ({valid.sum() * a.win_s / 60:.0f} min); left out: {int((flat & ~excl).sum())} flat, "
          f"{int(dropout.sum())} dropout (a channel below {a.drop_frac:.0%} of its median), {int(excl.sum())} excluded, "
          f"{int((~lfp_ok).sum())} outside the LFP")
    print(f"   reference LFP: median of ch {lch.tolist()}" + (f" (theta: ch {tch.tolist()})" if lt is not lg else "")
          + f"; theta {a.theta_band} Hz; " + ", ".join(f"{n} {lo:g}-{hi:g} Hz" for n, lo, hi in bands)
          + "; phase 0 deg = LFP peak, +-180 deg = trough")

    # ── 1. state ──
    print(f"\n1. STATE  theta windows = theta index >= {qt_hi:.2f} ({thw.sum()}), non-theta <= {qt_lo:.2f} ({ntw.sum()})")
    ratio_ch = 10 ** (np.nanmedian(lev_ch[thw], 0) - np.nanmedian(lev_ch[ntw], 0))
    print(f"   noise level theta / non-theta: x{10 ** (np.nanmedian(lev[thw]) - np.nanmedian(lev[ntw])):.3f}  (per channel "
          + " ".join(f"{v:.3f}" for v in ratio_ch) + f");  detection rate x{np.nanmedian(mua[thw]) / max(np.nanmedian(mua[ntw]), 1e-9):.2f}")
    r_th = partial_corr(lev[valid], theta_idx[valid], lmua[valid])
    print(f"   corr(level, theta index) {r_th[0]:+.2f}, given the detection rate {r_th[1]:+.2f};  "
          f"corr(level, log detection rate) {np.corrcoef(lev[valid], lmua[valid])[0, 1]:+.2f}")
    state_rows = []
    for k, n in enumerate(bnames):
        bp = bandpow[:, k]
        rr = partial_corr(lev[valid], bp[valid], lmua[valid])
        q1, q2 = np.quantile(bp[valid], [1 / 3, 2 / 3])
        tr = 10 ** (np.nanmedian(lev[valid & (bp >= q2)]) - np.nanmedian(lev[valid & (bp <= q1)]))
        state_rows.append((n, rr[0], rr[1], tr))
        print(f"   corr(level, {n} gamma power) {rr[0]:+.2f}, given the detection rate {rr[1]:+.2f};  noise level in the "
              f"top / bottom third of {n} power x{tr:.3f}")

    # ── 2. phase ──
    def phase_report(name, S, N, K, O, sel, centers):
        h = pooled(S[sel], N[sel])
        m, ph, mm = modulation(h.mean(1), centers)
        mc = np.array([modulation(h[:, c], centers)[0] for c in range(C)])
        pc = np.array([modulation(h[:, c], centers)[1] for c in range(C)])
        ns, nc = rotation_null(S[sel], N[sel], centers, a.shuffles, rng)
        rate = rate_profile(K[sel], O[sel], sr)
        mr, phr, mmr = modulation(rate, centers)
        nr = rate_null(K[sel], O[sel], centers, a.shuffles, rng, sr)
        p, pr = pvalue(m, ns), pvalue(mr, nr)
        print(f"   {name:14s} noise m {m:.3f} at {ph:+5.0f} deg (max/min x{mm:.3f}; null p95 {np.nanpercentile(ns, 95):.3f}, "
              f"p {p:.3f})   detections m {mr:.3f} at {phr:+5.0f} deg (x{mmr:.2f}; p {pr:.3f})")
        print(f"   {'':14s} per channel m " + " ".join(f"{v:.3f}" for v in mc) + "  at " + " ".join(f"{v:+.0f}" for v in pc)
              + f"  (null p95 per channel {np.nanmean(np.nanpercentile(nc, 95, 0)):.3f})")
        return dict(h=h, m=m, phase=ph, maxmin=mm, m_ch=mc, phase_ch=pc, null=ns, null_ch=nc, p=p,
                    rate=rate, m_rate=mr, phase_rate=phr, null_rate=nr, p_rate=pr, n_windows=int(sel.sum()))

    print("\n2. PHASE  (noise power per phase bin, each window normalised by its own mean; m from 1 + m cos(phase - phase0))")
    print("   gamma: in the band's bursts (its envelope in its top quantile), then in its EXCLUSIVE bursts (every other "
          "band below its own threshold): a strong neighbouring band leaks into a band's bursts, and its phase with it")
    PH = {"theta": phase_report("theta", S_th, N_th, K_th, O_th, thw, ct)}
    for k, n in enumerate(bnames):
        PH[n] = phase_report(f"{n} gamma", S_g[k], N_g[k], K_g[k], O_g[k], valid, cg)
        PH[f"{n}_excl"] = phase_report("  exclusive", S_x[k], N_x[k], K_x[k], O_x[k], valid, cg)

    # ── 3. amplitude ──
    print(f"\n3. AMPLITUDE  noise power and detection rate per {ne}-quantile of each gamma band's envelope "
          "(window-normalised noise: the fast co-variation)")
    AMP = {}
    vidx = np.flatnonzero(valid)
    blen = int(min(60, max(1, vidx.size // 20)))                   # bootstrap blocks: 60 windows, at least 20 blocks
    _, bl = np.unique(vidx // blen, return_inverse=True)
    nblk = int(bl.max()) + 1

    def ratio_ci(Sx, Nx):
        Sb = np.zeros((nblk,) + Sx.shape[1:]); Nb = np.zeros((nblk,) + Nx.shape[1:])
        np.add.at(Sb, bl, Sx); np.add.at(Nb, bl, Nx)
        boots = []
        for _ in range(200):
            pick = rng.integers(0, nblk, nblk)
            n_ = Nb[pick].sum(0)
            if n_[0] > 0 and n_[-1] > 0:
                hb = (Sb[pick].sum(0) / n_[:, None]).mean(1)
                boots.append(hb[-1] / hb[0])
        return tuple(np.percentile(boots, [2.5, 97.5])) if boots else (np.nan, np.nan)

    for k, n in enumerate(bnames):
        h = pooled(S_e[k][valid], N_e[k][valid])
        rate = rate_profile(K_e[k][valid], O_e[k][valid], sr)
        top_bot = h.mean(1)[-1] / h.mean(1)[0]
        lo95, hi95 = ratio_ci(S_e[k][vidx], N_e[k][vidx])
        AMP[n] = dict(h=h, rate=rate, top_bottom=top_bot, ci=(lo95, hi95), top_bottom_ch=h[-1] / h[0],
                      rate_top_bottom=rate[-1] / rate[0] if rate[0] > 0 else np.nan)
        print(f"   {n:5s} gamma: noise top/bottom x{top_bot:.3f} (95% {lo95:.3f}-{hi95:.3f}); per channel "
              + " ".join(f"{v:.3f}" for v in h[-1] / h[0]) + f";  detections x{AMP[n]['rate_top_bottom']:.2f}")
    # the joint regression: each band's slope with the other bands held fixed (neighbouring bands' envelopes
    # are correlated -- leakage, co-occurring bursts -- so the one-band curves above can share one effect)
    A_, B_ = XtX[valid].sum(0), XtY[valid].sum(0)
    beta = np.linalg.solve(A_, B_)                                   # (G+1, C): % per SD = 100 x beta
    XtXb = np.zeros((nblk, G + 1, G + 1)); XtYb = np.zeros((nblk, G + 1, C))
    np.add.at(XtXb, bl, XtX[vidx]); np.add.at(XtYb, bl, XtY[vidx])
    bb = []
    for _ in range(200):
        pick = rng.integers(0, nblk, nblk)
        with contextlib.suppress(np.linalg.LinAlgError):
            bb.append(np.linalg.solve(XtXb[pick].sum(0), XtYb[pick].sum(0))[1:].mean(1))
    bci = np.percentile(np.array(bb), [2.5, 97.5], axis=0) if bb else np.full((2, G), np.nan)
    print("   joint regression of the noise power on every band's envelope (z-scored over the session): % noise power "
          "per SD of the band, the other bands held fixed (unique), and alone (marginal)")
    for k, n in enumerate(bnames):
        idx = [0, k + 1]
        marg = np.linalg.solve(A_[np.ix_(idx, idx)], B_[idx])[1]
        AMP[n].update(unique_ch=100 * beta[k + 1], unique=100 * beta[k + 1].mean(), unique_ci=100 * bci[:, k],
                      marginal_ch=100 * marg, marginal=100 * marg.mean())
        print(f"   {n:5s} gamma: unique {100 * beta[k + 1].mean():+.2f}% (95% {100 * bci[0, k]:+.2f} to {100 * bci[1, k]:+.2f}),"
              f" marginal {100 * marg.mean():+.2f}%;  unique per channel " + " ".join(f"{100 * v:+.2f}" for v in beta[k + 1]))

    # ── 4. coupling ──
    print("\n4. COUPLING  gamma envelope by theta phase in the theta windows (the LFP's own theta-gamma coupling)")
    CP = {}
    for k, n in enumerate(bnames):
        hc = E_c[k][thw].sum(0) / np.maximum(N_c[thw].sum(0), 1)
        m, ph, mm = modulation(hc, ct)
        CP[n] = dict(h=hc, m=m, phase=ph, maxmin=mm)
        print(f"   {n:5s} gamma envelope: m {m:.3f} at {ph:+.0f} deg theta phase (max/min x{mm:.2f})")
    if G > 1:
        print("   correlation of the bands' instantaneous envelopes: " + ";  ".join(
            f"{bnames[i]}-{bnames[j]} {env_corr[i, j]:+.2f}" for i in range(G) for j in range(i + 1, G)))

    # ── outputs ──
    out = dict(t=tw, win_s=a.win_s, level_ch=lev_ch, mua=mua, theta_idx=theta_idx, bandpow=bandpow,
               band_names=np.array(bnames), bands=np.array([[lo, hi] for _, lo, hi in bands]),
               valid=valid, theta_win=thw, nontheta_win=ntw, flat=flat, dropout=dropout, excluded=excl,
               theta_centers=ct, gamma_centers=cg, env_thr=thr, env_edges=edges, env_corr=env_corr,
               ratio_ch=ratio_ch, state=np.array([[r1, r2, tr] for _, r1, r2, tr in state_rows]))
    for name, d in PH.items():
        for key in ("h", "m", "phase", "maxmin", "m_ch", "phase_ch", "null", "null_ch", "p", "rate", "m_rate",
                    "phase_rate", "null_rate", "p_rate"):
            out[f"phase_{name}_{key}"] = np.asarray(d[key])
    for name, d in AMP.items():
        for key in ("h", "rate", "top_bottom", "ci", "top_bottom_ch", "rate_top_bottom", "unique", "unique_ch",
                    "unique_ci", "marginal", "marginal_ch"):
            out[f"amp_{name}_{key}"] = np.asarray(d[key])
    for name, d in CP.items():
        for key in ("h", "m", "phase"):
            out[f"coupling_{name}_{key}"] = np.asarray(d[key])
    np.savez(a.out + ".npz", **out)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 3, figsize=(16, 9))
        deg_t, deg_g = np.degrees(ct), np.degrees(cg)
        d = PH["theta"]
        x0 = ax[0, 0]
        x0.plot(deg_t, d["h"].mean(1), "o-", color="k", label="noise power (channel mean)")
        for c in range(C):
            x0.plot(deg_t, d["h"][:, c], lw=0.8, alpha=0.5)
        x0.axhline(1, color="0.6", lw=0.8)
        x0.set_xlabel("theta phase (deg; 0 = LFP peak)"); x0.set_ylabel("noise power / window mean")
        x0.set_title(f"theta: noise m {d['m']:.3f} (p {d['p']:.3f}); detections m {d['m_rate']:.3f}")
        x0b = x0.twinx(); x0b.plot(deg_t, d["rate"], "--", color="tab:red"); x0b.set_ylabel("detections / s", color="tab:red")
        ax[0, 1].bar(np.arange(C), d["m_ch"], color="0.5")
        ax[0, 1].plot(np.arange(C), np.nanpercentile(d["null_ch"], 95, 0), "_", color="tab:red", ms=20, mew=2,
                      label="null 95%")
        ax[0, 1].set_xlabel("channel (group order)"); ax[0, 1].set_ylabel("theta modulation m of the noise")
        ax[0, 1].legend(); ax[0, 1].set_title("theta modulation per channel")
        cols = plt.cm.viridis(np.linspace(0, 0.85, max(G, 1)))
        for k, n in enumerate(bnames):
            dk, dx = PH[n], PH[f"{n}_excl"]
            ax[0, 2].plot(deg_g, dk["h"].mean(1), "o-", color=cols[k], label=f"{n}: m {dk['m']:.3f} (p {dk['p']:.3f})")
            ax[0, 2].plot(deg_g, dx["h"].mean(1), "--", color=cols[k], lw=1,
                          label=f"{n} exclusive: m {dx['m']:.3f} (p {dx['p']:.3f})")
        ax[0, 2].axhline(1, color="0.6", lw=0.8); ax[0, 2].legend(fontsize=7)
        ax[0, 2].set_xlabel("gamma phase (deg; 0 = LFP peak), bursts only"); ax[0, 2].set_ylabel("noise power / window mean")
        ax[0, 2].set_title("noise by gamma phase (dashed: exclusive bursts)")
        for k, n in enumerate(bnames):
            dk = AMP[n]
            ax[1, 0].plot(np.arange(1, ne + 1), dk["h"].mean(1), "o-", color=cols[k],
                          label=f"{n} noise (unique {dk['unique']:+.1f}%/SD)")
            ax[1, 0].plot(np.arange(1, ne + 1), dk["rate"] / np.nanmean(dk["rate"]), "--", color=cols[k], lw=0.8)
        ax[1, 0].axhline(1, color="0.6", lw=0.8); ax[1, 0].legend(fontsize=8)
        ax[1, 0].set_xlabel("gamma envelope quantile bin"); ax[1, 0].set_ylabel("noise power (solid), detection rate (dashed), / mean")
        ax[1, 0].set_title("noise by gamma amplitude")
        ax[1, 1].bar(np.arange(C), ratio_ch, color="tab:blue")
        ax[1, 1].axhline(1, color="0.6", lw=0.8); ax[1, 1].set_ylim(min(0.9, np.nanmin(ratio_ch) - 0.02), max(1.1, np.nanmax(ratio_ch) + 0.02))
        ax[1, 1].set_xlabel("channel (group order)"); ax[1, 1].set_ylabel("noise level, theta / non-theta windows")
        ax[1, 1].set_title("theta state")
        for k, n in enumerate(bnames):
            ax[1, 2].plot(deg_t, CP[n]["h"], "o-", color=cols[k], label=f"{n}: m {CP[n]['m']:.3f}")
        ax[1, 2].axhline(1, color="0.6", lw=0.8); ax[1, 2].legend(fontsize=8)
        ax[1, 2].set_xlabel("theta phase (deg)"); ax[1, 2].set_ylabel("gamma envelope / window mean")
        ax[1, 2].set_title("LFP theta-gamma coupling")
        fig.tight_layout(); fig.savefig(a.out + ".png", dpi=110)
    except Exception as e:                                           # noqa: BLE001 -- the numbers are written already
        _log(f"  figure skipped ({e})")
    _log(f"wrote {a.out}.npz, {a.out}.png in {time.time() - t_start:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
