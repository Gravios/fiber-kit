#!/usr/bin/env python3
"""fil_noise_states.py -- is a group's spike-band noise stationary, or does it come in states?

Streams one group's channels of the .fil in --win-s windows (default 1 s) and keeps the SPIKE-FREE
samples: those at least --guard-ms outside the window of every spike the group's .res detected.  It takes
them into the waveform space the pipeline sorts in (--variant: standard = the .fil itself; stderiv,
stderiv_S<n>, stderiv_C<n> = fiber_stderiv.space_transform with the session's sdiffPairs), and measures
per window:

  level      log10 of the robust noise SD (1.4826 x MAD), mean over channels
  tails      log10(plain SD / robust SD): heavy tails = sub-threshold (undetected) spikes in the background
  cm         common share: the fraction of a channel's noise variance that all channels share,
             (C x var(channel mean) / mean channel variance - 1) / (C - 1); 0 = independent, 1 = all common,
             below 0 = channels anti-correlated (a spatial derivative does that)
  tilt       log10(power 4-8 kHz / power 0.3-1 kHz): spectral shape
  bands      power in 0.3-1, 1-2, 2-4, 4-8 and 8+ kHz
  mua        the group's detections per second
  theta, ripple   with an LFP (<base>.lfp unless --lfp / --no-lfp): theta index log10(P 5-12 Hz / P 1-4 Hz)
             and log10 ripple-band (150-250 Hz) power of the window

Then four questions:

  1. STRUCTURE.  How much of the window-to-window variation of the level is real?  Each window's level is
     also measured on its even and odd 25-ms sub-blocks; their disagreement is the measurement noise of
     the window's level, so the real share = 1 - measurement variance / total variance.  Autocorrelation
     of the level at 1 s .. 10 min (a stationary background gives ~0 beyond one window).
  2. STATES.  A Gaussian mixture on the standardised (level, tails, cm, tilt) of the windows, K = 1 ..
     --max-states chosen by BIC.  Per state: occupancy, the features, the MUA rate and theta index, dwell
     times (runs of consecutive windows), and the real share WITHIN the state.  A small state whose runs
     are single windows is marked 'transient': windows that straddle a change of state, or hold a short
     event (a ripple), not a state of its own.
  3. FAST STRUCTURE (needs the LFP).  Noise power per theta-phase bin in theta windows (theta index above
     its --theta-q quantile); the noise level in 10-ms bins around ripple events (ripple-band envelope
     above mean + --ripple-sd SD) against random times.
  4. PIPELINE IMPACT.  The noise covariance a whitener would use -- block-Toeplitz over the session mask's
     L samples (fiber_lib.build_masks), lags 0..L-1 measured on spike-free sample pairs -- for the whole
     span, each state and each 12-min chunk (the pipeline's whitener granularity).  Under ONE whitener
     built from the whole span, each state's noise has covariance C_s^-1/2 C_state C_s^-1/2: all its
     eigenvalues 1 = the session whitener fits; a common factor = a level change only (radii, not
     directions); a spread = a change of shape.

Writes <out>.npz (per-window features, states, covariances), <out>.png, and prints the report.

Example (neurosuite session layout, run in the session directory):

  python3 tools/fil_noise_states.py sirotaA-jg-000005-20120312 7 --variant stderiv_C5 --out g7_noise
  python3 tools/fil_noise_states.py sirotaA-jg-000005-20120312 7 --variant standard --minutes 160-220 --out g7_std

A full session streams the whole .fil once (every channel's bytes pass the disk; ~10-20 min for 3 h);
--minutes restricts the span, --every N analyses one window in N.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

BANDS = ((300.0, 1000.0), (1000.0, 2000.0), (2000.0, 4000.0), (4000.0, 8000.0), (8000.0, None))
FEATS = ("level", "tails", "cm", "tilt")
SEG = 128                                   # samples per spectral segment
LAGS_S = (1, 2, 5, 10, 30, 60, 120, 300, 600)


def _log(*a):
    print(*a, file=sys.stderr, flush=True)


def free_mask(n, s0, spikes, pre, post):
    """Boolean (n,): sample s0 + i lies outside [t - pre, t + post] of every spike t."""
    i0 = np.searchsorted(spikes, s0 - post, "left")
    i1 = np.searchsorted(spikes, s0 + n + pre, "right")
    d = np.zeros(n + 1, np.int32)
    if i1 > i0:
        t = spikes[i0:i1] - s0
        a = np.clip(t - pre, 0, n); b = np.clip(t + post + 1, 0, n)
        np.add.at(d, a, 1); np.add.at(d, b, -1)
    return np.cumsum(d[:-1]) == 0


_DITHER = np.random.default_rng(12345)


def robust_sd(x):
    """1.4826 x MAD per column of x (n, C).  The .fil and the stderiv spaces are integer (int16), so a MAD of
    them takes only integer or half-integer values: a quantised level that a mixture model fits as
    near-zero-variance spikes (spurious states).  A uniform +-0.5 dither (variance 1/12 of a unit) makes the
    estimate continuous."""
    x = np.asarray(x, np.float64) + (_DITHER.random(np.shape(x)) - 0.5)
    med = np.median(x, 0)
    return 1.4826 * np.median(np.abs(x - med), 0)


def runs_of(mask):
    """(start, stop) of the True runs of a boolean array."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def window_spectrum(x, free, sr, max_seg=64):
    """Mean power spectrum (over channels and up to max_seg spike-free SEG-sample segments), Hann."""
    st, en = runs_of(free)
    segs = []
    for a, b in zip(st, en):
        for s in range(a, b - SEG + 1, SEG):
            segs.append(s)
            if len(segs) >= max_seg:
                break
        if len(segs) >= max_seg:
            break
    if not segs:
        return None
    w = np.hanning(SEG)
    X = np.stack([x[s:s + SEG] for s in segs])                  # (k, SEG, C)
    X = X - X.mean(1, keepdims=True)
    P = np.abs(np.fft.rfft(X * w[None, :, None], axis=1)) ** 2   # (k, F, C)
    return P.mean((0, 2))


def band_powers(psd, sr):
    f = np.fft.rfftfreq(SEG, 1.0 / sr)
    out = []
    for lo, hi in BANDS:
        hi = sr / 2.0 if hi is None else min(hi, sr / 2.0)
        m = (f >= lo) & (f < hi)
        out.append(float(psd[m].mean()) if m.any() else np.nan)
    return np.array(out)


def toeplitz_from_lags(lagc):
    """Block-Toeplitz covariance (L*C, L*C), sample-major (t*C + c), from lag covariances (L, C, C)."""
    L, C, _ = lagc.shape
    Cf = np.empty((L * C, L * C))
    for i in range(L):
        for j in range(L):
            d = j - i
            Cf[i * C:(i + 1) * C, j * C:(j + 1) * C] = lagc[d] if d >= 0 else lagc[-d].T
    return 0.5 * (Cf + Cf.T)


def inv_sqrt(C, floor=1e-6):
    ev, V = np.linalg.eigh(C)
    ev = np.maximum(ev, floor * max(float(ev.max()), 1e-30))
    return V @ np.diag(1.0 / np.sqrt(ev)) @ V.T


def mismatch(Cref, C):
    """Eigenvalues of Cref^-1/2 C Cref^-1/2: (geometric mean, min, max, spread = exp(SD of log))."""
    W = inv_sqrt(Cref)
    lam = np.clip(np.linalg.eigvalsh(W @ C @ W), 1e-12, None)
    ll = np.log(lam)
    return float(np.exp(ll.mean())), float(lam.min()), float(lam.max()), float(np.exp(ll.std())), lam


def acf(y, lags):
    y = np.asarray(y, float)
    y = np.where(np.isfinite(y), y, np.nanmean(y)) - np.nanmean(y)
    d = float((y * y).sum())
    return np.array([float((y[:-L] * y[L:]).sum()) / d if 0 < L < y.size else np.nan for L in lags])


def read_lfp(path, ntotal, chans, lfp_sr, start_s):
    """Median over `chans` of an int16 interleaved LFP file, float32, with its time axis (s)."""
    mm = np.memmap(path, np.int16, "r").reshape(-1, ntotal)
    x = np.empty(mm.shape[0], np.float32)
    step = int(lfp_sr * 600)
    for a in range(0, mm.shape[0], step):
        x[a:a + step] = np.median(np.asarray(mm[a:a + step][:, chans], np.float32), 1)
    return x, start_s + np.arange(x.size) / lfp_sr


def lfp_products(x, lfp_sr, ripple_sd, max_ripples, rng):
    """Theta phase (5-12 Hz, Hilbert), ripple-band envelope and ripple event times (s, LFP clock)."""
    from scipy.signal import butter, sosfiltfilt, hilbert
    th = sosfiltfilt(butter(3, (5.0, 12.0), "band", fs=lfp_sr, output="sos"), x)
    phase = np.angle(hilbert(th)).astype(np.float32)
    hi = min(250.0, 0.45 * lfp_sr)
    rp = sosfiltfilt(butter(3, (150.0, hi), "band", fs=lfp_sr, output="sos"), x)
    env = np.abs(hilbert(rp))
    k = max(1, int(round(0.010 * lfp_sr)))
    env = np.convolve(env, np.ones(k) / k, "same")
    thr = env.mean() + ripple_sd * env.std()
    above = env > thr
    st, en = runs_of(above)
    peaks = np.array([a + int(np.argmax(env[a:b])) for a, b in zip(st, en)], np.int64)
    if peaks.size:
        keep = [peaks[0]]
        for p in peaks[1:]:
            if p - keep[-1] >= int(0.1 * lfp_sr):
                keep.append(p)
        peaks = np.array(keep)
    if peaks.size > max_ripples:
        peaks = np.sort(rng.choice(peaks, max_ripples, replace=False))
    return phase, env, peaks


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
    ap.add_argument("--minutes", default=None, help="span to analyse, e.g. 160-220 (default: the whole file)")
    ap.add_argument("--win-s", type=float, default=1.0, help="window length in seconds (default 1)")
    ap.add_argument("--every", type=int, default=1, help="analyse one window in N (default 1 = all)")
    ap.add_argument("--guard-ms", type=float, default=1.0,
                    help="spike-free = at least this far outside a detected spike's window (default 1 ms)")
    ap.add_argument("--max-states", type=int, default=4, help="largest number of states tried (default 4)")
    ap.add_argument("--chunk-min", type=float, default=12.0, help="whitener chunk length for part 4 (default 12)")
    ap.add_argument("--lfp", default=None, help="LFP file (default <base>.lfp when it exists)")
    ap.add_argument("--no-lfp", action="store_true", help="skip the LFP even if <base>.lfp exists")
    ap.add_argument("--lfp-sr", type=float, default=None, help="LFP sampling rate (default the yaml's lfpSamplingRate, else 1250)")
    ap.add_argument("--lfp-channels", default=None, help="LFP channels for the median trace (default the group's)")
    ap.add_argument("--lfp-start-s", type=float, default=0.0, help="session time (s) of the LFP's first sample")
    ap.add_argument("--theta-q", type=float, default=0.7,
                    help="theta windows for the phase analysis: theta index above this quantile (default 0.7)")
    ap.add_argument("--ripple-sd", type=float, default=5.0, help="ripple threshold, SD above the mean envelope (default 5)")
    ap.add_argument("--max-ripples", type=int, default=300, help="ripple events used for the peri-event profile")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="output prefix (<out>.npz, <out>.png)")
    a = ap.parse_args(argv)

    from fiber_kit import neuro_io as nio, session_yaml as sy, fiber_stderiv as fst, fiber_lib as fl

    def _list(s):
        out = []
        for part in str(s).split(","):
            part = part.strip()
            if part:
                lo, _, hi = part.partition("-")
                out.extend(range(int(lo), int(hi or lo) + 1))
        return out

    rng = np.random.default_rng(a.seed)
    sp = sy.resolve_session_params(a.base, a.group, channels=(_list(a.channels) if a.channels else None),
                                   ntotal=a.ntotal, require=("channels", "ntotal"))
    gch = np.asarray(sp.channels, np.int64); C = gch.size; sr = float(sp.sr)
    nsamp, peak = int(sp.nsamp or 32), sp.peak
    tf = fst.space_transform(a.variant, sp.sdiff_pairs)
    mask_t = fl.build_masks(nsamp, peak).full
    L = int(mask_t.size)
    filp = a.fil or (sp.base + ".fil")
    fil = nio.open_signal(filp, int(sp.ntotal))
    T = fil.shape[0]
    if a.res:
        spikes = nio.read_res_file(a.res).astype(np.int64)
    else:
        spikes = np.asarray(nio.read_res(sp.base, a.group), np.int64)
    spikes = np.sort(spikes)
    g = int(round(a.guard_ms * 1e-3 * sr))
    pk = int(peak if peak is not None else nsamp // 2)
    pre, post = pk + g, nsamp - pk + g
    W = int(round(a.win_s * sr))
    s_lo, s_hi = 1, T
    if a.minutes:
        m0, _, m1 = a.minutes.partition("-")
        s_lo = max(1, int(float(m0) * 60 * sr)); s_hi = min(T, int(float(m1) * 60 * sr))
    starts = np.arange(s_lo, s_hi - W + 1, W)[:: max(1, a.every)]
    if starts.size < 10:
        raise SystemExit(f"fil_noise_states: only {starts.size} windows of {a.win_s:g} s in the span; widen "
                         "--minutes or shorten --win-s")
    contiguous = bool((np.diff(gch) == 1).all())
    blk = max(1, int(round(0.025 * sr)))
    _log(f"fil_noise_states: {filp} ({T / sr / 60:.1f} min, {sp.ntotal} ch), group {a.group} ch {gch.tolist()}, "
         f"variant {a.variant or 'standard'}, {starts.size} windows of {a.win_s:g} s, guard {a.guard_ms:g} ms, "
         f"{spikes.size} detections")

    # ── LFP (optional) ──
    lfp_on = False; lfp = None
    lfpp = a.lfp or (sp.base + ".lfp")
    if not a.no_lfp and os.path.exists(lfpp):
        lfp_sr = a.lfp_sr
        if lfp_sr is None and sp.yaml:
            try:
                import yaml as _yaml
                with open(sp.yaml) as fh:
                    lfp_sr = (_yaml.safe_load(fh).get("fieldPotentials") or {}).get("lfpSamplingRate")
            except Exception:                                        # noqa: BLE001 -- fall back to 1250
                lfp_sr = None
        lfp_sr = float(lfp_sr or 1250.0)
        try:
            lch = np.asarray(_list(a.lfp_channels), np.int64) if a.lfp_channels else gch
            lx, lt = read_lfp(lfpp, int(sp.ntotal), lch, lfp_sr, a.lfp_start_s)
            phase, env, rip = lfp_products(lx, lfp_sr, a.ripple_sd, a.max_ripples, rng)
            lfp = dict(x=lx, t0=a.lfp_start_s, sr=lfp_sr, phase=phase, env=env, rip=rip)
            lfp_on = True
            _log(f"  LFP {lfpp}: {lx.size / lfp_sr / 60:.1f} min at {lfp_sr:g} Hz, {rip.size} ripple events kept")
        except Exception as e:                                       # noqa: BLE001 -- reported, the rest runs
            _log(f"  LFP {lfpp} unusable ({e}); continuing without it")
    nb = 8                                                           # theta phase bins
    nw = starts.size
    F = {k: np.full(nw, np.nan) for k in ("level", "tails", "cm", "tilt", "mua", "free", "lev_even", "lev_odd",
                                          "theta", "ripple")}
    bands = np.full((nw, len(BANDS)), np.nan)
    rms_ch = np.full((nw, C), np.nan)
    lagc = np.zeros((nw, L, C, C), np.float64); lagn = np.zeros((nw, L), np.float64)
    ph_pow = np.zeros((nw, nb)); ph_n = np.zeros((nw, nb))
    t_start = time.time()
    for wi, s0 in enumerate(starts):
        if contiguous:
            raw = np.asarray(fil[s0 - 1:s0 + W, gch[0]:gch[-1] + 1], np.float32)
        else:
            raw = np.asarray(fil[s0 - 1:s0 + W][:, gch], np.float32)
        x = raw[1:] if tf is None else np.asarray(tf(raw[None]), np.float32)[0]
        free = free_mask(W, s0, spikes, pre, post)
        F["free"][wi] = free.mean()
        F["mua"][wi] = (np.searchsorted(spikes, s0 + W) - np.searchsorted(spikes, s0)) / a.win_s
        if free.sum() < 0.05 * W:
            continue
        xf = x[free]
        r = robust_sd(xf)
        if not np.all(r > 0):
            continue                                                 # a flat (dropout) window
        rms_ch[wi] = r
        F["level"][wi] = float(np.mean(np.log10(r)))
        F["tails"][wi] = float(np.mean(np.log10(xf.std(0) / r)))
        var = xf.var(0)
        F["cm"][wi] = float((C * xf.mean(1).var() / var.mean() - 1.0) / (C - 1)) if var.sum() > 0 and C > 1 else np.nan
        half = (np.arange(W) // blk) % 2 == 0
        for key, hm in (("lev_even", half), ("lev_odd", ~half)):
            sel = free & hm
            if sel.sum() > 100:
                rr = robust_sd(x[sel])
                if np.all(rr > 0):
                    F[key][wi] = float(np.mean(np.log10(rr)))
        psd = window_spectrum(x, free, sr)
        if psd is not None:
            bp = band_powers(psd, sr); bands[wi] = bp
            F["tilt"][wi] = float(np.log10(bp[3] / bp[0])) if bp[0] > 0 and bp[3] > 0 else np.nan
        xc = x - xf.mean(0)
        for lag in range(L):
            ok = free[:W - lag] & free[lag:]
            if ok.any():
                lagc[wi, lag] += xc[:W - lag][ok].T @ xc[lag:][ok]
                lagn[wi, lag] += ok.sum()
        if lfp_on:
            i0 = int((s0 / sr - lfp["t0"]) * lfp["sr"]); i1 = i0 + int(a.win_s * lfp["sr"])
            if i0 >= 0 and i1 <= lfp["x"].size and i1 - i0 >= 16:
                seg = lfp["x"][i0:i1] - lfp["x"][i0:i1].mean()
                P = np.abs(np.fft.rfft(seg * np.hanning(seg.size))) ** 2
                f = np.fft.rfftfreq(seg.size, 1.0 / lfp["sr"])
                pt, pdl = P[(f >= 5) & (f < 12)].mean(), P[(f >= 1) & (f < 4)].mean()
                F["theta"][wi] = float(np.log10(pt / pdl)) if pdl > 0 else np.nan
                F["ripple"][wi] = float(np.log10(np.mean(lfp["env"][i0:i1] ** 2) + 1e-12))
                tt = (s0 + np.flatnonzero(free)) / sr
                li = np.clip(((tt - lfp["t0"]) * lfp["sr"]).astype(np.int64), 0, lfp["x"].size - 1)
                b = ((lfp["phase"][li] + np.pi) / (2 * np.pi) * nb).astype(int) % nb
                p2 = (xc[free] ** 2).mean(1)
                ph_pow[wi] = np.bincount(b, p2, minlength=nb); ph_n[wi] = np.bincount(b, minlength=nb)
        if (wi + 1) % 300 == 0:
            _log(f"  {wi + 1}/{nw} windows, {time.time() - t_start:.0f} s")

    ok = np.isfinite(F["level"])
    tw = starts / sr                                                  # window start, s
    print(f"\n== {os.path.basename(sp.base)} group {a.group}, {a.variant or 'standard'}: {ok.sum()} usable windows of "
          f"{a.win_s:g} s ({(tw[-1] + a.win_s - tw[0]) / 60:.1f} min span); spike-free share median "
          f"{np.nanmedian(F['free']):.0%}")
    lev = F["level"]
    print(f"noise level: robust SD median {10 ** np.nanmedian(lev):.1f} (file units); p5-p95 of windows "
          f"x{10 ** (np.nanpercentile(lev, 95) - np.nanpercentile(lev, 5)):.2f}; common share median "
          f"{np.nanmedian(F['cm']):.2f} (0 = independent channels, 1 = all common); tails median "
          f"x{10 ** np.nanmedian(F['tails']):.3f}")

    # ── 1. structure ──
    def real_share(sel):
        e, o, l_ = F["lev_even"][sel], F["lev_odd"][sel], lev[sel]
        m = np.isfinite(e) & np.isfinite(o) & np.isfinite(l_)
        if m.sum() < 10:
            return np.nan, np.nan, np.nan
        noise = np.var(e[m] - o[m]) / 4.0
        tot = np.var(l_[m])
        return (max(0.0, 1.0 - noise / tot) if tot > 0 else np.nan), float(np.sqrt(tot)), float(np.sqrt(noise))
    rs, sd_tot, sd_noise = real_share(ok)
    print(f"\n1. STRUCTURE  SD of the window level {sd_tot:.4f} log10 units, of which measurement noise "
          f"{sd_noise:.4f} -> real share {rs:.0%}")
    step = a.win_s * max(1, a.every)
    lags = [int(round(Ls / step)) for Ls in LAGS_S if Ls / step >= 1 and Ls / step < ok.sum() / 2]
    ac = acf(np.where(ok, lev, np.nan), lags)
    print("   autocorrelation of the level  lag (s): " + "  ".join(f"{L_ * step:>6.0f}" for L_ in lags))
    print("                                          " + "  ".join(f"{v:6.2f}" for v in ac))
    if np.isfinite(F["mua"][ok]).all():
        print(f"   corr(level, log MUA rate) {np.corrcoef(lev[ok], np.log10(F['mua'][ok] + 0.1))[0, 1]:+.2f}; "
              f"corr(level, cm) {np.corrcoef(lev[ok & np.isfinite(F['cm'])], F['cm'][ok & np.isfinite(F['cm'])])[0, 1]:+.2f}"
              + (f"; corr(level, theta index) {np.corrcoef(lev[ok & np.isfinite(F['theta'])], F['theta'][ok & np.isfinite(F['theta'])])[0, 1]:+.2f}"
                 if lfp_on and np.isfinite(F['theta'][ok]).sum() > 10 else ""))

    # ── 2. states ──
    from sklearn.mixture import GaussianMixture
    Xf = np.column_stack([F[k] for k in FEATS])
    okf = ok & np.isfinite(Xf).all(1)
    Z = (Xf[okf] - Xf[okf].mean(0)) / (Xf[okf].std(0) + 1e-12)
    bics, models = [], []
    for K in range(1, max(1, a.max_states) + 1):
        if Z.shape[0] < 10 * K:
            break
        gm = GaussianMixture(K, covariance_type="full", n_init=3, random_state=a.seed).fit(Z)
        bics.append(gm.bic(Z)); models.append(gm)
    K = int(np.argmin(bics)) + 1
    gm = models[K - 1]
    lab = np.full(nw, -1); lab[okf] = gm.predict(Z)
    order = np.argsort([np.nanmean(lev[lab == k]) for k in range(K)])
    remap = {int(o): i for i, o in enumerate(order)}
    lab = np.array([remap.get(int(v), -1) for v in lab])
    print(f"\n2. STATES  BIC for K = 1..{len(bics)}: " + ", ".join(f"{b:.0f}" for b in bics) + f" -> K = {K}")
    print("   state  share  level(x)  tails(x)   cm    tilt   MUA/s" + ("  theta" if lfp_on else "")
          + "   dwell median / p90 (s)  runs  real share within")
    transient = []
    lv0 = np.nanmedian(lev[lab >= 0])
    for k in range(K):
        m = lab == k
        idx = np.flatnonzero(m)
        brk = np.flatnonzero(np.diff(idx) != 1)
        runs = np.diff(np.r_[0, brk + 1, idx.size])
        rsk = real_share(m)[0]
        tr = bool(np.median(runs) <= 1 and m.mean() / okf.mean() < 0.2)
        transient.append(tr)
        print(f"   {k:5d}  {m.mean() / okf.mean():5.0%}  {10 ** (np.nanmedian(lev[m]) - lv0):8.2f}  "
              f"{10 ** np.nanmedian(F['tails'][m]):8.3f}  {np.nanmedian(F['cm'][m]):5.2f}  {np.nanmedian(F['tilt'][m]):6.2f}"
              f"  {np.nanmedian(F['mua'][m]):6.1f}" + (f"  {np.nanmedian(F['theta'][m]):5.2f}" if lfp_on else "")
              + f"   {np.median(runs) * step:8.0f} / {np.percentile(runs, 90) * step:6.0f}  {runs.size:5d}  {rsk:6.0%}"
              + ("   transient" if tr else ""))

    # ── 3. fast structure ──
    peri = None; phase_mod = None
    if lfp_on:
        th_ok = np.isfinite(F["theta"]) & ok
        if th_ok.sum() >= 10:
            thw = th_ok & (F["theta"] >= np.nanquantile(F["theta"][th_ok], a.theta_q))
            pp = ph_pow[thw].sum(0) / np.maximum(ph_n[thw].sum(0), 1)
            phase_mod = pp / pp.mean()
            print(f"\n3. THETA PHASE  noise power by theta phase (8 bins from -pi, theta windows = top "
                  f"{1 - a.theta_q:.0%} of theta index): " + " ".join(f"{v:.3f}" for v in phase_mod)
                  + f"   (max/min x{phase_mod.max() / phase_mod.min():.3f})")
        rip_t = lfp["t0"] + lfp["rip"] / lfp["sr"]
        rip_t = rip_t[(rip_t * sr > s_lo + 0.4 * sr) & (rip_t * sr < s_hi - 0.4 * sr)]
        if rip_t.size >= 5:
            half_w = int(0.3 * sr); bw = int(0.010 * sr); nbins = 2 * half_w // bw

            def profile(times):
                acc = np.zeros(nbins); cnt = np.zeros(nbins)
                for tt in times:
                    c0 = int(tt * sr) - half_w
                    if contiguous:
                        rw = np.asarray(fil[c0 - 1:c0 + 2 * half_w, gch[0]:gch[-1] + 1], np.float32)
                    else:
                        rw = np.asarray(fil[c0 - 1:c0 + 2 * half_w][:, gch], np.float32)
                    xx = rw[1:] if tf is None else np.asarray(tf(rw[None]), np.float32)[0]
                    fr = free_mask(2 * half_w, c0, spikes, pre, post)
                    for b in range(nbins):
                        sl = slice(b * bw, (b + 1) * bw)
                        if fr[sl].sum() > 0.3 * bw:
                            rr = robust_sd(xx[sl][fr[sl]])
                            if np.all(rr > 0):
                                acc[b] += np.mean(np.log10(rr)); cnt[b] += 1
                return acc / np.maximum(cnt, 1)
            pr = profile(rip_t)
            base_t = rng.uniform(s_lo / sr + 1, s_hi / sr - 1, rip_t.size)
            pb = profile(base_t)
            peri = 10 ** (pr - pb.mean())
            print(f"   RIPPLES  {rip_t.size} events: noise level around the ripple peak (10-ms bins, -300..+300 ms) "
                  f"vs random times: peak x{peri.max():.2f} at {(np.argmax(peri) - nbins // 2) * 10:+d} ms, "
                  f"+-100 ms mean x{peri[nbins // 2 - 10:nbins // 2 + 10].mean():.2f}")

    # ── 4. pipeline impact: one whitener for every state / chunk? ──
    def cov_of(sel):
        n = lagn[sel].sum(0)
        if (n < 10).any():
            return None
        return toeplitz_from_lags(lagc[sel].sum(0) / n[:, None, None])
    Cs = cov_of(okf)
    print(f"\n4. WHITENER  covariance over L = {L} samples x {C} channels; eigenvalues of Cs^-1/2 C Cs^-1/2 "
          f"(Cs = the whole span): factor = geometric mean, spread = exp(SD log)")
    mm_rows = []
    for name, sel in [(f"state {k}", lab == k) for k in range(K)] + \
                     [(f"chunk {c} ({c * a.chunk_min:g}-{(c + 1) * a.chunk_min:g} min)",
                       okf & (tw // (a.chunk_min * 60) == c)) for c in np.unique((tw[okf] // (a.chunk_min * 60)).astype(int))]:
        Ck = cov_of(sel & okf)
        if Cs is None or Ck is None:
            continue
        fac, lo, hi, spr, lam = mismatch(Cs, Ck)
        mm_rows.append((name, fac, lo, hi, spr, lam))
        print(f"   {name:24s} factor {fac:5.2f}  min {lo:5.2f}  max {hi:5.2f}  spread x{spr:4.2f}")

    # ── outputs ──
    np.savez(a.out + ".npz", t=tw, win_s=a.win_s, **F, bands=bands, rms_ch=rms_ch,
             state=lab, bic=np.array(bics), transient=np.array(transient, bool), lagc=lagc.sum(0) / np.maximum(lagn.sum(0), 1)[:, None, None],
             phase_mod=(phase_mod if phase_mod is not None else np.array([])),
             peri=(peri if peri is not None else np.array([])),
             mismatch_names=np.array([r_[0] for r_ in mm_rows]),
             mismatch_eig=np.array([r_[5] for r_ in mm_rows]) if mm_rows else np.zeros((0, 0)))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 2, figsize=(14, 10))
        tm = tw / 60.0
        cols = plt.cm.viridis(np.linspace(0, 0.9, max(K, 1)))
        for k in range(K):
            m = lab == k
            ax[0, 0].plot(tm[m], 10 ** lev[m], ".", ms=2, color=cols[k], label=f"state {k}")
        ax[0, 0].set_ylabel("robust noise SD"); ax[0, 0].set_xlabel("min"); ax[0, 0].legend(markerscale=5)
        ax[0, 0].set_title("noise level per window")
        ax[1, 0].plot(tm, F["cm"], ".", ms=2, color="0.3", label="common share")
        ax[1, 0].set_ylabel("common share"); ax2 = ax[1, 0].twinx()
        ax2.plot(tm, F["tilt"], ".", ms=2, color="tab:orange"); ax2.set_ylabel("tilt (4-8 kHz / 0.3-1 kHz)")
        ax[1, 0].set_xlabel("min")
        ax[2, 0].plot(tm, F["mua"], ".", ms=2, color="tab:red"); ax[2, 0].set_ylabel("detections / s")
        if lfp_on:
            ax3 = ax[2, 0].twinx(); ax3.plot(tm, F["theta"], ".", ms=2, color="tab:blue"); ax3.set_ylabel("theta index")
        ax[2, 0].set_xlabel("min")
        ax[0, 1].semilogx([L_ * step for L_ in lags], ac, "o-")
        ax[0, 1].axhline(0, color="0.6", lw=0.8); ax[0, 1].set_xlabel("lag (s)"); ax[0, 1].set_ylabel("autocorrelation")
        ax[0, 1].set_title(f"level autocorrelation; real share {rs:.0%}")
        for k in range(K):
            m = lab == k
            ax[1, 1].hist(10 ** lev[m], 60, histtype="step", color=cols[k], label=f"state {k}")
        ax[1, 1].set_xlabel("robust noise SD"); ax[1, 1].legend()
        for name, _fac, _lo, _hi, _spr, lam in mm_rows:
            ax[2, 1].plot(np.sort(lam)[::-1], lw=1, label=name if name.startswith("state") else None,
                          color=("k" if name.startswith("state") else "0.7"))
        ax[2, 1].axhline(1, color="0.6", lw=0.8); ax[2, 1].set_yscale("log")
        ax[2, 1].set_xlabel("eigenvalue rank"); ax[2, 1].set_ylabel("noise variance under the span whitener")
        ax[2, 1].set_title("whitener mismatch (black: states, grey: chunks)"); ax[2, 1].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(a.out + ".png", dpi=110)
        if phase_mod is not None or peri is not None:
            fig2, bx = plt.subplots(1, 2, figsize=(11, 3.6))
            if phase_mod is not None:
                bx[0].bar(np.arange(nb), phase_mod); bx[0].axhline(1, color="0.6")
                bx[0].set_xlabel("theta phase bin (from -pi)"); bx[0].set_ylabel("noise power / mean")
            if peri is not None:
                bx[1].plot((np.arange(peri.size) - peri.size // 2) * 10, peri)
                bx[1].axhline(1, color="0.6"); bx[1].set_xlabel("ms from ripple peak"); bx[1].set_ylabel("noise level / random")
            fig2.tight_layout(); fig2.savefig(a.out + "_fast.png", dpi=110)
    except Exception as e:                                           # noqa: BLE001 -- the numbers are written already
        _log(f"  figure skipped ({e})")
    _log(f"wrote {a.out}.npz, {a.out}.png" + (f", {a.out}_fast.png" if lfp_on else "") + f" in {time.time() - t_start:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
