#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
#  fiber_adlp.py — amplitude-dependent low-pass (ADLP) of spike snippets.
#
#  A small spike's high-frequency content is mostly noise (the template spectrum
#  falls under the noise floor at a frequency that rises with amplitude), while a
#  large spike carries real shape detail there.  ADLP gives every spike its own
#  zero-phase low-pass, the cutoff a power law of its amplitude:
#
#      fc = clip(fc_ref * (A / A_ref) ** gamma, fc_min, fc_max)
#
#  A = baseline-corrected peak-to-peak inside [peak+amp_lo, peak+amp_hi), taken on
#  the spike's largest channel (scope 'spike': one cutoff for all channels, so the
#  spatial footprint keeps its ratios at every frequency) or per channel (scope
#  'channel': far, small channels get smoothed harder).  A_ref defaults to the
#  file's median A.  gamma = 0 is a uniform low-pass at fc_ref (the control that
#  separates "low-pass" from "amplitude-dependent"); fc >= Nyquist is passthrough.
#
#  The filter is filtfilt of an order-`order` digital Butterworth (bilinear, so its
#  squared magnitude is 1/(1+(tan(pi f/sr)/tan(pi fc/sr))^(2*order))), applied in
#  the frequency domain after odd-reflection padding of nSamples-1 (filtfilt's
#  padtype='odd'): zero-phase, continuous at the window edges, and equal to
#  scipy.signal.filtfilt(butter(order, fc), padlen=nSamples-1) per spike.  Writes a new waveform variant
#  .spk.<out>.<g> (int16), a per-channel temporal PCA basis .pca.<out>.<g> fitted
#  on the filtered snippets (same window as the input variant's basis), and a
#  sidecar .adlp.<out>.<g>.npz (per-spike amplitude + cutoff, parameters).
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import os
import time

import numpy as np

try:
    from . import neuro_io as nio
    from . import fiber_pca as fpca
    from . import session_yaml as sy
except ImportError:                                         # flat-module use
    import neuro_io as nio
    import fiber_pca as fpca
    import session_yaml as sy


def ptp_amplitude(w, peak, lo=-6, hi=10, base=12):
    """w (n,T,C) -> (n,C) peak-to-peak inside [peak+lo, peak+hi), baseline = mean of the first
    `base` samples (clipped so it ends before the window)."""
    T = w.shape[1]
    a0 = max(0, peak + lo); a1 = min(T, peak + hi); b1 = max(1, min(base, a0))
    seg = w[:, a0:a1, :] - w[:, :b1, :].mean(1, keepdims=True)
    return seg.max(1) - seg.min(1)


def cutoff(A, A_ref, fc_ref, gamma, fc_min, fc_max):
    """Power-law cutoff, clipped to [fc_min, fc_max]."""
    r = np.maximum(np.asarray(A, float), 1e-12) / float(A_ref)
    return np.clip(float(fc_ref) * np.power(r, float(gamma)), fc_min, fc_max)


def lowpass(w, sr, fc, order=2):
    """Zero-phase low-pass of each snippet at its own cutoff: filtfilt of a digital
    Butterworth, odd padding of T-1 samples.  w (n,T,C); fc (n,) per spike or (n,C) per
    channel, Hz.  fc >= sr/2 -> exact passthrough."""
    w = np.asarray(w, np.float64)
    n, T, C = w.shape
    P = T - 1                                               # longest odd reflection the window allows
    left = 2 * w[:, :1, :] - w[:, 1:P + 1, :][:, ::-1, :]
    right = 2 * w[:, -1:, :] - w[:, -P - 1:-1, :][:, ::-1, :]
    x = np.concatenate([left, w, right], axis=1)
    L = x.shape[1]
    f = np.fft.rfftfreq(L, 1.0 / sr)
    fc = np.asarray(fc, np.float64)
    if fc.ndim == 1:
        fc = np.repeat(fc[:, None], C, axis=1)
    nyq = sr / 2.0
    tf = np.tan(np.pi * np.minimum(f, nyq * (1.0 - 1e-9)) / sr)            # bilinear frequency warp
    tc = np.tan(np.pi * np.minimum(fc, nyq * (1.0 - 1e-9)) / sr)
    H = 1.0 / (1.0 + (tf[None, :, None] / tc[:, None, :]) ** (2 * int(order)))
    H = np.where(fc[:, None, :] >= nyq, 1.0, H)
    y = np.fft.irfft(np.fft.rfft(x, axis=1) * H, n=L, axis=1)
    return y[:, P:P + T, :]


def amplitude(w, peak, scope, lo, hi):
    A = ptp_amplitude(w, peak, lo, hi)
    return A.max(1) if scope == "spike" else A


def run(spk_in, spk_out, sr, peak, *, fc_ref, gamma, fc_min, fc_max, order=2, scope="spike",
        a_ref=None, amp_lo=-6, amp_hi=10, block=40000, ref_n=200000, seed=0, log=print):
    """Filter memmap `spk_in` (N,T,C int16) into `spk_out` (same shape, int16).  Returns
    (A, fc, a_ref): per-spike (or spike x channel) amplitude and cutoff, and the A_ref used."""
    N = spk_in.shape[0]
    if a_ref is None:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(N, min(N, ref_n), replace=False))
        a_ref = float(np.median(amplitude(np.asarray(spk_in[idx], np.float64), peak, "spike",
                                          amp_lo, amp_hi)))
        log(f"A_ref = {a_ref:.1f} (median spike ptp over {idx.size:,} snippets)")
    shp = (N,) if scope == "spike" else (N, spk_in.shape[2])
    A_all = np.empty(shp, np.float32); fc_all = np.empty(shp, np.float32)
    lo16, hi16 = np.iinfo(np.int16).min, np.iinfo(np.int16).max
    nclip = 0
    for s in range(0, N, block):
        w = np.asarray(spk_in[s:s + block], np.float64)
        A = amplitude(w, peak, scope, amp_lo, amp_hi)
        fc = cutoff(A, a_ref, fc_ref, gamma, fc_min, fc_max)
        y = np.rint(lowpass(w, sr, fc, order))
        nclip += int(((y < lo16) | (y > hi16)).sum())
        spk_out[s:s + block] = np.clip(y, lo16, hi16).astype(np.int16)
        A_all[s:s + block] = A; fc_all[s:s + block] = fc
    if nclip:
        log(f"warning: {nclip:,} samples clipped to int16")
    return A_all, fc_all, a_ref


def fit_pca_like(spk, template, n_fit=200000, seed=0):
    """Fit a per-channel temporal PCA on `spk` in the window of the `template` basis
    (recShift, data2use, nComp, centered).  Returns (means, evec)."""
    N = spk.shape[0]
    idx = np.sort(np.random.default_rng(seed).choice(N, min(N, n_fit), replace=False))
    win = fpca.extract_windows(spk[idx], template["recShift"], template["data2use"])
    return fpca.fit_basis(win, template["nComp"], centered=bool(template["centered"]))


def main():
    ap = argparse.ArgumentParser(
        description="Amplitude-dependent low-pass of spike snippets: every spike gets a zero-phase "
                    "low-pass whose cutoff is a power law of its amplitude.  Writes .spk.<out>.<g>, "
                    ".pca.<out>.<g> (fitted on the filtered snippets) and .adlp.<out>.<g>.npz.")
    sy.add_session_args(ap, channels=False, ntotal=False)
    ap.add_argument("--in-variant", default="standard",
                    help="waveform variant to filter (exact token; default standard)")
    ap.add_argument("--out-variant", default="standard_ADLP", help="variant token to write")
    ap.add_argument("--fc-ref", type=float, default=9000.0,
                    help="cutoff (Hz) at A = A_ref.  Default 9000: where the median g7 standard spike's "
                         "spectrum meets the pre-peak noise floor")
    ap.add_argument("--a-ref", type=float, default=None, help="reference amplitude (default: file median ptp)")
    ap.add_argument("--gamma", type=float, default=1.2,
                    help="cutoff exponent: fc ~ A**gamma (0 = uniform low-pass at --fc-ref)")
    ap.add_argument("--fc-min", type=float, default=2500.0, help="lowest cutoff (Hz)")
    ap.add_argument("--fc-max", type=float, default=None,
                    help="highest cutoff (Hz); default Nyquist, so large spikes pass unfiltered")
    ap.add_argument("--order", type=int, default=2, help="Butterworth order (applied forward+backward)")
    ap.add_argument("--scope", choices=["spike", "channel"], default="spike",
                    help="spike = one cutoff from the largest channel; channel = a cutoff per channel")
    ap.add_argument("--amp-lo", type=int, default=-6, help="amplitude window start, samples from peak")
    ap.add_argument("--amp-hi", type=int, default=10, help="amplitude window end, samples from peak")
    ap.add_argument("--block", type=int, default=40000, help="spikes per processing block")
    ap.add_argument("--no-pca", action="store_true", help="do not fit/write .pca.<out>.<g>")
    a = ap.parse_args()

    P = "▸ fiber-adlp"; IND = " " * (len(P) + 3)
    def log(m):  print(f"{P} · {m}")
    def det(k, v):  print(f"{IND}{k:<11} {v}")

    cfg = sy.resolve_session_params(a.session, a.group, nchan=a.nchan, nsamp=a.nsamp, sr=a.sr,
                                    require=())
    base, g = cfg["base"], a.group
    nsamp, nchan, sr, peak = int(cfg["nsamp"]), int(cfg["nchan"]), float(cfg["sr"]), int(cfg["peak"])
    fc_max = sr / 2.0 if a.fc_max is None else float(a.fc_max)
    src = nio.session_path(base, "spk", g, variant=a.in_variant)
    dst = nio.session_path(base, "spk", g, variant=a.out_variant)
    if not os.path.exists(src):
        raise SystemExit(f"fiber-adlp: no {src}")
    if os.path.abspath(src) == os.path.abspath(dst):
        raise SystemExit("fiber-adlp: --out-variant must differ from --in-variant")
    spk_in = nio.open_spk_file(src, nsamp, nchan)
    N = spk_in.shape[0]
    log(f"group {g} · {N:,} spikes · {a.in_variant} → {a.out_variant}")
    det("law", f"fc = clip({a.fc_ref:g} Hz · (A/A_ref)^{a.gamma:g}, {a.fc_min:g}, {fc_max:g}) · "
               f"order {a.order} zero-phase · scope {a.scope}")
    t0 = time.time()
    spk_out = np.memmap(dst, np.int16, "w+", shape=(N, nsamp, nchan))
    A, fc, a_ref = run(spk_in, spk_out, sr, peak, fc_ref=a.fc_ref, gamma=a.gamma, fc_min=a.fc_min,
                       fc_max=fc_max, order=a.order, scope=a.scope, a_ref=a.a_ref,
                       amp_lo=a.amp_lo, amp_hi=a.amp_hi, block=a.block, log=log)
    spk_out.flush()
    q = np.percentile(fc, [5, 25, 50, 75, 95])
    det("cutoff", " / ".join(f"{v / 1000:.1f}" for v in q) + " kHz (5/25/50/75/95%)")
    det("passthru", f"{100.0 * np.mean(fc >= sr / 2.0):.1f}% at Nyquist")
    det("wrote", dst)
    side = nio.session_path(base, "adlp", g, variant=a.out_variant) + ".npz"
    np.savez(side, amp=A, fc=fc, a_ref=a_ref, fc_ref=a.fc_ref, gamma=a.gamma, fc_min=a.fc_min,
             fc_max=fc_max, order=a.order, scope=a.scope, in_variant=a.in_variant,
             amp_lo=a.amp_lo, amp_hi=a.amp_hi)
    det("wrote", side)
    if not a.no_pca:
        try:
            tmpl = fpca.read_pcad(nio.session_path(base, "pca", g, variant=a.in_variant))
        except (FileNotFoundError, ValueError) as e:
            log(f"no input basis to copy the window from ({e}); --no-pca implied")
        else:
            means, evec = fit_pca_like(spk_out, tmpl)
            pdst = nio.session_path(base, "pca", g, variant=a.out_variant)
            fpca.write_pcad(pdst, means, evec, tmpl["recShift"], centered=tmpl["centered"],
                            method=tmpl["method"], n_input_channels=tmpl["nInputChannels"])
            det("wrote", f"{pdst}  ({tmpl['nComp']} comp · window {tmpl['recShift']}"
                         f"+{tmpl['data2use']})")
    log(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
