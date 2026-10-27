"""tools/fil_noise_rhythms.py on synthetic sessions whose noise is built to follow theta and gamma.

  * MODULATED file (120 s, 20 kHz, 8-channel group of a 10-channel file).  The LFP holds an 8-Hz theta for the
    first 60 s and a 2-Hz delta after; slow gamma (37 Hz, not a harmonic of theta) whose amplitude peaks at theta
    phase -90 deg; mid gamma (70 Hz) in random bursts over the whole file; no fast gamma.  The spike-band noise is:
      - x1.2 in the theta half, on every channel;
      - on channels 0-3 in the theta half, amplitude x (1 + 0.3 cos(theta - 180 deg)): power m 0.574 at 180 deg;
      - on channels 4-7 in the mid-gamma bursts, x 1.15 (1 + 0.4 cos(gamma - 90 deg)): power m 0.74 at +90 deg,
        and x1.43 power in the bursts.
    Detected spikes (masked by the tool) come at 30/s, x (1 + 0.6 cos(theta - 180 deg)) in the theta half.  The
    file is flat for 4 s (90-94 s) and channel 5 drops to 40% for 3 s (100-103 s).  The tool must find:
      - the 4 flat and 3 dropout windows, left out;
      - the theta windows in the theta half, and the noise x1.2 there;
      - theta-phase modulation on channels 0-3 only, at 180 deg, significant, with the detections at 180 deg;
      - mid-gamma-phase modulation on channels 4-7 only, at +90 deg, significant, in the bursts and in the
        exclusive bursts; none for slow gamma;
      - the mid bursts LEAK into the fast band (no fast gamma is built): the fast band's bursts show the mid
        modulation, and its exclusive bursts lose at least 80% of it;
      - the noise rising with the mid-gamma envelope on channels 4-7 only (+15% per SD there in the joint
        regression); not with slow gamma, whose amplitude is coupled to a theta phase (-90 deg) where the
        noise's theta modulation is at its mean, nor, once mid gamma is held fixed, with fast gamma;
      - the slow-gamma envelope coupled to theta phase at -90 deg.
  * the same file in stderiv_C5 (sdiffPairs from the yaml): it runs and still finds the theta modulation.
  * STATIONARY control (the same LFP, unmodulated noise, uniform spikes): no theta- or gamma-phase modulation, no
    amplitude relation, no theta-state change.
"""
import os
import sys
import tempfile

import numpy as np
from scipy.signal import butter, sosfilt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import fil_noise_rhythms as fnr    # noqa: E402

fails = 0
SR, LFP_SR, NTOT, DUR, HALF = 20000, 1250, 10, 120, 60
PAIRS = "0-2+4,1-3+5,2-4+0+6,3-5+1+7,4-6+2+7,5-7+3+6,6-7+3+5,7-5+6"


def check(ok, what):
    global fails
    print(("  ok:   " if ok else "  FAIL: ") + what); fails += not ok


def cdist(a, b):
    """Circular distance in degrees."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def bursts(rng):
    """Mid-gamma burst intervals (s): Poisson starts, 60-120 ms long, ~25% duty."""
    out, t = [], 0.2
    while t < DUR - 0.3:
        t += rng.exponential(0.27)
        d = rng.uniform(0.06, 0.12)
        out.append((t, t + d)); t += d
    return out


def gate(times, iv):
    """1 inside the bursts, raised-cosine edges of 5 ms."""
    g = np.zeros(times.size)
    e = 0.005
    for a, b in iv:
        i0, i1 = np.searchsorted(times, a - e), np.searchsorted(times, b + e)
        tt = times[i0:i1]
        up = np.clip((tt - (a - e)) / e, 0, 1); dn = np.clip(((b + e) - tt) / e, 0, 1)
        g[i0:i1] = np.maximum(g[i0:i1], np.minimum(up, dn) ** 2 * (3 - 2 * np.minimum(up, dn)))
    return g


def make_session(d, name, modulated, rng):
    T = DUR * SR
    t = np.arange(T) / SR
    theta = 2 * np.pi * 8.0 * t + 0.5 * np.sin(2 * np.pi * 0.3 * t)
    psi = 2 * np.pi * 70.0 * t + 0.3 * np.sin(2 * np.pi * 0.5 * t)
    iv = bursts(rng)
    g = gate(t, iv)
    th_half = t < HALF
    sos = butter(3, (300, 5000), "band", fs=SR, output="sos")
    ind = sosfilt(sos, rng.normal(0, 1, (T, 8)), axis=0); ind /= ind.std(0)
    cm = sosfilt(sos, rng.normal(0, 1, T)); cm /= cm.std()
    x = 30.0 * ind + (12.0 * cm)[:, None]
    del ind
    if modulated:
        x *= np.where(th_half, 1.2, 1.0)[:, None]
        x[:, :4] *= np.where(th_half, 1.0 + 0.3 * np.cos(theta - np.pi), 1.0)[:, None]
        x[:, 4:] *= (1.0 + g * (1.15 * (1.0 + 0.4 * np.cos(psi - np.pi / 2)) - 1.0))[:, None]
    # detected spikes (masked by the tool)
    cand = np.sort(rng.uniform(1.0, DUR - 1.0, int(48 * DUR)))
    th_c = 2 * np.pi * 8.0 * cand + 0.5 * np.sin(2 * np.pi * 0.3 * cand)
    keep = np.where(cand < HALF, rng.random(cand.size) < (1 + 0.6 * np.cos(th_c - np.pi)) / 1.6,
                    rng.random(cand.size) < 30 / 48) if modulated else rng.random(cand.size) < 30 / 48
    sp = np.unique((cand[keep] * SR).astype(np.int64))
    tmpl = -400 * np.exp(-0.5 * (np.arange(-15, 17) / 3.0) ** 2)
    for o in range(-15, 17):
        x[sp + o, :4] += tmpl[o + 15] * np.array([1.0, 0.8, 0.5, 0.3])
    # hardware faults: 4 s flat, 3 s of channel 5 at 40%
    x[(t >= 90) & (t < 94)] = 0.0
    x[(t >= 100) & (t < 103), 5] *= 0.4
    fil = np.zeros((T, NTOT), np.int16)
    fil[:, :8] = np.clip(np.round(x), -32767, 32767)
    fil[:, 8:] = np.round(rng.normal(0, 30, (T, 2)))
    fil[(t >= 90) & (t < 94)] = 0
    base = os.path.join(d, name)
    fil.tofile(base + ".fil")
    del fil, x
    sp.astype("<i8").tofile(base + ".res.1")
    # LFP
    tl = np.arange(DUR * LFP_SR) / LFP_SR
    thl = 2 * np.pi * 8.0 * tl + 0.5 * np.sin(2 * np.pi * 0.3 * tl)
    psl = 2 * np.pi * 70.0 * tl + 0.3 * np.sin(2 * np.pi * 0.5 * tl)
    th_l = tl < HALF
    lfp = np.where(th_l, 400 * np.cos(thl), 300 * np.cos(2 * np.pi * 2.0 * tl))
    slow_amp = np.where(th_l, 120 * (1 + np.cos(thl + np.pi / 2)) / 2, 20.0)
    lfp += slow_amp * np.cos(2 * np.pi * 37.0 * tl)
    lfp += 150 * gate(tl, iv) * np.cos(psl)
    lfp += rng.normal(0, 20, tl.size)
    np.repeat(np.round(lfp).astype(np.int16)[:, None], NTOT, 1).tofile(base + ".lfp")
    with open(base + ".yaml", "w") as fh:
        fh.write(f"acquisitionSystem: {{nBits: 16, nChannels: {NTOT}, samplingRate: {SR}}}\n"
                 f"fieldPotentials: {{lfpSamplingRate: {LFP_SR}}}\n"
                 "spikeDetection:\n  channelGroups:\n    - channels: [0, 1, 2, 3, 4, 5, 6, 7]\n"
                 f"      nSamples: 32\n      peakSampleIndex: 15\n      sdiffPairs: '{PAIRS}'\n")
    return base


def run(base, out, *extra):
    cwd = os.getcwd()
    os.chdir(os.path.dirname(base))
    try:
        rc = fnr.main([os.path.basename(base), "1", "--out", out, "--shuffles", "200", *extra])
    finally:
        os.chdir(cwd)
    return rc, np.load(out + ".npz")


def main():
    rng = np.random.default_rng(5)
    with tempfile.TemporaryDirectory() as d:
        base = make_session(d, "mod", True, rng)
        rc, Z = run(base, os.path.join(d, "mod_std"))
        check(rc == 0, "modulated file: the tool runs")
        t = Z["t"]
        flat, drop = Z["flat"], Z["dropout"]
        check(flat.sum() == 4 and set(np.round(t[flat]).astype(int)) == {90, 91, 92, 93},
              f"the 4 flat windows are left out ({flat.sum()} at {np.round(t[flat]).astype(int).tolist()} s)")
        check(drop.sum() == 3 and set(np.round(t[drop]).astype(int)) == {100, 101, 102},
              f"the 3 dropout windows are left out ({drop.sum()} at {np.round(t[drop]).astype(int).tolist()} s)")
        thw = Z["theta_win"]
        check(thw.sum() >= 25 and np.mean(t[thw] < HALF) == 1.0,
              f"theta windows lie in the theta half ({thw.sum()}, {np.mean(t[thw] < HALF):.0%})")
        ratio = 10 ** (np.nanmedian(np.nanmean(Z["level_ch"][thw], 1)) - np.nanmedian(np.nanmean(Z["level_ch"][Z["nontheta_win"]], 1)))
        check(1.12 < ratio < 1.32, f"noise level theta / non-theta x{ratio:.3f} (built x1.2-1.23)")
        m, ph, p = float(Z["phase_theta_m"]), float(Z["phase_theta_phase"]), float(Z["phase_theta_p"])
        mch = Z["phase_theta_m_ch"]
        check(0.2 < m < 0.36 and cdist(ph, 180) < 25 and p < 0.01,
              f"theta: channel-mean noise m {m:.3f} at {ph:+.0f} deg, p {p:.3f} (built 0.29 at 180 deg)")
        check(np.all((mch[:4] > 0.45) & (mch[:4] < 0.7)) and np.all(mch[4:] < 0.08),
              "theta: per channel m " + " ".join(f"{v:.2f}" for v in mch) + " (built 0.57 on 0-3, 0 on 4-7)")
        mr, phr, pr = float(Z["phase_theta_m_rate"]), float(Z["phase_theta_phase_rate"]), float(Z["phase_theta_p_rate"])
        check(0.45 < mr < 0.75 and cdist(phr, 180) < 25 and pr < 0.01,
              f"theta: detections m {mr:.3f} at {phr:+.0f} deg, p {pr:.3f} (built 0.6 at 180 deg)")
        mm, phm, pm = float(Z["phase_mid_m"]), float(Z["phase_mid_phase"]), float(Z["phase_mid_p"])
        mmch = Z["phase_mid_m_ch"]
        check(0.22 < mm < 0.45 and cdist(phm, 90) < 30 and pm < 0.01,
              f"mid gamma: channel-mean noise m {mm:.3f} at {phm:+.0f} deg, p {pm:.3f} (built ~0.35 at +90 deg)")
        check(np.all(mmch[4:] > 0.5) and np.all(mmch[:4] < 0.1),
              "mid gamma: per channel m " + " ".join(f"{v:.2f}" for v in mmch) + " (built 0.74 on 4-7, 0 on 0-3)")
        mx, phx, px = float(Z["phase_mid_excl_m"]), float(Z["phase_mid_excl_phase"]), float(Z["phase_mid_excl_p"])
        check(mx > 0.3 and cdist(phx, 90) < 30 and px < 0.01,
              f"mid gamma, exclusive bursts: still m {mx:.3f} at {phx:+.0f} deg, p {px:.3f}")
        ms_, ps_ = float(Z["phase_slow_m"]), float(Z["phase_slow_p"])
        msx, psx = float(Z["phase_slow_excl_m"]), float(Z["phase_slow_excl_p"])
        check(ps_ > 0.005 and ms_ < 0.03 and (psx > 0.005 or msx < 0.03),
              f"slow gamma: no noise phase modulation (m {ms_:.3f}, p {ps_:.3f}; exclusive m {msx:.3f}, p {psx:.3f})")
        mf, mfx = float(Z["phase_fast_m"]), float(Z["phase_fast_excl_m"])
        check(mf > float(np.nanpercentile(Z["phase_fast_null"], 99)) and mfx < 0.03 and mfx < 0.2 * mf,
              f"fast gamma: the mid bursts' leak shows in its bursts (m {mf:.3f}) and goes in its exclusive bursts "
              f"(m {mfx:.3f})")
        tb, tbch = float(Z["amp_mid_top_bottom"]), Z["amp_mid_top_bottom_ch"]
        check(1.12 < tb < 1.35 and np.all(tbch[4:] > 1.3) and np.all(np.abs(tbch[:4] - 1) < 0.06),
              f"mid gamma envelope: noise top/bottom x{tb:.3f}; per channel " + " ".join(f"{v:.2f}" for v in tbch)
              + " (built x1.43 on 4-7)")
        tbs = float(Z["amp_slow_top_bottom"])
        check(abs(tbs - 1) < 0.05, f"slow gamma envelope: no noise relation (x{tbs:.3f})")
        u = {n: float(Z[f"amp_{n}_unique"]) for n in ("slow", "mid", "fast")}
        uci = Z["amp_mid_unique_ci"]; uch = Z["amp_mid_unique_ch"]
        check(u["mid"] > 5 and uci[0] > 4 and np.all(uch[4:] > 12) and np.all(uch[:4] < 2)
              and abs(u["slow"]) < 2.5 and abs(u["fast"]) < 2.5,
              f"joint regression: unique % per SD slow {u['slow']:+.2f}, mid {u['mid']:+.2f} (95% {uci[0]:+.2f} to "
              f"{uci[1]:+.2f}; per channel " + " ".join(f"{v:+.1f}" for v in uch) + f"), fast {u['fast']:+.2f}")
        cm_, cph = float(Z["coupling_slow_m"]), float(Z["coupling_slow_phase"])
        check(cm_ > 0.5 and cdist(cph, -90) < 30, f"slow-gamma envelope coupled to theta: m {cm_:.2f} at {cph:+.0f} deg (built -90)")
        cmm = float(Z["coupling_mid_m"])
        check(cmm < 0.2, f"mid-gamma envelope not coupled to theta (m {cmm:.2f})")

        rc, Z5 = run(base, os.path.join(d, "mod_c5"), "--variant", "stderiv_C5", "--shuffles", "100")
        check(rc == 0 and float(Z5["phase_theta_p"]) < 0.02 and cdist(float(Z5["phase_theta_phase"]), 180) < 30,
              f"stderiv_C5: theta modulation still found (m {float(Z5['phase_theta_m']):.3f} at "
              f"{float(Z5['phase_theta_phase']):+.0f} deg, p {float(Z5['phase_theta_p']):.3f})")

        base0 = make_session(d, "flat", False, rng)
        rc, Z0 = run(base0, os.path.join(d, "flat_std"))
        check(rc == 0, "stationary control: the tool runs")
        bad = [(n, float(Z0[f"phase_{n}_m"]), float(Z0[f"phase_{n}_p"])) for n in ("theta", "slow", "mid", "fast")]
        check(all(p_ > 0.005 and m_ < 0.06 for _, m_, p_ in bad),
              "stationary control: no phase modulation (" + ", ".join(f"{n} m {m_:.3f} p {p_:.3f}" for n, m_, p_ in bad) + ")")
        tbs0 = [float(Z0[f"amp_{n}_top_bottom"]) for n in ("slow", "mid", "fast")]
        u0 = [float(Z0[f"amp_{n}_unique"]) for n in ("slow", "mid", "fast")]
        check(all(abs(v - 1) < 0.05 for v in tbs0) and all(abs(v) < 1.0 for v in u0),
              "stationary control: no amplitude relation (" + ", ".join(f"x{v:.3f}" for v in tbs0)
              + "; unique " + ", ".join(f"{v:+.2f}%" for v in u0) + ")")
        r0 = 10 ** (np.nanmedian(np.nanmean(Z0["level_ch"][Z0["theta_win"]], 1))
                    - np.nanmedian(np.nanmean(Z0["level_ch"][Z0["nontheta_win"]], 1)))
        check(abs(r0 - 1) < 0.03, f"stationary control: noise level theta / non-theta x{r0:.3f}")
        mr0, pr0 = float(Z0["phase_theta_m_rate"]), float(Z0["phase_theta_p_rate"])
        check(pr0 > 0.005 or mr0 < 0.1, f"stationary control: detections not theta-modulated (m {mr0:.3f}, p {pr0:.3f})")
    print(f"\ntest_fil_noise_rhythms: {'PASS' if fails == 0 else 'FAIL'} ({fails} failed)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
