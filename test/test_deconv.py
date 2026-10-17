"""fiber_deconv: detection-anchored sub-sample template deconvolution.

Unit checks (fast):
  * the VAR whitening filter is linear and shift-equivariant (what lets the
    pursuit treat the whitened signal as a sum of whitened templates);
  * two overlapping spikes 9 samples apart, at fractional times, in low noise:
    both units recovered, times within 0.15 sample, amplitudes within 5 %;
  * NEGATIVE CONTROL: pure noise pursued with a bank of templates that are not
    in it -- placements on at most 1 % of pseudo-detections;
  * calibrate_margin: refuses (inf) below min_n, returns the smallest margin
    whose top set agrees >= target.

The fiber-session integration is checked end to end by
test_session_deconv.py.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fiber_kit import fiber_deconv as fd          # noqa: E402
from fiber_kit import fiber_lib as fl             # noqa: E402

SR = 32552.0
C, NSAMP, PEAK = 8, 42, 21


def raw_template(depth, width, amp):
    t = (np.arange(NSAMP) - PEAK) / SR * 1e3
    w = -np.exp(-0.5 * (t / (0.12 * width)) ** 2) \
        + 0.35 * np.exp(-0.5 * ((t - 0.45 * width) / (0.25 * width)) ** 2)
    y = np.arange(C) * 20.0
    foot = np.exp(-0.5 * ((y - depth) / 28.0) ** 2) + 0.05
    T = np.zeros((NSAMP, C))
    for c in range(C):
        T[:, c] = np.interp(np.arange(NSAMP) - (y[c] - depth) / 60.0, np.arange(NSAMP), w) * foot[c]
    return amp * T


def place(fil, T, t, a=1.0):
    i = int(np.floor(t)); f = t - i
    buf = np.zeros((NSAMP + 8, C)); buf[4:4 + NSAMP] = T
    fil[i - PEAK - 4:i - PEAK - 4 + NSAMP + 8] += a * fd._frac_shift(buf, f)


def noise(rng, n, sd):
    from scipy.signal import lfilter
    L = np.linalg.cholesky(0.5 * np.eye(C) + 0.5 * np.exp(
        -np.abs(np.subtract.outer(range(C), range(C))) / 2.0))
    return lfilter([1.0], [1.0, -0.6, 0.15], rng.normal(0, sd, (n, C)) @ L.T, axis=0)


def transformed_template(T):
    return fl.fil_to_spkD_space(np.vstack([np.zeros((1, C)), T]))[1:]


def main():
    ok = 0

    def check(name, cond):
        nonlocal ok
        assert cond, name
        ok += 1
        print(f"  ok  {name}")

    rng = np.random.default_rng(0)

    # ── whitening filter: linear + shift-equivariant ──
    N = 200000
    sig = noise(rng, N, 14.0)
    rd = lambda a, b: np.vstack([np.zeros((max(0, -a), C)), sig[max(0, a):b]])  # noqa: E731
    nm = fd.fit_noise_model(rd, 0, N, np.array([], np.int64))
    x = rng.normal(size=(300, C)); y = rng.normal(size=(300, C))
    check("whiten is linear", np.allclose(fd.whiten(x + 2 * y, nm), fd.whiten(x, nm) + 2 * fd.whiten(y, nm)))
    xs = np.vstack([np.zeros((7, C)), x])
    check("whiten is shift-equivariant", np.allclose(fd.whiten(xs, nm)[7:], fd.whiten(x, nm)))
    wn = fd.whiten(fd.read_transformed(rd, 1000, 100000), nm)[nm.order:]
    ev = np.linalg.eigvalsh(np.cov(wn.T))      # allpairs has rank C-1: one null direction, the rest white
    check("whitened noise is white across channels (eigenvalues 0 once, else 1 +/- 0.05)",
          abs(ev[0]) < 1e-6 and np.abs(ev[1:] - 1).max() < 0.05)
    ac = np.mean(wn[1:] * wn[:-1], 0) / np.mean(wn * wn, 0)
    check("whitened noise is white in time (|lag-1 autocorr| < 0.05)", np.abs(ac).max() < 0.05)

    # ── two overlapping spikes at fractional times ──
    TA, TB = raw_template(40, 1.0, 400), raw_template(110, 0.7, 300)
    sig2 = noise(rng, N, 3.0)
    place(sig2, TA, 50000.30)
    place(sig2, TB, 50009.65, 0.9)
    rd2 = lambda a, b: sig2[max(0, a):b]                                         # noqa: E731
    nm2 = fd.fit_noise_model(rd2, 0, N, np.array([50000]))
    bank = fd.build_bank({0: transformed_template(TA), 1: transformed_template(TB)}, nm2, NSAMP, PEAK, R=8,
                         amp_sd={0: 0.2, 1: 0.2})
    r = fd.deconvolve(rd2, np.array([50000]), np.array([0]), NSAMP, PEAK, SR, nm=nm2, bank=bank,
                      s0=0, s1=N)
    pl = np.sort(r.placements, order="t")
    check("collision: exactly two placements", len(pl) == 2)
    check("collision: units 0 then 1", list(pl["unit"]) == [0, 1])
    check("collision: times within 0.15 sample",
          abs(pl["t"][0] - 50000.30) < 0.15 and abs(pl["t"][1] - 50009.65) < 0.15)
    check("collision: amplitudes within 5 %", abs(pl["amp"][0] - 1.0) < 0.05 and abs(pl["amp"][1] - 0.9) < 0.05)
    check("collision: detection labelled 0, partner 1 at +9.65 smp",
          r.primary[0] == 0 and r.partner[0] == 1 and abs(r.partner_dt[0] - 9.65) < 0.15)

    # ── NEGATIVE CONTROL: pure noise, templates absent ──
    bankn = fd.build_bank({k: transformed_template(raw_template(d, w, a))
                           for k, (d, w, a) in enumerate([(40, 1.0, 300), (100, 0.6, 200), (150, 1.1, 150)])},
                          nm, NSAMP, PEAK, R=4)
    fake = np.sort(rng.choice(np.arange(500, N - 500), 1500, replace=False))
    rn = fd.deconvolve(rd, fake, np.zeros(len(fake), int), NSAMP, PEAK, SR, nm=nm, bank=bankn, s0=0, s1=N)
    check(f"negative control: {len(rn.placements)} placements on {len(fake)} noise pseudo-detections (<= 1 %)",
          len(rn.placements) <= 0.01 * len(fake))

    # ── calibrate_margin ──
    check("calibrate_margin refuses below min_n", fd.calibrate_margin(np.arange(10.0), np.ones(10, bool)) == np.inf)
    m = np.r_[np.linspace(50, 30, 200), np.linspace(29, 0, 100)]
    agree = np.r_[np.ones(200, bool), np.zeros(100, bool)]
    check("calibrate_margin: lowest margin of the all-agree set", fd.calibrate_margin(m, agree, 0.999, 10.0) == 30.0)
    check("calibrate_margin honours the floor", fd.calibrate_margin(m, agree, 0.5, 40.0) >= 40.0)

    print(f"test_deconv: {ok} checks passed")


if __name__ == "__main__":
    main()
