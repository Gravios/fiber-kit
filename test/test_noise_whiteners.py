#!/usr/bin/env python3
# test_noise_whiteners.py -- the two whitener sources that need no .fil.
#
#   * neuro_io.read_nst / write_nst: NSTE v1 round trip; a wrong magic, a newer version
#     and a short body are refused.
#   * fiber_lib.nst_whitener: on noise windows with a known covariance (AR(1) in time,
#     correlated across channels), the masked sub-block it whitens with is the right one:
#     whitened masked noise has identity covariance and zero mean, and a wrong mask
#     (shifted) or a channel-major layout does not.
#   * fiber_lib.spk_chunk_whitener: snippets whose pre-peak samples are that same noise,
#     with a spike after the peak, give a whitener under which held-out noise windows of
#     the MASK span are white -- although the mask is wider than the pre-peak segment's
#     reach in a single snippet would allow without the stationary (block-Toeplitz) model.
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from fiber_kit import neuro_io as nio   # noqa: E402
from fiber_kit import fiber_lib as fl   # noqa: E402

fails = 0


def check(ok, what):
    global fails
    print(("  ok:   " if ok else "  FAIL: ") + what)
    fails += not ok


def ar_noise(n, T, C, rng, phi=0.6, mix=0.5, sd=100.0):
    """n windows of T samples x C channels: AR(1) in time, neighbouring channels mixed."""
    e = rng.normal(0, sd, (n, T + 50, C))
    x = np.zeros_like(e)
    for t in range(1, T + 50):
        x[:, t] = phi * x[:, t - 1] + e[:, t]
    x = x[:, 50:]
    return x + mix * np.roll(x, 1, axis=2) + 40.0          # + a DC offset per channel


def white_err(X, W, nmean):
    Z = (X - nmean) @ W
    C = np.cov(Z, rowvar=False)
    return float(np.abs(C - np.eye(C.shape[0])).max()), float(np.abs(Z.mean(0)).max())


def main():
    rng = np.random.default_rng(5)
    T, C, peak = 52, 4, 26
    mask = fl.build_masks(T, peak).full                     # 15 samples around the peak

    print("read_nst / write_nst:")
    with tempfile.TemporaryDirectory() as d:
        noise = ar_noise(20000, T, C, rng).reshape(20000, -1)
        st = dict(nsamp=T, nchan=C, peak=peak, guard=T, passes=2, n_requested=20000,
                  n_drawn=20000, n_kept=19990, n_rejected_flat=4, n_rejected_outlier=6,
                  zmax=5.0, seed=1.0, sigma=noise.reshape(-1, T, C).std((0, 1)),
                  mean=noise.mean(0).reshape(T, C), cov=np.cov(noise, rowvar=False))
        p = os.path.join(d, "s.nst.standard.1")
        nio.write_nst(p, st)
        back = nio.read_nst(p)
        check(all(np.array_equal(np.asarray(back[k]), np.asarray(st[k])) for k in st),
              "round trip is exact (every field)")
        raw = bytearray(open(p, "rb").read())
        for what, mut in (("wrong magic", lambda b: b.__setitem__(slice(0, 4), b"XXXX")),
                          ("newer version", lambda b: b.__setitem__(slice(4, 8),
                                                                    np.int32(nio.NST_VERSION[0] + 1).tobytes())),
                          ("short body", lambda b: b.__delitem__(slice(len(b) - 8, len(b))))):
            bb = bytearray(raw); mut(bb)
            q = os.path.join(d, "bad.nst"); open(q, "wb").write(bytes(bb))
            try:
                nio.read_nst(q); ok = False
            except ValueError:
                ok = True
            check(ok, f"{what} refused")
        check(nio.classify_type("nst") == "MethodSpecific", "nst is MethodSpecific (strict)")

    print("nst_whitener:")
    test = ar_noise(20000, T, C, rng)
    Xm = test[:, mask, :].reshape(len(test), -1)
    W, nmean, nk = fl.nst_whitener(st, mask)
    e_c, e_m = white_err(Xm, W, nmean)
    check(e_c < 0.06 and e_m < 0.05, f"held-out masked noise is white (cov err {e_c:.3f}, mean {e_m:.3f})")
    check(nk == 19990, "n_kept passed through")
    bool_mask = np.zeros(T, bool); bool_mask[mask] = True
    Wc, _, _ = fl.nst_whitener(st, bool_mask)
    check(np.allclose(Wc, W), "boolean mask accepted")
    # a channel-major reading of the same block is not the right whitener
    idx_cm = (np.arange(C)[:, None] * T + mask[None, :]).ravel()
    Ccm = st["cov"][np.ix_(idx_cm, idx_cm)]
    ev, V = np.linalg.eigh(Ccm); Wcm = V @ np.diag(ev ** -0.5) @ V.T
    check(white_err(Xm, Wcm, nmean)[0] > 0.3, "a channel-major sub-block does NOT whiten (layout matters)")
    try:
        fl.nst_whitener(st, np.arange(40, 60)); ok = False
    except ValueError:
        ok = True
    check(ok, "a mask past the .nst window is refused")

    print("spk_chunk_whitener:")
    n = 30000
    sn = ar_noise(n, T, C, rng)
    t = np.arange(T) - peak
    tmpl = -np.exp(-0.5 * (t / 2.0) ** 2) + 0.4 * np.exp(-0.5 * ((t - 7) / 4.0) ** 2)
    tmpl[t < -8] = 0.0                                       # the spike starts 8 samples before the peak
    amp = rng.uniform(500, 3000, n)
    waves = sn + amp[:, None, None] * tmpl[None, :, None] * np.array([1.0, 0.6, 0.3, 0.1])
    res = np.cumsum(rng.integers(200, 2000, n))              # isolated spikes
    Ws, ns_, nseg = fl.spk_chunk_whitener(waves, res, mask, peak)
    e_c, e_m = white_err(Xm, Ws, ns_)
    check(e_c < 0.08 and e_m < 0.05, f"held-out masked noise is white (cov err {e_c:.3f}, mean {e_m:.3f})")
    check(nseg == 20000, f"segments capped at max_seg ({nseg})")
    # the same estimate from the windows' SPIKE part is not a noise model
    Wbad, nbad, _ = fl.spk_chunk_whitener(np.roll(waves, -(peak - 6), axis=1), res, mask, peak)
    check(white_err(Xm, Wbad, nbad)[0] > 0.3, "segments that contain the spike do NOT give a noise whitener")
    close = res.copy(); close[1::2] = close[0::2] + 10      # every other spike right after another
    _, _, nseg2 = fl.spk_chunk_whitener(waves, close, mask, peak, max_seg=10 ** 9)
    check(nseg2 == n // 2, f"snippets right after another spike are skipped ({nseg2} of {n})")

    print(f"\ntest_noise_whiteners: {'FAIL' if fails else 'PASS'} ({fails} failed)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
