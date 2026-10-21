#!/usr/bin/env python3
# test_adlp.py -- fiber-adlp (amplitude-dependent low-pass of spike snippets).
#
#   * lowpass: a cutoff at Nyquist is an exact passthrough; a fixed cutoff equals
#     scipy filtfilt of the same digital Butterworth (odd padding, padlen T-1); it is
#     zero-phase (a symmetric pulse keeps its peak sample); per-channel cutoffs filter
#     each channel at its own frequency.
#   * cutoff: the power law, its clip, and gamma 0 = uniform.
#   * the CLI on a synthetic session: passthrough settings write a byte-identical
#     .spk (the negative control the experiment uses); default settings smooth small
#     spikes more than large ones; the .pca keeps the input basis's window; the
#     sidecar holds one amplitude and cutoff per spike.
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from fiber_kit import fiber_adlp as ad    # noqa: E402
from fiber_kit import fiber_pca as fpca   # noqa: E402

fails = 0


def check(ok, what):
    global fails
    print(("  ok:   " if ok else "  FAIL: ") + what)
    fails += not ok


def main():
    rng = np.random.default_rng(2)
    sr = 32552.0
    w = rng.normal(0, 200, (300, 52, 4))

    print("lowpass / cutoff:")
    check(np.abs(ad.lowpass(w, sr, np.full(300, sr)) - w).max() < 1e-9, "fc at Nyquist is a passthrough")
    from scipy.signal import butter, filtfilt
    b, a = butter(2, 5000 / (sr / 2))
    ref = filtfilt(b, a, w, axis=1, padtype="odd", padlen=51)
    got = ad.lowpass(w, sr, np.full(300, 5000.0), order=2)
    err = np.abs(ref - got).max() / np.abs(ref).max()
    check(err < 1e-9, f"fixed 5 kHz equals filtfilt(butter(2), padlen 51) (rel err {err:.1e})")
    t = np.arange(52) - 26
    pulse = np.repeat((-np.exp(-0.5 * (t / 1.5) ** 2))[None, :, None], 2, axis=2)
    y = ad.lowpass(pulse, sr, np.array([3000.0]))
    check(int(np.argmin(y[0, :, 0])) == 26, "zero phase: a symmetric pulse keeps its peak sample")
    check(y[0, 26, 0] > pulse[0, 26, 0], "the low-pass flattens the pulse")
    yc = ad.lowpass(pulse, sr, np.array([[3000.0, sr]]))
    check(np.abs(yc[0, :, 1] - pulse[0, :, 1]).max() < 1e-9 and np.abs(yc[0, :, 0] - y[0, :, 0]).max() < 1e-9,
          "per-channel cutoffs: each channel at its own frequency")
    fc = ad.cutoff(np.array([500.0, 1000.0, 2000.0, 1e6]), 1000.0, 9000.0, 1.2, 2500.0, 30000.0)
    check(np.isclose(fc[1], 9000.0) and np.isclose(fc[2], 9000.0 * 2 ** 1.2) and fc[0] < fc[1] < fc[2],
          "power law fc = fc_ref (A/A_ref)^gamma")
    check(fc[3] == 30000.0 and ad.cutoff(np.array([1.0]), 1000.0, 9000.0, 1.2, 2500.0, 16000.0)[0] == 2500.0,
          "clipped to [fc_min, fc_max]")
    check(np.allclose(ad.cutoff(np.array([10.0, 1e5]), 1000.0, 6000.0, 0.0, 2500.0, 16000.0), 6000.0),
          "gamma 0 is a uniform low-pass")

    print("CLI:")
    with tempfile.TemporaryDirectory() as d:
        base = os.path.join(d, "mini")
        n, ns, nc, peak = 2000, 52, 4, 26
        amp = np.exp(rng.uniform(np.log(300), np.log(6000), n))
        tmpl = -np.exp(-0.5 * (t / 2.0) ** 2)
        x = rng.normal(0, 150, (n, ns, nc)) + amp[:, None, None] * tmpl[None, :, None] * np.array([1, .6, .3, .1])
        x.round().astype(np.int16).tofile(base + ".spk.standard.1")
        np.sort(rng.integers(0, 10 ** 7, n)).astype("<i8").tofile(base + ".res.1")
        means, evec = fpca.fit_basis(x[:, 14:39, :], 3, centered=False)
        fpca.write_pcad(base + ".pca.standard.1", means, evec, 14, centered=0)
        open(base + ".yaml", "w").write(
            "acquisitionSystem: {nBits: 16, nChannels: 4, samplingRate: 32552}\n"
            "spikeDetection:\n  channelGroups:\n    - channels: [0, 1, 2, 3]\n"
            f"      nSamples: {ns}\n      peakSampleIndex: {peak}\n")
        env = {k: v for k, v in os.environ.items() if not k.startswith("FK_")}
        env["PYTHONPATH"] = os.path.join(HERE, "..", "src")

        def run(*args):
            r = subprocess.run([sys.executable, "-m", "fiber_kit.fiber_adlp", base, "1"] + list(args),
                               capture_output=True, text=True, env=env, timeout=600)
            if r.returncode:
                print(r.stdout[-2000:], r.stderr[-2000:])
            return r.returncode

        check(run("--out-variant", "standard_PASS", "--fc-min", "16276", "--no-pca") == 0, "passthrough run")
        check(open(base + ".spk.standard.1", "rb").read() == open(base + ".spk.standard_PASS.1", "rb").read(),
              "passthrough settings write a byte-identical .spk")
        check(not os.path.exists(base + ".pca.standard_PASS.1"), "--no-pca writes no basis")
        check(run() == 0, "default run")
        y = np.fromfile(base + ".spk.standard_ADLP.1", np.int16).reshape(n, ns, nc).astype(float)
        side = np.load(base + ".adlp.standard_ADLP.1.npz")
        check(side["fc"].shape == (n,) and side["amp"].shape == (n,), "sidecar: one amplitude and cutoff per spike")
        hf = np.abs(np.diff(y - x, axis=1)).mean((1, 2))      # what the filter removed, high-frequency
        small, large = amp < np.quantile(amp, 0.2), amp > np.quantile(amp, 0.8)
        check(np.median(hf[small]) > 3 * np.median(hf[large]) + 1e-9,
              f"small spikes smoothed more than large ones ({np.median(hf[small]):.1f} vs {np.median(hf[large]):.1f})")
        check(np.corrcoef(side["amp"], side["fc"])[0, 1] > 0.5, "cutoff rises with amplitude")
        b2 = fpca.read_pcad(base + ".pca.standard_ADLP.1")
        check((b2["recShift"], b2["data2use"], b2["nComp"]) == (14, 25, 3), "the .pca keeps the input basis's window")
        check(run("--out-variant", "standard") != 0, "refuses to overwrite its input")

    print(f"\ntest_adlp: {'FAIL' if fails else 'PASS'} ({fails} failed)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
