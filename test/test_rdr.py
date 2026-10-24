"""fiber-rdr: residual-derivative RMS against the clean amplitude line.

Synthetic one-group session (standard .spk, white noise sigma 60): six clean units of different
amplitudes (sub-sample jitter and a 4% gain spread, so their D rises with amplitude as real cells do) and
one MIXED cluster -- two cells with the same footprint, the second one 2/3 the amplitude with a
wider trough 2 samples later.

  * noise_sigma_d is exact on a .nst covariance (white noise: sigma * sqrt 2); the window lead-in
    fallback agrees within 5%;
  * the clean line fitted on all clusters (no labels) has the mixed cluster above it: ratio >= 2, every
    clean unit <= 1.5;
  * the CLI writes the table and flags exactly the mixed cluster, with the .nst and without it
    (the lead-in estimate of sigma_d is ~1% off per channel; that moves the ratio of the smallest unit,
    whose expected excess is tiny here, so only the flag and the mixed cluster's ratio are compared).
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from fiber_kit import fiber_rdr as rdr     # noqa: E402
from fiber_kit import neuro_io as nio      # noqa: E402

NS, NC, PEAK, SIG = 32, 4, 15, 60.0
fails = 0


def check(ok, what):
    global fails
    print(("  ok:   " if ok else "  FAIL: ") + what); fails += not ok


def shape(t0, w, amp, foot):
    t = np.arange(NS)
    s = -np.exp(-0.5 * ((t - t0) / w) ** 2) + 0.3 * np.exp(-0.5 * ((t - t0 - 5) / 2.5) ** 2)
    return amp * s[:, None] * np.asarray(foot)[None, :]


def build(td, rng):
    foot = [1.0, 0.7, 0.4, 0.15]
    W, lab = [], []
    for u, amp in enumerate((400, 700, 1000, 1400, 1800, 2300)):
        for _ in range(400):
            g = 1 + rng.normal(0, 0.04); j = rng.uniform(-0.5, 0.5)
            W.append(shape(PEAK - 1 + j, 1.4, amp * g, foot) + rng.normal(0, SIG, (NS, NC))); lab.append(u + 2)
    for i in range(600):                                            # cluster 8: two cells, one footprint
        j = rng.uniform(-0.5, 0.5)
        s = shape(PEAK - 1 + j, 1.4, 1200, foot) if i % 2 else shape(PEAK + 1 + j, 2.6, 800, foot)
        W.append(s + rng.normal(0, SIG, (NS, NC))); lab.append(8)
    W = np.clip(np.rint(np.array(W)), -32768, 32767).astype(np.int16); lab = np.array(lab)
    perm = rng.permutation(len(lab)); W, lab = W[perm], lab[perm]
    res = np.sort(rng.choice(np.arange(20000 * 3600), len(lab), replace=False))
    base = os.path.join(td, "mini")
    nio.write_spk(base, 1, W, variant="standard")
    nio.write_res_file(base + ".res.1", res)
    nio.write_clu_file(base + ".clu.stderiv.1.fs", lab)
    with open(base + ".yaml", "w") as fh:
        fh.write(f"acquisitionSystem: {{nBits: 16, nChannels: {NC}, samplingRate: 20000}}\n"
                 "spikeDetection:\n  channelGroups:\n    - channels: [0, 1, 2, 3]\n"
                 f"      nSamples: {NS}\n      peakSampleIndex: {PEAK}\n")
    return base, W, lab


def cli(base, *extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("FK_")}
    env["FK_CONFIG"] = os.devnull
    env["PYTHONPATH"] = os.path.join(HERE, "..", "src")
    r = subprocess.run([sys.executable, "-m", "fiber_kit.fiber_rdr", base, "1", "--clu-stage", "fs"] + list(extra),
                       capture_output=True, text=True, env=env, timeout=600)
    return r.returncode, r.stdout + r.stderr


def table(path):
    rows = {}
    with open(path) as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            v = dict(zip(hdr, line.rstrip("\n").split("\t")))
            rows[int(v["cluster"])] = (float(v["ratio"]), int(v["flag"]))
    return rows


def main():
    rng = np.random.default_rng(11)
    cov = np.eye(NS * NC) * SIG ** 2
    sd = rdr.noise_sigma_d(cov, NS, NC)
    check(np.allclose(sd, SIG * np.sqrt(2)), f"noise_sigma_d on a white .nst covariance = sigma*sqrt2 ({sd[0]:.2f})")
    with tempfile.TemporaryDirectory() as td:
        base, W, lab = build(td, rng)
        s_w, sd_w = rdr.noise_from_windows(W, n=3000, rng=rng)
        check(np.all(np.abs(sd_w / (SIG * np.sqrt(2)) - 1) < 0.05) and np.all(np.abs(s_w / SIG - 1) < 0.05),
              f"window lead-in noise estimate within 5% (sigma {s_w.round(1)}, sigma_d {sd_w.round(1)})")
        prof = {u: rdr.cluster_profile(W, np.flatnonzero(lab == u), peak=PEAK, sigma=np.full(NC, SIG),
                                       sigma_d=np.full(NC, SIG * np.sqrt(2)), rng=rng) for u in range(2, 9)}
        A = np.array([prof[u][0] for u in prof]); M = np.array([prof[u][1] for u in prof])
        d0, k = rdr.fit_clean_line(A, M)
        Q = np.nanmax(rdr.excess_ratio(A, M, d0, k), 1)
        clean, mixed = Q[:6], Q[6]
        check(mixed >= 2.0 and np.all(clean <= 1.5),
              f"clean line {d0:.3f} + {k:.5f} A^2: mixed cluster ratio {mixed:.2f}, clean {np.round(clean, 2).tolist()}")
        rc, out = cli(base)
        check(rc == 0 and "no .nst" in out, "CLI runs without a .nst (window lead-in noise)" + ("" if rc == 0 else "\n" + out[-2000:]))
        t1 = table(base + ".rdr.stderiv.1.fs.tsv")
        check({u for u, (r, f) in t1.items() if f} == {8}, f"CLI flags exactly the mixed cluster ({t1})")
        nio.write_nst(base + ".nst.standard.1", dict(nsamp=NS, nchan=NC, sigma=np.full(NC, SIG),
                                                     mean=np.zeros((NS, NC)), cov=cov))
        rc, out = cli(base, "--out", os.path.join(td, "nst.tsv"))
        t2 = table(os.path.join(td, "nst.tsv"))
        check(rc == 0 and "nst.standard.1" in out and abs(t2[8][0] - t1[8][0]) < 0.25
              and {u for u, (r, f) in t2.items() if f} == {8},
              "with the .nst: the same flag, the mixed ratio within 0.25 of the fallback's "
              f"({ {u: (t1[u][0], t2[u][0]) for u in t1} })")
    print(f"\ntest_rdr: {'FAIL' if fails else 'PASS'} ({fails} failed)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
