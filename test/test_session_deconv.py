"""fiber-session --deconv end to end.

A synthetic one-chunk session whose .fil really contains the spikes (four
units, collisions censored at detection), run three ways:
  * --deconv: exits 0, writes the .deconv.<method>.<elec>.<stage>.tsv manifest,
    and .fibers carries deconv_agree -- >= 0.95 on pure fibers, within 0.05 of
    the true purity on contaminated ones (it is a contamination measure);
  * --no-deconv: the deconv columns are NaN, no manifest, and every label is
    what the --deconv run wrote (--deconv alone changes nothing);
  * --deconv --deconv-assign: moves exactly the spikes the manifest marks,
    all from noise, onto their own cell's fiber, as new child atoms.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from fiber_kit import fiber_stderiv as fsd        # noqa: E402
from fiber_kit import neuro_io as nio             # noqa: E402
from test_deconv import (NSAMP, PEAK, SR, noise, place,   # noqa: E402
                         raw_template, transformed_template)


def main():
    ok = 0

    def check(name, cond):
        nonlocal ok
        assert cond, name
        ok += 1
        print(f"  ok  {name}")

    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as td:
        base = os.path.join(td, "syn")
        dur = int(90 * SR)
        units = [(40, 1.0, 420, 8.0), (100, 0.9, 320, 10.0), (130, 0.55, 260, 20.0), (70, 1.0, 200, 8.0)]
        times, labs = [], []
        for k, (_, _, _, rate) in enumerate(units):
            t = 1000.0
            while True:
                t += rng.exponential(SR / rate) + 0.003 * SR
                if t >= dur - 1000:
                    break
                times.append(t); labs.append(k)
        o = np.argsort(times); times = np.array(times)[o]; labs = np.array(labs)[o]
        fil = noise(rng, dur, 14.0)
        Ts = [raw_template(d, w, a) for d, w, a, _ in units]
        for t, k in zip(times, labs):
            place(fil, Ts[k], t)
        fil = np.clip(np.round(fil), -32768, 32767).astype(np.int16)
        fil.tofile(f"{base}.fil")
        amp = np.array([np.abs(transformed_template(Ts[k])).max() for k in labs])
        keep = np.ones(len(times), bool)
        for i in range(len(times)):                                  # censor the smaller of close pairs
            j0 = np.searchsorted(times, times[i] - 10); j1 = np.searchsorted(times, times[i] + 10, "right")
            keep[i] = not any(amp[j] > amp[i] for j in range(j0, j1) if j != i)
        res = np.round(times[keep]).astype(np.int64)
        res.tofile(f"{base}.res.1")
        idx = res[:, None] + np.arange(-PEAK - 1, NSAMP - PEAK)[None, :]
        spk = fsd.apply_stderiv_transform(fil[idx].astype(np.float64))
        spk.tofile(f"{base}.spk.stderiv.1")
        with open(f"{base}.yaml", "w") as f:
            f.write("acquisitionSystem: {nChannels: 8, samplingRate: 32552, nBits: 16}\n"
                    "spikeDetection:\n  channelGroups:\n"
                    "    - channels: [0,1,2,3,4,5,6,7]\n      nSamples: 42\n      peakSampleIndex: 21\n"
                    "      nFeatures: 3\n")
        env = {k: v for k, v in os.environ.items() if not k.startswith("FK_")}
        env["PYTHONPATH"] = os.pathsep.join(
            [os.path.join(os.path.dirname(__file__), "..", "src"), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)

        def run(extra, stage):
            return subprocess.run(
                [sys.executable, "-m", "fiber_kit.fiber_session", base, "1", "--method", "stderiv",
                 "--no-link", "--chunk-min", "5", "--overlap-min", "0", "--min-group", "60",
                 "--clu-stage", stage] + extra, capture_output=True, text=True, env=env, timeout=900)

        r1 = run(["--deconv"], "dc")
        out = r1.stdout + r1.stderr
        if r1.returncode != 0:
            print(out[-4000:])
        check("fiber-session --deconv exits 0", r1.returncode == 0)
        man = f"{base}.deconv.stderiv.1.dc.tsv"
        check("wrote the .deconv manifest", os.path.exists(man))
        with open(man) as fh:
            rows = fh.read().splitlines()
        check("manifest has the header and rows",
              rows[0].startswith("spike\tres\tchunk\tclu\tdeconv_clu") and len(rows) > 100)
        check("logged the deconv summary", "collision partners" in out)
        fib = np.load(f"{base}.fibers.stderiv.1", allow_pickle=False)
        tl = labs[keep]
        _, clu = nio.read_clu_file(f"{base}.clu.stderiv.1.dc")
        pur = {}
        for g in np.unique(clu[clu > 0]):
            cnt = np.bincount(tl[clu == g], minlength=len(units))
            pur[int(g) - 1] = cnt.max() / cnt.sum()
        big = fib["nspk"] >= 100
        ag = fib["deconv_agree"]
        check(f"deconv_agree finite on all {big.sum()} well-populated fibers",
              big.any() and np.isfinite(ag[big]).all())
        pp = np.array([pur.get(int(g), np.nan) for g in fib["gid"]])
        clean_f = big & (pp >= 0.99)
        check(f"pure fibers (>= 0.99, {clean_f.sum()}) are corroborated (deconv_agree >= 0.95)",
              clean_f.any() and (ag[clean_f] >= 0.95).all())
        dirty = big & (pp < 0.9)
        check(f"impure fibers (< 0.9, {dirty.sum()}) are flagged: deconv_agree within 0.05 of purity",
              (np.abs(ag[dirty] - pp[dirty]) <= 0.05).all())
        r2 = run(["--no-deconv"], "nodc")
        check("fiber-session --no-deconv exits 0", r2.returncode == 0)
        fib2 = np.load(f"{base}.fibers.stderiv.1", allow_pickle=False)
        check("--no-deconv: deconv_agree is NaN, no manifest",
              np.isnan(fib2["deconv_agree"]).all() and not os.path.exists(f"{base}.deconv.stderiv.1.nodc.tsv"))
        _, clu2 = nio.read_clu_file(f"{base}.clu.stderiv.1.nodc")
        check("--deconv without --deconv-assign leaves every label as --no-deconv has it",
              np.array_equal(clu, clu2))

        r3 = run(["--deconv", "--deconv-assign"], "dca")
        check("fiber-session --deconv --deconv-assign exits 0", r3.returncode == 0)
        _, clu3 = nio.read_clu_file(f"{base}.clu.stderiv.1.dca")
        moved = np.flatnonzero(clu3 != clu)
        with open(f"{base}.deconv.stderiv.1.dca.tsv") as fh:
            mrows = [ln.split("\t") for ln in fh.read().splitlines()[1:]]
        flagged = {int(r_[0]) for r_ in mrows if r_[9] == "1"}
        check(f"--deconv-assign moves exactly the {len(flagged)} spikes the manifest marks rescued, all from noise",
              set(moved.tolist()) == flagged and (clu[moved] == 0).all())
        maj = {int(g): np.bincount(tl[clu == g], minlength=len(units)).argmax() for g in np.unique(clu[clu > 0])}
        prec = np.mean([maj[int(clu3[i])] == tl[i] for i in moved]) if moved.size else 1.0
        check(f"rescued spikes land on their own cell's fiber ({prec:.3f} >= 0.95)", prec >= 0.95)
        _, clc3 = nio.read_clu_file(f"{base}.clc.stderiv.1.dca")
        check("rescued spikes are parked as new child atoms (not merged into the fibers' atoms)",
              not set(np.unique(clc3[moved]).tolist()) & set(np.unique(clc3[np.setdiff1d(np.flatnonzero(clu3 > 0), moved)]).tolist()))

    print(f"test_session_deconv: {ok} checks passed")


if __name__ == "__main__":
    main()
