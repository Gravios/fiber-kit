"""fiber-session: real spikes stay in the sort; only flagged artifacts reach cluster 0; cluster 1 stays free.

A synthetic one-group session (.fil + .spk.stderiv extracted from it): three dense units and one SPARSE
unit (25 spikes, below --fine-min-group 40), plus a mask from "fiber-flag-artifacts" marking 12 spikes.

  * control (no new flags): the sparse unit's spikes go to cluster 0 (the shedding being fixed);
  * no output ever uses cluster 1 (global fiber 0 used to be written as cluster 1, Klusters' noise
    cluster): every real fiber id is >= 2;
  * --keep-small-groups --assign-unsorted: nothing is shed -- cluster 0 is empty;
  * ... plus --exclude-clu <mask>: cluster 0 holds EXACTLY the masked spikes;
  * the sparse unit keeps its own cluster(s) rather than being absorbed by a dense unit.
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from fiber_kit import fiber_stderiv as fst    # noqa: E402
from fiber_kit import neuro_io as nio         # noqa: E402

SR = 20000
NSAMP, NCH, NTOTAL, PEAK = 32, 3, 4, 15
DUR_S = 60


def _build(td, rng):
    base = os.path.join(td, "mini")
    n_t = DUR_S * SR
    fil = rng.normal(0, 25, (n_t, NTOTAL))
    t = np.arange(-8, 9)
    shapes = [np.array([1.0, 0.5, 0.2]), np.array([0.3, 1.0, 0.6]), np.array([0.2, 0.4, 1.0]),
              np.array([0.6, 0.2, 0.9])]
    true, unit = [], []
    for u, (amp, sh, n) in enumerate(zip((2600.0, 1800.0, 2200.0, 3000.0), shapes, (260, 260, 260, 25))):
        tt = np.sort(rng.choice(np.arange(SR, n_t - SR, 7), n, replace=False))
        wav = -amp * np.exp(-0.5 * (t / 1.6) ** 2) + 0.35 * amp * np.exp(-0.5 * ((t - 5) / 2.5) ** 2)
        for s in tt:
            fil[s + t, :NCH] += wav[:, None] * sh[None, :]
        true.append(tt); unit.append(np.full(n, u))
    fil.round().clip(-32768, 32767).astype(np.int16).tofile(base + ".fil")
    true = np.concatenate(true); unit = np.concatenate(unit)
    o = np.argsort(true); true, unit = true[o], unit[o]
    true.astype(np.int64).tofile(base + ".res.1")
    from fiber_kit.fiber_realign import reextract_from_fil
    raw = reextract_from_fil(nio.open_signal(base + ".fil", NTOTAL), [0, 1, 2], true, NSAMP + 1, PEAK + 1)
    nio.write_spk(base, 1, raw[:, 1:, :], variant="standard")
    nio.write_spk(base, 1, fst.apply_stderiv_transform(raw), variant="stderiv")
    with open(base + ".yaml", "w") as fh:
        fh.write(f"acquisitionSystem: {{nBits: 16, nChannels: {NTOTAL}, samplingRate: {SR}}}\n"
                 "spikeDetection:\n  channelGroups:\n    - channels: [0, 1, 2]\n"
                 f"      nSamples: {NSAMP}\n      peakSampleIndex: {PEAK}\n")
    mask = np.zeros(true.size, np.int32)
    mask[rng.choice(np.flatnonzero(unit < 3), 12, replace=False)] = 1
    nio.write_clu_file(base + ".clu.standard.1.artifact", mask)
    return base, unit, mask


def _env():
    env = {k: v for k, v in os.environ.items() if not k.startswith("FK_")}
    env["FK_CONFIG"] = os.devnull
    env["PYTHONPATH"] = os.pathsep.join([os.path.join(HERE, "..", "src"),
                                         env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    return env


def session(base, *extra):
    r = subprocess.run([sys.executable, "-m", "fiber_kit.fiber_session", base, "1",
                        "--method", "stderiv", "--chunk-min", "0.5", "--overlap-min", "0",
                        "--min-group", "40", "--fine-min-group", "40", "--clu-stage", "fs",
                        "--realign", "off", "--whiten-source", "spk"] + list(extra),
                       capture_output=True, text=True, env=_env(), timeout=900)
    clu = nio.read_clu_file(f"{base}.clu.stderiv.1.fs")[1] if r.returncode == 0 else None
    return r.returncode, r.stdout + r.stderr, clu


def main():
    ok = 0

    def check(name, cond, out=""):
        nonlocal ok
        if not cond and out:
            print(out[-4000:])
        assert cond, name
        ok += 1
        print(f"  ok  {name}")

    rng = np.random.default_rng(5)
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "src"); os.mkdir(src)
        base0, unit, mask = _build(src, rng)

        def fresh(name):
            d = os.path.join(td, name); shutil.copytree(src, d)
            return os.path.join(d, "mini")

        rc, out, c0 = session(fresh("control"))
        check("control run exits 0", rc == 0, out)
        check(f"control: the sparse unit is shed to cluster 0 ({np.sum((c0 == 0) & (unit == 3))}/25)",
              np.sum((c0 == 0) & (unit == 3)) >= 20, out)
        check(f"control: no spike in cluster 1, real ids start at 2 (min {c0[c0 > 0].min()})",
              not np.any(c0 == 1) and c0[c0 > 0].min() >= 2, out)

        rc, out, c1 = session(fresh("keep"), "--keep-small-groups", "--assign-unsorted")
        check("--keep-small-groups --assign-unsorted exits 0", rc == 0, out)
        check(f"nothing shed: cluster 0 empty ({np.sum(c1 == 0)}), cluster 1 empty ({np.sum(c1 == 1)})",
              not np.any(c1 <= 1), out)
        # the sparse unit's spikes sit in clusters where IT is the majority, not inside a dense unit's cluster
        own = sum(int(np.sum((c1 == c) & (unit == 3))) for c in np.unique(c1[unit == 3])
                  if np.sum((c1 == c) & (unit == 3)) > 0.5 * np.sum(c1 == c))
        check(f"the sparse unit keeps cluster(s) of its own ({own}/25 in clusters it dominates)", own >= 20, out)
        check("the run reports the kept / assigned spikes", "small groups kept" in out or "assigned to their nearest" in out, out)

        rc, out, c2 = session(fresh("mask"), "--keep-small-groups", "--assign-unsorted",
                              "--exclude-clu", os.path.join(td, "mask", "mini.clu.standard.1.artifact"))
        check("... with --exclude-clu exits 0", rc == 0, out)
        check(f"cluster 0 holds exactly the {int(mask.sum())} flagged artifacts",
              np.array_equal(c2 == 0, mask == 1) and not np.any(c2 == 1), out)

    print(f"test_session_no_shed: {ok} checks passed")


if __name__ == "__main__":
    main()
