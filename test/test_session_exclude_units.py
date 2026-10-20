"""fiber-session --in-stage / --exclude-units / --exclude-as.

A synthetic one-chunk session (four units in a real .fil) and a 'curated'
input clu at stage `cur` whose clusters are the true units (ids 2-5) plus
noise (1).

  * parse_unit_list: ids, ranges, whitespace, duplicates; bad tokens refused;
  * NEGATIVE CONTROL: --in-stage without --exclude-units writes the same clu
    as a run with no input stage at all;
  * carry: the excluded units' spikes form exactly one output cluster each,
    no new fiber holds any of them, each is one atom of its own, and the
    log reports input -> output ids that match the written clu;
  * noise: the excluded spikes are cluster 0, the rest as in carry;
  * ids not in the input clu are reported, not fatal; excluding noise (1)
    leaves those spikes in cluster 0 under carry;
  * refusals: --exclude-units without --in-stage; an input clu whose length
    differs from the .res; an output that would overwrite the input clu;
    a malformed list.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from fiber_kit import fiber_session as fs           # noqa: E402
from fiber_kit import fiber_stderiv as fsd          # noqa: E402
from fiber_kit import neuro_io as nio               # noqa: E402
from test_deconv import NSAMP, PEAK, SR, noise, place, raw_template   # noqa: E402


def build(td, rng):
    base = os.path.join(td, "syn")
    dur = int(60 * SR)
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
    res = np.round(times).astype(np.int64)
    res.tofile(f"{base}.res.1")
    idx = res[:, None] + np.arange(-PEAK - 1, NSAMP - PEAK)[None, :]
    fsd.apply_stderiv_transform(fil[idx].astype(np.float64)).tofile(f"{base}.spk.stderiv.1")
    with open(f"{base}.yaml", "w") as f:
        f.write("acquisitionSystem: {nChannels: 8, samplingRate: 32552, nBits: 16}\n"
                "spikeDetection:\n  channelGroups:\n"
                "    - channels: [0,1,2,3,4,5,6,7]\n      nSamples: 42\n      peakSampleIndex: 21\n"
                "      nFeatures: 3\n")
    cur = labs + 2                                   # curated: units 2..5
    cur[rng.random(len(cur)) < 0.03] = 1             # a sprinkle of noise
    nio.write_clu(base, 1, cur, variant="stderiv", tag="cur")
    nio.write_clu(base, 1, cur[:-5], variant="stderiv", tag="short")
    return base, cur


def main():
    ok = 0

    def check(name, cond):
        nonlocal ok
        assert cond, name
        ok += 1
        print(f"  ok  {name}")

    # ── parser ──
    check("parse_unit_list: ids, ranges, spaces, duplicates",
          fs.parse_unit_list(" 7,3, 12-14,3 ") == [3, 7, 12, 13, 14])
    check("parse_unit_list: empty -> []", fs.parse_unit_list("") == [])
    for bad in ("3,x", "5-2", "-3", "3-", "1.5"):
        try:
            fs.parse_unit_list(bad)
            refused = False
        except ValueError:
            refused = True
        check(f"parse_unit_list refuses '{bad}'", refused)

    rng = np.random.default_rng(1)
    with tempfile.TemporaryDirectory() as td:
        base, cur = build(td, rng)
        env = {k: v for k, v in os.environ.items() if not k.startswith("FK_")}
        env["PYTHONPATH"] = os.pathsep.join([os.path.join(HERE, "..", "src"),
                                             env.get("PYTHONPATH", "")]).rstrip(os.pathsep)

        def run(extra, stage):
            r = subprocess.run([sys.executable, "-m", "fiber_kit.fiber_session", base, "1", "--method", "stderiv",
                                "--no-link", "--chunk-min", "5", "--overlap-min", "0", "--min-group", "60",
                                "--clu-stage", stage] + extra, capture_output=True, text=True, env=env, timeout=900)
            return r, r.stdout + r.stderr

        def clu(stage, kind="clu"):
            return np.asarray(nio.read_clu_file(nio.session_path(base, kind, 1, variant="stderiv", tag=stage))[1])

        r0, o0 = run([], "plain")
        check("baseline run exits 0", r0.returncode == 0)
        r1, o1 = run(["--in-stage", "cur"], "ctl")
        check("--in-stage alone exits 0 and says nothing is excluded",
              r1.returncode == 0 and "nothing excluded" in o1)
        check("NEGATIVE CONTROL: --in-stage without --exclude-units writes the baseline clu",
              np.array_equal(clu("plain"), clu("ctl")))

        ex = [2, 4]
        exm = np.isin(cur, ex)
        r2, o2 = run(["--in-stage", "cur", "--exclude-units", "2,4,99"], "carry")
        if r2.returncode:
            print(o2[-3000:])
        check("carry run exits 0", r2.returncode == 0)
        c2 = clu("carry")
        for u in ex:
            m = cur == u
            ids = np.unique(c2[m])
            check(f"unit {u}: its {m.sum()} spikes are one output cluster ({ids.tolist()})",
                  ids.size == 1 and ids[0] > 1)
            check(f"unit {u}: that cluster holds nothing else", np.array_equal(c2 == ids[0], m))
            check(f"unit {u}: log maps it to {ids[0]}", f"{u}->{ids[0]}" in o2)
        check("carried clusters are distinct", np.unique(c2[exm]).size == len(ex))
        check("unknown id 99 reported, not fatal", "ignored): 99" in o2)
        clc = clu("carry", "clc")
        check("each carried unit is exactly one atom of its own",
              all(np.unique(clc[cur == u]).size == 1 for u in ex)
              and not set(np.unique(clc[exm]).tolist()) & set(np.unique(clc[~exm]).tolist()))
        check("the remaining spikes were sorted into fibers (not left empty)", (c2[~exm] > 0).mean() > 0.5)

        r3, o3 = run(["--in-stage", "cur", "--exclude-units", "2,4", "--exclude-as", "noise"], "asnoise")
        check("noise run exits 0", r3.returncode == 0)
        c3 = clu("asnoise")
        check("noise: every excluded spike is cluster 0", (c3[exm] == 0).all())
        check("noise and carry sort the remaining spikes identically",
              np.array_equal(c3[~exm], c2[~exm]))

        r4, o4 = run(["--in-stage", "cur", "--exclude-units", "1"], "excl_noise")
        check("excluding input noise (1) under carry leaves those spikes in cluster 0",
              r4.returncode == 0 and (clu("excl_noise")[cur == 1] == 0).all())

        for name, extra, msg in (
                ("--exclude-units without --in-stage", ["--exclude-units", "2"], "give --in-stage"),
                ("input clu shorter than the .res", ["--in-stage", "short", "--exclude-units", "2"],
                 "carries its own .res"),
                ("malformed list", ["--in-stage", "cur", "--exclude-units", "2,x"], "is not an id"),
                ("missing input stage", ["--in-stage", "nope", "--exclude-units", "2"], "--in-stage:")):
            r, o = run(extra, "refused")
            check(f"refused: {name}", r.returncode != 0 and msg in o)
        r, o = run(["--in-stage", "cur", "--exclude-units", "2"], "cur")
        check("refused: output would overwrite the input clu",
              r.returncode != 0 and "would overwrite the input" in o
              and np.array_equal(clu("cur"), cur))

    print(f"test_session_exclude_units: {ok} checks passed")


if __name__ == "__main__":
    main()
