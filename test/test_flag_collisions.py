"""fiber-flag-collisions: probable collisions flagged before clustering, from off-spike energy.

A synthetic group of 6,000 detections of three units (52 x 8 records, noise sd 12), isolated in
time, of which
  * 600 hold a second unit's spike 10-20 samples from the peak (0.6-1.1 of its template), with
    no detection of its own  -> the score has to find them;
  * 200 are pairs of detections 15 samples apart                 -> the neighbour rule;
  * the rest are single spikes, amplitudes varying by 10 %.
Checked: at the default --score-min the score flags most of the hidden collisions and few single
spikes (rates printed; floors well under them), although one spike in ten is a collision; the
excess is where the partner is; every paired detection is flagged by the neighbour rule;
--max-share caps the flagged share; NEGATIVE CONTROL -- a group with no collision at all has under
2 % of its spikes flagged; the mask keeps an existing artifact entry, writes collisions as 2, is
redone (not accumulated) on a second run, and --dry-run writes nothing.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from fiber_kit import fiber_flag_collisions as ff  # noqa: E402
from fiber_kit import fiber_spkc as fs             # noqa: E402
from fiber_kit import neuro_io as nio              # noqa: E402

NS, NC, PEAK = 52, 8, 26


def templ(depth, amp, width):
    s = np.arange(NS)[:, None]; c = np.arange(NC)[None, :]
    w = -np.exp(-0.5 * ((s - PEAK) / (4.0 * width)) ** 2) + 0.3 * np.exp(-0.5 * ((s - PEAK - 9.0 * width) / (5.0 * width)) ** 2)
    return (amp * w * (np.exp(-0.5 * ((c * 20.0 - depth) / 28.0) ** 2) + 0.05)).astype(np.float32)


def group(rng, n, n_coll, n_pairs):
    T = [templ(40, 400, 1.0), templ(110, 320, 0.8), templ(75, 260, 1.2)]
    unit = rng.integers(0, 3, n)
    spk = np.stack([rng.normal(1.0, 0.1) * T[u] for u in unit]) + rng.normal(0, 12.0, (n, NS, NC)).astype(np.float32)
    res = 5000 + np.arange(n, dtype=np.int64) * 400
    coll = np.sort(rng.choice(n, n_coll, replace=False))
    lag = np.zeros(n, int)
    for i in coll:
        lag[i] = int(rng.integers(10, 21) * rng.choice([-1, 1]))
        spk[i] += rng.uniform(0.6, 1.1) * fs.shifted(T[(unit[i] + 1 + rng.integers(0, 2)) % 3], float(lag[i]))
    kind = np.zeros(n, int); kind[coll] = 1
    pair = rng.choice(np.setdiff1d(np.arange(n - 1), np.r_[coll, coll - 1]), n_pairs // 2, replace=False)
    pair = pair[np.isin(pair + 1, pair, invert=True)]
    for i in pair:                                             # i and i + 1 become two detections 15 samples apart
        res[i + 1] = res[i] + 15
        spk[i] += fs.shifted(T[unit[i + 1]], 15.0)
        spk[i + 1] += fs.shifted(T[unit[i]], -15.0)
        kind[[i, i + 1]] = 2
    return fs.to_int16(spk), res, kind, lag


def main():
    ok = 0

    def check(name, cond):
        nonlocal ok
        assert cond, name
        ok += 1
        print(f"  ok  {name}")

    rng = np.random.default_rng(4)
    spk, res, kind, lag = group(rng, 6000, 600, 200)
    flags, info = ff.flag(spk, res, PEAK, neighbour=0)                      # the score alone
    hit, fp = flags[kind == 1].mean(), flags[kind == 0].mean()
    print(f"      score alone, --score-min 12: {hit:.1%} of the hidden collisions flagged, {fp:.1%} of the single spikes")
    check("the score flags >= 85 % of the hidden collisions", hit >= 0.85)
    check("... and <= 3 % of the single spikes", fp <= 0.03)
    m = (kind == 1) & flags
    check("the excess is where the partner is (within 3 samples for >= 80 % of the flagged collisions)",
          (np.abs(info["lag"][m] - lag[m]) <= 3).mean() >= 0.80)
    flags2, info2 = ff.flag(spk, res, PEAK)
    check("the neighbour rule flags every detection with another one 15 samples away",
          flags2[kind == 2].all() and info2["by_neighbour"] == int((kind == 2).sum()) and info2["neighbour"] == NS // 2)
    check("it adds nothing else", np.array_equal(flags2 & (kind != 2), flags & (kind != 2)))
    loose, _ = ff.flag(spk, res, PEAK, score_min=6.0, neighbour=0)
    check("a lower --score-min flags more, and everything the higher one flagged", loose.sum() > flags.sum() and loose[flags].all())
    capped, ci = ff.flag(spk, res, PEAK, max_share=0.03, neighbour=0)
    check("--max-share 0.03 raises the threshold until the score flags at most 3 % of the spikes",
          ci["raised"] and capped.mean() <= 0.03 + 1e-9 and flags[capped].all())

    spk0, res0, kind0, _ = group(rng, 6000, 0, 0)
    f0, _ = ff.flag(spk0, res0, PEAK)
    print(f"      NEGATIVE CONTROL, no collision anywhere: {f0.mean():.2%} flagged")
    check("NEGATIVE CONTROL: without collisions under 2 % of the spikes are flagged", f0.mean() <= 0.02)
    check("nearest_detection: gaps on both sides, and for a .res that is not sorted",
          list(ff.nearest_detection([10, 14, 40, 41])) == [4, 4, 1, 1]
          and list(ff.nearest_detection([14, 10, 41, 40])) == [4, 4, 1, 1])

    with tempfile.TemporaryDirectory() as td:
        base = os.path.join(td, "syn")
        res.tofile(f"{base}.res.1")
        spk.tofile(f"{base}.spk.standard.1")
        with open(f"{base}.yaml", "w") as f:
            f.write("acquisitionSystem: {nChannels: 8, samplingRate: 32552, nBits: 16}\n"
                    "spikeDetection:\n  channelGroups:\n"
                    "    - channels: [0,1,2,3,4,5,6,7]\n      nSamples: 52\n      peakSampleIndex: 26\n      nFeatures: 3\n")
        env = {k: v for k, v in os.environ.items() if not k.startswith("FK_")}
        env["PYTHONPATH"] = os.pathsep.join([os.path.join(HERE, "..", "src"), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)

        def run(extra=()):
            return subprocess.run([sys.executable, "-m", "fiber_kit.fiber_flag_collisions", base, "1", *extra],
                                  capture_output=True, text=True, env=env, timeout=600)
        out = f"{base}.clu.standard.1.artifact"
        r = run(["--dry-run"])
        check("--dry-run reports and writes nothing", r.returncode == 0 and not os.path.exists(out) and "dry run" in r.stderr)
        art = np.zeros(len(res), np.int32)
        clean_row = int(np.flatnonzero(~flags2)[0]); coll_row = int(np.flatnonzero(flags2)[0])
        art[clean_row] = 1; art[coll_row] = 1                       # fiber-flag-artifacts was here first
        nio.write_clu_file(out, art)
        r = run()
        if r.returncode != 0:
            print((r.stdout + r.stderr)[-2000:])
        _, mask = nio.read_clu_file(out, n_spikes=len(res))
        want = np.where(flags2, ff.COLLISION, 0); want[[clean_row, coll_row]] = 1
        check("the mask: collisions as 2, the earlier artifact entries kept as 1, the rest 0",
              r.returncode == 0 and np.array_equal(mask, want))
        r = run(["--score-min", "40", "--neighbour", "0"])
        _, mask2 = nio.read_clu_file(out, n_spikes=len(res))
        strict, _ = ff.flag(spk, res, PEAK, score_min=40.0, neighbour=0)
        want2 = np.where(strict, ff.COLLISION, 0); want2[[clean_row, coll_row]] = 1
        check("a second run redoes the collisions (fewer at a higher --score-min), it does not add to them",
              r.returncode == 0 and np.array_equal(mask2, want2) and (mask2 == 2).sum() < (mask == 2).sum())
        os.remove(f"{base}.spk.standard.1")
        check("without the .spk: refuses, naming the file", run().returncode != 0)

    print(f"test_flag_collisions: {ok} checks passed")


if __name__ == "__main__":
    main()
