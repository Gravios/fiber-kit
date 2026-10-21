"""fiber-realign --check (report, write nothing) and --recut (re-cut at the current .res, move nothing).

The state a pre-0.51 commit left behind: spike times on the true peaks, every window cut one sample late
(the timestamp at peakSampleIndex instead of peakSampleIndex - 1).

  * --check says so and changes no file: with the .fil it locates the windows at peakSampleIndex
    instead of peakSampleIndex - 1, and a realign (aligning in that geometry) would move nothing and
    only re-cut; without the .fil it sees every unit's template one sample after the target and a
    realign moving every spike by +1;
  * --recut re-cuts the windows where neurosuite-3 would (the timestamp at peakSampleIndex - 1), keeps the
    .res, writes zero offsets and leaves the .clu alone;
  * after it --check finds the windows at the group's index, every template on the target and nothing
    to move;
  * the coincidence / inversion counters see same-unit and cross-unit pairs on one sample and a step
    backwards inside a unit.
Negative control: --check on correctly cut windows (a fresh extraction) already reports nothing to move.
"""
import glob
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from fiber_kit import fiber_realign as frl   # noqa: E402
from fiber_kit import neuro_io as nio        # noqa: E402

SR, NCH, NTOTAL, NSAMP, PK = 20000, 3, 4, 42, 21


def _session(td, late):
    rng = np.random.default_rng(11)
    base = os.path.join(td, "rc")
    n_t = 300000
    fil = rng.normal(0, 20, (n_t, NTOTAL))
    t = np.arange(-8, 9)
    wav = -3000 * np.exp(-0.5 * (t / 1.5) ** 2) + 900 * np.exp(-0.5 * ((t - 5) / 2.5) ** 2)
    true, lab = [], []
    for u, sh in ((2, [1.0, 0.6, 0.3]), (3, [0.3, 0.6, 1.0])):
        tt = np.sort(rng.choice(np.arange(1000 + 35 * u, n_t - 1000, 140), 300, replace=False))
        for s in tt:
            fil[s + t, :NCH] += wav[:, None] * np.array(sh)[None, :]
        true.append(tt); lab.append(np.full(tt.size, u))
    o = np.argsort(np.concatenate(true), kind="stable")
    true = np.concatenate(true)[o]; lab = np.concatenate(lab)[o]
    fil = fil.round().clip(-32768, 32767).astype(np.int16)
    fil.tofile(base + ".fil")
    true.astype(np.int64).tofile(base + ".res.1")
    c = PK if late else frl.timestamp_index(PK)          # late: what a pre-0.51 commit wrote
    nio.write_spk(base, 1, np.stack([fil[r - c:r - c + NSAMP, :NCH] for r in true]), variant="standard")
    nio.write_clu_file(base + ".clu.standard.1", lab)
    with open(base + ".yaml", "w") as fh:
        fh.write(f"acquisitionSystem: {{nBits: 16, nChannels: {NTOTAL}, samplingRate: {SR}}}\n"
                 "spikeDetection:\n  channelGroups:\n    - channels: [0, 1, 2]\n"
                 f"      nSamples: {NSAMP}\n      peakSampleIndex: {PK}\n")
    return base, fil, true


def _bytes(path):
    with open(path, "rb") as fh:
        return fh.read()


def _snapshot(base):
    return {p: _bytes(p) for p in sorted(glob.glob(base + ".*"))}


def _check(base, logs):
    args = [base, "1", "--clu", base + ".clu.standard.1", "--variant", "standard", "--min-score", "0.1",
            "--check"]
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        frl.main(args)
    logs.append(buf.getvalue())
    return buf.getvalue()


def main():
    ok = 0

    def check(name, cond, out=""):
        nonlocal ok
        if not cond and out:
            print(out[-3000:])
        assert cond, name
        ok += 1
        print(f"  ok  {name}")

    logs = []
    with tempfile.TemporaryDirectory() as td:
        base, fil, true = _session(td, late=True)
        before = _snapshot(base)
        out = _check(base, logs)
        check("--check writes nothing", _snapshot(base) == before, out)
        check("--check locates the windows at index 21, not the group's 20",
              "the spike time at index 21, NOT the group's 20" in out, out)
        check("--check: aligned in that geometry, every template is on target and a realign moves no spike",
              "2 of 2 units' templates peak on the target (index 21)" in out and "0 of 600 spikes" in out, out)
        os.rename(base + ".fil", base + ".fil.away")
        out = _check(base, logs)
        os.rename(base + ".fil.away", base + ".fil")
        check("--check without the .fil: every template one sample after the target",
              "0 of 2 units' templates peak on the target (index 20)" in out and "+1: 2" in out, out)
        check("--check without the .fil: a realign would move every spike by +1",
              "600 of 600 spikes (100.0%)" in out and "+1: 600" in out, out)

        frl.main([base, "1", "--clu", base + ".clu.standard.1", "--variant", "standard", "--recut",
                  "--variants", "standard"])
        spk = np.asarray(nio.open_spk_file(base + ".spk.standard.1", NSAMP, NCH))
        c0 = frl.timestamp_index(PK)
        fresh = np.stack([fil[r - c0:r - c0 + NSAMP, :NCH] for r in true])
        check("--recut: windows = a neurosuite-3 cut at the current .res", np.array_equal(spk, fresh))
        check("--recut: the .res is unchanged", np.array_equal(nio.read_res_file(base + ".res.1"), true))
        check("--recut: zero offsets", not np.any(np.load(base + ".offsets.1.npy")))
        check("--recut: the .clu is untouched",
              _bytes(base + ".clu.standard.1") == before[base + ".clu.standard.1"])
        out = _check(base, logs)
        check("after --recut, --check: the windows at the group's index, every template on the target, "
              "nothing to move", "the spike time at index 20, the group's geometry" in out
              and "2 of 2 units' templates peak on the target (index 20)" in out and "0 of 600 spikes" in out, out)

    with tempfile.TemporaryDirectory() as td:
        base, fil, true = _session(td, late=False)
        out = _check(base, logs)
        check("NEGATIVE CONTROL: a fresh extraction already has nothing to move",
              "0 of 600 spikes" in out and "2 of 2 units' templates peak on the target" in out, out)
        try:
            frl.main([base, "1", "--clu", base + ".clu.standard.1", "--recut", "--shift-spk"])
            refused = False
        except SystemExit as e:
            refused = "--recut re-reads the .fil" in str(e)
        check("--recut with --shift-spk is refused", refused)

    # the counters
    t = np.array([100, 100, 200, 200, 300, 290, 400, 410])
    lab = np.array([2, 2, 2, 3, 2, 2, 1, 1])
    same, cross = frl.time_coincidences(t, lab)
    check("time_coincidences: one same-unit pair, one cross-unit pair",
          same.tolist() == [[0, 1]] and cross.tolist() == [[2, 3]])
    n_all, n_unit = frl.time_inversions(t, lab)
    check("time_inversions: one step backwards, inside unit 2 (reserve bins ignored)", (n_all, n_unit) == (1, 1))
    print(f"test_realign_recut_check: {ok} checks passed")


if __name__ == "__main__":
    main()
