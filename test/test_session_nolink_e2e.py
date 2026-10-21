"""fiber-session --no-link runs END TO END (the chunk-disjoint plan path).

The plan templates run fiber-session with --no-link (linking is fiber-link /
fiber-anchor-link's job), and that path had never been exercised whole: the sort
finished, the .clu/.clc/.clp were written -- and then the drift_anchor_pairs
comprehension indexed anchor_links[c] on the bare [] the no-link branch left
behind, killing the stage with IndexError after its real work was done.
anchor_links now always carries one (possibly empty) list per adjacent chunk
pair, the shape link_chunks returns.

This drives the real CLI on a synthetic two-chunk session whose directory also
holds TWO stderiv-family .spk variants, so the run doubles as an end-to-end
check of the method-token threading (main AND the chunk worker resolve the
exact --method token instead of raising the family-ambiguity error).

It also checks the noise model end to end: the .spk is a stderiv_C5 one, so the
chunk whitener needs the group's sdiffPairs (here --sdiff-pairs, the session
has no YAML) and says which space it whitens in; without the pattern the stage
refuses cleanly instead of whitening in the all-pairs space.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

SR = 20000.0
NSAMP, NCH, NTOTAL = 32, 3, 4
SETS3 = "0-1+2,1-0+2,2-0+1"          # order-5 reference sets for the 3-channel group
PER_CHUNK = 300


def _build_session(td, rng):
    base = os.path.join(td, "mini")
    # spike times: two 0.5-min chunks, PER_CHUNK spikes each, well inside the record
    t0, t1 = int(2 * SR), int(58 * SR)
    res = np.sort(rng.integers(t0, t1, 2 * PER_CHUNK)).astype(np.int64)
    res.tofile(f"{base}.res.1")
    # waveforms: one unit, big negative peak on ch0 at the canonical peak sample
    w = np.zeros((len(res), NSAMP, NCH), np.int16)
    w[:, :, :] = rng.integers(-40, 40, w.shape).astype(np.int16)
    w[:, 12:20, 0] += (-3000 * np.hanning(8)).astype(np.int16)[None, :]
    w.tofile(f"{base}.spk.stderiv_C5.1")
    (w + np.int16(1)).tofile(f"{base}.spk.stderiv_C5_D34.1")   # second family member
    # .fil: mild noise so the chunk whitener sees a usable covariance
    nfil = int(60 * SR)
    fil = rng.integers(-30, 30, nfil * NTOTAL).astype(np.int16)
    fil.tofile(f"{base}.fil")
    return base


def main():
    ok = 0

    def check(name, cond):
        nonlocal ok
        assert cond, name
        ok += 1
        print(f"  ok  {name}")

    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as td:
        base = _build_session(td, rng)
        env = {k: v for k, v in os.environ.items() if not k.startswith("FK_")}
        env["PYTHONPATH"] = os.pathsep.join(
            [os.path.join(os.path.dirname(__file__), "..", "src"),
             env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
        argv = [sys.executable, "-m", "fiber_kit.fiber_session", base, "1",
                "--channels", "0,1,2", "--ntotal", str(NTOTAL), "--nsamp", str(NSAMP),
                "--sr", str(int(SR)), "--method", "stderiv_C5_D34",
                "--no-link", "--no-fine", "--chunk-min", "0.5", "--overlap-min", "0",
                "--min-group", "50", "--clu-stage", "nolink"]
        r0 = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=420)
        out0 = r0.stdout + r0.stderr
        check("a stderiv_C5 .spk without its sdiffPairs is refused, cleanly",
              r0.returncode != 0 and "sdiffPairs" in out0 and "Traceback" not in out0)
        r = subprocess.run(argv + ["--sdiff-pairs", SETS3],
                           capture_output=True, text=True, env=env, timeout=420)
        out = r.stdout + r.stderr
        if r.returncode != 0:
            print(out[-4000:])
        check("fiber-session --no-link exits 0 on the two-variant directory", r.returncode == 0)
        check("ran chunk-disjoint (the no-link mode)", "chunk-disjoint" in out)
        check("whitens in the .spk's own space (C5 sets from --sdiff-pairs)",
              f"noise model   stderiv, sdiffPairs {SETS3} (order 5)" in out)
        check("resolved the exact --method spk (no family-ambiguity error)",
              ".spk.stderiv_C5_D34.1" in out and "several 'stderiv' variants" not in out)
        clu = f"{base}.clu.stderiv_C5_D34.1.nolink"
        check("wrote the hierarchy clu under the token + stage", os.path.exists(clu))
        fib = np.load(f"{base}.fibers.stderiv_C5_D34.1", allow_pickle=False)
        check("emitted .fibers with EMPTY drift_anchor_pairs (0, 3)",
              fib["drift_anchor_pairs"].shape == (0, 3))

    print(f"test_session_nolink_e2e: {ok}/7 checks passed")


if __name__ == "__main__":
    main()
