"""fiber-session keeps out of Klusters' reserve ids and records where each fiber was written.

0 and 1 are the artefact and noise bins for every later stage (intrachunk reserve=(0, 1), consensus
parks its quarantine in 1, xcorr-merge and backbone-link read units as clu > 1, anchor-link holds atoms
0/1 out of the link and folds flat-label-1 spikes into noise).  On a synthetic two-chunk session with
linking:

  * no spike is in cluster 1, every fiber is >= 2, no atom (.clc) is 1;
  * the first global fiber (gid 0) -- written as cluster 1 until 0.51 -- is a cluster >= 2;
  * .fibers clu_id names each row's cluster in the written .clu: rows of one gid share it, it is a
    cluster of the .clu, and that cluster holds spikes in the row's chunk;
  * the merge-candidate list carries clu_a / clu_b (consistent with .fibers for any pair it lists);
  * anchor-link's chunk partition keeps every fiber-session atom (none held out as a reserve bin).
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from fiber_kit import fiber_anchor_link as fal   # noqa: E402
from fiber_kit import fiber_stderiv as fst       # noqa: E402
from fiber_kit import neuro_io as nio            # noqa: E402

SR = 20000
NSAMP, NCH, NTOTAL, PK = 32, 3, 4, 16
DUR_S = 60


def _session(td):
    rng = np.random.default_rng(23)
    base = os.path.join(td, "ids")
    n_t = DUR_S * SR
    fil = rng.normal(0, 25, (n_t, NTOTAL))
    t = np.arange(-8, 9)
    shapes = [np.array([1.0, 0.5, 0.2]), np.array([0.3, 1.0, 0.6]), np.array([0.2, 0.4, 1.0])]
    det = []
    for amp, sh in zip((2600.0, 1800.0, 2200.0), shapes):
        tt = np.sort(rng.choice(np.arange(SR, n_t - SR, 7), 300, replace=False))
        wav = -amp * np.exp(-0.5 * (t / 1.6) ** 2) + 0.35 * amp * np.exp(-0.5 * ((t - 5) / 2.5) ** 2)
        for s in tt:
            fil[s + t, :NCH] += wav[:, None] * sh[None, :]
        det.append(tt + rng.integers(-1, 2, tt.size))
    fil = fil.round().clip(-32768, 32767).astype(np.int16)
    fil.tofile(base + ".fil")
    det = np.sort(np.concatenate(det)).astype(np.int64)
    det.tofile(base + ".res.1")
    c0 = PK - 1                                            # neurosuite-3's window
    raw = np.stack([fil[d - c0 - 1:d - c0 + NSAMP, :NCH] for d in det])
    nio.write_spk(base, 1, raw[:, 1:, :], variant="standard")
    nio.write_spk(base, 1, fst.apply_stderiv_transform(raw), variant="stderiv")
    with open(base + ".yaml", "w") as fh:
        fh.write(f"acquisitionSystem: {{nBits: 16, nChannels: {NTOTAL}, samplingRate: {SR}}}\n"
                 "spikeDetection:\n  channelGroups:\n    - channels: [0, 1, 2]\n"
                 f"      nSamples: {NSAMP}\n      peakSampleIndex: {PK}\n")
    return base, det


def main():
    ok = 0

    def check(name, cond, out=""):
        nonlocal ok
        if not cond and out:
            print(out[-4000:])
        assert cond, name
        ok += 1
        print(f"  ok  {name}")

    env = {k: v for k, v in os.environ.items() if not k.startswith("FK_")}
    env["FK_CONFIG"] = os.devnull
    env["PYTHONPATH"] = os.pathsep.join([os.path.join(HERE, "..", "src"), env.get("PYTHONPATH", "")])
    with tempfile.TemporaryDirectory() as td:
        base, res = _session(td)
        r = subprocess.run([sys.executable, "-m", "fiber_kit.fiber_session", base, "1", "--method", "stderiv",
                            "--chunk-min", "0.5", "--overlap-min", "0.1", "--min-group", "40",
                            "--clu-stage", "fs", "--realign", "off", "--emit-merge-candidates"],
                           capture_output=True, text=True, env=env, timeout=900)
        out = r.stdout + r.stderr
        check("fiber-session (two chunks, linked) exits 0", r.returncode == 0, out)
        _, clu = nio.read_clu_file(base + ".clu.stderiv.1.fs")
        _, clc = nio.read_clu_file(base + ".clc.stderiv.1.fs")
        clu = np.asarray(clu); clc = np.asarray(clc)
        fibers = np.unique(clu[clu > 0])
        check(f"no spike in cluster 1; fibers are {fibers.min()}..{fibers.max()}",
              fibers.size > 0 and not (clu == 1).any() and fibers.min() >= 2, out)
        check("no atom is 1 (0 is noise only)", not (clc == 1).any() and np.array_equal(clc == 0, clu == 0))
        fib = np.load(nio.fibers_path(base, "stderiv", 1), allow_pickle=False)
        gid, cid, ch = fib["gid"], fib["clu_id"], fib["chunk"]
        check("the first global fiber is a cluster >= 2 (it was 1)", (cid[gid == 0] >= 2).all() and (gid == 0).any())
        same = all(np.unique(cid[gid == g]).size == 1 for g in np.unique(gid[gid >= 0]))
        check("rows of one gid share one clu_id", same)
        check("every clu_id is a cluster of the .clu", set(cid[cid > 0].tolist()) <= set(fibers.tolist()))
        chunk_s = 0.5 * 60 * SR; t0 = int(res.min())
        hold = [((clu == c) & (((res - t0) // chunk_s).astype(int) == k)).any()
                for c, k, n in zip(cid, ch, fib["nspk"]) if c > 0 and n >= 20]
        check(f"each row's cluster holds spikes in the row's chunk ({sum(hold)}/{len(hold)})", hold and all(hold))
        with open(f"{base}.merge_candidates.1.tsv") as fh:
            lines = fh.read().splitlines()
        head = lines[0].split("\t")
        rows = [ln.split("\t") for ln in lines[1:]]
        g2c = {int(g): int(c) for g, c in zip(gid, cid) if g >= 0}
        agree = all(int(r_[head.index("clu_a")]) == g2c.get(int(r_[1]), 0) and
                    int(r_[head.index("clu_b")]) == g2c.get(int(r_[2]), 0) for r_ in rows)
        check(f"the merge-candidate list names clusters (clu_a / clu_b) consistent with .fibers ({len(rows)} pairs)",
              "clu_a" in head and "clu_b" in head and agree)
        chunk_of, _ = fal.atom_chunks_and_rates(clc, res, SR, chunk_min=None)
        atoms = set(np.unique(clc[clc > 0]).tolist())
        check("anchor-link's partition keeps every fiber-session atom", atoms <= set(chunk_of))
    print(f"test_session_ids: {ok} checks passed")


if __name__ == "__main__":
    main()
