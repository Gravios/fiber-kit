"""fiber-validate-merges reads each candidate's cluster from the list, not gid + 1.

Two fibers, gid 0 and 1, written as clusters 2 and 5 (fibers start at 2 and the hierarchy save
compacts ids).  With clu_a / clu_b in the candidate list the pair is evaluated on clusters 2 and 5.
A pre-0.51 list without them falls back to gid + 1 and says so: it evaluates the pair on clusters 1
and 2 -- in a pre-0.51 sort cluster 1 is fiber 0, so the pair is kept rather than dropped as a
reserve bin; in this 0.51 sort they are the wrong trains (the negative control).
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from fiber_kit import neuro_io as nio   # noqa: E402

SR = 20000


def main():
    ok = 0

    def check(name, cond, out=""):
        nonlocal ok
        if not cond and out:
            print(out[-3000:])
        assert cond, name
        ok += 1
        print(f"  ok  {name}")

    env = {k: v for k, v in os.environ.items() if not k.startswith("FK_")}
    env["FK_CONFIG"] = os.devnull
    env["PYTHONPATH"] = os.pathsep.join([os.path.join(HERE, "..", "src"), env.get("PYTHONPATH", "")])
    rng = np.random.default_rng(2)
    with tempfile.TemporaryDirectory() as td:
        base = os.path.join(td, "vm")
        res = np.sort(rng.choice(np.arange(SR, 600 * SR), 900, replace=False)).astype(np.int64)
        clu = np.where(np.arange(res.size) % 3 == 0, 2, 5)
        clu[:40] = 1                                     # 40 noise spikes: what gid + 1 would read
        res.tofile(base + ".res.1")
        nio.write_clu_file(base + ".clu.1", clu)
        with open(base + ".yaml", "w") as fh:
            fh.write(f"acquisitionSystem: {{nBits: 16, nChannels: 4, samplingRate: {SR}}}\n"
                     "spikeDetection:\n  channelGroups:\n    - channels: [0, 1, 2]\n")
        new = base + ".new.tsv"; old = base + ".old.tsv"
        with open(new, "w") as fh:
            fh.write("chunk\tgid_a\tgid_b\tlocal_a\tlocal_b\tprofile_dist\tthreshold\tclu_a\tclu_b\n"
                     "0\t0\t1\t0\t1\t0.1000\t0.2000\t2\t5\n")
        with open(old, "w") as fh:
            fh.write("chunk\tgid_a\tgid_b\tlocal_a\tlocal_b\tprofile_dist\tthreshold\n"
                     "0\t0\t1\t0\t1\t0.1000\t0.2000\n")

        def run(cand):
            r = subprocess.run([sys.executable, "-m", "fiber_kit.validate_merge_candidates", base, "1",
                                "--cand", cand], capture_output=True, text=True, env=env, timeout=300)
            return r.returncode, r.stdout + r.stderr

        rc, out = run(new)
        n2, n5 = int((clu == 2).sum()), int((clu == 5).sum())
        line = [ln for ln in out.splitlines() if ln.strip().startswith("0-1")]
        check("with clu_a / clu_b the pair is read on clusters 2 and 5",
              rc == 0 and line and line[0].split()[2:4] == [str(n2), str(n5)] and "WARNING" not in out, out)
        rc, out = run(old)
        n1 = int((clu == 1).sum())
        line = [ln for ln in out.splitlines() if ln.strip().startswith("0-1")]
        check("NEGATIVE CONTROL: an old list falls back to gid + 1, with a warning, and reads clusters 1 and 2 "
              "(fiber 0 is kept: cluster 1 was a fiber before 0.51)",
              rc == 0 and "assuming cluster = gid + 1" in out and line
              and line[0].split()[2:4] == [str(n1), str(n2)], out)
    print(f"test_validate_merges_ids: {ok} checks passed")


if __name__ == "__main__":
    main()
