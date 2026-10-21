"""fiber-session --whiten-source END TO END: nst / auto / spk need no .fil.

A synthetic two-chunk, one-group session with a standard .spk and NO .fil:
  * auto with the .spk's .nst present whitens with it (the log names the file)
    and never opens the .fil;
  * spk whitens from the snippets' pre-peak samples, again without a .fil;
  * nst with the .nst missing, and a .nst whose window does not match the
    .spk, stop the stage with a message;
  * an opaque variant (fiber-adlp's standard_ADLP), which no .fil transform
    describes, sorts with spk, and auto (no .nst) refuses it up front with a
    message naming --whiten-source spk -- not a traceback in a worker.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fiber_kit import neuro_io as nio   # noqa: E402

SR = 20000.0
NSAMP, NCH, NTOTAL, PEAK = 32, 2, 4, 15
PER_CHUNK = 300


def _build_session(td, rng):
    base = os.path.join(td, "mini")
    t0, t1 = int(2 * SR), int(58 * SR)
    res = np.sort(rng.integers(t0, t1, 2 * PER_CHUNK)).astype(np.int64)
    res.tofile(f"{base}.res.1")
    w = rng.normal(0, 30, (len(res), NSAMP, NCH))
    w[:, 12:20, 0] += -3000 * np.hanning(8)[None, :]
    w[:, 12:20, 1] += -1200 * np.hanning(8)[None, :]
    w.round().astype(np.int16).tofile(f"{base}.spk.standard.1")
    noise = rng.normal(0, 30, (4000, NSAMP * NCH))
    nio.write_nst(f"{base}.nst.standard.1",
                  dict(nsamp=NSAMP, nchan=NCH, peak=PEAK, n_requested=4000, n_drawn=4000,
                       n_kept=4000, zmax=5.0, sigma=np.full(NCH, 30.0),
                       mean=noise.mean(0).reshape(NSAMP, NCH), cov=np.cov(noise, rowvar=False)))
    return base


def run(base, *extra, method="standard"):
    env = {k: v for k, v in os.environ.items() if not k.startswith("FK_")}
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.join(os.path.dirname(__file__), "..", "src"), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    r = subprocess.run(
        [sys.executable, "-m", "fiber_kit.fiber_session", base, "1",
         "--channels", "0,1", "--ntotal", str(NTOTAL), "--nsamp", str(NSAMP),
         "--sr", str(int(SR)), "--method", method, "--no-link", "--no-fine",
         "--chunk-min", "0.5", "--overlap-min", "0", "--min-group", "50",
         "--clu-stage", "wsrc"] + list(extra),
        capture_output=True, text=True, env=env, timeout=600)
    return r.returncode, r.stdout + r.stderr


def main():
    ok = 0

    def check(name, cond, out=""):
        nonlocal ok
        if not cond and out:
            print(out[-3000:])
        assert cond, name
        ok += 1
        print(f"  ok  {name}")

    rng = np.random.default_rng(1)
    with tempfile.TemporaryDirectory() as td:
        base = _build_session(td, rng)
        clu = f"{base}.clu.standard.1.wsrc"
        check("no .fil in the session", not os.path.exists(f"{base}.fil"))

        rc, out = run(base)                                   # default: auto
        check("auto with a .nst exits 0 without a .fil", rc == 0, out)
        check("auto whitened with the .spk's .nst", "noise statistics mini.nst.standard.1" in out, out)
        check("wrote the clu", os.path.exists(clu))
        os.remove(clu)

        rc, out = run(base, "--whiten-source", "spk")
        check("spk exits 0 without a .fil", rc == 0, out)
        check("spk whitened from the snippets", "snippet pre-peak" in out, out)

        os.rename(f"{base}.nst.standard.1", f"{base}.nst.standard.1.off")
        rc, out = run(base, "--whiten-source", "nst")
        check("nst without a .nst stops with a message", rc != 0 and "--whiten-source nst: no" in out, out)

        bad = nio.read_nst(f"{base}.nst.standard.1.off")
        nio.write_nst(f"{base}.nst.standard.1",
                      dict(bad, nsamp=NSAMP // 2, mean=np.zeros((NSAMP // 2, NCH)),
                           cov=np.eye(NSAMP // 2 * NCH)))
        rc, out = run(base, "--whiten-source", "nst")
        check("a .nst for another window stops with a message", rc != 0 and "window, the .spk is" in out, out)

        # an opaque variant (fiber-adlp's) has no .fil transform: spk runs, auto/fil are refused up front
        import shutil
        shutil.copy(f"{base}.spk.standard.1", f"{base}.spk.standard_ADLP.1")
        rc, out = run(base, "--whiten-source", "spk", method="standard_ADLP")
        check("spk sorts an opaque variant (no transform needed)", rc == 0, out)
        rc, out = run(base, method="standard_ADLP")
        check("auto without a .nst refuses an opaque variant, naming --whiten-source spk",
              rc != 0 and "--whiten-source spk" in out and "Traceback" not in out, out)

    print(f"test_session_whiten_source_e2e: {ok}/10 checks passed")


if __name__ == "__main__":
    main()
