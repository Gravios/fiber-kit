"""The int32 header of .clu / .clc / .clp files, and fiber-clu-header, which repairs it.

  * write_clu_file: the header of a .clu or .clc is the number of DISTINCT ids, so a chunk run's file --
    real clusters plus reserved clusters 16777116+chunk -- gets a header Klusters accepts (<= 65536).
    0.57.2 wrote the highest id + 1 (16777119 here) and Klusters refused the file ("Invalid or missing
    .clu header (nClusters=16777136)");
  * a .clp's header is its number of entries (= the highest atom id); `n_clusters=` is ignored, through
    write_clu too; clu_kind reads the kind token, not a suffix;
  * fiber-clu-header --check reports the files a 0.57.2 chunk run left and changes nothing; without
    --check it rewrites only the 4 header bytes (the ids are byte-identical).  It skips a .res, a
    .units.npz and a text .clu that the same shell glob catches, without touching them.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src")
sys.path.insert(0, SRC)
from fiber_kit import neuro_io as nio   # noqa: E402

RES = 16777116                          # fiber_chunks.RESERVED_LO
fails = 0


def check(ok, what):
    global fails
    print(("  ok:   " if ok else "  FAIL: ") + what); fails += not ok


def raw(path):
    return np.fromfile(path, "<i4")


def data(path):
    with open(path, "rb") as fh:
        return fh.read()


def cli(*args):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([SRC, env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    r = subprocess.run([sys.executable, "-m", "fiber_kit.fiber_clu_header", *args],
                       capture_output=True, text=True, env=env, timeout=120)
    return r.returncode, r.stdout + r.stderr


def main():
    ids = np.array([0, 1, 2, 2, 5, 7, RES, RES + 2, RES + 2], np.int64)    # 7 distinct, highest RES + 2
    with tempfile.TemporaryDirectory() as d:
        print("write_clu_file: the header is the number of distinct ids")
        p = os.path.join(d, "s.clu.stderiv.8.exp_final")
        nio.write_clu_file(p, ids)
        r = raw(p)
        check(r[0] == 7 and r[0] <= 65536, f"a chunk run's .clu: header {r[0]} = 7 distinct ids, within the "
                                            f"0-65536 Klusters accepts (0.57.2 wrote {RES + 3})")
        check(np.array_equal(r[1:], ids) and np.array_equal(nio.read_clu_file(p)[1], ids), "the ids are as given")
        q = os.path.join(d, "s.clc.stderiv.8.exp_final")
        nio.write_clu_file(q, np.array([1, 1, 3, 9, 9, 40], np.int64))
        check(raw(q)[0] == 4, f".clc: header {raw(q)[0]} = 4 distinct atom ids (not 41)")
        c = os.path.join(d, "s.clp.stderiv.8.exp_final")
        nio.write_clu_file(c, np.array([2, 2, 5, 0, RES], np.int64))
        check(raw(c)[0] == 5, f".clp: header {raw(c)[0]} = its 5 entries (the highest atom id), not its distinct ids")
        nio.write_clu_file(p, ids, n_clusters=RES + 3)
        check(raw(p)[0] == 7, "n_clusters= is ignored (it is accepted for old callers)")
        nio.write_clu(os.path.join(d, "s"), 8, ids, n_clusters=99, variant="stderiv", tag="w")
        check(raw(os.path.join(d, "s.clu.stderiv.8.w"))[0] == 7, "... through write_clu too")
        nio.write_clu_file(os.path.join(d, "s.clp.stderiv.8.fs.tmp"), np.array([3, 3], np.int64))
        check(raw(os.path.join(d, "s.clp.stderiv.8.fs.tmp"))[0] == 2, "a .clp written under a .tmp name is a .clp")
        check([nio.clu_kind(x) for x in ("b.clu.stderiv.8.clp", "b.clp.stderiv.8.fs.tmp", "b.clc.8", "/x/free")]
              == ["clu", "clp", "clc", "clu"], "clu_kind: the first kind token after the base; 'clu' when none")
        check(nio.write_clu_file(os.path.join(d, "s.clu.9"), np.zeros(0, np.int64)) and raw(os.path.join(d, "s.clu.9"))[0] == 0,
              "an empty .clu has header 0")

    with tempfile.TemporaryDirectory() as d:
        print("fiber-clu-header: the files a 0.57.2 chunk run left")
        old = {}
        for name, arr, hdr in (("m.clu.stderiv_C5_D34.8.exp_final", ids, RES + 3),
                               ("m.clc.stderiv_C5_D34.8.exp_final", np.array([0, 1, 1, 2, 3, 4, 5, 6, 7]), 8),
                               ("m.clp.stderiv_C5_D34.8.exp_final", np.array([2, 2, 5, 7, 7, RES, RES + 2]), RES + 3)):
            path = os.path.join(d, name)
            np.concatenate([[hdr], arr]).astype("<i4").tofile(path)
            old[path] = data(path)
        others = {os.path.join(d, "m.res.stderiv.8.exp_final"): np.arange(9, dtype="<i8").tobytes(),
                  os.path.join(d, "m.clu.stderiv_C5_D34.8.exp_final.units.npz"): b"PK\x03\x04" + bytes(60),
                  os.path.join(d, "m.clu.3"): b"3\n1\n2\n2\n"}
        for path, b in others.items():
            with open(path, "wb") as fh:
                fh.write(b)
        allf = sorted(old) + sorted(others)
        rc, out = cli("--check", *allf)
        check(rc == 1 and out.count("  OFF   ") == 2 and out.count("  skip  ") == 3
              and all(data(x) == b for x, b in old.items()),
              "--check: the .clu and the .clp are off, the .clc is right, 3 files skipped; nothing written"
              + ("" if rc == 1 else "\n" + out))
        rc, out = cli(*allf)
        clu, clc, clp = (raw(x) for x in sorted(old, key=lambda x: (".clu." not in x, ".clc." not in x)))
        check(rc == 0 and out.count("  fixed ") == 2 and clu[0] == 7 and clp[0] == 7 and clc[0] == 8,
              f"repaired: .clu header {clu[0]} (7 distinct ids), .clp {clp[0]} (7 entries); the .clc (gap-free, "
              f"so 8 either way) left at {clc[0]}"
              + ("" if rc == 0 else "\n" + out))
        check(all(data(x)[4:] == b[4:] for x, b in old.items()),
              "only the 4 header bytes changed: every id is byte-identical")
        check(all(data(x) == b for x, b in others.items()),
              "the .res, the .units.npz and the text .clu the glob caught are untouched")
        rc, out = cli("--check", *sorted(old))
        check(rc == 0 and "  OFF   " not in out, "--check after the repair: all right")
        rc, out = cli(os.path.join(d, "missing.clu.8"))
        check(rc == 0 and "  skip  " in out and "Traceback" not in out,
              "a missing file is skipped with a message, not a traceback")

    print(f"test_clu_header: {'PASS' if not fails else 'FAIL'} ({fails} failed)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
