"""fiber-profile: the py-spy stack parser, the summary, and (when py-spy can run here) a real record.

  * a process frame whose command line holds ';' and '"' is one frame (the exit-status shell's does);
  * the shell is dropped from the chain, the command under it is the parent, its children workers;
  * activity: importlib -> import, pickle -> pickle/IPC, a threading leaf -> waiting, else compute;
  * the computing is attributed to the package of the innermost frame (fiber_kit, numpy, ...);
  * record keeps the command's exit status (py-spy itself always exits 0) and follows pool workers.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from fiber_kit import fiber_profile as fp          # noqa: E402

fails = 0


def check(ok, what):
    global fails
    print(("  ok:   " if ok else "  FAIL: ") + what); fails += not ok


SH = 'process 10:"/bin/sh -c st=$1; shift; "$@"; echo $? > "$st" fiber-profile-exec /x.exit python3 -m fiber_kit.fiber_session a; b"'
KID = 'process 11:"/bin/sh -c st=$1; shift; "$@"; echo $? > "$st" fiber-profile-exec /x.exit python3 -m fiber_kit.fiber_session a; b"'
RAW = "\n".join([
    f'{SH};{KID};<module> (fiber_kit/fiber_session.py:3);_find_and_load (<frozen importlib._bootstrap>:1360) 50',
    f'{SH};{KID};main (fiber_kit/fiber_session.py:2766);Condition.wait (threading.py:359) 20',
    f'{SH};{KID};main (fiber_kit/fiber_session.py:2766);_ForkingPickler.dumps (multiprocessing/reduction.py:51) 5',
    f'{SH};{KID};process 12:"python3 w";_process_chunk (fiber_kit/fiber_session.py:900);_sum (numpy/_core/_methods.py:49) 30',
    f'{SH};{KID};process 13:"python3 w";_process_chunk (fiber_kit/fiber_session.py:900);estep (fiber_kit/klustakwik.py:125) 10',
]) + "\n"


def main():
    with tempfile.TemporaryDirectory() as td:
        raw = os.path.join(td, "r.raw")
        with open(raw, "w") as fh:
            fh.write(RAW)
        rows = fp.parse_raw(raw)
        check(len(rows) == 5, "five stacks parsed")
        check(rows[0][0] == [(11, "(the profiled command)")],
              "the shell is dropped; the command under it is relabelled and is the root")
        check(rows[3][0][-1] == (12, "python3 w"), "a worker under the command is its own process")
        check(rows[0][1][-1]["path"].startswith("<frozen importlib"), "the frames after the chain parse")
        s = fp.summarise(rows, 10, {"wall_s": 10.0, "cpu_s": 12.0})
        check(s["parent"]["pid"] == 11 and abs(s["parent"]["s"] - 7.5) < 1e-9, "parent 75 samples = 7.5 s")
        check(s["workers"]["n"] == 2 and abs(s["workers"]["s_max"] - 3.0) < 1e-9, "two workers, busiest 3 s")
        check(abs(s["workers"]["balance"] - 4.0 / 6.0) < 1e-9, "balance = 4 s / (2 x 3 s)")
        pa = s["activity_s"]["parent"]
        check((pa["import"], pa["waiting"], pa["pickle/IPC"]) == (5.0, 2.0, 0.5),
              "parent: import 5 s, waiting 2 s, pickling 0.5 s")
        check(s["compute_by_package_s"] == {"numpy": 3.0, "fiber_kit": 1.0}, "compute by package")
        check(abs(s["busy_threads_avg"] - 0.95) < 1e-9, "busy threads = (10 - 0.5 waiting... ) / wall")
        empty = os.path.join(td, "e.raw")              # py-spy may read the shell's command line as ""
        with open(empty, "w") as fh:
            fh.write('process 20:"";process 21:"python3 x";main (fiber_kit/x.py:1) 4\n')
        check(fp.parse_raw(empty)[0][0] == [(21, "python3 x")], "a shell with an empty command line is dropped too")
        txt = fp.format_summary(s)
        check("balance 0.67" in txt and "numpy" in txt, "the text report renders")

        if shutil.which("py-spy") is None:
            print("  skip: py-spy not installed -- no end-to-end record")
        else:
            work = os.path.join(td, "w.py")
            with open(work, "w") as fh:
                fh.write("import sys\nfrom concurrent.futures import ProcessPoolExecutor\n"
                         "def f(n):\n    return sum(i * i for i in range(n))\n"
                         "if __name__ == '__main__':\n"
                         "    with ProcessPoolExecutor(2) as ex:\n        list(ex.map(f, [3_000_000] * 4))\n"
                         "    sys.exit(3)\n")
            env = dict(os.environ, PYTHONPATH=os.path.join(HERE, "..", "src"))
            out = os.path.join(td, "prof")
            r = subprocess.run([sys.executable, "-m", "fiber_kit.fiber_profile", "record", "--out", out,
                                "--name", "w", "--", sys.executable, work],
                               capture_output=True, text=True, env=env, timeout=300)
            raws = [f for f in os.listdir(out) if f.endswith(".raw")] if os.path.isdir(out) else []
            if not raws and "ermission" in (r.stdout + r.stderr):
                print("  skip: py-spy may not ptrace here (no CAP_SYS_PTRACE)")
            else:
                check(r.returncode == 3, f"record exits with the command's status (got {r.returncode})")
                check(len(raws) == 1, "one .raw written")
                meta = json.load(open(os.path.join(out, raws[0][:-4] + ".json")))
                check(meta["exit"] == 3 and meta["wall_s"] > 0 and meta["cpu_s"] > 0, ".json: exit, wall, cpu")
                s = fp.summarise(fp.parse_raw(os.path.join(out, raws[0]), meta["shim"]), meta["rate"], meta)
                check(s["workers"]["n"] >= 1, f"pool workers sampled ({s['workers']['n']})")
    print(f"test_profile: {'FAIL' if fails else 'PASS'} ({fails} failed)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
