#!/usr/bin/env python3
"""test_runner_timing.py — fiber-pipeline's log says how long every step took.

The log had no times: how long a stage of an overnight run took had to be read off the
modification times of its output files.  Checked here, with stand-in stage programs first on PATH
(the runner starts a stage by its console name):

  * a step's header is followed by the clock time it started at, and its end by "<step> took <t>";
  * the run ends with "== timing ==", one line per step in order, and the total;
  * a step that fails says "<step> stopped after <t>" and the steps before it are still listed;
    the runner's exit status is the stage's;
  * a --dry-run prints none of it.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPE = os.path.join(ROOT, "scripts", "fiber-pipeline")

fails = 0


def check(ok, what, extra=""):
    global fails
    fails += not ok
    print(("  ok:   " if ok else "  FAIL: ") + what + (("\n        " + extra) if (not ok and extra) else ""))


def main():
    tmp = tempfile.mkdtemp()
    try:
        bindir = os.path.join(tmp, "bin")
        os.mkdir(bindir)
        for name, body in (("fiber-clean", "sleep 1\nexit 0\n"), ("fiber-polish", "sleep 1\nexit 3\n")):
            path = os.path.join(bindir, name)
            with open(path, "w") as fh:
                fh.write("#!/usr/bin/env bash\n" + body)
            os.chmod(path, 0o755)
        sess = os.path.join(tmp, "sess")
        os.mkdir(sess)
        for method in ("stderiv", "standard"):                 # the first step's input, so the plan's wiring check passes
            with open(os.path.join(sess, f"sess.clu.{method}.1.t0"), "wb") as fh:
                fh.write(b"\0" * 8)
        env = {k: v for k, v in os.environ.items() if not k.startswith(("FK_", "FIBER_"))}
        env.update(PYTHONPATH=os.path.join(ROOT, "src"), PATH=bindir + os.pathsep + os.environ["PATH"])

        def run(plan, *extra):
            path = os.path.join(sess, "plan.yaml")
            with open(path, "w") as fh:
                fh.write("pipeline:\n" + "".join(f"  - {{stage: {st}, name: {nm}, in: t{i}, out: t{i + 1}}}\n"
                                                  for i, (st, nm) in enumerate(plan)))
            return subprocess.run(["bash", PIPE, "1", "--plan", path, *extra], cwd=sess, env=env, capture_output=True, text=True)

        dur = r"\d+s"
        r = run([("fiber-clean", "first"), ("fiber-clean", "second")])
        out = r.stdout
        check(r.returncode == 0, f"a plan of two stand-in steps runs (rc {r.returncode})", (r.stderr or out)[-400:])
        check(len(re.findall(r"^   started \d{4}-\d\d-\d\d \d\d:\d\d:\d\d$", out, re.M)) == 2, "each step says when it started")
        check(re.search(rf"^   first took {dur}$", out, re.M) and re.search(rf"^   second took {dur}$", out, re.M),
              "each step says how long it took")
        m = re.search(rf"^== timing ==\n   first +{dur}\n   second +{dur}\n   total +({dur})$", out, re.M)
        check(m is not None, "the run ends with the list of steps, in order, and the total", out[-300:])
        check(m is not None and 2 <= int(m.group(1)[:-1]) <= 30, "the total is the run's wall time (two steps of one second)")
        check(out.index("first took") < out.index("[plan 2/2]"), "a step's time is printed before the next step's header")

        r = run([("fiber-clean", "first"), ("fiber-polish", "broken"), ("fiber-clean", "never")])
        check(r.returncode == 3, f"a failing step stops the run with its own exit status ({r.returncode})")
        check(re.search(rf"^   broken stopped after {dur}$", r.stderr, re.M) is not None, "it says how long it ran before it stopped",
              r.stderr[-300:])
        check(re.search(rf"^== timing ==\n   first +{dur}\n   total +{dur}$", r.stderr, re.M) is not None and "never" not in r.stdout,
              "the steps that finished are still listed, the one after is not run")

        r = run([("fiber-clean", "first")], "--dry-run")
        check(r.returncode == 0 and "started" not in r.stdout and "timing" not in r.stdout and " took " not in r.stdout,
              "a --dry-run prints no times")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"test_runner_timing: {'PASS' if not fails else 'FAIL'} ({fails} failed)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
