#!/usr/bin/env python3
"""test_runner_version.py — the runner script and the package it starts are one version.

pip copies scripts/fiber-pipeline to the bin directory at install time.  With an editable install
python keeps importing the checkout, so the copy on PATH grows older than the stages it starts
(a 0.67.9 copy once ran a 0.71.10 checkout's plans and did not know their stages).  Checked here:

  * `fk run` starts the checkout's script, not a `fiber-pipeline` that is first on PATH;
  * a runner of another version hands over to the checkout's script, and says so;
  * with no script to hand over to it stops (exit 2) and names both versions;
  * FK_RUNNER_ANY_VERSION=1 runs regardless.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPE = os.path.join(ROOT, "scripts", "fiber-pipeline")
sys.path.insert(0, os.path.join(ROOT, "src"))
from fiber_kit import plan_steps, __version__    # noqa: E402

fails = 0


def check(ok, what):
    global fails
    fails += not ok
    print(("  ok:   " if ok else "  FAIL: ") + what)


def run(cmd, cwd, **extra):
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"))
    for k in ("FK_CONFIG", "FK_RUNNER_ANY_VERSION", "FK_RUNNER_HANDED_OVER"):
        env.pop(k, None)
    env.update(extra)
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)


def main():
    check(os.path.samefile(plan_steps.runner_path(), PIPE), "runner_path() is the checkout's scripts/fiber-pipeline")
    tmp = tempfile.mkdtemp()
    try:
        # a stale copy first on PATH
        bindir = os.path.join(tmp, "bin")
        os.mkdir(bindir)
        stale = os.path.join(bindir, "fiber-pipeline")
        with open(stale, "w") as fh:
            fh.write("#!/usr/bin/env bash\necho STALE-RUNNER\n")
        os.chmod(stale, 0o755)
        sess = os.path.join(tmp, "sess")
        os.mkdir(sess)
        with open(os.path.join(sess, "fiber-kit.yaml"), "w") as fh:
            fh.write("pipeline:\n  - {stage: fiber-session, name: session, out: fiber_session}\n")
        fk = [sys.executable, "-c", "from fiber_kit.plan_steps import main; import sys; sys.exit(main(sys.argv[1:]))"]
        r = run(fk + ["run", "1", "--dry-run"], sess, PATH=bindir + os.pathsep + os.environ["PATH"])
        check(r.returncode == 0 and "STALE-RUNNER" not in r.stdout and "[plan 1/1]" in r.stdout,
              "fk run starts the checkout's runner with a stale one first on PATH")

        # the checkout's runner starts without a word about versions
        r = run(["bash", PIPE, "--help"], sess)
        check(r.returncode == 0 and "USAGE" in r.stdout.upper() and "this script is" not in r.stderr,
              "the runner of the package's version starts")

        # a runner of another version
        with open(PIPE) as fh:
            text = fh.read()
        check(f'FK_RUNNER_VERSION="{__version__}"' in text, "the runner carries the package's version")
        other = os.path.join(tmp, "fiber-pipeline-old")
        with open(other, "w") as fh:
            fh.write(re.sub(r'^FK_RUNNER_VERSION="[^"]+"', 'FK_RUNNER_VERSION="0.0.1"', text, flags=re.M))
        r = run(["bash", other, "--help"], sess)
        check(r.returncode == 0 and "USAGE" in r.stdout.upper() and "this script is 0.0.1" in r.stderr
              and os.path.normpath(PIPE) in r.stderr, "another version hands over to the checkout's script and says so")
        r = run(["bash", other, "--help"], sess, FK_RUNNER_HANDED_OVER="1")
        check(r.returncode == 2 and "0.0.1" in r.stderr and __version__ in r.stderr and "Reinstall" in r.stderr,
              "with nothing to hand over to it stops and names both versions")
        r = run(["bash", other, "--help"], sess, FK_RUNNER_ANY_VERSION="1")
        check(r.returncode == 0 and "USAGE" in r.stdout.upper() and "this script is" not in r.stderr,
              "FK_RUNNER_ANY_VERSION=1 runs regardless")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"test_runner_version: {'PASS' if not fails else 'FAIL'} ({fails} failed)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
