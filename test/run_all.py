#!/usr/bin/env python3
"""
run_all.py — one entry point for every check in this repo.

Six independent checks accumulated here with no way to run them together, which
is how a suite rots: a check nobody runs is a check nobody notices failing.

Tests are DISCOVERED, not listed.  Any test/test_*.py is picked up, so adding one
needs no edit here -- the same rule the doc generator and the startup smoke test
use for their own inputs.  The contract is the one the existing tests already
follow: exit non-zero on failure.

Also runs the checks that are not test files:
  * tools/gen_stages_doc.py --check --template  (docs/plans match the parsers)
  * ruff, when installed, using the config already in pyproject.toml

Usage:
    python3 test/run_all.py              # everything
    python3 test/run_all.py --fast       # skip the slow ones (see SLOW below)
    python3 test/run_all.py -v           # show each check's output
    NS3_ROOT=/path/to/neurosuite-3 python3 test/run_all.py
        # lets test_shared_contract compare the cross-repo tables directly
        # instead of only printing their hashes
"""
import argparse
import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Each spawns one interpreter per stage, so they cost about a minute.  Worth it
# in CI, not worth it on every save -- hence --fast.
SLOW = {"test_stage_startup.py"}
SLOW_EXTRA = {"gen_stages_doc --check"}


def run(name, cmd, verbose, timeout=1800):
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        out, code = (r.stdout + r.stderr), r.returncode
    except subprocess.TimeoutExpired:
        out, code = f"timed out after {timeout}s", 124
    except FileNotFoundError as e:
        return name, "skip", 0.0, str(e), []
    dt = time.time() - t0
    if verbose and out.strip():
        print("─" * 72)
        print(out.rstrip())
        print("─" * 72)
    lines = [ln.rstrip() for ln in out.strip().splitlines() if ln.strip()]
    tail = lines[-1][:78] if lines else ""
    # On failure the summary line is the least useful thing to show -- it says
    # how many failed, not which.  Keep the lines that name them.
    detail = []
    if code != 0:
        # Only lines that NAME a failing item.  Deliberately narrow: a loose match
        # picks up a linter's source-context lines and buries the thing you needed.
        detail = [ln.strip() for ln in lines
                  if ln.lstrip().startswith(("FAIL", "DRIFT", "ERROR"))][:12]
    return name, ("ok" if code == 0 else "FAIL"), dt, tail, detail


def main():
    ap = argparse.ArgumentParser(prog="run_all")
    ap.add_argument("--fast", action="store_true", help="skip the slow checks")
    ap.add_argument("-v", "--verbose", action="store_true", help="show each check's output")
    ap.add_argument("--strict-lint", action="store_true",
                    help="let ruff findings fail the suite (advisory by default)")
    a = ap.parse_args()

    checks = []
    for path in sorted(glob.glob(os.path.join(HERE, "test_*.py"))):
        name = os.path.basename(path)
        if a.fast and name in SLOW:
            continue
        checks.append((name, [sys.executable, path]))

    gen = os.path.join(ROOT, "tools", "gen_stages_doc.py")
    if os.path.exists(gen) and not (a.fast and "gen_stages_doc --check" in SLOW_EXTRA):
        checks.append(("gen_stages_doc --check",
                       [sys.executable, gen, "--check", "--template"]))

    if subprocess.run(["which", "ruff"], capture_output=True).returncode == 0:
        # Advisory unless --strict-lint.  There are ~90 findings after the config
        # was tuned to the documented house style, each wanting individual review;
        # a suite that is red from day one is a suite people stop reading, and the
        # tests above are the part that must stay trustworthy.
        checks.append(("ruff" + ("" if a.strict_lint else " (advisory)"),
                       ["ruff", "check", "src", "test", "tools"]))

    results = [run(n, c, a.verbose) for n, c in checks]

    print(f"\n{'check':38s} {'result':7s} {'time':>7s}  last line")
    print("-" * 100)
    failed = 0
    for name, status, dt, tail, _d in results:
        failed += status == "FAIL" and "advisory" not in name
        print(f"{name:38s} {status:7s} {dt:6.1f}s  {tail}")

    # Name the failures.  Without this a red suite tells you only that something
    # broke, and the next step is re-running the check by hand to find out what.
    for name, status, _dt, _tail, detail in results:
        if status == "FAIL" and detail and "advisory" not in name:
            print(f"\n  --- {name} ---")
            for ln in detail:
                print(f"    {ln[:96]}")

    if not os.environ.get("NS3_ROOT"):
        print("\nNS3_ROOT unset: test_shared_contract only printed its hashes rather than\n"
              "comparing them against the neurosuite-3 checkout.")
    print(f"\n{len(results) - failed}/{len(results)} passed"
          + (f", {failed} FAILED" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        # Ctrl-C during a subprocess otherwise dumps a traceback through
        # subprocess.communicate, which reads like the runner crashed.
        print("\ninterrupted", file=sys.stderr)
        sys.exit(130)
