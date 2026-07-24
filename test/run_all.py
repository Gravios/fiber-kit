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
        return name, "skip", 0.0, str(e)
    dt = time.time() - t0
    if verbose and out.strip():
        print("─" * 72)
        print(out.rstrip())
        print("─" * 72)
    tail = ""
    for line in reversed(out.strip().splitlines()):
        if line.strip():
            tail = line.strip()[:78]
            break
    return name, ("ok" if code == 0 else "FAIL"), dt, tail


def main():
    ap = argparse.ArgumentParser(prog="run_all")
    ap.add_argument("--fast", action="store_true", help="skip the slow checks")
    ap.add_argument("-v", "--verbose", action="store_true", help="show each check's output")
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
        checks.append(("ruff", ["ruff", "check", "src", "test", "tools"]))

    results = [run(n, c, a.verbose) for n, c in checks]

    print(f"\n{'check':38s} {'result':7s} {'time':>7s}  last line")
    print("-" * 100)
    failed = 0
    for name, status, dt, tail in results:
        failed += status == "FAIL"
        print(f"{name:38s} {status:7s} {dt:6.1f}s  {tail}")

    if not os.environ.get("NS3_ROOT"):
        print("\nNS3_ROOT unset: test_shared_contract only printed its hashes rather than\n"
              "comparing them against the neurosuite-3 checkout.")
    print(f"\n{len(results) - failed}/{len(results)} passed"
          + (f", {failed} FAILED" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
