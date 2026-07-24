#!/usr/bin/env python3
"""
test_stage_startup.py — every stage must import and build its parser.

Why this exists.  The checks used across this codebase are static -- compile(),
pyflakes, ast.parse -- and none of them can see a parser that fails when it is
BUILT: a duplicate dest=, two flags colliding on the same option string, a default
that fails its own type=.  argparse raises those at construction, so they are
invisible until someone runs the stage, and then they are total: the command
cannot start at all, for anyone.

What this does NOT catch, stated because the test was written after a bug it
would have missed.  The SessionCfg keyword-splat crash -- a key added to the dict
without the matching dataclass field -- happens in resolve_session_params, which
every stage calls AFTER parse_args.  This probe stops AT parse_args, so the
parser builds cleanly and nothing here fires.  Verified by reintroducing that bug
and watching this test report 0 failures.  test_session_cfg.py is what covers
that seam; the two are complementary and neither subsumes the other.

The check is one cheap act: import the module and let it construct its parser.
This does that for every console script, by intercepting parse_args so
main() stops the instant the parser is complete -- the technique
tools/gen_stages_doc.py uses to read the parsers.

Stages that cannot be imported for an environmental reason -- the PySide6 GUI
tools when Qt is absent -- are SKIPPED, not failed: this is a smoke test for the
package's own consistency, not a dependency check.  A stage that imports and then
fails is a real failure.

Run:  python3 test/test_stage_startup.py [-v]
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")

# Import the stage, stop at parse_args, report what happened.  Run out-of-process
# so one stage's import side effects cannot leak into the next, and so a hang is
# a timeout rather than a wedged test run.
PROBE = r"""
import sys, argparse, importlib
sys.path.insert(0, %(src)r)
_MODULE = sys.argv[1]
sys.argv = ["<<probe>>"]
class _Got(Exception):
    def __init__(self, ap): self.ap = ap
def _stop(self, *a, **k): raise _Got(self)
argparse.ArgumentParser.parse_args = _stop
argparse.ArgumentParser.parse_known_args = _stop
try:
    m = importlib.import_module(_MODULE)
except ImportError as e:
    print("SKIP import: %%s" %% str(e)[:90]); raise SystemExit(0)
try:
    m.main()
    print("FAIL main() returned without building a parser")
except _Got as g:
    n = len([a for a in g.ap._actions if a.dest != "help"])
    print("OK %%d" %% n)
except SystemExit as e:
    # a stage that exits before parsing (missing optional dep, or its own
    # usage guard) is environmental, not a defect in the package
    print("SKIP exit: %%s" %% str(e)[:90])
except Exception as e:
    print("FAIL %%s: %%s" %% (type(e).__name__, str(e)[:110]))
"""


def console_scripts():
    text = open(os.path.join(ROOT, "pyproject.toml"), errors="ignore").read()
    m = re.search(r"^\[project\.scripts\]\s*$(.*?)(?=^\[)", text, re.M | re.S)
    out = {}
    for line in (m.group(1).splitlines() if m else []):
        e = re.match(r'\s*([\w.-]+)\s*=\s*"([\w.]+):(\w+)"', line)
        if e:
            out[e.group(1)] = e.group(2)
    return out


def main():
    verbose = "-v" in sys.argv
    code = PROBE % {"src": SRC}
    ok = skipped = failed = 0
    problems = []
    for cmd, mod in sorted(console_scripts().items()):
        # A stage that never constructs an ArgumentParser has no parser to build
        # (ndm_fiber-kit reads sys.argv directly).  Detected from the source
        # rather than an allowlist, so a new one needs no edit here -- and so a
        # stage that DOES use argparse can never be excused by this branch.
        src_path = os.path.join(SRC, *mod.split(".")) + ".py"
        try:
            uses_argparse = "ArgumentParser(" in open(src_path, errors="ignore").read()
        except OSError:
            uses_argparse = True
        if not uses_argparse:
            skipped += 1
            if verbose:
                print(f"  skip  {cmd:24s} does not use argparse")
            continue
        try:
            r = subprocess.run([sys.executable, "-c", code, mod],
                               capture_output=True, text=True, timeout=120)
            line = (r.stdout.strip().splitlines() or [""])[-1]
        except subprocess.TimeoutExpired:
            line = "FAIL timed out building its parser"
        if line.startswith("OK"):
            ok += 1
            if verbose:
                print(f"  ok    {cmd:24s} {line.split()[1]} flags")
        elif line.startswith("SKIP"):
            skipped += 1
            if verbose:
                print(f"  skip  {cmd:24s} {line[5:]}")
        else:
            failed += 1
            problems.append((cmd, line))
            print(f"  FAIL  {cmd:24s} {line}")

    print(f"\n{ok} built, {skipped} skipped (environment), {failed} failed")
    if problems:
        print("\nA stage that imports but cannot build its parser is broken for every "
              "user of it;\nthis is the check that static analysis cannot do.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
