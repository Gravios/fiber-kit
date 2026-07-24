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
import ast
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
_MODULE, _ENTRY = sys.argv[1], sys.argv[2]
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
    getattr(m, _ENTRY)()
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


def entry_parses_args(path, fn="main"):
    """True when the module's ENTRY function actually reaches argparse.

    Checked on main() specifically, not the whole file: fiber_pipeline_editor
    builds an ArgumentParser elsewhere for its lint entry point while its main()
    goes straight to QApplication, so a file-wide search says yes and the probe
    then runs the GUI.  Checked per ENTRY POINT, since one module can back
    several commands with different functions.
    """
    try:
        tree = ast.parse(open(path, errors="ignore").read())
    except (OSError, SyntaxError):
        return True                      # unreadable: probe it and let it report
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name == fn:
            for m in ast.walk(n):
                if isinstance(m, ast.Call) and \
                        getattr(m.func, "attr", getattr(m.func, "id", "")) in (
                            "parse_args", "parse_known_args"):
                    return True
            return False
    return True


def console_scripts():
    text = open(os.path.join(ROOT, "pyproject.toml"), errors="ignore").read()
    m = re.search(r"^\[project\.scripts\]\s*$(.*?)(?=^\[)", text, re.M | re.S)
    out = {}
    for line in (m.group(1).splitlines() if m else []):
        e = re.match(r'\s*([\w.-]+)\s*=\s*"([\w.]+):(\w+)"', line)
        if e:
            # Keep the entry FUNCTION: fiber-plan-lint is
            # fiber_pipeline_editor:lint_main, and probing main() instead would
            # test a different program from the one the command runs.
            out[e.group(1)] = (e.group(2), e.group(3))
    return out


def main():
    verbose = "-v" in sys.argv
    code = PROBE % {"src": SRC}
    ok = skipped = failed = 0
    problems = []
    for cmd, (mod, fn) in sorted(console_scripts().items()):
        # Skip a stage whose main() never reaches parse_args -- ndm_fiber-kit reads
        # sys.argv directly, and the GUI editors build a QApplication and call
        # app.exec().  The probe stops AT parse_args, so for those it stops nowhere:
        # it runs the program.  On a machine without PySide6 that surfaced as a
        # tidy ImportError skip; with Qt installed it OPENS THE EDITOR and blocks
        # until the window is closed.  Detected from main()'s own body rather than
        # an allowlist, so a stage that does parse args can never be excused by it.
        src_path = os.path.join(SRC, *mod.split(".")) + ".py"
        if not entry_parses_args(src_path, fn):
            skipped += 1
            if verbose:
                print(f"  skip  {cmd:24s} does not use argparse")
            continue
        try:
            env = dict(os.environ, QT_QPA_PLATFORM="offscreen", MPLBACKEND="Agg")
            r = subprocess.run([sys.executable, "-c", code, mod, fn], env=env,
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
