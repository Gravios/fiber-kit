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
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")

sys.path.insert(0, os.path.join(ROOT, "tools"))
from stage_probe import console_scripts, entry_parses_args, run_probe  # noqa: E402

# Import the stage, stop at parse_args, report what happened.  Run out-of-process
# so one stage's import side effects cannot leak into the next, and so a hang is
# a timeout rather than a wedged test run.




def main():
    verbose = "-v" in sys.argv
    ok = skipped = failed = 0
    for cmd, (mod, fn) in sorted(console_scripts(ROOT).items()):
        src_path = os.path.join(SRC, *mod.split(".")) + ".py"
        # Nothing to probe when the entry never reaches parse_args: probing those
        # runs the program (ndm_fiber-kit reads sys.argv directly, the editor
        # builds a QApplication).  Derived from the entry's own body, so a stage
        # that DOES parse args can never be excused by it.
        if not entry_parses_args(src_path, fn):
            skipped += 1
            if verbose:
                print(f"  skip  {cmd:24s} entry does not parse args")
            continue
        status, payload = run_probe(SRC, mod, fn, timeout=120)
        if status == "json":
            ok += 1
            if verbose:
                print(f"  ok    {cmd:24s} {len(payload['rows'])} flags")
        elif status == "skip":
            skipped += 1
            if verbose:
                print(f"  skip  {cmd:24s} {payload}")
        else:
            failed += 1
            print(f"  FAIL  {cmd:24s} {payload}")

    print(f"\n{ok} built, {skipped} skipped (environment), {failed} failed")
    if failed:
        print("\nA stage that imports but cannot build its parser is broken for every "
              "user of it;\nthis is the check that static analysis cannot do.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
