#!/usr/bin/env python3
"""
stage_probe.py — the one way to get at a stage's ArgumentParser.

Two tools need this: tools/gen_stages_doc.py, to read the parsers and regenerate
docs/stages.md and the plan template, and test/test_stage_startup.py, to assert
every stage can build its parser at all.  They had a copy each, and the copies
drifted in exactly the way everything else in this codebase drifts when it is
mirrored: the smoke test learned to skip a stage whose entry never reaches
parse_args, and to force Qt offscreen, and the generator did not.  So the
generator went on running fiber_pipeline_editor's main(), which builds a
QApplication and calls app.exec() -- opening the editor window and blocking until
its 90 s timeout.  Twice, because two console scripts are backed by that module.
That was 180 s of a 202 s doc check, and two GUI windows in the middle of a test
run.

One implementation, imported by both.

The technique: import the module and call its ENTRY function with
ArgumentParser.parse_args patched to raise, which stops the moment every
add_argument has run and nothing else has.  Out of process, so a stage's import
side effects cannot leak and a hang is a timeout rather than a wedged run.
"""
import ast
import json
import os
import re
import subprocess
import sys

PROBE = r"""
import sys, json, argparse, importlib
sys.path.insert(0, %(src)r)
# argparse falls back to basename(sys.argv[0]) when a parser sets no prog=.
# Pin a sentinel so "did it set prog explicitly?" is answerable.
_AUTO = "<<autoprog>>"
_MODULE, _ENTRY = sys.argv[1], sys.argv[2]   # read BEFORE argv is replaced
sys.argv = [_AUTO]
class _Got(Exception):
    def __init__(self, ap): self.ap = ap
def _stop(self, *a, **k): raise _Got(self)
argparse.ArgumentParser.parse_args = _stop
argparse.ArgumentParser.parse_known_args = _stop
def dump(ap):
    pos, rows = [], []
    for a in ap._actions:
        if a.dest == "help":
            continue
        if not a.option_strings:
            pos.append(a.dest)
            continue
        rows.append(dict(flags=list(a.option_strings), dest=a.dest,
                         default=None if a.default is argparse.SUPPRESS else a.default,
                         suppressed=a.default is argparse.SUPPRESS,
                         choices=list(a.choices) if a.choices else None,
                         help=a.help or "",
                         const=a.__class__.__name__))
    return dict(prog=("" if ap.prog == _AUTO else ap.prog),
                description=ap.description or "",
                positionals=pos, rows=rows)
try:
    m = importlib.import_module(_MODULE)
except ImportError as e:
    print("SKIP import: %%s" %% str(e)[:90]); raise SystemExit(0)
try:
    getattr(m, _ENTRY)()
    print("FAIL entry returned without building a parser")
except _Got as g:
    print("JSON" + json.dumps(dump(g.ap)))
except SystemExit as e:
    print("SKIP exit: %%s" %% str(e)[:90])
except BaseException as e:
    print("FAIL %%s: %%s" %% (type(e).__name__, str(e)[:110]))
"""


def console_scripts(root):
    """command -> (module, entry function), from [project.scripts].

    The entry function matters: fiber-plan-lint is fiber_pipeline_editor:lint_main,
    and probing main() instead tests a different program from the one the command
    runs -- and, for that module, opens the GUI.
    """
    text = open(os.path.join(root, "pyproject.toml"), errors="ignore").read()
    m = re.search(r"^\[project\.scripts\]\s*$(.*?)(?=^\[)", text, re.M | re.S)
    out = {}
    for line in (m.group(1).splitlines() if m else []):
        e = re.match(r'\s*([\w.-]+)\s*=\s*"([\w.]+):(\w+)"', line)
        if e:
            out[e.group(1)] = (e.group(2), e.group(3))
    return out


def entry_parses_args(path, fn="main"):
    """True when that entry function's own body reaches argparse.

    False means there is nothing to probe -- ndm_fiber-kit reads sys.argv directly,
    fiber_pipeline_editor's main() goes straight to QApplication.  Probing those
    does not stop at parse_args; it RUNS them.  Checked on the entry function
    rather than the file, since one module can back several commands and
    fiber_pipeline_editor builds an ArgumentParser elsewhere.
    """
    try:
        tree = ast.parse(open(path, errors="ignore").read())
    except (OSError, SyntaxError):
        return True                       # unreadable: probe it and let it report
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name == fn:
            for m in ast.walk(n):
                if isinstance(m, ast.Call) and \
                        getattr(m.func, "attr", getattr(m.func, "id", "")) in (
                            "parse_args", "parse_known_args"):
                    return True
            return False
    return True


def probe_env():
    """Environment that cannot put a window on screen, whatever slips through."""
    return dict(os.environ, QT_QPA_PLATFORM="offscreen", MPLBACKEND="Agg")


def run_probe(sys_path, module, entry="main", timeout=120):
    """(status, payload): ('json', spec) | ('skip', why) | ('fail', why).

    `sys_path` is prepended to sys.path in the child VERBATIM -- it is the
    directory containing the package, e.g. <repo>/src.  Taking it literally rather
    than deriving it from a package directory removes an off-by-one that silently
    turned every stage into an ImportError skip, which reads as "0 built, 39
    skipped" and looks like an environment problem rather than a bug here.
    """
    code = PROBE % {"src": os.path.abspath(sys_path)}
    try:
        r = subprocess.run([sys.executable, "-c", code, module, entry],
                           env=probe_env(), capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return "fail", f"timed out after {timeout}s building its parser"
    line = ""
    for ln in r.stdout.splitlines():
        if ln.startswith(("JSON", "SKIP", "FAIL")):
            line = ln
    if line.startswith("JSON"):
        try:
            return "json", json.loads(line[4:])
        except ValueError:
            return "fail", "probe emitted unparseable JSON"
    if line.startswith("SKIP"):
        return "skip", line[5:]
    return "fail", line[5:] if line else "no output from probe"


def introspect(pkg_dir, module, entry="main", timeout=120):
    """The captured parser spec, or None when it cannot or must not be probed.

    `pkg_dir` is the package directory itself, e.g. <repo>/src/fiber_kit.
    """
    src_path = os.path.join(pkg_dir, *module.split(".")[1:]) + ".py"
    if not entry_parses_args(src_path, entry):
        return None
    status, payload = run_probe(os.path.join(pkg_dir, ".."), module, entry, timeout)
    return payload if status == "json" else None
