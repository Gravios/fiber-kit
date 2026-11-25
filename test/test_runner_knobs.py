#!/usr/bin/env python3
"""test_runner_knobs.py — the knobs the runner turns into flags do what they say.

`scripts/fiber-pipeline` reads some FK_* knobs into shell variables and passes them to the stage as
flags.  Two things went wrong with that route, both checked here against the runner's own text:

  * a switch the runner passes only when its knob is ON did nothing when the stage's default was
    already on: FK_SESSION_DIPSPLIT=0 left the dip split running (--dipsplit was simply not passed
    and fiber-session defaults to it).  Every switch flag must change the stage's default, in the
    state the runner passes it in, or be paired with the flag of the other state.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPE = os.path.join(ROOT, "scripts", "fiber-pipeline")
sys.path.insert(0, os.path.join(ROOT, "src"))
from fiber_kit import stages                                  # noqa: E402
from fiber_kit import fiber_pipeline_editor as ed             # noqa: E402

fails = 0
with open(PIPE) as _fh:
    RUN = _fh.read()

# shell variable -> (FK knob, the runner's built-in default)
VARS = {m.group(1): (m.group(2), m.group(3))
        for m in re.finditer(r'^([A-Z_0-9]+)="\$\{(FK_[A-Z0-9_]+):-([^}]*)\}"', RUN, re.M)}


def check(ok, what):
    global fails
    fails += not ok
    print(("  ok:   " if ok else "  FAIL: ") + what)


def runner_flags(stage):
    """What the runner's function of `stage` passes: [(flag, shell variable, 'value' | 'on' | 'off')]."""
    fn = stage.replace("-", "_")
    i = RUN.index(f"\n{fn}() {{")
    body = RUN[i:RUN.index("\n}\n", i)]
    out = [(f, v, "value") for f, v in re.findall(r'(--[a-z0-9-]+) "\$\{?([A-Z_0-9]+)\}?"', body)]
    for kind in ("on", "off"):
        for v, flags in re.findall(r'_%s "\$\{?([A-Z_0-9]+)[^"]*"\s+\w+\s*&&\s*\w+\+=\(([^)]*)\)' % kind, body):
            first = flags.split()[0]
            out.append((first, v, kind))
            out += [(f, v2, "value") for f, v2 in re.findall(r'(--[a-z0-9-]+) "\$\{?([A-Z_0-9]+)\}?"', flags)
                    if (f, v2, "value") not in out]
    return out


_PARSERS = {}


def parser(stage):
    if stage not in _PARSERS:
        _PARSERS[stage] = ed._capture_parser(stages.module(stage))
    return _PARSERS[stage]


def actions(stage):
    return {s: a for a in parser(stage)._actions for s in a.option_strings}


def main():
    n = 0
    for st in stages.names():
        acts = None
        for flag, var, kind in runner_flags(st):
            if kind == "value":
                continue
            acts = acts or actions(st)
            a = acts.get(flag)
            if a is None:
                check(False, f"{st}: the runner passes {flag}, which the stage does not take")
                continue
            n += 1
            # a switch whose flag restates the default is fine when the other state passes a flag that
            # changes it (--dipsplit when on, --no-dipsplit when off)
            other = [acts.get(f2) for f2, v2, k2 in runner_flags(st) if v2 == var and k2 not in ("value", kind)]
            if any(o is not None and o.nargs == 0 and o.default != o.const for o in other):
                continue
            if a.nargs == 0 and a.default == a.const:
                check(False, f"{st}: {flag} (passed when {var} is {kind}) sets {a.dest}={a.const!r}, already the "
                             f"default: the knob {VARS.get(var, ('?',))[0]} does nothing")
    check(n >= 15, f"{n} switch flags of the runner each change their stage's default")
    sess = dict((f, k) for f, _v, k in runner_flags("fiber-session"))
    check(sess.get("--dipsplit") == "on" and sess.get("--no-dipsplit") == "off",
          "FK_SESSION_DIPSPLIT: on passes --dipsplit, off passes --no-dipsplit")
    print(f"test_runner_knobs: {'PASS' if not fails else 'FAIL'} ({fails} failed)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
