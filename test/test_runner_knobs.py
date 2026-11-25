#!/usr/bin/env python3
"""test_runner_knobs.py — the knobs the runner turns into flags do what they say.

`scripts/fiber-pipeline` reads some FK_* knobs into shell variables and passes them to the stage as
flags.  Two things went wrong with that route, both checked here against the runner's own text:

  * a switch the runner passes only when its knob is ON did nothing when the stage's default was
    already on: FK_SESSION_DIPSPLIT=0 left the dip split running (--dipsplit was simply not passed
    and fiber-session defaults to it).  Every switch flag must change the stage's default, in the
    state the runner passes it in, or be paired with the flag of the other state.
  * a stage started by hand did not see those knobs at all: `fiber-session <base> 8` beside a
    profile ran on its argparse defaults.  config.RUNNER_KNOBS lists, per stage, the flag each
    such knob becomes, and config.apply_profile_knobs makes them the parser's defaults.  Checked:
    the table is what the runner's stage functions pass; and under a profile that sets every one
    of those knobs to a value of its own, the command the runner builds and the bare command parse
    to the same settings.
  * with NO profile the two still differ where the runner's built-in default is not the stage's
    (the stage's own defaults are an older configuration).  That list is pinned below, so it
    cannot grow unnoticed; which side is right is a decision, not a bug this test can settle.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPE = os.path.join(ROOT, "scripts", "fiber-pipeline")
sys.path.insert(0, os.path.join(ROOT, "src"))
from fiber_kit import stages, config as cfgmod                # noqa: E402
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


# Flags that take a path the runner builds (the artifact mask): not a knob's value, so not in the table.
RUNNER_ONLY = {"--exclude-clu", "--artifact-clu"}

PROBE = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
from fiber_kit import fiber_pipeline_editor as ed, stages
stage, dests = sys.argv[2], json.loads(sys.argv[3])
p = ed._capture_parser(stages.module(stage))
out = {}
for key, argv in json.loads(sys.argv[4]).items():
    ns = vars(p.parse_args(argv))
    out[key] = {d: ns.get(d) for d in dests}
print("RESULT" + json.dumps(out, default=str))
"""


def table_rows(stage):
    """The runner's flags of `stage` that carry a knob, as config.RUNNER_KNOBS writes them."""
    acts = actions(stage)
    return tuple((f, VARS[v][0], k) for f, v, k in runner_flags(stage)
                 if v in VARS and f in acts and f not in RUNNER_ONLY)


def altered(stage):
    """A value for every knob of the stage's table that is neither the runner's nor the stage's default."""
    acts, env = actions(stage), {}
    for flag, knob, kind in cfgmod.RUNNER_KNOBS[stage]:
        a = acts[flag]
        rdef = next(VARS[v][1] for v in VARS if VARS[v][0] == knob)
        if kind != "value":
            env.setdefault(knob, "0" if rdef.strip().lower() in ("1", "on", "yes", "true") else "1")
        elif a.choices:
            env[knob] = str([c for c in a.choices if str(c) not in (str(a.default), rdef)][0])
        elif a.type is int:
            env[knob] = str(int(float(rdef or a.default or 0)) + 3)
        elif a.type is float:
            env[knob] = repr(round(float(rdef or a.default or 0.0) + 0.0625, 4))
        elif a.type is None:
            env[knob] = (rdef or str(a.default or "")) + "x"
        else:                                                # a type of the stage's own (a number or a word)
            try:
                env[knob] = repr(round(float(rdef or a.default or 0.0) + 0.0625, 4))
            except (TypeError, ValueError):
                env[knob] = "0.0625"
    return env


def parsed(stage, tmp, config_text):
    """(settings of the runner's dry-run command, settings of the bare command) under one profile."""
    import json
    import subprocess
    cfg = os.path.join(tmp, "fiber-kit.yaml")
    with open(cfg, "w") as fh:
        fh.write(config_text)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("FK_", "FIBER_"))}
    env.update(PYTHONPATH=os.path.join(ROOT, "src"), FK_CONFIG=cfg)
    r = subprocess.run(["bash", PIPE, "1", stage, "--dry-run"], cwd=tmp, env=env, capture_output=True, text=True)
    line = [ln for ln in r.stdout.splitlines() if ln.strip().startswith("+ " + stage + " ")]
    if r.returncode != 0 or not line:
        return None, None, (r.stderr or r.stdout).strip().splitlines()[-1:]
    import shlex
    argv = shlex.split(line[0].strip()[2:])[1:]
    npos = next((i for i, t in enumerate(argv) if t.startswith("--")), len(argv))
    dests = sorted({actions(stage)[f].dest for f, _k, _kind in cfgmod.RUNNER_KNOBS[stage]})
    r = subprocess.run([sys.executable, "-c", PROBE, os.path.join(ROOT, "src"), stage, json.dumps(dests),
                        json.dumps({"runner": argv, "bare": argv[:npos]})], cwd=tmp, env=env, capture_output=True, text=True)
    res = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT")]
    if not res:
        return None, None, (r.stderr or r.stdout).strip().splitlines()[-1:]
    d = json.loads(res[0][6:])
    return d["runner"], d["bare"], None


# With no profile: settings where the runner's built-in default is not the stage's own.  A run by hand
# without a profile gets the right-hand value, a run through fiber-pipeline the left-hand one.
NO_PROFILE_DIFFERENCES = {      # setting: [through fiber-pipeline, by hand]
    "fiber-session": {"cfiber_gate": [True, False], "dip_alpha": [0.02, 0.01], "dip_dim": [6, 4], "dip_min": [30, 40],
                      "feature_align": ["xcorr", None], "incl_assign": [True, False], "inclusion_k": [2.5, 3.0],
                      "merge_method": ["sliding", "template"], "rkk_delete": [False, True]},
    "fiber-stochastic": {"feature_align": ["xcorr", None], "inclusion_k": [2.5, 3.0],
                         "merge_method": ["sliding", "template"]},
    "fiber-refine": {"chunk_minutes": [12.0, 0.0], "drop_min": [5, None], "feature_align": ["xcorr", None],
                     "fold_off_thr": [0.22, None], "large": [150, 800], "merge_min_sim": [0.96, 0.92],
                     "min_group": [30, 40], "split_var_mult": [1.5, 0.0]},
    "fiber-link": {"align_upsample": [4, 1], "cfiber_q": [0.9, None], "cos_thr": [0.75, 0.975]},
}


def main():
    import shutil
    import tempfile
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
    tmp = tempfile.mkdtemp()
    try:
        for st in sorted(cfgmod.RUNNER_KNOBS):
            check(tuple(cfgmod.RUNNER_KNOBS[st]) == table_rows(st),
                  f"{st}: config.RUNNER_KNOBS is what the runner passes ({len(cfgmod.RUNNER_KNOBS[st])} flags; "
                  f"{sorted(set(cfgmod.RUNNER_KNOBS[st]) ^ set(table_rows(st)))[:3]})")
            alt = altered(st)
            run, bare, err = parsed(st, tmp, "".join(f"{k}: {v}\n" for k, v in alt.items()))
            if err is not None:
                check(False, f"{st}: could not compare the runner's command with the bare one ({err})")
                continue
            diff = {d: (run[d], bare[d]) for d in run if run[d] != bare[d]}
            check(not diff, f"{st}: under a profile the runner's command and the bare command give the same "
                            f"{len(run)} settings ({dict(list(diff.items())[:4])})")
            moved = sum(1 for d in run if run[d] is not None)
            check(moved >= len(run) // 2, f"{st}: the profile's values reached the stage ({moved} of {len(run)} set)")
            run, bare, err = parsed(st, tmp, "")
            if err is not None:
                check(False, f"{st}: could not compare without a profile ({err})")
                continue
            diff = {d: [run[d], bare[d]] for d in run if run[d] != bare[d]}
            check(diff == NO_PROFILE_DIFFERENCES.get(st, {}),
                  f"{st}: without a profile the two differ in the {len(NO_PROFILE_DIFFERENCES.get(st, {}))} pinned "
                  f"settings only (now: {diff})")
        # the shipped profiles themselves: with one of them in force a stage started by hand must get
        # what fiber-pipeline would give it (a knob the profile leaves blank falls under the pinned list)
        import glob
        for prof in sorted(glob.glob(os.path.join(ROOT, "src", "fiber_kit", "fiber-kit*.yaml"))):
            with open(prof) as fh:
                text = fh.read()
            bad = {}
            for st in sorted(cfgmod.RUNNER_KNOBS):
                run, bare, err = parsed(st, tmp, text)
                if err is not None:
                    bad[st] = err
                    continue
                diff = {d: [run[d], bare[d]] for d in run if run[d] != bare[d]}
                extra = {d: v for d, v in diff.items() if NO_PROFILE_DIFFERENCES.get(st, {}).get(d) != v}
                if extra:
                    bad[st] = extra
            check(not bad, f"{os.path.basename(prof)}: every stage by hand = through fiber-pipeline ({bad})")
        missing = [st for st in stages.names() if st not in cfgmod.RUNNER_KNOBS and table_rows(st)]
        check(not missing, f"every stage the runner passes knobs to is in the table ({missing})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"test_runner_knobs: {'PASS' if not fails else 'FAIL'} ({fails} failed)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
