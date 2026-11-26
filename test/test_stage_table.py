#!/usr/bin/env python3
"""test_stage_table.py — the stage lists agree with the one table (fiber_kit/stages.py).

The runner's arrays and dispatch tables, the plan editor's catalogue, the console entry points and
`fiber-pipeline --list` each named the stages on their own, and had drifted: `fiber-plan-lint`
refused three of the four shipped profiles at step 0 (it did not know fiber-flag-artifacts),
`--list` stopped with "POST_SUMMARY[$i]: unbound variable", and a plan step fiber-stochastic was
dispatched by the runner but refused by its plan parser.  Checked here:

  * every stage of the table has a module file and a console entry point on that module;
  * the runner's five arrays are the table's five groups, in the table's order;
  * both dispatch tables of the runner, and a stage function, exist for every stage;
  * the editor's catalogue names exactly the table's stages;
  * `fiber-pipeline --list` prints the table and exits 0;
  * every shipped profile's plan lints without an error, and the runner's parser accepts a plan
    that runs every stage.
"""
import glob
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
PIPE = os.path.join(ROOT, "scripts", "fiber-pipeline")
sys.path.insert(0, SRC)
from fiber_kit import stages, plan_steps                     # noqa: E402
from fiber_kit import fiber_pipeline_editor as ed            # noqa: E402

fails = 0


def check(ok, what):
    global fails
    fails += not ok
    print(("  ok:   " if ok else "  FAIL: ") + what)


def main():
    names = stages.names()
    check(len(names) == len(set(names)) and len(names) >= 23, f"{len(names)} stages, each named once")
    check(all(s.group in dict(stages.GROUPS) for s in stages.STAGES), "every stage is in a known group")

    with open(os.path.join(ROOT, "pyproject.toml")) as fh:
        eps = dict(re.findall(r'^([a-z_][a-z0-9_-]*)\s*=\s*"(fiber_kit\.[a-z_0-9]+):main"', fh.read(), re.M))
    missing = [n for n in names if not os.path.isfile(os.path.join(SRC, *stages.module(n).split(".")) + ".py")]
    check(not missing, f"every stage has its module file ({missing})")
    wrong = [n for n in names if eps.get(n) != stages.module(n)]
    check(not wrong, f"every stage has a console entry point on its module ({wrong})")

    with open(PIPE) as fh:
        run = fh.read()
    for var, group in (("STAGES", "core"), ("CURATED_STAGES", "curated"), ("PRE_STAGES", "pre"),
                       ("ALT_STAGES", "alt"), ("POST_STAGES", "post")):
        m = re.search(r"^%s=\(([^)]*)\)" % var, run, re.M)
        got = m.group(1).split() if m else None
        check(got == stages.names(group), f"the runner's {var} is the table's '{group}' group ({got})")
    for fn in ("run_stage_direct", "run_plan_stage_direct"):
        body = run[run.index(fn + "() {"):]
        body = body[:body.index("\n}\n")]
        got = set(re.findall(r"^\s+(fiber-[a-z-]+)\)", body, re.M))
        check(got == set(names), f"{fn} dispatches exactly the table's stages ({sorted(got ^ set(names))})")
    nofn = [n for n in names if not re.search(r"^%s\(\) \{" % n.replace("-", "_"), run, re.M)]
    check(not nofn, f"the runner has a function for every stage ({nofn})")
    m = re.search(r'^SHIELD_STAGES="([^"]*)"', run, re.M)
    check(m is not None and set(m.group(1).split()) <= set(names), "the runner's SHIELD_STAGES are stages of the table")

    check(set(ed.CATALOG) == set(names), f"the editor's catalogue names the table's stages ({sorted(set(ed.CATALOG) ^ set(names))})")
    check(ed.STAGE_MODULES == {n: stages.module(n) for n in names}, "the editor's module map is the table's")

    # --in / --out / --method: one spelling on every stage (stages.IO)
    check(set(stages.IO) <= set(names), "stages.IO names stages of the table")
    for st in names:
        acts = ed._capture_parser(stages.module(st))._option_string_actions
        io = stages.IO.get(st, {})
        wrong = [f"{short}->{full}" for short, full in io.items() if acts.get(short) is None or acts.get(short) is not acts.get(full)]
        check(not wrong, f"{st}: {', '.join(io) or 'no short names'} {'are' if len(io) != 1 else 'is'} the flag stages.IO says ({wrong})")
        # nothing left out: a stage that takes the long name and has the short one free must be in IO
        free = [short for short, longs in (("--in", ("--in-stage", "--clu-stage")), ("--out", ("--out-stage", "--out-tag")),
                                           ("--method", ("--clu-method",)))
                if short not in acts and any(f in acts for f in longs)
                and not (short == "--in" and "--in-stage" not in acts and st in ("fiber-session", "fiber-stochastic"))]
        check(not free, f"{st}: no short name is missing ({free})")
        # and what the runner passes as the plan's in: / out: is what the short names stand for
        fn = st.replace("-", "_")
        body = run[run.index(f"\n{fn}() {{"):]
        body = body[:body.index("\n}\n")]
        for short, pat in (("--in", r'(--[a-z0-9-]+) "\$\{?(?:STEP_IN|min)\b'), ("--out", r'(--[a-z0-9-]+) "\$\{?(?:STEP_OUT|mout)\b')):
            passed = set(re.findall(pat, body))
            if short in io and passed:
                check(io[short] in passed, f"{st}: the runner passes the plan's {short[2:]}: as {io[short]} ({sorted(passed)})")

    tmp = tempfile.mkdtemp()
    try:
        env = dict(os.environ, PYTHONPATH=SRC)
        env.pop("FK_CONFIG", None)
        r = subprocess.run(["bash", PIPE, "--list"], cwd=tmp, env=env, capture_output=True, text=True)
        check(r.returncode == 0 and r.stdout.rstrip("\n") == stages.listing(), "fiber-pipeline --list prints the table and exits 0")
        r = subprocess.run([sys.executable, "-m", "fiber_kit.stages", "names", "pre"], cwd=tmp, env=env, capture_output=True, text=True)
        check(r.returncode == 0 and r.stdout.split() == stages.names("pre"), "python -m fiber_kit.stages names <group>")

        profiles = sorted(glob.glob(os.path.join(SRC, "fiber_kit", "fiber-kit*.yaml")))
        check(len(profiles) >= 4, f"{len(profiles)} shipped profiles found")
        for p in profiles:
            try:
                steps = ed.load_plan(p)
            except Exception as e:                           # noqa: BLE001 - the failure is the finding
                if "no 'pipeline:'" in str(e):
                    continue                                 # a profile without a plan has nothing to lint
                check(False, f"{os.path.basename(p)}: the plan loads ({e})")
                continue
            errors, _warnings = ed.lint(steps)
            check(not errors, f"{os.path.basename(p)}: {len(steps)} steps lint without an error ({errors[:2]})")

        plan = os.path.join(tmp, "every.yaml")
        with open(plan, "w") as fh:
            fh.write("pipeline:\n" + "".join(f"  - {{stage: {n}}}\n" for n in names))
        rows = plan_steps.runner_rows(plan, known=set(names))
        check(len(rows.splitlines()) == len(names), "the runner's plan parser accepts a plan that runs every stage")
        r = subprocess.run(["bash", PIPE, "1", "--plan", plan, "--dry-run"], cwd=tmp, env=env, capture_output=True, text=True)
        check(r.returncode == 0 and f"[plan {len(names)}/{len(names)}]" in r.stdout,
              f"a dry run of that plan reaches its last step ({(r.stderr or r.stdout).strip().splitlines()[-1:]})")
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"test_stage_table: {'PASS' if not fails else 'FAIL'} ({fails} failed)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
