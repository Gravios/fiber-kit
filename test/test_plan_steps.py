#!/usr/bin/env python3
# test_plan_steps.py -- unique plan step names (plan_steps), `fk list` / `fk run`, and
# fiber-pipeline's step selection (--step / --from / --to / positional step names).
#
#   * ids: <stage>-<k>, the k-th run of a stage in plan order (fiber-kit-exp.yaml: fiber-realign-1..3);
#   * explicit `name:` aliases; malformed, duplicate, numeric names and a name that is another
#     step's id are refused (plan_steps and fiber-plan-lint);
#   * selectors: id, name, position, a stage that runs once, ranges A..B (open ends), comma lists;
#     a stage that runs several times is refused with the ids to choose from;
#   * fiber-pipeline --dry-run runs exactly the selected steps with their plan tags/params, keeps the
#     plan's units bookkeeping for a fiber-link selected alone, accepts step names as positionals,
#     refuses step names when there is no plan; a bare stage name keeps its stand-alone meaning;
#   * `fk run --dry-run` forwards to the same run; fiber-plan-edit's model round-trips `name:`.
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from fiber_kit import plan_steps as ps   # noqa: E402

EXP = os.path.join(ROOT, "src", "fiber_kit", "fiber-kit-exp.yaml")
PIPE = os.path.join(ROOT, "scripts", "fiber-pipeline")
fails = 0


def check(ok, what):
    global fails
    print(("  ok:   " if ok else "  FAIL: ") + what)
    fails += not ok


def raises(fn, needle=""):
    try:
        fn()
    except ps.PlanError as e:
        return needle in str(e)
    return False


def write(d, name, text):
    p = os.path.join(d, name)
    with open(p, "w") as fh:
        fh.write(text)
    return p


def env():
    e = {k: v for k, v in os.environ.items() if not k.startswith("FK_")}
    e["PYTHONPATH"] = os.pathsep.join([os.path.join(ROOT, "src"), e.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    return e


def pipe(cwd, *args, cfg=None):
    e = env()
    e["FK_CONFIG"] = cfg or os.path.join(cwd, "none.yaml")
    r = subprocess.run(["bash", PIPE] + list(args) + ["--dry-run"], cwd=cwd, capture_output=True,
                       text=True, env=e, timeout=300)
    return r.returncode, r.stdout + r.stderr


def headers(out):
    return [ln.split("] ", 1)[1].split(" ==")[0] for ln in out.splitlines() if ln.startswith("== [plan ")]


PLAN = """pipeline:
  - {stage: fiber-session,    out: s}
  - {stage: fiber-realign,    in: s}
  - {stage: fiber-intrachunk, in: s, out: chunkA}
  - {stage: fiber-realign,    in: chunkA, name: realign-after-chunk}
  - {stage: fiber-link,       in: s, out: linked}
  - {stage: fiber-consensus,  in: linked, out: cons, params: {quarantine-only: 1}}
"""


def main():
    print("ids and names:")
    steps = ps.load_steps(EXP)
    ids = [s["id"] for s in steps]
    check(ids[:3] == ["fiber-session-1", "fiber-realign-1", "fiber-consensus-1"]
          and ids.count("fiber-realign-1") == 1 and "fiber-realign-3" in ids and "fiber-cpos-2" in ids,
          "fiber-kit-exp.yaml: <stage>-<k> ids (" + ", ".join(ids[:3]) + ", ...)")
    check(len(set(ids)) == len(ids), "every id unique")
    with tempfile.TemporaryDirectory() as d:
        p = write(d, "p.yaml", PLAN)
        st = ps.load_steps(p)
        check(st[3]["id"] == "fiber-realign-2" and st[3]["name"] == "realign-after-chunk", "explicit name kept beside the id")
        for bad, needle in (("1fiber", None), ("a b", "letters"), ("7", "number"),
                            ("fiber-realign-1", "id"), ("realign-after-chunk", "already")):
            txt = PLAN.replace("  - {stage: fiber-session,    out: s}",
                               "  - {stage: fiber-session,    out: s, name: '%s'}" % bad)
            q = write(d, "bad.yaml", txt)
            if needle is None:
                check(ps.load_steps(q)[0]["name"] == bad, f"name {bad!r} accepted")
            else:
                check(raises(lambda: ps.load_steps(q), needle), f"name {bad!r} refused ({needle})")

        print("selectors:")
        r = lambda *s: ps.resolve(st, list(s))              # noqa: E731
        check(r("fiber-realign-2") == [3] and r("realign-after-chunk") == [3], "by id and by name")
        check(r("3") == [2] and r("fiber-link") == [4], "by position; a stage that runs once by its bare name")
        check(raises(lambda: r("fiber-realign"), "fiber-realign-1, fiber-realign-2"),
              "a stage that runs twice is refused, naming its ids")
        check(r("fiber-realign-1..fiber-link-1") == [1, 2, 3, 4] and r("..2") == [0, 1] and r("5..") == [4, 5],
              "ranges, open at either end")
        check(r("fiber-link-1,1", "1") == [0, 4], "comma lists, de-duplicated, in plan order")
        check(raises(lambda: r("fiber-link-1..2"), "backwards") and raises(lambda: r("9"), "does not exist")
              and raises(lambda: r("nope"), "fk list"), "backwards range, missing position, unknown name refused")

        print("fk list / fk run:")
        o = subprocess.run([sys.executable, "-m", "fiber_kit.plan_steps"], capture_output=True, text=True,
                           env=env())
        check(o.returncode == 2, "python -m plan_steps without 'rows' prints usage")
        lst = subprocess.run([sys.executable, "-c", "import sys; from fiber_kit.plan_steps import main; "
                              "sys.exit(main(sys.argv[1:]))", "list", "--plan", p], capture_output=True,
                             text=True, env=env())
        check(lst.returncode == 0 and "fiber-realign-2" in lst.stdout and "realign-after-chunk" in lst.stdout
              and "--quarantine-only 1" in lst.stdout, "fk list shows ids, names and flags")
        nm = subprocess.run([sys.executable, "-c", "import sys; from fiber_kit.plan_steps import main; "
                             "sys.exit(main(sys.argv[1:]))", "list", "--plan", p, "--names"],
                            capture_output=True, text=True, env=env())
        check(nm.stdout.split() == ["fiber-session-1", "fiber-realign-1", "fiber-intrachunk-1",
                                    "realign-after-chunk", "fiber-link-1", "fiber-consensus-1"],
              "fk list --names: one name per step (its own name when it has one)")
        e2 = env(); e2["FK_CONFIG"] = os.path.join(d, "none.yaml")
        noplan = subprocess.run([sys.executable, "-c", "import sys; from fiber_kit.plan_steps import main; "
                                 "sys.exit(main(sys.argv[1:]))", "list"], capture_output=True, text=True,
                                env=e2, cwd=d)
        check(noplan.returncode == 2 and "no plan" in noplan.stderr, "fk list without a plan says so")

        print("fiber-pipeline:")
        sess = os.path.join(d, "mysess"); os.mkdir(sess)
        rc, out = pipe(sess, "5", "--plan", p, "--step", "realign-after-chunk")
        check(rc == 0 and headers(out) == ["fiber-realign-2 (realign-after-chunk)"]
              and "mysess.clu.stderiv.5.chunkA" in out, "--step <name>: that step only, with its plan 'in'")
        rc, out = pipe(sess, "5", "--plan", p, "fiber-link-1")
        check(rc == 0 and headers(out) == ["fiber-link-1"] and "mysess.clu.stderiv.5.chunkA.units.npz" in out,
              "positional step; a lone fiber-link still reads the units its plan intrachunk emitted")
        rc, out = pipe(sess, "5", "--plan", p, "--from", "fiber-realign-2", "--to", "fiber-link-1")
        check(rc == 0 and headers(out) == ["fiber-realign-2 (realign-after-chunk)", "fiber-link-1"],
              "--from / --to: the span")
        rc, out = pipe(sess, "5", "--plan", p, "--step", "fiber-consensus-1")
        check(rc == 0 and "--quarantine-only 1" in out, "the step's params are passed")
        rc, out = pipe(sess, "5", "--plan", p)
        check(rc == 0 and len(headers(out)) == 6, "no selector: the whole plan")
        rc, out = pipe(sess, "5", "--plan", p, "--step", "fiber-realign")
        check(rc != 0 and "fiber-realign-1, fiber-realign-2" in out, "ambiguous bare stage refused")
        rc, out = pipe(sess, "5", "fiber-realign-2")
        check(rc == 2 and "no plan" in out, "a step name without any plan is refused")
        rc, out = pipe(sess, "5", "fiber-realign-2", cfg=p)
        check(rc == 0 and headers(out) == ["fiber-realign-2 (realign-after-chunk)"],
              "positional step from the plan embedded in $FK_CONFIG")
        rc, out = pipe(sess, "5", "fiber-realign", cfg=p)
        check(rc == 0 and not headers(out) and "== fiber-realign" in out,
              "a bare stage name still runs that stage stand-alone (not a plan step)")
        e3 = env(); e3["FK_CONFIG"] = p; e3["PATH"] = os.path.join(ROOT, "scripts") + os.pathsep + e3["PATH"]
        fr = subprocess.run([sys.executable, "-c", "import sys; from fiber_kit.plan_steps import main; "
                             "sys.exit(main(sys.argv[1:]))", "run", "5", "fiber-intrachunk-1", "--dry-run"],
                            capture_output=True, text=True, env=e3, cwd=sess)
        check(fr.returncode == 0 and headers(fr.stdout) == ["fiber-intrachunk-1"], "fk run --dry-run forwards to fiber-pipeline")
        fkrun = lambda *x, cfg=p: subprocess.run(                      # noqa: E731
            [sys.executable, "-c", "import sys; from fiber_kit.plan_steps import main; sys.exit(main(sys.argv[1:]))",
             "run", *x, "--dry-run"], capture_output=True, text=True, cwd=sess,
            env=dict(e3, FK_CONFIG=cfg))
        fr = fkrun("5")
        check(fr.returncode == 0 and len(headers(fr.stdout)) == 6, "fk run 5: the whole plan")
        fr = fkrun("5", "--from", "fiber-realign-2", "--to", "fiber-link-1")
        check(fr.returncode == 0 and headers(fr.stdout) == ["fiber-realign-2 (realign-after-chunk)", "fiber-link-1"],
              "fk run 5 --from / --to: the span")
        fr = fkrun("5", cfg=os.path.join(d, "none.yaml"))
        check(fr.returncode == 0 and not headers(fr.stdout) and "== fiber-session" in fr.stdout,
              "fk run 5 without a plan: the canonical stages, as fiber-pipeline 5")
        fr = fkrun("5", "fiber-realign-2", cfg=os.path.join(d, "none.yaml"))
        check(fr.returncode == 2 and "no plan" in fr.stderr, "fk run 5 <step> without a plan is refused")

        print("fiber-plan-edit model / fiber-plan-lint:")
        try:
            from fiber_kit import fiber_pipeline_editor as ed
        except Exception as e:                                       # noqa: BLE001 - editor deps absent
            print(f"  skip: editor module not importable ({e})")
        else:
            steps2 = ed.load_plan(p)
            check(steps2[3].name == "realign-after-chunk", "load_plan reads name")
            check("name: realign-after-chunk" in ed.dump_plan(steps2, ordered=False), "dump_plan writes it back")
            steps2[0].name = "realign-after-chunk"
            errs, _ = ed.lint(steps2)
            check(any("already" in x for x in errs), "lint: a duplicate name is an error")

    print(f"\ntest_plan_steps: {'FAIL' if fails else 'PASS'} ({fails} failed)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
