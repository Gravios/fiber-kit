#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
#  plan_steps.py — unique, addressable names for the steps of a pipeline plan,
#  and the `fk` command (`fk list`, `fk run`).
#
#  A plan may run a stage several times (fiber-kit-exp.yaml runs fiber-realign
#  three times, fiber-cpos twice), so the stage name does not say WHICH run is
#  meant.  Every step therefore gets an id:
#
#      <stage>-<k>     k = the step's occurrence of that stage in plan order,
#                      from 1: fiber-realign-1, fiber-realign-2, fiber-realign-3
#
#  plus, optionally, its own `name:` in the plan (a stable alias that survives
#  reordering, which the positional id does not).  A step is selected by its id,
#  its name, its 1-based position, or -- when the stage occurs once -- the bare
#  stage name.  Names must be unique, may not reuse another step's id, and use
#  letters, digits, '.', '_' and '-'.
#
#  Qt-free and dependency-light (PyYAML), so scripts/fiber-pipeline's plan parser
#  imports it, as do `fk list` / `fk run` and fiber-plan-lint.
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys

TAG_FIELDS = ("in", "out", "units", "spk", "cpos", "drift")
NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class PlanError(ValueError):
    """A plan, a step name or a step selector that cannot be used."""


def load_steps(path):
    """The plan's steps, in order: a list of dicts with stage, the tag fields, params, args and name
    (all present, '' / {} / [] when absent).  Accepts a plan file or a fiber-kit.yaml whose
    `pipeline:` section is the plan."""
    try:
        import yaml
    except ImportError:
        raise PlanError("reading a plan needs PyYAML (pip install pyyaml)") from None
    try:
        with open(path) as fh:
            doc = yaml.safe_load(fh) or {}
    except OSError as e:
        raise PlanError(f"cannot read plan {path}: {e}") from None
    except Exception as e:                                   # noqa: BLE001 - any YAML error
        raise PlanError(f"cannot parse plan {path}: {e}") from None
    raw = doc.get("pipeline") if isinstance(doc, dict) else doc
    if not isinstance(raw, list) or not raw:
        raise PlanError(f"{path}: no 'pipeline:' list of steps")
    steps = []
    for i, st in enumerate(raw, 1):
        if not isinstance(st, dict) or not st.get("stage"):
            raise PlanError(f"{path}: step {i} is not a mapping with a 'stage'")
        d = {"stage": str(st["stage"])}
        for k in TAG_FIELDS:
            v = st.get(k, "")
            d[k] = "" if v is None else str(v)
        d["params"] = dict(st.get("params") or {})
        d["args"] = [str(a) for a in (st.get("args") or [])]
        nm = st.get("name", "")
        d["name"] = "" if nm is None else str(nm)
        steps.append(d)
    assign_ids(steps)
    return steps


def assign_ids(steps):
    """Give every step its positional id (<stage>-<k>) in place and check the explicit names.
    `steps` are dicts with 'stage' and 'name' (or objects with .stage / .name).  Returns the ids."""
    def get(s, k):
        return s.get(k, "") if isinstance(s, dict) else getattr(s, k, "")
    count = {}
    ids = []
    for s in steps:
        st = get(s, "stage")
        count[st] = count.get(st, 0) + 1
        ids.append(f"{st}-{count[st]}")
    errors = name_errors(steps, ids)
    if errors:
        raise PlanError("; ".join(errors))
    for s, i in zip(steps, ids):
        if isinstance(s, dict):
            s["id"] = i
    return ids


def name_errors(steps, ids=None):
    """Problems with the steps' explicit names (empty list = fine)."""
    def get(s, k):
        return s.get(k, "") if isinstance(s, dict) else getattr(s, k, "")
    if ids is None:
        count = {}
        ids = []
        for s in steps:
            st = get(s, "stage")
            count[st] = count.get(st, 0) + 1
            ids.append(f"{st}-{count[st]}")
    errors = []
    seen = {}
    idset = {i: n for n, i in enumerate(ids, 1)}
    for n, s in enumerate(steps, 1):
        nm = str(get(s, "name") or "")
        if not nm:
            continue
        if not NAME_RE.match(nm):
            errors.append(f"step {n}: name {nm!r} may only hold letters, digits, '.', '_' and '-'")
        elif nm.isdigit():
            errors.append(f"step {n}: name {nm!r} is a number (numbers select steps by position)")
        elif nm in seen:
            errors.append(f"step {n}: name {nm!r} is already the name of step {seen[nm]}")
        elif nm in idset and idset[nm] != n:
            errors.append(f"step {n}: name {nm!r} is step {idset[nm]}'s id")
        seen.setdefault(nm, n)
    return errors


def resolve(steps, selectors):
    """Indices (0-based, plan order, no repeats) of the steps the selectors name.  A selector is a
    step id, a step name, a 1-based position, a stage that occurs once, or a range A..B of any of
    these (inclusive; either end may be left open).  Comma-separated lists are split."""
    by_key = {}
    for i, s in enumerate(steps):
        by_key[s["id"]] = i
        if s["name"]:
            by_key[s["name"]] = i
    stage_runs = {}
    for i, s in enumerate(steps):
        stage_runs.setdefault(s["stage"], []).append(i)

    def one(tok):
        if tok in by_key:
            return by_key[tok]
        if tok.isdigit():
            n = int(tok)
            if 1 <= n <= len(steps):
                return n - 1
            raise PlanError(f"step {n} does not exist (the plan has {len(steps)} steps)")
        runs = stage_runs.get(tok)
        if runs and len(runs) == 1:
            return runs[0]
        if runs:
            raise PlanError(f"{tok!r} runs {len(runs)} times in this plan; name one: "
                            + ", ".join(steps[i]["id"] for i in runs))
        raise PlanError(f"no step {tok!r} in this plan (`fk list` shows the step names)")

    picked = []
    for sel in selectors:
        for tok in [t.strip() for t in str(sel).split(",") if t.strip()]:
            if ".." in tok:
                a, b = tok.split("..", 1)
                lo = one(a) if a else 0
                hi = one(b) if b else len(steps) - 1
                if hi < lo:
                    raise PlanError(f"range {tok!r} runs backwards (step {lo + 1} .. step {hi + 1})")
                rng = range(lo, hi + 1)
            else:
                rng = [one(tok)]
            for i in rng:
                if i not in picked:
                    picked.append(i)
    return sorted(picked)


def default_plan(plan=None):
    """The plan fiber-pipeline would run: an explicit file, else the `pipeline:` section of
    $FK_CONFIG (default ./fiber-kit.yaml).  None when there is none."""
    if plan:
        return plan
    cfg = os.environ.get("FK_CONFIG") or os.path.join(os.getcwd(), "fiber-kit.yaml")
    if os.path.isfile(cfg):
        with open(cfg) as fh:
            if any(re.match(r"^pipeline:\s*$", ln) for ln in fh):
                return cfg
    return None


def step_args(s):
    """The step's extra stage flags, as fiber-pipeline passes them (params, then args)."""
    args = []
    for k, v in s["params"].items():
        flag = "--" + str(k).replace("_", "-").lstrip("-")
        if v is True or v is None or v == "":
            args.append(flag)
        else:
            args += [flag, str(v)]
    return args + list(s["args"])


def runner_rows(path, selectors=(), known=None):
    """fiber-pipeline's plan rows: one \\x1f-separated line per step -- selected flag (1 when the
    step runs: all steps without selectors), id, name, stage, in, out, units, spk, cpos, drift,
    shell-quoted args.  `known`: the stages the runner can dispatch; any other stage is refused
    before a step runs."""
    steps = load_steps(path)
    if known:
        for n, s in enumerate(steps, 1):
            if s["stage"] not in known:
                raise PlanError(f"plan step {n}: unknown stage {s['stage']!r} (known: {', '.join(sorted(known))})")
    sel = set(resolve(steps, selectors)) if selectors else set(range(len(steps)))
    rows = []
    for i, s in enumerate(steps):
        rows.append("\x1f".join([
            "1" if i in sel else "0", s["id"], s["name"], s["stage"],
            *(s[k] for k in TAG_FIELDS), " ".join(shlex.quote(a) for a in step_args(s))]))
    return "\n".join(rows)


# ───────────────────────────── the `fk` command ─────────────────────────────
def _table(steps, selected=None):
    head = ["#", "step", "name", "in", "out", "flags"]
    rows = []
    for i, s in enumerate(steps):
        extra = [f"{k}={s[k]}" for k in ("units", "cpos", "spk", "drift") if s[k]]
        flags = " ".join(extra + [shlex.quote(a) for a in step_args(s)])
        mark = "" if selected is None else ("*" if i in selected else " ")
        rows.append([f"{mark}{i + 1}", s["id"], s["name"], s["in"] or "-", s["out"] or "-", flags])
    w = [max(len(str(r[c])) for r in rows + [head]) for c in range(len(head) - 1)]
    out = ["  ".join(h.ljust(w[c]) for c, h in enumerate(head[:-1])) + "  " + head[-1]]
    for r in rows:
        out.append("  ".join(str(r[c]).ljust(w[c]) for c in range(len(head) - 1)) + "  " + r[-1])
    return "\n".join(o.rstrip() for o in out)


def _cmd_list(a):
    path = default_plan(a.plan)
    if not path:
        raise PlanError("no plan: give --plan <file>, or run where fiber-kit.yaml (or $FK_CONFIG) has a "
                        "'pipeline:' section")
    steps = load_steps(path)
    selected = set(resolve(steps, a.select)) if a.select else None
    if a.names:
        for i, s in enumerate(steps):
            if selected is None or i in selected:
                print(s["name"] or s["id"])
        return 0
    print(f"plan: {path}  ({len(steps)} steps)")
    print(_table(steps, selected))
    return 0


def _cmd_run(a):
    path = default_plan(a.plan)
    if not path:
        raise PlanError("no plan to run steps of: give --plan <file>, or run where fiber-kit.yaml (or "
                        "$FK_CONFIG) has a 'pipeline:' section")
    steps = load_steps(path)
    resolve(steps, a.steps)                                  # fail here, with the plan in hand
    exe = shutil.which("fiber-pipeline") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts", "fiber-pipeline")
    cmd = [exe, str(a.elec), "--plan", path] + [x for s in a.steps for x in ("--step", s)]
    if a.method:
        cmd += ["--method", a.method]
    if a.dry_run:
        cmd.append("--dry-run")
    return subprocess.call(cmd)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="fk", description="fiber-kit pipeline plans: list the steps by their unique names, run chosen "
                               "steps.  A step is named <stage>-<k> (its k-th run of that stage, in plan "
                               "order) or by its own `name:` in the plan; a stage that runs once is also its "
                               "bare stage name.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("list", help="list the plan's steps and their names")
    pl.add_argument("select", nargs="*", help="only these steps (names, positions, ranges A..B)")
    pl.add_argument("--plan", help="plan file (default: the 'pipeline:' section of $FK_CONFIG or "
                                   "./fiber-kit.yaml, as fiber-pipeline)")
    pl.add_argument("--names", action="store_true", help="print only the step names, one per line")
    pr = sub.add_parser("run", help="run chosen steps of the plan (fiber-pipeline <elec> --plan P --step ...)")
    pr.add_argument("elec", type=int, help="electrode group")
    pr.add_argument("steps", nargs="+", help="steps: names, positions, ranges A..B (e.g. fiber-realign-2, "
                                             "fiber-consensus-1..fiber-cpos-2)")
    pr.add_argument("--plan", help="plan file (default as for `fk list`)")
    pr.add_argument("--method", "--variant", dest="method", help="session method token for the run")
    pr.add_argument("-n", "--dry-run", action="store_true", help="print the stage commands only")
    a = ap.parse_args(argv)
    try:
        return _cmd_list(a) if a.cmd == "list" else _cmd_run(a)
    except PlanError as e:
        print(f"fk: {e}", file=sys.stderr)
        return 2


def _runner_main(argv, known=None):
    """`rows <plan> [selector ...]` -- fiber-pipeline's plan parser (it passes `known`)."""
    if len(argv) < 2 or argv[0] != "rows":
        print("usage: python3 -m fiber_kit.plan_steps rows <plan> [selector ...]", file=sys.stderr)
        return 2
    try:
        sys.stdout.write(runner_rows(argv[1], argv[2:], known))
    except PlanError as e:
        print(f"fiber-pipeline: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_runner_main(sys.argv[1:]))
