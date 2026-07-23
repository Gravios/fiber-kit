#!/usr/bin/env python3
"""
gen_stages_doc.py — regenerate the per-stage tables in docs/stages.md from the
argument parsers, and fail if the committed doc has drifted from them.

WHY STATIC.  Thirty-four of the thirty-seven stages build their ArgumentParser
inline inside main(), so there is no importable parser object for
sphinx-argparse / argparse-manpage / argdown to reach without refactoring every
one of them.  Reading the AST needs no imports at all, which also means this runs
in CI without numpy, scipy or CUDA installed and cannot be broken by an import
side effect.

WHAT IT PRESERVES.  Section headings, stage ORDER and every hand-written
paragraph are editorial and are left exactly as they are.  Only each stage's
"Positional:" line and its flag table are rewritten, keyed by the
`### `fiber-name`` heading already in the file.  A stage that exists in the code
but has no heading in the doc is REPORTED, not inserted: where it belongs in the
narrative is a judgement the generator does not get to make.

LIMITS, stated because they are the reason to read the output rather than trust
it.  A default that is a literal (97% of them) or a module-level constant is
resolved exactly.  One that is a runtime lookup -- _knob_default(...) reading the
session config -- cannot be, and is rendered `(from config)`.

Usage:
    python3 tools/gen_stages_doc.py            # rewrite docs/stages.md in place
    python3 tools/gen_stages_doc.py --check    # exit 1 if the doc has drifted
"""
import argparse
import ast
import os
import re
import sys

SRC = "src/fiber_kit"
DOC = "docs/stages.md"
PYPROJECT = "pyproject.toml"


def console_scripts(path=PYPROJECT):
    """command name -> module, from [project.scripts].

    This, not prog=, is what the doc headings are keyed on, and rightly: it is the
    name a user actually types.  Parsed with a regex rather than tomllib so the
    generator runs on any Python 3 without an import.
    """
    text = open(path, errors="ignore").read()
    m = re.search(r"^\[project\.scripts\]\s*$(.*?)(?=^\[)", text, re.M | re.S)
    if not m:
        return {}
    out = {}
    for line in m.group(1).splitlines():
        e = re.match(r'\s*([\w.-]+)\s*=\s*"([\w.]+):(\w+)"', line)
        if e:
            cmd, mod, _fn = e.groups()
            out[cmd] = mod.split(".")[-1]
    return out


# ── reading the parsers ──────────────────────────────────────────────────────
def module_constants(tree):
    """Module-level NAME = <literal>, so `default=MIN_SPIKES` can be resolved."""
    out = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 \
                and isinstance(n.targets[0], ast.Name) \
                and isinstance(n.value, ast.Constant):
            out[n.targets[0].id] = n.value.value
    return out


def render_default(node, consts, action):
    if action in ("store_true", "store_false"):
        return "flag (off)" if action == "store_true" else "flag (on)"
    if node is None:
        return "\u2014"
    if isinstance(node, ast.Constant):
        v = node.value
        if v is None or v == "":
            return "\u2014" if v is None else '`""`'
        return f"`{v}`"
    if isinstance(node, ast.Name) and node.id in consts:
        return f"`{consts[node.id]}`"
    if isinstance(node, (ast.List, ast.Tuple)):
        try:
            return f"`{[e.value for e in node.elts]}`"
        except AttributeError:
            pass
    return "(from config)"



# ── flags contributed by shared helpers ──────────────────────────────────────
# Several stages call sy.add_session_args(ap) or fiber_stochastic.add_arguments(ap),
# which add flags from ANOTHER module.  Reading only the stage's own file loses
# them -- fiber-session's positionals and --channels/--ntotal/--nsamp all arrive
# this way.  So the helpers are parsed once and spliced in where they are called,
# honouring the keyword switches that suppress individual flags
# (add_session_args(ap, channels=False, ...)).
HELPERS = {
    "add_session_args": ("session_yaml.py", "add_session_args"),
    "add_arguments":    ("fiber_stochastic.py", "add_arguments"),
}


def helper_rows(src_dir, fname, consts_cache={}):
    """(positionals, rows, gate_of_row) for a shared flag-adding helper.

    gate_of_row[i] is the keyword that suppresses row i when passed False, taken
    from the `if <kw>:` the add_argument sits under, so a caller's
    channels=False is reflected.
    """
    mod, fn = HELPERS[fname]
    path = os.path.join(src_dir, mod)
    if not os.path.exists(path):
        return [], [], []
    key = (path, fn)
    if key in consts_cache:
        return consts_cache[key]
    tree = ast.parse(open(path, errors="ignore").read())
    consts = module_constants(tree)
    target = next((n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == fn), None)
    if target is None:
        consts_cache[key] = ([], [], [])
        return consts_cache[key]

    positionals, rows, gates = [], [], []

    def walk(node, gate):
        for ch in ast.iter_child_nodes(node):
            g = gate
            if isinstance(ch, ast.If) and isinstance(ch.test, ast.Name):
                g = ch.test.id
            if isinstance(ch, ast.Call) and getattr(ch.func, "attr", "") == "add_argument":
                names = [a.value for a in ch.args
                         if isinstance(a, ast.Constant) and isinstance(a.value, str)]
                if names:
                    row = build_row(ch, names, consts)
                    if row is None:
                        positionals.append(names[0])
                        gates.append(gate)
                    else:
                        rows.append(row)
                        gates.append(gate)
            walk(ch, g)
    walk(target, None)
    consts_cache[key] = (positionals, rows, gates)
    return consts_cache[key]


def build_row(node, names, consts):
    """One table row, or None when `names[0]` is a positional."""
    if not names[0].startswith("-"):
        return None
    kw = {k.arg: k.value for k in node.keywords}
    action = kw["action"].value if isinstance(kw.get("action"), ast.Constant) else None
    try:
        help_ = ast.literal_eval(kw["help"]) if "help" in kw else ""
    except Exception:
        help_ = ""
    choices = None
    if "choices" in kw and isinstance(kw["choices"], (ast.List, ast.Tuple)):
        try:
            choices = [e.value for e in kw["choices"].elts]
        except AttributeError:
            choices = None
    # argparse.BooleanOptionalAction silently creates a --no-<flag> twin; the doc
    # has always listed both, so render both or every one of them looks deleted.
    boolopt = (isinstance(kw.get("action"), ast.Attribute)
               and kw["action"].attr == "BooleanOptionalAction")
    if boolopt:
        base = names[0][2:]
        flag = f"`--{base}` / `--no-{base}`"
        action = "store_true"   # rendered as flag (on/off) from the default below
    else:
        flag = f"`{names[0]}`"
        if len(names) > 1:
            flag += " (" + ", ".join(f"`{a}`" for a in names[1:]) + ")"
    text = " ".join(str(help_ or "").split())
    if choices:
        text += (" \u2014 " if text else "") + "choices: " + ", ".join(f"`{c}`" for c in choices)
    if boolopt:
        d = kw.get("default")
        on = isinstance(d, ast.Constant) and d.value is True
        return (flag, "flag (on)" if on else "flag (off)", text)
    return (flag, render_default(kw.get("default"), consts, action), text)


def parse_stage(path):
    """(prog, description, positionals, rows) or None if the file is not a stage."""
    tree = ast.parse(open(path, errors="ignore").read())
    consts = module_constants(tree)
    prog = desc = None
    positionals, rows = [], []
    dynamic = False

    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "ArgumentParser":
            for kw in n.keywords:
                if kw.arg == "prog" and isinstance(kw.value, ast.Constant):
                    prog = kw.value.value
                if kw.arg == "description":
                    try:
                        desc = ast.literal_eval(kw.value)
                    except Exception:
                        desc = None

    for n in ast.walk(tree):
        if not (isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add_argument"):
            continue
        names = [a.value for a in n.args
                 if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        if not names:
            # A computed flag name -- e.g. fiber_backbone_link builds one per entry
            # of a module-level _KNOBS dict.  The name cannot be recovered without
            # evaluating the module, so this stage's table is left ALONE rather
            # than rewritten with the rows silently missing: a --check that passed
            # while thirteen flags were absent would be worse than no check.
            dynamic = True
            continue
        row = build_row(n, names, consts)
        if row is None:
            positionals.append(names[0])
        else:
            rows.append(row)

    # Splice in flags added by a shared helper, honouring switches like
    # add_session_args(ap, channels=False) that suppress individual ones.
    for n in ast.walk(tree):
        fname = getattr(n.func, "attr", getattr(n.func, "id", "")) if isinstance(n, ast.Call) else ""
        # <Class>.add_arguments(ap) -- config.py builds one flag per dataclass field,
        # so the names exist only at runtime.  Same rule as a computed flag name:
        # leave the stage's table alone rather than rewrite it short.
        if fname == "add_arguments" and isinstance(getattr(n, "func", None), ast.Attribute) \
                and isinstance(n.func.value, ast.Name) and n.func.value.id[:1].isupper():
            dynamic = True
            continue
        if fname not in HELPERS:
            continue
        off = {k.arg for k in n.keywords
               if isinstance(k.value, ast.Constant) and k.value is not None
               and k.value.value is False}
        hpos, hrows, hgates = helper_rows(os.path.dirname(path), fname)
        gi = 0
        for pn in hpos:
            if hgates[gi] not in off:
                positionals.append(pn)
            gi += 1
        for r in hrows:
            if hgates[gi] not in off:
                rows.append(r)
            gi += 1

    if prog is None and desc is None and not rows:
        return None
    if prog is None:
        prog = os.path.basename(path)[:-3].replace("_", "-")
    return prog, desc, positionals, rows, dynamic


def render_block(stage):
    _prog, _desc, positionals, rows, _dyn = stage
    out = []
    if positionals:
        out.append("Positional: " + ", ".join(f"`{p}`" for p in positionals))
        out.append("")
    out.append("| flag | default | description |")
    out.append("|---|---|---|")
    for flag, default, text in rows:
        out.append(f"| {flag} | {default} | {text.replace('|', chr(92) + '|')} |")
    return "\n".join(out)


# ── rewriting the doc, in place, per heading ─────────────────────────────────
HEAD = re.compile(r"^### `([a-z0-9-]+)`\s*$", re.M)


def rebuild(doc_text, stages):
    """Replace each stage block's Positional line + table; keep everything else."""
    heads = [(m.group(1), m.start(), m.end()) for m in HEAD.finditer(doc_text)]
    if not heads:
        return doc_text, [], []
    out, used, missing_in_code, dynamic_skipped = [], set(), [], []
    prev_end = 0
    for i, (name, hs, he) in enumerate(heads):
        body_end = heads[i + 1][1] if i + 1 < len(heads) else len(doc_text)
        out.append(doc_text[prev_end:he])
        body = doc_text[he:body_end]
        st = stages.get(name)
        if st is None or st[4]:
            (missing_in_code if st is None else dynamic_skipped).append(name)
            out.append(body)
            prev_end = body_end
            if st is not None:
                used.add(name)
            continue
        used.add(name)
        # keep prose before the Positional line / table, and any trailing
        # "## " section heading that belongs to the NEXT section
        m = re.search(r"^(Positional:.*?$|\| flag \| default \| description \|)", body, re.M)
        prose = body[:m.start()] if m else body
        tail = ""
        t = re.search(r"\n(## .*)$", body, re.S)
        if t:
            tail = "\n" + t.group(1)
        if not prose.endswith("\n\n"):
            prose = prose.rstrip("\n") + "\n\n"
        out.append(prose + render_block(st) + "\n" + tail)
        prev_end = body_end
    out.append(doc_text[prev_end:])
    return "".join(out), sorted(set(stages) - used), missing_in_code, dynamic_skipped


def main():
    ap = argparse.ArgumentParser(prog="gen_stages_doc")
    ap.add_argument("--check", action="store_true",
                    help="do not write; exit 1 if docs/stages.md has drifted from the parsers")
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--doc", default=DOC)
    a = ap.parse_args()

    # Key every stage by its INSTALLED command name.  A module reachable only as
    # a library (session_yaml, klustakwik) is deliberately absent from the doc and
    # must not be reported as undocumented.
    cmds = console_scripts()
    stages, progmismatch = {}, []
    for cmd, mod in sorted(cmds.items()):
        p = os.path.join(a.src, mod + ".py")
        if not os.path.exists(p):
            continue
        st = parse_stage(p)
        if not st:
            continue
        if st[0] != cmd:
            progmismatch.append((cmd, st[0]))
        stages[cmd] = st

    doc = open(a.doc, errors="ignore").read()
    new, undocumented, unknown, dynamic = rebuild(doc, stages)

    for n in undocumented:
        print(f"  note: stage '{n}' has a parser but no `### `{n}`` heading in {a.doc} "
              f"(placement is editorial \u2014 add the heading where it belongs)")
    for n in unknown:
        print(f"  note: heading '{n}' in {a.doc} has no matching console script")
    for n in dynamic:
        print(f"  note: '{n}' builds flag names at runtime; its table is left as written "
              f"and is NOT checked")
    for cmd, prog in progmismatch:
        print(f"  note: {cmd} sets prog='{prog}' \u2014 its own --help prints the wrong "
              f"command name")

    if a.check:
        if new != doc:
            print(f"DRIFT: {a.doc} does not match the argument parsers. "
                  f"Run: python3 tools/gen_stages_doc.py")
            return 1
        print(f"OK: {a.doc} matches the argument parsers "
              f"({len(stages)} stages, {len(unknown)} unmatched heading(s))")
        return 0

    if new != doc:
        open(a.doc, "w").write(new)
        print(f"rewrote {a.doc} from {len(stages)} parsers")
    else:
        print(f"{a.doc} already matches the parsers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
