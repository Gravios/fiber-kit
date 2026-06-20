#  fiber_pipeline_editor.py — Qt6 (>= 6.8) node-graph editor for fiber-kit pipeline plans.
#
#  Edits the scripts/fiber-pipeline `--plan` YAML (an ordered list of steps; stages may repeat and
#  be reordered).  Layout:
#     LEFT    palette of pipeline stages (double-click to add a node)
#     CENTRE  node canvas: each step is a node, connectors wire one step's OUT tag to the next's IN
#     RIGHT   inspector: edit the selected node's structural tags (in/out/units/cpos/spk) and its
#             tunable params (only the ones you tick are written to the plan; the rest stay default)
#  Save writes the plan YAML (in dependency / topological order).
#
#  The model core below (catalog, PlanStep, load/dump/topo/validate/auto_layout) is Qt-free and
#  unit-testable; the PySide6 view is only built when Qt is present.  Run: `fiber-plan-edit [plan.yaml]`.

import sys
import os
from dataclasses import dataclass, field

try:
    import yaml
except ImportError:                                              # pragma: no cover
    yaml = None

TAG_FIELDS = ("in", "out", "units", "cpos", "spk")


# ───────────────────────────── stage catalog (UI metadata) ─────────────────────────────
def _intrachunk_params():
    """Pull the intrachunk knobs straight from the typed IntrachunkConfig so the editor never drifts
    from the code; fall back to a static list if config.py is unavailable."""
    try:
        from dataclasses import fields as _dcf
        try:
            from .config import IntrachunkConfig
        except ImportError:
            from config import IntrachunkConfig
        out = []
        for f in _dcf(IntrachunkConfig):
            m = f.metadata
            flag = m.get("cli") or f.name.replace("_", "-")
            choices = list(m.get("choices") or [])
            typ = "choice" if choices else {float: "float", int: "int", str: "str"}.get(m.get("type", float), "float")
            out.append(dict(flag=flag, type=typ, default=("" if f.default is None else f.default),
                            help=m.get("help", ""), choices=choices))
        return out
    except Exception:
        return [dict(flag="gate", type="choice", default="cfiber", help="shape gate",
                     choices=["cosine", "mmd", "kcov", "cfiber"]),
                dict(flag="cos-thr", type="float", default=0.85, help="cosine recall prefilter", choices=[]),
                dict(flag="amp-gate", type="float", default=1.10, help="log-amplitude gate (nat-log; 0=off)", choices=[])]


def _p(flag, typ, default, help, choices=()):
    return dict(flag=flag, type=typ, default=default, help=help, choices=list(choices))


CATALOG = {
    "fiber-session": dict(input=False, tags=["out"], params=[
        _p("fine-method", "choice", "rkk", "per-fiber fine clusterer", ["rkk", "kk"]),
        _p("chunk-min", "int", 12, "chunk length (min); drift << site pitch per chunk"),
        _p("merge-method", "choice", "sliding", "coarse merge method", ["sliding"]),
        _p("merge-corr", "float", 0.90, "coarse-fiber template-correlation merge (0.88-0.93)"),
        _p("cfiber-q", "float", 0.90, "within-fiber cfiber shape-veto quantile (0.85-0.95)"),
        _p("feature-align", "choice", "xcorr", "sub-sample align before featurize", ["xcorr", "centroid", "off"]),
        _p("inclusion-k", "float", 2.5, "core radius = median + k*MAD (2.0-3.0; lower=purer)"),
        _p("dip-dim", "int", 6, "dip-test PCA dims (4-8)"),
        _p("dip-alpha", "float", 0.02, "dip-test p to bisect (0.01-0.03; >0.05 splits noise)"),
        _p("dip-min", "int", 30, "min spikes to bisect (30-60)")]),
    "fiber-realign": dict(input=True, tags=["in"], params=[]),   # in-place; structural only
    "fiber-refine": dict(input=True, tags=["in", "out"], params=[
        _p("large", "int", 150, "split only clusters >= this many spikes (100-300)"),
        _p("min-group", "int", 30, "min spikes per split piece (25-50)"),
        _p("merge-min-sim", "float", 0.96, "merge-back similarity (0.92-0.97; higher keeps over-splits)"),
        _p("split-var-mult", "float", 1.5, "split clusters w/ top-3 feat-var > x*median (1.3-2.0)"),
        _p("split-min-corr", "float", 0.93, "min split-piece internal corr to keep (0.90-0.95)"),
        _p("chunk-minutes", "int", 12, "refine chunk length (min)"),
        _p("fold-off-thr", "float", 0.22, "inter-channel-offset fold veto, samples (0.20-0.25)"),
        _p("dedup-stale", "choice", "quarantine", "stale per-spike file handling", ["quarantine", "error", "skip"]),
        _p("merge-warp-recall", "float", 0.9, "warp-recall group-delay floor (0.85-0.95; empty=off)"),
        _p("merge-amp-thr", "float", 0.7, "warp-recall amplitude-profile floor (0.6-0.8)"),
        _p("merge-warp-thr", "str", "", "warp precision gate on cosine merges (empty=off)")]),
    "fiber-cpos": dict(input=True, tags=["in", "out", "spk"], params=[]),   # localizes from raw .spk; structural
    "fiber-intrachunk": dict(input=True, tags=["in", "out", "cpos"], params=_intrachunk_params()),
    "fiber-link": dict(input=True, tags=["in", "out", "units", "cpos"], params=[
        _p("cos-thr", "float", 0.75, "cosine prefilter (0.70-0.85; recall)"),
        _p("pos-thr", "float", 1.5, "position gate (1.0-2.0)"),
        _p("off-thr", "float", 1.0, "inter-channel offset gate, samples (0.8-1.2)"),
        _p("max-gap", "int", 2, "max chunk gap to bridge (1-4)"),
        _p("amp-gate", "float", 1.39, "log-amplitude gate, nat-log; ln4=1.39 -> 4x (0=off)"),
        _p("cfiber-q", "float", 0.90, "cfiber co-gate quantile (0.85-0.95; empty=off)"),
        _p("warp-thr", "str", "", "warp co-gate (empty=off)")]),
}
STAGES = list(CATALOG)


# ── derive each stage's fields from its module's argparse, so the inspector tracks the modules ──
STAGE_MODULES = {
    "fiber-session": "fiber_kit.fiber_session",
    "fiber-realign": "fiber_kit.fiber_realign",
    "fiber-refine": "fiber_kit.fiber_refine",
    "fiber-cpos": "fiber_kit.fiber_cpos",
    "fiber-intrachunk": "fiber_kit.fiber_intrachunk",
    "fiber-link": "fiber_kit.fiber_link",
}

# flags the PLAN expresses structurally (node tags / edges) or that come from the session, not per-node tuning
_STRUCTURAL_FLAGS = {
    "channels", "ntotal", "nchan", "nsamp", "sr", "peak", "probe", "base", "elec", "session",
    "dir", "yaml", "verbose", "quiet", "help", "out-clu", "out-res", "out-variant",
    "in", "in-clu", "out", "clu", "cpos", "spk", "units", "fil", "fil-offset",
}


def _is_structural_flag(flag):
    return flag in _STRUCTURAL_FLAGS or flag.endswith("-method") or flag.endswith("-stage")


class _CaptureParser(Exception):
    pass


def _capture_parser(modname):
    """Import a stage module and capture its argparse parser WITHOUT running the stage, by intercepting
    parse_args/parse_known_args -- which every main() calls right after building the parser."""
    import argparse as _ap
    import importlib
    real_pa, real_pk = _ap.ArgumentParser.parse_args, _ap.ArgumentParser.parse_known_args
    box = {}

    def grab(self, *a, **k):
        box["p"] = self
        raise _CaptureParser()

    _ap.ArgumentParser.parse_args = grab
    _ap.ArgumentParser.parse_known_args = grab
    try:
        importlib.import_module(modname).main()
    except _CaptureParser:
        pass
    except SystemExit:
        pass
    finally:
        _ap.ArgumentParser.parse_args = real_pa
        _ap.ArgumentParser.parse_known_args = real_pk
    return box.get("p")


def introspect_stage(stage):
    """Tuning flags a stage's module actually exposes (the source of truth), or None if it can't be imported.
    Valued optionals only; structural plumbing and valueless toggles are dropped.  Each entry carries every
    alias so the curated overlay can match regardless of which spelling it used."""
    import argparse as _ap
    modname = STAGE_MODULES.get(stage)
    if not modname:
        return None
    try:
        parser = _capture_parser(modname)
    except Exception:
        return None
    if parser is None:
        return None
    out = []
    for a in parser._actions:
        if not a.option_strings or a.dest == "help" or a.nargs == 0:
            continue
        aliases = [s.lstrip("-") for s in a.option_strings]
        flag = max(aliases, key=len)
        if _is_structural_flag(flag):
            continue
        typ = "choice" if a.choices else {int: "int", float: "float"}.get(a.type, "str")
        default = a.default
        if default is None or default == _ap.SUPPRESS:
            default = ""
        out.append(dict(flag=flag, aliases=aliases, type=typ, default=default,
                        help=(a.help or "").strip(), choices=list(a.choices) if a.choices else []))
    return out


_STAGE_PARAMS_CACHE = {}


def stage_accepted_flags(stage):
    """Every option flag the stage's module accepts (canonical + aliases, structural included), or None if
    the module can't be imported.  The linter uses this to catch param flags the stage would reject at
    runtime; introspect_stage's tuning-only filter is too narrow for that."""
    modname = STAGE_MODULES.get(stage)
    if not modname:
        return None
    try:
        parser = _capture_parser(modname)
    except Exception:
        return None
    if parser is None:
        return None
    flags = set()
    for a in parser._actions:
        for s in a.option_strings:
            flags.add(s.lstrip("-"))
    return flags


def stage_params(stage):
    """The inspector's field list for a stage, derived live from its module so flags that are added, renamed
    or removed track automatically.  Module flags that still match a curated entry keep the curated help and
    (when the module default is suppressed, as intrachunk's are) the curated default, and are marked
    'primary' to show first; the rest are 'advanced'.  Falls back to the static curated catalog when the
    module can't be introspected (e.g. a minimal editor-only environment)."""
    if stage in _STAGE_PARAMS_CACHE:
        return _STAGE_PARAMS_CACHE[stage]
    derived = introspect_stage(stage)
    if derived is None:
        result = [dict(p, primary=True) for p in CATALOG[stage]["params"]]
    else:
        curated = {p["flag"]: p for p in CATALOG[stage]["params"]}
        primary, advanced = [], []
        for d in derived:
            cm = next((curated[al] for al in d["aliases"] if al in curated), None)
            entry = dict(flag=d["flag"], type=d["type"], default=d["default"],
                         help=d["help"], choices=d["choices"])
            if cm is not None:
                if (entry["default"] == "" or entry["default"] is None) and cm.get("default") not in (None, ""):
                    entry["default"] = cm["default"]
                if cm.get("help"):
                    entry["help"] = cm["help"]
                entry["primary"] = True
                primary.append(entry)
            else:
                entry["primary"] = False
                advanced.append(entry)
        result = primary + advanced
    _STAGE_PARAMS_CACHE[stage] = result
    return result


# ───────────────────────────── plan model (Qt-free) ─────────────────────────────
@dataclass
class PlanStep:
    stage: str
    tags: dict = field(default_factory=dict)        # subset of TAG_FIELDS -> str
    params: dict = field(default_factory=dict)      # flag -> value
    args: list = field(default_factory=list)        # raw extra flags, passed through verbatim
    x: float = 0.0
    y: float = 0.0

    def tag(self, k):
        return str(self.tags.get(k, ""))


def load_plan(path):
    """Parse a --plan YAML into a list of PlanStep."""
    if yaml is None:
        raise RuntimeError("PyYAML required to read plans (pip install pyyaml)")
    with open(path) as fh:
        doc = yaml.safe_load(fh) or {}
    steps_raw = doc.get("pipeline") if isinstance(doc, dict) else doc
    if not isinstance(steps_raw, list):
        raise ValueError("plan has no 'pipeline:' list of steps")
    steps = []
    for i, st in enumerate(steps_raw):
        if not isinstance(st, dict) or st.get("stage") not in CATALOG:
            raise ValueError("plan step %d: missing or unknown 'stage' (%r)" % (i, st))
        tags = {k: str(st[k]) for k in TAG_FIELDS if k in st and st[k] is not None}
        params = dict(st.get("params") or {})
        args = list(st.get("args") or [])
        pos = st.get("pos") or [0.0, 0.0]
        try:
            x, y = float(pos[0]), float(pos[1])
        except (TypeError, ValueError, IndexError):
            x, y = 0.0, 0.0
        steps.append(PlanStep(stage=st["stage"], tags=tags, params=params, args=args, x=x, y=y))
    return steps


def derive_edges(steps):
    """Edges implied by the tag wiring: an edge (src, dst, kind) exists when dst's in/units/cpos tag
    equals the OUT tag of an earlier step.  '' (base over-cluster) and the session source are not edges."""
    edges = []
    for j, dst in enumerate(steps):
        for kind in ("in", "units", "cpos"):
            t = dst.tag(kind)
            if not t:
                continue
            src = None                                   # latest earlier producer of this tag
            for i in range(j - 1, -1, -1):
                if steps[i].tag("out") == t:
                    src = i
                    break
            if src is not None:
                edges.append((src, j, kind))
    return edges


def topo_order(steps):
    """Topological order honouring the tag dependencies, stable on the current list order (so a load
    -> save round-trip of an already-valid plan is a no-op).  Raises on a dependency cycle."""
    n = len(steps)
    deps = {i: set() for i in range(n)}
    for src, dst, _ in derive_edges(steps):
        deps[dst].add(src)
    order, done = [], set()
    progressed = True
    while len(done) < n and progressed:
        progressed = False
        for i in range(n):                               # stable: lowest current index first
            if i in done:
                continue
            if deps[i] <= done:
                order.append(steps[i]); done.add(i); progressed = True
    if len(done) < n:
        raise ValueError("plan has a dependency cycle among: %s"
                         % ", ".join(steps[i].stage for i in range(n) if i not in done))
    return order


def validate(steps):
    """Return a list of human-readable warnings (unproduced inputs, duplicate out tags, etc.)."""
    warns = []
    produced = []
    seen_out = {}
    for i, s in enumerate(steps):
        for kind in ("in", "units", "cpos"):
            t = s.tag(kind)
            if t and t not in produced:
                warns.append("step %d (%s): '%s'='%s' is not produced by any earlier step"
                             % (i + 1, s.stage, kind, t))
        out = s.tag("out")
        if "out" in CATALOG[s.stage]["tags"]:
            inplace = out == s.tag("in")                 # cpos augments the same clu stage (out==in); not a fresh producer
            if out in seen_out and not inplace:
                warns.append("step %d (%s): out tag '%s' already produced by step %d (later steps "
                             "will read the earlier producer)" % (i + 1, s.stage, out, seen_out[out] + 1))
            seen_out[out] = i
            if out not in produced:
                produced.append(out)
    try:
        topo_order(steps)
    except ValueError as e:
        warns.append(str(e))
    return warns


def lint(steps):
    """Static checks on a plan, beyond validate(): returns (errors, warnings).  Errors would break a run
    (cycle, a param flag the stage rejects, an uncoercible numeric value); warnings are advisory (an input
    no earlier step produces -- it may exist on disk -- duplicate out tags, modules not importable so flags
    could not be checked)."""
    errors, warnings = [], []
    # structural: reuse validate(), but a dependency cycle is a hard error
    for w in validate(steps):
        (errors if "dependency cycle" in w else warnings).append(w)
    # per-step flag and value checks against the real module CLIs
    for i, s in enumerate(steps):
        accepted = stage_accepted_flags(s.stage)
        if accepted is None:
            if s.params:
                warnings.append("step %d (%s): module not importable; param flags not checked" % (i + 1, s.stage))
            continue
        types = {}
        derived = introspect_stage(s.stage) or []
        for d in derived:
            for al in d["aliases"]:
                types[al] = d["type"]
        for flag, val in s.params.items():
            if flag not in accepted:
                errors.append("step %d (%s): unknown flag --%s (the stage would reject it)" % (i + 1, s.stage, flag))
                continue
            typ = types.get(flag)
            if typ in ("int", "float") and str(val).strip() != "":
                try:
                    float(val)
                except (TypeError, ValueError):
                    errors.append("step %d (%s): --%s expects %s but value is %r"
                                  % (i + 1, s.stage, flag, typ, val))
    return errors, warnings


def lint_main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Statically check fiber-kit pipeline plans (stage names, tag wiring, cycles, and -- "
                    "against the real stage modules -- unknown or mistyped param flags).  Exit 1 on errors; "
                    "with --strict, also on warnings.  Accepts plan files or a fiber-kit.yaml with a "
                    "'pipeline:' section.")
    ap.add_argument("plans", nargs="+", help="plan YAML file(s) to lint")
    ap.add_argument("--strict", action="store_true", help="treat warnings as failures (CI gate)")
    ap.add_argument("-q", "--quiet", action="store_true", help="print only failures")
    a = ap.parse_args()
    import sys
    rc = 0
    for path in a.plans:
        try:
            steps = load_plan(path)
        except Exception as e:                                   # noqa: BLE001 - report any load failure as an error
            print("%s: ERROR could not load plan: %s" % (path, e))
            rc = 1
            continue
        errors, warnings = lint(steps)
        if errors or warnings or not a.quiet:
            status = "FAIL" if errors else ("WARN" if warnings else "ok")
            print("%s: %s (%d steps)" % (path, status, len(steps)))
        for e in errors:
            print("  ERROR %s" % e)
        for w in warnings:
            print("  warn  %s" % w)
        if errors or (a.strict and warnings):
            rc = 1
    sys.exit(rc)


def dump_plan(steps, ordered=True, header=True):
    """Serialise to plan YAML text (in topological order by default)."""
    if yaml is None:
        raise RuntimeError("PyYAML required to write plans (pip install pyyaml)")
    seq = topo_order(steps) if ordered else list(steps)
    has_layout = any(s.x or s.y for s in steps)              # only record positions once the editor has laid out
    rows = []
    for s in seq:
        d = {"stage": s.stage}
        for k in CATALOG[s.stage]["tags"]:               # in/out always (core wiring); units/cpos/spk only if set
            v = s.tags.get(k, "")
            if k in ("in", "out") or v != "":
                d[k] = v
        if s.params:
            d["params"] = dict(s.params)
        if s.args:
            d["args"] = list(s.args)
        if has_layout:
            d["pos"] = [int(round(s.x)), int(round(s.y))]    # editor layout; ignored by the runner
        rows.append(d)
    body = yaml.safe_dump({"pipeline": rows}, sort_keys=False, default_flow_style=False, width=120)
    if not header:
        return body
    head = ("# fiber-kit pipeline plan (edited by fiber-plan-edit).  Steps run in this order; a stage may\n"
            "# repeat with different in/out tags and params.  Run:  fiber-pipeline <elec> --plan thisfile.yaml\n")
    return head + body


PRIMARY_MARKER = "# ---- primary pipeline plan (run by `fiber-pipeline <elec>` when no --plan is given) ----"


def find_pipeline_exe():
    """Locate the fiber-pipeline runner: prefer the installed console script, else the scripts/ copy next
    to this package (so the editor previews correctly from a source checkout too)."""
    import os
    import shutil
    exe = shutil.which("fiber-pipeline")
    if exe:
        return [exe]
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.normpath(os.path.join(here, "..", "..", "scripts", "fiber-pipeline"))
    if os.path.isfile(cand):
        return ["bash", cand]
    return None


def render_dry_run(steps, elec="5", exe=None, timeout=20):
    """Run the REAL fiber-pipeline in --dry-run on the current plan and return (text, ok).  This shells out
    to the runner itself -- the preview is exactly what would execute, never a reimplementation that could
    drift.  Dry-run touches no session files; it runs in a scratch dir with a placeholder session name."""
    import os
    import subprocess
    import tempfile
    cmd = exe or find_pipeline_exe()
    if not cmd:
        return ("fiber-pipeline not found on PATH.\nInstall fiber-kit (pip install -e .) or run the editor "
                "from a source checkout so scripts/fiber-pipeline is reachable."), False
    if topo_order is not None:
        try:
            topo_order(steps)
        except ValueError as e:
            return "plan is not runnable:\n  %s" % e, False
    tmpd = tempfile.mkdtemp(prefix="fk-preview-")
    planfile = os.path.join(tmpd, "plan.yaml")
    try:
        with open(planfile, "w") as fh:
            fh.write(dump_plan(steps))
        env = dict(os.environ, FK_DIR=tmpd, FK_SESS="SESSION")
        p = subprocess.run(cmd + [str(elec), "--plan", planfile, "--dry-run"],
                           cwd=tmpd, env=env, capture_output=True, text=True, timeout=timeout)
        out = p.stdout or ""
        if p.stderr:
            out += ("\n" if out else "") + p.stderr
        return out.rstrip() or "(no output)", p.returncode == 0
    except subprocess.TimeoutExpired:
        return "preview timed out after %ds" % timeout, False
    except Exception as e:                                       # noqa: BLE001 - surface any failure in the pane
        return "preview failed: %s" % e, False
    finally:
        for path in (planfile, tmpd):
            try:
                os.remove(path) if os.path.isfile(path) else os.rmdir(path)
            except OSError:
                pass


def _strip_pipeline_block(text):
    """Remove an existing top-level `pipeline:` block (and our marker comment) from fiber-kit.yaml text,
    leaving every FK_* line and comment untouched.  A block = the `pipeline:` line plus the following
    list items ('- ...'), their indented continuations, and blank lines."""
    lines = text.splitlines(keepends=True)
    out, i, n = [], 0, len(lines)
    while i < n:
        ln = lines[i]
        if ln.startswith("pipeline:"):
            i += 1
            while i < n and (lines[i].strip() == "" or lines[i][:1] in (" ", "\t", "-")):
                i += 1
            continue
        if ln.rstrip("\n") == PRIMARY_MARKER:
            i += 1
            continue
        out.append(ln)
        i += 1
    return "".join(out)


def embed_plan_in_config(cfg_text, steps):
    """Return fiber-kit.yaml text with the plan written in as the primary `pipeline:` section, replacing
    any prior one and preserving the existing FK_* knobs and comments."""
    body = dump_plan(steps, header=False)            # 'pipeline:\n- ...'
    base = _strip_pipeline_block(cfg_text or "").rstrip("\n")
    if base:
        return base + "\n\n" + PRIMARY_MARKER + "\n" + body
    return PRIMARY_MARKER + "\n" + body


def auto_layout(steps, dx=220.0, dy=130.0):
    """Layered left-to-right layout: column = longest-path depth in the dependency DAG; rows stack
    nodes sharing a column.  Writes x/y onto each step and returns them."""
    n = len(steps)
    deps = {i: set() for i in range(n)}
    for src, dst, _ in derive_edges(steps):
        deps[dst].add(src)
    depth = [0] * n
    for _ in range(n):                                   # relax longest paths
        for i in range(n):
            depth[i] = max([0] + [depth[d] + 1 for d in deps[i]])
    rows_in_col = {}
    for i in range(n):
        c = depth[i]
        r = rows_in_col.get(c, 0)
        rows_in_col[c] = r + 1
        steps[i].x = c * dx
        steps[i].y = r * dy
    return steps


def new_step(stage, x=0.0, y=0.0):
    tags = {k: "" for k in CATALOG[stage]["tags"]}
    return PlanStep(stage=stage, tags=tags, x=x, y=y)


def duplicate_step(step):
    """A copy of a step (same params/tags), nudged in position and given a fresh out tag so it is a
    distinct producer -- the building block for running a stage twice with different values."""
    new = PlanStep(stage=step.stage, tags=dict(step.tags), params=dict(step.params),
                   args=list(step.args), x=step.x + 36.0, y=step.y + 36.0)
    if new.tags.get("out"):
        new.tags["out"] = new.tags["out"] + "_copy"
    return new


# ───────────────────────────── Qt view (PySide6 >= 6.8) ─────────────────────────────
def _require_qt():
    try:
        from PySide6 import QtCore
    except ImportError:
        sys.exit("fiber-plan-edit needs PySide6 >= 6.8  —  pip install 'PySide6>=6.8'")
    v = tuple(int(p) for p in QtCore.qVersion().split(".")[:2])
    if v < (6, 8):
        sys.exit("fiber-plan-edit needs Qt >= 6.8 (found %s).  Upgrade PySide6." % QtCore.qVersion())
    return v


def _build_qt():
    from PySide6 import QtCore, QtGui, QtWidgets
    Qt = QtCore.Qt

    NODE_W, NODE_H, PORT_R = 168.0, 76.0, 6.0

    class EdgeItem(QtWidgets.QGraphicsPathItem):
        def __init__(self, src_node, dst_node, kind):
            super().__init__()
            self.src, self.dst, self.kind = src_node, dst_node, kind
            self.setZValue(-1)
            col = {"in": "#6fb1ff", "units": "#ffb86f", "cpos": "#b6f06f"}.get(kind, "#888")
            self.setPen(QtGui.QPen(QtGui.QColor(col), 2.0))
            self.update_path()

        def update_path(self):
            a = self.src.out_port_scene_pos()
            b = self.dst.in_port_scene_pos()
            path = QtGui.QPainterPath(a)
            mx = (a.x() + b.x()) * 0.5
            path.cubicTo(QtCore.QPointF(mx, a.y()), QtCore.QPointF(mx, b.y()), b)
            self.setPath(path)

        def shape(self):                                 # widen the clickable area for right-click removal
            stroker = QtGui.QPainterPathStroker()
            stroker.setWidth(10)
            return stroker.createStroke(self.path())

        def contextMenuEvent(self, event):
            menu = QtWidgets.QMenu()
            act = menu.addAction("Remove connection (%s)" % self.kind)
            if menu.exec(event.screenPos()) is act:
                self.dst.editor.remove_edge(self)
            event.accept()

    class NodeItem(QtWidgets.QGraphicsObject):
        moved = QtCore.Signal()

        def __init__(self, step, editor):
            super().__init__()
            self.step, self.editor = step, editor
            self.setFlags(self.GraphicsItemFlag.ItemIsMovable | self.GraphicsItemFlag.ItemIsSelectable
                          | self.GraphicsItemFlag.ItemSendsGeometryChanges)
            self.setPos(step.x, step.y)
            self.setZValue(1)

        def boundingRect(self):
            return QtCore.QRectF(0, 0, NODE_W, NODE_H)

        def paint(self, p, opt, w=None):
            r = self.boundingRect()
            sel = self.isSelected()
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            p.setPen(QtGui.QPen(QtGui.QColor("#e6c14b" if sel else "#3a3a3a"), 2.0))
            p.setBrush(QtGui.QColor("#2b2f36"))
            p.drawRoundedRect(r, 8, 8)
            p.setBrush(QtGui.QColor("#3a4150"))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(QtCore.QRectF(0, 0, NODE_W, 24), 8, 8)
            p.setPen(QtGui.QColor("#f0f0f0"))
            f = p.font(); f.setBold(True); p.setFont(f)
            p.drawText(QtCore.QRectF(10, 2, NODE_W - 20, 22), Qt.AlignmentFlag.AlignVCenter, self.step.stage)
            f.setBold(False); p.setFont(f)
            p.setPen(QtGui.QColor("#a9c7ff"))
            info = []
            for k in ("in", "out", "units"):
                if k in CATALOG[self.step.stage]["tags"]:
                    info.append("%s:%s" % (k, self.step.tag(k) or "''"))
            p.drawText(QtCore.QRectF(10, 30, NODE_W - 20, 18), Qt.AlignmentFlag.AlignVCenter, "  ".join(info))
            np = len(self.step.params)
            if np:
                p.setPen(QtGui.QColor("#9fe0a0"))
                p.drawText(QtCore.QRectF(10, 50, NODE_W - 20, 18), Qt.AlignmentFlag.AlignVCenter,
                           "%d param%s set" % (np, "" if np == 1 else "s"))
            # ports
            p.setPen(Qt.PenStyle.NoPen)
            if CATALOG[self.step.stage]["input"]:
                p.setBrush(QtGui.QColor("#6fb1ff"))
                p.drawEllipse(self._in_port_rect())
            p.setBrush(QtGui.QColor("#9fe0a0"))
            p.drawEllipse(self._out_port_rect())

        def _in_port_rect(self):
            return QtCore.QRectF(-PORT_R, NODE_H / 2 - PORT_R, 2 * PORT_R, 2 * PORT_R)

        def _out_port_rect(self):
            return QtCore.QRectF(NODE_W - PORT_R, NODE_H / 2 - PORT_R, 2 * PORT_R, 2 * PORT_R)

        def in_port_scene_pos(self):
            return self.mapToScene(QtCore.QPointF(0, NODE_H / 2))

        def out_port_scene_pos(self):
            return self.mapToScene(QtCore.QPointF(NODE_W, NODE_H / 2))

        def hit_out_port(self, scene_pos):
            return self._out_port_rect().adjusted(-4, -4, 4, 4).contains(self.mapFromScene(scene_pos))

        def hit_in_port(self, scene_pos):
            return (CATALOG[self.step.stage]["input"]
                    and self._in_port_rect().adjusted(-4, -4, 4, 4).contains(self.mapFromScene(scene_pos)))

        def itemChange(self, change, value):
            if change == self.GraphicsItemChange.ItemPositionHasChanged:
                self.step.x, self.step.y = self.pos().x(), self.pos().y()
                self.editor.refresh_edges()
            elif change == self.GraphicsItemChange.ItemSelectedHasChanged and value:
                self.editor.select_node(self)
            return super().itemChange(change, value)

        def contextMenuEvent(self, event):
            menu = QtWidgets.QMenu()
            a_dup = menu.addAction("Duplicate node")
            a_del = menu.addAction("Delete node")
            chosen = menu.exec(event.screenPos())
            if chosen is a_del:
                self.editor.delete_node(self)
            elif chosen is a_dup:
                self.editor.duplicate_node(self)
            event.accept()

    class Scene(QtWidgets.QGraphicsScene):
        def __init__(self, editor):
            super().__init__()
            self.editor = editor
            self.temp_line = None
            self.temp_src = None

        def mousePressEvent(self, ev):
            if ev.button() == Qt.MouseButton.LeftButton:
                for it in self.items(ev.scenePos()):
                    if isinstance(it, NodeItem) and it.hit_out_port(ev.scenePos()):
                        self.temp_src = it
                        self.temp_line = self.addLine(QtCore.QLineF(it.out_port_scene_pos(), ev.scenePos()),
                                                      QtGui.QPen(QtGui.QColor("#e6c14b"), 2, Qt.PenStyle.DashLine))
                        self.temp_line.setZValue(5)
                        return
            super().mousePressEvent(ev)

        def mouseMoveEvent(self, ev):
            if self.temp_line is not None:
                self.temp_line.setLine(QtCore.QLineF(self.temp_src.out_port_scene_pos(), ev.scenePos()))
                return
            super().mouseMoveEvent(ev)

        def mouseReleaseEvent(self, ev):
            if self.temp_line is not None:
                self.removeItem(self.temp_line)
                target = None
                for it in self.items(ev.scenePos()):
                    if isinstance(it, NodeItem) and it is not self.temp_src and it.hit_in_port(ev.scenePos()):
                        target = it
                        break
                if target is not None:
                    self.editor.connect_nodes(self.temp_src, target)
                self.temp_line = None
                self.temp_src = None
                return
            super().mouseReleaseEvent(ev)

        def contextMenuEvent(self, ev):
            if self.itemAt(ev.scenePos(), QtGui.QTransform()) is not None:
                super().contextMenuEvent(ev)             # let the node/edge under the cursor handle it
                return
            menu = QtWidgets.QMenu()
            sub = menu.addMenu("Add stage")
            for s in STAGES:
                sub.addAction(s)
            chosen = menu.exec(ev.screenPos())
            if chosen is not None and chosen.text() in CATALOG:
                self.editor.add_stage(chosen.text())
            ev.accept()

    class Inspector(QtWidgets.QWidget):
        def __init__(self, editor):
            super().__init__()
            self.editor = editor
            self.node = None
            self.v = QtWidgets.QVBoxLayout(self)
            self.title = QtWidgets.QLabel("— no node selected —")
            f = self.title.font(); f.setBold(True); f.setPointSize(f.pointSize() + 1); self.title.setFont(f)
            self.v.addWidget(self.title)
            self.form_host = QtWidgets.QWidget()
            self.v.addWidget(self.form_host)
            self.v.addStretch(1)

        def show_node(self, node):
            self.node = node
            old = self.form_host.layout()
            if old is not None:
                QtWidgets.QWidget().setLayout(old)
            form = QtWidgets.QFormLayout(self.form_host)
            form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
            if node is None:
                self.title.setText("— no node selected —")
                return
            spec = CATALOG[node.step.stage]
            self.title.setText("%s" % node.step.stage)
            # structural tags
            for k in spec["tags"]:
                e = QtWidgets.QLineEdit(node.step.tags.get(k, ""))
                e.setPlaceholderText("(base over-cluster)" if k == "in" else "")
                e.textChanged.connect(lambda t, kk=k: self._set_tag(kk, t))
                form.addRow("%s tag" % k, e)
            # params: derived live from the stage's module; tick to include in the plan
            params = stage_params(node.step.stage)
            primary = [p for p in params if p.get("primary", True)]
            advanced = [p for p in params if not p.get("primary", True)]
            if params:
                form.addRow(QtWidgets.QLabel("<b>params</b> (ticked = written to plan)"))
            for pm in primary:
                self._add_param_row(form, node, pm)
            if advanced:
                adv_host = QtWidgets.QWidget()
                adv_form = QtWidgets.QFormLayout(adv_host)
                adv_form.setContentsMargins(0, 0, 0, 0)
                adv_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
                for pm in advanced:
                    self._add_param_row(adv_form, node, pm)
                start_open = not primary                  # nothing curated -> show the module's flags outright
                adv_host.setVisible(start_open)
                btn = QtWidgets.QToolButton()
                btn.setStyleSheet("QToolButton{border:none;color:#9cdcff;}")
                btn.setCheckable(True)
                btn.setChecked(start_open)

                def _label(on, n=len(advanced)):
                    return ("\u25be " if on else "\u25b8 ") + "advanced module flags (%d)" % n
                btn.setText(_label(start_open))
                btn.toggled.connect(lambda on, b=btn, w=adv_host: (w.setVisible(on), b.setText(_label(on))))
                form.addRow(btn)
                form.addRow(adv_host)

        def _add_param_row(self, form, node, pm):
            row = QtWidgets.QWidget(); h = QtWidgets.QHBoxLayout(row); h.setContentsMargins(0, 0, 0, 0)
            cb = QtWidgets.QCheckBox()
            on = pm["flag"] in node.step.params
            cb.setChecked(on)
            ed = self._editor_for(pm, node.step.params.get(pm["flag"], pm["default"]))
            ed.setEnabled(on)
            cb.toggled.connect(lambda c, p=pm, e=ed: self._toggle_param(p, e, c))
            self._wire_editor(pm, ed)
            h.addWidget(cb); h.addWidget(ed, 1)
            lab = QtWidgets.QLabel("--%s" % pm["flag"]); lab.setToolTip(pm["help"])
            form.addRow(lab, row)

        def _editor_for(self, pm, value):
            if pm["type"] == "choice":
                c = QtWidgets.QComboBox(); c.addItems([str(x) for x in pm["choices"]])
                i = c.findText(str(value))
                if i >= 0:
                    c.setCurrentIndex(i)
                return c
            e = QtWidgets.QLineEdit(str(value))
            return e

        def _wire_editor(self, pm, ed):
            if isinstance(ed, QtWidgets.QComboBox):
                ed.currentTextChanged.connect(lambda t, p=pm: self._set_param(p, t))
            else:
                ed.textChanged.connect(lambda t, p=pm: self._set_param(p, t))

        def _coerce(self, pm, t):
            if pm["type"] == "int":
                return int(float(t))
            if pm["type"] == "float":
                return float(t)
            return t

        def _set_tag(self, k, t):
            if self.node is None:
                return
            self.node.step.tags[k] = t
            self.node.update()
            self.editor.refresh_edges()

        def _toggle_param(self, pm, ed, on):
            ed.setEnabled(on)
            if self.node is None:
                return
            if on:
                try:
                    self.node.step.params[pm["flag"]] = self._coerce(
                        pm, ed.currentText() if isinstance(ed, QtWidgets.QComboBox) else ed.text())
                except ValueError:
                    self.node.step.params[pm["flag"]] = pm["default"]
            else:
                self.node.step.params.pop(pm["flag"], None)
            self.node.update()
            self.editor.schedule_preview()

        def _set_param(self, pm, t):
            if self.node is None or pm["flag"] not in self.node.step.params:
                return
            try:
                self.node.step.params[pm["flag"]] = self._coerce(pm, t)
            except ValueError:
                pass
            self.editor.schedule_preview()

    class Palette(QtWidgets.QListWidget):
        def __init__(self, editor):
            super().__init__()
            self.editor = editor
            for s in STAGES:
                self.addItem(s)
            self.itemDoubleClicked.connect(lambda it: editor.add_stage(it.text()))

    class PreviewDock(QtWidgets.QDockWidget):
        """Shows the exact command sequence the current plan would run, via the real fiber-pipeline in
        --dry-run.  Refreshes on demand, and (when 'auto' is on) shortly after each edit."""
        def __init__(self, editor):
            super().__init__("Dry-run preview", editor)
            self.editor = editor
            w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w); v.setContentsMargins(4, 4, 4, 4)
            bar = QtWidgets.QHBoxLayout()
            bar.addWidget(QtWidgets.QLabel("elec"))
            self.elec = QtWidgets.QLineEdit("5"); self.elec.setFixedWidth(52)
            self.elec.setValidator(QtGui.QIntValidator(0, 9999, self))
            self.elec.textChanged.connect(lambda *_: editor.schedule_preview())
            bar.addWidget(self.elec)
            self.auto = QtWidgets.QCheckBox("auto"); self.auto.setChecked(True)
            bar.addWidget(self.auto)
            btn = QtWidgets.QPushButton("Refresh"); btn.clicked.connect(editor.run_preview)
            bar.addWidget(btn); bar.addStretch(1)
            v.addLayout(bar)
            self.out = QtWidgets.QPlainTextEdit(); self.out.setReadOnly(True)
            mono = QtGui.QFont("monospace"); mono.setStyleHint(QtGui.QFont.StyleHint.Monospace)
            self.out.setFont(mono)
            self.out.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
            v.addWidget(self.out)
            self.setWidget(w)
            self.visibilityChanged.connect(lambda vis: editor.run_preview() if vis else None)

    class Editor(QtWidgets.QMainWindow):
        def __init__(self, path=None):
            super().__init__()
            self.path = path
            self.steps = []
            self.nodes = []
            self.edges = []
            self.setWindowTitle("fiber-kit pipeline editor")
            self.resize(1200, 720)

            self.scene = Scene(self)
            self.view = QtWidgets.QGraphicsView(self.scene)
            self.view.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            self.view.setDragMode(QtWidgets.QGraphicsView.DragMode.RubberBandDrag)
            self.view.setBackgroundBrush(QtGui.QColor("#202329"))
            self.setCentralWidget(self.view)

            pal = QtWidgets.QDockWidget("Stages", self)
            pal.setWidget(Palette(self))
            self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, pal)

            self.inspector = Inspector(self)
            insp = QtWidgets.QDockWidget("Inspector", self)
            scroll = QtWidgets.QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(self.inspector)
            insp.setWidget(scroll)
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, insp)

            self.preview = PreviewDock(self)
            self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.preview)
            self._preview_timer = QtCore.QTimer(self)
            self._preview_timer.setSingleShot(True); self._preview_timer.setInterval(450)
            self._preview_timer.timeout.connect(self.run_preview)

            self._menus()
            self.scene.selectionChanged.connect(self._on_sel)
            if path and os.path.isfile(path):
                self.load(path)
            else:
                self.statusBar().showMessage("new plan — double-click a stage on the left to add it")

        def _menus(self):
            m = self.menuBar().addMenu("&File")
            def act(name, slot, sc=None):
                a = QtGui.QAction(name, self); a.triggered.connect(slot)
                if sc:
                    a.setShortcut(sc)
                m.addAction(a); return a
            act("&New", self.new, "Ctrl+N")
            act("&Open…", self.open_dialog, "Ctrl+O")
            act("&Save", self.save, "Ctrl+S")
            act("Save Plan &As…", self.save_as, "Ctrl+Shift+S")
            m.addSeparator()
            act("Make &Primary (write into fiber-kit.yaml)…", self.make_primary)
            m.addSeparator()
            em = self.menuBar().addMenu("&Edit")
            em.addAction(QtGui.QAction("&Delete node", self, triggered=self.delete_selected, shortcut="Del"))
            em.addAction(QtGui.QAction("Du&plicate node", self, triggered=self._duplicate_selected, shortcut="Ctrl+D"))
            QtGui.QShortcut(QtGui.QKeySequence(Qt.Key.Key_Backspace), self, activated=self.delete_selected)
            mm = self.menuBar().addMenu("&Plan")
            mm.addAction(QtGui.QAction("Auto-&layout", self, triggered=self.relayout, shortcut="Ctrl+L"))
            mm.addAction(QtGui.QAction("&Validate", self, triggered=self.do_validate, shortcut="Ctrl+R"))
            mm.addAction(QtGui.QAction("Dry-run &preview", self, triggered=self.show_preview, shortcut="Ctrl+P"))

        # ---- model <-> scene ----
        def _clear(self):
            self.scene.clear(); self.nodes = []; self.edges = []

        def _add_node(self, step):
            node = NodeItem(step, self)
            self.scene.addItem(node)
            self.nodes.append(node)
            return node

        def refresh_edges(self):
            for e in self.edges:
                self.scene.removeItem(e)
            self.edges = []
            by_index = {i: self.nodes[i] for i in range(len(self.nodes))}
            for src, dst, kind in derive_edges(self.steps):
                e = EdgeItem(by_index[src], by_index[dst], kind)
                self.scene.addItem(e); self.edges.append(e)
            self.schedule_preview()

        def schedule_preview(self):
            if getattr(self, "preview", None) is not None and self.preview.isVisible() \
                    and self.preview.auto.isChecked():
                self._preview_timer.start()

        def run_preview(self):
            if getattr(self, "preview", None) is None:
                return
            elec = self.preview.elec.text() or "5"
            text, ok = render_dry_run(self.steps, elec=elec)
            self.preview.out.setPlainText(text)
            self.preview.out.setStyleSheet("" if ok else "QPlainTextEdit{color:#ffb4b4;}")
            self.statusBar().showMessage("dry-run preview refreshed" if ok else "dry-run preview: error")

        def show_preview(self):
            self.preview.show(); self.preview.raise_(); self.run_preview()

        def select_node(self, node):
            self.inspector.show_node(node)

        def _on_sel(self):
            sel = [it for it in self.scene.selectedItems() if isinstance(it, NodeItem)]
            self.inspector.show_node(sel[0] if len(sel) == 1 else None)

        # ---- actions ----
        def add_stage(self, stage):
            c = self.view.mapToScene(self.view.viewport().rect().center())
            step = new_step(stage, c.x(), c.y())
            # sensible default out tag so wiring is easy
            if "out" in CATALOG[stage]["tags"] and not step.tags.get("out"):
                step.tags["out"] = {"fiber-session": "", "fiber-refine": "refine", "fiber-cpos": "refine",
                                    "fiber-intrachunk": "refine_intrachunk",
                                    "fiber-link": "refine_linked"}.get(stage, stage.replace("fiber-", ""))
            self.steps.append(step)
            self._add_node(step)
            self.refresh_edges()
            self.statusBar().showMessage("added %s" % stage)

        def connect_nodes(self, src_node, dst_node):
            out = src_node.step.tag("out")
            dst_node.step.tags["in"] = out
            dst_node.update()
            self.refresh_edges()
            self.inspector.show_node(dst_node)
            self.statusBar().showMessage("wired %s.out='%s' -> %s.in" % (src_node.step.stage, out, dst_node.step.stage))

        def delete_node(self, node):
            if node not in self.nodes:
                return
            self.scene.removeItem(node)
            self.nodes.remove(node)
            self.steps.remove(node.step)                 # nodes[i] and steps[i] stay parallel
            if self.inspector.node is node:
                self.inspector.show_node(None)
            self.refresh_edges()
            self.statusBar().showMessage("deleted %s (downstream tags left intact -- Validate to review)" % node.step.stage)

        def delete_selected(self):
            for node in [it for it in self.scene.selectedItems() if isinstance(it, NodeItem)]:
                self.delete_node(node)

        def _duplicate_selected(self):
            for node in [it for it in self.scene.selectedItems() if isinstance(it, NodeItem)]:
                self.duplicate_node(node)

        def duplicate_node(self, node):
            step = duplicate_step(node.step)
            self.steps.append(step)
            new = self._add_node(step)
            self.refresh_edges()
            new.setSelected(True)
            self.statusBar().showMessage("duplicated %s -> out='%s'" % (step.stage, step.tags.get("out", "")))

        def remove_edge(self, edge):
            edge.dst.step.tags[edge.kind] = ""           # edges are derived from tags; clearing the tag drops it
            edge.dst.update()
            self.refresh_edges()
            if self.inspector.node is edge.dst:
                self.inspector.show_node(edge.dst)
            self.statusBar().showMessage("removed %s connection into %s" % (edge.kind, edge.dst.step.stage))

        def relayout(self):
            auto_layout(self.steps)
            for n in self.nodes:
                n.setPos(n.step.x, n.step.y)
            self.refresh_edges()
            self.view.fitInView(self.scene.itemsBoundingRect().adjusted(-40, -40, 40, 40),
                                Qt.AspectRatioMode.KeepAspectRatio)

        def do_validate(self):
            w = validate(self.steps)
            if not w:
                QtWidgets.QMessageBox.information(self, "Validate", "Plan looks consistent.")
            else:
                QtWidgets.QMessageBox.warning(self, "Validate", "\n".join("• " + x for x in w))

        def new(self):
            self.path = None; self.steps = []; self._clear()
            self.setWindowTitle("fiber-kit pipeline editor — (new)")
            self.statusBar().showMessage("new plan")

        def open_dialog(self):
            p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open plan", "", "YAML (*.yaml *.yml);;All (*)")
            if p:
                self.load(p)

        def load(self, path):
            try:
                self.steps = load_plan(path)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Open", "Could not read plan:\n%s" % e); return
            self.path = path
            if not any(s.x or s.y for s in self.steps):  # legacy / hand-written plan with no layout: arrange it
                auto_layout(self.steps)
            self._clear()
            for s in self.steps:
                self._add_node(s)
            for n in self.nodes:                          # place at saved (or just-computed) positions
                n.setPos(n.step.x, n.step.y)
            self.refresh_edges()
            self.setWindowTitle("fiber-kit pipeline editor — %s" % os.path.basename(path))
            self.statusBar().showMessage("loaded %d steps from %s" % (len(self.steps), path))

        def save(self):
            if not self.path:
                return self.save_as()
            try:
                text = dump_plan(self.steps)
            except ValueError as e:
                QtWidgets.QMessageBox.critical(self, "Save", "Cannot serialise plan:\n%s" % e); return
            with open(self.path, "w") as fh:
                fh.write(text)
            self.statusBar().showMessage("saved %s" % self.path)

        def save_as(self):
            p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save plan as", self.path or "fiber-plan.yaml",
                                                         "YAML (*.yaml *.yml)")
            if p:
                self.path = p
                self.setWindowTitle("fiber-kit pipeline editor — %s" % os.path.basename(p))
                self.save()

        def make_primary(self):
            try:
                topo_order(self.steps)                   # refuse to promote a cyclic plan
            except ValueError as e:
                QtWidgets.QMessageBox.critical(self, "Make primary", "Plan is not runnable:\n%s" % e); return
            default = os.environ.get("FK_CONFIG") or os.path.join(
                os.path.dirname(self.path) if self.path else "", "fiber-kit.yaml") or "fiber-kit.yaml"
            p, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Make primary — write plan into fiber-kit.yaml (FK_* knobs & comments preserved)",
                default, "YAML (*.yaml *.yml)")
            if not p:
                return
            cfg = ""
            if os.path.isfile(p):
                with open(p) as fh:
                    cfg = fh.read()
            with open(p, "w") as fh:
                fh.write(embed_plan_in_config(cfg, self.steps))
            self.statusBar().showMessage("wrote primary plan into %s — `fiber-pipeline <elec>` now runs it by default" % p)

    return QtWidgets, Editor


def main():
    _require_qt()
    from PySide6 import QtWidgets
    _qtw, Editor = _build_qt()
    app = QtWidgets.QApplication(sys.argv)
    path = sys.argv[1] if len(sys.argv) > 1 else None
    win = Editor(path)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
