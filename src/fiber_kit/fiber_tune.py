#!/usr/bin/env python3
"""fiber-tune -- search a plan's knobs against a ground truth on a subset of chunks, then confirm the best.

    fiber-tune search  <session> <elec> --spec tune.yaml [--max-trials N] [--dry-run]
    fiber-tune confirm <session> <elec> --spec tune.yaml [--top 3] [--chunks all] [--keep-ws]
    fiber-tune show    <session> <elec> --spec tune.yaml

A thin driver on fiber-bench: every trial is a `fiber-bench run` of the plan with one set of overrides,
under bench's hashed stage tags, so a step whose parameters did not change is not run again (changing a
merge knob re-runs the merges, not fiber-session) and a search can be stopped and resumed.

SPEC (yaml):
    name:      exp-g6                  # results go to <session dir>/fk-bench/tune/<name>/
    config:    fiber-kit-exp.yaml      # base config with a `pipeline:` plan (relative to the spec file)
    truth:     {clu: gt.clu, res: gt.res, units: "2-34", exclude: "", tol: 5, int_rate: 5.0}
    chunks:    "2-3"                   # FK_CHUNKS of the search (0-based session chunks; "" = all)
    score_at:  psame                   # plan step whose output is scored (default: the last step with `out:`)
    objective: "all_acc - 0.01*merges" # expression over the metric names below
    set:       {}                      # fixed overrides for every trial
    start:     {}                      # overrides the search starts from (default: the base config)
    passes:    2                       # coordinate-search passes over every knob
    min_gain:  0.002                   # an objective gain below this does not move the search
    search_realign: false              # true = every trial runs the FULL plan in a workspace (see below)
    stages:                            # knob groups, searched in this order (upstream first)
      - name: session
        knobs: {FK_SESSION_COARSE_DR: [0.07, 0.09, 0.12, 0.15], FK_SESSION_INCL_ASSIGN: [0, 1]}
      - name: psame
        knobs: {FK_PSAME_THETA: [0.3, 0.5, 0.7]}

SEARCH.  Coordinate ascent: from `start`, each knob in turn is tried at each of its listed values with
every other knob held at the current best; the best value is kept when it raises the objective by at
least `min_gain`.  Groups are visited in spec order, then the whole list again (`passes`), stopping
early when a pass changes nothing.  Not a global optimum: interactions between knobs are seen only
through the passes.

IN-PLACE REALIGN.  fiber-realign steps without `out:` rewrite the session's .res/.spk.  A search would
then start each trial from data the previous trial moved, so `search` drops those steps from the plan
(and fiber-bench turns fiber-session's own end-of-run realign off): the search compares parameters on
the original alignment.  `confirm` re-runs the best configurations with the FULL plan -- realign steps
and fiber-session's own end-of-run realign included -- each in a private WORKSPACE
(fk-bench/tune/<name>/ws/<hash>/): the .res/.spk/.fet/offset files are copied, every other session file
is symlinked, stage outputs are not carried over.  A workspace costs one copy of those files (removed
afterwards unless --keep-ws).

The realign changes what the later stages see.  On g6 0316 (one 6-min chunk, the same parameters) the
search scored ARI 0.675 / accuracy 0.376 and the workspace run ARI 0.503 / accuracy 0.417, so the
search's ranking is a guide and `confirm` decides.  `search_realign: true` runs EVERY trial as a
workspace run instead: faithful, but no step is shared between trials (each re-sorts the chunks).

METRICS (per trial, truth spikes in chunks not run are left out, as fiber-bench score):
    all_acc pyr_acc int_acc   mean over truth units of the best single cluster's accuracy
    all_rec all_prec ...      that cluster's recall / precision (also pyr_ / int_)
    all_n80 ...               units with accuracy >= 0.8;   all_frag80: clusters holding 80% of a unit
    merges merges_pyr_pyr merges_with_int
    ari purity completeness   on the truth spikes: ARI over sorted spikes, purity of the sorted spikes,
                              completeness = share of truth spikes in their unit's best single cluster
    unsorted n_clusters       share of truth spikes in clusters 0/1; clusters >= 2
The objective is an arithmetic expression over these names (+ - * /, numbers, min, max, abs); a NaN or
missing name makes it -inf.

OUTPUT (fk-bench/tune/<name>/): trials.jsonl (every trial: overrides, metrics, objective, phase),
trials.tsv, best.yaml -- the base config with the best overrides applied and the plan as given (realign
steps included, no FK_CHUNKS) -- and summary.md.
"""
import argparse
import ast
import datetime
import json
import math
import operator
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np

try:
    from . import fiber_bench as fb, fiber_chunks as fch, neuro_io as nio, session_yaml as sy
except ImportError:                      # run as a script from src/fiber_kit
    import fiber_bench as fb
    import fiber_chunks as fch
    import neuro_io as nio
    import session_yaml as sy

_LP = "▸ fiber-tune"
SPEC_KEYS = {"name", "config", "truth", "chunks", "score_at", "objective", "set", "start", "passes", "min_gain",
             "stages", "method", "search_realign"}
# files fiber-realign / fiber-session rewrite in place: copied into a workspace (everything else is symlinked)
_WS_COPY = re.compile(r"\.(res|spk|fet|off|offsets)(\.|$)")
# stage outputs: not carried into a workspace (the run makes its own)
_WS_SKIP = re.compile(r"\.(clu|clc|clp|cpos|fibers|qrn|psame|rdr|deconv|lmx)\.")


# ───────────────────────────── objective ─────────────────────────────
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
_UN = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_FN = {"min": min, "max": max, "abs": abs}


def compile_objective(expr):
    """Parse `expr` once; returns f(metrics) -> float (-inf when a name is missing or the value is NaN).
    Only arithmetic on metric names and numbers is allowed."""
    tree = ast.parse(str(expr), mode="eval")

    def check(n):
        if isinstance(n, ast.Expression):
            return check(n.body)
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            return check(n.left) and check(n.right)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _UN:
            return check(n.operand)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return True
        if isinstance(n, ast.Name):
            return True
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FN and not n.keywords:
            return all(check(a) for a in n.args)
        raise ValueError(f"objective {expr!r}: {type(n).__name__} is not allowed (arithmetic on metric names only)")
    check(tree)

    def ev(n, m):
        if isinstance(n, ast.Expression):
            return ev(n.body, m)
        if isinstance(n, ast.BinOp):
            a, b = ev(n.left, m), ev(n.right, m)
            if isinstance(n.op, ast.Div) and b == 0:
                return float("nan")
            return _BIN[type(n.op)](a, b)
        if isinstance(n, ast.UnaryOp):
            return _UN[type(n.op)](ev(n.operand, m))
        if isinstance(n, ast.Constant):
            return float(n.value)
        if isinstance(n, ast.Name):
            v = m.get(n.id)
            return float("nan") if v is None or isinstance(v, str) else float(v)
        return _FN[n.func.id](*[ev(a, m) for a in n.args])

    def f(metrics):
        try:
            v = float(ev(tree, metrics))
        except (TypeError, ValueError, OverflowError):
            return float("-inf")
        return v if math.isfinite(v) else float("-inf")
    return f


# ───────────────────────────── spec and plans ─────────────────────────────
def load_spec(path):
    with open(path) as fh:
        spec = fb._yaml_load(fh) or {}
    bad = set(spec) - SPEC_KEYS
    if bad:
        raise SystemExit(f"{_LP} · {path}: unknown key(s) {sorted(bad)} (known: {sorted(SPEC_KEYS)})")
    if not spec.get("config"):
        raise SystemExit(f"{_LP} · {path}: `config:` (the base config with a plan) is required")
    here = os.path.dirname(os.path.abspath(path))
    spec["config"] = spec["config"] if os.path.isabs(spec["config"]) else os.path.join(here, spec["config"])
    tr = dict(spec.get("truth") or {})
    for k in ("clu", "res"):
        if tr.get(k) and not os.path.isabs(tr[k]):
            tr[k] = os.path.join(here, tr[k])
    spec["truth"] = tr
    spec.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    spec.setdefault("objective", "all_acc")
    spec["set"] = dict(spec.get("set") or {}); spec["start"] = dict(spec.get("start") or {})
    spec["passes"] = int(spec.get("passes", 2)); spec["min_gain"] = float(spec.get("min_gain", 0.002))
    spec["chunks"] = "" if spec.get("chunks") in (None, "", "all") else str(spec["chunks"])
    spec["search_realign"] = bool(spec.get("search_realign", False))
    stages = spec.get("stages") or []
    for g in stages:
        if not isinstance(g, dict) or not isinstance(g.get("knobs"), dict) or not g["knobs"]:
            raise SystemExit(f"{_LP} · {path}: every `stages:` entry needs a `knobs:` mapping of knob -> [values]")
        g["knobs"] = {k: (v if isinstance(v, list) else [v]) for k, v in g["knobs"].items()}
    spec["stages"] = stages
    compile_objective(spec["objective"])                    # fail early on a bad expression
    return spec


def check_knobs(spec, cfg):
    """Every knob the spec sets must be one the base config lists (the config lists every knob the code
    reads), so a typo cannot silently do nothing."""
    names = set(spec["set"]) | set(spec["start"]) | {k for g in spec["stages"] for k in g["knobs"]}
    bad = sorted(k for k in names if k not in cfg and k != "FK_CHUNKS")
    if bad:
        raise SystemExit(f"{_LP} · knob(s) not in {os.path.basename(spec['config'])}: {bad}")
    # the chunk grid is what `chunks:` indexes: searching it would score different spans of the recording
    moving = sorted(k for g in spec["stages"] for k in g["knobs"] if k in ("FK_CHUNKS", "FK_SESSION_CHUNK_MIN"))
    if moving and spec["chunks"]:
        raise SystemExit(f"{_LP} · {moving} cannot be searched on a chunk subset (the chunk indices would move); "
                         "set it under `set:` or search it with `chunks: all`")


def default_score_at(plan):
    """The last step that writes a new stage (`out:`) and is not fiber-realign."""
    steps = [s for s in plan if s.get("out") and s["stage"] != "fiber-realign"]
    if not steps:
        raise SystemExit(f"{_LP} · the plan has no step with `out:` to score")
    return steps[-1].get("name") or steps[-1]["stage"]


def search_plan(plan, score_at):
    """The plan up to `score_at`, without in-place fiber-realign steps (they rewrite the shared .res/.spk)."""
    idx = next((i for i, s in enumerate(plan) if score_at in (s.get("name"), s["stage"])), None)
    if idx is None:
        raise SystemExit(f"{_LP} · score_at {score_at!r} names no step of the plan "
                         f"({', '.join(s.get('name') or s['stage'] for s in plan)})")
    return [dict(s) for s in plan[:idx + 1] if not (s["stage"] == "fiber-realign" and not s.get("out"))]


def _fmt(v):
    """An override value as fiber-bench's --set reads it back (yaml), type-preserving."""
    return json.dumps(v)


def _same(a, b):
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)


# ───────────────────────────── truth and scoring ─────────────────────────────
class Truth:
    """The truth file matched to one .res (re-matched when a workspace's realigned .res differs)."""

    def __init__(self, tr, sr):
        if not tr.get("clu") or not tr.get("res"):
            raise SystemExit(f"{_LP} · `truth:` needs clu and res")
        _, self.glab = nio.read_clu_file(tr["clu"])
        self.gres = nio.read_res_file(tr["res"])
        self.units = (fb.parse_units(tr.get("units"), tr.get("exclude", "")) if tr.get("units")
                      else sorted(int(u) for u in np.unique(self.glab) if u > 1 and
                                  int(u) not in fb.parse_units(tr.get("exclude", ""))))
        self.tol = int(tr.get("tol", 5))
        self.types = fb.type_units(self.gres, self.glab, self.units, sr, float(tr.get("int_rate", 5.0)),
                                   fb.parse_units(tr["interneurons"]) if tr.get("interneurons") else None)
        self._cache = {}

    def matched(self, res):
        key = (res.size, int(res[:1000].sum()), int(res[-1000:].sum()))
        if key not in self._cache:
            self._cache = {key: fb.match_gt(res, self.gres, self.glab, self.units, self.tol)}
        return self._cache[key]


def score_labels(clu, gt, truth):
    """fiber-bench's truth metrics plus ARI / purity / completeness / unsorted on the truth spikes."""
    clu = np.asarray(clu, np.int64)
    rsv = fch.is_reserved(clu)
    g = np.where(rsv, -1, gt)
    T, _ = fb.truth_metrics(clu, g, truth.units, truth.types)
    m = (g >= 0) & np.isin(g, truth.units)
    gl, cl = g[m], clu[m]; s = cl >= 2
    out = dict(T)
    out["n_clusters"] = int(np.unique(clu[(clu >= 2) & ~rsv]).size)
    out["unsorted"] = float((~s).mean()) if m.any() else float("nan")
    if s.any():
        from sklearn.metrics import adjusted_rand_score
        out["ari"] = float(adjusted_rand_score(gl[s], cl[s]))
        _, ci = np.unique(cl[s], return_inverse=True); _, ui = np.unique(gl[s], return_inverse=True)
        M = np.zeros((ui.max() + 1, ci.max() + 1), np.int64); np.add.at(M, (ui, ci), 1)
        out["purity"] = float(M.max(0).sum() / s.sum())
        out["completeness"] = float(M.max(1).sum() / m.sum())
    else:
        out.update(ari=float("nan"), purity=float("nan"), completeness=float("nan"))
    return out


# ───────────────────────────── trials ─────────────────────────────
class Tuner:
    def __init__(self, session, elec, spec, clu_method="stderiv_C5_D34", dry_run=False, log=print):
        self.spec, self.elec, self.dry_run, self.log = spec, int(elec), dry_run, log
        sp = sy.resolve_session_params(session, elec)
        self.session, self.base, self.sr = session, sp["base"], sp["sr"]
        self.cfg_full = fb.load_config(spec["config"])
        check_knobs(spec, self.cfg_full)
        plan = list(self.cfg_full.get("pipeline") or [])
        if not plan:
            raise SystemExit(f"{_LP} · {spec['config']} has no `pipeline:` plan")
        self.score_at = spec.get("score_at") or default_score_at(plan)
        self.method = spec.get("method") or self.cfg_full.get("FK_REALIGN_VARIANT") or clu_method
        self.dir = os.path.join(fb.bench_dir(self.base), "tune", spec["name"]); os.makedirs(self.dir, exist_ok=True)
        # the search config: base + fixed overrides, plan cut at score_at without in-place realigns
        cfg = dict(self.cfg_full); cfg.update(spec["set"]); cfg["pipeline"] = search_plan(plan, self.score_at)
        self.cfg_search_path = os.path.join(self.dir, "search-base.yaml")
        self._write_yaml(self.cfg_search_path, cfg, f"fiber-tune {spec['name']}: search base (no in-place realign)")
        self.cfg_search = fb.load_config(self.cfg_search_path)
        # the confirm config: base + fixed overrides, the FULL plan (in-place realign steps included)
        cfg = dict(self.cfg_full); cfg.update(spec["set"])
        self.cfg_confirm_path = os.path.join(self.dir, "confirm-base.yaml")
        self._write_yaml(self.cfg_confirm_path, cfg, f"fiber-tune {spec['name']}: confirm base (full plan)")
        self.cfg_confirm = fb.load_config(self.cfg_confirm_path)
        self.objective = compile_objective(spec["objective"])
        self.truth = None if dry_run else Truth(spec["truth"], self.sr)
        self.trials = self._read_trials()

    # ── bookkeeping ──
    @staticmethod
    def _write_yaml(path, cfg, header):
        import yaml
        with open(path, "w") as fh:
            fh.write(f"# {header}\n")
            yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=None, width=200)

    def _read_trials(self):
        p = os.path.join(self.dir, "trials.jsonl"); out = []
        if os.path.exists(p):
            with open(p) as fh:
                out = [json.loads(x) for x in fh if x.strip()]
        return out

    def _record(self, rec):
        self.trials.append(rec)
        with open(os.path.join(self.dir, "trials.jsonl"), "a") as fh:
            fh.write(json.dumps(rec, default=float) + "\n")

    def plan_of(self, overrides, cfg_base=None, upto=None, allow_inplace=False, keep_realign=False):
        cfg = dict(self.cfg_search if cfg_base is None else cfg_base); cfg.update(overrides)
        if self.spec["chunks"]:
            cfg["FK_CHUNKS"] = str(self.spec["chunks"])
        return fb.build_run(cfg, upto or self.score_at, keep_realign=keep_realign, allow_inplace=allow_inplace)

    # ── one trial ──
    def evaluate(self, overrides, phase="search", note=""):
        overrides = dict(overrides)
        if self.spec.get("search_realign"):
            return self.run_full(overrides, phase, note=note)
        cfg_run, steps, h = self.plan_of(overrides)
        tag = steps[-1]["out_tag"]
        for r in self.trials:
            if r.get("phase") == phase and r.get("hash") == h and r.get("ok"):
                return r
        t0 = time.time()
        ns = argparse.Namespace(session=self.session, elec=self.elec, clu_method=self.method,
                                config=self.cfg_search_path, set=[f"{k}={_fmt(v)}" for k, v in overrides.items()],
                                grid=[], sweep=None, upto=self.score_at, keep_realign=False, allow_inplace=False,
                                force=False, keep_going=True, dry_run=self.dry_run, chunks=self.spec["chunks"] or None)
        fb.cmd_run(ns)
        meth = cfg_run.get("FK_REALIGN_VARIANT") or self.method
        path = f"{self.base}.clu.{meth}.{self.elec}.{tag}"
        rec = dict(trial=len([r for r in self.trials if r.get("phase") == phase]) + 1, phase=phase, hash=h, tag=tag,
                   overrides=overrides, note=note, when=datetime.datetime.now().isoformat(timespec="seconds"),
                   fiber_kit=fb._fk_version())
        if self.dry_run:
            return dict(rec, ok=False, objective=float("-inf"), metrics={})
        if not os.path.exists(path):
            rec.update(ok=False, objective=float("-inf"), metrics={}, seconds=round(time.time() - t0, 1))
            self._record(rec); self.log(f"    FAILED (no {os.path.basename(path)}; see fk-bench/runs/{h}/run.log)")
            return rec
        res = nio.read_res(self.base, self.elec)
        _, clu = nio.read_clu_at(self.base, self.elec, variant=meth, tag=tag, n_spikes=res.size)
        M = score_labels(clu, self.truth.matched(res), self.truth)
        rec.update(ok=True, metrics=M, objective=self.objective(M), seconds=round(time.time() - t0, 1))
        self._record(rec)
        return rec

    def _line(self, r):
        M = r.get("metrics") or {}
        return (f"obj {r['objective']:.4f} | acc {M.get('all_acc', float('nan')):.3f} ari {M.get('ari', float('nan')):.3f} "
                f"pur {M.get('purity', float('nan')):.3f} comp {M.get('completeness', float('nan')):.3f} "
                f"merges {M.get('merges', '-')} clusters {M.get('n_clusters', '-')}")

    # ── coordinate search ──
    def search(self, max_trials=None):
        cur = dict(self.spec["start"])
        self.log(f"{_LP} · {self.spec['name']}: objective `{self.spec['objective']}`, scored at step {self.score_at!r}, "
                 f"chunks {self.spec['chunks'] or 'all'}")
        best = self.evaluate(cur, note="start")
        self.log(f"  start {json.dumps(cur) if cur else '(base config)'}: {self._line(best)}")
        n = 1
        for p in range(self.spec["passes"]):
            moved = False
            for g in self.spec["stages"]:
                for knob, vals in g["knobs"].items():
                    now = cur.get(knob, self.cfg_search.get(knob))
                    tried = []
                    for v in vals:
                        if _same(v, now):
                            continue
                        if max_trials and n >= max_trials:
                            self.log(f"{_LP} · --max-trials {max_trials} reached"); return self.finish(cur, best)
                        r = self.evaluate(dict(cur, **{knob: v}), note=f"pass {p + 1} {g.get('name', '')}: {knob}={v}")
                        n += 1; tried.append((r["objective"], v, r))
                        self.log(f"  [{g.get('name', '')}] {knob}={v!s:<10} {self._line(r)}")
                    if not tried:
                        continue
                    o, v, r = max(tried, key=lambda t: t[0])
                    # a failed trial (-inf) never moves the search; from a failed start, any finite trial does
                    if math.isfinite(o) and (not math.isfinite(best["objective"]) or
                                             o >= best["objective"] + self.spec["min_gain"]):
                        cur[knob] = v; best = r; moved = True
                        self.log(f"  -> {knob} = {v}  (objective {o:.4f})")
            if not moved:
                self.log(f"{_LP} · pass {p + 1} changed nothing: done"); break
        return self.finish(cur, best)

    def finish(self, cur, best):
        cfg = dict(self.cfg_full); cfg.update(self.spec["set"]); cfg.update(cur)
        hdr = (f"fiber-tune {self.spec['name']} best: objective {best['objective']:.4f} "
               f"({self.spec['objective']}) on chunks {self.spec['chunks'] or 'all'}\n"
               f"# overrides vs {os.path.basename(self.spec['config'])}: {json.dumps(dict(self.spec['set'], **cur))}")
        self._write_yaml(os.path.join(self.dir, "best.yaml"), cfg, hdr)
        self.write_tables()
        self.log(f"{_LP} · best {json.dumps(cur) if cur else '(base config)'}: {self._line(best)}")
        self.log(f"{_LP} · -> {os.path.join(self.dir, 'best.yaml')}, trials.tsv, summary.md")
        return cur, best

    # ── confirm with the full plan in private workspaces ──
    def make_workspace(self, h):
        src = os.path.dirname(os.path.abspath(self.base)); ws = os.path.join(self.dir, "ws", h)
        os.makedirs(ws, exist_ok=True); copied = 0
        for f in sorted(os.listdir(src)):
            p = os.path.join(src, f); q = os.path.join(ws, f)
            if os.path.isdir(p) or os.path.lexists(q) or _WS_SKIP.search(f):
                continue
            if _WS_COPY.search(f):
                if subprocess.run(["cp", "--reflink=auto", p, q], capture_output=True).returncode != 0:
                    shutil.copy2(p, q)
                copied += os.path.getsize(q)
            else:
                os.symlink(p, q)
        return ws, copied

    def run_full(self, overrides, phase, note="", extra=None, keep_ws=False):
        """One configuration with the FULL plan (in-place realign steps and fiber-session's own realign kept)
        in a private workspace.  Re-uses an earlier workspace run of the same hash, from any phase."""
        cfg_run, steps, h = self.plan_of(overrides, cfg_base=self.cfg_confirm, allow_inplace=True, keep_realign=True)
        done = next((x for x in self.trials if x.get("mode") == "workspace" and x.get("hash") == h and x.get("ok")), None)
        if done:
            return done
        ws, nbytes = self.make_workspace(h)
        wbase = os.path.join(ws, os.path.basename(self.base))
        self.log(f"    workspace {ws} ({nbytes / 1e9:.2f} GB copied)")
        t0 = time.time()
        ns = argparse.Namespace(session=wbase, elec=self.elec, clu_method=self.method, config=self.cfg_confirm_path,
                                set=[f"{k}={_fmt(v)}" for k, v in overrides.items()], grid=[], sweep=None,
                                upto=self.score_at, keep_realign=True, allow_inplace=True, force=False,
                                keep_going=True, dry_run=self.dry_run, chunks=self.spec["chunks"] or None)
        fb.cmd_run(ns)
        tag = steps[-1]["out_tag"]; meth = cfg_run.get("FK_REALIGN_VARIANT") or self.method
        rec = dict(trial=len([r for r in self.trials if r.get("phase") == phase]) + 1, phase=phase, mode="workspace",
                   hash=h, tag=tag, overrides=dict(overrides), note=note, chunks=self.spec["chunks"], workspace=ws,
                   when=datetime.datetime.now().isoformat(timespec="seconds"), fiber_kit=fb._fk_version(), **(extra or {}))
        path = f"{wbase}.clu.{meth}.{self.elec}.{tag}"
        if self.dry_run:
            return dict(rec, ok=False, objective=float("-inf"), metrics={})
        if os.path.exists(path):
            res = nio.read_res(wbase, self.elec)
            _, clu = nio.read_clu_at(wbase, self.elec, variant=meth, tag=tag, n_spikes=res.size)
            M = score_labels(clu, self.truth.matched(res), self.truth)
            rec.update(ok=True, metrics=M, objective=self.objective(M))
        else:
            rec.update(ok=False, metrics={}, objective=float("-inf"))
            self.log(f"    FAILED (see {os.path.join(ws, 'fk-bench')})")
        rec["seconds"] = round(time.time() - t0, 1)
        self._record(rec)
        if not keep_ws:
            for f in os.listdir(ws):
                q = os.path.join(ws, f)
                if os.path.isfile(q) and not os.path.islink(q) and _WS_COPY.search(f):
                    os.remove(q)                           # the copies; stage outputs stay for inspection
        return rec

    def confirm(self, top=3, chunks=None, keep_ws=False):
        ok = [r for r in self.trials if r.get("phase") == "search" and r.get("ok")]
        if not ok:
            raise SystemExit(f"{_LP} · no search trials in {self.dir}: run `fiber-tune search` first")
        seen, picks = set(), []
        start = next((r for r in ok if r.get("note") == "start"), None)
        for r in ([start] if start else []) + sorted(ok, key=lambda r: -r["objective"]):
            key = json.dumps(r["overrides"], sort_keys=True)
            if key not in seen and (len(picks) < top + (1 if start else 0)):
                seen.add(key); picks.append(r)
        if chunks is not None:
            self.spec = dict(self.spec, chunks="" if chunks in ("", "all") else str(chunks))
        self.log(f"{_LP} · confirm: {len(picks)} configuration(s), full plan up to {self.score_at!r} (realign "
                 f"steps included), chunks {self.spec['chunks'] or 'all'}")
        out = []
        for r in picks:
            self.log(f"  {json.dumps(r['overrides']) if r['overrides'] else '(start)'}")
            rec = self.run_full(r["overrides"], "confirm", note="confirm", keep_ws=keep_ws,
                                extra=dict(search_hash=r["hash"], search_objective=r["objective"]))
            out.append(rec)
            if rec.get("ok"):
                self.log(f"    {self._line(rec)}  (search: {r['objective']:.4f})")
        self.write_tables()
        return out

    # ── tables ──
    def write_tables(self):
        rows = []
        for r in self.trials:
            row = dict(phase=r.get("phase"), trial=r.get("trial", ""), hash=r.get("hash"), ok=r.get("ok"),
                       objective=r.get("objective"), overrides=json.dumps(r.get("overrides", {}), sort_keys=True),
                       seconds=r.get("seconds"), note=r.get("note", ""))
            for k in ("all_acc", "pyr_acc", "int_acc", "all_rec", "all_prec", "all_n80", "all_frag80", "merges",
                      "merges_pyr_pyr", "ari", "purity", "completeness", "unsorted", "n_clusters", "units_scored"):
                row[k] = (r.get("metrics") or {}).get(k, "")
            rows.append(row)
        fb._tsv_write(os.path.join(self.dir, "trials.tsv"), rows)
        S = [r for r in self.trials if r.get("ok")]
        lines = [f"# fiber-tune {self.spec['name']}", "",
                 f"- base config: `{self.spec['config']}`", f"- objective: `{self.spec['objective']}`",
                 f"- scored at: `{self.score_at}`; search chunks: `{self.spec['chunks'] or 'all'}`",
                 f"- trials: {len([r for r in S if r['phase'] == 'search'])} search, "
                 f"{len([r for r in S if r['phase'] == 'confirm'])} confirm", ""]
        for ph in ("search", "confirm"):
            R = sorted([r for r in S if r["phase"] == ph], key=lambda r: -r["objective"])[:15]
            if not R:
                continue
            lines += [f"## {ph} (best first)", "", "| objective | acc | ARI | purity | completeness | merges | clusters | overrides |",
                      "|---|---|---|---|---|---|---|---|"]
            for r in R:
                M = r["metrics"]
                lines.append(f"| {r['objective']:.4f} | {M.get('all_acc', float('nan')):.3f} | {M.get('ari', float('nan')):.3f} | "
                             f"{M.get('purity', float('nan')):.3f} | {M.get('completeness', float('nan')):.3f} | "
                             f"{M.get('merges', '')} | {M.get('n_clusters', '')} | `{json.dumps(r['overrides'], sort_keys=True)}` |")
            lines.append("")
        with open(os.path.join(self.dir, "summary.md"), "w") as fh:
            fh.write("\n".join(lines))


# ───────────────────────────── CLI ─────────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser(prog="fiber-tune", description="Search a plan's knobs against a ground truth "
                                 "on a subset of chunks (via fiber-bench), then confirm the best with the full plan.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("session"); p.add_argument("elec", type=int)
        p.add_argument("--spec", required=True, help="search spec (yaml; see the module docstring)")
        p.add_argument("--clu-method", default="stderiv_C5_D34",
                       help="method token when the config sets no FK_REALIGN_VARIANT (default stderiv_C5_D34)")
    p = sub.add_parser("search", help="coordinate search over the spec's knobs (no in-place realign)")
    common(p)
    p.add_argument("--max-trials", type=int, default=None, help="stop after this many trials (resumable)")
    p.add_argument("--dry-run", action="store_true", help="print the stage commands of the first trial only")
    p = sub.add_parser("confirm", help="re-run the best configurations with the full plan in private workspaces")
    common(p)
    p.add_argument("--top", type=int, default=3, help="best distinct configurations to confirm (plus the start)")
    p.add_argument("--chunks", default=None, help="chunks for the confirmation (default: the spec's; 'all' = every chunk)")
    p.add_argument("--keep-ws", action="store_true", help="keep the workspaces' copied .res/.spk/.fet files")
    p.add_argument("--dry-run", action="store_true", help="print the stage commands only")
    p = sub.add_parser("show", help="rewrite trials.tsv / summary.md and print the summary")
    common(p)
    a = ap.parse_args(argv)
    spec = load_spec(a.spec)
    t = Tuner(a.session, a.elec, spec, clu_method=a.clu_method, dry_run=getattr(a, "dry_run", False))
    if a.cmd == "search":
        if a.dry_run:
            t.evaluate(dict(spec["start"]), note="dry-run"); return 0
        t.search(a.max_trials)
    elif a.cmd == "confirm":
        t.confirm(a.top, a.chunks, a.keep_ws)
    else:
        t.write_tables()
        with open(os.path.join(t.dir, "summary.md")) as fh:
            print(fh.read())
    return 0


if __name__ == "__main__":
    sys.exit(main())
