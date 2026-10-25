#!/usr/bin/env python3
"""fiber-bench -- run, score and compare pipeline-parameter variants, each under a config HASH.

    fiber-bench run    <session> <elec> --config fiber-kit-exp.yaml [--set K=V ...] [--grid K=v1,v2 ...]
                                        [--sweep sweep.yaml] [--upto session] [--dry-run]
    fiber-bench score  <session> <elec> --gt-clu OLD.clu --gt-res OLD.res [--gt-units 2-32 --gt-exclude 7,18]
                                        [--stages fiber_session,p_t50]
    fiber-bench report <session> <elec> [--out report.html]
    fiber-bench list   <session> <elec>

HASHES.  Every clustering step of the plan writes its output under `<out>_<hash>`, where the hash is the
first 8 hex digits of a SHA-1 over
  * the step's stage name and its plan `params:`;
  * every FK_* key that stage reads -- fiber-session: every key NOT owned by another stage (so a new
    session knob is never missed); the others: their own prefixes (FK_XCM_, FK_RDR_, FK_FLAG_, ...);
  * FK_REALIGN_VARIANT (the method token every file name carries);
  * the hash of the step that produced its input (`in:`), and for fiber-session with FK_SESSION_EXCLUDE
    the artifact mask's step -- a chain, so changing an upstream knob changes every downstream hash and
    changing a downstream knob leaves the upstream outputs (and their cache) alone.
Identical parameters give identical tags: a step whose output already exists is skipped, so a sweep
can be stopped and resumed.  The hash covers PARAMETERS, not code: the fiber-kit version is recorded
with every run instead.

ISOLATION.  Each run gets its own config file (<session dir>/fk-bench/runs/<hash>/fiber-kit.yaml: the
base config, the overrides, and the plan with hashed tags), and fiber-pipeline runs with every FK_*
variable stripped from its environment (an exported FK_* would override the yaml and break the hash).
fiber-session's end-of-run realign (FK_SESSION_REALIGN) rewrites the shared .res/.spk IN PLACE, so
every later run would start from data an earlier run moved: the harness forces it `off` (and the hash
records that) unless --keep-realign.  For the same reason in-place plan steps (fiber-realign without
`out:`) end the run unless --allow-inplace.

METRICS (score).  Against a ground-truth .clu/.res (spikes matched one-to-one within --gt-tol samples):
per truth unit, its best single cluster's accuracy n_uc / (n_u + g_c - n_uc), recall, precision, and how
many clusters hold 80% of it (frag80); merges = clusters holding >= 20% and >= 50 spikes of a second
truth unit.  Units are typed pyramidal / interneuron (rate >= --int-rate Hz, or --interneurons) and
every summary is given for each type.  Without truth: cluster count, shares in clusters 0/1, refractory
violations, the fiber-rdr excess ratio, the own-gain IQR and the residual-variance non-uniformity per
cluster -- and, when truth is given, how well each of those predicts a cluster's impurity (AUC), so a
truth-free objective can be chosen for sessions without a curation.
"""
import argparse
import datetime
import hashlib
import itertools
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

try:
    from . import neuro_io as nio, session_yaml as sy, fiber_chunks as _fch
except ImportError:                                              # script / direct execution
    import neuro_io as nio, session_yaml as sy, fiber_chunks as _fch

_LP = "▸ fiber-bench"

# FK_* prefixes owned by each stage.  fiber-session owns everything NOT listed for another stage.
STAGE_PREFIXES = {
    "fiber-flag-artifacts": ("FK_FLAG_",),
    "fiber-xcorr-merge": ("FK_XCM_",),
    "fiber-realign": ("FK_REALIGN_",),
    "fiber-rdr": ("FK_RDR_",),
    "fiber-backbone-link": ("FK_BBLINK_",),
    "fiber-anchor-link": ("FK_ALINK_",),
    "fiber-consensus": ("FK_CONSENSUS_", "FK_CONS_"),
    "fiber-lowmix": ("FK_LOWMIX_",),
    "fiber-intrachunk": ("FK_INTRA_",),
    "fiber-link": ("FK_LINK_", "FK_MERGE_"),
    "fiber-refine": ("FK_REFINE_",),
    "fiber-peel": ("FK_PEEL_",),
    "fiber-stochastic": ("FK_STOCH_",),
}
_FOREIGN_TO_SESSION = tuple(p for st, ps in STAGE_PREFIXES.items() if st != "fiber-realign" for p in ps)
GLOBAL_KEYS = ("FK_REALIGN_VARIANT", "FK_CHUNKS", "FK_SESSION_CHUNK_MIN")   # every stage's hash carries these


# ───────────────────────────── config, hashes, plans ─────────────────────────────
def load_config(path):
    import yaml
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def stage_params(stage, cfg):
    """The FK_* keys (sorted) that `stage` reads, plus the global keys."""
    keys = [k for k in cfg if isinstance(k, str) and k.startswith("FK_")]
    if stage == "fiber-session":
        own = [k for k in keys if not k.startswith(_FOREIGN_TO_SESSION)]
    else:
        pre = STAGE_PREFIXES.get(stage)
        own = [k for k in keys if pre and k.startswith(pre)]
    own = sorted(set(own) | {k for k in GLOBAL_KEYS if k in cfg})
    return {k: cfg[k] for k in own}


def _h(obj):
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:8]


def step_hash(step, cfg, upstream):
    return _h({"stage": step["stage"], "params": step.get("params") or {},
               "knobs": stage_params(step["stage"], cfg), "upstream": sorted(upstream)})


def _truthy(v):
    return str(v).strip().lower() not in ("", "0", "false", "off", "no", "none")


def build_run(cfg, upto=None, keep_realign=False, allow_inplace=False):
    """Hash the plan in `cfg` (its `pipeline:` list) up to and including step `upto` (a step name or a
    stage; default: the first fiber-session step).  Returns (cfg_out, steps, final_hash), where steps
    carry 'hash', 'out_tag' and the rewritten 'in'.  cfg_out has the hashed plan and the flag stage tag."""
    cfg = dict(cfg)
    if not keep_realign:
        cfg["FK_SESSION_REALIGN"] = "off"
    plan = list(cfg.get("pipeline") or [])
    if not plan:
        raise SystemExit(f"{_LP} · the config has no `pipeline:` plan")
    if upto is None:
        upto = next((s.get("name") or s["stage"] for s in plan if s["stage"] == "fiber-session"), None)
    idx = next((i for i, s in enumerate(plan) if upto in (s.get("name"), s["stage"])), None)
    if idx is None:
        raise SystemExit(f"{_LP} · --upto {upto!r} names no step of the plan "
                         f"({', '.join(s.get('name') or s['stage'] for s in plan)})")
    by_out, steps, flag_hash = {}, [], None
    for s in plan[:idx + 1]:
        s = dict(s)
        st = s["stage"]
        if st == "fiber-realign" and not s.get("out") and not allow_inplace:
            raise SystemExit(f"{_LP} · step {s.get('name') or st!r} realigns IN PLACE (rewrites the shared .res/.spk); "
                             "stop the run before it (--upto) or pass --allow-inplace")
        up = []
        if s.get("in"):
            up.append(by_out.get(s["in"], ("external", s["in"]))[0])
        if st == "fiber-session" and _truthy(cfg.get("FK_SESSION_EXCLUDE", 0)) and flag_hash:
            up.append(flag_hash)
        h = step_hash(s, cfg, up)
        s["hash"] = h
        if s.get("in") and s["in"] in by_out:
            s["in"] = by_out[s["in"]][1]
        if st == "fiber-flag-artifacts":
            tag = f"{cfg.get('FK_FLAG_STAGE', 'artifact')}_{h}"
            cfg["FK_FLAG_STAGE"] = tag; s["out_tag"] = tag; flag_hash = h
        elif s.get("out"):
            tag = f"{s['out']}_{h}"
            by_out[s["out"]] = (h, tag); s["out"] = tag; s["out_tag"] = tag
        steps.append(s)
    cfg["pipeline"] = [{k: v for k, v in s.items() if k not in ("hash", "out_tag")} for s in steps]
    return cfg, steps, steps[-1]["hash"]


def parse_overrides(sets, grids, sweep_file):
    """[{}] -> list of override dicts: --set applied to all, the cartesian product of --grid, and the
    entries of a sweep file (a list of dicts, or a dict of lists = another grid)."""
    import yaml
    base = {}
    for kv in sets or []:
        k, v = kv.split("=", 1); base[k.strip()] = yaml.safe_load(v) if v.strip() != "" else ""
    axes = []
    for kv in grids or []:
        k, vs = kv.split("=", 1)
        axes.append([(k.strip(), yaml.safe_load(v) if v.strip() != "" else "") for v in vs.split(",")])
    combos = [dict(c) for c in itertools.product(*axes)] if axes else [{}]
    if sweep_file:
        with open(sweep_file) as fh:
            sw = yaml.safe_load(fh)
        if isinstance(sw, dict):
            sw = [dict(c) for c in itertools.product(*[[(k, v) for v in (vs if isinstance(vs, list) else [vs])]
                                                      for k, vs in sw.items()])]
        combos = [dict(c, **e) for c in combos for e in sw]
    return [dict(base, **c) for c in combos]


# ───────────────────────────── registry ─────────────────────────────
def bench_dir(base):
    d = os.path.join(os.path.dirname(os.path.abspath(base)) or ".", "fk-bench")
    os.makedirs(d, exist_ok=True)
    return d


def read_registry(base):
    p = os.path.join(bench_dir(base), "runs.jsonl")
    out = []
    if os.path.exists(p):
        with open(p) as fh:
            for line in fh:
                if line.strip():
                    out.append(json.loads(line))
    return out


def _append(base, rec):
    with open(os.path.join(bench_dir(base), "runs.jsonl"), "a") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")


def _fk_version():
    try:
        from . import __version__ as v
    except Exception:                                            # pragma: no cover
        v = "?"
    return v


def _pipeline_exe():
    here = os.path.dirname(os.path.abspath(__file__))
    for c in (os.path.join(here, "..", "..", "scripts", "fiber-pipeline"), shutil.which("fiber-pipeline")):
        if c and os.path.exists(c):
            return os.path.abspath(c)
    raise SystemExit(f"{_LP} · fiber-pipeline not found")


def cmd_run(a):
    import yaml
    cfg0 = load_config(a.config)
    cfg_session = sy.resolve_session_params(a.session, a.elec)
    base = cfg_session["base"]
    method = cfg0.get("FK_REALIGN_VARIANT") or a.clu_method
    combos = parse_overrides(a.set, a.grid, a.sweep)
    if a.chunks is not None:                                        # --chunks: one more override, in the hash
        _fch.parse(a.chunks)                                        # validate early
        combos = [dict(c, FK_CHUNKS=str(a.chunks)) for c in combos]
    print(f"{_LP} · {len(combos)} variant(s) of {os.path.basename(a.config)} up to step {a.upto or 'fiber-session'}")
    for ov in combos:
        cfg = dict(cfg0); cfg.update(ov)
        cfg_run, steps, h = build_run(cfg, a.upto, a.keep_realign, a.allow_inplace)
        meth = cfg_run.get("FK_REALIGN_VARIANT") or method
        todo = []
        for s in steps:
            spk_m = cfg_run.get("FK_FLAG_SPK", "standard") if s["stage"] == "fiber-flag-artifacts" else meth
            path = f"{base}.clu.{spk_m}.{a.elec}.{s['out_tag']}" if s.get("out_tag") else None
            s["path"] = path
            if path is None or not os.path.exists(path) or a.force:
                todo.append(s)
        desc = ", ".join(f"{k}={v}" for k, v in ov.items()) or "(base config)"
        tags = " -> ".join(s.get("out_tag", s["stage"]) for s in steps)
        if not todo:
            print(f"  [{h}] {desc}: cached ({tags})")
            if not any(r.get("hash") == h and r.get("ok") for r in read_registry(base)):
                # outputs exist but no successful record (e.g. the sweep was killed between the stage
                # finishing and the registry write): register them, or `score` would never see them
                _append(base, dict(hash=h, overrides=ov, base_config=os.path.abspath(a.config), elec=a.elec, method=meth,
                                   steps=[{k: s.get(k) for k in ("stage", "name", "in", "out_tag", "hash")} for s in steps],
                                   final_tag=steps[-1].get("out_tag"), ok=True, seconds=None, fiber_kit=_fk_version(),
                                   when=datetime.datetime.now().isoformat(timespec="seconds"),
                                   keep_realign=bool(a.keep_realign), note="registered from existing outputs"))
            continue
        rdir = os.path.join(bench_dir(base), "runs", h); os.makedirs(rdir, exist_ok=True)
        cfg_file = os.path.join(rdir, "fiber-kit.yaml")
        run_cfg = dict(cfg_run); run_cfg["pipeline"] = [p for p, s in zip(cfg_run["pipeline"], steps) if s in todo]
        with open(cfg_file, "w") as fh:
            fh.write(f"# fiber-bench run {h}: {desc}\n# base config {os.path.abspath(a.config)}\n")
            yaml.safe_dump(run_cfg, fh, sort_keys=False, default_flow_style=None, width=200)
        env = {k: v for k, v in os.environ.items() if not k.startswith("FK_")}
        env.update(FK_CONFIG=cfg_file, FK_DIR=os.path.dirname(os.path.abspath(base)), FK_SESS=os.path.basename(base))
        cmd = [_pipeline_exe(), str(a.elec)] + (["--dry-run"] if a.dry_run else [])
        print(f"  [{h}] {desc}: running {len(todo)}/{len(steps)} step(s) ({tags})", flush=True)
        t0 = time.time()
        with open(os.path.join(rdir, "run.log"), "a") as log:
            r = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(base)), env=env, stdout=log,
                               stderr=subprocess.STDOUT)
        dt = time.time() - t0
        ok = r.returncode == 0 and (a.dry_run or all(s["path"] is None or os.path.exists(s["path"]) for s in steps))
        print(f"      {'ok' if ok else 'FAILED'} in {dt:.0f}s (log {os.path.join(rdir, 'run.log')})", flush=True)
        if a.dry_run:
            with open(os.path.join(rdir, "run.log")) as fh:
                print(fh.read()[-3000:])
            continue
        _append(base, dict(hash=h, overrides=ov, base_config=os.path.abspath(a.config), elec=a.elec, method=meth,
                           steps=[{k: s.get(k) for k in ("stage", "name", "in", "out_tag", "hash")} for s in steps],
                           final_tag=steps[-1].get("out_tag"), ok=bool(ok), seconds=round(dt, 1),
                           fiber_kit=_fk_version(), when=datetime.datetime.now().isoformat(timespec="seconds"),
                           keep_realign=bool(a.keep_realign)))
        if not ok and not a.keep_going:
            raise SystemExit(f"{_LP} · run {h} failed; see its log (--keep-going to continue the sweep)")
    return 0


# ───────────────────────────── scoring ─────────────────────────────
def parse_units(spec, exclude=""):
    out = set()
    for part in str(spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-"); out |= set(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return sorted(out - set(parse_units(exclude) if exclude else []))


def match_gt(res, gt_res, gt_lab, units, tol=5):
    """One-to-one match of the truth file's spikes to `res` within +-tol samples, nearest offset first
    (every truth-file spike competes, as in the reference build of gt.npy; spikes of units outside
    `units` are then dropped).  Returns gt (len(res),): the truth unit of each spike, -1 = no truth."""
    res = np.asarray(res, np.int64); gt_res = np.asarray(gt_res, np.int64); gt_lab = np.asarray(gt_lab)
    order = np.argsort(res, kind="stable"); rs = res[order]
    tr, tl = gt_res, gt_lab                    # match EVERY truth-file spike, keep the truth units after
    units = np.asarray(units)
    out = np.full(res.size, -1, np.int64); used_new = np.zeros(res.size, bool); used_gt = np.zeros(tr.size, bool)
    for d in sorted(set(range(-tol, tol + 1)), key=abs):
        tgt = tr + d
        j = np.searchsorted(rs, tgt)
        ok = (j < rs.size) & ~used_gt
        ok[ok] &= rs[j[ok]] == tgt[ok]
        i_gt = np.flatnonzero(ok); i_new = order[j[i_gt]]
        free = ~used_new[i_new]
        # duplicate targets inside one offset pass: keep the first
        _, first = np.unique(i_new[free], return_index=True)
        sel = np.flatnonzero(free)[first]
        out[i_new[sel]] = tl[i_gt[sel]]; used_new[i_new[sel]] = True; used_gt[i_gt[sel]] = True
    out[~np.isin(out, units)] = -1
    return out


def type_units(gt_res, gt_lab, units, sr, int_rate=5.0, interneurons=None):
    span = max((float(np.max(gt_res)) - float(np.min(gt_res))) / sr, 1e-9)
    types = {}
    for u in units:
        n = int(np.sum(gt_lab == u))
        if interneurons is not None:
            types[u] = "int" if u in interneurons else "pyr"
        else:
            types[u] = "int" if n / span >= int_rate else "pyr"
    return types


def truth_metrics(clu, gt, units, types, merge_min=50, merge_frac=0.2):
    """score.py semantics: per truth unit, the best single cluster by accuracy (clusters 0/1 unassigned).
    Units with no truth spike among the scored spikes (e.g. outside the chunks of a chunk run) are left out."""
    clu = np.asarray(clu); gt = np.asarray(gt)
    present = {int(x) for x in np.unique(gt[gt >= 0])}
    units = [u for u in units if int(u) in present]
    gt = np.where(np.isin(gt, units), gt, -1)
    lab = gt >= 0
    c = np.where(clu >= 2, clu, -1)[lab]; u = gt[lab]
    C, ci = np.unique(c, return_inverse=True)
    U = np.asarray(units); ui = np.searchsorted(U, u)
    M = np.zeros((U.size, C.size), np.int64); np.add.at(M, (ui, ci), 1)
    noise = C == -1; Mc = M.copy(); Mc[:, noise] = 0
    g = Mc.sum(0); nu = np.maximum(M.sum(1), 1)
    acc = Mc / np.maximum(nu[:, None] + g[None, :] - Mc, 1)
    b = acc.argmax(1) if C.size else np.zeros(U.size, int)
    rows = []
    for k in range(U.size):
        r = Mc[k]; srt = np.sort(r)[::-1]
        frag = int(np.searchsorted(np.cumsum(srt), 0.8 * nu[k]) + 1) if srt.sum() >= 0.8 * nu[k] else -1
        rows.append(dict(unit=int(U[k]), type=types[int(U[k])], n=int(M[k].sum()), acc=float(acc[k, b[k]]),
                         rec=float(Mc[k, b[k]] / nu[k]), prec=float(Mc[k, b[k]] / max(g[b[k]], 1)), frag80=frag,
                         unassigned=float(M[k, noise].sum() / nu[k]), best_cluster=int(C[b[k]])))
    merges = []
    for j in np.flatnonzero(~noise):
        col = Mc[:, j]; o = np.argsort(col)[::-1]
        if col.size > 1 and col[o[1]] >= merge_min and col[o[1]] >= merge_frac * col.sum():
            merges.append((int(C[j]), int(U[o[0]]), int(U[o[1]])))
    S = {}
    for grp, sel in (("all", lambda r: True), ("pyr", lambda r: r["type"] == "pyr"), ("int", lambda r: r["type"] == "int")):
        R = [r for r in rows if sel(r)]
        if not R:
            continue
        a_ = np.array([r["acc"] for r in R])
        S[f"{grp}_acc"] = float(a_.mean()); S[f"{grp}_n80"] = int((a_ >= 0.8).sum())
        S[f"{grp}_rec"] = float(np.mean([r["rec"] for r in R])); S[f"{grp}_prec"] = float(np.mean([r["prec"] for r in R]))
        fr = [r["frag80"] for r in R if r["frag80"] > 0]
        S[f"{grp}_frag80"] = float(np.mean(fr)) if fr else float("nan")
    ty = {r["unit"]: r["type"] for r in rows}
    S["units_scored"] = len(rows)
    S["merges"] = len(merges)
    S["merges_pyr_pyr"] = sum(1 for m in merges if ty[m[1]] == "pyr" and ty[m[2]] == "pyr")
    S["merges_with_int"] = S["merges"] - S["merges_pyr_pyr"]
    S["merge_list"] = " ".join(f"{m[1]}+{m[2]}" for m in merges)
    return S, rows


def cluster_purity(clu, gt, min_lab=30):
    out = {}
    lab = gt >= 0
    for c in np.unique(clu[clu >= 2]):
        g = gt[(clu == c) & lab]
        if g.size >= min_lab:
            v, n = np.unique(g, return_counts=True); out[int(c)] = (int(v[n.argmax()]), float(n.max() / n.sum()))
    return out


def free_metrics(clu, res, spk, sr, peak, nsamp, nchan, *, base=None, elec=None, refrac_ms=1.5, min_n=40,
                 cap=1000, rdr_thr=2.0, chunk_min=12.0, seed=0, do_waveform=True):
    """Truth-free per-cluster measures and their summary.  Returns (summary, per-cluster rows)."""
    try:
        from . import fiber_rdr as rdr, fiber_xcorr_merge as xm
    except ImportError:
        import fiber_rdr as rdr, fiber_xcorr_merge as xm
    clu = np.asarray(clu); res = np.asarray(res)
    rng = np.random.default_rng(seed)
    ev = ~_fch.is_reserved(clu)                                    # spikes this sort computed (fiber_chunks)
    ids, cnt = np.unique(clu[(clu >= 2) & ev], return_counts=True)
    S = dict(n_spikes=int(ev.sum()), n_clusters=int(ids.size), frac_c0=float(np.mean(clu[ev] == 0)) if ev.any() else 0.0,
             frac_c1=float(np.mean(clu[ev] == 1)) if ev.any() else 0.0,
             median_size=float(np.median(cnt)) if cnt.size else 0.0, frac_small=float(np.mean(cnt < min_n)) if cnt.size else 0.0)
    r = int(round(refrac_ms * 1e-3 * sr))
    order = np.argsort(clu, kind="stable"); bounds = np.searchsorted(clu[order], ids)
    rows = []
    sigma = sigma_d = None
    if do_waveform:
        nstp = nio.session_path(base, "nst", elec, variant="standard") if base else ""
        if nstp and os.path.exists(nstp):
            st = nio.read_nst(nstp); sigma, sigma_d = np.asarray(st["sigma"], float), rdr.noise_sigma_d(st["cov"], nsamp, nchan)
        else:
            sigma, sigma_d = rdr.noise_from_windows(spk, rng=rng)
    chunk = ((res - res.min()) // (chunk_min * 60.0 * sr)).astype(int) if chunk_min > 0 else None
    for u, c0, n in zip(ids, bounds, cnt):
        idx = order[c0:c0 + n]
        t = np.sort(res[idx]); isi = np.diff(t)
        row = dict(cluster=int(u), n=int(n), isi_viol=float(np.mean(isi < r)) if isi.size else 0.0)
        if do_waveform and n >= min_n:
            A, Mv, _ = rdr.cluster_profile(spk, idx, peak=peak, sigma=sigma, sigma_d=sigma_d, chunk=chunk, cap=cap, rng=rng)
            row["_A"], row["_M"] = A, Mv
            _T, _sd, giqr, rvu = xm._tmpl(spk, idx, cap=300, ref_sample=peak, rng=rng)
            row["gain_iqr"] = float(giqr); row["rvu_max"] = float(np.max(rvu))
        rows.append(row)
    big = [x for x in rows if "_A" in x]
    if big:
        A = np.array([x["_A"] for x in big]); Mv = np.array([x["_M"] for x in big])
        d0, k = rdr.fit_clean_line(A, Mv)
        Q = rdr.excess_ratio(A, Mv, d0, k)
        for x, q in zip(big, Q):
            x["rdr"] = float(np.nanmax(q)) if np.isfinite(q).any() else float("nan")
        rv = np.array([x["rdr"] for x in big]); gi = np.array([x["gain_iqr"] for x in big]); rm = np.array([x["rvu_max"] for x in big])
        S.update(rdr_median=float(np.nanmedian(rv)), rdr_flagged=float(np.nanmean(rv >= rdr_thr)), rdr_line_k=float(np.sqrt(k)),
                 gain_iqr_median=float(np.median(gi)), rvu_median=float(np.median(rm)), rvu_over_05=float(np.mean(rm > 0.5)))
    for x in rows:
        x.pop("_A", None); x.pop("_M", None)
    iv = np.array([x["isi_viol"] for x in rows if x["n"] >= min_n])
    S["isi_viol_median"] = float(np.median(iv)) if iv.size else float("nan")
    S["frac_isi_over_1pct"] = float(np.mean(iv > 0.01)) if iv.size else float("nan")
    return S, rows


def _auc(y, s):
    y = np.asarray(y, bool); s = np.asarray(s, float); ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    if y.sum() == 0 or (~y).sum() == 0:
        return float("nan")
    from scipy.stats import rankdata
    rk = rankdata(s)
    return float((rk[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def _tsv_write(path, rows, cols=None):
    cols = cols or sorted({k for r in rows for k in r})
    with open(path, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join("" if r.get(c) is None else (f"{r[c]:.6g}" if isinstance(r.get(c), float) else str(r[c]))
                               for c in cols) + "\n")


_STR_COLS = {"tag", "hash", "method", "overrides", "gt_sig", "merge_list", "type"}


def _tsv_read(path):
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        out = []
        for line in fh:
            v = line.rstrip("\n").split("\t")
            r = {}
            for k, x in zip(hdr, v):
                if k in _STR_COLS or x == "":
                    r[k] = x
                    continue
                try:
                    r[k] = float(x)
                except ValueError:
                    r[k] = x
            out.append(r)
        return out


def cmd_score(a):
    sp = sy.resolve_session_params(a.session, a.elec)
    base, SR, NS, NC, PK = sp["base"], sp["sr"], sp["nsamp"], sp["nchan"], sp["peak"]
    res = nio.read_res(base, a.elec)
    spk = nio.open_spk_file(nio.session_path(base, "spk", a.elec, variant=a.spk_method), NS, NC)
    reg = read_registry(base)
    targets = []                                                   # (tag, method, hash, overrides)
    for r in reg:
        if r.get("ok") and r.get("final_tag"):
            targets.append((r["final_tag"], r.get("method") or a.clu_method, r["hash"], r.get("overrides", {})))
    for t in [x for x in (a.stages or "").split(",") if x]:
        targets.append((t, a.clu_method, "baseline", {}))
    seen, uniq = set(), []
    for t in targets:
        if t[0] not in seen:
            seen.add(t[0]); uniq.append(t)
    gt = types = units = None
    if a.gt_clu:
        _, glab = nio.read_clu_file(a.gt_clu)
        gres = nio.read_res_file(a.gt_res)
        units = parse_units(a.gt_units, a.gt_exclude) if a.gt_units else sorted(int(u) for u in np.unique(glab) if u > 1)
        gt = match_gt(res, gres, glab, units, a.gt_tol)
        types = type_units(gres, glab, units, SR, a.int_rate, parse_units(a.interneurons) if a.interneurons else None)
        print(f"{_LP} · truth: {len(units)} units ({sum(t == 'int' for t in types.values())} interneurons: "
              f"{[u for u in units if types[u] == 'int']}), {int((gt >= 0).sum())} of {res.size} spikes matched")
    bd = bench_dir(base)
    old = {r["tag"]: r for r in _tsv_read(os.path.join(bd, "metrics.tsv"))}
    gsig = _h([a.gt_clu, a.gt_res, units, a.gt_tol, a.int_rate, a.interneurons]) if a.gt_clu else "none"
    summaries, unit_rows, clus_rows = [], [], []
    for tag, meth, h, ov in uniq:
        if tag in old and old[tag].get("gt_sig") == gsig and not a.force:
            summaries.append(old[tag]); print(f"  {tag}: cached"); continue
        try:
            _, clu = nio.read_clu_at(base, a.elec, variant=meth, tag=tag, n_spikes=res.size)
        except Exception as e:                                     # noqa: BLE001
            print(f"  {tag}: cannot read ({e})"); continue
        t0 = time.time()
        S = dict(tag=tag, hash=h, method=meth, overrides=json.dumps(ov, sort_keys=True), gt_sig=gsig)
        rsv = _fch.is_reserved(clu)
        S["chunks_left_out"] = int(len(np.unique(clu[rsv])))
        F, crow = free_metrics(clu, res, spk, SR, PK, NS, NC, base=base, elec=a.elec, do_waveform=not a.no_waveform)
        S.update(F)
        if gt is not None:
            gt_ev = np.where(rsv, -1, gt)                         # truth spikes in chunks not computed: not scored
            T, urows = truth_metrics(clu, gt_ev, units, types)
            S.update(T)
            for r in urows:
                unit_rows.append(dict(tag=tag, **r))
            pur = cluster_purity(clu, gt_ev)
            for r in crow:
                if r["cluster"] in pur:
                    r["truth_unit"], r["purity"] = pur[r["cluster"]]
            lab = [r for r in crow if "purity" in r]
            y = [r["purity"] < 0.9 for r in lab]
            for m in ("rdr", "gain_iqr", "rvu_max", "isi_viol"):
                S[f"auc_{m}"] = _auc(y, [r.get(m, np.nan) for r in lab])
        for r in crow:
            clus_rows.append(dict(tag=tag, **r))
        summaries.append(S)
        print(f"  {tag}: {F['n_clusters']} clusters" + (f", pyr acc {S.get('pyr_acc', float('nan')):.3f}, "
              f"merges {S.get('merges')} ({S.get('merges_pyr_pyr')} pyr-pyr)" if gt is not None else "")
              + f" | {time.time() - t0:.0f}s", flush=True)
    done = {r["tag"] for r in summaries}
    keep = [r for r in old.values() if r["tag"] not in done] + summaries
    _tsv_write(os.path.join(bd, "metrics.tsv"), keep)
    if unit_rows:
        prev = [r for r in _tsv_read(os.path.join(bd, "units.tsv")) if r["tag"] not in {u["tag"] for u in unit_rows}]
        _tsv_write(os.path.join(bd, "units.tsv"), prev + unit_rows)
    if clus_rows:
        prev = [r for r in _tsv_read(os.path.join(bd, "clusters.tsv")) if r["tag"] not in {u["tag"] for u in clus_rows}]
        _tsv_write(os.path.join(bd, "clusters.tsv"), prev + clus_rows)
    print(f"{_LP} · metrics -> {os.path.join(bd, 'metrics.tsv')}")
    return 0


# ───────────────────────────── report ─────────────────────────────
_KEY_METRICS = ["pyr_acc", "pyr_rec", "pyr_prec", "pyr_frag80", "merges_pyr_pyr", "int_acc", "n_clusters",
                "frac_c0", "frac_c1", "rdr_flagged", "rvu_over_05", "frac_isi_over_1pct"]


def _png(fig):
    import base64
    import io
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def cmd_report(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sp = sy.resolve_session_params(a.session, a.elec)
    bd = bench_dir(sp["base"])
    M = _tsv_read(os.path.join(bd, "metrics.tsv"))
    if not M:
        raise SystemExit(f"{_LP} · no metrics yet: run `fiber-bench score` first")
    U = _tsv_read(os.path.join(bd, "units.tsv")); C = _tsv_read(os.path.join(bd, "clusters.tsv"))
    for r in M:
        r["_ov"] = json.loads(r.get("overrides") or "{}") if isinstance(r.get("overrides"), str) else {}
    params = sorted({k for r in M for k in r["_ov"]})
    has_truth = any(np.isfinite(_num(r.get("pyr_acc"))) for r in M)
    figs = []
    if has_truth:                                                  # accuracy vs false merges
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for r in M:
            x, y = _num(r.get("merges_pyr_pyr")), _num(r.get("pyr_acc"))
            base_ = r.get("hash") == "baseline"
            ax.scatter(x, y, s=40, marker="s" if base_ else "o", color="#555" if base_ else "#2a6fdb")
            ax.annotate(r["tag"] if base_ else str(r.get("hash")), (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("pyramidal-pyramidal merges (false)"); ax.set_ylabel("pyramidal accuracy (mean best-cluster)")
        ax.set_title("accuracy vs false merges (up-left is better)"); ax.grid(alpha=0.3)
        figs.append(("Accuracy against false merges", _png(fig)))
    if params:                                                     # parameter effects
        show = (["pyr_acc", "pyr_rec", "pyr_prec", "merges_pyr_pyr"] if has_truth else []) + ["n_clusters", "rdr_flagged"]
        fig, axs = plt.subplots(len(params), len(show), figsize=(2.6 * len(show), 2.2 * len(params)), squeeze=False)
        for i, p in enumerate(params):
            R = [r for r in M if p in r["_ov"]]
            xs = [r["_ov"][p] for r in R]
            num = all(isinstance(x, (int, float)) for x in xs)
            cats = sorted({str(x) for x in xs})
            for j, m in enumerate(show):
                ax = axs[i][j]
                X = [float(x) if num else cats.index(str(x)) for x in xs]
                ax.scatter(X, [_num(r.get(m)) for r in R], s=18, color="#2a6fdb")
                if not num:
                    ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats, fontsize=7, rotation=30)
                if i == len(params) - 1 or True:
                    ax.set_xlabel(p.replace("FK_", ""), fontsize=7)
                if j == 0:
                    ax.set_ylabel("")
                ax.set_title(m, fontsize=8); ax.tick_params(labelsize=7); ax.grid(alpha=0.3)
        fig.tight_layout()
        figs.append(("Parameter effects (each point one run; other parameters may differ)", _png(fig)))
    if U:                                                          # per-unit accuracy heatmap
        tags = [r["tag"] for r in M]
        units = sorted({int(_num(u["unit"])) for u in U}, key=lambda u: (next((x["type"] for x in U if int(_num(x["unit"])) == u), ""), u))
        Z = np.full((len(tags), len(units)), np.nan)
        for u in U:
            if u["tag"] in tags:
                Z[tags.index(u["tag"]), units.index(int(_num(u["unit"])))] = _num(u["acc"])
        fig, ax = plt.subplots(figsize=(0.32 * len(units) + 3, 0.28 * len(tags) + 1.5))
        im = ax.imshow(Z, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ty = {int(_num(x["unit"])): x["type"] for x in U}
        ax.set_xticks(range(len(units))); ax.set_xticklabels([f"{u}{'*' if ty.get(u) == 'int' else ''}" for u in units], fontsize=7)
        ax.set_yticks(range(len(tags))); ax.set_yticklabels(tags, fontsize=7)
        ax.set_xlabel("truth unit (* interneuron)"); fig.colorbar(im, ax=ax, label="best-cluster accuracy")
        figs.append(("Per-unit accuracy", _png(fig)))
    if C and has_truth:                                            # truth-free measures against purity
        fig, axs = plt.subplots(1, 3, figsize=(11, 3.3))
        for ax, m in zip(axs, ("rdr", "gain_iqr", "rvu_max")):
            P = [(_num(r.get(m)), _num(r.get("purity"))) for r in C if np.isfinite(_num(r.get("purity")))]
            if P:
                x, y = np.array(P).T
                ax.scatter(x, y, s=6, alpha=0.4, color="#2a6fdb")
            ax.set_xlabel(m); ax.set_ylabel("truth purity"); ax.grid(alpha=0.3)
        fig.tight_layout()
        figs.append(("Truth-free measures against cluster purity (all scored runs)", _png(fig)))
    cols = ["tag", "hash"] + params + [m for m in _KEY_METRICS if any(m in r for r in M)] + \
           [c for c in ("auc_rdr", "auc_gain_iqr", "auc_rvu_max", "auc_isi_viol", "merge_list") if any(c in r for r in M)]
    def cell(r, c):
        if c in params:                                            # parameter values exactly as given
            return str(r["_ov"].get(c, ""))
        v = r.get(c, "")
        if isinstance(v, float):
            if not np.isfinite(v):
                return ""
            return str(int(v)) if v.is_integer() else f"{v:.3f}"
        return str(v)
    rows_html = "\n".join("<tr>" + "".join(f"<td>{cell(r, c)}</td>" for c in cols) + "</tr>" for r in M)
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>fiber-bench report</title>
<style>body{{font-family:system-ui,sans-serif;margin:16px;background:#fff;color:#222}} table{{border-collapse:collapse;font-size:12px}}
td,th{{border:1px solid #ddd;padding:3px 6px;text-align:right}} th{{cursor:pointer;background:#f3f3f3;position:sticky;top:0}}
td:first-child{{text-align:left}} img{{max-width:100%}} h2{{font-size:16px;margin-top:28px}}</style></head><body>
<h1 style="font-size:20px">fiber-bench: {os.path.basename(sp['base'])} group {a.elec}</h1>
<p>{len(M)} sort(s); parameters that differ between runs: {', '.join(params) or 'none'}.  Click a column header to sort.</p>
<table id="t"><thead><tr>{''.join(f'<th>{c}</th>' for c in cols)}</tr></thead><tbody>
{rows_html}</tbody></table>
{''.join(f'<h2>{t}</h2><img src="{s}">' for t, s in figs)}
<script>
document.querySelectorAll('#t th').forEach((th,i)=>th.onclick=()=>{{const tb=th.closest('table').tBodies[0];
const rows=[...tb.rows];const asc=th.dataset.asc!=='1';th.dataset.asc=asc?'1':'0';
rows.sort((a,b)=>{{const x=a.cells[i].innerText,y=b.cells[i].innerText;const nx=parseFloat(x),ny=parseFloat(y);
const r=(isNaN(nx)||isNaN(ny))?x.localeCompare(y):nx-ny;return asc?r:-r}});rows.forEach(r=>tb.appendChild(r))}});
</script></body></html>"""
    out = a.out or os.path.join(bd, "report.html")
    with open(out, "w") as fh:
        fh.write(html)
    print(f"{_LP} · report -> {out}")
    return 0


def cmd_list(a):
    sp = sy.resolve_session_params(a.session, a.elec)
    for r in read_registry(sp["base"]):
        ov = ", ".join(f"{k}={v}" for k, v in (r.get("overrides") or {}).items()) or "(base)"
        sec = r.get("seconds")
        print(f"{r['hash']}  {'ok ' if r.get('ok') else 'BAD'}  {sec if sec is not None else '-':>7}s  {r.get('final_tag')}  {ov}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fiber-bench",
                                 description="Run, score and compare pipeline-parameter variants under a config hash.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("session"); p.add_argument("elec", type=int)
        p.add_argument("--clu-method", default="stderiv_C5_D34", help="method token of the scored .clu (default stderiv_C5_D34)")
    p = sub.add_parser("run", help="run the plan (up to --upto) for each parameter variant, under hashed stage tags")
    common(p)
    p.add_argument("--config", required=True, help="base config with a `pipeline:` plan (e.g. fiber-kit-exp.yaml)")
    p.add_argument("--set", action="append", default=[], help="K=V override for every variant (repeatable)")
    p.add_argument("--grid", action="append", default=[], help="K=v1,v2,... sweep axis (repeatable; cartesian product)")
    p.add_argument("--sweep", default=None, help="yaml: a list of override dicts, or a dict of value lists")
    p.add_argument("--upto", default=None, help="last plan step to run (name or stage; default the first fiber-session)")
    p.add_argument("--keep-realign", action="store_true", help="keep FK_SESSION_REALIGN (rewrites .res/.spk in place!)")
    p.add_argument("--allow-inplace", action="store_true", help="allow in-place fiber-realign steps in the run")
    p.add_argument("--force", action="store_true", help="re-run steps whose output exists")
    p.add_argument("--keep-going", action="store_true", help="continue the sweep after a failed run")
    p.add_argument("--dry-run", action="store_true", help="print the stage commands only")
    p.add_argument("--chunks", default=None, help="run only these 0-based chunks (FK_CHUNKS, e.g. 14-16); the "
                                                  "other chunks' spikes go to reserved clusters and are not scored")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("score", help="metrics for every registered run (and --stages baselines)")
    common(p)
    p.add_argument("--stages", default="", help="extra stage tags to score as baselines (comma list)")
    p.add_argument("--spk-method", default="standard", help="waveform axis for the truth-free measures")
    p.add_argument("--gt-clu", default=None, help="ground-truth .clu (e.g. the old curation)")
    p.add_argument("--gt-res", default=None, help="its .res")
    p.add_argument("--gt-units", default="", help="truth units, e.g. 2-32 (default every unit > 1)")
    p.add_argument("--gt-exclude", default="", help="units to leave out of the truth, e.g. 7,18")
    p.add_argument("--gt-tol", type=int, default=5, help="match tolerance, samples")
    p.add_argument("--int-rate", type=float, default=5.0, help="truth units firing >= this (Hz) are interneurons")
    p.add_argument("--interneurons", default="", help="explicit interneuron unit list (overrides --int-rate)")
    p.add_argument("--no-waveform", action="store_true", help="skip the waveform measures (rdr, gain IQR, RVU)")
    p.add_argument("--force", action="store_true", help="re-score tags already in metrics.tsv")
    p.set_defaults(fn=cmd_score)
    p = sub.add_parser("report", help="HTML report of the scored runs")
    common(p)
    p.add_argument("--out", default=None, help="output HTML (default <session dir>/fk-bench/report.html)")
    p.set_defaults(fn=cmd_report)
    p = sub.add_parser("list", help="list the registered runs")
    common(p)
    p.set_defaults(fn=cmd_list)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
