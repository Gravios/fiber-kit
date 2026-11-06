#!/usr/bin/env python3
"""fiber-tune: the search harness's pieces, without running the pipeline.

  1. The objective: arithmetic on metric names; a missing name or NaN gives -inf; anything but
     arithmetic (a call outside min/max/abs, an attribute) is refused.
  2. The search plan: cut at score_at, in-place fiber-realign steps dropped (they rewrite the shared
     .res/.spk), steps with `out:` kept; the default score_at of the shipped exp plan is psame.
  3. Override values survive fiber-bench's --set round trip with their types, so the trial's hash (computed
     here) is the hash fiber-bench run gives its outputs -- otherwise the scored tag would not exist.
  4. Coordinate search on a known objective whose best A depends on B reaches the optimum through a
     second pass; a resumed search re-uses every trial (no new evaluation).
  5. A confirm workspace copies .res/.spk/.fet, symlinks everything else, leaves stage outputs out.
  6. Scoring: perfect labels score 1 everywhere; spikes in reserved (not-run) chunks are not scored; a
     merge of two units is counted.
Exits nonzero on failure.
"""
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from fiber_kit import fiber_tune as ft, fiber_bench as fb, fiber_chunks as fch  # noqa: E402

fails = 0


def check(name, ok, detail=""):
    global fails
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail and not ok else ""))
    fails += 0 if ok else 1


# 1. objective
f = ft.compile_objective("all_acc - 0.01*merges + max(ari, 0)")
check("objective value", abs(f(dict(all_acc=0.8, merges=3, ari=0.5)) - (0.8 - 0.03 + 0.5)) < 1e-12)
check("missing metric -> -inf", f(dict(all_acc=0.8, merges=3)) == float("-inf"))
check("NaN metric -> -inf", f(dict(all_acc=float("nan"), merges=0, ari=0.1)) == float("-inf"))
for bad in ("__import__('os')", "all_acc.real", "[all_acc][0]", "open('x')"):
    try:
        ft.compile_objective(bad); ok = False
    except ValueError:
        ok = True
    check(f"objective refuses {bad!r}", ok)

# 2. search plan
exp = fb.load_config(os.path.join(os.path.dirname(ft.__file__), "fiber-kit-exp.yaml"))
plan = exp["pipeline"]
sa = ft.default_score_at(plan)
check("default score_at of the exp plan is psame", sa == "psame", sa)
sp = ft.search_plan(plan, sa)
check("search plan ends at score_at", (sp[-1].get("name") or sp[-1]["stage"]) == "psame")
check("search plan has no in-place realign", not any(s["stage"] == "fiber-realign" and not s.get("out") for s in sp))
check("search plan keeps the clustering steps", [s.get("name") for s in sp if s.get("out")] ==
      [s.get("name") for s in plan[:[i for i, s in enumerate(plan) if s.get("name") == "psame"][0] + 1] if s.get("out")])
cfg = dict(exp); cfg["pipeline"] = sp
_, steps, _ = fb.build_run(cfg, "psame")                    # must not raise (no in-place step left)
check("fiber-bench accepts the search plan without --allow-inplace", steps[-1]["out_tag"].startswith("exp_psame_"))

# 3. override round trip -> identical hash
vals = {"FK_SESSION_COARSE_DR": 0.09, "FK_SESSION_INCL_ASSIGN": 0, "FK_REALIGN_VARIANT": "stderiv_C5_D34",
        "FK_PSAME_THETA": 0.5, "FK_SESSION_REALIGN": "off", "FK_CONS_MODE": ""}
back = fb.parse_overrides([f"{k}={ft._fmt(v)}" for k, v in vals.items()], [], None)[0]
check("override values keep their types through --set", all(type(back[k]) is type(v) and back[k] == v for k, v in vals.items()),
      str({k: (back[k], vals[k]) for k in vals if back[k] != vals[k] or type(back[k]) is not type(vals[k])}))
c1 = dict(cfg); c1.update(vals); c1["FK_CHUNKS"] = "2-3"
c2 = dict(cfg); c2.update(back); c2["FK_CHUNKS"] = str("2-3")
check("trial hash == fiber-bench run hash", fb.build_run(c1, "psame")[2] == fb.build_run(c2, "psame")[2])

# 4. coordinate search on a known objective
with tempfile.TemporaryDirectory() as td:
    t = object.__new__(ft.Tuner)
    t.spec = dict(name="t", config="base.yaml", objective="x", set={}, start={}, passes=3, min_gain=1e-6, chunks="",
                  stages=[dict(name="session", knobs={"A": [0.05, 0.09, 0.15]}), dict(name="merge", knobs={"B": [0, 1, 2]})])
    t.cfg_search = {"A": 0.15, "B": 0}; t.cfg_full = {"A": 0.15, "B": 0, "pipeline": []}
    t.dir = td; t.trials = []; t.log = lambda *_: None; t.score_at = "x"
    calls = []

    def fake(ov, phase="search", note=""):
        a = ov.get("A", 0.15); b = ov.get("B", 0)
        h = json.dumps(dict(A=a, B=b), sort_keys=True)
        for r in t.trials:
            if r["hash"] == h:
                return r
        calls.append(h)
        # interaction: the best A depends on B (0.09 while B < 2, 0.05 at B = 2), so the optimum (0.05, 2)
        # needs a second pass -- the first sets A = 0.09, then B = 2, then the second moves A
        obj = -100 * (a - (0.09 if b < 2 else 0.05)) ** 2 - (b - 2) ** 2
        r = dict(phase=phase, hash=h, ok=True, objective=obj, overrides=dict(ov), metrics={}, note=note)
        t.trials.append(r); return r
    t.evaluate = fake
    cur, best = t.search()
    check("coordinate search reaches the optimum through a second pass", cur == {"A": 0.05, "B": 2}, str(cur))
    n = len(calls); t.search()
    check("a resumed search evaluates nothing new", len(calls) == n, f"{len(calls) - n} new")
    with open(os.path.join(td, "best.yaml")) as fh:
        check("best.yaml written with the best overrides", "A: 0.05" in fh.read())

    # failed trials (-inf) never move the search, even from a failed start
    t.trials = []; t.cfg_search = {"A": 0.15, "B": 0}
    t.evaluate = lambda ov, phase="search", note="": dict(phase=phase, hash=json.dumps(ov, sort_keys=True), ok=False,
                                                        objective=float("-inf"), overrides=dict(ov), metrics={}, note=note)
    cur, _ = t.search()
    check("failed trials never move the search", cur == {}, str(cur))

# 5. workspace
with tempfile.TemporaryDirectory() as td:
    src = os.path.join(td, "sess"); os.makedirs(src)
    names = ["s.yaml", "s.res.stderiv_S3.6", "s.spk.stderiv_C5.6", "s.fet.stderiv_C5_D34.6", "s.offsets.6.npy",
             "s.fil", "s.nst.stderiv_C5.6", "p.probe", "s.clu.stderiv_C5_D34.6.fs_clean", "s.fibers.stderiv_C5_D34.6.x.npz"]
    for n_ in names:
        with open(os.path.join(src, n_), "w") as fh:
            fh.write(n_)
    t = object.__new__(ft.Tuner); t.base = os.path.join(src, "s"); t.dir = os.path.join(td, "tune")
    ws, nb = t.make_workspace("abc")
    kind = {n_: ("link" if os.path.islink(os.path.join(ws, n_)) else "copy" if os.path.exists(os.path.join(ws, n_)) else "none")
            for n_ in names}
    want = {"s.yaml": "link", "s.res.stderiv_S3.6": "copy", "s.spk.stderiv_C5.6": "copy", "s.fet.stderiv_C5_D34.6": "copy",
            "s.offsets.6.npy": "copy", "s.fil": "link", "s.nst.stderiv_C5.6": "link", "p.probe": "link",
            "s.clu.stderiv_C5_D34.6.fs_clean": "none", "s.fibers.stderiv_C5_D34.6.x.npz": "none"}
    check("workspace: copies .res/.spk/.fet/offsets, links the rest, skips stage outputs", kind == want,
          str({k: (kind[k], want[k]) for k in want if kind[k] != want[k]}))
    with open(os.path.join(ws, "s.res.stderiv_S3.6"), "w") as fh:
        fh.write("moved")
    with open(os.path.join(src, "s.res.stderiv_S3.6")) as fh:
        check("writing the workspace .res leaves the session's alone", fh.read() == "s.res.stderiv_S3.6")

# 6. scoring
rng = np.random.default_rng(0)
gt = rng.integers(2, 6, 3000); gt[:200] = -1                 # units 2..5, some spikes without truth
tr = object.__new__(ft.Truth); tr.units = [2, 3, 4, 5]; tr.types = dict.fromkeys(tr.units, "pyr")
M = ft.score_labels(np.where(gt >= 0, gt + 10, 1), gt, tr)
check("perfect labels score 1", all(abs(M[k] - 1) < 1e-12 for k in ("all_acc", "ari", "purity", "completeness")) and M["merges"] == 0,
      str({k: M[k] for k in ("all_acc", "ari", "purity", "completeness", "merges")}))
clu = np.where(gt >= 0, gt + 10, 1); clu[1500:] = fch.reserved_id(1)     # second half: a chunk not run
M2 = ft.score_labels(clu, gt, tr)
check("reserved-chunk spikes are not scored", abs(M2["all_acc"] - 1) < 1e-12 and M2["n_clusters"] == 4)
clu = np.where(gt >= 0, gt + 10, 1); clu[gt == 3] = 12                   # unit 3 merged into unit 2's cluster
M3 = ft.score_labels(clu, gt, tr)
check("a merge is counted and lowers purity", M3["merges"] == 1 and M3["purity"] < 0.8, f"{M3['merges']}, {M3['purity']:.3f}")
print("OK (0 failures)" if not fails else f"FAILED ({fails})")
sys.exit(1 if fails else 0)
