"""Clean-core metrics (fiber_bench.core_metrics), fiber-gt-review's disagreement rows, and the shipped
core-search spec.

  * core_metrics: an unsorted spike costs only coverage; a curator-noise spike in a core is reported,
    not charged; two cells in one cluster cut core precision; a cell split in two cuts link;
  * cross_refractory / self_refractory: ~0 for one refractory train split in two, ~1 for two
    independent trains, NaN without power;
  * review(): a merged pair of cells -> "sort mixes two cells"; a cell split at random -> "sort did not
    link" by refractory evidence; a cell split in time -> "disjoint in time"; a curated unit that is two
    cells -> `unit`; an identical sort -> no disagreement rows (negative control);
  * plans/tune/core-search.yaml loads and passes check_knobs against fiber-kit-exp.yaml;
    plans/tune/hybrid-search.yaml likewise against fiber-kit-best.yaml, searches the realign gate
    first with search_realign on (0.67.21), and keeps the hybrid on/off control last.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fiber_kit import fiber_bench as fb, fiber_gt_review as gr, fiber_tune as ft   # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
fails = 0
def check(ok, what):
    global fails
    print(("  ok:   " if ok else "  FAIL: ") + what); fails += not ok


SR = 20000.0
rng = np.random.default_rng(3)


def train(rate, T, dead_ms=2.0):
    """Poisson train with a dead time (samples)."""
    isi = rng.exponential(1.0 / rate, int(rate * T * 1.5) + 10) + dead_ms * 1e-3
    t = np.cumsum(isi); t = t[t < T]
    return np.round(t * SR).astype(np.int64)


def main():
    # ── core_metrics ──
    clu = np.array([2] * 90 + [0] * 10 + [2] * 5 + [3] * 50 + [4] * 50 + [5] * 100)
    gt = np.array([10] * 90 + [10] * 10 + [0] * 5 + [11] * 50 + [11] * 50 + [12] * 100)
    S, R = fb.core_metrics(clu, gt, [10, 11, 12], {10: "pyr", 11: "pyr", 12: "pyr"})
    r = {x["unit"]: x for x in R}
    check(r[10]["core_prec"] == 1.0 and abs(r[10]["coverage"] - 0.9) < 1e-9 and r[10]["link"] == 1.0,
          "unsorted spikes cost only coverage (core precision 1, link 1, coverage 0.9)")
    check(abs(r[10]["noise_share"] - 5 / 95) < 1e-9, "curator-noise spikes in the core are reported, not charged")
    check(r[11]["link"] == 0.5 and not r[11]["clean"], "a cell split over two clusters: link 0.5, not clean")
    check(r[10]["clean"] and r[12]["clean"] and S["all_clean"] == 2, "clean = precision >= 0.95 and link >= 0.8")
    clu2 = clu.copy(); clu2[clu2 == 5] = 2                  # cell 12 merged into cell 10's cluster
    S2, R2 = fb.core_metrics(clu2, gt, [10, 11, 12], {10: "pyr", 11: "pyr", 12: "pyr"})
    check(abs({x["unit"]: x for x in R2}[10]["core_prec"] - 90 / 190) < 1e-9, "two cells in one cluster cut core precision")
    check(S2["core_purity"] < S["core_purity"], "core_purity falls with the merge")

    # ── refractory statistics ──
    T = 600.0
    a = train(8.0, T)
    half = rng.random(a.size) < 0.5
    x1, e1 = gr.cross_refractory(a[half], a[~half], SR)
    b = train(8.0, T)
    x2, e2 = gr.cross_refractory(a, b, SR)
    check(x1 < 0.1 and e1 >= 3, f"one refractory train split in two: cross ratio {x1:.2f} (expected {e1:.0f} pairs)")
    check(0.6 < x2 < 1.5, f"two independent trains: cross ratio {x2:.2f}")
    xs, _ = gr.self_refractory(a, SR); xm, _ = gr.self_refractory(np.sort(np.r_[a, b]), SR)
    check(xs < 0.1 and xm > 0.3, f"self refractory: one cell {xs:.2f}, two cells {xm:.2f}")
    x3, _ = gr.cross_refractory(train(0.2, T), train(0.2, T), SR)
    check(x3 != x3, "two sparse trains: no power -> NaN")

    # ── review() on a synthetic session ──
    cells = {u: train(rate, T) for u, rate in ((2, 8.0), (3, 8.0), (4, 8.0), (5, 6.0))}
    both = np.sort(np.r_[train(6.0, T), train(6.0, T)])               # a curated unit that is two cells
    cells[6] = both
    noise = train(3.0, T)
    res = np.concatenate([*cells.values(), noise]); gt = np.concatenate([np.full(t.size, u) for u, t in cells.items()]
                                                                         + [np.zeros(noise.size, np.int64)])
    o = np.argsort(res, kind="stable"); res, gt = res[o], gt[o]
    units = [2, 3, 4, 5, 6]
    rows0, _ = gr.review(gt.copy(), gt, units, res, SR)
    check([r["kind"] for r in rows0] == ["unit"] and rows0[0]["unit"] == 6,
          "identical sort: no disagreements; the two-cell curated unit is flagged (`unit`)")
    sort = gt.copy()
    sort[gt == 3] = 2                                                 # two cells merged
    i4 = np.flatnonzero(gt == 4); sort[i4[rng.random(i4.size) < 0.5]] = 40       # concurrent split
    i5 = np.flatnonzero(gt == 5); sort[i5[res[i5] > np.median(res[i5])]] = 50    # split in time
    rows, sets = gr.review(sort, gt, units, res, SR)
    byk = {}
    for r_ in rows:
        byk.setdefault(r_["kind"], []).append(r_)
    check(any(r_["verdict"] == "sort mixes two cells" and {r_["unit"], r_["unit_b"]} == {2, 3} for r_ in byk.get("mixed", [])),
          "merged pair -> sort mixes two cells")
    sp = {r_["unit"]: r_["verdict"] for r_ in byk.get("split", [])}
    check(sp.get(4) == "sort did not link", f"concurrent split of one cell -> sort did not link ({sp.get(4)})")
    check(sp.get(5, "").startswith("sort did not link (disjoint"), f"split in time -> disjoint ({sp.get(5)})")
    ids = [r_["set_a"] for r_ in rows if r_.get("set_a") is not None] + [r_["set_b"] for r_ in rows if r_.get("set_b") is not None]
    check(set(ids) <= set(sets) and min(sets) >= 2, "every row's spike sets are review-.clu clusters (ids from 2)")

    # ── fiber-tune: the core metrics are scored, the metric set is versioned, rescore re-scores ──
    import tempfile
    from fiber_kit import neuro_io as nio
    td = tempfile.mkdtemp(); base = os.path.join(td, "S")
    nio.write_res_file(f"{base}.res.6", res)
    nio.write_clu_file(f"{base}.gtc", gt); nio.write_res_file(f"{base}.gtr", res)
    Tr = ft.Truth(dict(clu=f"{base}.gtc", res=f"{base}.gtr"), SR)
    check(Tr.id.endswith("|" + ft.SCORE_VERSION), "the truth id carries the metric-set version")
    M = ft.score_labels(sort, Tr.matched(res), Tr, Tr.matched_all(res))
    check(all(k in M for k in ("all_core_prec", "all_link", "all_cov", "all_clean", "core_purity")),
          "score_labels reports the clean-core metrics")
    nio.write_clu(base, 6, sort, variant="stderiv", tag="t_abc")
    tn = object.__new__(ft.Tuner)
    tn.base, tn.elec, tn.method, tn.truth, tn.cfg_search = base, 6, "stderiv", Tr, {}
    tn.dir = td; tn.log = lambda *x: None; tn.trials = []
    tn.objective = ft.compile_objective("all_link")
    tn.write_tables = lambda: None
    tn._record(dict(phase="search", trial=1, hash="abc", tag="t_abc", ok=True, overrides={}, note="old",
                    metrics={"all_acc": 0.5}, objective=0.5, truth="old-truth"))
    tn.rescore()
    cur = [r for r in tn.trials if tn._current(r)]
    check(len(cur) == 1 and "all_link" in cur[0]["metrics"] and cur[0]["objective"] == cur[0]["metrics"]["all_link"],
          "rescore: a trial scored under an older truth / metric set is scored again from its .clu")
    with open(os.path.join(td, "trials.jsonl")) as fh:
        check(len(fh.readlines()) == 2, "rescore appends the new record (the old one stays in trials.jsonl)")

    # ── the shipped spec ──
    spec = ft.load_spec(os.path.join(ROOT, "plans", "tune", "core-search.yaml"))
    cfg = fb.load_config(spec["config"])
    plans = {k: ft.derive_plan(cfg["pipeline"], v, spec["_here"]) for k, v in spec["plans"].items()}
    try:
        ft.check_knobs(spec, cfg, plans); okk = True
    except SystemExit as e:
        okk = False; print("   ", e)
    check(okk, "core-search.yaml: every knob is in fiber-kit-exp.yaml, every step parameter names a step")
    spec = ft.load_spec(os.path.join(ROOT, "plans", "tune", "hybrid-search.yaml"))
    cfg_h = fb.load_config(spec["config"])
    plans_h = {k: ft.derive_plan(cfg_h["pipeline"], v, spec["_here"]) for k, v in spec["plans"].items()}
    try:
        ft.check_knobs(spec, cfg_h, plans_h); okk = True
    except SystemExit as e:
        okk = False; print("   ", e)
    check(okk, "hybrid-search.yaml: every knob is in fiber-kit-best.yaml, every step parameter names a step")
    check(spec["config"].endswith("fiber-kit-best.yaml") and not any(st.get("name") == "psame-chunk" for st in plans_h["best"]),
          "hybrid-search.yaml starts from the best profile (the previous fk-tune winner, psame-chunk dropped)")
    kn = [k for st in spec["stages"] for k in st["knobs"]]
    check(all(k in kn for k in ("FK_SESSION_FEAT_HYBRID", "FK_SESSION_NEIGH_PASSES", "FK_SESSION_NEIGH_K",
                                "FK_SESSION_NEIGH_COS", "FK_SESSION_RKK_PTS_PER_START", "FK_SESSION_RKK_STARTS",
                                "FK_SESSION_RKK_MIN", "FK_SESSION_RKK_REALIGN_ITERS", "FK_PSAME_ALIGN_PAIRS")),
          "hybrid-search.yaml searches Block N, the rkk retry, the rkk effort knobs and the psame pair alignment")
    check(ft._same(spec["start"].get("FK_SESSION_FEAT_HYBRID"), 1) and ft._same(spec["start"].get("FK_SESSION_HYBRID_SCALE"), 2.0)
          and ft._same(spec["start"].get("FK_SESSION_HYBRID_PCS"), 3),
          "hybrid-search.yaml starts from the clean run's winner (hybrid on, scale 2, 3 PCs)")
    check(spec["stages"][0].get("name") == "realign" and spec["stages"][1].get("name") == "neigh"
          and spec["stages"][-1].get("name") == "hybrid"
          and list(spec["stages"][-1]["knobs"]) == ["FK_SESSION_FEAT_HYBRID"],
          "hybrid-search.yaml searches the realign gate first, then Block N, and keeps the hybrid control last")
    check(spec["search_realign"] is True
          and all(k in kn for k in ("FK_REALIGN_MIN_SCORE", "FK_REALIGN_MAX_SHIFT")),
          "hybrid-search.yaml searches the realign knobs AND runs the realign in the search -- they are "
          "no-ops without search_realign (the search plan has no fiber-realign step, and fiber-session's "
          "rkk realign loop uses fl.align_xcorr and reads none of them)")
    spec = ft.load_spec(os.path.join(ROOT, "plans", "tune", "core-search.yaml"))
    f = ft.compile_objective(spec["objective"])
    hi = f(dict(pyr_link=0.9, pyr_core_prec=0.96, pyr_cov=0.6, merges_pyr_pyr=0))
    lo = f(dict(pyr_link=0.9, pyr_core_prec=0.90, pyr_cov=0.9, merges_pyr_pyr=0))
    check(hi > lo, "core objective: precision below 0.95 costs more than extra coverage gains")
    print("OK (0 failures)" if not fails else f"FAILED ({fails})")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
