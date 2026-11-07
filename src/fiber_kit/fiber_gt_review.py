#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════════════════════
#  fiber-gt-review -- where a sort and a curated ground truth disagree, say which one is likely wrong,
#  as a list to step through in Klusters.
#
#  A curation is not perfect, and every score against it inherits its mistakes.  This compares one sort
#  (a .clu stage) with the curated .clu, finds every place the two disagree by a meaningful number of
#  spikes, and tests the two spike sets involved with evidence that does not depend on either labelling:
#
#    cos    roll-cosine (+-4 samples) of the two sets' mean RAW waveforms (.spk.standard)
#    xref   cross-refractory ratio: pairs of spikes, one from each set, (0.6, 1.5] ms apart, over the
#           number expected if the two trains were independent (over the time both sets are present).
#           One cell cannot fire twice within its refractory period: ~0 = one cell, ~1 = two cells.
#           Left blank when fewer than 3 such pairs are expected (no power).  Without the .spk, the
#           verdicts rest on xref alone.
#
#  THREE KINDS OF DISAGREEMENT (row `kind`):
#    mixed    a sort cluster holds two curated units (each >= --min-n spikes and >= --frac of the
#             cluster's unit spikes).  one cell (cos >= 0.95, xref <= 0.2) -> "curation split one cell";
#             two cells (xref >= 0.5) -> "sort mixes two cells".
#    noise    a sort cluster holds >= --min-n spikes the curator left in a noise cluster (0/1) beside its
#             main unit.  Same tests against the main unit's spikes in the cluster: one cell ->
#             "curation left the unit's spikes in noise"; otherwise "sort holds noise".
#    split    a curated unit is the main unit of two sort clusters (each >= --min-n of its spikes).
#             Disjoint in time, or one cell -> "sort did not link"; concurrent and two cells ->
#             "curation merged two cells".
#    unit     a curated unit's OWN train violates its refractory period (self ratio >= 0.25 with power;
#             two equal cells give 0.5, a cell with ~15% foreign spikes 0.25):
#             it is several cells, or heavily contaminated -- whatever the sort does with it.
#    Anything in between is "unclear" -- look at it.  A call made on the waveform alone (the pair fires too
#    rarely for the refractory test: expected < 3 pairs, the usual case for two pyramidal cells) carries
#    "(waveform only)": look-alike neighbours reach cosine 0.95, so treat those as leads, not answers.
#
#  OUTPUT
#    <base>.gtreview.<group>.<tag>.tsv   one row per disagreement, largest first; `set_a` / `set_b` are
#                                        cluster ids in the review .clu
#    <base>.clu.<method>.<group>.<tag>_gtreview   (--write-clu, default on) every spike set named in the
#                                        table as its own cluster (all other spikes 0), so the rows can be
#                                        opened in Klusters beside the sort and the curation
#  Read-only for the sort and the curation.
# ════════════════════════════════════════════════════════════════════════════
import argparse
import os
import numpy as np

try:
    from . import neuro_io as nio, session_yaml as sy, fiber_bench as fb, fiber_chunks as fch
    from .fiber_psame import rollcos
except ImportError:
    import neuro_io as nio, session_yaml as sy, fiber_bench as fb, fiber_chunks as fch
    from fiber_psame import rollcos

_LP = "▸ fiber-gt-review"
ONE_CELL_COS, ONE_CELL_XREF, TWO_CELL_XREF = 0.95, 0.2, 0.5
WAVE_TWO_COS = 0.90                       # waveform-only call of two cells below this cosine
SELF_FLAG = 0.25                          # a curated unit's own ratio at or above this is flagged: k equal
                                          # cells give 1 - 1/k (0.5 for two), a cell with a share f of
                                          # foreign spikes about 2f(1 - f) (0.25 at f ~ 0.15)


def cross_refractory(ta, tb, sr, lo_ms=0.6, hi_ms=1.5, min_expected=3.0):
    """(observed / expected pairs with |ta - tb| in (lo, hi] ms, expected) over the span both trains cover;
    ratio NaN when fewer than min_expected pairs are expected."""
    ta = np.sort(np.asarray(ta, np.int64)); tb = np.sort(np.asarray(tb, np.int64))
    if ta.size == 0 or tb.size == 0:
        return float("nan"), 0.0
    t0, t1 = max(ta[0], tb[0]), min(ta[-1], tb[-1])
    if t1 <= t0:
        return float("nan"), 0.0
    a = ta[(ta >= t0) & (ta <= t1)]; b = tb[(tb >= t0) & (tb <= t1)]
    lo, hi = lo_ms * 1e-3 * sr, hi_ms * 1e-3 * sr
    obs = 0
    for s in (1, -1):                                     # b after a, b before a
        if s == 1:
            n = np.searchsorted(b, a + hi, "right") - np.searchsorted(b, a + lo, "right")
        else:
            n = np.searchsorted(b, a - lo, "left") - np.searchsorted(b, a - hi, "left")
        obs += int(n.sum())
    exp = a.size * b.size * 2.0 * (hi - lo) / float(t1 - t0)
    return (obs / exp if exp >= min_expected else float("nan")), exp


def self_refractory(t, sr, lo_ms=0.6, hi_ms=1.5, min_expected=3.0):
    """(observed / expected intervals in (lo, hi] ms between any two spikes of ONE train, expected): ~0 for
    one cell, 1 - 1/k for k equal independent cells, ~2f(1 - f) for a cell with a share f of foreign
    spikes.  NaN without power."""
    t = np.sort(np.asarray(t, np.int64))
    if t.size < 2:
        return float("nan"), 0.0
    lo, hi = lo_ms * 1e-3 * sr, hi_ms * 1e-3 * sr
    obs = int((np.searchsorted(t, t + hi, "right") - np.searchsorted(t, t + lo, "right")).sum())
    T = float(t[-1] - t[0])
    exp = t.size * (t.size - 1) * (hi - lo) / T if T > 0 else 0.0
    return (obs / exp if exp >= min_expected else float("nan")), exp


def isi_violations(t, sr, ms=1.5):
    t = np.sort(np.asarray(t, np.int64))
    return float(np.mean(np.diff(t) < ms * 1e-3 * sr)) if t.size > 1 else float("nan")


def evidence(cos, xref):
    """'one' / 'two' (refractory decides, the waveform does not contradict), 'one?' / 'two?' (waveform only:
    the refractory test has no power -- most pyramidal pairs, which fire too rarely), or 'unclear'.
    The waveform alone is weak: look-alike neighbours reach cosine 0.95 (g6 0316)."""
    has_c, has_x = cos == cos, xref == xref
    if has_x and xref <= ONE_CELL_XREF and (not has_c or cos >= ONE_CELL_COS):
        return "one"
    if has_x and xref >= TWO_CELL_XREF:
        return "two"
    if not has_x and has_c:
        return "one?" if cos >= ONE_CELL_COS else ("two?" if cos < WAVE_TWO_COS else "unclear")
    return "unclear"


_TEXT = {"mixed": ("curation split one cell", "sort mixes two cells"),
         "noise": ("curation left the unit's spikes in noise", "sort holds noise"),
         "split": ("sort did not link", "curation merged two cells")}


def verdict(kind, cos, xref, overlap_frac=1.0):
    """The row's reading, from evidence(); '(waveform only)' marks a call without refractory power.  A
    curated unit in two sort clusters that never overlap in time is a linking miss whatever the evidence."""
    if kind == "split" and overlap_frac < 0.1:
        return "sort did not link (disjoint in time)"
    e = evidence(cos, xref)
    if e == "unclear":
        return "unclear"
    one, two = _TEXT[kind]
    txt = one if e.startswith("one") else two
    return txt + (" (waveform only)" if e.endswith("?") else "")


def review(clu, gt_all, units, res, sr, spk=None, min_n=50, frac=0.1, cap=500, rng=None):
    """The disagreement rows (dicts) and the spike sets they name {set id: spike indices}."""
    rng = rng or np.random.default_rng(0)
    clu = np.asarray(clu, np.int64); g = np.asarray(gt_all, np.int64); res = np.asarray(res, np.int64)
    units = set(int(u) for u in units)
    ok = (clu >= 2) & ~fch.is_reserved(clu) & (g >= 0)
    lab = np.where(np.isin(g, list(units)), g, -2)        # -2 = the curator's noise / artefact (0/1, not a unit)
    key = np.stack([clu[ok], lab[ok]], 1)
    pairs, inv = np.unique(key, axis=0, return_inverse=True)
    idx_ok = np.flatnonzero(ok)
    order = np.argsort(inv.ravel(), kind="stable")
    bounds = np.r_[0, np.cumsum(np.bincount(inv.ravel(), minlength=len(pairs)))]
    members = {(int(c), int(u)): idx_ok[order[bounds[k]:bounds[k + 1]]] for k, (c, u) in enumerate(pairs.tolist())}
    by_c = {}
    for (c, u), ix in members.items():
        by_c.setdefault(c, {})[u] = ix
    tmpl = {}

    def T(k):
        if spk is None:
            return None
        if k not in tmpl:
            ix = members[k]
            if ix.size > cap:
                ix = np.sort(rng.choice(ix, cap, replace=False))
            tmpl[k] = np.asarray(spk[ix], float).mean(0)
        return tmpl[k]

    def cos(ka, kb):
        a, b = T(ka), T(kb)
        return float(rollcos(a, b)) if a is not None else float("nan")

    def span(ix):
        t = res[ix]
        return float(t.min() / sr / 60.0), float(t.max() / sr / 60.0)

    sets, rows = {}, []

    def sid(k):
        if k not in sets:
            sets[k] = len(sets) + 2                       # review .clu ids from 2
        return sets[k]

    def row(kind, ka, kb, extra):
        a, b = members[ka], members[kb]
        x, ex = cross_refractory(res[a], res[b], sr)
        (a0, a1), (b0, b1) = span(a), span(b)
        ov = max(0.0, min(a1, b1) - max(a0, b0)) / max(min(a1 - a0, b1 - b0), 1e-9)
        c = cos(ka, kb)
        rows.append(dict(kind=kind, verdict=verdict(kind, c, x, ov), n_a=int(a.size), n_b=int(b.size),
                         cos=c, xref=x, xref_expected=ex, xref_observed=int(round(x * ex)) if x == x else "", isi_viol_a=isi_violations(res[a], sr),
                         isi_viol_b=isi_violations(res[b], sr), a_min=f"{a0:.1f}-{a1:.1f}", b_min=f"{b0:.1f}-{b1:.1f}",
                         overlap=float(min(ov, 1.0)), set_a=sid(ka), set_b=sid(kb), **extra))

    for c, comp in sorted(by_c.items()):
        uu = {u: ix for u, ix in comp.items() if u != -2}
        n_units = sum(ix.size for ix in uu.values())
        if not uu:
            continue
        main = max(uu, key=lambda u: uu[u].size)
        big = sorted([u for u, ix in uu.items() if ix.size >= max(min_n, frac * n_units)], key=lambda u: -uu[u].size)
        for i in range(len(big)):
            for j in range(i + 1, len(big)):
                row("mixed", (c, big[i]), (c, big[j]),
                    dict(sort_cluster=c, sort_cluster_b=c, unit=big[i], unit_b=big[j]))
        nz = comp.get(-2)
        if nz is not None and nz.size >= max(min_n, frac * (n_units + nz.size)) and uu[main].size >= min_n:
            row("noise", (c, main), (c, -2), dict(sort_cluster=c, sort_cluster_b=c, unit=main, unit_b=-1))
    owners = {}
    for c, comp in by_c.items():
        uu = {u: ix for u, ix in comp.items() if u != -2}
        if uu:
            main = max(uu, key=lambda u: uu[u].size)
            if uu[main].size >= min_n:
                owners.setdefault(main, []).append(c)
    for u, cs in sorted(owners.items()):
        cs = sorted(cs, key=lambda c: -members[(c, u)].size)
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                row("split", (cs[i], u), (cs[j], u), dict(sort_cluster=cs[i], sort_cluster_b=cs[j], unit=u, unit_b=u))
    for u in sorted(units):                              # the curated units' own refractory periods
        t = res[g == u]
        if t.size < min_n:
            continue
        x, ex = self_refractory(t, sr)
        if x == x and x >= SELF_FLAG:
            t0, t1 = float(t.min() / sr / 60.0), float(t.max() / sr / 60.0)
            rows.append(dict(kind="unit", verdict="curated unit violates refractory (several cells, or contaminated)",
                             unit=u, n_a=int(t.size), n_b=0, xref=x, xref_expected=ex, cos=float("nan"),
                             xref_observed=int(round(x * ex)),
                             isi_viol_a=isi_violations(t, sr), a_min=f"{t0:.1f}-{t1:.1f}",
                             rate_hz=float(t.size / max((t1 - t0) * 60.0, 1e-9))))
    rows.sort(key=lambda r: (["unit", "mixed", "noise", "split"].index(r["kind"]), -(r["n_a"] + r.get("n_b", 0))))
    return rows, {s: members[k] for k, s in sets.items()}


COLS = ["kind", "verdict", "rate_hz", "sort_cluster", "sort_cluster_b", "unit", "unit_b", "n_a", "n_b", "cos", "xref",
        "xref_observed", "xref_expected", "isi_viol_a", "isi_viol_b", "a_min", "b_min", "overlap", "set_a", "set_b"]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fiber-gt-review",
                                 description="List where a sort and a curated ground truth disagree, with "
                                             "waveform and refractory evidence for which one is wrong.")
    ap.add_argument("session"); ap.add_argument("elec", type=int)
    ap.add_argument("--clu-method", default="stderiv_C5_D34", help="method token of the sort's .clu")
    ap.add_argument("--clu-stage", required=True, help="stage tag of the sort to compare (e.g. exp_psame)")
    ap.add_argument("--gt-clu", required=True, help="curated .clu")
    ap.add_argument("--gt-res", required=True, help="its .res")
    ap.add_argument("--gt-units", default="", help="curated units (default every unit > 1)")
    ap.add_argument("--gt-exclude", default="", help="units to leave out")
    ap.add_argument("--gt-tol", type=int, default=5, help="spike match tolerance (samples)")
    ap.add_argument("--min-n", type=int, default=50, help="spikes a set needs to be listed")
    ap.add_argument("--frac", type=float, default=0.1, help="... and this share of its sort cluster")
    ap.add_argument("--spk-variant", default="standard", help="waveforms for the cosine (default raw standard)")
    ap.add_argument("--no-clu", action="store_true", help="do not write the review .clu")
    a = ap.parse_args(argv)

    cfg = sy.resolve_session_params(a.session, a.elec)
    base, sr, ns, nc = cfg["base"], cfg["sr"], cfg["nsamp"], cfg["nchan"]
    res = nio.read_res(base, a.elec)
    _, clu = nio.read_clu_at(base, a.elec, variant=a.clu_method, tag=a.clu_stage, n_spikes=res.size)
    _, glab = nio.read_clu_file(a.gt_clu)
    gres = nio.read_res_file(a.gt_res)
    if glab.size != gres.size:
        raise SystemExit(f"{_LP} · {os.path.basename(a.gt_clu)} holds {glab.size:,} labels but "
                         f"{os.path.basename(a.gt_res)} {gres.size:,} times")
    units = fb.parse_units(a.gt_units, a.gt_exclude) if a.gt_units else sorted(int(u) for u in np.unique(glab) if u > 1)
    g_all = fb.match_gt(res, gres, glab, None, a.gt_tol)
    print(f"{_LP} · .{a.clu_stage} vs {os.path.basename(a.gt_clu)}: {int((g_all >= 0).sum()):,} of {res.size:,} spikes "
          f"matched, {len(units)} curated units")
    spk = None
    sp = nio.session_path(base, "spk", a.elec, variant=a.spk_variant)
    if os.path.exists(sp):
        spk = nio.open_spk_file(sp, ns, nc)
        if spk.shape[0] != res.size:
            print(f"{_LP} · {os.path.basename(sp)} holds {spk.shape[0]:,} spikes, the .res {res.size:,}: no cosine")
            spk = None
    else:
        print(f"{_LP} · no {os.path.basename(sp)}: no cosine (refractory evidence only)")
    rows, sets = review(clu, g_all, units, res, sr, spk=spk, min_n=a.min_n, frac=a.frac)
    out = f"{base}.gtreview.{a.elec}.{a.clu_stage}.tsv"
    fb._tsv_write(out, rows, COLS)
    from collections import Counter
    cnt = Counter((r["kind"], r["verdict"]) for r in rows)
    print(f"{_LP} · {len(rows)} disagreement(s); wrote {os.path.basename(out)}")
    for (k, v), n in sorted(cnt.items()):
        print(f"    {k:<6} {v:<42} {n}")
    if not a.no_clu and sets:
        lab = np.zeros(res.size, np.int64)
        for s, ix in sets.items():
            lab[ix] = s
        tag = f"{a.clu_stage}_gtreview"
        nio.write_clu(base, a.elec, lab, variant=a.clu_method, tag=tag)
        print(f"{_LP} · wrote {os.path.basename(nio.session_path(base, 'clu', a.elec, variant=a.clu_method, tag=tag))} "
              f"({len(sets)} spike sets; ids = the table's set_a / set_b)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
