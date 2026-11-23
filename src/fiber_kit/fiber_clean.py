#!/usr/bin/env python3
# ══════════════════════════════════════════════════════════════════════════════
#  fiber_clean.py  —  keep the clean part of a sort, send the rest to the noise
#
#  The two Klusters tools of 2026-11-20..22 as one pass over a flat .clu:
#    1. CORE CUT (fiber_corecut, Klusters' "Split Off Spurious Spikes") on every
#       cluster of at least --cc-min spikes, on the RAW .spk.standard windows:
#       the spikes past the cut go to the noise cluster.
#    2. QUALITY of what is left (fiber_corecut.residual_variance, the waveform
#       view's quality dot; the session's method .spk, up to --q-spikes evenly
#       spaced spikes): a cluster at or above --q-dirty is DIRTY and goes to the
#       noise whole.
#    3. A cluster left with fewer than --min-keep spikes goes to the noise, and
#       so do the clusters named in --junk (fiber-polish's residue cluster).
#  Cluster 0 (artefacts), the noise cluster itself and the chunks' reserved
#  clusters are left as they are.  Writes a flat .clu (the .clc/.clp hierarchy is
#  not rebuilt) and <base>.clean.<method>.<g>.<stage>.tsv with one row per input
#  cluster.
#
#  What it buys and costs, measured on sirotaA-jg-000005-20120312 group 8
#  (fs_polish, against the curated file, time-local purity; analysis/
#  fs-polish-clean-resort): purity 93.0 -> 95.8 %, clusters >= 90 % pure 57 -> 66 %,
#  curated spikes held 77.1 -> 53.7 %.  The cut errs on the side of splitting: it
#  took 29 % of the kept clusters' spikes, most of them spikes of the same unit
#  (which is what fiber-unmix then puts back).  The quality value is a weak
#  indicator on its own (AUC 0.71-0.74 for < 90 % against >= 98 % pure).
# ══════════════════════════════════════════════════════════════════════════════
import argparse
import os
import sys

import numpy as np

try:
    from . import fiber_chunks as fch
    from . import fiber_corecut as fcc
    from . import neuro_io as nio
    from . import session_yaml as sy
except ImportError:
    import fiber_chunks as fch
    import fiber_corecut as fcc
    import neuro_io as nio
    import session_yaml as sy

REPORT_COLS = ["cluster", "n_in", "n_kept", "cut_channel", "quality", "kind"]


def _even(idx, n):
    return idx if len(idx) <= n else idx[(np.arange(n) * len(idx)) // n]


def clean(clu, raw, qual, sr, *, noise_id=1, junk=(), keep_ids=(), core_cut=True, cc_min=200,
          cc_max=40000, q_dirty=0.52, q_spikes=1000, min_keep=50, log=None):
    """One cleaning pass (see the header).

    clu: labels.  raw(ix) -> (len(ix), nSamp, nCh) RAW waveforms for the core cut;
    qual(ix) -> the waveforms the quality value is taken on.  Returns (labels, rows)
    with rows = one dict (REPORT_COLS) per input cluster; kind is kept | passed |
    dirty | small | junk."""
    log = log or (lambda s: None)
    clu = np.asarray(clu, np.int64)
    out = clu.copy()
    real = (clu > 1) & (clu != noise_id) & ~fch.is_reserved(clu)
    junk = {int(x) for x in junk}; keep_ids = {int(x) for x in keep_ids}
    rows = []
    ids, cnt = np.unique(clu[real], return_counts=True)
    for u, c in zip(ids, cnt):
        u, c = int(u), int(c)
        j = np.flatnonzero(clu == u)
        row = dict(cluster=u, n_in=c, n_kept=c, cut_channel=-1, quality=float("nan"), kind="kept")
        rows.append(row)
        if u in keep_ids:
            row["kind"] = "passed"; continue
        if u in junk:
            out[j] = noise_id; row.update(n_kept=0, kind="junk"); continue
        keep = np.ones(c, bool)
        if core_cut and c >= cc_min:
            r = fcc.analyse(raw(j), sr, min_spikes=cc_min, max_ordered=cc_max)
            if r["ok"]:
                keep = r["keep"]; row["cut_channel"] = r["cut_channel"]
        jk = j[keep]
        out[j[~keep]] = noise_id
        if len(jk) < min_keep:
            out[jk] = noise_id; row.update(n_kept=0, kind="small"); continue
        q = fcc.residual_variance(qual(_even(jk, q_spikes))) if q_dirty > 0 else float("nan")
        row["quality"] = q
        if q_dirty > 0 and not (q < q_dirty):                    # NaN (no value) counts as dirty
            out[jk] = noise_id; row.update(n_kept=0, kind="dirty"); continue
        row["n_kept"] = int(len(jk))
    return out, rows


def write_report(path, rows):
    with nio.atomic_write(path, "w") as f:
        f.write("\t".join(REPORT_COLS) + "\n")
        for r in rows:
            q = "" if not np.isfinite(r["quality"]) else f"{r['quality']:.4f}"
            f.write(f"{r['cluster']}\t{r['n_in']}\t{r['n_kept']}\t{r['cut_channel']}\t{q}\t{r['kind']}\n")
    return path


def _id_list(text):
    return tuple(int(x) for x in str(text).replace(",", " ").split())


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Keep the clean part of a sort: core cut per cluster (Klusters' Split Off Spurious "
                    "Spikes), then the quality value of what is left; cut-off spikes, dirty clusters and "
                    "clusters left too small go to the noise cluster.")
    ap.add_argument("base"); ap.add_argument("elec", type=int)
    nio.add_clu_args(ap, method_default="stderiv", stage_default="fiber_session", stage_dest="clu_stage",
                     stage_alias=None)
    ap.add_argument("--res-variant", default="", help="pin the .res variant ('' = the shared .res)")
    ap.add_argument("--noise-id", type=int, default=1, help="cluster the removed spikes go to (default 1)")
    ap.add_argument("--junk", default="", help="clusters sent to the noise whole, e.g. fiber-polish's residue cluster")
    ap.add_argument("--keep-ids", default="", help="clusters passed through untouched")
    ap.add_argument("--core-cut", type=int, default=1, help="1 = core cut per cluster (default), 0 = quality only")
    ap.add_argument("--cc-min", type=int, default=200, help="smallest cluster the core cut runs on (default 200)")
    ap.add_argument("--cc-max", type=int, default=40000,
                    help="most spikes of a cluster that are ordered; above it an evenly spaced subset (default 40000)")
    ap.add_argument("--q-dirty", type=float, default=0.52,
                    help="a cluster whose quality value is at or above this is dirty (default 0.52, the red of "
                         "Klusters' quality dot; 0 = no quality test)")
    ap.add_argument("--q-spikes", type=int, default=1000, help="spikes per cluster the quality value is taken on")
    ap.add_argument("--min-keep", type=int, default=50, help="a cluster left with fewer spikes goes to the noise")
    ap.add_argument("--out-tag", default=None, help="output stage tag (default <clu-stage>_clean)")
    a = ap.parse_args(argv)
    log = lambda s: print(s, file=sys.stderr, flush=True)  # noqa: E731

    cfg = sy.resolve_session_params(a.base, a.elec, verbose=False)
    base = cfg["base"]; NS, NC = int(cfg["nsamp"]), int(cfg["nchan"]); sr = float(cfg["sr"])
    res = nio.read_res_at(base, a.elec, variant=a.res_variant) if a.res_variant else nio.read_res(base, a.elec)
    n = len(res)
    _, clu = nio.resolve_clu(a, base, a.elec, n_spikes=n, stage_dest="clu_stage")
    clu = np.asarray(clu, np.int64)
    try:
        rawmm, rs = nio.open_spk(base, a.elec, NS, NC, prefer=nio.prefer_standard())
    except FileNotFoundError:
        raise SystemExit("fiber-clean needs the RAW waveforms (.spk.standard.<group>): the core cut is "
                         "made on them") from None
    if rawmm.shape[0] != n:
        raise SystemExit(f"{os.path.basename(rs.path)} holds {rawmm.shape[0]:,} spikes, the .res {n:,}")
    qmm, qname = rawmm, os.path.basename(rs.path)
    if nio.parse_variant_token(a.clu_method).family != "standard":
        r = nio.resolve(base, "spk", a.elec, a.clu_method)
        if r.found and r.variant == a.clu_method:
            m = nio.open_spk_file(r.path, NS, NC)
            if m.shape[0] != n:
                raise SystemExit(f"{os.path.basename(r.path)} holds {m.shape[0]:,} spikes, the .res {n:,}")
            qmm, qname = m, os.path.basename(r.path)
    if a.q_dirty > 0 and qmm is rawmm and nio.parse_variant_token(a.clu_method).family != "standard":
        log(f"[clean] no .spk of method {a.clu_method}: the quality value is taken on the raw windows, where "
            f"it reads lower (--q-dirty {a.q_dirty} was set on stderiv waveforms)")
    raw = lambda ix: np.asarray(rawmm[np.asarray(ix)], np.float32)   # noqa: E731
    qual = lambda ix: np.asarray(qmm[np.asarray(ix)], np.float32)    # noqa: E731

    log(f"[clean] {os.path.basename(base)} group {a.elec}: {n:,} spikes, stage '{a.clu_stage}' | core cut "
        f"{'on >= ' + str(a.cc_min) if a.core_cut else 'off'}, dirty at quality >= {a.q_dirty:g} ({qname}), "
        f"kept >= {a.min_keep}")
    new, rows = clean(clu, raw, qual, sr, noise_id=a.noise_id, junk=_id_list(a.junk),
                      keep_ids=_id_list(a.keep_ids), core_cut=bool(a.core_cut), cc_min=a.cc_min,
                      cc_max=a.cc_max, q_dirty=a.q_dirty, q_spikes=a.q_spikes, min_keep=a.min_keep, log=log)
    for kind in ("kept", "passed", "dirty", "small", "junk"):
        rr = [r for r in rows if r["kind"] == kind]
        if not rr:
            continue
        nin = sum(r["n_in"] for r in rr); nk = sum(r["n_kept"] for r in rr)
        log(f"[clean]   {kind:<6} {len(rr):>5} clusters, {nin:>9,} spikes in"
            + (f", {nk:,} kept after the cut ({1 - nk / max(nin, 1):.1%} cut off)" if kind == "kept" else ""))
    tag = a.out_tag or (f"{a.clu_stage}_clean" if a.clu_stage else "clean")
    path = nio.write_clu(base, a.elec, new, variant=a.clu_method, tag=tag)
    rep = write_report(nio.session_path(base, "clean", a.elec, variant=a.clu_method, tag=tag) + ".tsv", rows)
    nkept = sum(1 for r in rows if r["kind"] in ("kept", "passed"))
    log(f"[clean] wrote {os.path.basename(path)}: {nkept} clusters, {int((new != clu).sum()):,} spikes moved to "
        f"cluster {a.noise_id} (now {int((new == a.noise_id).sum()):,}, {np.mean(new == a.noise_id):.1%}) "
        f"-- flat .clu; the .clc/.clp hierarchy is not rebuilt")
    log(f"[clean] wrote {os.path.basename(rep)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
