#!/usr/bin/env python3
# ══════════════════════════════════════════════════════════════════════════════
#  fiber_unmix.py  —  explain the spikes OUTSIDE the kept units on the .fil
#
#  After a sort has been cleaned (dirty clusters and cut-off spikes sent to the
#  noise cluster) the noise is mostly not new cells: on sirotaA-jg-000005-20120312
#  group 8 about seven in ten of its spikes fit one kept unit as well as that
#  unit's own spikes do, and a good part of the rest are two kept units summed
#  (analysis/fs-polish-clean-resort, 2026-11-23).  Re-sorting that noise rebuilds
#  poorer copies of the same cells.  This stage explains it instead:
#
#    * MASK.  The spikes of the kept units are not touched and not pursued: they
#      only supply the templates.  The stage works on the rows the input .clu
#      holds in a noise cluster (--noise-ids, default 0 and 1) -- optionally less
#      the rows an earlier round's .eap already explains (--skip-marked).
#    * ON THE .FIL, NOT THE SNIPPET.  Each of those detections is pursued on the
#      continuous filtered signal with fiber_deconv (noise-whitened, sub-sample
#      phases, amplitude priors, joint amplitude refit), over a segment that also
#      covers its neighbours.  A partner 15 samples away is fitted whole, where a
#      52-sample .spk window cuts it off.  Nothing is subtracted from any file:
#      the subtraction is in memory, per segment.
#    * PER TIME BLOCK.  Templates, noise model, pursuit threshold and the
#      residual a clean fit leaves are all taken per --block-min of recording
#      from the kept units' spikes of that block, so a drifting cell is matched
#      by its template of the time.
#
#  Each pursued detection gets one KIND:
#    one          one kept unit explains it (residual no larger than a kept spike
#                 leaves): a spike of that unit the sort did not hold
#    collision    a kept unit plus a second, non-equivalent kept unit in the same
#                 waveform (the second one detected itself, or hidden)
#    shadow       no unit sits on the detection, but its window is explained by
#                 neighbouring spikes: a second detection of somebody's flank
#    partial      a kept unit sits on it and a remainder is left: a candidate for
#                 a collision with a cell that has no template yet
#    unexplained  nothing known fits: a cell without a template, or not a spike
#  `partial` and `unexplained` keep their noise label -- they are what the next
#  round's sort should look at.  The others leave the noise cluster.
#
#  Output: the stage's flat .clu, a manifest <base>.unmix.<method>.<g>.<tag>.tsv
#  with one row per pursued detection, and (--eap) the stage's .eap membership:
#  every kept spike holds its own class at offset 0 (the mask), every explained
#  detection the classes found in it at their sample offsets.
#
#  Limits.  Only DETECTED events are examined: a spike that never crossed the
#  threshold because it sits under a kept spike is not found here (that needs a
#  subtraction from the recording).  Two spikes within a few samples of each
#  other are close to one larger spike and are mostly read as `one`.  The stage
#  is as good as its templates: an impure kept unit explains what it should not.
# ══════════════════════════════════════════════════════════════════════════════
import argparse
import os
import sys

import numpy as np

try:
    from . import fiber_chunks as fch
    from . import fiber_decollide as fdc
    from . import fiber_deconv as fd
    from . import neuro_io as nio
    from . import session_yaml as sy
except ImportError:
    import fiber_chunks as fch
    import fiber_decollide as fdc
    import fiber_deconv as fd
    import neuro_io as nio
    import session_yaml as sy

NONE, ONE, COLLISION, SHADOW, PARTIAL, UNEXPLAINED = 0, 1, 2, 3, 4, 5
KIND_NAMES = {NONE: "none", ONE: "one", COLLISION: "collision", SHADOW: "shadow",
              PARTIAL: "partial", UNEXPLAINED: "unexplained"}
MANIFEST_COLS = ["spike", "res", "kind", "k1", "a1", "dt1", "gain", "margin", "resid",
                 "k2", "a2", "dt2", "k2_detected"]
RESID_FALLBACK = 1.5      # residual bound of a block with too few kept spikes to measure one


def kept_mask(clu, noise_ids=(0, 1)):
    """Rows in a kept unit: not in a noise cluster and not in a chunk's reserved cluster."""
    clu = np.asarray(clu)
    return ~np.isin(clu, list(noise_ids)) & ~fch.is_reserved(clu) & (clu >= 0)


def _isolated(res, rows, gap):
    """Of the sorted detections `res`, the `rows` with no other detection within `gap` samples."""
    prev = np.r_[np.iinfo(np.int64).min // 2, res[:-1]]
    nxt = np.r_[res[1:], np.iinfo(np.int64).max // 2]
    return rows[((res - prev)[rows] > gap) & ((nxt - res)[rows] > gap)]


def classify(r, res_sub, todo_sub, nsamp, peak, resid_max):
    """Kinds of the detections `todo_sub` (indices into the pursued set) from a
    fiber_deconv Result `r` over the detections `res_sub`.

    `nsamp`, `peak` are the .spk window's.  A placement is IN a detection's
    waveform when its peak lies inside the detection's window or within a
    quarter window of either end (its flank is then in the snippet).  A
    detection's first component is the placement anchored on it; failing that,
    the strongest placement in its waveform that no detection claims (the unit
    the detection is a mis-timed sighting of).  Its second component is the
    strongest remaining placement of a non-equivalent unit in its waveform,
    detected or not.  Returns a dict of arrays over `todo_sub`."""
    P = r.placements
    o = np.argsort(P["t"], kind="stable")
    pt, pu, pa, pg, pan = P["t"][o], P["unit"][o], P["amp"][o], P["gain"][o], P["anchor"][o]
    before, after = peak + nsamp // 4, (nsamp - peak) + nsamp // 4
    n = len(todo_sub)
    out = dict(kind=np.full(n, UNEXPLAINED, np.int8), k1=np.full(n, -1, np.int64),
               a1=np.full(n, np.nan), dt1=np.full(n, np.nan), gain=np.full(n, np.nan),
               margin=np.full(n, np.nan), resid=np.full(n, np.nan), k2=np.full(n, -1, np.int64),
               a2=np.full(n, np.nan), dt2=np.full(n, np.nan), k2_det=np.zeros(n, np.int8))
    for i, d in enumerate(todo_sub):
        td = float(res_sub[d])
        lo, hi = np.searchsorted(pt, [td - before, td + after])
        near = [j for j in range(lo, hi) if pan[j] != d]
        k1 = int(r.primary[d])
        if k1 >= 0:
            out["a1"][i] = r.amp[d]; out["dt1"][i] = r.dt[d]
            out["gain"][i] = r.gain[d]; out["margin"][i] = r.margin[d]
        else:
            free = [j for j in near if pan[j] < 0]
            if free:
                j = max(free, key=lambda q: pg[q])
                k1 = int(pu[j]); near.remove(j)
                out["a1"][i] = pa[j]; out["dt1"][i] = pt[j] - td; out["gain"][i] = pg[j]
        own = r.equiv.get(k1, {k1}) if k1 >= 0 else set()
        other = [j for j in near if int(pu[j]) not in own]
        rs = float(r.resid[d])
        out["resid"][i] = rs
        clean = bool(rs <= resid_max)                 # NaN (no window) is not clean
        if other:
            j = max(other, key=lambda q: pg[q])
            out["k2"][i] = int(pu[j]); out["a2"][i] = pa[j]; out["dt2"][i] = pt[j] - td
            out["k2_det"][i] = 1 if pan[j] >= 0 else 0
        out["k1"][i] = k1
        if k1 >= 0:
            out["kind"][i] = (COLLISION if other else ONE) if clean else PARTIAL
        else:
            out["kind"][i] = SHADOW if (other and clean) else UNEXPLAINED
    return out


def unmix(read_raw, res, clu, nsamp, peak, sr, *, todo=None, noise_ids=(0, 1), n_samples=None,
          block_min=30.0, blocks=None, R=4, order=4, refrac_ms=1.0, thr_floor=16.0, null_q=0.995,
          min_n=30, equiv_cos=0.93, resid_q=0.95, calib_n=300, tol=4, extend=8, seed=0,
          transform=None, log=None):
    """Explain the detections `todo` (default: the rows of a noise cluster) by
    the kept units of `clu`, on the continuous signal `read_raw` (see the header).

    extend: the templates (and the window the residual is taken over) reach this
    many samples beyond the .spk window on each side.  A spike's tail does not
    end where its snippet does, and a tail the template lacks is left in the
    signal as if it were somebody else's.

    res must be sorted.  `blocks` restricts the run to those block indices (the
    other blocks' rows stay NONE).  Returns a dict of per-spike arrays (kind, k1,
    a1, dt1, gain, margin, resid, k2, a2, dt2, k2_det; NONE / -1 / NaN where a
    row was not pursued) plus `blocks`: one dict per block with its counts,
    pursuit threshold, residual bound and the kept spikes' self-agreement."""
    log = log or (lambda s: None)
    res = np.asarray(res, np.int64); clu = np.asarray(clu, np.int64)
    N = res.size
    if N and np.any(np.diff(res) < 0):
        raise ValueError("unmix: the .res is not sorted")
    kept = kept_mask(clu, noise_ids)
    if todo is None:
        todo = np.isin(clu, list(noise_ids))
    todo = np.asarray(todo, bool) & ~kept
    if n_samples is None:
        n_samples = int(res.max()) + 4 * nsamp if N else 0
    out = dict(kind=np.zeros(N, np.int8), k1=np.full(N, -1, np.int64), a1=np.full(N, np.nan),
               dt1=np.full(N, np.nan), gain=np.full(N, np.nan), margin=np.full(N, np.nan),
               resid=np.full(N, np.nan), k2=np.full(N, -1, np.int64), a2=np.full(N, np.nan),
               dt2=np.full(N, np.nan), k2_det=np.zeros(N, np.int8))
    stats = []
    blk = max(1, int(round(float(block_min) * 60.0 * float(sr))))
    nb = int(np.ceil(n_samples / blk)) if N else 0
    ext = max(0, int(extend))
    nst, pkt = nsamp + 2 * ext, peak + ext           # the templates' window
    ctx = nst + nst // 2
    rng = np.random.default_rng(seed)
    for b in range(nb):
        if blocks is not None and b not in blocks:
            continue
        lo, hi = b * blk, min((b + 1) * blk, int(n_samples))
        i0, i1 = np.searchsorted(res, [lo, hi])
        rows = np.arange(i0, i1)
        td = rows[todo[rows]]
        if td.size == 0:
            continue
        st = dict(block=b, n=int(td.size), units=0, thr=float("nan"), resid_max=float("nan"),
                  agree=float("nan"), n_calib=0)
        rb = res[rows]
        lab = np.where(kept[rows], clu[rows], -1)
        try:
            model = fd.build_model(read_raw, rb, lab, nst, pkt, s0=lo, s1=hi, R=R, order=order,
                                   min_n=min_n, transform=transform, seed=seed, equiv_cos=equiv_cos)
        except ValueError:
            out["kind"][td] = UNEXPLAINED               # no kept unit has a template in this block
            st["counts"] = {UNEXPLAINED: int(td.size)}
            stats.append(st)
            log(f"[unmix] block {b + 1}/{nb}: {td.size:,} spikes, no kept unit with >= {min_n} "
                f"spikes -- all unexplained")
            continue
        nm, bank, equiv = model
        thr, _ = fd.null_threshold(read_raw, lo, hi, rb, bank, nm, q=null_q, floor=thr_floor,
                                   seed=seed + 1, transform=transform)
        kw = dict(s0=lo, s1=hi, R=R, order=order, refrac_ms=refrac_ms, tol=tol, transform=transform,
                  seed=seed, nm=nm, bank=bank, equiv=equiv, thr=thr)
        # what a clean fit leaves: the kept units' own isolated spikes, pursued the same way
        cal = _isolated(rb, np.flatnonzero(np.isin(lab, bank.keys)), nst)
        if cal.size > calib_n:
            cal = np.sort(rng.choice(cal, calib_n, replace=False))
        rmax = RESID_FALLBACK
        if cal.size:
            rc = fd.deconvolve(read_raw, rb[cal], lab[cal], nst, pkt, sr, **kw)
            okc = np.isfinite(rc.resid)
            if okc.sum() >= 20:
                rmax = float(np.quantile(rc.resid[okc], resid_q))
            st["agree"] = float(np.mean([int(p) in equiv.get(int(l), ()) for l, p in zip(lab[cal], rc.primary)]))
            st["n_calib"] = int(cal.size)
        # the detections to explain, and every detection close enough to share a window with one
        tt = rb[todo[rows]]
        near = np.searchsorted(tt, rb + ctx, "right") > np.searchsorted(tt, rb - ctx, "left")
        sub = np.flatnonzero(near)
        r = fd.deconvolve(read_raw, rb[sub], lab[sub], nst, pkt, sr, **kw)
        tsub = np.flatnonzero(todo[rows[sub]])
        c = classify(r, rb[sub], tsub, nsamp, peak, rmax)
        g = rows[sub[tsub]]
        for key in ("kind", "k1", "a1", "dt1", "gain", "margin", "resid", "k2", "a2", "dt2", "k2_det"):
            out[key][g] = c[key]
        st.update(units=len(bank.keys), thr=float(thr), resid_max=rmax,
                  counts={int(k): int(v) for k, v in zip(*np.unique(c["kind"], return_counts=True))})
        stats.append(st)
        cn = st["counts"]
        log(f"[unmix] block {b + 1}/{nb}: {td.size:,} spikes, {len(bank.keys)} units | "
            + "  ".join(f"{KIND_NAMES[k]} {cn.get(k, 0):,}" for k in (ONE, COLLISION, SHADOW, PARTIAL, UNEXPLAINED))
            + f" | thr {thr:.1f}, resid <= {rmax:.2f}, kept spikes named as their unit {st['agree']:.1%}"
              f" (n {st['n_calib']})")
    out["blocks"] = stats
    return out


def relabel(clu, u, *, one_as="cluster", margin_min=10.0, tol=4):
    """The stage's .clu from the input labels and unmix()'s result.

    partial / unexplained / not-pursued rows keep their label.  collision and
    shadow rows each go to one new cluster.  `one` rows go, by `one_as`:
      cluster  to one new cluster
      units    to one new cluster per unit they fit (to look at next to that unit)
      assign   INTO the unit, when the label is confident (margin >= margin_min)
               and the spike sits on its detection (|dt| <= tol); the others to
               the one new cluster
    Returns (labels, ids) with ids = dict(one, collision, shadow, units={unit: id})."""
    new = np.asarray(clu, np.int64).copy()
    nxt = fch.next_free_id(new)
    ids = dict(one=nxt, collision=nxt + 1, shadow=nxt + 2, units={})
    kind = u["kind"]
    new[kind == COLLISION] = ids["collision"]
    new[kind == SHADOW] = ids["shadow"]
    one = kind == ONE
    if one_as == "assign":
        sure = one & (u["margin"] >= margin_min) & (np.abs(u["dt1"]) <= tol)
        new[sure] = u["k1"][sure]
        new[one & ~sure] = ids["one"]
    elif one_as == "units":
        for j, k in enumerate(np.unique(u["k1"][one])):
            ids["units"][int(k)] = nxt + 3 + j
            new[one & (u["k1"] == k)] = nxt + 3 + j
    else:
        new[one] = ids["one"]
    if nxt + 2 + len(ids["units"]) >= fch.RESERVED_LO:
        raise ValueError("unmix: the new cluster ids would enter the reserved block")
    return new, ids


def manifest_rows(res, u):
    """One dict (MANIFEST_COLS) per pursued detection, in spike order."""
    rows = []
    for i in np.flatnonzero(u["kind"] != NONE):
        rows.append(dict(spike=int(i), res=int(res[i]), kind=KIND_NAMES[int(u["kind"][i])],
                         k1=int(u["k1"][i]), a1=float(u["a1"][i]), dt1=float(u["dt1"][i]),
                         gain=float(u["gain"][i]), margin=float(u["margin"][i]),
                         resid=float(u["resid"][i]), k2=int(u["k2"][i]), a2=float(u["a2"][i]),
                         dt2=float(u["dt2"][i]), k2_detected=int(u["k2_det"][i])))
    return rows


def write_manifest(path, rows):
    def fmt(v):
        if isinstance(v, str):
            return v
        if isinstance(v, float):
            return "" if not np.isfinite(v) else f"{v:.4g}"
        return str(int(v))
    with nio.atomic_write(path, "w") as f:
        f.write("\t".join(MANIFEST_COLS) + "\n")
        for r in rows:
            f.write("\t".join(fmt(r[c]) for c in MANIFEST_COLS) + "\n")
    return path


def eap_manifest(u):
    """unmix()'s result as fiber_decollide.write_eap_membership rows: the classes
    present in each explained detection at their integer sample offsets (a slot
    without a unit is -1 and writes nothing).  A shadow holds its neighbour only."""
    rows = []
    for i in np.flatnonzero(np.isin(u["kind"], (ONE, COLLISION, SHADOW, PARTIAL))):
        k1, k2 = int(u["k1"][i]), int(u["k2"][i])
        rows.append(dict(source_idx=int(i), k1=k1, tau1=int(np.rint(u["dt1"][i])) if k1 >= 0 else 0,
                         k2=k2, tau2=int(np.rint(u["dt2"][i])) if k2 >= 0 else 0))
    return rows


def _id_list(text):
    return tuple(int(x) for x in str(text).replace(",", " ").split())


def _block_set(text):
    if not text:
        return None
    s = set()
    for part in str(text).replace(",", " ").split():
        a, _, b = part.partition("-")
        s.update(range(int(a), int(b or a) + 1))
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Explain the spikes outside the kept units (the noise cluster) as kept units and "
                    "their sums, on the .fil: one unit, a collision of two, a neighbour's flank, a unit "
                    "plus a remainder, or nothing known.  Kept spikes are not touched.")
    ap.add_argument("base"); ap.add_argument("elec", type=int)
    nio.add_clu_args(ap, method_default="stderiv", stage_default="fiber_session", stage_dest="clu_stage",
                     stage_alias=None)
    ap.add_argument("--res-variant", default="", help="pin the .res variant ('' = the shared .res)")
    ap.add_argument("--signal", default="", help="the filtered recording (default <base>.fil)")
    ap.add_argument("--noise-ids", default="0,1", help="clusters whose spikes are explained (default 0,1); "
                                                         "every other non-reserved cluster is a kept unit")
    ap.add_argument("--skip-marked", action="store_true",
                    help="leave out the noise rows the INPUT stage's .eap already holds a class for")
    ap.add_argument("--block-min", type=float, default=30.0,
                    help="minutes per block: templates, noise model and thresholds are per block (default 30)")
    ap.add_argument("--blocks", default="", help="run only these 0-based blocks, e.g. 0-2,7 (default all)")
    ap.add_argument("--phases", type=int, default=4, help="sub-sample phases per template (1 = whole samples)")
    ap.add_argument("--order", type=int, default=4, help="order of the VAR noise-whitening filter")
    ap.add_argument("--refrac-ms", type=float, default=1.0, help="a unit is not placed twice within this")
    ap.add_argument("--floor", type=float, default=16.0, help="lowest pursuit gain threshold")
    ap.add_argument("--null-q", type=float, default=0.995, help="quantile of the off-spike gain taken as threshold")
    ap.add_argument("--min-n", type=int, default=30, help="spikes a kept unit needs in a block for a template")
    ap.add_argument("--equiv-cos", type=float, default=0.93,
                    help="templates at least this similar count as one unit (a second component must differ)")
    ap.add_argument("--resid-q", type=float, default=0.95,
                    help="a fit is clean when its residual is within this quantile of the kept spikes' own")
    ap.add_argument("--extend", type=int, default=8,
                    help="samples the templates reach beyond the .spk window on each side (default 8)")
    ap.add_argument("--calib-n", type=int, default=300, help="kept spikes per block the residual bound is measured on")
    ap.add_argument("--one-as", choices=("cluster", "units", "assign"), default="cluster",
                    help="where one-unit spikes go: one new cluster (default), one new cluster per unit, or "
                         "into the unit itself when --margin-min is met")
    ap.add_argument("--margin-min", type=float, default=10.0,
                    help="--one-as assign: least gain margin over the best other unit")
    ap.add_argument("--eap", choices=("all", "resolved", "off"), default="all",
                    help="the stage's .eap membership: kept spikes (own class, offset 0) and explained "
                         "detections (default), the explained detections only, or none.  The matrix is "
                         "spikes x template classes bytes")
    ap.add_argument("--n-cells", type=int, default=128, help="template-class columns a fresh .eap starts with / grows by")
    ap.add_argument("--out-tag", default=None, help="output stage tag (default <clu-stage>_unmix)")
    a = ap.parse_args(argv)
    log = lambda s: print(s, file=sys.stderr, flush=True)  # noqa: E731

    cfg = sy.resolve_session_params(a.base, a.elec, verbose=False, require=("channels", "ntotal", "nsamp", "sr"))
    base = cfg["base"]; NS = int(cfg["nsamp"]); sr = float(cfg["sr"])
    peak = int(cfg["peak"]) if cfg.get("peak") is not None else NS // 2
    res = nio.read_res_at(base, a.elec, variant=a.res_variant) if a.res_variant else nio.read_res(base, a.elec)
    res = np.asarray(res, np.int64)
    n = res.size
    _, clu = nio.resolve_clu(a, base, a.elec, n_spikes=n, stage_dest="clu_stage")
    clu = np.asarray(clu, np.int64)
    if clu.size != n:
        raise SystemExit(f"the .clu holds {clu.size:,} spikes, the .res {n:,}")
    sig = a.signal or f"{base}.fil"
    if not os.path.exists(sig):
        raise SystemExit(f"fiber-unmix needs the filtered recording: {sig} not found (--signal)")
    mm = nio.open_signal(sig, int(cfg["ntotal"]))
    read_raw = fd.memmap_reader(mm, cfg["channels"])
    noise_ids = _id_list(a.noise_ids)
    kept = kept_mask(clu, noise_ids)
    todo = np.isin(clu, list(noise_ids))
    tag = a.out_tag or (f"{a.clu_stage}_unmix" if a.clu_stage else "unmix")
    if a.skip_marked:
        cur = nio.read_eap(base, a.elec, tag=a.clu_stage or "")
        if cur["ok"] and int(cur["nSpikes"]) == n:
            marked = (np.asarray(cur["cells"]) != nio.EAP_ABSENT).any(1)
            log(f"[unmix] --skip-marked: {int((todo & marked).sum()):,} noise rows already hold a class")
            todo &= ~marked
        else:
            log("[unmix] --skip-marked: the input stage has no .eap of this spike count -- nothing skipped")
    log(f"[unmix] {os.path.basename(base)} group {a.elec}: {n:,} spikes, stage '{a.clu_stage}' | "
        f"{int(kept.sum()):,} in {len(np.unique(clu[kept]))} kept units, {int(todo.sum()):,} to explain | "
        f"{os.path.basename(sig)}, {NS} samples, peak {peak}")
    u = unmix(read_raw, res, clu, NS, peak, sr, todo=todo, noise_ids=noise_ids, n_samples=mm.shape[0],
              block_min=a.block_min, blocks=_block_set(a.blocks), R=a.phases, order=a.order,
              refrac_ms=a.refrac_ms, thr_floor=a.floor, null_q=a.null_q, min_n=a.min_n,
              equiv_cos=a.equiv_cos, resid_q=a.resid_q, calib_n=a.calib_n, extend=a.extend, log=log)
    done = u["kind"] != NONE
    nd = max(1, int(done.sum()))
    log("[unmix] " + "  ".join(f"{KIND_NAMES[k]} {int((u['kind'] == k).sum()):,} ({(u['kind'] == k).sum() / nd:.1%})"
                               for k in (ONE, COLLISION, SHADOW, PARTIAL, UNEXPLAINED))
        + f"  of {int(done.sum()):,} pursued")
    coll = u["kind"] == COLLISION
    if coll.any():
        log(f"[unmix] collisions: second unit detected itself {int((u['k2_det'][coll] == 1).sum()):,}, "
            f"hidden {int((u['k2_det'][coll] == 0).sum()):,}; median |lag| "
            f"{float(np.nanmedian(np.abs(u['dt2'][coll] - np.nan_to_num(u['dt1'][coll])))):.1f} samples")
    new, ids = relabel(clu, u, one_as=a.one_as, margin_min=a.margin_min)
    path = nio.write_clu(base, a.elec, new, variant=a.clu_method, tag=tag)
    man = write_manifest(nio.session_path(base, "unmix", a.elec, variant=a.clu_method, tag=tag) + ".tsv",
                         manifest_rows(res, u))
    moved = int((new != clu).sum())
    where = {"cluster": f"one-unit spikes -> cluster {ids['one']}",
             "units": f"one-unit spikes -> clusters {ids['one'] + 3}.. (one per unit, {len(ids['units'])})",
             "assign": f"one-unit spikes -> their unit when margin >= {a.margin_min:g}, else cluster {ids['one']}"}
    log(f"[unmix] wrote {os.path.basename(path)}: {moved:,} spikes left the noise | {where[a.one_as]}, "
        f"collisions -> {ids['collision']}, shadows -> {ids['shadow']}; partial and unexplained keep their label")
    log(f"[unmix] wrote {os.path.basename(man)}")
    if a.eap != "off":
        own = (np.flatnonzero(kept), clu[kept]) if a.eap == "all" else None
        em = fdc.write_eap_membership(base, a.elec, parent_tag=tag, n_spikes=n, manifest=eap_manifest(u),
                                      n_cells=a.n_cells, own=own)
        if em:
            log(f"[unmix] wrote {os.path.basename(em['eap'])} ({em['cells']:,} cells, "
                f"{os.path.getsize(em['eap']) / 1e6:.0f} MB) + {os.path.basename(em['tcl'])}")
        else:
            log("[unmix] .eap not written (nothing to mark, or its spike count disagrees)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
