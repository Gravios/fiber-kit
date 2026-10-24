#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════════════════════
#  fiber_xcorr_merge.py -- link/merge clusters by KLUSTERS-STYLE cross-correlation
#  cosine (max cosine over CIRCULAR TIME SHIFTS of the two mean waveforms,
#  max_s cos(t_a, roll(t_b, s))), as a CONFIDENCE-ORDERED AGGLOMERATION: the
#  highest-cosine pair at/above the threshold fuses first, the combined spikes are
#  RE-ALIGNED and the merged template recomputed, and only then is the next pair
#  picked -- so each decision uses a clean, current template (the shape fiber-peel
#  uses).  A merge is vetoed when the refractory cross-correlogram is powered yet
#  shows no dip (two co-located but INDEPENDENT cells).  A pure clu relabel; .res/.spk
#  untouched.
#
#  Meant to run AFTER fiber-backbone-link, to fuse chains the linker left split when
#  they are the same cell at cosine ~1.0.  Mean waveforms are amplitude-PRESERVING
#  (DC removed per channel, then L2 cosine) so the across-channel amplitude profile
#  stays part of identity.  Templates are built on the STANDARD .spk (curation axis),
#  realigned per fragment AND after every merge.
#
#  THRESHOLD / g5 note: the default cos-thr is a conservative 0.99.  Scored against a
#  curated g5 sort, lowering it below ~0.995 fused co-located same-site fragments
#  (their footprints are near-identical, and on sparse g5 the refractory gate is
#  powerless -- it abstains).  BUT that curated sort is INCOMPLETE, so some of the
#  "impure" merges may in fact be correct; treat the GT purity as a lower bound and
#  tune cos-thr on your own review.  A merge is safest where the refractory gate has
#  power (denser sessions) or between genuinely near-identical same-cell chains.
#
#  Knobs read FK_XCM_* (CLI/plan-param > FK_* env > global fiber-kit.yaml > default),
#  so the operating point in fiber-kit-exp.yaml applies on a direct call.  Reuses
#  fiber_ccg.refractory_gate and fiber_geometry; res read variant-aware.  Optional
#  --gt-clu scores purity+completeness like fiber-defrag.
# ════════════════════════════════════════════════════════════════════════════
import argparse
import os
import numpy as np
from collections import defaultdict

try:
    from . import neuro_io as nio, fiber_geometry as fg, fiber_lib as fl, session_yaml as sy, fiber_ccg as cg
    from . import fiber_link_core as flc
    from . import fiber_consolidate as _fc
    from . import config as cfgmod
except ImportError:
    import neuro_io as nio, fiber_geometry as fg, fiber_lib as fl, session_yaml as sy, fiber_ccg as cg
    import fiber_link_core as flc
    import fiber_consolidate as _fc
    import config as cfgmod

_LP = "\u25b8 fiber-xcorr-merge"

_KNOBS = {
    "FK_XCM_COS_THR": ("cos_thr", float, 0.99),
    "FK_XCM_SHIFT": ("shift", int, 4),
    "FK_XCM_REFRAC_MS": ("refrac_ms", float, 2.0),
    "FK_XCM_REFRAC_THR": ("refrac_thr", float, 0.3),
    "FK_XCM_REFRAC_MIN_EXP": ("refrac_min_exp", float, 5.0),
    "FK_XCM_MIN_N": ("min_n", int, 40),
    "FK_XCM_SPK_CAP": ("spk_cap", int, 300),
    "FK_XCM_CX_SCALE": ("complexity_scale", float, 0.0),
    "FK_XCM_BAND_THR": ("band_thr", float, 0.5),
    "FK_XCM_OFF_THR": ("off_thr", float, 0.0),
    "FK_XCM_OFF_AMP_FRAC": ("off_amp_frac", float, 0.3),
    # how a cluster's spikes are aligned before its template is taken, at the start and after every
    # merge: peak = each spike to its own peak; klusters = Klusters' realign (xcorr to the cluster template)
    "FK_XCM_MERGE_ALIGN": ("merge_align", str, "peak"),
    # --merge-align klusters only: a spike is moved to its best lag only if its normalised xcorr with the
    # template reaches this.  Klusters' own default is 0.70; with the template pre-rolled to
    # peakSampleIndex - 1 while the raw trough sits 2-3 samples later, every spike below the threshold
    # stays 2-3 samples behind the rest of its cluster (g6 0316: 35-94% of the spikes of dirty units),
    # which widens the template and makes a false second mode.  0.10 -- the curator's Klusters setting and
    # fiber-realign's FK_REALIGN_MIN_SCORE in the exp plan -- moves all but <= 2.4% of them.
    "FK_XCM_MERGE_ALIGN_MIN_SCORE": ("merge_align_min_score", float, 0.10),
    # own-gain spread veto (raw/standard waveforms): g = sum(w*x*T)/sum(w*T^2), w=|T| -- the Klusters strip
    # gain of each spike against its cluster's median template.  Two co-shaped cells at different amplitudes
    # widen the IQR of g.  giqr_thr>0: refuse a merge whose merged IQR exceeds it; giqr_rise>0: refuse a
    # merge whose merged IQR exceeds the larger of the two parts' by more than this.  0 = off.
    "FK_XCM_GIQR_THR": ("giqr_thr", float, 0.0),
    "FK_XCM_GIQR_RISE": ("giqr_rise", float, 0.0),
    # per-channel RESIDUAL-VARIANCE UNIFORMITY veto: residual = aligned spike - merged template, within
    # +/- rvu_win samples of the peak; per channel c the residual variance v_c(t) over the window should be
    # ~flat (noise).  Two co-shaped cells at different amplitudes leave a residual that scales with the
    # waveform, so v_c(t) peaks with the spike.  Non-uniformity = std(v_c)/mean(v_c) over the window.
    # rvu_thr>0: refuse a merge if ANY channel's non-uniformity exceeds it (each channel has its own veto);
    # rvu_rise>0: refuse if any channel exceeds the larger of the two parts' on that channel by more than
    # this.  0 = off.
    "FK_XCM_RVU_THR": ("rvu_thr", float, 0.0),
    "FK_XCM_RVU_RISE": ("rvu_rise", float, 0.0),
    "FK_XCM_RVU_WIN": ("rvu_win", int, 12),
    # rvu_knn=1: instead of vetoing a candidate merge that fails the residual-variance test, run a
    # WAVEFORM KNN SPLIT on each of the two candidate clusters (fiber_consolidate.knn_pass -- Klusters'
    # split-by-knn-vs-references with own-cluster exclusion) against the other one and its nearest
    # clusters by roll cosine (rvu_knn_refs references in all):
    # spike buckets that vote (>= rvu_knn_vote of rvu_knn_k neighbours) for a reference fold into it when
    # their median matches it in shape AND amplitude, else become a new cluster.  The two parts are rebuilt
    # from what stays, and the merge is retried once; if it still fails, the pair is vetoed.
    "FK_XCM_RVU_KNN": ("rvu_knn", int, 0),
    "FK_XCM_RVU_KNN_K": ("rvu_knn_k", int, 10),
    "FK_XCM_RVU_KNN_VOTE": ("rvu_knn_vote", float, 0.6),
    "FK_XCM_RVU_KNN_REFS": ("rvu_knn_refs", int, 8),
    "FK_XCM_RVU_KNN_FOLD": ("rvu_knn_fold", float, 0.9),
    "FK_XCM_RVU_KNN_MAX": ("rvu_knn_max", int, 20000),   # candidates larger than this are vetoed, not split (memory)
    # drift transform emitted from this stage's own cross-chunk merges (see the --emit-drift block in
    # main): 1 = fit + write <base>.fibers.<method>.<group>.<out-stage> for fiber-backbone-link
    # --drift-fibers, 0 = off.  drift_k is the transform dimensionality; drift_chunk_min must match the
    # chunking the consumer bins by.
    "FK_XCM_EMIT_DRIFT": ("emit_drift", int, 1),
    "FK_XCM_DRIFT_K": ("drift_k", int, 6),
    "FK_XCM_DRIFT_CHUNK_MIN": ("drift_chunk_min", float, 12.0),
}


def _knob_default(name, typ, fb, gcfg):
    v = gcfg.get(name)
    if v in (None, ""):
        v = os.environ.get(name, "")
    return fb if v in (None, "") else typ(v)


def _klusters_align(w, ref_sample, min_score=0.10):
    """Klusters' per-cluster realign of a spike sample (fiber_realign.klusters_offsets: iterative
    normalised xcorr to the cluster's own truncated-mean template, pre-rolled to peakSampleIndex - 1,
    threshold `min_score` (Klusters' default is 0.70; see FK_XCM_MERGE_ALIGN_MIN_SCORE), 2 passes), applied as the circular roll Klusters' working copy uses.  In memory only:
    a template is built from it, nothing is written."""
    try:
        from . import fiber_realign as fr
    except ImportError:
        import fiber_realign as fr
    w = np.asarray(w)
    if len(w) < 2:
        return w.astype(float)
    _off, ioff = fr.klusters_offsets(w, np.full(len(w), 2), int(ref_sample) - 1, min_n=1,
                                    min_score=min_score)
    return fr.roll_spikes(w, ioff).astype(float)


def _tmpl(spk, idx, *, cap, ref_sample, rng, align="peak", rvu_win=12, align_min_score=0.10):
    """Realigned median template of a cluster (STANDARD spk).  Called for every cluster
    and again after every merge -- this is the 'realignment after each merger'.

    align='peak': each spike to its own peak (fiber_lib.realign); align='klusters': the sample is
    aligned as Klusters' realign aligns a cluster -- every spike to the cluster's template by xcorr --
    so a merged cluster whose fragments sat at different offsets gets one consistent alignment."""
    if idx.size > cap:
        idx = rng.choice(idx, cap, replace=False)
    raw = np.asarray(spk[np.sort(idx)])
    al = _klusters_align(raw, ref_sample, align_min_score) if align == "klusters" else fl.realign(np.asarray(raw, float))
    w = fg.mutual_center_spikes(fg.denoise(al), ref_sample=ref_sample)
    T = np.median(w, 0)
    # median template, per-sample sigma (band co-gate), own-gain IQR, per-channel residual-variance non-uniformity
    return T, w.std(0), _gain_iqr(w, T), _resid_var_nonuniformity(w, T, ref_sample, rvu_win)


def _resid_var_nonuniformity(w, T, ref_sample, win):
    """Per channel: std/mean over samples ref_sample +/- win of the residual variance var_spikes(w - T)."""
    lo, hi = max(0, ref_sample - win), min(T.shape[0], ref_sample + win + 1)
    v = np.var(np.asarray(w, float)[:, lo:hi, :] - T[None, lo:hi, :], axis=0)     # (window, nch)
    return v.std(0) / np.maximum(v.mean(0), 1e-12)


def _gain_iqr(w, T):
    """IQR of the Klusters strip gain g = sum(w*x*T)/sum(w*T^2), w=|T|, of the aligned spikes `w` against T."""
    t = _fc._tpl_terms(T)
    if t is None or len(w) < 4:
        return float("nan")
    g = _fc._score_block(np.asarray(w, float).reshape(len(w), -1), t)[1]
    q1, q3 = np.quantile(g, [0.25, 0.75])
    return float(q3 - q1)


def _unit_flat(Tc):
    F = Tc.reshape(Tc.shape[0], -1)
    return F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)


def roll_cos_matrix(T, shift):
    """Symmetric best-over-circular-shift cosine matrix (C[i,j] = max_s cos(T_i, roll(T_j, s)))."""
    Tc = T - T.mean(axis=1, keepdims=True)
    A = _unit_flat(Tc)
    best = np.full((T.shape[0], T.shape[0]), -1.0)
    for s in range(-shift, shift + 1):
        np.maximum(best, A @ _unit_flat(np.roll(Tc, s, axis=1)).T, out=best)
    return np.maximum(best, best.T)


def roll_cos_row(t, T, shift):
    """max_s cos(t, roll(T_k, s)) for every k -> (m,)."""
    tc = t - t.mean(0)
    a = tc.ravel(); a = a / (np.linalg.norm(a) + 1e-12)
    Tc = T - T.mean(axis=1, keepdims=True)
    best = np.full(T.shape[0], -1.0)
    for s in range(-shift, shift + 1):
        np.maximum(best, _unit_flat(np.roll(Tc, s, axis=1)) @ a, out=best)
    return best


def agglomerate(spk, ids, idx0, times0, duration, *, cos_thr, shift, refrac, refrac_thr,
                refrac_min_exp, censor, cap, ref_sample, rng, cx_scale=0.0, band_thr=0.0,
                off_thr=0.0, off_amp_frac=0.3, align="peak", align_min_score=0.10, giqr_thr=0.0, giqr_rise=0.0, rvu_thr=0.0, rvu_rise=0.0, rvu_win=12,
                rvu_knn=0, res=None, knn_k=10, knn_vote=0.6, knn_refs=8, knn_fold=0.9, knn_max=20000, min_n=15,
                next_id=None, stats=None):
    """Confidence-ordered roll-cosine agglomeration with realign-after-merge + refractory veto.
    If cx_scale>0 the required cosine is raised for LOW-complexity (shift-insensitive) pairs, whose
    high roll-shift cosine is weak evidence.  Returns (mapping id->group id, n_merge, n_veto).

    off_thr>0 adds an INTER-CHANNEL-TIMING co-veto (fiber_geometry.interchannel_offsets +
    offset_distance): the roll-cosine already scores the amplitude FOOTPRINT, but two co-located
    cells at slightly different positions can match in footprint while differing in per-channel
    spike timing -- the drift-robust half of identity.  A merge is refused when the RMS inter-channel
    offset difference of the two mean templates exceeds off_thr samples.  Requires RAW/standard
    templates (which this module builds), because the stderiv trough timing is noisy; the offsets
    are recomputed after each merge, like the template.  Scale note: this uses the "xcorr" method,
    so off_thr is on that lag scale (~0.2-0.3), NOT the trough scale (~1.0).  0 = off (unchanged)."""
    m0 = len(ids)
    m = m0 + (max(64, m0) if rvu_knn else 0)               # spare slots for clusters the knn split creates
    if rvu_knn and res is None:
        raise ValueError("rvu_knn needs the spike times (res)")
    _ms = [_tmpl(spk, idx0[k], cap=cap, ref_sample=ref_sample, rng=rng, align=align, rvu_win=rvu_win,
                 align_min_score=align_min_score)
           for k in range(m0)]
    T = np.zeros((m,) + _ms[0][0].shape); SD = np.zeros_like(T)
    GI = np.full(m, np.nan); RV = np.zeros((m, T.shape[2]))
    for k, x in enumerate(_ms):
        T[k], SD[k], GI[k], RV[k] = x
    n_gveto = n_rveto = 0
    C = np.full((m, m), -np.inf); C[:m0, :m0] = roll_cos_matrix(T[:m0], shift)
    np.fill_diagonal(C, -np.inf)
    cx = np.zeros(m) if cx_scale > 0 else None
    if cx is not None:
        cx[:m0] = [fg.waveform_complexity(T[k]) for k in range(m0)]
    cx_ref = (float(np.median(cx[:m0])) + 1e-9) if cx is not None else 1.0
    off = ([fg.interchannel_offsets(T[k], amp_frac=off_amp_frac, method="xcorr") for k in range(m0)] + [None] * (m - m0)
           if off_thr > 0 else None)
    sid = list(ids) + [None] * (m - m0)                   # output id of every slot
    nid = int(next_id if next_id is not None else max(ids) + 1)
    parent = list(range(m)); alive = np.zeros(m, bool); alive[:m0] = True; used = m0
    idx = [idx0[k].copy() for k in range(m0)] + [np.zeros(0, np.int64)] * (m - m0)
    tt = [times0[k] for k in range(m0)] + [np.zeros(0)] * (m - m0)
    vetoed = set(); n_merge = n_veto = 0
    knn_tried = set(); n_knn = n_knn_moved = n_knn_new = 0

    def _refresh(k):
        """Rebuild slot k's template / stats / cosine row from idx[k] (drops it if too small)."""
        if idx[k].size < 4:
            alive[k] = False; C[k, :] = -np.inf; C[:, k] = -np.inf; return
        T[k], SD[k], GI[k], RV[k] = _tmpl(spk, idx[k], cap=cap, ref_sample=ref_sample, rng=rng, align=align,
                                          rvu_win=rvu_win, align_min_score=align_min_score)
        tt[k] = np.sort(res[idx[k]]) if res is not None else tt[k]
        if cx is not None:
            cx[k] = fg.waveform_complexity(T[k])
        if off is not None:
            off[k] = fg.interchannel_offsets(T[k], amp_frac=off_amp_frac, method="xcorr")
        ck = roll_cos_row(T[k], T, shift); ck[~alive] = -np.inf; ck[k] = -np.inf
        C[k, :] = ck; C[:, k] = ck

    def _knn_split_one(src_k, partner):
        """Waveform knn split of cluster src_k against its partner and its nearest other clusters
        (Klusters' split-by-knn-vs-references: the source is one cluster, the others are references)."""
        nonlocal used, nid, n_knn_moved, n_knn_new
        src = idx[src_k]
        if src.size > knn_max or src.size < 2 * min_n:
            return 0
        row = C[src_k].copy(); row[[src_k, partner]] = -np.inf; row[~alive] = -np.inf
        refs = [partner] + [int(r) for r in np.argsort(-row)[:max(knn_refs - 1, 0)] if np.isfinite(row[r])]
        pool = [rng.choice(idx[r], min(idx[r].size, cap), replace=False) for r in refs]
        loc = np.concatenate([src] + pool)
        lab = np.concatenate([np.zeros(src.size, int)] + [np.full(p.size, q + 1) for q, p in enumerate(pool)])
        new, _, _, _ = _fc.knn_pass(lambda ix: np.asarray(spk[loc[np.asarray(ix)]], np.float32), lab,
                                    np.arange(loc.size), k=knn_k, thr=knn_vote, minref=min(min_n, 15),
                                    minnew=min_n, fold_thr=knn_fold, scorr=1.0, exclude=(), next_id=len(refs) + 1)
        ns = new[:src.size]
        moved = ns != 0
        if not moved.any():
            return 0
        for v in np.unique(ns[moved]):
            sp = src[ns == v]
            if 1 <= v <= len(refs):                        # folds into an existing reference cluster
                r = refs[v - 1]; idx[r] = np.concatenate([idx[r], sp]); _refresh(r)
            elif used < m:                                 # a distinct bucket: a new cluster
                k = used; used += 1; idx[k] = sp; sid[k] = nid; nid += 1; alive[k] = True; n_knn_new += 1
                _refresh(k)
            else:                                          # out of spare slots: leave the bucket in place
                moved[ns == v] = False
        idx[src_k] = src[~moved]; _refresh(src_k)
        n_knn_moved += int(moved.sum())
        return int(moved.sum())

    def _knn_split(i, j):
        """Split each part of the candidate i+j against the other part and its neighbours."""
        n = _knn_split_one(i, j)
        if alive[j] and alive[i]:
            n += _knn_split_one(j, i)
        elif alive[j]:
            n += _knn_split_one(j, int(np.argmax(C[j])))
        return n
    while True:
        k = int(np.argmax(C)); val = float(C.flat[k])
        if val < cos_thr:
            break
        i, j = divmod(k, m)
        if i > j:
            i, j = j, i
        if cx is not None:                                 # complexity-scaled threshold: simpler pairs must match harder
            thr_eff = cos_thr + cx_scale * (1.0 - cos_thr) * max(0.0, 1.0 - min(cx[i], cx[j]) / cx_ref)
            if val < thr_eff:
                C[i, j] = C[j, i] = -np.inf; continue
        if (i, j) in vetoed:
            C[i, j] = C[j, i] = -np.inf; continue
        if refrac > 0 and cg.refractory_gate(tt[i], tt[j], duration, refrac, thr=refrac_thr,
                                              min_exp=refrac_min_exp, censor=censor)["verdict"] == "veto":
            vetoed.add((i, j)); C[i, j] = C[j, i] = -np.inf; n_veto += 1; continue
        if off is not None and fg.offset_distance(off[i], off[j]) > off_thr:  # inter-channel timing disagrees -> distinct cells
            C[i, j] = C[j, i] = -np.inf; n_veto += 1; continue
        if band_thr > 0:                                   # median+/-sigma band-overlap co-gate (backbone-link method)
            bov = fg.band_overlap(T[i], SD[i], T[j], SD[j])
            if not np.isfinite(bov) or bov < band_thr:
                C[i, j] = C[j, i] = -np.inf; continue
        # merge j into i, then RE-ALIGN the combined spikes and recompute the template
        ij = np.concatenate([idx[i], idx[j]])
        Tm, SDm, Gm, RVm = _tmpl(spk, ij, cap=cap, ref_sample=ref_sample, rng=rng, align=align, rvu_win=rvu_win,
                 align_min_score=align_min_score)
        if (giqr_thr > 0 and Gm > giqr_thr) or (giqr_rise > 0 and Gm > np.nanmax([GI[i], GI[j]]) + giqr_rise):
            C[i, j] = C[j, i] = -np.inf; n_gveto += 1; continue   # amplitude-mixed: two co-shaped cells
        if (rvu_thr > 0 and np.any(RVm > rvu_thr)) or \
           (rvu_rise > 0 and np.any(RVm > np.maximum(RV[i], RV[j]) + rvu_rise)):
            if rvu_knn and (i, j) not in knn_tried:        # split the candidate by waveform knn, then retry once
                knn_tried.add((i, j)); n_knn += 1
                if _knn_split(i, j):
                    continue
            C[i, j] = C[j, i] = -np.inf; n_rveto += 1; continue   # some channel's residual variance peaks with the spike
        idx[i] = ij
        T[i], SD[i], GI[i], RV[i] = Tm, SDm, Gm, RVm
        tt[i] = np.sort(np.concatenate([tt[i], tt[j]]))
        alive[j] = False; parent[j] = i; idx[j] = np.zeros(0, np.int64); n_merge += 1
        if cx is not None:
            cx[i] = fg.waveform_complexity(T[i])
        if off is not None:
            off[i] = fg.interchannel_offsets(T[i], amp_frac=off_amp_frac, method="xcorr")
        ci = roll_cos_row(T[i], T, shift); ci[~alive] = -np.inf; ci[i] = -np.inf
        C[i, :] = ci; C[:, i] = ci; C[j, :] = -np.inf; C[:, j] = -np.inf

    def root(x):
        while parent[x] != x:
            x = parent[x]
        return x
    if stats is not None:
        stats["gain_vetoes"] = n_gveto; stats["rvu_vetoes"] = n_rveto
        stats.update(knn_splits=n_knn, knn_moved=n_knn_moved, knn_new=n_knn_new)
        stats["members"] = {sid[k]: idx[k] for k in range(used) if parent[k] == k and idx[k].size}
    return {ids[k]: ids[root(k)] for k in range(m0)}, n_merge, n_veto


def _score(new, gt, keep):
    m = keep & (gt > 1)
    u = new[m]; p = gt[m]
    tot = pur = 0; pp = defaultdict(int); best = defaultdict(int)
    for uu in np.unique(u):
        pl = p[u == uu]; v, c = np.unique(pl, return_counts=True)
        tot += pl.size; pur += c.max()
        for vv, cc in zip(v, c):
            pp[vv] += cc; best[vv] = max(best[vv], cc)
    comp = float(np.mean([best[v] / pp[v] for v in pp])) if pp else float("nan")
    return (pur / tot if tot else float("nan")), comp


def main():
    gcfg = cfgmod.load_global_config()
    ap = argparse.ArgumentParser(prog="fiber-xcorr-merge",
                                 description="Confidence-ordered Klusters roll-shift cosine merge (realign after each merge).")
    ap.add_argument("session"); ap.add_argument("elec", type=int)
    ap.add_argument("--clu-method", default="stderiv")
    ap.add_argument("--clu-stage", "--variant", dest="clu_stage", default="backbone_linked",
                    help="input .clu stage tag (e.g. the fiber-backbone-link output)")
    ap.add_argument("--in-clu", default=None, help="explicit input .clu path (overrides --clu-method/--clu-stage)")
    ap.add_argument("--spk-method", "--spk-variant", dest="spk_variant", default="standard", help="waveform axis for templates (curation axis)")
    ap.add_argument("--out-stage", "--out-tag", dest="out_tag", default="xcorr_merged",
                    help="post-fiber stage tag of the output .clu")
    ap.add_argument("--refrac-censor-ms", type=float, default=0.0, help="detection censor window (ms)")
    ap.add_argument("--nsamp", type=int, default=None, help="override; default from <session>.yaml")
    ap.add_argument("--nchan", type=int, default=None, help="override; default from <session>.yaml")
    ap.add_argument("--ref-sample", type=int, default=None, help="override; default = peak from <session>.yaml")
    ap.add_argument("--gt-stage", "--gt-clu", dest="gt_clu", default=None,
                    help="post-fiber stage tag (or path) of the curated .clu to score purity+completeness")
    ap.add_argument("--seed", type=int, default=0)
    for name, (dest, typ, fb) in _KNOBS.items():
        ap.add_argument("--" + dest.replace("_", "-"), dest=dest, type=typ, default=_knob_default(name, typ, fb, gcfg),
                        help=f"{name} (default {_knob_default(name, typ, fb, gcfg)})")
    a = ap.parse_args()
    if a.merge_align not in ("peak", "klusters"):
        raise SystemExit(f"{_LP} · --merge-align must be peak or klusters, not {a.merge_align!r}")
    rng = np.random.default_rng(a.seed)

    cfg = sy.resolve_session_params(a.session, a.elec)
    base = cfg["base"]; elec = a.elec; SR = cfg["sr"]
    NS = a.nsamp if a.nsamp else cfg["nsamp"]              # per-group counts from <session>.yaml (groups differ, e.g. 8 vs 9 ch)
    NC = a.nchan if a.nchan else cfg["nchan"]
    PK = a.ref_sample if a.ref_sample is not None else cfg["peak"]
    res = nio.read_res(base, elec)
    if a.in_clu:
        _, clu = nio.read_clu_file(a.in_clu, n_spikes=res.size)
    else:
        _, clu = nio.read_clu_at(base, elec, variant=a.clu_method, tag=a.clu_stage, n_spikes=res.size)
    spk = nio.open_spk_file(nio.session_path(base, "spk", elec, variant=a.spk_variant), NS, NC)
    assert res.size == clu.size == spk.shape[0], "res/clu/spk length mismatch"
    duration = float(res.max()) if res.size else 1.0
    refrac = cg.refrac_samples(a.refrac_ms, SR); censor = cg.refrac_samples(a.refrac_censor_ms, SR)

    ids0, cnt = np.unique(clu[clu > 1], return_counts=True)
    big = [int(u) for u, c in zip(ids0, cnt) if c >= a.min_n]
    print(f"{_LP} · {base} elec {elec} | {a.spk_variant} .spk | {len(big)}/{len(ids0)} clusters >= {a.min_n} spk"
          f" | roll-cos>={a.cos_thr} shift+/-{a.shift} refrac={a.refrac_ms}ms realign-after-merge ({a.merge_align})")
    if len(big) < 2:
        raise SystemExit(f"{_LP} · too few clusters to merge")

    idx0 = [np.flatnonzero(clu == u) for u in big]
    times0 = [np.sort(res[ix]) for ix in idx0]
    _st = {}
    mapping, n_merge, n_veto = agglomerate(spk, big, idx0, times0, duration, cos_thr=a.cos_thr, shift=a.shift,
                                           refrac=refrac, refrac_thr=a.refrac_thr, refrac_min_exp=a.refrac_min_exp,
                                           censor=censor, cap=a.spk_cap, ref_sample=PK, rng=rng,
                                           cx_scale=a.complexity_scale, band_thr=a.band_thr,
                                           off_thr=a.off_thr, off_amp_frac=a.off_amp_frac, align=a.merge_align,
                                           align_min_score=a.merge_align_min_score,
                                           giqr_thr=a.giqr_thr, giqr_rise=a.giqr_rise,
                                           rvu_thr=a.rvu_thr, rvu_rise=a.rvu_rise, rvu_win=a.rvu_win,
                                           rvu_knn=a.rvu_knn, res=res, knn_k=a.rvu_knn_k, knn_vote=a.rvu_knn_vote,
                                           knn_refs=a.rvu_knn_refs, knn_fold=a.rvu_knn_fold, knn_max=a.rvu_knn_max, min_n=a.min_n,
                                           next_id=int(clu.max()) + 1, stats=_st)
    new = clu.copy().astype(np.int64)
    knn_moved = _st.get("knn_moved", 0) > 0
    if knn_moved:                                          # per-spike moves: rebuild from the final members
        for u, ix in _st["members"].items():
            new[ix] = u
    else:
        for u in big:
            if mapping[u] != u:
                new[clu == u] = mapping[u]
    n_in = len(np.unique(clu[clu > 1])); n_out = len(np.unique(new[new > 1]))
    print(f"{_LP} · merged {n_in - n_out} clusters ({n_in} -> {n_out}); merges {n_merge}, refractory vetoes {n_veto}"
          + (f", gain-IQR vetoes {_st.get('gain_vetoes', 0)}" if (a.giqr_thr > 0 or a.giqr_rise > 0) else "")
          + (f", residual-variance vetoes {_st.get('rvu_vetoes', 0)}" if (a.rvu_thr > 0 or a.rvu_rise > 0) else "")
          + (f"; knn splits {_st.get('knn_splits', 0)}: {_st.get('knn_moved', 0)} spikes moved, "
             f"{_st.get('knn_new', 0)} new clusters" if a.rvu_knn else ""))
    nio.write_clu(base, elec, new, variant=a.clu_method, tag=a.out_tag)
    print(f"{_LP} · wrote {os.path.basename(nio.session_path(base, 'clu', elec, variant=a.clu_method, tag=a.out_tag))}")

    # ── carry the .clc/.clp hierarchy through the merge ──
    #   A merge changes only WHICH FIBER a child belongs to, never the atoms, so the child layer is
    #   copied verbatim and each child's parent goes through the SAME mapping the .clu did.  Without
    #   this the out-stage has a .clu with no matching .clc/.clp and the triple cannot be loaded as a
    #   unit (FiberHierarchy.load raises on the missing sibling).
    from .fiber_refiberize import FiberHierarchy as _FH
    try:
        _hier = _FH.load(base, elec, variant=a.clu_method, tag=a.clu_stage)
    except (FileNotFoundError, OSError):
        _hier = None
        print(f"{_LP} · no .clc/.clp at .{a.clu_stage} - wrote .clu only (no hierarchy to carry)")
    if _hier is not None:
        _child = np.asarray(_hier.child, np.int64).copy()
        _newpar = {c: int(mapping.get(int(f), int(f))) for c, f in _hier.parent.items()}
        if knn_moved:                                      # per-spike moves: each (atom -> new cluster) bucket
            nxt_c = int(max(_newpar) if _newpar else 0) + 1    # becomes a child atom of its own, parented to
            mv = np.flatnonzero((_child > 0) & (new != np.array([_newpar.get(int(c), 0) for c in _child])))
            key = {}                                       # the cluster it moved to (Klusters-inspectable)
            for i in mv:
                kk = (int(_child[i]), int(new[i]))
                if kk not in key:
                    key[kk] = nxt_c; _newpar[nxt_c] = kk[1]; nxt_c += 1
                _child[i] = key[kk]
            print(f"{_LP} · knn moves parked as {len(key):,} new child atoms")
        _FH(_child, _newpar).save(base, elec, variant=a.clu_method, tag=a.out_tag,
                                  renumber=False, backup=False)
        print(f"{_LP} · hierarchy carried: {len(_newpar):,} children -> "
              f"{len(set(_newpar.values())):,} fibers (.clc unchanged, .clp remapped)")

    # ── optional: fit + emit the drift transform from OUR OWN cross-chunk merges ──
    #   An accepted merge between clusters living in different chunks IS a same-cell correspondence --
    #   the xcorr-merge analog of fiber-session's overlap anchors -- so this stage can fit the very same
    #   drift object without fiber-session having linked at all.  Shared primitive (flc.fit_drift_transforms),
    #   shared key names, so fiber-backbone-link consumes it unchanged via --drift-fibers.
    if a.emit_drift:
        if a.spk_variant != "standard":
            print(f"{_LP} · --emit-drift skipped: the drift basis is RAW (stderiv breaks the "
                  f"amplitude-distance law and mutes drift), got {a.spk_variant}")
        else:
            K = int(a.drift_k)
            chunk_s = float(a.drift_chunk_min) * 60.0 * SR
            t0 = float(res.min()) if res.size else 0.0
            ch_of = {u: int((float(np.median(res[ix])) - t0) // chunk_s) for u, ix in zip(big, idx0)}
            nP = max((max(ch_of.values()) + 1 if ch_of else 0) - 1, 0)
            grp = defaultdict(list)
            for u in big:
                grp[mapping[u]].append(u)
            anchor_links = [[] for _ in range(nP)]
            for _g, ms in grp.items():                       # every merged group = one cell
                byc = defaultdict(list)
                for u in ms:
                    byc[ch_of[u]].append(u)
                for c in sorted(byc):
                    if c + 1 in byc and 0 <= c < nP:
                        for f in byc[c]:
                            for g2 in byc[c + 1]:
                                anchor_links[c].append((f, g2))
            npair = sum(len(x) for x in anchor_links)
            # RAW mean templates -- the same recipe fiber-session fits on (realign, no denoise/centering)
            rt = {}
            for u, ix in zip(big, idx0):
                sel = np.sort(rng.choice(ix, a.spk_cap, replace=False) if ix.size > a.spk_cap else ix)
                rt[(ch_of[u], u)] = fl.realign(np.asarray(spk[sel], float)).mean(0).ravel()
            if len(rt) >= K + 2 and npair:
                X = np.array(list(rt.values())); mu = X.mean(0)
                basis = np.linalg.svd(X - mu, full_matrices=False)[2][:K]
                pos = {k: (v - mu) @ basis.T for k, v in rt.items()}
                R, t, resid, na = flc.fit_drift_transforms(pos, anchor_links, K)
                dpath = nio.fibers_path(base, a.clu_method, elec, stage=a.out_tag)
                # Through a file handle, as fiber-session writes its .fibers: np.savez(<path>) appends
                # '.npz', so the table landed at <path>.npz while fiber-pipeline hands backbone-link
                # the canonical <path> -- and the drift step was silently skipped.
                with open(dpath, "wb") as _fh:
                    np.savez(_fh, drift_pca_basis=basis.astype(np.float32), drift_pca_mean=mu.astype(np.float32),
                             drift_R=R.astype(np.float32), drift_t=t.astype(np.float32),
                             drift_resid=resid.astype(np.float32), drift_nanchor=na,
                             meta_drift_k=np.array(K), meta_chunk_min=np.array(float(a.drift_chunk_min)))
                _med = float(np.nanmedian(resid)) if np.isfinite(resid).any() else float("nan")
                print(f"{_LP} · drift fit from {npair} cross-chunk merges: {int(np.sum(na > 0))}/{max(nP, 1)} "
                      f"chunk-pairs fit, median residual {_med:.2f}, anchors/pair median "
                      f"{int(np.median(na)) if na.size else 0}")
                print(f"{_LP} · wrote {os.path.basename(dpath)}  (fiber-backbone-link --drift-fibers)")
            else:
                print(f"{_LP} · --emit-drift: too few cross-chunk merges to fit ({npair} anchor pairs, "
                      f"{len(rt)} templates; need >= {K + 2} templates and >= 1 pair)")

    if a.gt_clu:
        _, gt = (nio.read_clu_file(a.gt_clu, n_spikes=res.size) if (os.path.sep in a.gt_clu or a.gt_clu.endswith(".clu"))
                 else nio.read_clu_at(base, elec, variant=a.clu_method, tag=a.gt_clu, n_spikes=res.size))
        keep = np.isin(clu, np.array(big))
        for tag, lab in (("before", clu.astype(np.int64)), ("after ", new)):
            pur, comp = _score(lab, np.asarray(gt), keep)
            print(f"{_LP} · GT {tag} (INCOMPLETE): spike-weighted purity {pur:.3f} | per-parent completeness {comp:.3f}")


if __name__ == "__main__":
    main()
