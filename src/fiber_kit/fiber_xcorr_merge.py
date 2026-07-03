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
    from . import config as cfgmod
except ImportError:
    import neuro_io as nio, fiber_geometry as fg, fiber_lib as fl, session_yaml as sy, fiber_ccg as cg
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
}


def _knob_default(name, typ, fb, gcfg):
    v = gcfg.get(name)
    if v in (None, ""):
        v = os.environ.get(name, "")
    return fb if v in (None, "") else typ(v)


def _tmpl(spk, idx, *, cap, ref_sample, rng):
    """Realigned median template of a cluster (STANDARD spk).  Called for every cluster
    and again after every merge -- this is the 'realignment after each merger'."""
    if idx.size > cap:
        idx = rng.choice(idx, cap, replace=False)
    w = fg.mutual_center_spikes(fg.denoise(fl.realign(np.asarray(spk[np.sort(idx)], float))), ref_sample=ref_sample)
    return np.median(w, 0)


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
                refrac_min_exp, censor, cap, ref_sample, rng, cx_scale=0.0):
    """Confidence-ordered roll-cosine agglomeration with realign-after-merge + refractory veto.
    If cx_scale>0 the required cosine is raised for LOW-complexity (shift-insensitive) pairs, whose
    high roll-shift cosine is weak evidence.  Returns (mapping id->group id, n_merge, n_veto)."""
    m = len(ids)
    T = np.stack([_tmpl(spk, idx0[k], cap=cap, ref_sample=ref_sample, rng=rng) for k in range(m)])
    C = roll_cos_matrix(T, shift)
    np.fill_diagonal(C, -np.inf)
    cx = np.array([fg.waveform_complexity(T[k]) for k in range(m)]) if cx_scale > 0 else None
    cx_ref = (float(np.median(cx)) + 1e-9) if cx is not None else 1.0
    parent = list(range(m)); alive = np.ones(m, bool)
    idx = [idx0[k].copy() for k in range(m)]; tt = [times0[k] for k in range(m)]
    vetoed = set(); n_merge = n_veto = 0
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
        # merge j into i, then RE-ALIGN the combined spikes and recompute the template
        idx[i] = np.concatenate([idx[i], idx[j]])
        T[i] = _tmpl(spk, idx[i], cap=cap, ref_sample=ref_sample, rng=rng)
        tt[i] = np.sort(np.concatenate([tt[i], tt[j]]))
        alive[j] = False; parent[j] = i; n_merge += 1
        if cx is not None:
            cx[i] = fg.waveform_complexity(T[i])
        ci = roll_cos_row(T[i], T, shift); ci[~alive] = -np.inf; ci[i] = -np.inf
        C[i, :] = ci; C[:, i] = ci; C[j, :] = -np.inf; C[:, j] = -np.inf

    def root(x):
        while parent[x] != x:
            x = parent[x]
        return x
    return {ids[k]: ids[root(k)] for k in range(m)}, n_merge, n_veto


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
    ap.add_argument("--spk-variant", default="standard", help="waveform axis for templates (curation axis)")
    ap.add_argument("--out-tag", default="xcorr_merged", help="output .clu stage tag")
    ap.add_argument("--refrac-censor-ms", type=float, default=0.0, help="detection censor window (ms)")
    ap.add_argument("--nsamp", type=int, default=42); ap.add_argument("--nchan", type=int, default=8)
    ap.add_argument("--ref-sample", type=int, default=21)
    ap.add_argument("--gt-clu", default=None, help="curated .clu tag/path to score purity+completeness")
    ap.add_argument("--seed", type=int, default=0)
    for name, (dest, typ, fb) in _KNOBS.items():
        ap.add_argument("--" + dest.replace("_", "-"), dest=dest, type=typ, default=_knob_default(name, typ, fb, gcfg),
                        help=f"{name} (default {_knob_default(name, typ, fb, gcfg)})")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    cfg = sy.resolve_session_params(a.session, a.elec)
    base = cfg["base"]; elec = a.elec; SR = cfg["sr"]
    res = nio.read_res(base, elec)
    if a.in_clu:
        _, clu = nio.read_clu_file(a.in_clu, n_spikes=res.size)
    else:
        _, clu = nio.read_clu_at(base, elec, variant=a.clu_method, tag=a.clu_stage, n_spikes=res.size)
    spk = nio.open_spk_file(nio.session_path(base, "spk", elec, variant=a.spk_variant), a.nsamp, a.nchan)
    assert res.size == clu.size == spk.shape[0], "res/clu/spk length mismatch"
    duration = float(res.max()) if res.size else 1.0
    refrac = cg.refrac_samples(a.refrac_ms, SR); censor = cg.refrac_samples(a.refrac_censor_ms, SR)

    ids0, cnt = np.unique(clu[clu > 1], return_counts=True)
    big = [int(u) for u, c in zip(ids0, cnt) if c >= a.min_n]
    print(f"{_LP} · {base} elec {elec} | {a.spk_variant} .spk | {len(big)}/{len(ids0)} clusters >= {a.min_n} spk"
          f" | roll-cos>={a.cos_thr} shift+/-{a.shift} refrac={a.refrac_ms}ms realign-after-merge")
    if len(big) < 2:
        raise SystemExit(f"{_LP} · too few clusters to merge")

    idx0 = [np.flatnonzero(clu == u) for u in big]
    times0 = [np.sort(res[ix]) for ix in idx0]
    mapping, n_merge, n_veto = agglomerate(spk, big, idx0, times0, duration, cos_thr=a.cos_thr, shift=a.shift,
                                           refrac=refrac, refrac_thr=a.refrac_thr, refrac_min_exp=a.refrac_min_exp,
                                           censor=censor, cap=a.spk_cap, ref_sample=a.ref_sample, rng=rng,
                                           cx_scale=a.complexity_scale)
    new = clu.copy().astype(np.int64)
    for u in big:
        if mapping[u] != u:
            new[clu == u] = mapping[u]
    n_in = len(np.unique(clu[clu > 1])); n_out = len(np.unique(new[new > 1]))
    print(f"{_LP} · merged {n_in - n_out} clusters ({n_in} -> {n_out}); merges {n_merge}, refractory vetoes {n_veto}")
    nio.write_clu(base, elec, new, variant=a.clu_method, tag=a.out_tag)
    print(f"{_LP} · wrote {os.path.basename(nio.session_path(base, 'clu', elec, variant=a.clu_method, tag=a.out_tag))}")

    if a.gt_clu:
        _, gt = (nio.read_clu_file(a.gt_clu, n_spikes=res.size) if (os.path.sep in a.gt_clu or a.gt_clu.endswith(".clu"))
                 else nio.read_clu_at(base, elec, variant=a.clu_method, tag=a.gt_clu, n_spikes=res.size))
        keep = np.isin(clu, np.array(big))
        for tag, lab in (("before", clu.astype(np.int64)), ("after ", new)):
            pur, comp = _score(lab, np.asarray(gt), keep)
            print(f"{_LP} · GT {tag} (INCOMPLETE): spike-weighted purity {pur:.3f} | per-parent completeness {comp:.3f}")


if __name__ == "__main__":
    main()
