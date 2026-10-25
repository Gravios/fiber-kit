#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════════════════════
#  fiber_lowmix.py — split low-amplitude units that are MIXTURES of sources,
#  return the parts that are copies of existing units, quarantine the rest.
#
#  Finding (g6 reference session, fix_final; docs/methods.md "Low-amplitude
#  mixtures"): a low-amplitude unit is mostly not a weak unit.  Its spikes split
#  into groups of DIFFERENT waveform shape (noise-corrected cosine between groups
#  median 0.63), each larger than the unit's own median (x1.25) -- averaging
#  differently shaped sources cancels, which is what made the unit look small.
#  A third of the groups are copies (cos >= 0.95, amplitude ratio ~1) of another
#  unit; a quarter match no unit.  The groups are invisible in the raw space: the
#  shared structure of low-amplitude spikes is the background (spike-free noise
#  snippets carry it as strongly), so the split is done after WHITENING with the
#  spatio-temporal covariance of spike-free .fil snippets.
#
#  Per selected unit:
#    1. waveforms: standard .spk, baseline (first samples) removed, per-channel
#       Legendre smoothing (degree chosen on split halves: 20 for 42 samples);
#    2. whiten with the noise covariance, PCA(3), GMM with k = BIC-best <= k_max;
#    3. NULL: the unit's own template with each spike's fitted amplitude, timing
#       (d/dt) and position (channel-gradient) coefficients + real noise snippets
#       scaled to the residual energy, through the identical path.  The unit is
#       split only if the real BIC gain per spike beats the null's
#       (gain > null_mult * null + null_add);
#    4. each group vs every unit template (+-max_shift samples):
#         copy   cos >= copy_cos and the refractory test does not veto the move
#                -> spikes return to that unit
#         keep   the largest non-copy group keeps the unit's id; other groups
#                with cos >= novel_cos become new units
#         novel  cos < novel_cos -> quarantined to noise (1) by default, kept as
#                its own child (atom) in the .clc so it stays reviewable.
#
#  Writes .clu (+ .clc/.clp when the input stage has a hierarchy) and a
#  <base>.lmx.<group>.<tag> table, one row per group.
#  Knobs FK_LOWMIX_* (CLI > env > $FK_CONFIG > default).
# ════════════════════════════════════════════════════════════════════════════
import argparse
import os
import sys

import numpy as np
from numpy.polynomial import legendre as _leg

try:
    from . import neuro_io as nio, session_yaml as sy, config as cfgmod
except ImportError:                                    # run as a script
    import neuro_io as nio
    import session_yaml as sy
    import config as cfgmod

RESERVE = (0, 1)


# ── waveform primitives ──────────────────────────────────────────────────────
def smoother(ns, deg):
    """(ns, ns) projector onto Legendre polynomials of degree <= deg over the window."""
    if deg <= 0 or deg >= ns - 1:
        return np.eye(ns)
    V = _leg.legvander(np.linspace(-1, 1, ns), int(deg))
    return V @ np.linalg.pinv(V)


def prep(W, P, n_base=6):
    """int16 (n, ns, nc) -> float32, per-channel baseline (median of the first n_base samples) removed,
    Legendre-smoothed."""
    W = np.asarray(W, np.float32)
    W = W - np.median(W[:, :n_base], axis=1, keepdims=True)
    return np.einsum("st,itc->isc", P, W).astype(np.float32)


def noise_snippets(read_rows, n_rows, res, ns, n, rng, guard=None):
    """n spike-free windows of ns samples: start times at least `guard` samples (default ns) from any
    detected spike.  read_rows(a, b) -> (b-a, nc) float array of the group's channels."""
    guard = ns if guard is None else int(guard)
    r = np.sort(np.asarray(res, np.int64))
    out = []
    tries = 0
    while len(out) < n and tries < 50:
        t = rng.integers(guard, max(guard + 1, n_rows - ns - guard), size=4 * n)
        j = np.searchsorted(r, t)
        prev = np.where(j > 0, t - r[np.maximum(j - 1, 0)], np.iinfo(np.int64).max)
        nxt = np.where(j < r.size, r[np.minimum(j, r.size - 1)] - t, np.iinfo(np.int64).max)
        ok = (prev > guard) & (nxt > ns + guard)
        out.extend(t[ok][: n - len(out)].tolist())
        tries += 1
    return np.stack([read_rows(s, s + ns) for s in out]) if out else np.zeros((0, ns, 0), np.float32)


def whitener(noise_flat, floor=1e-4):
    """Whitening map (D, d) from the noise covariance; directions with variance below floor*max (the
    Legendre-smoothed noise has rank < D) are dropped."""
    C = np.cov(np.asarray(noise_flat, np.float64).T)
    ev, E = np.linalg.eigh(C)
    keep = ev > ev.max() * floor
    return (E[:, keep] / np.sqrt(ev[keep])).astype(np.float64)


def nuisance_basis(w):
    """template, its time derivative, and the position-shift term s(t) b'(c) of its rank-1 factors."""
    U, s, Vt = np.linalg.svd(w, full_matrices=False)
    return np.stack([w, np.gradient(w, axis=0), np.outer(U[:, 0] * s[0], np.gradient(Vt[0]))]).reshape(3, -1)


def nuisance_fit(W, w):
    """Per-spike LS fit of the nuisance basis; returns (fitted, residual) as (n, D) arrays."""
    N = nuisance_basis(w)
    X = W.reshape(len(W), -1)
    a = np.linalg.lstsq(N.T, X.T, rcond=None)[0].T
    M = a @ N
    return M, X - M


# ── the split ────────────────────────────────────────────────────────────────
def _gmm_bic(F, k_max, seed):
    from sklearn.mixture import GaussianMixture
    fits = [GaussianMixture(k, random_state=seed).fit(F) for k in range(1, k_max + 1)]
    bic = np.array([g.bic(F) for g in fits])
    k = int(np.argmin(bic))
    return fits[k], k + 1, float((bic[0] - bic[k]) / len(F))


def _features(Z, Wh, ref):
    """whitened, PCA(3)-reduced, unit-variance features; `ref` (from the real spikes) fixes the
    projection so real and null are measured the same way."""
    Zw = (Z - ref["centre"]) @ Wh
    if "p" not in ref:
        mu = Zw.mean(0)
        ref["mu"] = mu
        ref["p"] = np.linalg.svd(Zw[: ref.get("n_pca", 4000)] - mu, full_matrices=False)[2][:3]
    F = (Zw - ref["mu"]) @ ref["p"].T
    if "sd" not in ref:
        ref["sd"] = F.std(0) + 1e-12
    return F / ref["sd"]


def split_unit(W, noise_flat, Wh, *, k_max=4, n_fit=3000, null_mult=2.0, null_add=0.02, seed=0):
    """W (n, ns, nc) prepped waveforms of one unit.  Returns dict(labels, k, gain, gain_null, split)."""
    rng = np.random.default_rng(seed)
    X = W.reshape(len(W), -1).astype(np.float64)
    if len(X) < 4 * k_max:
        return dict(labels=np.zeros(len(X), int), k=1, gain=0.0, gain_null=0.0, split=False)
    ref = dict(centre=np.median(X, 0))
    F = _features(X, Wh, ref)
    sub = rng.choice(len(F), min(n_fit, len(F)), replace=False)
    gm, k, gain = _gmm_bic(F[sub], k_max, seed)
    # null: same per-spike amplitude/timing/position, real noise at the residual's energy
    M, R = nuisance_fit(W, np.median(W, 0))
    nz = noise_flat[rng.integers(0, len(noise_flat), len(X))]
    scale = np.sqrt(np.sum(R ** 2) / max(np.sum(nz ** 2), 1e-12))
    Fn = _features(M + scale * nz, Wh, dict(ref))
    _, _, gain_null = _gmm_bic(Fn[sub], k_max, seed)
    split = k > 1 and gain > null_mult * gain_null + null_add
    labels = gm.predict(F) if split else np.zeros(len(X), int)
    return dict(labels=labels, k=k if split else 1, gain=gain, gain_null=gain_null, split=split)


def best_match(m, templates, exclude, max_shift):
    """(cos, unit, shift) of the template closest in shape to m over +-max_shift samples."""
    best = (-1.0, None, 0)
    mv = m.ravel(); nm = np.linalg.norm(mv) + 1e-12
    for v, t in templates.items():
        if v == exclude:
            continue
        for sh in range(-max_shift, max_shift + 1):
            tt = np.roll(t, sh, 0).ravel()
            c = float(mv @ tt / nm / (np.linalg.norm(tt) + 1e-12))
            if c > best[0]:
                best = (c, int(v), sh)
    return best


def _cross(ta, tb, lo, hi):
    """(a, b) pairs with lo < |t_b - t_a| <= hi; tb sorted."""
    up = np.searchsorted(tb, ta + hi, "right") - np.searchsorted(tb, ta + lo, "right")
    dn = np.searchsorted(tb, ta - lo, "left") - np.searchsorted(tb, ta - hi, "left")
    return int(up.sum() + dn.sum())


def _auto(t, lo, hi):
    """ordered pairs within one train with lo < dt <= hi (t sorted)."""
    return int((np.searchsorted(t, t + hi, "right") - np.searchsorted(t, t + lo, "right")).sum())


def local_refractory(t_group, t_target, sr, *, refrac_ms=1.5, censor_ms=0.6, base_ms=(3.0, 6.0),
                     thr=0.3, own_mult=1.5, min_exp=3.0):
    """Would moving t_group into t_target keep the target's refractory dip?
    Coincidences in (censor, refrac] against the LOCAL rate in base_ms (scaled to the same width):
    a uniform-rate expectation under-predicts bursty CA1 trains, so it abstains where the local one
    has power.  The bar is the target's OWN ratio (a curated unit is rarely clean): veto when the
    cross ratio exceeds max(thr, own_mult * own ratio).  Returns dict(verdict, ratio, own, obs, exp)."""
    ta = np.sort(np.asarray(t_group, np.int64)); tb = np.sort(np.asarray(t_target, np.int64))
    c, r_ = censor_ms * sr / 1e3, refrac_ms * sr / 1e3
    b0, b1 = base_ms[0] * sr / 1e3, base_ms[1] * sr / 1e3
    w = (r_ - c) / (b1 - b0)
    obs = _cross(ta, tb, c, r_); exp = _cross(ta, tb, b0, b1) * w
    o_own = _auto(tb, c, r_); e_own = _auto(tb, b0, b1) * w
    own = o_own / e_own if e_own > 0 else np.nan
    ratio = obs / exp if exp > 0 else np.nan
    if exp < min_exp:
        verdict = "abstain"
    elif ratio > max(thr, own_mult * (own if np.isfinite(own) else 0.0)):
        verdict = "veto"
    else:
        verdict = "allow"
    return dict(verdict=verdict, ratio=ratio, own=own, obs=obs, exp=exp)


def decide(groups, *, copy_cos=0.95, novel_cos=0.90, copy_on_abstain=False):
    """groups: list of dicts with n, cos, target, refrac ('allow'|'abstain'|'veto').  Adds 'verdict':
    copy | keep | new | novel.  The largest non-copy group keeps the unit (verdict 'keep').  A shape copy
    returns only on a refractory 'allow' (or 'abstain' with copy_on_abstain); otherwise it stays a new unit."""
    ok = ("allow", "abstain") if copy_on_abstain else ("allow",)
    for g in groups:
        g["verdict"] = "copy" if (g["cos"] >= copy_cos and g["refrac"] in ok) else None
    rest = [g for g in groups if g["verdict"] is None]
    if rest:
        core = max(rest, key=lambda g: g["n"])
        core["verdict"] = "keep"
        for g in rest:
            if g is not core:
                g["verdict"] = "new" if g["cos"] >= novel_cos else "novel"
    return groups


# ── knobs ────────────────────────────────────────────────────────────────────
_KNOBS = {
    "FK_LOWMIX_AMP_QUANTILE": ("amp_quantile", float, 1 / 3,
                               "units at or below this quantile of peak-to-peak amplitude (among units with "
                               ">= --min-n spikes) are examined; 1 = every unit"),
    "FK_LOWMIX_MIN_N": ("min_n", int, 150, "spikes a unit needs to be examined"),
    "FK_LOWMIX_MIN_GROUP": ("min_group", int, 30, "smallest group acted on; smaller groups stay with the unit"),
    "FK_LOWMIX_DEG": ("deg", int, 20, "Legendre degree per channel (20 for 42 samples: split-half optimum)"),
    "FK_LOWMIX_K_MAX": ("k_max", int, 4, "most groups per unit"),
    "FK_LOWMIX_NULL_MULT": ("null_mult", float, 2.0, "split only if BIC gain > NULL_MULT * null gain + NULL_ADD"),
    "FK_LOWMIX_NULL_ADD": ("null_add", float, 0.02, "see NULL_MULT (BIC gain per spike)"),
    "FK_LOWMIX_NOISE_N": ("noise_n", int, 20000, "spike-free .fil snippets for the noise covariance"),
    "FK_LOWMIX_COPY_COS": ("copy_cos", float, 0.95, "shape cosine that makes a group a copy of another unit"),
    "FK_LOWMIX_NOVEL_COS": ("novel_cos", float, 0.90,
                            "below this a non-core group matches no unit: quarantined (--novel noise) or kept"),
    "FK_LOWMIX_MAX_SHIFT": ("max_shift", int, 6, "+-samples searched when matching templates"),
    "FK_LOWMIX_TEMPLATE_MIN_N": ("template_min_n", int, 100, "spikes a unit needs to serve as a match template"),
    "FK_LOWMIX_REFRAC_MS": ("refrac_ms", float, 1.5,
                            "a copy returns only if its coincidences with the target in (0.6, REFRAC] ms stay "
                            "below max(0.3, 1.5 x the target's own ratio), against the local 3-6 ms rate "
                            "(0 = off: shape alone decides)"),
    "FK_LOWMIX_REFRAC_MIN_EXP": ("refrac_min_exp", float, 3.0,
                                 "coincidences that must be expected before the refractory test may veto"),
    "FK_LOWMIX_COPY_ON_ABSTAIN": ("copy_on_abstain", int, 0,
                                  "1 = a shape copy also returns when the refractory test has no power; 0 = it "
                                  "stays its own unit (measured on g6: shape-only copies sit at 0.68 of the local "
                                  "rate against their target -- half of them are not the same neuron)"),
    "FK_LOWMIX_NOVEL": ("novel", str, "noise", "noise = novel groups to noise (1), as a reviewable .clc child; "
                                               "keep = each novel group becomes its own unit"),
    "FK_LOWMIX_SEED": ("seed", int, 0, "seed for subsampling, noise snippets and the GMM"),
}


def _knob_default(name, typ, fallback, gcfg):
    """FK_* env var, then the $FK_CONFIG yaml, then the default (config.knob_value: one precedence
    and one validation for every stage)."""
    return cfgmod.knob_value(name, typ, fallback, gcfg)


def write_output(base, elec, method, in_stage, out_stage, clu, new, child_moves, log=print):
    """.clu, plus the .clc/.clp hierarchy when the input stage has one: every acted-on group becomes
    one new child whose parent is its destination, so all other atoms keep theirs.  The rebuilt .clu
    must reproduce `new` exactly, else only the .clu is written (as fiber-consensus does)."""
    try:
        from .fiber_refiberize import FiberHierarchy
    except ImportError:
        from fiber_refiberize import FiberHierarchy
    try:
        h = FiberHierarchy.load(base, elec, variant=method, tag=in_stage)
    except (FileNotFoundError, OSError):
        log(f"[lowmix] no .clc/.clp at .{in_stage}: writing the .clu only")
        return nio.write_clu(base, elec, new, variant=method, tag=out_stage)
    child = h.child.copy()
    if child.size != new.size:
        log(f"[lowmix] WARNING .clc has {child.size} rows, .clu {new.size}: hierarchy not carried")
        return nio.write_clu(base, elec, new, variant=method, tag=out_stage)
    parent = dict(h.parent)
    nxt = int(child.max()) + 1
    for idx, dest in child_moves:
        child[idx] = nxt; parent[nxt] = int(dest); nxt += 1
    h2 = FiberHierarchy(child, parent)
    rebuilt, _, _ = h2.refiberize(renumber=False)
    if not np.array_equal(rebuilt, new):
        log(f"[lowmix] WARNING hierarchy disagrees with the new .clu on {int((rebuilt != new).sum())} spikes: "
            f".clu only")
        return nio.write_clu(base, elec, new, variant=method, tag=out_stage)
    paths = h2.save(base, elec, variant=method, tag=out_stage, renumber=False, backup=False)
    log(f"[lowmix] hierarchy carried: {len(child_moves)} new children")
    return paths["clu"]


def main(argv=None):
    gcfg = cfgmod.load_global_config()
    ap = argparse.ArgumentParser(description="Split low-amplitude mixture units after noise whitening; "
                                             "return copies to their unit, quarantine novel groups.")
    ap.add_argument("base"); ap.add_argument("elec", type=int)
    ap.add_argument("--signal", required=True, help="continuous filtered int16 file (BASE.fil) for the noise")
    ap.add_argument("--signal-start", type=int, default=0, help="absolute sample of the signal's first row")
    ap.add_argument("--total-channels", type=int, default=None)
    ap.add_argument("--channels", default=None, help="group channels, e.g. 40-47 (default: session yaml)")
    nio.add_clu_args(ap, method_default="stderiv_C5_D34", stage_default="", stage_dest="clu_stage", stage_alias=None)
    ap.add_argument("--spk-variant", default="standard", help="waveforms to split on (raw: the amplitude law holds)")
    ap.add_argument("--res-variant", default="", help="pin the .res variant ('' = the shared .res)")
    ap.add_argument("--units", default=None, help="comma-separated units to examine (overrides the amplitude pick)")
    for name, (dest, typ, fb, hlp) in _KNOBS.items():
        d = _knob_default(name, typ, fb, gcfg)
        ap.add_argument("--" + dest.replace("_", "-"), dest=dest, type=typ, default=d, help=f"{hlp} [{name}; default {d}]")
    ap.add_argument("--out-tag", default=None, help="output stage tag (default <clu-stage>_lowmix)")
    cfgmod.check_choice_defaults(ap)
    a = ap.parse_args(argv)
    if a.novel not in ("noise", "keep"):
        ap.error("--novel must be noise or keep")
    log = lambda s: print(s, file=sys.stderr, flush=True)  # noqa: E731

    cfg = sy.resolve_session_params(a.base, a.elec, channels=a.channels, ntotal=a.total_channels, verbose=False)
    base = cfg["base"]; NS, NC, SR = int(cfg["nsamp"]), int(cfg["nchan"]), float(cfg["sr"])
    chans = cfg["channels"]; ntot = cfg["ntotal"]
    res = (nio.read_res_at(base, a.elec, variant=a.res_variant) if a.res_variant else nio.read_res(base, a.elec))
    res = np.asarray(res, np.int64)
    _, clu = nio.resolve_clu(a, base, a.elec, n_spikes=res.size, stage_dest="clu_stage")
    clu = np.asarray(clu, np.int64)
    rs = nio.resolve(base, "spk", a.elec, a.spk_variant)
    if not rs.found:
        raise SystemExit(f"no .spk for '{a.spk_variant}' (looked for {rs.path})")
    if rs.variant != a.spk_variant:
        log(f"[lowmix] .spk '{a.spk_variant}' resolved to '{rs.variant}' ({os.path.basename(rs.path)})")
    spk = nio.open_spk_file(rs.path, NS, NC)
    if spk.shape[0] != res.size:
        raise SystemExit(f".spk.{a.spk_variant} has {spk.shape[0]} spikes, .res {res.size}")
    P = smoother(NS, a.deg)
    rng = np.random.default_rng(a.seed)
    sig = nio.open_signal(a.signal, ntot)
    rel = res - a.signal_start
    pre = int(cfg.get("peak") or NS // 2)
    noise = noise_snippets(lambda s, e: np.asarray(sig[s:e][:, chans], np.float32), sig.shape[0],
                           rel[(rel >= 0) & (rel < sig.shape[0])] - pre, NS, a.noise_n, rng)
    if len(noise) < 1000:
        raise SystemExit(f"only {len(noise)} spike-free noise snippets found in {a.signal}")
    noise_flat = prep(noise, P).reshape(len(noise), -1).astype(np.float64)
    Wh = whitener(noise_flat)
    log(f"[lowmix] noise: {len(noise)} spike-free snippets, whitened dims {Wh.shape[1]}")

    units, cnt = np.unique(clu[clu > max(RESERVE)], return_counts=True)
    tmpl, ptp = {}, {}
    for u, n in zip(units, cnt):
        if n < min(a.template_min_n, a.min_n):
            continue
        iu = np.flatnonzero(clu == u)
        if iu.size > 600:
            iu = np.sort(rng.choice(iu, 600, replace=False))
        t = np.median(prep(spk[iu], P), 0)
        ptp[int(u)] = float(np.ptp(t, 0).max())
        if n >= a.template_min_n:
            tmpl[int(u)] = t
    if a.units:
        pick = [int(x) for x in a.units.split(",") if x.strip()]
    else:
        elig = {u: p for u, p in ptp.items() if (clu == u).sum() >= a.min_n}
        thr = np.quantile(list(elig.values()), a.amp_quantile) if elig else -np.inf
        pick = sorted(u for u, p in elig.items() if p <= thr)
    log(f"[lowmix] examining {len(pick)} unit(s) (amp quantile {a.amp_quantile:g}, min-n {a.min_n}); "
        f"{len(tmpl)} match templates")

    new = clu.copy(); nxt = int(clu.max()) + 1; moves = []; rows = []
    for u in pick:
        iu = np.flatnonzero(clu == u)
        W = prep(spk[iu], P)
        s = split_unit(W, noise_flat, Wh, k_max=a.k_max, null_mult=a.null_mult, null_add=a.null_add, seed=a.seed)
        if not s["split"]:
            rows.append(dict(unit=u, group=0, n=iu.size, ptp=ptp.get(u, np.nan), gain=s["gain"],
                             gain_null=s["gain_null"], cos=np.nan, target=-1, shift=0, refrac="-", verdict="unsplit",
                             dest=u))
            continue
        groups = []
        for gi in range(s["k"]):
            m = s["labels"] == gi
            if m.sum() == 0:
                continue
            med = np.median(W[m], 0)
            c, v, sh = best_match(med, tmpl, u, a.max_shift)
            verdict = "allow" if a.refrac_ms <= 0 else "abstain"
            if a.refrac_ms > 0 and v is not None and m.sum() >= a.min_group:
                verdict = local_refractory(res[iu[m]], res[clu == v], SR, refrac_ms=a.refrac_ms,
                                           min_exp=a.refrac_min_exp)["verdict"]
            groups.append(dict(g=gi, idx=iu[m], n=int(m.sum()), ptp=float(np.ptp(med, 0).max()), cos=c,
                               target=v, shift=sh, refrac=verdict))
        small = [g for g in groups if g["n"] < a.min_group]
        big = decide([g for g in groups if g["n"] >= a.min_group], copy_cos=a.copy_cos, novel_cos=a.novel_cos,
                     copy_on_abstain=bool(a.copy_on_abstain))
        for g in small:
            g["verdict"] = "small"
        for g in big + small:
            if g["verdict"] in ("keep", "small"):
                dest = u
            elif g["verdict"] == "copy":
                dest = g["target"]
            elif g["verdict"] == "novel" and a.novel == "noise":
                dest = 1
            else:
                dest = nxt; nxt += 1
            g["dest"] = dest
            new[g["idx"]] = dest
            if g["verdict"] not in ("keep", "small"):
                moves.append((g["idx"], dest))
            rows.append(dict(unit=u, group=g["g"], n=g["n"], ptp=g["ptp"], gain=s["gain"], gain_null=s["gain_null"],
                             cos=g["cos"], target=g["target"] if g["target"] is not None else -1, shift=g["shift"],
                             refrac=g["refrac"], verdict=g["verdict"], dest=dest))
        log(f"[lowmix] unit {u}: {s['k']} groups (BIC gain {s['gain']:.3f} vs null {s['gain_null']:.3f}) -> "
            + ", ".join(f"{g['n']}:{g['verdict']}" + (f"->{g['dest']}" if g['verdict'] != 'keep' else '')
                        for g in big + small))
    tag = a.out_tag or (f"{a.clu_stage}_lowmix" if a.clu_stage else "lowmix")
    p = write_output(base, a.elec, a.clu_method, a.clu_stage, tag, clu, new, moves, log=log)
    tp = f"{base}.lmx.{a.elec}.{tag}"
    cols = ["unit", "group", "n", "ptp", "gain", "gain_null", "cos", "target", "shift", "refrac", "verdict", "dest"]
    with open(tp, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join((f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c])) for c in cols) + "\n")
    from collections import Counter
    vc = Counter(r["verdict"] for r in rows)
    nmov = {k: sum(r["n"] for r in rows if r["verdict"] == k) for k in ("copy", "new", "novel")}
    print(f"[lowmix] {len(pick)} unit(s): {vc.get('unsplit', 0)} unsplit; groups {dict(vc)}; spikes returned "
          f"{nmov['copy']}, new units {nmov['new']}, novel {nmov['novel']} "
          f"({'to noise' if a.novel == 'noise' else 'kept'}) | wrote {os.path.basename(p)} and {os.path.basename(tp)}")


if __name__ == "__main__":
    main()
