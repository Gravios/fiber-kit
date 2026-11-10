#!/usr/bin/env python3
"""fiber-manifold -- the OFF-SPIKE-CHANNEL FIELD MANIFOLD of a unit, fitted in sliding windows, written as
children (.clc/.clp) and as per-spike feature columns (.fet.manifold.<g>) Klusters can plot.

What it looks at.  After a unit's own template is subtracted, the channels where the unit has little
power (the "off-spike" channels) still carry structure: the fields of neighbouring cells that fired within
the waveform window (before, coincident with, or after the unit's spike), and the unit's own variable
far field.  The standard per-channel PCA features give those channels almost no budget and smear a
time-jittered event over every component, so the structure is invisible in the feature view although
it is plainly there in aggregate.  This stage models it explicitly, per spike:

    r(t, ch) = A * g((t - tau) / w) * exp(-|ch - p| / lam)        on the off-spike channels only

    g    the unit's own dominant-channel spike shape (unit norm); w in {0.8, 1, 1.3} widths
    tau  the field's lag relative to the unit's trough, +-FK_MANIFOLD_LAG_MS
    p    the field source's position along the group (channel-index units, continuous)
    lam  its decay length (channels);  A its amplitude (projection);  SNR the matched-filter score in
         noise-sd units, whitened with the group's noise covariance restricted to the off channels

The fit is the best (tau, p, lam, w) by matched filter (coarse grid, then a refinement around the best
cell), per spike.  A field is "present" when SNR >= FK_MANIFOLD_SNR (noise windows put through the
same filter bank give ~3% false positives at 4.0; the stage prints the noise rate it measured).

Drift.  Windows of FK_MANIFOLD_WINDOW_MIN minutes every FK_MANIFOLD_STEP_MIN minutes slide over the
unit; each window has its own template (the unit drifts in amplitude and position), and each spike takes
its fit from the window whose centre is nearest.  The noise covariance comes from the group's .nst (the
spike-free statistics ndm_extractspikes -N writes; in a Klusters sub-session it is linked in) or, with
--signal, from spike-free snippets of the .fil inside each window.

What it writes (stage tag <in>_manifold, or --out-tag):
  .clu          the input assignment, unchanged (this stage never moves a spike between units)
  .clc / .clp   children of every examined unit: per time bin x class, class = plain (no field) or
                field at (position bin, timing before|coincident|after the dead time); at most
                FK_MANIFOLD_MAX_CHILDREN per unit (time bins are coarsened until it fits).  Units not
                examined keep their atoms (or become one atom each when the input has no hierarchy).
  .fet.manifold.<g>   one row per spike of the group (zeros for spikes of units not examined), columns
                snr*100 | lag_us | position*100 | decay*100 | amplitude | window | class | isi_flag |
                amp_dom | time  -- a feature space Klusters lists under "Space" (its .spk is the
                standard one), so tau against p, or SNR against the dominant-channel amplitude, is one
                lasso away.  isi_flag: for pairs closer than FK_MANIFOLD_REFRAC_MS, 1 = the member
                farther from the window's median waveform, 2 = the nearer; 0 otherwise.
  .mft.<method>.<g>.<tag>.tsv   the children table (atom, unit, time bin, class, n) and, per unit and
                window, the template's dominant-channel trough and the field-bearing fraction.

CLI:  fiber-manifold <base> <elec> [--clu-method M --clu-stage S | --in-clu PATH] [--units 5,8]
                     [--signal BASE.fil] [--out-tag T] [--FK knobs as --window-min 20 ...]
"""
import argparse
import os
import sys

import numpy as np

try:
    from . import neuro_io as nio, session_yaml as sy, config as cfgmod, backend as bk
except ImportError:                                    # run as a script
    import neuro_io as nio
    import session_yaml as sy
    import config as cfgmod
    import backend as bk

RESERVE = (0, 1)
LAMS = (0.7, 1.2, 2.0, 3.5, 8.0)
WIDTHS = (0.8, 1.0, 1.3)
CLASS_NAMES = ("plain", "before", "coincident", "after")      # timing classes; position bins prefix them

_KNOBS = {
    "FK_MANIFOLD_WINDOW_MIN": ("window_min", float, 20.0, "sliding window length (minutes)"),
    "FK_MANIFOLD_STEP_MIN": ("step_min", float, 5.0, "window step (minutes); each spike takes the nearest window's fit"),
    "FK_MANIFOLD_MIN_N": ("min_n", int, 2000, "spikes a unit needs to be examined"),
    "FK_MANIFOLD_MIN_WINDOW_N": ("min_window_n", int, 300, "spikes a window needs for its own template; fewer: the neighbour's"),
    "FK_MANIFOLD_OFF_FRAC": ("off_frac", float, 0.4, "a channel is off-spike when its template trough is below this "
                                                     "fraction of the deepest channel's (used when --off-channels is unset)"),
    "FK_MANIFOLD_MIN_OFF": ("min_off", int, 3, "at least this many off-spike channels (the shallowest are added)"),
    "FK_MANIFOLD_SNR": ("snr", float, 4.0, "field present when the matched-filter SNR (noise-sd units) is at least this"),
    "FK_MANIFOLD_LAG_MS": ("lag_ms", float, 0.9, "largest |lag| searched (ms)"),
    "FK_MANIFOLD_DEAD_MS": ("dead_ms", float, 0.6, "|lag| below this is 'coincident' (the detector's dead time)"),
    "FK_MANIFOLD_POS_BINS": ("pos_bins", int, 3, "position bins over the off-channel span (classes = bins x 3 timings)"),
    "FK_MANIFOLD_MAX_CHILDREN": ("max_children", int, 200, "most children per unit (time bins coarsen to fit)"),
    "FK_MANIFOLD_TIME_BINS": ("time_bins", int, 12, "time bins the plain class is split into (0 = one plain child)"),
    "FK_MANIFOLD_REFRAC_MS": ("refrac_ms", float, 2.0, "ISI below this flags the pair's farther-from-median member (0 = off)"),
    "FK_MANIFOLD_NOISE_N": ("noise_n", int, 20000, "noise snippets for the SNR calibration (drawn from the .nst or the .fil)"),
    "FK_MANIFOLD_WRITE_FET": ("write_fet", int, 1, "write the .fet.manifold.<g> feature space (1) or not (0)"),
    "FK_MANIFOLD_SEED": ("seed", int, 0, "random seed (noise draws)"),
    "FK_MANIFOLD_JOBS": ("jobs", int, 0, "units examined in parallel (processes); 0 = one per CPU, 1 = serial"),
}


def _knob_default(name, typ, fallback, gcfg):
    return cfgmod.knob_value(name, typ, fallback, gcfg)


# ── model pieces ─────────────────────────────────────────────────────────────
def shifted(v, tau):
    """v (NS, ...) shifted by tau samples (zeros enter)."""
    out = np.zeros_like(v)
    ns = v.shape[0]
    if tau >= 0:
        out[tau:] = v[:ns - tau]
    else:
        out[:ns + tau] = v[-tau:]
    return out


def stretched(g, w):
    """g (NS,) resampled about its trough by factor w (w > 1: wider), unit norm."""
    ns = g.size
    t0 = int(np.argmin(g))
    x = np.arange(ns)
    out = np.interp((x - t0) / w + t0, x, g, left=0.0, right=0.0)
    n = np.linalg.norm(out)
    return out / n if n > 0 else out


def unit_shape(T, dom, half=15):
    """The unit's own temporal shape on its dominant channel: the template's trace, zero outside
    +-half samples of the trough, unit norm."""
    g = T[:, dom].astype(np.float64).copy()
    t0 = int(np.argmin(g))
    g[: max(0, t0 - half)] = 0.0
    g[t0 + half + 1:] = 0.0
    n = np.linalg.norm(g)
    return g / n if n > 0 else g


class FilterBank:
    """Whitened, unit-norm filters for every (tau, p, lam, w) of a grid, plus their raw norms (for the
    amplitude) and parameters."""

    def __init__(self, g, Wh, off_idx, lags, ps, lams, widths):
        self.par = []
        F = []
        self.nrm = []
        chs = np.asarray(off_idx, float)
        for w in widths:
            gw = stretched(g, w)
            for tau in lags:
                gt = shifted(gw, int(tau))
                for p in ps:
                    for lam in lams:
                        sp = np.exp(-np.abs(chs - p) / lam)
                        sp /= np.linalg.norm(sp)
                        f = (gt[:, None] * sp[None, :]).ravel() @ Wh
                        nn = np.linalg.norm(f)
                        if nn <= 0:
                            continue
                        self.par.append((tau, p, lam, w))
                        F.append(f / nn)
                        self.nrm.append(nn)
        self.par = np.array(self.par)
        self.F = np.stack(F, 1).astype(np.float32)
        self.nrm = np.array(self.nrm)

    def score(self, Rw):
        """Matched-filter scores: the one dense product of the stage.  On the GPU when fiber-kit's
        backend is on (FIBER_KIT_GPU=1 and CuPy + a device), else numpy (multi-threaded BLAS)."""
        xp = bk.xp()
        S = bk.asarray(Rw, np.float32) @ bk.asarray(self.F, np.float32)
        j = S.argmax(1)
        s = S[xp.arange(S.shape[0]), j]
        return bk.asnumpy(s), bk.asnumpy(j)


class Searcher:
    """Two-stage search: a coarse bank, then a refinement bank around each coarse winner (cached)."""

    def __init__(self, g, Wh, off_idx, lag_max, ps, sr):
        self.g, self.Wh, self.off_idx = g, Wh, off_idx
        self.coarse = FilterBank(g, Wh, off_idx, np.arange(-lag_max, lag_max + 1, 3), ps[::2], (0.7, 2.0, 5.0), (1.0,))
        self.pmin, self.pmax = float(ps.min()), float(ps.max())
        self.cache = {}

    def _refine(self, tau0, p0):
        key = (int(tau0), round(float(p0), 2))
        if key not in self.cache:
            ps = np.clip(np.array([p0 - 0.5, p0, p0 + 0.5]), self.pmin, self.pmax)
            self.cache[key] = FilterBank(self.g, self.Wh, self.off_idx, np.arange(tau0 - 2, tau0 + 3), ps, LAMS, WIDTHS)
        return self.cache[key]

    def search(self, Rw):
        s0, j0 = self.coarse.score(Rw)
        snr = np.zeros(len(Rw), np.float32)
        par = np.zeros((len(Rw), 4))
        amp = np.zeros(len(Rw), np.float32)
        for jj in np.unique(j0):
            m = np.flatnonzero(j0 == jj)
            tau0, p0, _, _ = self.coarse.par[jj]
            bank = self._refine(tau0, p0)
            s, j = bank.score(Rw[m])
            snr[m] = s
            par[m] = bank.par[j]
            amp[m] = s / bank.nrm[j]
        return snr, par, amp


def whitener_from_cov(C, floor=1e-4):
    ev, E = np.linalg.eigh(np.asarray(C, np.float64))
    ev = np.maximum(ev, ev.max() * floor)
    return (E / np.sqrt(ev)).astype(np.float32)


def sub_cov(cov, ns, nc, chans):
    """The (ns*nc, ns*nc) .nst covariance (element t*nc + c) restricted to channels `chans`."""
    idx = np.array([t * nc + c for t in range(ns) for c in chans])
    return cov[np.ix_(idx, idx)]


def noise_snippets_fil(sig, chans, res_rel, ns, n, rng, guard=None):
    """n spike-free windows (ns, len(chans)) from an open signal; starts away from every spike."""
    guard = ns if guard is None else int(guard)
    r = np.sort(np.asarray(res_rel, np.int64))
    out = []
    tries = 0
    while len(out) < n and tries < 50:
        t = rng.integers(guard, max(guard + 1, sig.shape[0] - ns - guard), size=4 * n)
        j = np.searchsorted(r, t)
        prev = np.where(j > 0, t - r[np.maximum(j - 1, 0)], np.iinfo(np.int64).max)
        nxt = np.where(j < r.size, r[np.minimum(j, r.size - 1)] - t, np.iinfo(np.int64).max)
        ok = (prev > guard) & (nxt > ns + guard)
        out.extend(t[ok][: n - len(out)].tolist())
        tries += 1
    if not out:
        return np.zeros((0, ns, len(chans)), np.float32)
    return np.stack([np.asarray(sig[s:s + ns][:, chans], np.float32) for s in out])


def fit_template(X, strong):
    """LS-scaled template of a window: scale on the strong channels, template = the per-sample MEDIAN of
    the 70% best fits (a median: the fields a minority of the spikes carry do not leak into the template,
    so a plain spike's residual stays plain)."""
    step = max(1, len(X) // 6000)                               # the median over <= ~6000 spikes is a median
    T = np.median(X[::step], 0)
    def scale(T):
        den = (T[:, strong] ** 2).sum()
        a = np.einsum("nsc,sc->n", X[:, :, strong], T[:, strong]) / den if den > 0 else np.ones(len(X))
        return np.clip(a, 0.5, 1.6)
    a = scale(T)
    R = X - a[:, None, None] * T
    nr = np.linalg.norm(R.reshape(len(X), -1), axis=1)
    good = nr <= np.percentile(nr, 70)
    gi = np.flatnonzero(good)[::step]
    T = np.median(X[gi] / a[gi, None, None], 0)
    a = scale(T)
    return T, a


# ── the stage ────────────────────────────────────────────────────────────────
def nearest_window(t, centres):
    """Index of the centre nearest to each t (centres sorted)."""
    j = np.clip(np.searchsorted(centres, t), 1, len(centres) - 1)
    left = j - 1
    return np.where(np.abs(t - centres[left]) <= np.abs(t - centres[j]), left, j) if len(centres) > 1 else np.zeros(len(t), int)


def examine_unit(iu, res, spk, sr, ns, nc, a, noise_cov_fn, log, rng, off_fixed=None):
    """Fit the field manifold of one unit in sliding windows.  Returns per-spike arrays (snr, par
    (tau, p, lam, w), amp, window, amp_dom) and a per-window table."""
    t = res[iu] / sr
    n = len(iu)
    # the unit's session-wide template decides the strong / off channels and the shape g
    sub = iu if n <= 20000 else np.sort(rng.choice(iu, 20000, replace=False))
    Xs = np.asarray(spk[sub]).astype(np.float32)
    Tall = Xs.mean(0)
    trough = -Tall.min(0)
    dom = int(trough.argmax())
    order = np.argsort(-trough)
    strong = order[: max(1, min(5, int((trough >= a.off_frac * trough.max()).sum())))]
    if off_fixed:
        off = sorted(int(c) for c in off_fixed if 0 <= int(c) < nc)
    else:
        off = [c for c in range(nc) if trough[c] < a.off_frac * trough.max()]
        k = 0
        while len(off) < a.min_off and k < nc:                  # the shallowest channels fill up
            c = int(order[nc - 1 - k])
            if c not in off:
                off.append(c)
            k += 1
        off = sorted(off)
    if not off:
        off = [int(order[-1])]
    g = unit_shape(Tall, dom)
    lag_max = int(round(a.lag_ms * 1e-3 * sr))
    ps = np.arange(min(off) - 0.5, max(off) + 0.51, 0.5)
    # windows
    t0, t1 = float(t.min()), float(t.max())
    W, S = a.window_min * 60.0, a.step_min * 60.0
    centres = np.arange(t0 + W / 2, max(t1 - W / 2, t0 + W / 2) + 1e-9, S) if t1 - t0 > W else np.array([(t0 + t1) / 2])
    win_of = nearest_window(t, centres)
    snr = np.zeros(n, np.float32)
    par = np.zeros((n, 4))
    amp = np.zeros(n, np.float32)
    amp_dom = np.zeros(n, np.float32)
    rows = []
    searcher = None
    noise_snr = None
    for wi, c in enumerate(centres):
        m = np.flatnonzero(win_of == wi)
        if m.size == 0:
            continue
        lo, hi = c - W / 2, c + W / 2
        sel = np.flatnonzero((t >= lo) & (t <= hi))             # the window's spikes (overlap with neighbours)
        if sel.size < a.min_window_n:                            # too few: widen to the nearest spikes
            sel = np.argsort(np.abs(t - c))[: a.min_window_n]
        X = np.asarray(spk[iu[sel]]).astype(np.float32)
        T, aa = fit_template(X, strong)
        Xm = np.asarray(spk[iu[m]]).astype(np.float32)
        den = (T[:, strong] ** 2).sum()
        am = np.clip(np.einsum("nsc,sc->n", Xm[:, :, strong], T[:, strong]) / den, 0.5, 1.6)
        R = (Xm - am[:, None, None] * T)[:, :, off].reshape(len(m), -1)
        amp_dom[m] = -Xm[:, :, dom].min(1)
        C, Nz = noise_cov_fn(lo, hi, off)
        Wh = whitener_from_cov(C)
        if searcher is None or noise_cov_fn.per_window:
            searcher = Searcher(g, Wh, off, lag_max, ps, sr)
            if noise_snr is None or noise_cov_fn.per_window:
                # the SNR the same bank gives pure noise: the .fil snippets, or draws from the covariance
                nz = Nz if Nz is not None else rng.multivariate_normal(
                    np.zeros(C.shape[0]), C, size=min(a.noise_n, 5000)).astype(np.float32)
                noise_snr, _, _ = searcher.search(nz.reshape(len(nz), -1) @ Wh)
        s, p_, ap = searcher.search(R @ Wh)
        snr[m], par[m], amp[m] = s, p_, ap
        rows.append(dict(window=wi, t_min=lo / 60, t_max=hi / 60, n=int(m.size), trough_dom=float(-T[:, dom].min()),
                         field_frac=float((s >= a.snr).mean())))
    nf = float((noise_snr >= a.snr).mean()) if noise_snr is not None else float("nan")
    log(f"[manifold] unit: {n} spikes, {len(centres)} windows of {a.window_min:g} min every {a.step_min:g}; off-spike "
        f"channels {off} (dominant {dom}); field SNR >= {a.snr:g}: {100 * float((snr >= a.snr).mean()):.1f}% "
        f"(noise: {100 * nf:.1f}%)")
    return dict(snr=snr, par=par, amp=amp, amp_dom=amp_dom, window=win_of, rows=rows, off=off, dom=dom,
                n_windows=len(centres), noise_frac=nf)


def isi_flags(iu, res, spk, sr, refrac_ms, rng):
    """1 = farther-from-median member of a pair closer than refrac_ms, 2 = the nearer, 0 otherwise.
    Distance = whitened by the unit's own spike covariance (diagonal), to the median of 3000 spikes."""
    n = len(iu)
    flags = np.zeros(n, np.int8)
    if refrac_ms <= 0 or n < 10:
        return flags
    t = res[iu] / sr
    d = np.diff(t) * 1e3
    v = np.flatnonzero(d < refrac_ms)
    if v.size == 0:
        return flags
    sub = iu if n <= 3000 else np.sort(rng.choice(iu, 3000, replace=False))
    Xs = np.asarray(spk[sub]).astype(np.float32).reshape(len(sub), -1)
    med = np.median(Xs, 0)
    sd = Xs.std(0) + 1e-6
    ix = np.unique(np.r_[v, v + 1])
    X = np.asarray(spk[iu[ix]]).astype(np.float32).reshape(len(ix), -1)
    dist = dict(zip(ix.tolist(), np.linalg.norm((X - med) / sd, axis=1).tolist()))
    for i in v:
        far, near = (i, i + 1) if dist[i] >= dist[i + 1] else (i + 1, i)
        flags[far] = 1
        if flags[near] == 0:
            flags[near] = 2
    return flags


def segment(fit, t_sec, a, sr):
    """Per-spike class id and time bin -> child index; at most max_children children.
    class 0 = plain; field classes = 1 + pos_bin * 3 + timing (timing: before, coincident, after)."""
    snr, par = fit["snr"], fit["par"]
    lag_ms = par[:, 0] / sr * 1e3
    off = fit["off"]
    p = par[:, 1]
    edges = np.linspace(min(off) - 0.5, max(off) + 0.5, a.pos_bins + 1)
    pos = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, a.pos_bins - 1)
    tim = np.where(lag_ms < -a.dead_ms, 0, np.where(lag_ms > a.dead_ms, 2, 1))
    cls = np.where(snr < a.snr, 0, 1 + pos * 3 + tim)
    n_cls = 1 + 3 * a.pos_bins
    nb = max(1, a.time_bins)
    t0, t1 = float(t_sec.min()), float(t_sec.max()) + 1e-6
    while True:
        tb = np.minimum(((t_sec - t0) / (t1 - t0) * nb).astype(int), nb - 1) if a.time_bins > 0 else np.zeros(len(t_sec), int)
        key = tb * n_cls + cls
        uniq = np.unique(key)
        if uniq.size <= a.max_children or nb == 1:
            break
        nb = max(1, nb // 2)
    child = np.searchsorted(uniq, key)
    names = []
    for k in uniq:
        tbk, ck = divmod(int(k), n_cls)
        if ck == 0:
            names.append((tbk, "plain"))
        else:
            pb, ti = divmod(ck - 1, 3)
            names.append((tbk, f"p{pb}/{CLASS_NAMES[1 + ti]}"))
    return child, names, cls, tb


def write_hierarchy(base, elec, method, in_stage, out_stage, clu, examined, log):
    """.clu (unchanged) + .clc/.clp: examined units' children replace their atoms; other units keep
    theirs (or become one atom each when the input has no hierarchy)."""
    try:
        from .fiber_refiberize import FiberHierarchy
    except ImportError:
        from fiber_refiberize import FiberHierarchy
    n = clu.size
    child = None
    parent = {}
    try:
        h = FiberHierarchy.load(base, elec, variant=method, tag=in_stage)
        if h.child.size == n:
            child = h.child.copy().astype(np.int64)
            parent = {int(k): int(v) for k, v in h.parent.items()}
    except (FileNotFoundError, OSError, ValueError):
        pass
    if child is None:
        child = clu.astype(np.int64).copy()                         # one atom per unit, same id
        parent = {int(u): int(u) for u in np.unique(clu) if u >= 1}
    nxt = int(max(child.max(), max(parent) if parent else 0)) + 1
    table = []
    for u, (iu, cidx, names) in examined.items():
        for ci, (tb, name) in enumerate(names):
            m = iu[cidx == ci]
            child[m] = nxt
            parent[nxt] = int(u)
            table.append((nxt, int(u), int(tb), name, int(m.size)))
            nxt += 1
    # atoms that lost every spike are dropped from the parent map by save(); the examined units' old atoms go
    live = set(np.unique(child).tolist())
    parent = {k: v for k, v in parent.items() if k in live}
    h2 = FiberHierarchy(child, parent)
    paths = h2.save(base, elec, variant=method, tag=out_stage, renumber=False, backup=False)
    nio.write_clu(base, elec, clu, variant=method, tag=out_stage)
    log(f"[manifold] wrote {os.path.basename(paths['clc'])} (+ .clp, .clu): {len(table)} children for {len(examined)} unit(s)")
    return table


_CTX = {}          # the per-run context the forked workers inherit (fork start method: no pickling)


def _run_unit(u):
    return _CTX["one_unit"](u)


def main(argv=None):
    gcfg = cfgmod.load_global_config()
    ap = argparse.ArgumentParser(description="Fit the off-spike-channel field manifold of each unit in sliding windows; "
                                             "write it as children and as a .fet.manifold feature space.")
    ap.add_argument("base"); ap.add_argument("elec", type=int)
    ap.add_argument("--signal", default=None, help="continuous filtered int16 file (BASE.fil) for per-window noise; "
                                                   "default: the group's .nst, else the unit's own baseline")
    ap.add_argument("--signal-start", type=int, default=0, help="absolute sample of the signal's first row")
    ap.add_argument("--total-channels", type=int, default=None)
    ap.add_argument("--channels", default=None, help="group channels, e.g. 40-47 (default: session yaml)")
    nio.add_clu_args(ap, method_default="stderiv_C5_D34", stage_default="", stage_dest="clu_stage", stage_alias=None)
    ap.add_argument("--spk-variant", default="standard", help="waveforms the fields are read from (raw)")
    ap.add_argument("--res-variant", default="", help="pin the .res variant ('' = the shared .res)")
    ap.add_argument("--units", default=None, help="comma-separated units to examine (default: every unit with >= --min-n)")
    ap.add_argument("--off-channels", default=None, help="off-spike channels as group indices, e.g. 3,4,5,6,7 (default: by --off-frac)")
    for name, (dest, typ, fb, hlp) in _KNOBS.items():
        d = _knob_default(name, typ, fb, gcfg)
        ap.add_argument("--" + dest.replace("_", "-"), dest=dest, type=typ, default=d, help=f"{hlp} [{name}; default {d}]")
    ap.add_argument("--out-tag", default=None, help="output stage tag (default <clu-stage>_manifold)")
    ap.add_argument("--fet-method", default="manifold", help="method token of the written feature space")
    cfgmod.check_choice_defaults(ap)
    a = ap.parse_args(argv)
    log = lambda s: print(s, file=sys.stderr, flush=True)  # noqa: E731
    rng = np.random.default_rng(a.seed)

    cfg = sy.resolve_session_params(a.base, a.elec, channels=a.channels, ntotal=a.total_channels, verbose=False,
                                    require=("channels",))
    base = cfg["base"]; NS, NC, SR = int(cfg["nsamp"]), int(cfg["nchan"]), float(cfg["sr"])
    chans = cfg["channels"]; ntot = cfg.get("ntotal")
    res = (nio.read_res_at(base, a.elec, variant=a.res_variant) if a.res_variant else nio.read_res(base, a.elec))
    res = np.asarray(res, np.int64)
    _, clu = nio.resolve_clu(a, base, a.elec, n_spikes=res.size, stage_dest="clu_stage")
    clu = np.asarray(clu, np.int64)
    rs = nio.resolve(base, "spk", a.elec, a.spk_variant)
    if not rs.found:
        raise SystemExit(f"no .spk for '{a.spk_variant}' (looked for {rs.path})")
    spk = nio.open_spk_file(rs.path, NS, NC)
    if spk.shape[0] != res.size:
        raise SystemExit(f".spk.{a.spk_variant} has {spk.shape[0]} spikes, .res {res.size}")
    if a.off_channels:
        a.off_fixed = [int(x) for x in a.off_channels.split(",")]
    else:
        a.off_fixed = None

    # ---- the noise covariance source
    sig = None
    nst = None
    if a.signal and os.path.isfile(a.signal) and ntot:
        sig = nio.open_signal(a.signal, ntot)
        log(f"[manifold] noise from spike-free snippets of {os.path.basename(a.signal)}, per window")
    else:
        rn = nio.resolve(base, "nst", a.elec, rs.variant)
        if rn.found:
            try:
                nst = nio.read_nst(rn.path)
                if nst["nsamp"] != NS or nst["nchan"] != NC:
                    log(f"[manifold] {os.path.basename(rn.path)} is {nst['nsamp']}x{nst['nchan']}, the .spk {NS}x{NC}: not used")
                    nst = None
                else:
                    log(f"[manifold] noise covariance from {os.path.basename(rn.path)}")
            except ValueError as e:
                log(f"[manifold] {e}: .nst not used")
                nst = None
    rel = res - a.signal_start
    pre = int(cfg.get("peak") or NS // 2)

    class NoiseCov:
        per_window = sig is not None
        def __call__(self, lo, hi, off):
            if sig is not None:
                lo_s, hi_s = int(lo * SR) - a.signal_start, int(hi * SR) - a.signal_start
                lo_s, hi_s = max(0, lo_s), min(sig.shape[0], hi_s)
                seg = sig[lo_s:hi_s]
                r = rel[(rel >= lo_s) & (rel < hi_s)] - lo_s - pre
                Nz = noise_snippets_fil(seg, [chans[c] for c in off], r, NS, min(a.noise_n, 6000), rng)
                if len(Nz) >= 500:
                    Nz = Nz - Nz.mean(0)
                    return np.cov(Nz.reshape(len(Nz), -1).T), Nz
            if nst is not None:
                return sub_cov(nst["cov"], NS, NC, off), None
            # fallback: the unit's own baseline (first 6 samples) as white noise per channel
            return np.eye(NS * len(off)) * float(self.sigma2), None
    noise_cov_fn = NoiseCov()

    units, cnt = np.unique(clu[clu > max(RESERVE)], return_counts=True)
    if a.units:
        pick = [int(x) for x in a.units.split(",") if x.strip()]
    else:
        pick = [int(u) for u, n in zip(units, cnt) if n >= a.min_n]
    log(f"[manifold] examining {len(pick)} unit(s): {pick[:12]}{'...' if len(pick) > 12 else ''}")

    n_fet = 10
    fet = np.zeros((res.size, n_fet), np.int64)
    fet[:, -1] = res
    examined = {}
    unit_rows = []

    def one_unit(u):
        iu = np.flatnonzero(clu == u)
        if iu.size < 10:
            return None
        iu = iu[np.argsort(res[iu])]
        if nst is None and sig is None:
            Xb = np.asarray(spk[iu[:2000]]).astype(np.float32)
            noise_cov_fn.sigma2 = float(Xb[:, :6, :].var())
        rng_u = np.random.default_rng(a.seed + int(u))
        fit = examine_unit(iu, res, spk, SR, NS, NC, a, noise_cov_fn, log, rng_u, off_fixed=a.off_fixed)
        flags = isi_flags(iu, res, spk, SR, a.refrac_ms, rng_u)
        cidx, names, cls, tb = segment(fit, res[iu] / SR, a, SR)
        return u, iu, fit, flags, cidx, names, cls, tb

    jobs = a.jobs if a.jobs > 0 else (os.cpu_count() or 1)
    import multiprocessing as mp
    if jobs > 1 and len(pick) > 1 and not bk.gpu_enabled() and "fork" in mp.get_all_start_methods():
        # units are independent: one process each (fork: the memmaps and arrays are inherited, not
        # copied; the BLAS threads of each worker are limited so the processes do not fight)
        _CTX["one_unit"] = one_unit
        os.environ.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 1) // min(jobs, len(pick)))))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", os.environ["OMP_NUM_THREADS"])
        with mp.get_context("fork").Pool(min(jobs, len(pick))) as pool:
            results = pool.map(_run_unit, pick, chunksize=1)
    else:
        results = [one_unit(u) for u in pick]
    for r in results:
        if r is None:
            continue
        u, iu, fit, flags, cidx, names, cls, tb = r
        examined[u] = (iu, cidx, names)
        fet[iu, 0] = np.round(fit["snr"] * 100)
        fet[iu, 1] = np.round(fit["par"][:, 0] / SR * 1e6)
        fet[iu, 2] = np.round(fit["par"][:, 1] * 100)
        fet[iu, 3] = np.round(fit["par"][:, 2] * 100)
        fet[iu, 4] = np.round(fit["amp"])
        fet[iu, 5] = fit["window"]
        fet[iu, 6] = cls
        fet[iu, 7] = flags
        fet[iu, 8] = np.round(fit["amp_dom"])
        for r in fit["rows"]:
            unit_rows.append(dict(unit=u, **r))
        log(f"[manifold] unit {u}: {len(names)} children ({np.unique(cls).size} classes x {np.unique(tb).size} time bins); "
            f"ISI<{a.refrac_ms:g} ms flagged {int((flags == 1).sum())}")

    tag = a.out_tag or (f"{a.clu_stage}_manifold" if a.clu_stage else "manifold")
    table = write_hierarchy(base, a.elec, a.clu_method, a.clu_stage, tag, clu, examined, log)
    if a.write_fet:
        fp = nio.write_fet(base, a.elec, fet, variant=a.fet_method)
        log(f"[manifold] wrote {os.path.basename(fp)}: columns snr*100 | lag_us | position*100 | decay*100 | amplitude | "
            f"window | class | isi_flag | amp_dom | time")
    tp = f"{base}.mft.{a.clu_method}.{a.elec}.{tag}.tsv"
    with open(tp, "w") as f:
        f.write("# children\natom\tunit\ttime_bin\tclass\tn\n")
        for row in table:
            f.write("\t".join(str(x) for x in row) + "\n")
        f.write("# windows\nunit\twindow\tt_min\tt_max\tn\ttrough_dom\tfield_frac\n")
        for r in unit_rows:
            f.write(f"{r['unit']}\t{r['window']}\t{r['t_min']:.1f}\t{r['t_max']:.1f}\t{r['n']}\t{r['trough_dom']:.0f}\t{r['field_frac']:.3f}\n")
    log(f"[manifold] wrote {os.path.basename(tp)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
