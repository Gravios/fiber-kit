#!/usr/bin/env python3
"""fiber-envelope: a collision-robust per-unit variance envelope + near-collision detection.

Why
---
The per-channel spread a curator expects of a single unit -- and that a merge must not exceed --
is estimated in fiber-kit as the plain per-sample std over the cluster's aligned spikes
(`fiber_geometry.dispersion_profile`).  That estimate is biased by NEAR COLLISIONS: a second
spike from another unit inside the 42-sample window but not coincident with the detected one.
On g6 ~3.5 % of spikes carry a second spike within +-0.65 ms that has no .res row (detector
dead time), and a few % more inside the window; at 5x noise amplitude a 5 % contamination
inflates the plain variance ~2x at the samples it lands on.  An inflated band widens every
band-overlap / envelope gate, so false merges pass and the contamination reads as "noise".

What
----
1. NOISE FLOOR from off-spike windows of the continuous trace (never from cluster members),
   so near collisions cannot bias it by construction.
2. Per-spike AMPLITUDE a and JITTER b by weighted least squares against the template and its
   time derivative:  x_i ~ a_i T + b_i T'  (first-order Taylor: b = -a*delta).
3. A time-LOCALISED BUMP score on the residual: the max over a short sliding window of the
   channel-summed, variance-normalised residual energy.  A near collision is a bump at some
   offset; a single unit's own variability is not.  The threshold is the empirical null of
   the same statistic on off-spike noise windows (or, without noise windows, a robust
   median + k*MAD outlier rule on the cluster's own scores).
4. ITERATE: template, a/b spreads and the residual variance are re-estimated on the spikes
   that pass, so the envelope converges without circular contamination.
5. The ENVELOPE  sigma^2(t,c) = var_noise(c) + var_a T^2 + var_b T'^2 + excess(t,c), where the
   excess is the robust (MAD) residual variance above the noise floor -- the unit's genuine
   shape variability.  `excess_ratio` per channel is the "variance beyond expected" a false
   merge produces.
6. NEAR-COLLISION PARTNERS: a flagged spike's residual over an EXTENDED window is matched
   against the other units' extended templates at every shift, NON-circularly (a shifted atom
   is zero outside its support -- np.roll would wrap the partner's tail back into the window).
   A partner is accepted when it explains the bump (post-subtraction score below threshold)
   at a plausible amplitude.  Partners can be written as .eap membership (Klusters' amber
   collision ring) through fiber_decollide.write_eap_membership.
7. PAIR SEPARATION for merge gating: d' between two units' templates in units of their
   envelopes, corrected for template-estimation noise, plus the worst single channel.

Space-agnostic: works on any linear transform of the continuous trace (fil, or the stderiv
spkD space via `transform="stderiv"`), as long as the noise floor is taken in the same space.
"""
import argparse
import sys
from dataclasses import dataclass, field

import numpy as np

try:
    from . import fiber_adapt as fad
    from . import fiber_stderiv as fst
    from . import neuro_io as nio
except ImportError:                       # script use
    import fiber_adapt as fad
    import fiber_stderiv as fst
    import neuro_io as nio

MAD_K = 1.4826


# ── snippet extraction ──────────────────────────────────────────────────────────

def space_transform(space, sdiff_pairs=None):
    """Waveform-space transform for a variant token, applied to raw (filtered) windows that carry
    ONE extra leading sample: callable (n, L+1, C) -> (n, L, C), or None for the untransformed space.

      standard | fil      -> None (the .fil window itself, = .spk.standard)
      stderiv             -> SDIFF_ALLPAIRS + temporal first difference
      stderiv_C<4|5>      -> the session's sdiffPairs pattern (partner map / reference sets)
    A trailing _D<lag><dims> (feature-space suffix) is ignored: the waveform is the same.  All stderiv
    orders go through fiber_stderiv.apply_stderiv_transform, the shared mirror of the C++ extractor,
    so the windows are the ones the .spk on disk was built from (before any fiber-realign shift)."""
    if space in (None, "", "standard", "fil"):
        return None
    spec = nio.parse_variant_token(space)
    if spec.family in ("standard", "fil"):
        return None
    if spec.family != "stderiv":
        raise ValueError(f"unsupported waveform space {space!r}")
    if spec.kind is None:
        return lambda r: fst.apply_stderiv_transform(r).astype(np.float32)
    if spec.kind != "C":
        raise ValueError(f"{space!r}: only the session's custom pattern (_C) is supported")
    if not sdiff_pairs:
        raise ValueError(f"{space!r} needs the session's sdiffPairs pattern")
    order = 5 if fst.sdiff_spec_uses_sets(sdiff_pairs) else 4
    if spec.order != order:
        raise ValueError(f"{space!r} says order {spec.order} but sdiffPairs parses as order {order}")
    if order == 5:
        sets = fst.parse_sdiff_sets(sdiff_pairs)
        return lambda r: fst.apply_stderiv_transform(r, sets=sets).astype(np.float32)
    partner, _ = fst.parse_sdiff_pairs(sdiff_pairs)
    return lambda r: fst.apply_stderiv_transform(r, partner=partner).astype(np.float32)


def extract(trace, times, lo, hi, transform=None, chunk=4096):
    """Snippets trace[t+lo : t+hi] for each t (relative to trace start) -> (n, hi-lo, C) float32.
    `trace` may be a memmap; only the needed rows are read.  transform: None (as is), a callable
    from space_transform (computed exactly with one extra leading sample), or a variant token."""
    times = np.asarray(times, np.int64)
    if isinstance(transform, str):
        transform = space_transform(transform)
    pad = 0 if transform is None else 1
    rel = np.arange(lo - pad, hi)
    out = np.empty((times.size, hi - lo, trace.shape[1]), np.float32)
    for s in range(0, times.size, chunk):
        idx = times[s:s + chunk, None] + rel[None, :]
        w = np.asarray(trace[idx.ravel()], np.float32).reshape(idx.shape + (trace.shape[1],))
        if transform is not None:
            w = transform(w)
        out[s:s + chunk] = w
    return out


# ── 1. noise floor ──────────────────────────────────────────────────────────────

@dataclass
class Noise:
    var: np.ndarray            # (C,) per-channel noise variance
    windows: np.ndarray        # (m, L, C) off-spike windows (for the null calibration)


def noise_floor(trace, spike_times, lo, hi, n=3000, guard=48, transform=None, seed=0):
    """Per-channel noise variance from off-spike windows [t+lo, t+hi): no detected spike within
    `guard` samples of the window.  Windows still contain sub-threshold spikes; that IS the
    background a unit's spikes ride on, so they are kept."""
    T = trace.shape[0]
    sp = np.sort(np.asarray(spike_times, np.int64))
    rng = np.random.default_rng(seed)
    pad = 0 if transform is None else 1
    cand = rng.integers(-lo + pad + 1, T - hi - 1, size=n * 8)
    i = np.searchsorted(sp, cand + lo - guard, "left")
    j = np.searchsorted(sp, cand + hi + guard, "right")
    ok = cand[j == i][:n]
    if ok.size < 50:
        raise RuntimeError(f"noise_floor: only {ok.size} off-spike windows found")
    W = extract(trace, ok, lo, hi, transform)
    W = W - W.mean(1, keepdims=True)
    return Noise(var=W.reshape(-1, W.shape[-1]).var(0).astype(np.float64), windows=W)


# ── 2-3. per-spike fit and bump score ───────────────────────────────────────────

def fit_amp_jitter(X, T, dT, w):
    """Weighted LS of every spike on [T, T'] -> a, b (n,), residual R (n, L, C).  w: (C,) = 1/var."""
    Tw, dTw = T * w, dT * w
    A11, A12, A22 = (T * Tw).sum(), (T * dTw).sum(), (dT * dTw).sum()
    y1 = np.einsum("nlc,lc->n", X, Tw)
    y2 = np.einsum("nlc,lc->n", X, dTw)
    det = A11 * A22 - A12 * A12
    if abs(det) < 1e-12 * max(A11 * A22, 1e-30):
        a = y1 / max(A11, 1e-30); b = np.zeros_like(a)
    else:
        a = (A22 * y1 - A12 * y2) / det
        b = (A11 * y2 - A12 * y1) / det
    R = X - a[:, None, None] * T[None] - b[:, None, None] * dT[None]
    return a, b, R


def bump_scores(R, v, width=8):
    """Max over a sliding window of the channel-summed normalised residual energy.
    R (n, L, C), v (L, C) or (C,) -> score (n,), centre sample of the max (n,)."""
    E = (R * R / v).sum(-1)                                    # (n, L)
    cs = np.concatenate([np.zeros((E.shape[0], 1)), np.cumsum(E, 1)], 1)
    S = cs[:, width:] - cs[:, :-width]                         # (n, L-width+1)
    k = S.argmax(1)
    return S[np.arange(S.shape[0]), k], k + width // 2


def _mad_var(x, axis=0):
    med = np.median(x, axis=axis, keepdims=True)
    return (MAD_K * np.median(np.abs(x - med), axis=axis)) ** 2


def align_to_template(X, T, w, core, maxlag):
    """Integer lag per spike in [-maxlag, maxlag] maximising the weighted cross-correlation of the
    spike's core window with T.  X is extended by >= maxlag on both sides of `core`."""
    lo, hi = core.start, core.stop
    best = np.full(X.shape[0], -np.inf); lag = np.zeros(X.shape[0], int)
    Tw = T * w
    for s in range(-maxlag, maxlag + 1):
        c = np.einsum("nlc,lc->n", X[:, lo + s:hi + s], Tw)
        m = c > best; best[m] = c[m]; lag[m] = s
    return lag


def take_lagged(X, lag, sl):
    """Per-spike X[i, sl.start+lag_i : sl.stop+lag_i]."""
    idx = np.arange(sl.start, sl.stop)[None, :] + lag[:, None]
    return np.take_along_axis(X, idx[:, :, None], axis=1)


# ── 4-5. the envelope ──────────────────────────────────────────────────────────

@dataclass
class Envelope:
    template: np.ndarray       # (L, C) median of the clean spikes (core window)
    dtemplate: np.ndarray      # (L, C) its time derivative
    var_noise: np.ndarray      # (C,)
    var_a: float               # robust variance of the amplitude scale
    var_b: float               # robust variance of the derivative (jitter) coefficient
    excess: np.ndarray         # (L, C) robust residual variance above the noise floor
    sigma: np.ndarray          # (L, C) robust envelope = sqrt(noise + a/b terms + excess)
    sigma_naive: np.ndarray    # (L, C) plain std over ALL aligned spikes (dispersion_profile-like)
    n: int                     # spikes in the cluster
    n_clean: int               # spikes that passed the bump test
    flagged: np.ndarray        # (n,) bool, near-collision candidates
    score: np.ndarray          # (n,) bump score (final iteration)
    bump_pos: np.ndarray       # (n,) sample of the bump within the core window
    a: np.ndarray              # (n,) amplitude scale
    b: np.ndarray              # (n,) derivative coefficient
    lag: np.ndarray            # (n,) integer alignment lag
    threshold: float
    template_ext: np.ndarray = None   # (Lx, C) extended-window template (for partner search)
    var_clean: np.ndarray = None      # (L, C) PLAIN variance of the aligned clean spikes
    var_all: np.ndarray = None        # (L, C) PLAIN variance of ALL aligned spikes
    var_kept: np.ndarray = None       # (L, C) ... of all spikes except EXPLAINED near collisions (run())
    # ── cell layer (fit_envelope(times=...)) ──
    shape_amp: np.ndarray = None      # (L, C) S: shape change per unit of amplitude scale (beyond scaling)
    drift: np.ndarray = None          # (L, C) D: template change per second within the window
    var_pred: np.ndarray = None       # (L, C) variance of the cell model's deterministic part (clean spikes)
    adapt: dict = None                # adaptation law a ~ b0 + b1*EWMA(tau): tau, b0, b1, r2, corr
    meta: dict = field(default_factory=dict)

    def amp_model(self):
        """Amplitude-scaled expected variance (L, C) of ONE source, with the same per-spike factors on
        every channel, so a closer unit (larger T) gets a proportionally wider band and a distant one
        stays noise-dominated.  Without the cell layer: noise_c + var_a T^2 + var_b T'^2.  With it
        (fit_envelope(times=...)): noise_c + the spike-to-spike variance of
        a_i T + b_i T' + (a_i - <a>) S + (t_i - <t>) D -- amplitude-dependent shape (a burst's later,
        smaller spikes are also differently SHAPED) and within-window drift are then part of what one
        cell is allowed."""
        if self.var_pred is not None:
            return self.var_noise[None, :] + self.var_pred
        return self.var_noise[None, :] + self.var_a * self.template ** 2 + self.var_b * self.dtemplate ** 2

    def shape_gain(self):
        """|S| / |T|: how strongly the shape (not just the size) changes with amplitude.  None without
        the cell layer."""
        if self.shape_amp is None:
            return None
        return float(np.linalg.norm(self.shape_amp) / max(np.linalg.norm(self.template), 1e-12))

    def channel_ratio(self, which="kept"):
        """Per-channel observed / amplitude-predicted variance, energy-weighted over the window:
        sum_t var / sum_t amp_model.  ~1 on every channel for one unit at ANY distance; a second
        source shows up as > 1 on the channels where it differs from the first.

        which: 'kept'  -- all spikes except the near collisions a partner EXPLAINED (default; falls
                          back to 'all' before run() has searched partners).  A merged minority stays
                          in: a foreign spike very unlike the template IS flagged by the bump test, but
                          no shifted partner explains it, so it is counted here.
               'clean' -- only spikes that passed the bump test (hides a strongly different minority)
               'all'   -- every spike (near collisions inflate it)"""
        v = {"kept": self.var_kept if self.var_kept is not None else self.var_all,
             "clean": self.var_clean, "all": self.var_all}[which]
        return v.sum(0) / self.amp_model().sum(0)

    def excess_ratio(self):
        """Per-channel mean (over the core window) of residual variance / noise variance on the
        clean spikes -- ~1 for a well-modelled single unit; >1 where the cluster spreads beyond what
        noise + amplitude + jitter explain (a merged second unit, or shape adaptation)."""
        return ((self.excess + self.var_noise[None]) / self.var_noise[None]).mean(0)


def null_threshold(noise_windows, T, dT, var_noise, width=8, q=0.99):
    """Empirical (1-q) tail of the bump score on pure-noise windows fitted to the template.
    Windows longer than the template are cut to their CENTRED core-length part."""
    w = 1.0 / var_noise
    Lc = T.shape[0]
    s0 = (noise_windows.shape[1] - Lc) // 2
    Wn = noise_windows[:, s0:s0 + Lc]
    _, _, Rn = fit_amp_jitter(Wn, T, dT, w)
    s, _ = bump_scores(Rn, var_noise[None, :], width)
    return float(np.quantile(s, q)), s


ADAPT_TAUS = np.logspace(np.log10(0.02), np.log10(2.0), 16)


def adaptation_law(times, amp, keep=None, taus=ADAPT_TAUS):
    """Fit the spike-frequency-adaptation law of fiber_adapt to a unit's own train: amplitude
    a_i ~ b0 + b1 * EWMA_tau(prior spikes), best tau by R^2 over `taus`.  times in seconds.
    Returns dict(tau, b0, b1, r2, corr) (r2 = 0 when no tau helps)."""
    t = np.asarray(times, float); y = np.asarray(amp, float)
    keep = np.ones(t.size, bool) if keep is None else np.asarray(keep, bool)
    o = np.argsort(t, kind="stable")
    A = fad.ewma_multi(t[o], taus)
    best = dict(tau=float("nan"), b0=float(y[keep].mean()) if keep.any() else 0.0, b1=0.0, r2=0.0, corr=0.0)
    yk = y[o][keep[o]]; tot = ((yk - yk.mean()) ** 2).sum()
    if yk.size < 10 or tot <= 0:
        return best
    for i, tau in enumerate(taus):
        s = A[i][keep[o]]
        if s.std() < 1e-9:
            continue
        M = np.c_[np.ones(s.size), s]; bb = np.linalg.lstsq(M, yk, rcond=None)[0]
        r2 = 1 - ((yk - M @ bb) ** 2).sum() / tot
        if r2 > best["r2"]:
            best = dict(tau=float(tau), b0=float(bb[0]), b1=float(bb[1]), r2=float(r2),
                        corr=float(np.corrcoef(s, yk)[0, 1]))
    return best


def fit_envelope(X, core, var_noise=None, noise_windows=None, *, maxlag=4, width=8, q=0.99,
                 iters=3, k_mad=4.0, ext=None, times=None):
    """Robust envelope of ONE unit.

    X           (n, Lx, C) snippets extracted with margin around the core window.
    core        slice of the core window inside X (must leave >= maxlag on both sides).
    var_noise   (C,) noise floor; None -> self-calibrated (lowest-decile robust variance per channel)
    noise_windows  (m, >=L, C) off-spike windows for the empirical null (centred core part used);
                None -> median+k*MAD rule.
    ext         slice of the extended window (relative to X) for the extended template (partner
                search); must leave >= maxlag on both sides.  None -> no extended template.
    times       (n,) spike times in seconds -> also fit the CELL layer on the clean spikes:
                amplitude-dependent shape S, within-window drift D (both unit-level (L, C)
                directions regressed from the residual), and the adaptation law of the unit's train.
                amp_model() / channel_ratio() then measure spread beyond what one cell does.
    """
    X = np.asarray(X, np.float64)
    n, Lx, C = X.shape
    L = core.stop - core.start
    if var_noise is None:
        v0 = _mad_var(X[:, core].reshape(-1, L, C), axis=0)
        var_noise = np.quantile(v0, 0.1, axis=0)
    var_noise = np.maximum(np.asarray(var_noise, np.float64), 1e-12)
    w = 1.0 / var_noise
    keep = np.ones(n, bool)
    T = np.median(X[:, core], 0)
    lag = np.zeros(n, int)
    thr = np.inf; score = np.zeros(n); pos = np.zeros(n, int)
    for _ in range(max(1, iters)):
        lag = align_to_template(X, T, w, core, maxlag)
        Xc = take_lagged(X, lag, core)
        T = np.median(Xc[keep], 0)
        dT = np.gradient(T, axis=0)
        a, b, R = fit_amp_jitter(Xc, T, dT, w)
        vres = np.maximum(_mad_var(R[keep], axis=0), var_noise[None, :])
        score, pos = bump_scores(R, vres, width)
        if noise_windows is not None:
            # spikes are scored against vres (noise + the unit's own robust excess), noise windows
            # against var_noise: both are then unit-variance residuals, so one null serves both.
            thr, _ = null_threshold(noise_windows, T, dT, var_noise, width, q)
        else:
            med = np.median(score[keep]); mad = MAD_K * np.median(np.abs(score[keep] - med))
            thr = med + k_mad * mad
        new = score <= thr
        if new.sum() < max(10, 0.2 * n):     # never let the clean set collapse
            break
        if np.array_equal(new, keep):
            keep = new
            break
        keep = new
    Xc = take_lagged(X, lag, core)
    T = np.median(Xc[keep], 0); dT = np.gradient(T, axis=0)
    a, b, R = fit_amp_jitter(Xc, T, dT, w)
    var_a = float(_mad_var(a[keep])); var_b = float(_mad_var(b[keep]))
    excess = np.maximum(_mad_var(R[keep], axis=0) - var_noise[None, :], 0.0)
    sigma = np.sqrt(var_noise[None, :] + var_a * T ** 2 + var_b * dT ** 2 + excess)
    naive = Xc.std(0)
    var_clean = Xc[keep].var(0)
    var_all = Xc.var(0)
    Text = None
    if ext is not None:
        Text = np.median(take_lagged(X, lag, ext)[keep], 0)
    Sdir = Ddir = var_pred = adapt = None
    if times is not None:
        t = np.asarray(times, float)
        P = a[:, None, None] * T[None] + b[:, None, None] * dT[None]
        da = a - a[keep].mean()
        Sdir = (da[keep, None, None] * (Xc - P)[keep]).sum(0) / max(float((da[keep] ** 2).sum()), 1e-12)
        P = P + da[:, None, None] * Sdir[None]
        dt = t - t[keep].mean()
        Ddir = (dt[keep, None, None] * (Xc - P)[keep]).sum(0) / max(float((dt[keep] ** 2).sum()), 1e-12)
        P = P + dt[:, None, None] * Ddir[None]
        var_pred = P[keep].var(0)
        adapt = adaptation_law(t, a, keep)
    return Envelope(template=T, dtemplate=dT, var_noise=var_noise, var_a=var_a, var_b=var_b,
                    excess=excess, sigma=sigma, sigma_naive=naive, n=n, n_clean=int(keep.sum()),
                    flagged=~keep, score=score, bump_pos=pos, a=a, b=b, lag=lag, threshold=thr,
                    template_ext=Text, var_clean=var_clean, var_all=var_all, shape_amp=Sdir,
                    drift=Ddir, var_pred=var_pred, adapt=adapt)


def robust_dispersion_profile(waves, var_noise=None, noise_windows=None, maxlag=3, **kw):
    """Robust counterpart of fiber_geometry.dispersion_profile for spk-only stacks (no continuous
    context): returns the Envelope (None for < 10 spikes); `.sigma` is the band and `.template` the
    matching median, both (L-2*maxlag, C) -- the window shrinks by the alignment margin.  Feed the
    pair to band_overlap in place of (median, dispersion_profile)."""
    w = np.asarray(waves, float)
    if len(w) < 10:
        return None
    return fit_envelope(w, slice(maxlag, w.shape[1] - maxlag), var_noise, noise_windows,
                        maxlag=maxlag, **kw)


# ── 6. near-collision partners ─────────────────────────────────────────────────

def _shift_atom(Text, s, Lx):
    """Template Text (Lx, C) delayed by s samples, zero outside its support (non-circular)."""
    out = np.zeros_like(Text)
    if s >= 0:
        out[s:] = Text[:Lx - s]
    else:
        out[:Lx + s] = Text[-s:]
    return out


def partner_dictionary(others, Lx, *, smax=None, min_shift=2, exclude=None):
    """Shifted, zero-padded atoms of every unit's extended template -> (atoms (K*S, Lx, C), keys, shifts)."""
    smax = (Lx // 2 - 4) if smax is None else int(smax)
    keys = [k for k in sorted(others) if k != exclude and others[k].template_ext is not None]
    shifts = [s for s in range(-smax, smax + 1) if abs(s) >= min_shift]
    atoms = np.stack([_shift_atom(others[k].template_ext, s, Lx) for k in keys for s in shifts])
    return atoms, keys, shifts


def partner_gain_null(noise_windows, atoms, var_noise, amp=(0.4, 2.5), q=0.99, chunk=1024):
    """(q) quantile of the BEST single-atom gain on pure-noise extended windows: an explanation must
    beat what the dictionary finds in noise alone, or a low-amplitude template 'explains' any bump."""
    w = 1.0 / var_noise
    A = (atoms * w[None, None, :]).reshape(len(atoms), -1)
    norms = (atoms * atoms * w[None, None, :]).sum((1, 2)) + 1e-12
    best = []
    for s0 in range(0, len(noise_windows), chunk):
        G = np.asarray(noise_windows[s0:s0 + chunk], np.float64).reshape(-1, A.shape[1]) @ A.T
        c = G / norms[None]
        best.append(np.where((c >= amp[0]) & (c <= amp[1]), G * c, 0.0).max(1))
    return float(np.quantile(np.concatenate(best), q))


def find_partners(Xext, own, env, others, var_noise, *, core_in_ext, width=8, smax=None,
                  min_shift=2, amp=(0.4, 2.5), gain_thr=0.0, chunk=2048):
    """Explain flagged spikes' bumps by another unit's template at some shift.

    Xext         (m, Lx, C) flagged spikes' extended windows, ALREADY aligned (own lag applied).
    own          unit id (excluded from the dictionary).
    env          the unit's Envelope (template_ext, a, b used through `ab`).
    others       {unit: Envelope} with template_ext set (the dictionary).
    core_in_ext  slice of the core window inside the extended window.
    gain_thr     minimum noise-normalised energy the partner atom must explain (partner_gain_null).
    Returns dict of arrays (m,): partner (-1 = none), shift (samples, partner trough relative to the
    detected spike), coef, gain, score_after, explained (bool).
    """
    Lx, C = env.template_ext.shape
    w = 1.0 / var_noise
    atoms, keys, shifts = partner_dictionary(others, Lx, smax=smax, min_shift=min_shift, exclude=own)
    atoms_w = atoms * w[None, None, :]
    norms = (atoms * atoms_w).sum((1, 2)) + 1e-12
    A = atoms_w.reshape(len(atoms), -1)
    m = Xext.shape[0]
    out = dict(partner=np.full(m, -1), shift=np.zeros(m, int), coef=np.zeros(m), gain=np.zeros(m),
               score_after=np.full(m, np.inf), explained=np.zeros(m, bool))
    if m == 0 or not keys:
        return out
    Tx = env.template_ext; dTx = np.gradient(Tx, axis=0)
    vres = np.maximum(env.excess + var_noise[None, :], var_noise[None, :])
    cs = core_in_ext
    for s0 in range(0, m, chunk):
        Xe = np.asarray(Xext[s0:s0 + chunk], np.float64)
        # own amplitude/jitter re-fit on the core (as in fit_envelope), residual over the extended window
        a, b, _ = fit_amp_jitter(Xe[:, cs], env.template, env.dtemplate, w)
        Re = Xe - a[:, None, None] * Tx[None] - b[:, None, None] * dTx[None]
        G = Re.reshape(len(Re), -1) @ A.T                       # (mm, K*S)
        coef = G / norms[None]
        gain = np.where((coef >= amp[0]) & (coef <= amp[1]), G * coef, -np.inf)
        j = gain.argmax(1)
        ok = np.isfinite(gain[np.arange(len(j)), j])
        c = coef[np.arange(len(j)), j]
        g = np.where(ok, gain[np.arange(len(j)), j], 0.0)
        Rp = Re - c[:, None, None] * atoms[j]
        sc, _ = bump_scores(Rp[:, cs], vres, width)
        sl = slice(s0, s0 + len(Re))
        out["partner"][sl] = np.where(ok, np.asarray(keys)[j // len(shifts)], -1)
        out["shift"][sl] = np.asarray(shifts)[j % len(shifts)]
        out["coef"][sl] = c
        out["gain"][sl] = g
        out["score_after"][sl] = sc
        out["explained"][sl] = ok & (sc <= env.threshold) & (g > gain_thr)
    return out


# ── 7. pair separation (merge gate) ───────────────────────────────────────────

def separation(ea, eb, maxlag=3, sigma="robust"):
    """d' between two units: sqrt( sum_{t,c} D^2 / s^2  -  L*C*(1/na + 1/nb) ), D = Ta - Tb at the best
    integer lag (|lag| <= maxlag, overlap region only), s^2 = mean of the two envelopes.  The
    subtracted term is the expected value for two estimates of the SAME template (estimation
    noise), so a same-cell pair sits near 0.  Also returns the worst single channel's d'.
    sigma: 'robust' (Envelope.sigma) or 'naive' (Envelope.sigma_naive)."""
    Ta, Tb = ea.template, eb.template
    Sa = ea.sigma if sigma == "robust" else ea.sigma_naive
    Sb = eb.sigma if sigma == "robust" else eb.sigma_naive
    L, C = Ta.shape
    best = None
    for s in range(-maxlag, maxlag + 1):
        a0, b0 = max(0, s), max(0, -s)
        l = L - abs(s)
        D = Ta[a0:a0 + l] - Tb[b0:b0 + l]
        v = 0.5 * (Sa[a0:a0 + l] ** 2 + Sb[b0:b0 + l] ** 2)
        q = D * D / v
        d2 = q.sum() - l * C * (1.0 / ea.n_clean + 1.0 / eb.n_clean)
        if best is None or d2 < best[0]:
            best = (d2, q, l)
    d2, q, l = best
    dc = q.sum(0) - l * (1.0 / ea.n_clean + 1.0 / eb.n_clean)
    return float(np.sqrt(max(d2, 0.0))), float(np.sqrt(max(dc.max(), 0.0)))


# ── CLI ───────────────────────────────────────────────────────────────────────

def run(trace, res, clu, *, pre, post, margin=32, maxlag=4, transform=None, min_n=50,
        max_per=3000, width=8, q=0.99, seed=0, partners=True, sr=None, log=print):
    """Envelopes for every unit with >= min_n spikes + partner search for its flagged spikes.
    res: spike times relative to `trace` start; clu: labels (<= 1 = noise, skipped).
    sr: sampling rate -> the cell layer (amplitude-dependent shape, drift, adaptation law) is fit too.
    Returns ({unit: Envelope}, per-spike table dict, noise)."""
    lo, hi = -pre - margin - maxlag, post + margin + maxlag
    core = slice(margin + maxlag, margin + maxlag + pre + post)
    ext = slice(maxlag, hi - lo - maxlag)
    core_in_ext = slice(margin, margin + pre + post)
    ok = (res + lo - 1 >= 0) & (res + hi < trace.shape[0])
    # off-spike windows over the EXTENDED span: core part for the floor + bump null, all of it for the
    # partner-gain null
    noise = noise_floor(trace, res, -pre - margin, post + margin, guard=pre + post,
                        transform=transform, seed=seed)
    noise.var = noise.windows[:, margin:margin + pre + post].reshape(-1, noise.windows.shape[-1]).var(0)
    rng = np.random.default_rng(seed)
    envs, rows = {}, {}
    for u in np.unique(clu[ok & (clu > 1)]):
        idx = np.flatnonzero(ok & (clu == u))
        if idx.size < min_n:
            continue
        fit_idx = np.sort(rng.choice(idx, max_per, replace=False)) if idx.size > max_per else idx
        X = extract(trace, res[fit_idx], lo, hi, transform)
        env = fit_envelope(X, core, noise.var, noise.windows, maxlag=maxlag, width=width, q=q, ext=ext,
                           times=None if sr is None else res[fit_idx] / float(sr))
        if fit_idx.size != idx.size:            # score every spike against the converged envelope
            X = extract(trace, res[idx], lo, hi, transform)
            w = 1.0 / env.var_noise
            lag = align_to_template(X, env.template, w, core, maxlag)
            a, b, R = fit_amp_jitter(take_lagged(X, lag, core), env.template, env.dtemplate, w)
            sc, pos = bump_scores(R, np.maximum(env.excess + env.var_noise[None], env.var_noise[None]), width)
            env.lag, env.a, env.b, env.score, env.bump_pos = lag, a, b, sc, pos
            env.flagged = sc > env.threshold
        env.meta["idx"] = idx
        if sr is not None:
            raw = extract(trace, res[idx[:: max(1, idx.size // 500)]], -pre, post, None).mean(0)
            env.meta["type"] = type_features(raw, res[idx] / float(sr), sr)
        envs[int(u)] = env
        rows[int(u)] = X if partners else None
        log(f"unit {int(u):4d}: n={idx.size:6d} flagged={env.flagged.mean():.3f} "
            f"chan_ratio_clean_max={env.channel_ratio('clean').max():.2f} naive/robust={np.median(env.sigma_naive / env.sigma):.2f}")
    table = dict(idx=[], unit=[], score=[], flagged=[], bump_pos=[], partner=[], shift=[], coef=[],
                 explained=[])
    gain_thr = 0.0
    if partners and envs:
        Lx = pre + post + 2 * margin
        atoms, _, _ = partner_dictionary(envs, Lx)
        gain_thr = partner_gain_null(noise.windows, atoms, noise.var, q=q)
        log(f"partner gain threshold (noise q={q}): {gain_thr:.1f}")
    for u, env in envs.items():
        idx = env.meta["idx"]; f = env.flagged
        part = dict(partner=np.full(idx.size, -1), shift=np.zeros(idx.size, int),
                    coef=np.zeros(idx.size), explained=np.zeros(idx.size, bool))
        if partners and f.any():
            X = rows[u]
            Xext = take_lagged(X[f], env.lag[f], ext)
            p = find_partners(Xext, u, env, envs, env.var_noise, core_in_ext=core_in_ext, width=width,
                              gain_thr=gain_thr)
            for k in part:
                part[k][f] = p[k]
            keep_k = ~part["explained"]
            Xc = take_lagged(X[keep_k], env.lag[keep_k], core)
            env.var_kept = Xc.var(0)
        table["idx"].append(idx); table["unit"].append(np.full(idx.size, u))
        table["score"].append(env.score); table["flagged"].append(f)
        table["bump_pos"].append(env.bump_pos - pre); table["partner"].append(part["partner"])
        table["shift"].append(part["shift"] + 0); table["coef"].append(part["coef"])
        table["explained"].append(part["explained"])
        rows[u] = None
    table = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in table.items()}
    o = np.argsort(table["idx"], kind="stable")
    return envs, {k: v[o] for k, v in table.items()}, noise


# ── 8. cell type ───────────────────────────────────────────────────────────────

def type_features(raw_template, times, sr=32552.0, duration=None):
    """Cell-type evidence for one unit: trough-to-peak width (ms) of the UNTRANSFORMED (filtered) mean
    waveform on its largest channel, burst index (fraction of ISIs < 10 ms) and rate (Hz).  On g6 the
    narrow units (width < 0.35 ms) are the high-rate, non-adapting interneurons; the broad, bursting,
    adapting units are pyramidal cells.  Width needs the raw window -- the stderiv transform's temporal
    difference shifts and narrows the extremum (docs/morphology.md, "Peak indices")."""
    W = np.asarray(raw_template, float)
    pc = int(np.argmax(np.ptp(W, 0))); x = W[:, pc]
    tr = int(np.argmin(x)); width = int(np.argmax(x[tr:])) / float(sr) * 1e3
    t = np.sort(np.asarray(times, float))
    isi = np.diff(t)
    dur = float(duration) if duration else max(float(t[-1] - t[0]), 1e-9) if t.size > 1 else 1.0
    return dict(width_ms=float(width), burst=float((isi < 0.010).mean()) if isi.size else 0.0,
                rate_hz=float(t.size / dur), narrow=bool(width < 0.35))


def cell_type_conflict(types, narrow_ms=0.35):
    """cannot_link for merge_units: a narrow-spiking unit never merges with a broad-spiking one."""
    def f(ua, ub):
        a, b = types.get(ua), types.get(ub)
        if not a or not b:
            return None
        if (a["width_ms"] < narrow_ms) != (b["width_ms"] < narrow_ms):
            return f"width {a['width_ms']:.2f}/{b['width_ms']:.2f} ms"
        return None
    return f


# ── 9. cell-aware merging ──────────────────────────────────────────────────────

def adaptation_coupling(tA, aA, tB, aB, taus=ADAPT_TAUS):
    """Does ONE adaptation law over the MERGED train explain both fragments' amplitudes as well as each
    fragment's own law (with its own mean) does?  Amplitudes must be on a common scale (e.g. from a
    fit of the union).  Fragments of one adapting cell split by energy band keep r2_joint close to
    r2_sep -- the joint state predicts which spikes are the small, late-in-burst ones; two cells do
    not.  Returns (r2_joint, r2_sep)."""
    t = np.r_[np.asarray(tA, float), np.asarray(tB, float)]
    y = np.r_[np.asarray(aA, float), np.asarray(aB, float)]
    src = np.r_[np.zeros(len(tA)), np.ones(len(tB))]
    o = np.argsort(t, kind="stable"); t, y, src = t[o], y[o], src[o]
    tot = float(((y - y.mean()) ** 2).sum())
    if tot <= 0 or y.size < 10:
        return 0.0, 0.0
    best = 0.0
    for s_ in fad.ewma_multi(t, taus):
        M = np.c_[np.ones(s_.size), s_]; bb = np.linalg.lstsq(M, y, rcond=None)[0]
        best = max(best, 1 - float(((y - M @ bb) ** 2).sum()) / tot)
    rs = 0.0
    for g in (0, 1):
        m = src == g; tt, yy = t[m], y[m]
        r_ = float(((yy - yy.mean()) ** 2).sum())
        for s_ in fad.ewma_multi(tt, taus):
            M = np.c_[np.ones(s_.size), s_]; bb = np.linalg.lstsq(M, yy, rcond=None)[0]
            r_ = min(r_, float(((yy - M @ bb) ** 2).sum()))
        rs += r_
    return float(best), float(1 - rs / tot)


def merge_evidence(XA, tA, XB, tB, core, noise, envA, envB, maxlag=4):
    """Evidence that two units are one cell, beyond template separation:
      inflation   amplitude-scaled spread of the UNION (cell model, clean spikes) minus the larger of
                  the two parts' -- ~<= 0 for fragments of one cell (a fragment is a compact piece of
                  the cell's spread), > 0 when a second source is added.  On g6 fiber_session pairs, a
                  0.1 cut vetoed 32 pairs of which 31 were different GT cells.
      adapt_joint / adapt_sep   adaptation_coupling on the union's common amplitude scale."""
    U = fit_envelope(np.concatenate([XA, XB]), core, noise.var, noise.windows, maxlag=maxlag,
                     times=np.r_[tA, tB])
    infl = float(U.channel_ratio("clean").max()
                 - max(envA.channel_ratio("clean").max(), envB.channel_ratio("clean").max()))
    na = len(XA)
    aj, asep = adaptation_coupling(tA, U.a[:na], tB, U.a[na:])
    return dict(inflation=infl, adapt_joint=aj, adapt_sep=asep, shape_gain=U.shape_gain())


def merge_units(trace, res, clu, envs, noise, *, theta=7.0, pre=21, post=21, margin=32, maxlag=4,
                transform=None, sr=32552.0, knn=8, dmax=15.0, infl_thr=0.1, cannot_link=None,
                log=print):
    """Average-linkage merge of units on envelope d' (`separation`), with merge inflation as a
    cannot-link: pairs among each unit's `knn` nearest (d' < dmax) get merge_evidence, and a pair whose
    union inflates the amplitude-scaled spread by more than `infl_thr` cannot end up in one cluster.
    Every unit needs the cell layer (run(..., sr=...)).  cannot_link: optional callable (ua, ub) ->
    reason string or None, applied to every pair (e.g. cell_type_conflict) -- a further cannot-link.
    Returns (new labels, pair rows); a merged group takes the smallest member id, units not in `envs`
    keep theirs."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    units = sorted(envs); n = len(units)
    lo, hi = -pre - margin - maxlag, post + margin + maxlag
    core = slice(margin + maxlag, margin + maxlag + pre + post)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = separation(envs[units[i]], envs[units[j]])[0]
    cand = sorted({tuple(sorted((i, int(j)))) for i in range(n) for j in np.argsort(D[i])[1:knn + 1]
                   if D[i, j] < dmax}) if n > 1 else []
    cache = {}
    def data(u):
        if u not in cache:
            idx = envs[u].meta["idx"]
            cache[u] = (extract(trace, res[idx], lo, hi, transform), res[idx] / float(sr))
        return cache[u]
    rows = []
    for i, j in cand:
        ua, ub = units[i], units[j]
        (XA, tA), (XB, tB) = data(ua), data(ub)
        ev = merge_evidence(XA, tA, XB, tB, core, noise, envs[ua], envs[ub], maxlag=maxlag)
        veto = ev["inflation"] > infl_thr
        rows.append(dict(ua=ua, ub=ub, dprime=float(D[i, j]), vetoed=bool(veto),
                         reason=f"inflation {ev['inflation']:.3f}" if veto else "", **ev))
        if veto:
            D[i, j] = D[j, i] = D[i, j] + 1e6
    if cannot_link is not None:
        for i in range(n):
            for j in range(i + 1, n):
                why = cannot_link(units[i], units[j])
                if why:
                    D[i, j] = D[j, i] = D[i, j] + 1e6
                    rows.append(dict(ua=units[i], ub=units[j], dprime=float(D[i, j] - 1e6), vetoed=True,
                                     inflation=None, adapt_joint=None, adapt_sep=None, shape_gain=None,
                                     reason=str(why)))
    new = np.asarray(clu).copy()
    if n > 1:
        cl = fcluster(linkage(squareform(D, checks=False), "average"), float(theta), "distance")
        groups = {}
        for u, c in zip(units, cl):
            groups.setdefault(int(c), []).append(u)
        for mem in groups.values():
            tgt = min(mem)
            for u in mem:
                new[np.asarray(clu) == u] = tgt
        gid = {u: min(groups[int(c)]) for u, c in zip(units, cl)}
        for r in rows:
            r["merged"] = gid[r["ua"]] == gid[r["ub"]]
        log(f"merge: {n} units -> {len(groups)} at d' < {theta}; {sum(r['vetoed'] for r in rows)} "
            f"of {len(rows)} candidate pairs vetoed by inflation > {infl_thr}")
    return new, rows


def write_merge_table(path, rows):
    cols = ("ua", "ub", "dprime", "inflation", "adapt_joint", "adapt_sep", "shape_gain", "vetoed", "merged",
            "reason")
    def fmt(c, v):
        if v is None or v == "":
            return "-"
        if c == "reason":
            return str(v)
        if isinstance(v, (bool, np.bool_)) or c in ("ua", "ub"):
            return str(int(v))
        return f"{v:.4f}"
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(fmt(c, r.get(c)) for c in cols) + "\n")
    return path


def write_tables(base, elec, tag, envs, table, res_abs):
    """<base>.env.<elec>[.<tag>]  one row per unit;  <base>.nco.<elec>[.<tag>]  one row per scored
    spike (near-collision table).  Both TAB-separated with a header."""
    sfx = f".{elec}" + (f".{tag}" if tag else "")
    p_env = f"{base}.env{sfx}"
    with open(p_env, "w") as f:
        nc = next(iter(envs.values())).sigma.shape[1]
        f.write("unit\tn\tn_clean\tflag_frac\tthreshold\tsd_amp\tsd_deriv\tnaive_over_robust\t"
                "chan_ratio_max\tshape_gain\tadapt_tau\tadapt_r2\twidth_ms\tburst\trate_hz\t" + "\t".join(f"excess_c{c}" for c in range(nc)) + "\t"
                + "\t".join(f"chan_ratio_c{c}" for c in range(nc)) + "\n")
        for u, e in sorted(envs.items()):
            f.write(f"{u}\t{e.n}\t{e.n_clean}\t{e.flagged.mean():.4f}\t{e.threshold:.2f}\t"
                    f"{np.sqrt(e.var_a):.4f}\t{np.sqrt(e.var_b):.4f}\t"
                    f"{np.median(e.sigma_naive / e.sigma):.3f}\t{e.channel_ratio().max():.3f}\t"
                    + (f"{e.shape_gain():.3f}\t{e.adapt['tau']:.3f}\t{e.adapt['r2']:.3f}\t" if e.adapt
                       else "-\t-\t-\t")
                    + (f"{e.meta['type']['width_ms']:.3f}\t{e.meta['type']['burst']:.3f}\t{e.meta['type']['rate_hz']:.2f}\t"
                       if e.meta.get("type") else "-\t-\t-\t")
                    + "\t".join(f"{x:.3f}" for x in e.excess_ratio()) + "\t"
                    + "\t".join(f"{x:.3f}" for x in e.channel_ratio()) + "\n")
    p_nco = f"{base}.nco{sfx}"
    with open(p_nco, "w") as f:
        f.write("spike\ttime\tunit\tscore\tflagged\tbump_pos\tpartner\tshift\tcoef\texplained\n")
        for i in range(table["idx"].size):
            f.write(f"{int(table['idx'][i])}\t{int(res_abs[int(table['idx'][i])])}\t{int(table['unit'][i])}\t"
                    f"{table['score'][i]:.2f}\t{int(table['flagged'][i])}\t{int(table['bump_pos'][i])}\t"
                    f"{int(table['partner'][i])}\t{int(table['shift'][i])}\t{table['coef'][i]:.3f}\t"
                    f"{int(table['explained'][i])}\n")
    return p_env, p_nco


def eap_manifest(table):
    """Explained near collisions -> fiber_decollide manifest rows (k1 at 0, k2 at its shift)."""
    m = table["explained"].astype(bool)
    return [dict(source_idx=int(i), k1=int(u), tau1=0, k2=int(p), tau2=int(s))
            for i, u, p, s in zip(table["idx"][m], table["unit"][m], table["partner"][m], table["shift"][m])]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Collision-robust per-unit variance envelopes + near-collision partners.")
    ap.add_argument("base"); ap.add_argument("elec", type=int)
    ap.add_argument("--signal", required=True,
                    help="continuous filtered int16 file to read windows from (e.g. BASE.fil)")
    ap.add_argument("--total-channels", type=int,
                    help="channels in the signal file (default: from the session yaml)")
    ap.add_argument("--channels", help="group channels, e.g. 40-47 or 40,41,... (default: session yaml)")
    ap.add_argument("--signal-start", type=int, default=0,
                    help="absolute sample of the signal file's first row (for chunks)")
    ap.add_argument("--space", default=None,
                    help="waveform space: standard | stderiv | stderiv_C<order> (default: the "
                         "--clu-method token without its _D suffix, i.e. the clu's own space)")
    ap.add_argument("--sdiff-pairs", default=None,
                    help="sdiffPairs pattern for a _C space (default: the session yaml's group entry)")
    ap.add_argument("--res-variant", default="stderiv", help="variant of the spike-time .res")
    nio.add_clu_args(ap, method_default="stderiv_C5_D34", stage_default="",
                     stage_dest="clu_stage", stage_alias=None)
    ap.add_argument("--pre", type=int, default=21, help="core window samples before the detected trough")
    ap.add_argument("--post", type=int, default=21, help="core window samples from the trough on")
    ap.add_argument("--margin", type=int, default=32, help="extended-window margin for partners")
    ap.add_argument("--min-n", type=int, default=50, help="skip units with fewer spikes")
    ap.add_argument("--max-per", type=int, default=3000,
                    help="spikes per unit used to FIT the envelope (all spikes are scored)")
    ap.add_argument("--q", type=float, default=0.99, help="null quantile of the bump threshold (off-spike noise windows)")
    ap.add_argument("--out-tag", default="", help="tag for the .env/.nco tables")
    ap.add_argument("--sr", type=float, default=None, help="sampling rate (default: session yaml, else 32552)")
    ap.add_argument("--merge", type=float, default=None, metavar="THETA",
                    help="also merge units: average linkage on envelope d' cut at THETA (7 on g6), with "
                         "merge inflation as a cannot-link; writes a new .clu stage + .envmerge table")
    ap.add_argument("--infl-thr", type=float, default=0.1,
                    help="merge inflation above which a pair cannot be merged (default 0.1)")
    ap.add_argument("--type-veto", action="store_true",
                    help="never merge a narrow- with a broad-spiking unit (off by default: on g6 fiber_session "
                         "fragments the per-fragment width is too noisy and the veto cost 0.11 ARI)")
    ap.add_argument("--merge-tag", default=None,
                    help="stage tag of the merged .clu (default: <clu-stage>.envmerge)")
    ap.add_argument("--eap", action="store_true",
                    help="write explained near collisions as .eap membership on the analysed --clu-stage "
                         "(Klusters' collision ring)")
    a = ap.parse_args(argv)
    try:
        from . import session_yaml as sy
    except ImportError:
        import session_yaml as sy
    info = {}
    if sy.find_session_yaml(a.base) is not None:
        info = sy.load_session(a.base, a.elec)
    if a.channels:
        ch = a.channels
        chans = (list(range(int(ch.split("-")[0]), int(ch.split("-")[1]) + 1))
                 if "-" in ch and "," not in ch else [int(x) for x in ch.split(",")])
    else:
        chans = info.get("channels")
    ntot = a.total_channels or info.get("ntotal")
    if not chans or not ntot:
        ap.error("--channels/--total-channels not given and no session yaml to read them from")
    if a.space is None:
        sp = nio.parse_variant_token(a.clu_method)
        a.space = sp.family if sp.kind is None else f"{sp.family}_{sp.kind}{sp.order}"
    transform = space_transform(a.space, a.sdiff_pairs or info.get("sdiff_pairs"))
    print(f"waveform space: {a.space}" + (f" (sdiffPairs {a.sdiff_pairs or info.get('sdiff_pairs')})"
                                          if "_C" in a.space else ""))
    sig = nio.open_signal(a.signal, ntot)

    class _View:                       # group-channel view of the memmap, rows on demand
        shape = (sig.shape[0], len(chans))
        def __getitem__(self, rows):
            return np.asarray(sig[rows][:, chans], np.float32)
    res_abs = nio.read_res_at(a.base, a.elec, variant=a.res_variant)
    _, clu = nio.resolve_clu(a, a.base, a.elec, n_spikes=res_abs.size, stage_dest="clu_stage")
    rel = res_abs - a.signal_start
    inwin = (rel >= 0) & (rel < sig.shape[0])
    lab = np.where(inwin, clu, 0)
    sr = a.sr or info.get("sr") or 32552.0
    view = _View(); rel0 = np.where(inwin, rel, 0)
    log = lambda s: print(s, file=sys.stderr, flush=True)  # noqa: E731
    envs, table, noise = run(view, rel0, lab, pre=a.pre, post=a.post, margin=a.margin,
                             transform=transform, min_n=a.min_n, max_per=a.max_per, q=a.q, sr=sr,
                             log=log)
    p_env, p_nco = write_tables(a.base, a.elec, a.out_tag, envs, table, res_abs)
    print(f"wrote {p_env} and {p_nco}")
    nflag = int(table["flagged"].sum()); nexp = int(table["explained"].sum())
    print(f"{table['idx'].size} spikes scored, {nflag} flagged, {nexp} explained by a partner")
    if a.merge is not None:
        types = {u: e.meta["type"] for u, e in envs.items() if e.meta.get("type")}
        new, rows = merge_units(view, rel0, lab, envs, noise, theta=a.merge, pre=a.pre, post=a.post,
                                margin=a.margin, transform=transform, sr=sr, infl_thr=a.infl_thr,
                                cannot_link=cell_type_conflict(types) if a.type_veto else None, log=log)
        out = np.where(inwin, new, clu)
        tag = a.merge_tag or (f"{a.clu_stage}.envmerge" if a.clu_stage else "envmerge")
        p_clu = nio.write_clu(a.base, a.elec, out, variant=a.clu_method, tag=tag)
        p_tab = write_merge_table(f"{a.base}.envmerge.{a.elec}" + (f".{a.out_tag}" if a.out_tag else ""), rows)
        print(f"merged: {np.unique(clu[inwin & (clu > 1)]).size} -> {np.unique(out[inwin & (out > 1)]).size} units "
              f"in the window; wrote {p_clu} and {p_tab}")
    if a.eap:
        try:
            from . import fiber_decollide as fd
        except ImportError:
            import fiber_decollide as fd
        r = fd.write_eap_membership(a.base, a.elec, parent_tag=a.clu_stage, n_spikes=res_abs.size,
                                    manifest=eap_manifest(table))
        print(f"eap: {r}")


if __name__ == "__main__":
    main()
