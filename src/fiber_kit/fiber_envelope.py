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
    from . import fiber_stderiv as fst
    from . import neuro_io as nio
except ImportError:                       # script use
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
    meta: dict = field(default_factory=dict)

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


def fit_envelope(X, core, var_noise=None, noise_windows=None, *, maxlag=4, width=8, q=0.99,
                 iters=3, k_mad=4.0, ext=None):
    """Robust envelope of ONE unit.

    X           (n, Lx, C) snippets extracted with margin around the core window.
    core        slice of the core window inside X (must leave >= maxlag on both sides).
    var_noise   (C,) noise floor; None -> self-calibrated (lowest-decile robust variance per channel)
    noise_windows  (m, >=L, C) off-spike windows for the empirical null (centred core part used);
                None -> median+k*MAD rule.
    ext         slice of the extended window (relative to X) for the extended template (partner
                search); must leave >= maxlag on both sides.  None -> no extended template.
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
    Text = None
    if ext is not None:
        Text = np.median(take_lagged(X, lag, ext)[keep], 0)
    return Envelope(template=T, dtemplate=dT, var_noise=var_noise, var_a=var_a, var_b=var_b,
                    excess=excess, sigma=sigma, sigma_naive=naive, n=n, n_clean=int(keep.sum()),
                    flagged=~keep, score=score, bump_pos=pos, a=a, b=b, lag=lag, threshold=thr,
                    template_ext=Text)


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
        max_per=3000, width=8, q=0.99, seed=0, partners=True, log=print):
    """Envelopes for every unit with >= min_n spikes + partner search for its flagged spikes.
    res: spike times relative to `trace` start; clu: labels (<= 1 = noise, skipped).
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
        env = fit_envelope(X, core, noise.var, noise.windows, maxlag=maxlag, width=width, q=q, ext=ext)
        if fit_idx.size != idx.size:            # score every spike against the converged envelope
            X = extract(trace, res[idx], lo, hi, transform)
            w = 1.0 / env.var_noise
            lag = align_to_template(X, env.template, w, core, maxlag)
            a, b, R = fit_amp_jitter(take_lagged(X, lag, core), env.template, env.dtemplate, w)
            sc, pos = bump_scores(R, np.maximum(env.excess + env.var_noise[None], env.var_noise[None]), width)
            env.lag, env.a, env.b, env.score, env.bump_pos = lag, a, b, sc, pos
            env.flagged = sc > env.threshold
        env.meta["idx"] = idx
        envs[int(u)] = env
        rows[int(u)] = X if partners else None
        log(f"unit {int(u):4d}: n={idx.size:6d} flagged={env.flagged.mean():.3f} "
            f"excess_max={env.excess_ratio().max():.2f} naive/robust={np.median(env.sigma_naive / env.sigma):.2f}")
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
        table["idx"].append(idx); table["unit"].append(np.full(idx.size, u))
        table["score"].append(env.score); table["flagged"].append(f)
        table["bump_pos"].append(env.bump_pos - pre); table["partner"].append(part["partner"])
        table["shift"].append(part["shift"] + 0); table["coef"].append(part["coef"])
        table["explained"].append(part["explained"])
        rows[u] = None
    table = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in table.items()}
    o = np.argsort(table["idx"], kind="stable")
    return envs, {k: v[o] for k, v in table.items()}, noise


def write_tables(base, elec, tag, envs, table, res_abs):
    """<base>.env.<elec>[.<tag>]  one row per unit;  <base>.nco.<elec>[.<tag>]  one row per scored
    spike (near-collision table).  Both TAB-separated with a header."""
    sfx = f".{elec}" + (f".{tag}" if tag else "")
    p_env = f"{base}.env{sfx}"
    with open(p_env, "w") as f:
        f.write("unit\tn\tn_clean\tflag_frac\tthreshold\tsd_amp\tsd_deriv\tnaive_over_robust\t"
                + "\t".join(f"excess_c{c}" for c in range(next(iter(envs.values())).sigma.shape[1]))
                + "\n")
        for u, e in sorted(envs.items()):
            f.write(f"{u}\t{e.n}\t{e.n_clean}\t{e.flagged.mean():.4f}\t{e.threshold:.2f}\t"
                    f"{np.sqrt(e.var_a):.4f}\t{np.sqrt(e.var_b):.4f}\t"
                    f"{np.median(e.sigma_naive / e.sigma):.3f}\t"
                    + "\t".join(f"{x:.3f}" for x in e.excess_ratio()) + "\n")
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
    envs, table, _ = run(_View(), np.where(inwin, rel, 0), lab, pre=a.pre, post=a.post,
                         margin=a.margin, transform=transform,
                         min_n=a.min_n, max_per=a.max_per, q=a.q,
                         log=lambda s: print(s, file=sys.stderr, flush=True))
    p_env, p_nco = write_tables(a.base, a.elec, a.out_tag, envs, table, res_abs)
    print(f"wrote {p_env} and {p_nco}")
    nflag = int(table["flagged"].sum()); nexp = int(table["explained"].sum())
    print(f"{table['idx'].size} spikes scored, {nflag} flagged, {nexp} explained by a partner")
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
