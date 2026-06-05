#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════════════════════
#  fiber_adapt.py — spike-frequency-adaptation (EWMA-τ) model + de-adaptation.
#
#  For RS (pyramidal) cells the fiber's ENERGY curve is largely produced by
#  spike-frequency adaptation: a spike following recent firing at short ISI is
#  attenuated, so it sits at lower energy.  A leaky integrator over spike times
#      a[i] = exp(-Δt/τ)·(a[i-1] + 1)              (state from PRIOR spikes)
#  tracks that history.  A fiber is "adapting" when corr(a, energy-position)
#  exceeds ~0.2 (FS/interneurons sit near 0 — adaptation does not drive them).
#  De-adaptation fits the law r = g(a) and rescales each spike's amplitude to
#  the un-adapted (a=0) reference, collapsing the energy spread.  This attacks
#  cross-energy fragmentation at its cause (for RS cells) rather than stitching
#  the curve geometrically.  Single-τ is enough (prior: ~270–620 ms; a fast+slow
#  term did not change R); the whitener stays history-independent.
# ════════════════════════════════════════════════════════════════════════════
import numpy as np
try:
    from . import fiber_lib as fl
except ImportError:
    import fiber_lib as fl


def ewma_multi(ts, taus):
    """Adaptation state for MANY taus at once -> (len(taus), len(ts)).

    The recurrence a[i] = exp(-Δt_i/τ)·(a[i-1]+1) is sequential in i but
    independent across τ, so we run the single i-loop once with the τ axis
    vectorized — bit-identical to calling ewma() per τ (no cumprod, so it stays
    numerically stable for long trains), ~12x faster over the 30-τ adapt grid."""
    ts = np.asarray(ts, float); taus = np.asarray(taus, float); n = len(ts)
    A = np.zeros((len(taus), n))
    if n < 2:
        return A
    dec = np.exp(-np.diff(ts)[None, :] / taus[:, None])    # (T, n-1)
    for i in range(1, n):
        A[:, i] = dec[:, i - 1] * (A[:, i - 1] + 1.0)
    return A


def ewma(ts, tau):
    """Adaptation state from time-ordered spike times (seconds)."""
    return ewma_multi(ts, [tau])[0]


def adapt_fit(t, r, taus=None):
    """Fit τ maximizing |corr(EWMA state, energy position)|.  position=1 at low
    energy (most-adapted).  Returns (corr, tau, a) with a in t's original order."""
    if taus is None:
        taus = np.logspace(np.log10(0.005), np.log10(2.0), 30)
    o = np.argsort(t, kind='mergesort'); ts = t[o]; rs = r[o]
    pos = 1.0 - np.argsort(np.argsort(rs)) / (len(rs) - 1 + 1e-9)
    best = (0.0, float(taus[len(taus) // 2]), np.zeros(len(t)))
    A = ewma_multi(ts, taus)                              # all taus at once (exact, ~12x)
    for ti, tau in enumerate(taus):
        a = A[ti]
        if a.std() < 1e-9:
            continue
        c = np.corrcoef(a, pos)[0, 1]
        if not np.isnan(c) and abs(c) > abs(best[0]):
            best = (float(c), float(tau), a)
    corr, tau, a_ord = best
    a = np.empty(len(t)); a[o] = a_ord
    return corr, tau, a


def fiber_adaptation(waves, res, W, nmean, mask, sr, taus=None):
    """Adaptation fingerprint of one fiber (no rescaling): (corr, tau, a, r)."""
    n = len(waves)
    Xw = (fl.realign(waves)[:, mask, :].reshape(n, -1) - nmean) @ W
    r = np.linalg.norm(Xw, axis=1); t = np.asarray(res, float) / sr
    corr, tau, a = adapt_fit(t, r, taus)
    return corr, tau, a, r


def deadapt(waves, res, W, nmean, mask, sr, taus=None, min_corr=0.2, clip=(0.2, 5.0)):
    """De-adapt one unit's waveforms.  Fit r = b0 + b1·a, rescale each spike's
    amplitude by r_ref/r_pred (r_ref = a=0 reference) to collapse the energy
    curve.  RS cells (|corr|>=min_corr) are de-adapted; FS cells returned as-is.
    Returns (deadapted_waves, info{tau,corr,adapting,collapse,n})."""
    waves = np.asarray(waves, float); n = len(waves)
    corr, tau, a, r = fiber_adaptation(waves, res, W, nmean, mask, sr, taus)
    info = dict(tau=float(tau), corr=float(corr), n=int(n),
                adapting=bool(abs(corr) >= min_corr), collapse=0.0)
    if not info['adapting']:
        return waves.copy(), info
    A = np.vstack([np.ones(n), a]).T
    b = np.linalg.lstsq(A, r, rcond=None)[0]
    r_ref = b[0]; r_pred = np.maximum(A @ b, 1e-6)
    s = np.clip(r_ref / r_pred, clip[0], clip[1])
    cv0 = r.std() / (r.mean() + 1e-9); cv1 = (r * s).std() / ((r * s).mean() + 1e-9)
    info['collapse'] = float(1 - cv1 / (cv0 + 1e-9))
    return waves * s[:, None, None], info


def adaptation_residual(waves, res, W, nmean, mask, sr, taus=None):
    """Per-spike adaptation-sequence consistency.  z_i = (r_i - g(a_i))/sigma
    with a_i the causal EWMA of PRECEDING spikes and g the fitted law.  |z| large
    => the spike's energy does not fit its recent neighbors (collision /
    contamination / misassignment).  Returns (z, info)."""
    corr, tau, a, r = fiber_adaptation(waves, res, W, nmean, mask, sr, taus)
    A = np.vstack([np.ones(len(a)), a]).T
    b = np.linalg.lstsq(A, r, rcond=None)[0]
    e = r - A @ b
    sigma = 1.4826 * np.median(np.abs(e - np.median(e))) + 1e-9
    z = e / sigma
    snr = abs(b[1]) * float(np.std(a)) / sigma
    return z, dict(tau=float(tau), corr=float(corr), b0=float(b[0]), b1=float(b[1]),
                   sigma=float(sigma), snr=float(snr))
