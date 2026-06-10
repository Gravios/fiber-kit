#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════════════════════
#  fiber_geometry.py — fine-geometry fiber signature + strict link veto.
#
#  The cross-chunk linker must NOT regroup units the per-chunk sort separated.
#  A single mean-template vector is too coarse: distinct units routinely reach
#  cosine 0.6-0.85 (worst 0.98 == the same-unit median), so no template
#  threshold is both strict and useful.  The fiber GEOMETRY -- the unit-waveform
#  direction d(r) traced over spike energy r, as a CURVE rather than one point --
#  separates same-unit-over-time from different-unit at AUC ~0.99 and admits a
#  strict cut (curve distance <= ~1.3 -> 95% same-unit recall, ~1.5% false-merge
#  on real g5 data; the residual false merges are genuinely near-identical units).
#
#  Curves are built from UN-WHITENED masked templates (per-chunk whiteners
#  differ; raw footprints are comparable across chunks), then embedded in a
#  canonical PCA(3) direction basis fit on the pooled curve points at link time.
#  Distance is scale-free (mean per-point tip distance / mean curve length) so it
#  does not reward high-energy units.  Intended use: a VETO on top of the
#  overlap-anchor backbone (shared spikes = the positive identity evidence) --
#  never union two per-chunk fibers whose curves disagree, which also blocks the
#  union-find from chaining two geometrically-distinct fibers through one
#  spurious adjacent link.
# ════════════════════════════════════════════════════════════════════════════
import numpy as np

DEFAULT_NQ = 5
DEFAULT_GEO_THR = 1.3        # calibrated on real g5: 95% recall, ~1.5% false-merge


def fiber_curve(masked_templates_by_spike, energy, nq=DEFAULT_NQ):
    """d(r) curve as a stack of UN-WHITENED mean templates over energy quantiles.

    masked_templates_by_spike: (nspike, P) per-spike realigned masked waveforms
        flattened over (samples, channels) -- raw .spk space, not whitened.
    energy: (nspike,) per-spike energy (e.g. L2 norm of the masked waveform).
    Returns (nq, P): the mean raw template in each energy quantile bin (low->high
    energy).  Bins with too few spikes fall back to the fiber-wide mean."""
    Xs = np.asarray(masked_templates_by_spike, float)
    r = np.asarray(energy, float)
    edges = np.quantile(r, np.linspace(0, 1, nq + 1))
    edges[0] = -np.inf; edges[-1] = np.inf
    glob = Xs.mean(0)
    cur = np.empty((nq, Xs.shape[1]))
    for k in range(nq):
        m = (r >= edges[k]) & (r < edges[k + 1])
        cur[k] = Xs[m].mean(0) if m.sum() >= 5 else glob
    return cur


def geometry_basis(curves, k=3):
    """Canonical PCA(k) direction basis over pooled, unit-normalised curve points.
    `curves`: iterable of (nq, P) arrays.  Returns (mean: (P,), P_basis: (P, k))."""
    pts = np.vstack([c for c in curves])
    pts = pts / (np.linalg.norm(pts, axis=1, keepdims=True) + 1e-12)   # directions
    mu = pts.mean(0)
    _, _, Vt = np.linalg.svd(pts - mu, full_matrices=False)
    return mu, Vt[:k].T


def _embed(curve, mu, P):
    d = curve / (np.linalg.norm(curve, axis=1, keepdims=True) + 1e-12)
    return (d - mu) @ P


def _clen(emb):
    return float(np.mean([np.linalg.norm(emb[i + 1] - emb[i]) for i in range(len(emb) - 1)])) + 1e-9


def curve_distance(curve_i, curve_j, mu, P):
    """Scale-free distance between two fiber curves in canonical PCA(k): mean
    per-quantile tip distance normalised by the mean curve length.  Lower = more
    likely the same unit; ~0.96 same-unit vs ~5.6 different-unit on real g5."""
    ei, ej = _embed(curve_i, mu, P), _embed(curve_j, mu, P)
    return float(np.mean(np.linalg.norm(ei - ej, axis=1)) / (0.5 * (_clen(ei) + _clen(ej))))


def geo_veto(curve_i, curve_j, mu, P, thr=DEFAULT_GEO_THR):
    """True == REFUSE the link (curves too distinct to be the same unit)."""
    return curve_distance(curve_i, curve_j, mu, P) > thr
