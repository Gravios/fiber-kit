#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════════════════════
#  fiber_split.py — recursive residual splitting with a held-out acceptance gate.
#
#  A node's spikes are explained by a SINGLE shared fiber d(r) (energy-dependent
#  direction).  What that fiber cannot predict — the residual X - r·d(r) — is
#  where a second, envelope-similar unit hides (see fiber_adapt for the energy
#  structure that is legitimately removed first).  A candidate binary split is
#  proposed in the residual subspace and ACCEPTED only if it lowers OUT-OF-SAMPLE
#  residual energy more than a random split of the same node (a per-node null),
#  so the test does not reward the trivial decrease from simply adding clusters.
#  No labels are used; the objective is unsupervised residual energy.
# ════════════════════════════════════════════════════════════════════════════
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

try:
    from . import fiber_tracer as ft
    from . import fiber_adapt as fa
except ImportError:
    import fiber_tracer as ft
    import fiber_adapt as fa


def shared_fiber_residual(X, fit=None):
    """Residual of X to one shared fiber d(r) fit on `fit` rows (default all)."""
    Xf = X if fit is None else X[fit]
    tr = ft.trajectory(Xf)
    r = np.linalg.norm(X, axis=1)
    return X - r[:, None] * ft.predict_many(tr, r), tr


def _cv_residual_energy(X, labels, k=2, seed=0):
    """Σ over clusters of out-of-sample ‖X − shared-fiber recon‖²: fit d(r) on
    each fold's train rows, score the held-out rows, every row tested once."""
    rng = np.random.RandomState(seed); E = 0.0
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]; m = len(idx)
        if m < 2 * k:                                          # too small to fold: in-sample
            R, _ = shared_fiber_residual(X[idx]); E += float((R ** 2).sum()); continue
        fold = rng.permutation(m) % k
        for f in range(k):
            tr_i, te_i = idx[fold != f], idx[fold == f]
            trj = ft.trajectory(X[tr_i]); r = np.linalg.norm(X[te_i], axis=1)
            E += float(((X[te_i] - r[:, None] * ft.predict_many(trj, r)) ** 2).sum())
    return E


def _candidate_split(X, ncomp=8, seed=0):
    """Propose a binary split in the residual-to-shared-fiber subspace."""
    R, _ = shared_fiber_residual(X)
    k = min(ncomp, R.shape[1], max(1, len(R) - 1))
    Z = PCA(k).fit_transform(R)
    return KMeans(2, n_init=4, random_state=seed).fit_predict(Z)


def accept_split(X, margin=0.02, min_child=40, n_null=3, seed=0):
    """Accept a residual split iff it cuts held-out residual energy more than a
    random split of the same node.  Returns (labels|None, base, e_split, e_rand)."""
    base = _cv_residual_energy(X, np.zeros(len(X), int), seed=seed)
    lab = _candidate_split(X, seed=seed)
    if np.bincount(lab, minlength=2).min() < min_child:
        return None, base, base, base
    e_split = _cv_residual_energy(X, lab, seed=seed)
    rng = np.random.RandomState(seed + 1)
    e_rand = float(np.mean([_cv_residual_energy(X, rng.permutation(len(X)) % 2, seed=seed)
                            for _ in range(n_null)]))
    red, red_rand = (base - e_split) / base, (base - e_rand) / base
    return (lab if red > red_rand + margin else None), base, e_split, e_rand


def recursive_split(X, min_n=150, max_depth=6, margin=0.02, seed=0, _depth=0):
    """Recursively split a node's spikes; returns leaf labels 0..K-1.  Stops when
    the node is too small, too deep, or no residual split beats its random null."""
    n = len(X)
    if n < 2 * min_n or _depth >= max_depth:
        return np.zeros(n, int)
    lab, *_ = accept_split(X, margin=margin, min_child=max(40, min_n // 3), seed=seed + _depth)
    if lab is None:
        return np.zeros(n, int)
    out = np.empty(n, int); nxt = 0
    for c in (0, 1):
        idx = np.where(lab == c)[0]
        sub = recursive_split(X[idx], min_n, max_depth, margin, seed, _depth + 1)
        out[idx] = sub + nxt; nxt += int(sub.max()) + 1
    return out


def total_residual_energy(X, labels, seed=0):
    """Held-out total residual energy of a labelling (the unsupervised objective)."""
    return _cv_residual_energy(X, np.asarray(labels), seed=seed)
