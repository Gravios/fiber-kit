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
except ImportError:
    import fiber_tracer as ft


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


# ════════════════════════════════════════════════════════════════════════════
#  Cluster-splitting primitives lifted out of fiber_session.
#
#  These are algorithms, not a stage: dip-test bisection, offset-overlay
#  splitting, variance-driven recursion and the shape features they judge on.
#  They lived in fiber_session only because that is where they were first
#  written, which made a 1800-line CLI stage the de-facto library -- fiber_refine
#  imported it purely to reach these, and reached them through their PRIVATE
#  names (fs._dipsplit_rec, fs._nudge_split, ...), so the module had no say over
#  what was and was not its interface.
#
#  Moved verbatim: the bodies are byte-identical to what fiber_session ran.  The
#  dependency closure is self-contained -- numpy, sklearn, diptest, fiber_lib,
#  fiber_tracer, fiber_pca, fiber_cfiber -- with no call back into fiber_session,
#  which is what made the move possible without also moving the stage.
#
#  Names keep their leading underscore so every existing reference, internal and
#  through the fiber_session aliases below, still resolves; the underscore now
#  means "internal to the splitting layer" rather than "internal to a stage".
# ════════════════════════════════════════════════════════════════════════════
try:
    from . import fiber_lib as fl
except ImportError:
    import fiber_lib as fl
try:
    from . import fiber_pca as _fpca
except ImportError:
    import fiber_pca as _fpca
try:
    from . import fiber_cfiber as fcf
except ImportError:
    import fiber_cfiber as fcf
# Guarded exactly as fiber_session guarded it: diptest is optional, and a bare
# import here would turn an optional dependency into a hard one for every module
# that imports the splitting layer.
try:
    import diptest as _diptest
    _HAVE_DIP = True
except Exception:
    _diptest = None
    _HAVE_DIP = False

def _dipsplit_rec(F, idx, min_size=40, alpha=0.01, depth=0, maxd=4):
    """Recursive DipSplit: 2-means -> project on the centroid axis -> Hartigan dip
    test -> split iff p<alpha and both halves substantial.  Catches bimodal clusters
    BIC left merged.  F: (n,dim) low-dim features; idx: indices into F."""
    n = len(idx)
    if not _HAVE_DIP or n < 2 * min_size or depth > maxd: return [idx]
    km = KMeans(2, n_init=4, random_state=0).fit(F[idx])
    a = idx[km.labels_ == 0]; b = idx[km.labels_ == 1]
    if len(a) < min_size or len(b) < min_size: return [idx]
    dr = km.cluster_centers_[1] - km.cluster_centers_[0]; dr /= np.linalg.norm(dr) + 1e-9
    _, p = _diptest.diptest(np.ascontiguousarray(F[idx] @ dr))
    if p < alpha:
        return (_dipsplit_rec(F, a, min_size, alpha, depth + 1, maxd) +
                _dipsplit_rec(F, b, min_size, alpha, depth + 1, maxd))
    return [idx]

def _aligned_pca(waves, mask, k, basis=None):
    """Realign a (sub)cluster to its OWN median by iterated circular cross-correlation
    (fiber_lib.align_xcorr, the channel-summed sub-sample aligner) and return SHAPE features.
    If a global ndm_pca `basis` is given the median-aligned waveforms are projected onto it
    (shared basis across chunks/nodes); else the top-k scores of a per-call local SVD.  The
    integer dominant-channel fl.realign locks a sub-cluster onto the PARENT's peak; re-aligning
    to this node's own median before featurizing lets a deeper bisection be measured on correct
    alignment.  The xcorr realignment itself is unchanged -- only the feature projection moves to
    the global basis."""
    al = fl.align_xcorr(waves, ref="median", iters=6, maxlag=6)
    if basis is not None:
        F = _fpca.cluster_features(al, basis, realign=False, dims=k)
        if F is not None:
            return F
    w = al[:, mask, :].reshape(len(waves), -1)
    w = w - w.mean(0)
    U, S, _ = np.linalg.svd(w, full_matrices=False)
    return U[:, :k] * S[:k]

def _dipsplit_realign(waves, mask, dim, min_size=40, alpha=0.01, depth=0, maxd=4, basis=None):
    """Recursive DipSplit that REALIGNS EACH NODE to its own median before deciding the split:
    the 2-means centroid axis and dip test are recomputed from this sub-cluster's median-aligned
    SHAPE features (_aligned_pca: global ndm_pca basis when given, else local SVD), so every
    bisection is judged on its own alignment instead of the parent's (the per-step realign).
    Returns a list of index arrays into `waves`."""
    n = len(waves)
    if not _HAVE_DIP or n < 2 * min_size or depth > maxd:
        return [np.arange(n)]
    F = _aligned_pca(waves, mask, dim, basis=basis)         # realign THIS node + featurize
    km = KMeans(2, n_init=4, random_state=0).fit(F)
    a = np.flatnonzero(km.labels_ == 0); b = np.flatnonzero(km.labels_ == 1)
    if len(a) < min_size or len(b) < min_size:
        return [np.arange(n)]
    dr = km.cluster_centers_[1] - km.cluster_centers_[0]; dr /= np.linalg.norm(dr) + 1e-9
    _, p = _diptest.diptest(np.ascontiguousarray(F @ dr))
    if p >= alpha:
        return [np.arange(n)]
    out = []
    for loc in (a, b):
        for piece in _dipsplit_realign(waves[loc], mask, dim, min_size, alpha, depth + 1, maxd, basis=basis):
            out.append(loc[piece])
    return out

def _amp_spread(waves, mask):
    """Peak amplitude and number of signal channels of the median-aligned template — the
    low-amplitude / broad-noise gate for the nudge split."""
    T = np.median(fl.align_xcorr(waves, ref="median", iters=6, maxlag=6), 0)
    ptp = T.max(0) - T.min(0)
    amp = float(ptp.max())
    nch = int((ptp > 0.25 * amp).sum())
    return amp, nch

def _nudge_split(waves, mask, dim, min_size, alpha, max_nudge=3):
    """Offset-overlay split for low-amplitude clusters.  Two neurons of similar shape a few
    samples apart are MERGED by median realignment — it collapses the offset, so the per-node
    realign dipsplit returns them as one (validated: ARI 0.00).  Split on each spike's alignment
    LAG to the cluster median instead — the 'nudge' each spike wants: bimodal lags = two
    temporally-offset sub-units (ARI 0.98).  Self-gating: clean clusters have unimodal lags and
    are returned whole (0% spurious splits in test).  Each offset sub-cluster is then realigned to
    its own median and dip-refined (reusing _dipsplit_realign)."""
    n = len(waves)
    if not _HAVE_DIP or n < 2 * min_size:
        return [np.arange(n)]
    _, sh = fl.align_xcorr(waves, ref="median", iters=6, maxlag=max_nudge, return_shifts=True)
    parts = _dipsplit_rec(np.asarray(sh, float).reshape(-1, 1), np.arange(n), min_size, alpha)
    if len(parts) == 1:                                  # unimodal lags -> no offset overlay
        return [np.arange(n)]
    out = []
    for p in parts:                                      # refine each offset sub-unit on its own alignment
        for piece in _dipsplit_realign(waves[p], mask, dim, min_size, alpha):
            out.append(p[piece])
    return out

def _variance_split(waves, W, nmean, mask, n_grid, peak, margin, min_n, dims,
                    depth=0, max_depth=4):
    """Variance-driven auto-split: recursively bisect a fiber WHILE its per-channel
    residual-variance profile is peaked (channel-localized contamination) AND each
    bisection lowers the mean per-channel residual variance by >= margin.

    The measure is the stopping criterion, so it finds the right number of shape
    sub-units (no rkk over-fragmentation) and never splits on energy (an
    energy-only difference leaves the trajectory residual flat). Each split is on
    the trajectory residual WEIGHTED toward the high-variance channels, so the
    bisection looks where the contamination actually is. Returns a list of index
    arrays into `waves`."""
    n = len(waves)
    if n < 2 * min_n or depth >= max_depth:
        return [np.arange(n)]
    prof = ft.channel_residual_profile(waves, W, nmean, mask, n_grid=n_grid)
    vc = prof['per_channel']; med = float(np.median(vc)) + 1e-12
    if vc.max() / med < peak:                         # flat profile -> no shape contamination
        return [np.arange(n)]
    wch = np.sqrt(np.maximum(vc - med, 0.0))          # weight discriminative channels (excess over floor)
    if not np.any(wch > 0):
        return [np.arange(n)]
    F = (prof['residual'] * wch[None, None, :]).reshape(n, -1); F = F - F.mean(0)
    U, S, _ = np.linalg.svd(F, full_matrices=False); Fr = U[:, :dims] * S[:dims]
    km = KMeans(2, n_init=5, random_state=0).fit_predict(Fr)
    if np.bincount(km).min() < min_n:
        return [np.arange(n)]
    _, _, red = ft.split_meanvar(waves, km, W, nmean, mask, n_grid=n_grid, min_n=min_n)
    if red < margin:                                  # bisection doesn't reduce the measure -> stop
        return [np.arange(n)]
    out = []
    for k in (0, 1):
        idx = np.flatnonzero(km == k)
        for piece in _variance_split(waves[idx], W, nmean, mask, n_grid, peak, margin,
                                     min_n, dims, depth + 1, max_depth):
            out.append(idx[piece])
    return out

def _cfiber_edge_filter(edges, fine, waves, mask, q=0.90, modes=(2, 3, 4, -1, -2, -3)):
    """Veto candidate fragment-merge edges whose affine-invariant cfiber SHAPE disagrees.
    cfiber AUC on well-populated g5 units is ~0.998, so a shape mismatch beyond the within-
    fiber split-half null is strong evidence of two cells.  The veto threshold is CALIBRATED
    per chunk from that null (quantile q), so it adapts to the chunk's noise rather than a
    fixed constant.  Edges where either fiber is too small to estimate a stable shape are
    LEFT ALONE (the gate only vetoes when it is confident).  Returns the filtered edges."""
    if not edges:
        return edges
    theta = fcf.channel_angles(waves.shape[2])
    mi = np.asarray(mask); win = slice(int(mi.min()), int(mi.max()) + 1)
    rng = np.random.default_rng(0)
    def shp(idx):
        if len(idx) < 6:
            return None
        t = fl.realign(waves[idx]).mean(0)
        z = fcf.complex_loop(t[None], theta, win)[0]
        s, _, _, _ = fcf.shape_descriptor(z[None], modes)
        return s[0]
    nodes = sorted({u for e in edges for u in e})
    S = {}; nulls = []
    for u in nodes:
        ix = np.flatnonzero(fine == u); S[u] = shp(ix)
        if len(ix) >= 12:
            pp = rng.permutation(len(ix)); h = len(pp) // 2
            a = shp(ix[pp[:h]]); b = shp(ix[pp[h:]])
            if a is not None and b is not None:
                nulls.append(float(np.linalg.norm(a - b)))
    thr = float(np.quantile(nulls, q)) if nulls else np.inf
    return [(i, j) for (i, j) in edges
            if S.get(i) is None or S.get(j) is None or float(np.linalg.norm(S[i] - S[j])) <= thr]
