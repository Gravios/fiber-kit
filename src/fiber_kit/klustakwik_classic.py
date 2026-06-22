# ---------------------------------------------------------------------------
#  klustakwik_classic.py
#
#  Faithful Python port of KiloKlustaKwik's *classic two-phase CEM* branch
#  (the `-ChunkMinutes 0` path) from neurosuite-3 src/kiloklustakwik/src/KK.cpp.
#  fiber-kit and neurosuite-3 stay code-separate: this is a native
#  reimplementation, not a shell-out to the KiloKlustaKwik binary.
#
#  Why this exists: the merge-only CEM in klustakwik.py (`rkk`) collapses two
#  co-active welded cells into one cluster — it has no split step, so it can
#  never exceed its initial cluster count, and its BIC merge over-penalises in
#  high D.  KiloKlustaKwik separates such welds with *random* init because of
#  two things this port reproduces:
#    1. TrySplits  — at convergence, try bisecting every cluster in the dims
#       where it is most BIMODAL (Sarle's b, not variance), accept the split
#       iff the penalised full-dim score improves.
#    2. AIC penalty (PenaltyMix = 0) — far less merge-happy than BIC, so a
#       genuine split survives ConsiderDeletion.
#
#  Faithful to KK.cpp: Penalty (KK.cpp:160), MStep (178), EStep cost (285),
#  CStep + Class2 (515), ConsiderDeletion (555), TrySplits (731), ComputeScore
#  (997), RunEMLoop (1023), CEMTwoPhase (1400).  GPU/time-shift/refeaturize
#  paths are intentionally omitted (not relevant to in-memory clustering).
# ---------------------------------------------------------------------------
import numpy as np
from scipy.linalg import solve_triangular, cholesky

HUGE = 1e30
_LOG2PI = float(np.log(2.0 * np.pi))


def _penalty(K, nDims, nPoints, penaltyMix):
    """KK.cpp:160.  AIC/BIC blend over (cov + mean + weight) params per real cluster."""
    if K <= 1:
        return 0.0
    nParams = (nDims * (nDims + 1) // 2 + nDims + 1) * (K - 1)
    return ((1.0 - penaltyMix) * nParams * 2.0
            + penaltyMix * nParams * np.log(nPoints) / 2.0)


def _mstep(X, Class, Kmax, noise_point):
    """KK.cpp:178.  Weights/means/covs; drop real clusters with <= nDims members."""
    N, D = X.shape
    cnt = np.bincount(Class, minlength=Kmax).astype(np.int64)
    alive = cnt > 0
    for c in range(1, Kmax):                       # noise (0) is never size-killed
        if alive[c] and cnt[c] <= D:
            alive[c] = False
    alive[0] = True                                # noise cluster always survives
    tot = N + noise_point
    W = np.zeros(Kmax)
    Mean = np.zeros((Kmax, D))
    Cov = np.zeros((Kmax, D, D))
    for c in range(Kmax):
        if not alive[c]:
            continue
        W[c] = (cnt[c] + (noise_point if c == 0 else 0.0)) / tot
        if c == 0:
            continue
        m = Class == c
        Xc = X[m]
        Mean[c] = Xc.mean(0)
        if cnt[c] > 1:
            d = Xc - Mean[c]
            Cov[c] = (d.T @ d) / (cnt[c] - 1)
    return W, Mean, Cov, alive


def _estep(X, W, Mean, Cov, alive, reg):
    """KK.cpp:285.  LogP[c,p] = 0.5*logdet(Cov_c) - log(W_c) + D/2*log2pi + 0.5*maha.
    Cluster 0 is uniform noise: LogP[0,p] = -log(W_0).  Singular clusters die."""
    N, D = X.shape
    Kmax = W.shape[0]
    LogP = np.full((Kmax, N), HUGE)
    alive = alive.copy()
    log2piHalf = _LOG2PI * D * 0.5
    if alive[0] and W[0] > 0:
        LogP[0, :] = -np.log(W[0])
    for c in range(1, Kmax):
        if not alive[c] or W[c] <= 0:
            continue
        try:
            L = cholesky(Cov[c] + reg * np.eye(D), lower=True)
        except np.linalg.LinAlgError:
            alive[c] = False
            continue
        logRootDet = np.sum(np.log(np.diag(L)))
        root = solve_triangular(L, (X - Mean[c]).T, lower=True)   # (D, N)
        maha = np.einsum('dn,dn->n', root, root)
        LogP[c, :] = logRootDet - np.log(W[c]) + log2piHalf + 0.5 * maha
    return LogP, alive


def _cstep(LogP, alive):
    """KK.cpp:515.  Assign to best alive cluster; record 2nd-best (Class2)."""
    rows = np.where(alive)[0]
    if rows.size == 0:
        z = np.zeros(LogP.shape[1], np.int64)
        return z, z
    sub = LogP[rows]                                # (nAlive, N)
    order = np.argsort(sub, axis=0)                 # ascending cost
    Class = rows[order[0]]
    Class2 = rows[order[1]] if rows.size > 1 else rows[order[0]]
    return Class.astype(np.int64), Class2.astype(np.int64)


def _compute_score(LogP, Class, K, nDims, nPoints, penaltyMix):
    """KK.cpp:997.  score = Penalty(K) + sum_p LogP[Class_p, p].  Lower is better."""
    return (_penalty(K, nDims, nPoints, penaltyMix)
            + LogP[Class, np.arange(LogP.shape[1])].sum())


def _consider_deletion(LogP, Class, Class2, alive, nDims, nPoints,
                       penaltyMix, min_clusters):
    """KK.cpp:555.  Delete the cluster whose reassignment loss to 2nd-best is
    cheaper than the penalty of one fewer cluster.  Returns (Class, alive)."""
    Kalive = int(alive.sum())
    if Kalive <= min_clusters + 1:                  # +1 for the noise cluster
        return Class, alive, False
    p = np.arange(LogP.shape[1])
    loss_p = LogP[Class2, p] - LogP[Class, p]
    DeletionLoss = np.where(alive, 0.0, HUGE)       # KK.cpp:559 alive->0, dead->HUGE
    np.add.at(DeletionLoss, Class, loss_p)
    DeletionLoss[~alive] = HUGE
    DeletionLoss[0] = HUGE                          # never delete noise
    cand = int(np.argmin(DeletionLoss))
    minLoss = DeletionLoss[cand]
    deltaPen = (_penalty(Kalive, nDims, nPoints, penaltyMix)
                - _penalty(Kalive - 1, nDims, nPoints, penaltyMix))
    if minLoss < deltaPen:
        Class = Class.copy()
        moved = Class == cand
        Class[moved] = Class2[moved]
        alive = alive.copy()
        alive[cand] = False
        return Class, alive, True
    return Class, alive, False


def _bimodal_dims(Xc, n_spatial, k_select):
    """KK.cpp:773.  Sarle's bimodality coefficient per spatial dim; top k_select.
    b = (skew^2 + 1) / (excess_kurt + 3(n-1)^2/((n-2)(n-3))).  Picks two-mode
    dims, not merely high-variance ones."""
    n = Xc.shape[0]
    if k_select >= n_spatial or n < 4:
        return np.arange(n_spatial)
    S = Xc[:, :n_spatial]
    mu = S.mean(0)
    d = S - mu
    m2 = (d ** 2).mean(0)
    m3 = (d ** 3).mean(0)
    m4 = (d ** 4).mean(0)
    kurt_corr = 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    with np.errstate(divide='ignore', invalid='ignore'):
        skew = m3 / np.power(m2, 1.5)
        kurt = m4 / (m2 * m2) - 3.0
        denom = kurt + kurt_corr
        b = np.where((m2 > 1e-12) & (denom > 1e-9), (skew * skew + 1.0) / denom, 0.0)
    sel = np.sort(np.argsort(-b)[:k_select])
    return sel


def _run_em_loop(X, Class, alive, *, enable_splits, max_iter, n_spatial,
                 penaltyMix, reg, noise_point, min_clusters, max_clusters,
                 split_every, split_depth, depth, rng):
    """KK.cpp:1023.  MStep -> EStep -> CStep -> ConsiderDeletion until converged;
    TrySplits at convergence / every split_every iters when enabled."""
    N, D = X.shape
    Kmax = alive.shape[0]
    iters = 0
    while True:
        W, Mean, Cov, alive = _mstep(X, Class, Kmax, noise_point)
        LogP, alive = _estep(X, W, Mean, Cov, alive, reg)
        newClass, Class2 = _cstep(LogP, alive)
        nChanged = int((newClass != Class).sum())
        Class = newClass
        Class, alive, deleted = _consider_deletion(LogP, Class, Class2, alive, D, N,
                                                    penaltyMix, min_clusters)
        iters += 1
        converged = (nChanged == 0) and not deleted
        if iters >= max_iter:
            break
        did_split = 0
        if (enable_splits and split_every > 0
                and (iters % split_every == split_every - 1 or converged)):
            Class, alive, did_split = _try_splits(
                X, Class, alive, n_spatial=n_spatial, penaltyMix=penaltyMix,
                reg=reg, noise_point=noise_point, min_clusters=min_clusters,
                max_clusters=max_clusters, split_every=split_every,
                split_depth=split_depth, depth=depth, rng=rng)
        if converged and not did_split:
            break
    return Class, alive


def _try_splits(X, Class, alive, *, n_spatial, penaltyMix, reg, noise_point,
                min_clusters, max_clusters, split_every, split_depth, depth, rng):
    """KK.cpp:731.  For each real cluster: select bimodal dims, run a 2-start
    sub-CEM there, and if the bisection lowers the FULL-dim penalised score,
    commit it (binary split: sub-cluster 1 stays, the rest become a new id)."""
    N, D = X.shape
    Kmax = alive.shape[0]
    if int(alive.sum()) - 1 >= max_clusters:        # -1 for noise
        return Class, alive, 0
    k_select = min(n_spatial, max(6, n_spatial // 2))
    # baseline full-dim score
    W, Mean, Cov, al0 = _mstep(X, Class, Kmax, noise_point)
    LogP0, al0 = _estep(X, W, Mean, Cov, al0, reg)
    base_score = _compute_score(LogP0, Class, int(al0.sum()), D, N, penaltyMix)
    did_split = 0
    for c in [cc for cc in np.where(alive)[0] if cc > 0]:
        if int(alive.sum()) - 1 >= max_clusters:
            break
        idx = np.where(Class == c)[0]
        if idx.size < 2 * (D + 1):
            continue
        Xc = X[idx]
        sel = _bimodal_dims(Xc, n_spatial, k_select)
        # sub-space = selected spatial dims (+ time col if present)
        cols = list(sel) + ([n_spatial] if D > n_spatial else [])
        Sub = Xc[:, cols]
        # unsplit baseline vs split, both inside the sub-space CEM
        unsplit = _klustakwik_core(Sub, rng, penaltyMix=penaltyMix, reg=reg,
                                   n_start=2, max_clusters=1, min_clusters=1,
                                   enable_splits=False, noise_point=0.0,
                                   split_every=split_every, split_depth=split_depth,
                                   depth=depth + 1, _return_score=True)
        do_recurse = depth < split_depth
        split_lab, split_score = _klustakwik_core(
            Sub, rng, penaltyMix=penaltyMix, reg=reg, n_start=13,
            max_clusters=max_clusters, min_clusters=1,
            enable_splits=do_recurse, noise_point=0.0,
            split_every=split_every, split_depth=split_depth,
            depth=depth + 1, _return_score=True, _return_labels=True)
        if split_score >= unsplit:
            continue
        # propose binary split: K2 sub-cluster 1 -> keep c, everything else -> new id
        new_id = next((j for j in range(1, Kmax) if not alive[j]), -1)
        if new_id < 0:
            break
        trial = Class.copy()
        trial[idx[split_lab != 1]] = new_id
        trial_alive = alive.copy()
        trial_alive[new_id] = True
        # settle 3 iters in full dims, splits off, then score
        trial, trial_alive = _run_em_loop(
            X, trial, trial_alive, enable_splits=False, max_iter=3,
            n_spatial=n_spatial, penaltyMix=penaltyMix, reg=reg,
            noise_point=noise_point, min_clusters=min_clusters,
            max_clusters=max_clusters, split_every=split_every,
            split_depth=split_depth, depth=depth + 1, rng=rng)
        Wt, Mt, Ct, alt = _mstep(X, trial, Kmax, noise_point)
        LogPt, alt = _estep(X, Wt, Mt, Ct, alt, reg)
        new_score = _compute_score(LogPt, trial, int(alt.sum()), D, N, penaltyMix)
        if new_score < base_score:
            Class, alive = trial, alt
            base_score = new_score
            did_split = 1
    return Class, alive, did_split


def _klustakwik_core(X, rng, *, penaltyMix, reg, n_start, max_clusters,
                     min_clusters, enable_splits, noise_point, split_every,
                     split_depth, depth, _return_score=False, _return_labels=False):
    """One CEM run from random init on X (last col treated as time iff present
    and n_spatial<D).  Returns labels and/or final score."""
    N, D = X.shape
    n_spatial = D - 1 if D > 1 else D
    Kmax = max(n_start, max_clusters) + 2
    # random init: classes 1..(n_start-1); cluster 0 reserved for noise
    n_real = max(1, n_start - 1)
    Class = rng.integers(1, n_real + 1, size=N).astype(np.int64) if n_real > 1 \
        else np.ones(N, dtype=np.int64)
    alive = np.zeros(Kmax, bool)
    alive[:n_start] = True
    alive[0] = True
    # Phase 1: spatial dims only (exclude time)
    Class, alive = _run_em_loop(
        X[:, :n_spatial] if n_spatial < D else X, Class, alive,
        enable_splits=enable_splits, max_iter=500, n_spatial=n_spatial,
        penaltyMix=penaltyMix, reg=reg, noise_point=noise_point,
        min_clusters=min_clusters, max_clusters=max_clusters,
        split_every=split_every, split_depth=split_depth, depth=depth, rng=rng)
    # Phase 2: short merge pass over full dims (time reintroduced), splits off
    if n_spatial < D:
        Class, alive = _run_em_loop(
            X, Class, alive, enable_splits=False, max_iter=30, n_spatial=n_spatial,
            penaltyMix=penaltyMix, reg=reg, noise_point=noise_point,
            min_clusters=min_clusters, max_clusters=max_clusters,
            split_every=split_every, split_depth=split_depth, depth=depth, rng=rng)
    W, Mean, Cov, alive = _mstep(X, Class, Kmax, noise_point)
    LogP, alive = _estep(X, W, Mean, Cov, alive, reg)
    Class, _ = _cstep(LogP, alive)
    score = _compute_score(LogP, Class, int(alive.sum()), D, N, penaltyMix)
    if _return_labels:
        return Class, score
    if _return_score:
        return score
    return Class


def klustakwik_classic(X, *, max_clusters=12, min_clusters=2, n_start=2,
                       penalty_mix=0.0, reg_frac=1e-2, seed=42, has_time=False,
                       split_every=10, split_depth=1, n_runs=5):
    """Cluster X with the classic two-phase CEM + TrySplits, random init.
    X: (N, D) features.  If has_time, the last column is the time dimension
    (excluded from Phase 1 spatial clustering, used in the Phase 2 merge).
    `n_runs` random restarts are run (KK's -nRuns); the lowest-score solution
    is returned -- this is what makes random init reliable.
    Returns 0-based contiguous per-point labels (0 = noise/MUA, >=1 = units)."""
    X = np.ascontiguousarray(X, dtype=float)
    N, D = X.shape
    if not has_time:                                # append a constant time col so
        X = np.hstack([X, np.zeros((N, 1))])        # Phase-1/Phase-2 split is clean
    reg = reg_frac * float(np.median(np.var(X[:, :-1], 0)))
    best_lab, best_score = None, np.inf
    for run in range(max(1, n_runs)):
        rng = np.random.default_rng(seed + run)
        lab, score = _klustakwik_core(
            X, rng, penaltyMix=penalty_mix, reg=reg, n_start=n_start,
            max_clusters=max_clusters, min_clusters=min_clusters,
            enable_splits=True, noise_point=0.0, split_every=split_every,
            split_depth=split_depth, depth=0, _return_labels=True)
        if score < best_score:
            best_score, best_lab = score, lab
    uniq = np.unique(best_lab)                       # contiguous 0-based relabel
    remap = {u: i for i, u in enumerate(uniq)}
    return np.array([remap[u] for u in best_lab], dtype=np.int64)
