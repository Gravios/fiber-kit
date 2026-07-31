#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════════════════════
#  klustakwik.py — classic KlustaKwik CEM, in parity with kiloklustakwik's
#                  classic RANDOM branch (-InitMethod random).
#
#  Every formula here is transcribed from neurosuite-3's kiloklustakwik KK.cpp
#  (CPU path) and carries the line it came from.  The two repositories stay
#  strictly separate: this is a reimplementation against a read specification,
#  never shared code.  When KK.cpp moves, those references are the re-check list.
#
#  Loop (KK.cpp RunEMLoop:1023):
#      do { MStep; EStep; nChanged = CStep; ConsiderDeletion; ComputeScore;
#           maybe TrySplits }
#      while (nChanged > 0 || !lastStepFull || didSplit)
#
#  THE NOISE CLUSTER.  Cluster 0 is a UNIFORM density, LogP = -log(Weight[0])
#  for every point (KK.cpp:295-300), and every Gaussian loop starts at c = 1
#  (KK.cpp:150, 303, 362, 584, 803).  It is classic KlustaKwik's reject option:
#  a point fitting no Gaussian well lands there rather than being forced into
#  the least-bad cluster.  The previous version of this file had no such
#  cluster, so its CEM had to place every spike in some Gaussian and clusters
#  kept a foreign tail no amount of iterating could shed.
#
#  SCORE IS MINIMISED, as in the C++ (LogP is -log p, not log p).
# ════════════════════════════════════════════════════════════════════════════
import numpy as np

try:
    from scipy.linalg import solve_triangular as _solve_tri
    _HAVE_SCIPY = True
except ImportError:                                    # pragma: no cover
    _HAVE_SCIPY = False

# ── defaults, from kiloklustakwik KlustaKwik.cpp ────────────────────────────
MAX_POSSIBLE_CLUSTERS = 500      # KlustaKwik.cpp:45
RANDOM_SEED           = 1        # KlustaKwik.cpp:48
FULL_STEP_EVERY       = 10       # KlustaKwik.cpp:54
CHANGED_THRESH        = 0.05     # KlustaKwik.cpp:55
MAX_ITER              = 500      # KlustaKwik.cpp:58
PENALTY_MIX           = 0.0      # KlustaKwik.cpp:78   0 = AIC, 1 = BIC
SPLIT_EVERY           = 8        # KlustaKwik.cpp:772
NOISE_POINT           = 1        # KK.cpp:105  (0 in split sub-objects, KK.cpp:703)
SPLIT_RECURSE_DEPTH   = 1        # TrySplits recursion cap (KK.cpp:903 call site)
HUGE_SCORE            = 1e32


class _KK:
    """One CEM problem — the C++ KK object, field names kept where they aid traceability."""

    def __init__(self, X, n_starting_clusters, penalty_mix=PENALTY_MIX,
                 noise_point=NOISE_POINT, max_possible=MAX_POSSIBLE_CLUSTERS,
                 rng=None, min_clusters_alive=1, split_depth=0):
        self.X = np.ascontiguousarray(X, dtype=float)
        self.N, self.D = self.X.shape
        self.maxK = int(max(max_possible, n_starting_clusters + 2))
        self.nstart = int(n_starting_clusters)
        self.pmix = float(penalty_mix)
        self.noise_point = int(noise_point)
        self.min_alive = int(min_clusters_alive)
        self.split_depth = int(split_depth)
        self.rng = np.random.default_rng(RANDOM_SEED) if rng is None else rng
        self.cls = np.zeros(self.N, dtype=int)
        self.cls2 = np.zeros(self.N, dtype=int)
        self.alive = np.zeros(self.maxK, dtype=bool)
        self.W = np.zeros(self.maxK)
        self.mean = np.zeros((self.maxK, self.D))
        self.cov = np.zeros((self.maxK, self.D, self.D))
        self.logp = np.full((self.maxK, self.N), HUGE_SCORE)
        self.full_step = 1

    def alive_ids(self):
        return np.flatnonzero(self.alive)      # ascending == C++ AliveIndex order (KK.cpp:147)

    def _first_free(self):
        free = np.flatnonzero(~self.alive)
        return int(free[0]) if free.size else None

    # ── Penalty (KK.cpp:160) ───────────────────────────────────────────────
    def penalty(self, n):
        if n <= 1:
            return 0.0
        nparams = (self.D * (self.D + 1) // 2 + self.D + 1) * (n - 1)
        return ((1.0 - self.pmix) * nparams * 2.0
                + self.pmix * nparams * np.log(self.N) / 2.0)

    # ── MStep (KK.cpp:178) ─────────────────────────────────────────────────
    def mstep(self):
        cnt = np.bincount(self.cls, minlength=self.maxK)
        for c in self.alive_ids():                 # KK.cpp:189  delete c>0 with n <= nDims
            if c > 0 and cnt[c] <= self.D:
                self.alive[c] = False
        cnt = np.bincount(self.cls, minlength=self.maxK)
        denom = self.N + self.noise_point
        for c in self.alive_ids():                 # KK.cpp:199  weights
            self.W[c] = ((cnt[c] + self.noise_point) / denom) if c == 0 else (cnt[c] / denom)
        for c in self.alive_ids():                 # KK.cpp:213-257  mean + sample covariance
            if c == 0 or cnt[c] == 0:
                continue
            Xc = self.X[self.cls == c]
            self.mean[c] = Xc.mean(0)
            if cnt[c] > 1:
                d = Xc - self.mean[c]
                self.cov[c] = (d.T @ d) / (cnt[c] - 1)     # (n-1) denominator, KK.cpp:253

    # ── EStep (KK.cpp:285) ─────────────────────────────────────────────────
    def estep(self):
        self.logp[:] = HUGE_SCORE
        self.logp[0] = -np.log(self.W[0]) if self.W[0] > 0 else HUGE_SCORE   # KK.cpp:295-300
        log2pi_half = 0.5 * self.D * np.log(2.0 * np.pi)                      # KK.cpp:106
        for c in self.alive_ids():
            if c == 0:
                continue
            if self.W[c] <= 0:
                self.alive[c] = False
                continue
            try:                                   # Cholesky failure kills the cluster (KK.cpp:303)
                L = np.linalg.cholesky(self.cov[c])
            except np.linalg.LinAlgError:
                self.alive[c] = False
                continue
            logrootdet = np.log(np.diag(L)).sum()                        # KK.cpp:361
            base = logrootdet - np.log(self.W[c]) + log2pi_half          # KK.cpp:372
            d = (self.X - self.mean[c]).T
            root = _solve_tri(L, d, lower=True) if _HAVE_SCIPY else np.linalg.solve(L, d)
            self.logp[c] = base + 0.5 * (root ** 2).sum(0)               # LogP = base + Mahal/2

    # ── CStep (KK.cpp:515) ─────────────────────────────────────────────────
    def cstep(self):
        ids = self.alive_ids()
        if ids.size == 0:
            return 0
        order = np.argsort(self.logp[ids], axis=0, kind="stable")   # ties -> lowest id (C++ '<')
        top = ids[order[0]]
        self.cls2 = ids[order[1]] if ids.size > 1 else top.copy()
        nchanged = int((top != self.cls).sum())
        self.cls = top
        return nchanged

    # ── ConsiderDeletion (KK.cpp:555) ──────────────────────────────────────
    def consider_deletion(self):
        ids = self.alive_ids()
        if ids.size <= self.min_alive:
            return
        loss = np.where(self.alive, 0.0, HUGE_SCORE)
        idx = np.arange(self.N)
        for c in ids:                              # KK.cpp:571  cost of moving each point to Class2
            if c == 0:
                continue
            m = self.cls == c
            if m.any():
                loss[c] = float((self.logp[self.cls2[m], idx[m]] - self.logp[c, idx[m]]).sum())
        loss[0] = HUGE_SCORE                       # noise cluster is never a candidate
        cand = int(np.argmin(loss))
        if loss[cand] >= HUGE_SCORE:
            return
        dpen = self.penalty(ids.size) - self.penalty(ids.size - 1)       # KK.cpp:597
        if loss[cand] < dpen:
            m = self.cls == cand
            self.cls[m] = self.cls2[m]
            self.alive[cand] = False

    # ── ComputeScore (KK.cpp:997) ──────────────────────────────────────────
    def compute_score(self):
        return float(self.penalty(int(self.alive.sum()))
                     + self.logp[self.cls, np.arange(self.N)].sum())

    # ── TrySplits (KK.cpp:731) ─────────────────────────────────────────────
    def try_splits(self):
        did = 0
        score = self.compute_score()
        for c in list(self.alive_ids()):
            if c == 0:
                continue
            idx = np.flatnonzero(self.cls == c)
            if idx.size <= self.D + 2:
                continue
            free = self._first_free()
            if free is None:
                break                              # KK.cpp:896  no free clusters, abandon
            Xc = self.X[idx]
            k1 = _KK(Xc, 2, self.pmix, 0, rng=self.rng, split_depth=self.split_depth + 1)
            unsplit = k1.cem(recurse=0)            # KK.cpp:901-902  1-cluster baseline
            k2 = _KK(Xc, 13, self.pmix, 0, rng=self.rng, split_depth=self.split_depth + 1)
            split = k2.cem(recurse=1 if self.split_depth < SPLIT_RECURSE_DEPTH else 0)  # :903
            if not (split < unsplit):              # KK.cpp:906
                continue
            trial = self.cls.copy()                # K3: cluster c -> {c, free}  (KK.cpp:909-915)
            trial[idx] = np.where(k2.cls == 1, c, free)
            k3 = _KK(self.X, 2, self.pmix, self.noise_point, self.maxK, self.rng,
                     self.min_alive, self.split_depth)
            k3.cls = trial
            k3.alive[:] = False
            k3.alive[np.unique(trial)] = True
            k3.alive[0] = True
            k3.mstep(); k3.estep(); k3.cstep()
            new_score = k3.compute_score()
            if new_score < score:                  # KK.cpp:919  accept only if full score improves
                self.cls = k3.cls
                self.alive = k3.alive.copy()
                score = new_score
                did = 1
        return did

    # ── RunEMLoop (KK.cpp:1023) ────────────────────────────────────────────
    def run_em_loop(self, enable_splits, max_iter=0):
        max_iter = MAX_ITER if max_iter <= 0 else max_iter
        it = 0
        self.full_step = 1
        while True:
            self.mstep()
            self.estep()
            nchanged = self.cstep()
            self.consider_deletion()
            it += 1
            last_full = self.full_step
            self.full_step = int(nchanged > CHANGED_THRESH * self.N
                                 or nchanged == 0
                                 or it % FULL_STEP_EVERY == 0)
            if it > max_iter:
                break
            did_split = 0
            if enable_splits and SPLIT_EVERY > 0 and (
                    it % SPLIT_EVERY == SPLIT_EVERY - 1 or (nchanged == 0 and last_full)):
                did_split = self.try_splits()
            if not (nchanged > 0 or not last_full or did_split):
                break
        self.mstep(); self.estep()
        return self.compute_score()

    # ── CEM, classic RANDOM init (KK.cpp:1114; InitMethod random KK.cpp:1447) ──
    def cem(self, recurse=1):
        if self.nstart > 1:                        # KK.cpp:1119  irand(1, nStartingClusters-1)
            self.cls = self.rng.integers(1, self.nstart, size=self.N)
        else:
            self.cls = np.zeros(self.N, dtype=int)
        self.alive[:] = False
        self.alive[:self.nstart] = True            # KK.cpp:1124  cluster 0 alive but empty
        return self.run_em_loop(enable_splits=bool(recurse))


def _sweep(lo, hi):
    """nStartingClusters values to try.  The classic driver walks EVERY K in
    [MinClusters, MaxClusters]; that is O(hi-lo) full CEMs, affordable in OpenMP C++ and not in
    NumPy.  Walk every value up to a span of 12, then geometrically, so both endpoints still run."""
    if hi - lo <= 12:
        return list(range(lo, hi + 1))
    out, k = [lo], lo
    while k < hi:
        k = max(k + 1, int(round(k * 1.5)))
        out.append(min(k, hi))
    return sorted(set(out))


def _compact(lab):
    """Renumber real clusters to 1..K, ALWAYS reserving label 0 for the noise cluster.

    Label 0 must mean "noise" unconditionally, including when the noise cluster ended up empty --
    which is the normal, healthy outcome.  Renumbering from 0 whenever cluster 0 happens to be
    unpopulated would silently hand the noise label to a real cluster, and every consumer that
    treats 0 as a reject bucket would then discard a genuine unit.
    """
    lab = np.asarray(lab, int)
    remap, nxt = {0: 0}, 1
    for u in np.unique(lab):
        if u != 0:
            remap[u] = nxt
            nxt += 1
    return np.array([remap[u] for u in lab], dtype=int)


def klustakwik(X, max_clusters=200, min_size=None, seed=RANDOM_SEED,
               penalty_mix=PENALTY_MIX, min_clusters=2, n_starts=1,
               splits=True, noise=True, verbose=False, delete=True,
               init_labels=None, **_legacy):
    """Classic KlustaKwik CEM in parity with kiloklustakwik's -InitMethod random branch.

    Returns per-point labels (0-based).  LABEL 0 IS THE NOISE CLUSTER — points that no Gaussian
    explains better than a uniform density.  Callers that assumed every label was a real cluster
    must now treat 0 as a reject bucket; that is the point of the change, and it is what the C++
    has always done.

    max_clusters / min_clusters  sweep nStartingClusters as the classic driver does, keeping the
                                 best (lowest) score; n_starts random restarts per K.
    penalty_mix                  0 = AIC (the kiloklustakwik default), 1 = BIC.
    splits                       TrySplits on/off (SplitEvery).  Off is far cheaper and is the
                                 honest setting when the caller wants only the CEM.
    noise                        keep the uniform noise cluster (NoisePoint=1); False restores
                                 this file's pre-parity behaviour of forcing every point.
    min_size, delete             accepted for call compatibility only.  The classic rule is a
                                 fixed nClassMembers <= nDims deletion inside MStep (KK.cpp:189),
                                 so there is no separate size threshold to honour.
    """
    X = np.ascontiguousarray(X, dtype=float)
    N = X.shape[0]
    # Normalise every dimension to [0,1] (kSv.dataMin/dataMax, KK.h:1105).  This is NOT cosmetic:
    # the noise cluster's density is taken to be 1 (LogP = -log w0, KK.cpp:297), which is only the
    # uniform density if the data occupies the unit hypercube.  Without it the constant -log w0
    # beats every broad-covariance Gaussian at random init and the first E-step sends the whole
    # set to cluster 0.
    _lo = X.min(0); _rng_ = X.max(0) - _lo
    X = (X - _lo) / np.where(_rng_ > 0, _rng_, 1.0)
    rng = np.random.default_rng(seed)
    npoint = NOISE_POINT if noise else 0
    if init_labels is not None:                    # seeded start: one CEM, no sweep
        init = np.asarray(init_labels, int)
        kk = _KK(X, max(int(init.max()) + 2, 2), penalty_mix, npoint, rng=rng)
        kk.cls = np.where(init >= 0, init + 1, 0).astype(int)
        kk.alive[:] = False
        kk.alive[np.unique(kk.cls)] = True
        kk.alive[0] = True
        kk.run_em_loop(enable_splits=splits)
        return _compact(kk.cls)
    lo = max(2, int(min_clusters))
    hi = max(lo, int(max_clusters))
    best_score, best_cls = np.inf, None
    for k0 in _sweep(lo, hi):
        for _st in range(max(1, int(n_starts))):
            # Independent stream per (K0, start).  Sharing one stream across the sweep lets
            # TrySplits' sub-CEMs consume draws, so every later init depends on how many splits
            # were probed earlier -- runs stop being reproducible per K0, and a draw that strands
            # a whole group in the noise cluster (an absorbing sink: TrySplits skips cluster 0,
            # KK.cpp:762 `cc = 1`, so points there are only recovered if some live Gaussian
            # reaches them) can win the sweep.
            kk = _KK(X, k0, penalty_mix, npoint,
                     rng=np.random.default_rng((int(seed), int(k0), int(_st))))
            sc = kk.cem(recurse=1 if splits else 0)
            if verbose:
                print(f"  K0={k0}: score {sc:.7g}, {int(kk.alive.sum())} alive")
            if sc < best_score:
                best_score, best_cls = sc, kk.cls.copy()
    return _compact(best_cls if best_cls is not None else np.zeros(N, int))


if __name__ == "__main__":
    import argparse
    try:
        from . import neuro_io as nio, fiber_pca as fpca, fiber_lib as fl
    except ImportError:
        import neuro_io as nio, fiber_pca as fpca, fiber_lib as fl
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("elec", type=int)
    ap.add_argument("--nsamp", type=int, default=32); ap.add_argument("--nchan", type=int, default=8)
    ap.add_argument("--dims", type=int, default=12)
    ap.add_argument("--method", default="standard")
    ap.add_argument("--no-global-basis", action="store_true")
    ap.add_argument("--max-clusters", type=int, default=200)
    ap.add_argument("--min-clusters", type=int, default=2)
    ap.add_argument("--penalty-mix", type=float, default=PENALTY_MIX)
    ap.add_argument("--n-starts", type=int, default=1)
    ap.add_argument("--no-splits", action="store_true")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--realign", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    mm, _ = nio.open_spk(a.base, a.elec, a.nsamp, a.nchan)
    spk = np.asarray(mm, float)
    basis = None if a.no_global_basis else fpca.read_cluster_basis(a.base, a.elec, a.method)
    F = fpca.cluster_features(spk, basis, realign=a.realign) if basis is not None else None
    if F is None:
        F = fpca.local_features(spk, a.dims, mask=fl.MASK_FULL, realign=a.realign)
        print(f"[klustakwik] features: local SVD-{a.dims}")
    else:
        print(f"[klustakwik] features: global basis '{a.method}'")
    lab = klustakwik(F, max_clusters=a.max_clusters, min_clusters=a.min_clusters,
                     penalty_mix=a.penalty_mix, n_starts=a.n_starts,
                     splits=not a.no_splits, seed=a.seed, verbose=True)
    out = a.out or f"{a.base}.clu.{a.elec}"
    nio.write_clu_file(out, (lab + 1).astype(np.int32))
    print(f"{len(np.unique(lab))} clusters (label 0 = noise) -> {out}")
