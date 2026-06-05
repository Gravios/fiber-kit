#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════════════════════
#  klustakwik.py  —  classic KlustaKwik-style CEM, NON-chunked, random init.
#
#  The whole session at once (no chunking): over-cluster from random seeds, run
#  classification-EM (hard assignment, full-covariance Gaussians), delete tiny
#  clusters, then BIC-merge adjacent clusters until no merge improves the score.
#  This is the baseline the chunking+fiber machinery is meant to beat on drifty
#  data — a drifting unit smears, so non-chunked CEM tends to either split it
#  across the drift or merge it with a neighbour.
#
#  Merge criterion (cheap, exact for fitted Gaussians; the (n)D terms cancel):
#      merge i,j  iff  n_ij*logdet(S_ij) - n_i*logdet(S_i) - n_j*logdet(S_j)  <  pen*nparams
#  with pen = log(N) (BIC) or 2 (AIC), nparams = D + D(D+1)/2.
# ════════════════════════════════════════════════════════════════════════════
import numpy as np


def _fit(X, lab, K, reg):
    N, D = X.shape
    mu = np.zeros((K, D)); logdet = np.full(K, np.inf); icov = np.zeros((K, D, D))
    cnt = np.zeros(K, int); ok = np.zeros(K, bool)
    for c in range(K):
        m = lab == c; n = int(m.sum()); cnt[c] = n
        if n < D + 2: continue
        Xc = X[m]; mu[c] = Xc.mean(0)
        cov = np.cov(Xc.T) + reg
        sgn, ld = np.linalg.slogdet(cov)
        if sgn <= 0: continue
        logdet[c] = ld; icov[c] = np.linalg.inv(cov); ok[c] = True
    return mu, icov, logdet, cnt, ok


def _assign(X, mu, icov, logdet, cnt, N):
    K = len(cnt); best = np.full(N, -np.inf); lab = np.zeros(N, int)
    lw = np.where(cnt > 0, np.log(np.maximum(cnt, 1) / N), -np.inf)
    for c in range(K):
        if cnt[c] < 1 or not np.isfinite(logdet[c]): continue
        d = X - mu[c]; maha = np.einsum('ij,jk,ik->i', d, icov[c], d)
        sc = lw[c] - 0.5 * logdet[c] - 0.5 * maha
        upd = sc > best; best[upd] = sc[upd]; lab[upd] = c
    return lab, best


def _cem(X, lab, reg, N, min_size, pen, npar, max_iter=40):
    prev = None
    for _ in range(max_iter):
        K = lab.max() + 1
        mu, icov, logdet, cnt, ok = _fit(X, lab, K, reg)
        keep = [c for c in range(K) if ok[c] and cnt[c] >= min_size]
        if not keep: break
        lab, best = _assign(X, mu[keep], icov[keep], logdet[keep], cnt[keep], N)
        cost = -best.sum() + 0.5 * pen * npar * len(keep)
        if prev is not None and abs(prev - cost) < 1e-3 * (abs(prev) + 1): break
        prev = cost
    return lab


def klustakwik(X, max_clusters=200, min_size=20, seed=42, reg_frac=1e-2,
               penalty='bic', max_iter=40, merge_rounds=12, verbose=False):
    """Returns per-point labels (0-based). Random-seed init, classification-EM,
    BIC-merge.  X: (N, D) features (e.g. PCA of masked waveforms)."""
    rng = np.random.default_rng(seed); N, D = X.shape
    reg = reg_frac * float(np.median(np.var(X, 0))) * np.eye(D)
    npar = D + D * (D + 1) / 2.0
    pen = np.log(N) if penalty == 'bic' else 2.0
    K0 = min(max_clusters, max(2, N // (min_size * 3)))
    seeds = X[rng.choice(N, K0, replace=False)]
    lab = np.zeros(N, int); best = np.full(N, np.inf)
    for c in range(K0):                                   # nearest random seed
        d = ((X - seeds[c]) ** 2).sum(1); upd = d < best; best[upd] = d[upd]; lab[upd] = c
    lab = _cem(X, lab, reg, N, min_size, pen, npar, max_iter)
    for mr in range(merge_rounds):
        K = lab.max() + 1
        mu, icov, logdet, cnt, ok = _fit(X, lab, K, reg)
        val = [c for c in range(K) if ok[c]]
        if len(val) < 2: break
        mus = np.array([mu[c] for c in val])
        cand = set()
        for a in range(len(val)):                         # each cluster's nearest neighbour
            dd = ((mus - mus[a]) ** 2).sum(1); dd[a] = np.inf
            b = int(dd.argmin()); cand.add((min(a, b), max(a, b)))
        merges = []
        for a, b in cand:
            ca, cb = val[a], val[b]; m = (lab == ca) | (lab == cb)
            cov = np.cov(X[m].T) + reg; sgn, ld = np.linalg.slogdet(cov)
            if sgn <= 0: continue
            lhs = int(m.sum()) * ld - cnt[ca] * logdet[ca] - cnt[cb] * logdet[cb]
            if lhs < pen * npar: merges.append((lhs - pen * npar, ca, cb))
        if not merges: break
        merges.sort(); used = set(); did = 0
        for _, ca, cb in merges:
            if ca in used or cb in used: continue
            lab[lab == cb] = ca; used.update((ca, cb)); did += 1
        uniq = np.unique(lab); remap = {u: i for i, u in enumerate(uniq)}
        lab = np.array([remap[u] for u in lab])
        lab = _cem(X, lab, reg, N, min_size, pen, npar, max_iter)
        if verbose: print(f"  merge round {mr}: merged {did}, K={lab.max()+1}")
        if did == 0: break
    return lab


if __name__ == "__main__":
    # CLI: cluster the whole session (no chunking) on PCA of masked .spkD waveforms.
    import argparse, os, fiber_lib as fl
    try:
        from . import neuro_io as nio
    except ImportError:
        import neuro_io as nio
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("elec", type=int)
    ap.add_argument("--nsamp", type=int, default=32); ap.add_argument("--nchan", type=int, default=8)
    ap.add_argument("--dims", type=int, default=12, help="PCA dims (try 3-6 for lowered-dim runs)")
    ap.add_argument("--max-clusters", type=int, default=200); ap.add_argument("--min-size", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--realign", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    mm, _ = nio.open_spk(a.base, a.elec, a.nsamp, a.nchan)
    spk = np.asarray(mm, float)
    w = (fl.realign(spk) if a.realign else spk)[:, fl.MASK_FULL, :].reshape(len(spk), -1)
    w = w - w.mean(0); U, S, _ = np.linalg.svd(w, full_matrices=False); F = U[:, :a.dims] * S[:a.dims]
    lab = klustakwik(F, max_clusters=a.max_clusters, min_size=a.min_size, seed=a.seed, verbose=True)
    clu = (lab + 1).astype(np.int32)
    out = a.out or f"{a.base}.clu.{a.elec}"
    nio.write_clu_file(out, clu)
    print(f"{len(np.unique(lab))} clusters -> {out}")
