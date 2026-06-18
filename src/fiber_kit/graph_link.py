#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════════════════════
#  graph_link.py — global, graph-based merge for the over-clustered set.
#
#  Replaces the per-pair co-gate stack (cogated_links: mutual-NN position +
#  pos_thr + off_thr + cosine) with a GLOBAL formulation that is robust to the
#  per-pair gate miscalibration which leaves high-cosine blocks unmerged on
#  sessions whose geometry/drift differ from the calibration set.
#
#  Four composable pieces, each validated on real g5 (180-210 min window, 282
#  over-split fragments, whitened K=24):
#
#    1. global_ev / ev_agglomerate  — explained variance vs cluster count.
#       Cluster EV is a rank-r PCA RECONSTRUCTION (mean + top-r PCs), NOT the
#       fiber residual (that is a direction-vs-energy model whose residual is
#       not a valid variance term — it goes negative).  rank>=1 credits the
#       within-cluster manifold (drift / adaptation / broad-unit post-peak:
#       ~+0.08 EV over the centroid on g5) so the objective does not over-split
#       curved cells.  Operating point = argmax EV(n) - lam*n.
#
#    2. discriminative_affinity     — edge weight from a feature vector that
#       concatenates the masked waveform AND the discriminative axes (offset /
#       position / direction).  Folds pos_thr/off_thr into edge WEIGHTS instead
#       of hard vetoes, so similar-but-different units do not form the strong
#       bridge edges that chain a naive cosine graph (cosine>=0.85 connected
#       components fuse 206/282 fragments into one blob).
#
#    3. spectral_partition          — normalized graph Laplacian; k from the
#       eigengap (k~=25 on g5, which AGREES with the EV-lam*n knee of 20-30 —
#       two independent global criteria converge).  Cuts the weakest bottleneck
#       rather than a hard edge threshold, so it does not chain through cliques.
#
#    4. coherent_path_link (A*)     — least-incoherence path on the kNN manifold
#       graph in a METRIC feature space (position/direction/offset, whitened),
#       with an admissible Euclidean-to-target heuristic so A* is provably
#       optimal (cosine is NOT a metric and would break admissibility — hence
#       the metric space).  Links i,j iff their optimal path stays under a
#       coherence budget: keeps the transitivity that heals true blocks while
#       rejecting chains that must cross a manifold discontinuity.
#
#  Also: splittability_bic — GMM 1-vs-2 BIC gain in the top stderiv PCs, the
#  KlustaKwik-faithful contamination/quality flag (AUC ~0.86-0.89 on g5).
# ════════════════════════════════════════════════════════════════════════════
import heapq
import numpy as np


# ── 1. global explained variance vs cluster count ───────────────────────────
def _rank_r_resid(X, rank):
    """Residual SSE of a rank-`rank` PCA reconstruction (mean + top-r PCs) of X."""
    if len(X) == 0:
        return 0.0
    m = X.mean(0); Xz = X - m
    if rank <= 0 or len(X) <= rank:
        return float((Xz ** 2).sum())
    Vt = np.linalg.svd(Xz, full_matrices=False)[2][:rank]
    return float(((Xz - (Xz @ Vt.T) @ Vt) ** 2).sum())


def global_ev(features_by_cluster, total_var, rank=1):
    """Global explained variance of a partition: 1 - sum_cluster resid_rank / TV.

    features_by_cluster: iterable of (n_c, d) arrays (one per cluster, in the
    SAME whitened space `total_var` was computed in).  rank>=1 credits the
    within-cluster manifold; rank=0 is the centroid model."""
    r = sum(_rank_r_resid(np.asarray(X, float), rank) for X in features_by_cluster)
    return 1.0 - r / (total_var + 1e-12)


def ev_agglomerate(features_by_cluster, lam, rank=1, metric="cosine", n_grid=None):
    """Greedy agglomeration to argmax EV(n) - lam*n.

    Merges clusters in the order given by average-linkage on their templates
    (cheap, fixed order), recording the EV(n) curve, and returns the partition
    at the EV-lam*n optimum.

    Returns dict(labels (n_clusters0,), n_opt, curve [(n, ev), ...], best_ev).
    `labels[i]` is the merged-group id of input cluster i at the optimum."""
    from scipy.cluster.hierarchy import linkage, fcluster
    feats = [np.asarray(X, float) for X in features_by_cluster]
    K = len(feats)
    allX = np.concatenate(feats, 0); gm = allX.mean(0)
    TV = float(((allX - gm) ** 2).sum())
    tmpl = np.array([X.mean(0) for X in feats])
    Z = linkage(tmpl, method="average", metric=metric)
    ns = sorted({int(n) for n in np.unique(
        np.round(np.geomspace(1, K, num=min(K, 40 if n_grid is None else n_grid))).astype(int))}, reverse=True)
    curve = []; best = (-np.inf, K, None)
    for n in ns:
        lab = fcluster(Z, n, "maxclust")
        groups = {}
        for ci, g in enumerate(lab):
            groups.setdefault(int(g), []).append(ci)
        ev = global_ev([np.concatenate([feats[c] for c in mem], 0) for mem in groups.values()], TV, rank)
        nn = len(groups); curve.append((nn, ev))
        score = ev - lam * nn
        if score > best[0]:
            best = (score, nn, lab.copy())
    return dict(labels=best[2], n_opt=best[1], best_ev=global_ev(
        [np.concatenate([feats[c] for c in
         [i for i in range(K) if best[2][i] == g]], 0) for g in np.unique(best[2])], TV, rank),
        curve=sorted(curve))


# ── 2. discriminative affinity ──────────────────────────────────────────────
def discriminative_affinity(feature_vectors, sigma=None, knn=None, self_tuning=True):
    """Gaussian affinity over a discriminative feature vector per cluster.

    feature_vectors: (K, d) — e.g. concat(masked-waveform, offset, position,
    direction), each block pre-scaled to comparable variance.

    self_tuning=True (default) uses the Zelnik-Manor & Perona LOCAL scale:
    sigma_i = distance to the `knn`-th neighbour (knn=7 if unset), and
    A_ij = exp(-d2_ij / (sigma_i sigma_j)).  A single global sigma (the median
    pairwise distance) over-connects a densely-tiled over-split cloud and
    collapses the eigengap; the local scale adapts to varying density so the
    spectral cut tracks the real bottlenecks.  Pass self_tuning=False and a
    `sigma` for a fixed global scale.  If `knn` is set the affinity is sparsified
    to the knn nearest neighbours (symmetrised)."""
    F = np.asarray(feature_vectors, float); n = len(F)
    d2 = ((F[:, None, :] - F[None, :, :]) ** 2).sum(-1)
    if self_tuning:
        kk = knn if knn is not None else min(7, n - 1)
        sd = np.sort(np.sqrt(d2), axis=1)[:, kk]                 # dist to kk-th neighbour
        denom = sd[:, None] * sd[None, :] + 1e-12
        A = np.exp(-d2 / denom)
    else:
        if sigma is None:
            iu = np.triu_indices(n, 1); sigma = np.sqrt(np.median(d2[iu]) + 1e-12) + 1e-12
        A = np.exp(-d2 / (2.0 * sigma ** 2))
    np.fill_diagonal(A, 0.0)
    if knn is not None and knn < n - 1:
        keep = np.zeros_like(A, bool)
        for i in range(n):
            keep[i, np.argsort(A[i])[::-1][:knn]] = True
        A = A * (keep | keep.T)
    return A


# ── 3. spectral partition with eigengap k ───────────────────────────────────
def spectral_partition(A, k=None, kmax=60, random_state=0):
    """Normalized-Laplacian spectral clustering.  k from the largest eigengap if
    None.  Returns dict(labels, k, eigvals)."""
    A = np.asarray(A, float); n = len(A)
    d = A.sum(1); Dm = np.diag(1.0 / np.sqrt(d + 1e-12))
    L = np.eye(n) - Dm @ A @ Dm
    ev, evec = np.linalg.eigh(L); order = np.argsort(ev); ev = ev[order]; evec = evec[:, order]
    if k is None:
        m = min(kmax, n - 1)
        k = int(np.argmax(np.diff(ev[:m + 1]))) + 1
        k = max(1, k)
    if k <= 1:
        return dict(labels=np.zeros(n, int), k=1, eigvals=ev)
    from sklearn.cluster import KMeans
    U = evec[:, :k]
    U = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)   # row-normalize (Ng-Jordan-Weiss)
    lab = KMeans(k, n_init=10, random_state=random_state).fit(U).labels_
    return dict(labels=lab, k=k, eigvals=ev)


# ── 4. A* least-incoherence coherent-path linkage ───────────────────────────
def _knn_graph(coords, knn):
    """Symmetric kNN graph: adjacency dict node -> list of (neighbour, euclid dist)."""
    coords = np.asarray(coords, float); n = len(coords)
    adj = {i: {} for i in range(n)}
    for i in range(n):
        d = np.sqrt(((coords - coords[i]) ** 2).sum(1)); nn = np.argsort(d)[1:knn + 1]
        for j in nn:
            w = float(d[j]); adj[i][int(j)] = w; adj[int(j)][i] = min(adj[int(j)].get(i, w), w)
    return adj


def astar_path_cost(adj, coords, src, dst):
    """A* shortest path cost src->dst on the metric kNN graph `adj`, heuristic =
    Euclidean(node, dst) (admissible: Euclidean <= geodesic).  inf if no path."""
    coords = np.asarray(coords, float)
    h = lambda u: float(np.sqrt(((coords[u] - coords[dst]) ** 2).sum()))
    g = {src: 0.0}; pq = [(h(src), 0.0, src)]; seen = set()
    while pq:
        f, gu, u = heapq.heappop(pq)
        if u == dst:
            return gu
        if u in seen:
            continue
        seen.add(u)
        for v, w in adj[u].items():
            ng = gu + w
            if ng < g.get(v, np.inf):
                g[v] = ng; heapq.heappush(pq, (ng + h(v), ng, v))
    return np.inf


def coherent_path_link(coords, budget, knn=8, candidate_pairs=None):
    """Link clusters whose A* least-incoherence path on the metric kNN manifold
    graph stays within `budget`.  coords: (K, d) metric features (whitened
    position/direction/offset).  Keeps transitivity (a chain of coherent steps
    links the ends) while rejecting a chain forced to cross a manifold gap (a
    single large edge inflates the path cost above budget).

    ── CAVEAT (measured on g5) ──────────────────────────────────────────────
    This is GAP-DEPENDENT, like its sibling laplacian_link.  When units are
    separated manifolds it cleanly stops at the gap; when the over-split
    fragments densely TILE the feature space (no gap), geodesics are short
    everywhere and a loose budget chains the cloud into one blob — exactly as a
    naive connected-components does.  Set `budget` to a small multiple of the
    LOCAL scale (e.g. 1.5-2x the median kNN edge length) so only genuinely
    adjacent fragments link, and prefer spectral_partition (which cuts the
    weakest bottleneck and does not need a gap) when the cloud is dense.

    candidate_pairs: optional iterable of (i,j) to test; defaults to all pairs
    sharing a kNN edge of either endpoint (cheap).  Returns list of (i,j)."""
    adj = _knn_graph(coords, knn); K = len(coords)
    if candidate_pairs is None:
        candidate_pairs = set()
        for i in range(K):
            for j in adj[i]:
                candidate_pairs.add((min(i, j), max(i, j)))
    links = []
    for i, j in candidate_pairs:
        if astar_path_cost(adj, coords, i, j) <= budget:
            links.append((int(i), int(j)))
    return links


def median_knn_edge(coords, knn=8):
    """Median nearest-neighbour edge length — the LOCAL scale for sizing a
    coherent_path_link budget (try budget = 1.5-2x this)."""
    coords = np.asarray(coords, float)
    nn1 = []
    for i in range(len(coords)):
        d = np.sqrt(((coords - coords[i]) ** 2).sum(1)); nn1.append(np.sort(d)[1])
    return float(np.median(nn1))


def link_groups(links, K):
    """Connected components of an edge list over K nodes -> list of index groups."""
    parent = list(range(K))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for i, j in links:
        parent[find(i)] = find(j)
    comp = {}
    for i in range(K):
        comp.setdefault(find(i), []).append(i)
    return list(comp.values())


# ── splittability (contamination / re-split flag) ───────────────────────────
def splittability_bic(X, n_pcs=15, max_comp=2):
    """Per-spike BIC gain of a `max_comp`-vs-1 component full-covariance Gaussian
    fit in the top-`n_pcs` PCs of X.  >0 favours splitting (contaminated).

    KlustaKwik-faithful (KlustaKwik is a penalized Gaussian mixture); validated
    AUC ~0.86-0.89 on g5 with n_pcs~=15.  Use the full fet dimensionality the
    clusterer uses; >5 PCs matters for the subtler (cos>0.95) contamination."""
    from sklearn.mixture import GaussianMixture
    X = np.asarray(X, float); n = len(X)
    if n < 30:
        return 0.0
    Xc = X - X.mean(0)
    Vt = np.linalg.svd(Xc, full_matrices=False)[2][:n_pcs]
    F = Xc @ Vt.T
    try:
        b1 = GaussianMixture(1, covariance_type="full", reg_covar=1e-4, random_state=0).fit(F).bic(F)
        b2 = GaussianMixture(max_comp, covariance_type="full", reg_covar=1e-4,
                             n_init=2, random_state=0).fit(F).bic(F)
    except Exception:
        return 0.0
    return float((b1 - b2) / n)


if __name__ == "__main__":          # self-test on synthetic data (no real data needed)
    rng = np.random.default_rng(0)
    # three well-separated units, each over-split into 3 fragments
    centers = rng.standard_normal((3, 12)) * 5.0
    feats = []; truth = []
    for u in range(3):
        for _ in range(3):
            feats.append(centers[u] + rng.standard_normal((60, 12))); truth.append(u)
    truth = np.array(truth)
    out = ev_agglomerate(feats, lam=0.002, rank=1)
    print("ev_agglomerate -> n_opt =", out["n_opt"], "(expect ~3)  EV =", round(out["best_ev"], 3))
    tmpl = np.array([X.mean(0) for X in feats])
    A = discriminative_affinity(tmpl, knn=4)
    sp = spectral_partition(A)
    print("spectral_partition -> k =", sp["k"], "(expect ~3)")
    links = coherent_path_link(tmpl, budget=8.0, knn=4)
    print("coherent_path_link -> groups =", len(link_groups(links, len(tmpl))), "(expect ~3)")
    cont = np.vstack([feats[0], feats[3]])      # two different units merged
    print("splittability_bic(contaminated) =", round(splittability_bic(cont, n_pcs=8), 3),
          " (clean) =", round(splittability_bic(feats[0], n_pcs=8), 3))
