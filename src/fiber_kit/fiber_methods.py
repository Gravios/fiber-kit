#!/usr/bin/env python3
"""fiber_methods.py -- drift/identity methods validated on g5, wired as opt-in.

All OFF by default; each is enabled by a fiber-link flag (or env var) so the
baseline is untouched and every method can be A/B'd against it.

  dr_features            template-DR (PCA of mutual-centred templates): a far better
                         candidate / identity space than physical position
                         (g5: linking-decision 97%% vs <31%%; consecutive-chunk NN
                         candidate cosine 0.966 / 69%% clean vs 0.888 / 23%% for
                         position-NN -- it fixes the poor-candidate bottleneck at source).
  complete_footprint     counterfactual off-probe channel completion for edge units
                         whose footprint drifts off the array end: rank-1 temporal
                         shape x spatial field (1/r^2 by default; 1/r selectable).
                         Restores a truncated template's cosine match (g5: 0.83->0.91).
  estimate_drift_global  global collinearity drift -- Laplacian-regularised LS on
                         template-anchored same-neuron pairs, with an OPTIONAL
                         distance attenuation alpha(dist) (apparent drift = alpha(r)*D).
                         Alternative to the consecutive-accumulated estimate_drift,
                         which compounds error on sparse partitions.
"""
import numpy as np

try:
    from . import fiber_geometry as fg
except ImportError:
    import fiber_geometry as fg


# == template-DR identity / candidate space ===================================
def dr_basis(tmpl, k=10):
    """Fit a k-D PCA basis on mutual-centred, flattened templates.  Returns
    (mean (D,), components (k,D))."""
    tc = np.stack([fg.mutual_center(t) for t in tmpl]).reshape(len(tmpl), -1)
    mean = tc.mean(0)
    _, _, Vt = np.linalg.svd(tc - mean, full_matrices=False)
    return mean, Vt[:min(k, Vt.shape[0])]


def dr_features(tmpl, k=10, basis=None):
    """Mutual-centred templates (M,nsamp,nchan) -> (F (M,k) z-scored, basis).
    Pass `basis` to project onto a fixed basis instead of refitting."""
    if basis is None:
        basis = dr_basis(tmpl, k)
    mean, comp = basis
    tc = np.stack([fg.mutual_center(t) for t in tmpl]).reshape(len(tmpl), -1)
    F = (tc - mean) @ comp.T
    return F / (F.std(0) + 1e-9), basis


# == counterfactual off-probe footprint completion ===========================
def complete_footprint(tmpl, chpos, missing, *, field="inv_sq", partner=None):
    """Render unobserved off-probe channels of a truncated template.

    tmpl (nsamp,nchan); `missing` = indices of channels that drifted off the array.
    A spike footprint is rank-1 (one temporal shape * a spatial field), so the
    waveform on a channel that does not exist is shape(t) * field(position).  We
    fit the field (1/r^2 default; d^2=(y-y0)^2+r^2) to the OBSERVED channels and
    render the missing ones.  If `partner` (a same-neuron template that DID observe
    those positions, drift-aligned) is given, its real channels are used instead of
    the model (cross-view completion -- phase-correct, preferred); the field is the
    fallback for positions neither view saw.  Returns a completed copy.
    """
    T = np.asarray(tmpl, float).copy()
    miss = sorted(int(m) for m in missing)
    if partner is not None:                                   # cross-view: use real channels
        P = np.asarray(partner, float)
        for c in miss:
            if c < P.shape[1] and np.any(P[:, c]):
                T[:, c] = P[:, c]
        miss = [c for c in miss if not np.any(T[:, c])]       # only model what the partner lacked too
        if not miss:
            return T
    obs = np.array([c for c in range(T.shape[1]) if c not in set(miss)])
    if len(obs) < 3:
        return T
    To = T[:, obs]
    U, S, Vt = np.linalg.svd(To - To.mean(0), full_matrices=False)
    s = U[:, 0] * S[0]; w = Vt[0]
    if np.sum(s * To[:, int(np.argmax(np.abs(w)))]) < 0:
        s, w = -s, -w
    aw = np.abs(w); co = np.asarray(chpos, float)[obs]
    p = 2.0 if field == "inv_sq" else 1.0
    try:
        from scipy.optimize import least_squares
        sol = least_squares(
            lambda q: q[0] / ((co - q[1]) ** 2 + q[2] ** 2) ** (p / 2.0) - aw,
            [float(aw.max()) * 1e3, float(co[int(np.argmax(aw))]), 30.0],
            bounds=([0, co.min() - 80, 3.0], [1e12, co.max() + 80, 200.0])).x
    except Exception:
        return T
    A, y0, r = sol; sgn = np.sign(w[int(np.argmin(np.abs(co - y0)))] + 1e-9)
    cm = np.asarray(chpos, float)[np.array(miss)]
    wp = A / ((cm - y0) ** 2 + r ** 2) ** (p / 2.0)
    for i, c in enumerate(miss):
        T[:, c] = s * (sgn * wp[i])
    return T


def truncated_channels(tmpl, chpos, *, edge_frac=0.5):
    """Heuristic: which array-end channels carry enough amplitude that the footprint
    likely extends off-probe (candidates the linker should complete before matching).
    Returns the off-probe virtual-channel hint (which END, and how many)."""
    pp = tmpl.max(0) - tmpl.min(0)
    ends = []
    if pp[0] >= edge_frac * pp.max():
        ends.append("lo")
    if pp[-1] >= edge_frac * pp.max():
        ends.append("hi")
    return ends


# == global collinearity drift (+ optional distance attenuation) ==============
def _anchor_pairs(tmpl, chunk, chunks, *, cos_thr=0.95, gap=2):
    """Template-anchored same-neuron cross-chunk pairs: mutual cosine-NN >= cos_thr.
    Drift-independent (cosine on mutual-centred templates)."""
    tc = np.stack([fg.mutual_center(t) for t in tmpl]).reshape(len(tmpl), -1)
    tc = tc / (np.linalg.norm(tc, axis=1, keepdims=True) + 1e-9)
    anc = []
    for k in range(len(chunks) - 1):
        for g in range(1, gap + 1):
            if k + g >= len(chunks):
                break
            ai = np.flatnonzero(chunk == chunks[k]); bi = np.flatnonzero(chunk == chunks[k + g])
            if len(ai) < 2 or len(bi) < 2:
                continue
            C = tc[ai] @ tc[bi].T
            ja = C.argmax(1); jb = C.argmax(0)
            for u in range(len(ai)):
                j = int(ja[u])
                if C[u, j] >= cos_thr and int(jb[j]) == u:
                    anc.append((int(ai[u]), int(bi[j])))
    return list(set(anc))


def estimate_drift_global(y0, logA, w, chunk, chunks, tmpl, *, lam=3.0,
                          dist=None, n_dist_bins=4, cos_thr=0.95, gap=2, iters=12):
    """Depth drift D per chunk by maximising the collinearity of template-anchored
    same-neuron trajectories, with a discrete-Laplacian (D'') smoothness term:

        min_D  sum_anchors ((y_i - y_j) - alpha(dist)*(D[c_i]-D[c_j]))^2
               + lam * sum_c (D[c-1] - 2 D[c] + D[c+1])^2

    If `dist` (per-unit distance from the array) is given, the apparent-drift
    attenuation alpha(dist) is estimated jointly (alternating LS, gauged alpha=1 at
    the nearest bin) -- a rigid tissue drift registers fully on near-probe units and
    is attenuated for far ones.  Returns {chunk_id: D_um} (a drop-in for estimate_drift).
    On a session with too little drift to resolve alpha (drift ~ localisation noise),
    alpha collapses toward 1 and this reduces to the rigid global solve.
    """
    chunks = np.asarray(chunks); C = len(chunks)
    cidx = {int(c): i for i, c in enumerate(chunks)}
    anc = _anchor_pairs(tmpl, chunk, chunks, cos_thr=cos_thr, gap=gap)
    if len(anc) < C:
        from .fiber_link import estimate_drift  # fallback if too few anchors
        return estimate_drift(y0, logA, w, chunk, chunks)
    ca = np.array([cidx[int(chunk[a])] for a, _ in anc])
    cb = np.array([cidx[int(chunk[b])] for _, b in anc])
    dy = np.array([y0[a] - y0[b] for a, b in anc], float)
    if dist is not None:
        dp = np.array([0.5 * (dist[a] + dist[b]) for a, b in anc], float)
        ed = np.quantile(dp, np.linspace(0, 1, n_dist_bins + 1)); ed[0] -= 1; ed[-1] += 1
        binid = np.clip(np.digitize(dp, ed) - 1, 0, n_dist_bins - 1)
    else:
        binid = np.zeros(len(anc), int); n_dist_bins = 1
    alpha = np.ones(n_dist_bins)

    def solve_D(ak):
        rows = []; rhs = []
        for k in range(len(anc)):
            r = np.zeros(C); r[ca[k]] += ak[k]; r[cb[k]] -= ak[k]; rows.append(r); rhs.append(dy[k])
        for c in range(1, C - 1):
            r = np.zeros(C); r[c - 1] += 1; r[c] -= 2; r[c + 1] += 1
            rows.append(lam * r); rhs.append(0.0)
        g = np.zeros(C); g[0] = 1e6; rows.append(g); rhs.append(0.0)     # gauge D[0]=0
        return np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]

    D = solve_D(alpha[binid])
    if dist is not None:
        for _ in range(iters):
            dD = D[ca] - D[cb]; sig = np.abs(dD) > 1.0
            new = np.array([
                (np.sum(dy[(binid == b) & sig] * dD[(binid == b) & sig]) /
                 (np.sum(dD[(binid == b) & sig] ** 2) + 1e-9)) if ((binid == b) & sig).sum() >= 5
                else alpha[b] for b in range(n_dist_bins)])
            new = np.clip(new / (new[0] + 1e-9), 0.0, 2.0)               # gauge near-bin=1, bound
            if np.max(np.abs(new - alpha)) < 1e-3:
                alpha = new; break
            alpha = new; D = solve_D(alpha[binid])
    return {int(chunks[i]): float(D[i]) for i in range(C)}
