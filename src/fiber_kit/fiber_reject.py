# ════════════════════════════════════════════════════════════════════════════
#  fiber_reject.py — outlier spike reassignment as a pipeline step.
#
#  An over-split fragment carries a few spikes that belong to a neighbouring unit (or to
#  noise).  Those contaminants drag every per-cluster statistic — the localized position
#  most visibly (see _pca_profile's median fix), but also templates and signatures.  This
#  step flags, per cluster, the spikes whose feature vector is a robust outlier of the
#  cluster, then EITHER reassigns each to the cluster it best agrees with OR sends it to the
#  noise cluster — and writes a new .clu stage.
#
#  Features = the .pca scores (nCh*nComp), i.e. exactly the standard .fet the sort used (read
#  the basis with fiber_pca.read_pca and project the .spk; equals the on-disk .fet scores).
#  Distance is a robust diagonal Mahalanobis: per-cluster median centre and 1.4826*MAD scale,
#  so the gate is a chi^2 quantile on the standardized squared distance.
#
#  PROTOTYPE: single pass, diagonal (per-dim) scale, amplitude/standard feature space by
#  default.  Pass --feat-method stderiv to reassign in the clustering (stderiv) feature space.
# ════════════════════════════════════════════════════════════════════════════
import argparse
import numpy as np
from scipy.stats import chi2

try:
    from . import neuro_io as nio, session_yaml as sy, fiber_pca as fpca
except ImportError:
    import neuro_io as nio, session_yaml as sy, fiber_pca as fpca


def spike_features(spk, basis):
    """Per-spike feature vectors = the .pca scores (nCh*nComp) — the sort's standard .fet."""
    win = fpca.extract_windows(np.asarray(spk, np.float64), int(basis["recShift"]), int(basis["data2use"]))
    return fpca.project(win, basis)                          # (N, nCh*nComp), channel-major


def _models(labels, fet, ids, min_n, shrink, support_fraction=0.75, robust=True):
    """Per-cluster centre + inverse covariance.  By default a ROBUST (Minimum Covariance
    Determinant) estimate: a plain covariance is inflated by the very contaminants we want to
    flag (masking — validated: ~9% recovery with np.cov vs ~95% with MCD), because the outliers
    enter the scatter.  MCD fits the covariance of the clean core, so contaminants fall outside
    the chi^2 gate.  Falls back to a median-centred empirical covariance when MCD is unavailable
    or the cluster is too small (< 2*D); covariance is shrunk toward its diagonal + a ridge for
    conditioning."""
    D = fet.shape[1]
    MCD = None
    if robust:
        try:
            from sklearn.covariance import MinCovDet
            MCD = MinCovDet
        except ImportError:
            MCD = None
    cen, prec = {}, {}
    for c in ids:
        X = fet[labels == c].astype(np.float64)
        if len(X) < min_n:
            continue
        mu = C = None
        if MCD is not None and len(X) >= 2 * D:
            try:
                m = MCD(support_fraction=support_fraction, random_state=0).fit(X)
                mu, C = m.location_, m.covariance_
            except Exception:
                mu = C = None
        if mu is None:                                       # fallback: median centre + empirical cov
            mu = np.median(X, 0); C = np.cov(X - mu, rowvar=False)
            if C.ndim == 0:
                C = np.array([[float(C)]])
        C = (1.0 - shrink) * C + shrink * np.diag(np.diag(C)) + 1e-6 * np.eye(D)
        cen[c] = mu
        prec[c] = np.linalg.inv(C)
    return cen, prec


def _maha2(X, mu, P):
    """Squared Mahalanobis distance of each row of X to (mu, precision P)."""
    Xc = np.asarray(X, np.float64) - mu
    return np.einsum("ij,jk,ik->i", Xc, P, Xc)


def reject(labels, fet, *, noise_id=1, chi2_p=0.9999, reassign=True, min_n=50, shrink=0.1,
           support_fraction=0.75, robust=True):
    """Flag per-cluster feature outliers (robust Mahalanobis > chi^2(D, chi2_p)) and reassign them.
    Returns (new_labels, stats).  If reassign, each outlier moves to the cluster whose model it is
    closest to, provided that cluster is different and within the gate; else it goes to noise_id.
    Clusters with < min_n spikes have no model — neither pruned nor reassignment targets."""
    labels = np.asarray(labels).copy()
    fet = np.asarray(fet, np.float64)
    D = fet.shape[1]
    t2 = float(chi2.ppf(chi2_p, D))
    ids = [int(c) for c in np.unique(labels) if c > 1]
    cen, prec = _models(labels, fet, ids, min_n, shrink, support_fraction, robust)
    cids = list(cen)
    st = dict(n_out=0, n_re=0, n_noise=0, t2=t2, D=D, n_clusters=len(cids))
    if not cids:
        return labels, st
    out_idx, out_src = [], []
    for c in cids:
        m = np.flatnonzero(labels == c)
        d = _maha2(fet[m], cen[c], prec[c])
        o = m[d > t2]
        out_idx.append(o); out_src.append(np.full(len(o), c))
    out_idx = np.concatenate(out_idx) if out_idx else np.array([], int)
    out_src = np.concatenate(out_src) if out_src else np.array([], int)
    st["n_out"] = int(len(out_idx))
    if len(out_idx) == 0:
        return labels, st
    if reassign:                                            # nearest cluster model for each outlier
        best_d = np.full(len(out_idx), np.inf); best_c = np.full(len(out_idx), -1)
        Xo = fet[out_idx]
        for c in cids:
            d = _maha2(Xo, cen[c], prec[c])
            better = d < best_d
            best_d[better] = d[better]; best_c[better] = c
        move = (best_c != out_src) & (best_d <= t2)
        labels[out_idx[move]] = best_c[move]
        labels[out_idx[~move]] = noise_id
        st["n_re"] = int(move.sum()); st["n_noise"] = int((~move).sum())
    else:
        labels[out_idx] = noise_id; st["n_noise"] = int(len(out_idx))
    return labels, st


def main():
    ap = argparse.ArgumentParser(
        description="Reassign per-cluster outlier spikes to a better-fitting cluster or to noise.")
    sy.add_session_args(ap, nchan=False, sr=False, peak=True)
    ap.add_argument("--clu-method", default="stderiv", help="variant of the input .clu (before the group)")
    ap.add_argument("--clu-stage", default="refine", help="stage of the input .clu (after the group)")
    ap.add_argument("--feat-method", default="standard",
                    help=".pca/.spk variant for the feature space (standard=amplitude, stderiv=clustering)")
    ap.add_argument("--spk", default=None, help="explicit .spk path (else <base>.spk.<feat-method>.<group>)")
    ap.add_argument("--out-stage", default=None, help="output .clu stage (default: <clu-stage>_reject)")
    ap.add_argument("--noise-id", type=int, default=1, help="cluster id outliers fall to when no cluster fits")
    ap.add_argument("--chi2-p", type=float, default=0.9999, help="per-spike outlier gate (chi^2 quantile)")
    ap.add_argument("--shrink", type=float, default=0.1,
                    help="covariance shrinkage toward diagonal (0=full cov, 1=diagonal) for conditioning")
    ap.add_argument("--support-fraction", type=float, default=0.75,
                    help="MCD clean-core fraction (1 - max contamination the robust cov tolerates)")
    ap.add_argument("--no-robust", action="store_true",
                    help="use a plain (non-robust) covariance instead of MCD (faster; masks contaminants)")
    ap.add_argument("--no-reassign", action="store_true",
                    help="send every outlier to noise (skip cross-cluster reassignment)")
    ap.add_argument("--min-n", type=int, default=50, help="min spikes for a cluster to get a robust model")
    a = ap.parse_args()

    cfg = sy.resolve_session_params(a.session, a.group, channels=a.channels, ntotal=a.ntotal, nsamp=a.nsamp)
    base, elec = cfg["base"], a.group
    nsamp = cfg["nsamp"]; nch = len(cfg["channels"])
    if nsamp is None:
        raise SystemExit("[reject] nSamples not in <session>.yaml; pass --nsamp")

    res = nio.read_res(base, elec)
    _, labels = nio.read_clu_at(base, elec, variant=a.clu_method, tag=a.clu_stage, n_spikes=len(res))
    spk_path = a.spk or nio.session_path(base, "spk", elec, variant=a.feat_method)
    spk = nio.open_spk_file(spk_path, nsamp, nch)
    basis = fpca.read_pca(base, elec, prefer=[a.feat_method, ""])   # the eigenvectors the .fet was made with
    fet = spike_features(spk, basis)
    n = min(len(labels), len(fet)); labels, fet = labels[:n], fet[:n]

    new, st = reject(labels, fet, noise_id=a.noise_id, chi2_p=a.chi2_p,
                     reassign=not a.no_reassign, min_n=a.min_n, shrink=a.shrink,
                     support_fraction=a.support_fraction, robust=not a.no_robust)
    out_stage = a.out_stage or (f"{a.clu_stage}_reject" if a.clu_stage else "reject")
    out = nio.write_clu(base, elec, new, variant=a.clu_method, tag=out_stage)
    print(f"[reject] {st['n_out']} outliers over {st['n_clusters']} clusters "
          f"(D={st['D']}, chi2_p={a.chi2_p}, t^2={st['t2']:.1f}): "
          f"{st['n_re']} reassigned, {st['n_noise']} -> noise(id {a.noise_id}); wrote {out}")


if __name__ == "__main__":
    main()
