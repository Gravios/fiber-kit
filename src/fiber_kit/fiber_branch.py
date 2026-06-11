# ════════════════════════════════════════════════════════════════════════════
#  fiber_branch.py — detect units whose spikes do NOT lie on a single fiber.
#
#  The fiber d(r) (fiber_geometry.fiber_curve) models a unit's waveform variation along ONE
#  coordinate: energy.  A unit is well-described by a single fiber iff, after subtracting d(r),
#  the per-spike residual is unimodal noise.  A second, *energy-independent* mode in the residual
#  means the neuron emits two distinct waveform classes at matched energy -- a fiber BRANCH the
#  single curve averages away.  This is the one place the kernel split's sensitivity (a liability
#  for grouping, see fiber_intrachunk gate) is an asset: we WANT sub-structure orthogonal to the
#  energy axis the fiber already explains.
#
#  Detection gates (a candidate branch must clear ALL):
#    - residual PC1 is bimodal      : 2-Gaussian BIC beats 1-Gaussian by >= dbic_min,
#                                     separation >= sep_min, minority weight >= minw_min;
#    - energy-INDEPENDENT           : |corr(class, energy)| <= ecorr_max  (else it is just the
#                                     energy ladder the fiber already captures);
#    - depth-COHERENT               : the two modes share location, |Δdepth| <= depth_um  (else
#                                     it is two neurons mis-merged -- a curation issue, not a branch);
#    - kernel-confirmed             : kcov(mode_A, mode_B) >= kcov_mult x within-mode null.
#
#  IMPORTANT — operate on RAW (standard) waveforms via the real fiber_curve fit, never a crude
#  per-bin rank-1 proxy: validated on g5, the proxy reports ~22% branched units but the genuine
#  d(r) fit reports ~1% (the proxy under-fits the curve and manufactures bimodal residuals).
#  So on a well-behaved single-shank session expect very few branches; the value is flagging the
#  rare genuine multi-class unit and confirming the fiber model is otherwise adequate.
# ════════════════════════════════════════════════════════════════════════════
import argparse
import numpy as np

try:
    from . import fiber_geometry as fg, fiber_lib as fl, fiber_intrachunk as fic, neuro_io as nio, session_yaml as sy
except ImportError:
    import fiber_geometry as fg, fiber_lib as fl, fiber_intrachunk as fic, neuro_io as nio, session_yaml as sy

try:
    from sklearn.mixture import GaussianMixture as _GMM
except ImportError:
    _GMM = None


def energy_depth(spk_unit):
    """Per-spike energy-weighted channel centroid (raw amplitudes), in CHANNEL units.
    spk_unit: (n, nsamp, nchan).  Multiply by the probe pitch for microns."""
    e = (np.asarray(spk_unit, float) ** 2).sum(1)            # (n, nchan)
    return (e * np.arange(e.shape[1])).sum(1) / (e.sum(1) + 1e-9)


def fiber_residuals(spk_unit, *, nq=fg.DEFAULT_NQ):
    """Realign a unit's raw waveforms, fit the real d(r) curve, and return the per-spike
    off-fiber residual and energy.  spk_unit: (n, nsamp, nchan) RAW waveforms."""
    W = fl.realign(np.asarray(spk_unit, float)).reshape(len(spk_unit), -1)
    r = np.linalg.norm(W, axis=1)
    cur = fg.fiber_curve(W, r, nq)                            # (nq, P) energy-quantile mean templates
    edges = np.quantile(r, np.linspace(0, 1, nq + 1)); edges[0] = -np.inf; edges[-1] = np.inf
    b = np.clip(np.digitize(r, edges) - 1, 0, nq - 1)
    return W - cur[b], r


def detect_branch(resid, energy, depth, *, pitch=20.0, dbic_min=20.0, sep_min=2.2,
                  minw_min=0.18, ecorr_max=0.25, depth_um=8.0, kcov_mult=4.0):
    """Decide whether a unit's off-fiber residual hides a genuine second waveform class.
    Returns dict(branched, labels, n_branch, dbic, sep, minw, energy_corr, depth_um, kcov, kcov_null)."""
    if _GMM is None:
        raise ImportError("fiber_branch needs scikit-learn (GaussianMixture)")
    Rc = resid - resid.mean(0)
    pc = Rc @ np.linalg.svd(Rc, full_matrices=False)[2][0]
    z = ((pc - pc.mean()) / (pc.std() + 1e-9)).reshape(-1, 1)
    b1 = _GMM(1).fit(z).bic(z)
    g = _GMM(2, n_init=2, random_state=0).fit(z); b2 = g.bic(z)
    mu = g.means_.ravel(); sd = np.sqrt(g.covariances_.ravel()); w = g.weights_; lab = g.predict(z)
    sep = abs(mu[0] - mu[1]) / np.sqrt((sd ** 2).mean() + 1e-9)
    ec = abs(np.corrcoef(lab, energy)[0, 1]) if len(set(lab)) > 1 else 1.0
    dd = abs(depth[lab == 0].mean() - depth[lab == 1].mean()) * pitch if len(set(lab)) > 1 else 0.0
    A, B = resid[lab == 0], resid[lab == 1]
    kcov = kn = 0.0
    if len(A) >= 16 and len(B) >= 16:
        kcov = fic.kernel_twosample(A[:200], B[:200], "kcov")
        h = len(A) // 2; kn = fic.kernel_twosample(A[:h], A[h:2 * h], "kcov")
    branched = bool((b1 - b2) > dbic_min and sep > sep_min and w.min() > minw_min
                    and ec < ecorr_max and dd < depth_um and kcov > kcov_mult * (kn + 1e-9))
    return dict(branched=branched, labels=lab, n_branch=2 if branched else 1,
                dbic=float(b1 - b2), sep=float(sep), minw=float(w.min()),
                energy_corr=float(ec), depth_um=float(dd), kcov=float(kcov), kcov_null=float(kn))


def branch_units(spk_raw, members, *, min_n=400, nq=fg.DEFAULT_NQ, pitch=20.0, **gates):
    """Scan units (members = list of per-unit spike-row arrays) for fiber branches.
    Yields (unit_index, report dict) for units with >= min_n spikes."""
    for u, rows in enumerate(members):
        if len(rows) < min_n:
            continue
        sp = spk_raw[rows]
        resid, energy = fiber_residuals(sp, nq=nq)
        rep = detect_branch(resid, energy, energy_depth(sp), pitch=pitch, **gates)
        rep["unit"] = u; rep["n"] = len(rows)
        yield u, rep


def main():
    ap = argparse.ArgumentParser(description="Flag units whose spikes branch off the single fiber "
                                             "d(r) (a second, energy-independent, depth-coherent waveform class).")
    ap.add_argument("session"); ap.add_argument("group", type=int)
    ap.add_argument("--clu", default=None, help="unit-defining .clu (e.g. the .intrachunk or .linked clu); "
                                                "default resolves the canonical sort")
    ap.add_argument("--clu-method", default="stderiv"); ap.add_argument("--clu-stage", default="refine.linked")
    ap.add_argument("--min-n", type=int, default=400, help="skip units with fewer spikes (branch test needs samples)")
    ap.add_argument("--pitch", type=float, default=20.0, help="probe site pitch (um) for the depth-coherence gate")
    ap.add_argument("--depth-um", type=float, default=8.0); ap.add_argument("--sep-min", type=float, default=2.2)
    ap.add_argument("--ecorr-max", type=float, default=0.25); ap.add_argument("--dbic-min", type=float, default=20.0)
    ap.add_argument("--out", default=None, help="write a per-unit branch report .npz")
    a = ap.parse_args()

    cfg = sy.resolve_session_params(a.session, a.group, require=("ntotal",))
    base, elec = cfg["base"], a.group
    nsamp = int(cfg["nsamp"]); nch = int(cfg["nchan"])
    spk, r = nio.open_spk_raw(base, elec, nsamp, nch)         # RAW waveforms (refuses stderiv)
    if a.clu:
        _, ids = nio.read_clu_file(a.clu)
    else:
        _, ids = nio.read_clu_at(base, elec, variant=a.clu_method, tag=a.clu_stage)
    order = np.argsort(ids, kind="stable"); cs = ids[order]
    uq, st = np.unique(cs, return_index=True); en = np.r_[st[1:], len(cs)]
    members = [order[st[k]:en[k]] for k, c in enumerate(uq) if int(c) > 1]   # 0/1 reserved
    uids = [int(c) for c in uq if int(c) > 1]

    hits = []
    for i, rep in ((i, rep) for i, (_, rep) in enumerate(branch_units(
            spk, members, min_n=a.min_n, pitch=a.pitch, depth_um=a.depth_um,
            sep_min=a.sep_min, ecorr_max=a.ecorr_max, dbic_min=a.dbic_min))):
        if rep["branched"]:
            hits.append((uids[i], rep))
    scanned = sum(len(m) >= a.min_n for m in members)
    print(f"[branch] scanned {scanned} units (>= {a.min_n} spikes) from {r.path} -> {len(hits)} branched")
    for cid, rep in sorted(hits, key=lambda x: -x[1]["dbic"]):
        print(f"  unit {cid:5d} n={rep['n']:5d}  dBIC {rep['dbic']:6.0f}  sep {rep['sep']:.1f}  "
              f"Δdepth {rep['depth_um']:.1f}um  energy-corr {rep['energy_corr']:.2f}  "
              f"kcov {rep['kcov']:.3f} vs null {rep['kcov_null']:.3f}")
    if a.out:
        np.savez(a.out, units=np.array([c for c, _ in hits]),
                 dbic=np.array([r["dbic"] for _, r in hits]),
                 sep=np.array([r["sep"] for _, r in hits]),
                 depth_um=np.array([r["depth_um"] for _, r in hits]))
        print(f"[branch] wrote {a.out}")


if __name__ == "__main__":
    main()
