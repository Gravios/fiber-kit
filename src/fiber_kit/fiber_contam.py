#  fiber_contam.py — contamination QC for an existing sort (no re-clustering).
#
#  Refractory/ISI checks miss two-cell contamination whenever the two cells
#  never fire close together (well-separated trains, or one early / one late).
#  This pass reads contamination off the WAVEFORM distribution instead, using
#  the per-channel temporal-derivative structure of the (already first-
#  difference) stderiv spikes.  Validated on g5 group 5 first-36-min curated
#  block; the numbers in the comments are from that block.
#
#    detect      : the top within-cluster SVD component of the windowed multi-
#                  channel stderiv AND of its temporal SECOND derivative
#                  (diff of stderiv); bimodality coefficient (BC) of that
#                  component.  The second derivative exposes subtle mixtures the
#                  first derivative misses (cl 41 BC 0.28->0.55, cl 37 0.44->0.59).
#    calibrate   : a per-cluster single-mode Gaussian surrogate (matched n and
#                  per-feature std) sets the null; excess = BC - null95.  (A
#                  Hartigan dip test is a drop-in alternative null; the surrogate
#                  is what is validated here so it is the default.)
#    reject burst: a single bursting unit goes bimodal in AMPLITUDE (burst spikes
#                  attenuate), not in shape.  Split, then compare the two sub-
#                  templates by cosine: a burst splits one cell into SCALED copies
#                  (cosine >= burst_cos), two cells differ in SHAPE at any energy
#                  (so different-amplitude cells are still split, unlike an
#                  amplitude-alignment test).  (g5: 2531 is a burster; 2551,37,
#                  27,2558,2550 are real two-cell mixes.)
#    split       : a flagged cluster is split on the multi-channel windowed-SVD
#                  component — single-channel localisation loses power because
#                  the cells differ across several channels at once, and the
#                  offending energy is NOT confined to the faint channels.
#
#  Output: a ranked text/TSV report; optional --split writes a new staged .clu
#  with each flagged cluster divided into two sub-ids for review.

import argparse
import numpy as np

try:
    from . import fiber_session as fs, neuro_io as nio, session_yaml as sy
except ImportError:                                              # script / direct execution
    import fiber_session as fs, neuro_io as nio, session_yaml as sy

WIN_HALF       = 13      # samples either side of peak for the analysis window
N_PC           = 4       # top within-cluster SVD components scanned for bimodality
N_NULL         = 16      # single-mode surrogate draws for the per-cluster null
BURST_COS      = 0.92    # sub-template shape cosine at/above this == one bursting cell (scaled copies)
EXCESS_MARGIN  = 0.03    # BC must exceed null95 by this to flag
MIN_SPIKES     = 60      # clusters smaller than this are not scored


def bimodality_coefficient(x):
    """Sarle's BC = (skew^2 + 1) / (kurt + 3(n-1)^2/((n-2)(n-3))).
    Uniform == 0.555; > ~0.555 suggests bimodality.  Sign-invariant."""
    x = np.asarray(x, float)
    x = x - x.mean()
    n = x.size
    s = x.std()
    if n < 12 or s < 1e-9:
        return 0.0
    g = (x ** 3).mean() / s ** 3                                  # skewness
    k = (x ** 4).mean() / s ** 4 - 3.0                            # excess kurtosis
    return float((g * g + 1.0) / (k + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


def _baseline(W):
    """W (n, nsamp, nch) -> per-channel pre-spike baseline removed."""
    return W - W[:, :6, :].mean(1, keepdims=True)


def _window(nsamp, peak):
    lo = max(0, peak - WIN_HALF)
    hi = min(nsamp, peak + WIN_HALF)
    return slice(lo, hi)


def _top_components(Fc, n_pc, rng):
    """Right singular vectors and projections of centred Fc (n, d)."""
    if Fc.shape[0] > 2000:                                        # subsample rows for the basis only
        Fb = Fc[rng.permutation(Fc.shape[0])[:2000]]
    else:
        Fb = Fc
    try:
        _, _, Vt = np.linalg.svd(Fb, full_matrices=False)
    except np.linalg.LinAlgError:
        return None, None
    k = min(n_pc, Vt.shape[0])
    return Vt[:k], Fc @ Vt[:k].T                                  # (k, d), (n, k)


def _surrogate_null95(Fc, n_pc, n_null, rng):
    """95th pct of max-over-PCs BC for a single-mode Gaussian matched to Fc."""
    n, d = Fc.shape
    sd = Fc.std(0)
    vals = []
    for _ in range(n_null):
        S = sd * rng.standard_normal((n, d))                     # single mode by construction
        S -= S.mean(0)
        _, proj = _top_components(S, n_pc, rng)
        if proj is None:
            continue
        vals.append(max(bimodality_coefficient(proj[:, j]) for j in range(proj.shape[1])))
    return float(np.quantile(vals, 0.95)) if vals else 1.0


def _rep_features(Wb, peak, second):
    """Windowed multi-channel feature matrix for a representation.
    second=False -> stderiv (first derivative); True -> its temporal second
    derivative (diff along time).  Returns row-centred feature matrix Fc."""
    X = np.diff(Wb, axis=1) if second else Wb
    p = peak - (1 if second else 0)
    F = X[:, _window(X.shape[1], p), :].reshape(X.shape[0], -1)
    return F - F.mean(0)


def score_cluster(W, peak, *, n_pc=N_PC, n_null=N_NULL, burst_cos=BURST_COS, rng=None):
    """Contamination QC for one cluster's stderiv spikes W (n, nsamp, nch).

    Returns dict(bc, null95, excess, align, nature, rep, proj) where `proj` is
    the per-spike score on the chosen bimodal component (used by --split).
    `nature` is 'two-cell' for a shape-axis mixture, 'burst' for an amplitude-
    axis (single bursting unit), 'clean' when nothing exceeds the null."""
    if rng is None:
        rng = np.random.default_rng(0)
    Wb = _baseline(np.asarray(W, float))
    flat_full = Wb.reshape(Wb.shape[0], -1)
    tmpl = flat_full.mean(0)
    t_hat = tmpl / (np.linalg.norm(tmpl) + 1e-12)                # amplitude (template) direction
    best = None
    for second in (False, True):
        Fc = _rep_features(Wb, peak, second)
        if Fc.shape[0] < 12 or Fc.shape[1] < 2:
            continue
        Vt, proj = _top_components(Fc, n_pc, rng)
        if Vt is None:
            continue
        null95 = _surrogate_null95(Fc, n_pc, n_null, rng)
        for j in range(proj.shape[1]):
            bc = bimodality_coefficient(proj[:, j])
            cand = dict(bc=bc, null95=null95, excess=bc - null95,
                        rep=("d2" if second else "d1"), proj=proj[:, j])
            # the dominant bimodal component decides detection; its nature is read
            # from the actual split difference below (amplitude vs footprint).
            if best is None or cand["excess"] > best["excess"]:
                best = cand
    if best is None:
        return dict(bc=0.0, null95=1.0, excess=-1.0, align=0.0, sub_cos=1.0, nature="clean", rep="d1", proj=None)
    # nature: split on the chosen component, then ask whether the two sub-clusters
    # differ in AMPLITUDE (delta parallel to template -> bursting single unit) or
    # in FOOTPRINT/SHAPE (delta orthogonal -> two cells).  Full-waveform, all channels.
    pr = best["proj"]
    hi = pr >= np.median(pr)
    if hi.sum() >= 5 and (~hi).sum() >= 5:
        a = flat_full[~hi].mean(0)
        b = flat_full[hi].mean(0)
        # amplitude-INVARIANT shape match of the two modes: a true burst splits one
        # cell into scaled copies (cosine ~ 1); two cells differ in shape at any energy.
        best["sub_cos"] = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        delta = b - a
        best["align"] = float(abs(delta @ t_hat) / (np.linalg.norm(delta) + 1e-12))   # diagnostic only
    else:
        best["sub_cos"] = 1.0
        best["align"] = 1.0
    if best["excess"] <= EXCESS_MARGIN:
        best["nature"] = "clean"
    elif best["sub_cos"] >= burst_cos:
        best["nature"] = "burst"
    else:
        best["nature"] = "two-cell"
    return best


def split_recursive(W, peak, *, max_clusters=6, min_size=MIN_SPIKES, n_pc=N_PC,
                    n_null=N_NULL, burst_cos=BURST_COS, rng=None):
    """QC-gated recursive split.  Repeatedly split any part the QC flags as a
    two-cell mixture on its bimodal axis, stopping when every part is clean (or
    below 2*min_size, or max_clusters reached).  Returns sub-labels (n,) in
    0..k-1.  This is the detector driving the splitting phase: a part is divided
    only while the contamination signal survives, so a clean unit is never split
    and a burster (amplitude axis) is left intact."""
    if rng is None:
        rng = np.random.default_rng(0)
    W = np.asarray(W, float)
    final, work = [], [np.arange(len(W))]
    while work:
        idx = work.pop()
        if len(idx) < 2 * min_size or (len(final) + len(work) + 1) >= max_clusters:
            final.append(idx)
            continue
        s = score_cluster(W[idx], peak, n_pc=n_pc, n_null=n_null,
                          burst_cos=burst_cos, rng=rng)
        if s["nature"] == "two-cell" and s["proj"] is not None:
            hi = s["proj"] >= np.median(s["proj"])
            if hi.sum() >= min_size and (~hi).sum() >= min_size:
                work.append(idx[hi])
                work.append(idx[~hi])
                continue
        final.append(idx)
    labels = np.zeros(len(W), int)
    for k, idx in enumerate(final):
        labels[idx] = k
    return labels


def _isi_violation_pct(res_samp, sr, refractory_ms=2.0):
    if len(res_samp) < 3:
        return 0.0
    t = np.sort(res_samp.astype(np.float64)) / sr
    return 100.0 * float(np.mean(np.diff(t) < refractory_ms / 1000.0))


def scan(spk, clu, res, sr, peak, *, min_spikes=MIN_SPIKES, n_pc=N_PC, n_null=N_NULL,
         burst_cos=BURST_COS, seed=0):
    """Score every cluster.  Returns list of per-cluster dicts sorted by excess."""
    rng = np.random.default_rng(seed)
    rows = []
    for cid in np.unique(clu):
        if cid <= 0:                                             # 0 = noise/unsorted, <0 = ignored
            continue
        idx = np.flatnonzero(clu == cid)
        if idx.size < min_spikes:
            continue
        sc = score_cluster(spk[idx], peak, n_pc=n_pc, n_null=n_null,
                            burst_cos=burst_cos, rng=rng)
        sc["cluster"] = int(cid)
        sc["n"] = int(idx.size)
        sc["isi_pct"] = _isi_violation_pct(res[idx], sr)
        # temporal-split rate ratio: min/max spikes across the two recording halves
        tt = res[idx].astype(np.float64)
        mid = 0.5 * (tt.min() + tt.max())
        a, b = int((tt < mid).sum()), int((tt >= mid).sum())
        sc["rate_ratio"] = float(min(a, b) / max(a, b)) if max(a, b) else 0.0
        sc["_idx"] = idx
        rows.append(sc)
    rows.sort(key=lambda r: -r["excess"])
    return rows


def main():
    ap = argparse.ArgumentParser(
        description="Contamination QC for an existing sort: flag two-cell mixtures that ISI "
                    "checks miss, via the per-channel derivative-distribution bimodality of the "
                    "stderiv spikes (amplitude/burst axis rejected).  No re-clustering.")
    sy.add_session_args(ap)
    ap.add_argument("--clu-method", default="stderiv",
                    help="feature space before the group (default stderiv)")
    ap.add_argument("--clu-stage", "--variant", dest="variant", default="refine",
                    help="fiber stage after the group: read <base>.clu.<clu-method>.<elec>.<variant> "
                         "(default refine; '' = no stage)")
    ap.add_argument("--in-clu", default=None, help="explicit .clu path (overrides --clu-method/--variant)")
    ap.add_argument("--min-cluster", type=int, default=MIN_SPIKES, help="skip clusters smaller than this")
    ap.add_argument("--n-pc", type=int, default=N_PC, help="top within-cluster SVD components scanned")
    ap.add_argument("--n-null", type=int, default=N_NULL, help="single-mode surrogate draws for the null")
    ap.add_argument("--burst-cos", type=float, default=BURST_COS,
                    help="sub-template shape cosine at/above this is one bursting cell, not flagged")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write the ranked table to this TSV path")
    ap.add_argument("--split", action="store_true",
                    help="write a new staged .clu (tag '<variant>.csplit') with each flagged "
                         "two-cell cluster recursively QC-split into sub-ids on its bimodal axis")
    ap.add_argument("--max-split", type=int, default=6,
                    help="max sub-clusters a single flagged cluster may be split into (default 6)")
    a = ap.parse_args()

    cfg = sy.resolve_session_params(a.session, a.group, channels=a.channels, ntotal=a.ntotal,
                                    nchan=a.nchan, nsamp=a.nsamp, sr=a.sr)
    base = cfg["base"]; elec = a.group
    nchan, nsamp, sr, peak = cfg["nchan"], cfg["nsamp"], cfg["sr"], cfg["peak"]

    res = fs.read_res(base, elec)
    if a.in_clu:
        _, clu = nio.read_clu_file(a.in_clu, n_spikes=len(res))
    else:
        _, clu = nio.read_clu_at(base, elec, variant=a.clu_method, tag=a.variant, n_spikes=len(res))
    spk, spkpath = fs.open_spkD(base, elec, nsamp, nchan)
    assert spk.shape[0] == len(res) == len(clu), \
        f".res {len(res)} / .clu {len(clu)} / {spkpath} {spk.shape[0]} mismatch"
    print(f"loaded {len(res)} spikes, {len(np.unique(clu[clu > 0]))} clusters ({spkpath})")

    rows = scan(spk, clu, res, sr, peak, min_spikes=a.min_cluster, n_pc=a.n_pc,
                n_null=a.n_null, burst_cos=a.burst_cos, seed=a.seed)

    hdr = f"{'clu':>5} {'n':>6} {'rep':>4} {'BC':>6} {'null95':>7} {'excess':>7} " \
          f"{'subcos':>6} {'ISI%':>6} {'rate':>5}  nature"
    lines = [hdr]
    nflag = 0
    for r in rows:
        flag = r["nature"] == "two-cell"
        nflag += flag
        lines.append(f"{r['cluster']:>5} {r['n']:>6} {r['rep']:>4} {r['bc']:>6.3f} "
                     f"{r['null95']:>7.3f} {r['excess']:>+7.3f} {r['sub_cos']:>6.2f} "
                     f"{r['isi_pct']:>6.2f} {r['rate_ratio']:>5.2f}  "
                     f"{r['nature'].upper() if flag else r['nature']}")
    report = "\n".join(lines)
    print(report)
    print(f"\n{nflag}/{len(rows)} clusters flagged two-cell contamination "
          f"(shape-axis bimodality above null; bursters excluded).")
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(report + "\n")
        print(f"wrote {a.out}")

    if a.split:
        new = clu.copy()
        nxt = int(clu.max()) + 1
        nsplit = nparts = 0
        for r in rows:
            if r["nature"] != "two-cell":
                continue
            idx = r["_idx"]
            sub = split_recursive(spk[idx], peak, max_clusters=a.max_split,
                                  min_size=a.min_cluster, n_pc=a.n_pc, n_null=a.n_null,
                                  burst_cos=a.burst_cos, rng=np.random.default_rng(a.seed))
            k = int(sub.max())
            if k == 0:
                continue                                         # recursion kept it whole
            for s in range(1, k + 1):                            # part 0 keeps the original id
                new[idx[sub == s]] = nxt
                nxt += 1
            nsplit += 1
            nparts += k + 1
        tag = (a.variant + ".csplit") if a.variant else "csplit"
        nio.write_clu(base, elec, new.astype(np.int64),
                      n_clusters=int(len(np.unique(new[new > 0]))),
                      variant=a.clu_method, tag=tag)
        print(f"--split: {nsplit} flagged clusters -> {nparts} parts "
              f"(QC-gated, recursive); wrote .clu.{a.clu_method}.{elec}.{tag}")


if __name__ == "__main__":
    main()
