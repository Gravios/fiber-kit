#!/usr/bin/env python3
"""Score a candidate clustering against a ground-truth clustering.

Cluster *count* is a deceptive quality signal -- a method can hit the right number of clusters while
absorbing distinct cells (over-merge) or shattering one cell across fragments (over-split).  This scorer
measures the thing that actually matters by comparing a candidate ``.clu`` to a curated ground-truth
``.clu`` over the same spikes:

  * Adjusted Rand Index and V-measure (homogeneity / completeness) -- overall agreement.
  * PAIRWISE precision / recall / F1 -- the over-merge vs over-split axis directly:
        low pairwise recall    -> the candidate SPLITS ground-truth units (fragments)
        low pairwise precision -> the candidate MERGES distinct ground-truth units
  * Per-unit best-match recall / precision / F1, and counts of split GT units and merged candidates.

Candidate and ground truth are aligned by spike index when they have equal length, or by ``.res``
timestamp intersection when a ``.res`` is given for each (so a ground truth covering only part of the
session -- e.g. a curated head -- still scores correctly).  This module is import-clean: ``score`` and the
metric helpers are pure-numpy and have no I/O or argparse side effects, so other stages can call them.
"""
import numpy as np

from . import neuro_io as nio


# ───────────────────────────── metric core (pure numpy) ─────────────────────────────
def contingency(gt, cand):
    """Return (gt_ids, cand_ids, table) where table[i, j] = #spikes with ground-truth gt_ids[i] and
    candidate cand_ids[j].  Both inputs are 1-D integer label arrays of equal length."""
    gt = np.asarray(gt); cand = np.asarray(cand)
    if gt.shape != cand.shape:
        raise ValueError("label arrays must be the same length (%d vs %d)" % (gt.size, cand.size))
    gt_ids = np.unique(gt); cand_ids = np.unique(cand)
    gi = {v: i for i, v in enumerate(gt_ids)}
    ci = {v: j for j, v in enumerate(cand_ids)}
    table = np.zeros((gt_ids.size, cand_ids.size), dtype=np.int64)
    for g, c in zip(gt, cand):
        table[gi[g], ci[c]] += 1
    return gt_ids, cand_ids, table


def _comb2(x):
    x = np.asarray(x, dtype=np.float64)
    return x * (x - 1.0) / 2.0


def adjusted_rand(table):
    """Adjusted Rand Index from a contingency table (1.0 = identical partition; ~0 = chance)."""
    n = table.sum()
    if n < 2:
        return 1.0
    sum_ij = _comb2(table).sum()
    sa = _comb2(table.sum(1)).sum()
    sb = _comb2(table.sum(0)).sum()
    expected = sa * sb / _comb2(n)
    maxidx = 0.5 * (sa + sb)
    if maxidx == expected:
        return 1.0
    return float((sum_ij - expected) / (maxidx - expected))


def v_measure(table):
    """(homogeneity, completeness, v_measure) -- classes are ground-truth rows, clusters are candidate
    columns.  homogeneity=1 each candidate cluster holds a single GT unit; completeness=1 each GT unit
    lands in a single candidate cluster."""
    n = float(table.sum())
    if n == 0:
        return 1.0, 1.0, 1.0
    a = table.sum(1).astype(np.float64)      # GT (class) sizes
    b = table.sum(0).astype(np.float64)      # candidate (cluster) sizes

    def _ent(p):
        p = p[p > 0] / n
        return float(-(p * np.log(p)).sum())

    h_c = _ent(a); h_k = _ent(b)
    nz = table[table > 0].astype(np.float64)
    rows, cols = np.nonzero(table)
    bj = b[cols]; ai = a[rows]
    h_c_given_k = float(-((nz / n) * np.log(nz / bj)).sum())
    h_k_given_c = float(-((nz / n) * np.log(nz / ai)).sum())
    homo = 1.0 if h_c == 0 else 1.0 - h_c_given_k / h_c
    comp = 1.0 if h_k == 0 else 1.0 - h_k_given_c / h_k
    v = 0.0 if (homo + comp) == 0 else 2.0 * homo * comp / (homo + comp)
    return homo, comp, v


def pairwise_prf(table):
    """Pairwise (precision, recall, f1) over spike pairs.  recall = fraction of same-GT pairs the
    candidate keeps together (low -> over-split); precision = fraction of same-candidate pairs that are
    truly same-GT (low -> over-merge)."""
    same_both = _comb2(table).sum()
    same_gt = _comb2(table.sum(1)).sum()
    same_cand = _comb2(table.sum(0)).sum()
    recall = 1.0 if same_gt == 0 else float(same_both / same_gt)
    precision = 1.0 if same_cand == 0 else float(same_both / same_cand)
    f1 = 0.0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def per_unit_match(gt_ids, cand_ids, table, split_tol=0.2):
    """Per ground-truth unit: best-matching candidate, recall, precision, F1, fragment count, split flag.
    A unit is 'split' when its largest single candidate piece is < (1 - split_tol) of the unit."""
    rows = []
    a = table.sum(1)
    b = table.sum(0)
    for i, g in enumerate(gt_ids):
        size = int(a[i])
        if size == 0:
            continue
        j = int(np.argmax(table[i]))
        overlap = int(table[i, j])
        recall = overlap / size
        precision = overlap / int(b[j]) if b[j] else 0.0
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        nfrag = int((table[i] > 0).sum())
        rows.append(dict(gt=int(g), size=size, best_cand=int(cand_ids[j]), recall=recall,
                         precision=precision, f1=f1, n_fragments=nfrag, split=recall < (1.0 - split_tol)))
    return rows


def merged_candidates(gt_ids, cand_ids, table, merge_frac=0.2):
    """Candidate clusters that absorb >= 2 ground-truth units, each contributing >= merge_frac of the
    candidate's spikes."""
    out = []
    b = table.sum(0)
    for j, c in enumerate(cand_ids):
        size = int(b[j])
        if size == 0:
            continue
        contrib = [(int(gt_ids[i]), int(table[i, j])) for i in range(len(gt_ids))
                   if table[i, j] >= merge_frac * size and table[i, j] > 0]
        if len(contrib) >= 2:
            out.append(dict(cand=int(c), size=size,
                            gt_units=[g for g, _ in contrib], counts=[n for _, n in contrib]))
    return out


def align_by_res(cand_labels, cand_res, gt_labels, gt_res):
    """Restrict both label arrays to spikes whose .res timestamps are common to candidate and ground
    truth (inner join on time).  Use when the ground truth covers only part of the session."""
    cand_res = np.asarray(cand_res); gt_res = np.asarray(gt_res)
    common, ci, gi = np.intersect1d(cand_res, gt_res, return_indices=True)
    return np.asarray(cand_labels)[ci], np.asarray(gt_labels)[gi], common.size


def score(cand_labels, gt_labels, gt_noise=(0, 1), cand_noise=(), split_tol=0.2, merge_frac=0.2):
    """Compare candidate vs ground-truth label arrays (equal length, index-aligned).  Scoring is restricted
    to spikes whose ground-truth label is a real unit (not in gt_noise); cand_noise, if given, likewise
    drops candidate-noise spikes from scope.  Returns a summary dict."""
    cand = np.asarray(cand_labels); gt = np.asarray(gt_labels)
    if cand.shape != gt.shape:
        raise ValueError("candidate and ground-truth must be the same length; align by .res first")
    keep = ~np.isin(gt, list(gt_noise))
    if cand_noise:
        keep &= ~np.isin(cand, list(cand_noise))
    cand = cand[keep]; gt = gt[keep]
    gt_ids, cand_ids, table = contingency(gt, cand)
    pp, pr, pf = pairwise_prf(table)
    homo, comp, vm = v_measure(table)
    units = per_unit_match(gt_ids, cand_ids, table, split_tol=split_tol)
    merges = merged_candidates(gt_ids, cand_ids, table, merge_frac=merge_frac)
    f1s = np.array([u["f1"] for u in units]) if units else np.array([0.0])
    sizes = np.array([u["size"] for u in units]) if units else np.array([1])
    return dict(
        n_scored=int(gt.size), n_gt_units=int(gt_ids.size), n_cand_clusters=int(cand_ids.size),
        ari=adjusted_rand(table), homogeneity=homo, completeness=comp, v_measure=vm,
        pairwise_precision=pp, pairwise_recall=pr, pairwise_f1=pf,
        mean_f1=float(f1s.mean()), median_f1=float(np.median(f1s)),
        weighted_f1=float(np.average(f1s, weights=sizes)),
        n_gt_split=int(sum(u["split"] for u in units)), n_cand_merged=len(merges),
        units=units, merges=merges)


# ───────────────────────────── reporting ─────────────────────────────
def format_report(s, top=8):
    L = []
    L.append("scored %d spikes : %d ground-truth units vs %d candidate clusters"
             % (s["n_scored"], s["n_gt_units"], s["n_cand_clusters"]))
    L.append("  ARI %.4f   V-measure %.4f  (homogeneity %.3f / completeness %.3f)"
             % (s["ari"], s["v_measure"], s["homogeneity"], s["completeness"]))
    L.append("  pairwise  precision %.4f  recall %.4f  F1 %.4f" %
             (s["pairwise_precision"], s["pairwise_recall"], s["pairwise_f1"]))
    diag = []
    if s["pairwise_recall"] < 0.95:
        diag.append("OVER-SPLIT (%d GT units fragmented)" % s["n_gt_split"])
    if s["pairwise_precision"] < 0.95:
        diag.append("OVER-MERGE (%d candidates absorb multiple units)" % s["n_cand_merged"])
    L.append("  diagnosis: " + (", ".join(diag) if diag else "clean (no dominant split/merge)"))
    L.append("  per-unit F1  mean %.3f  median %.3f  size-weighted %.3f"
             % (s["mean_f1"], s["median_f1"], s["weighted_f1"]))
    worst = sorted(s["units"], key=lambda u: u["f1"])[:top]
    if worst:
        L.append("  worst %d ground-truth units:" % len(worst))
        L.append("    %-6s %-7s %-7s %-7s %-7s %-5s" % ("gt", "size", "recall", "prec", "F1", "frags"))
        for u in worst:
            L.append("    %-6d %-7d %-7.3f %-7.3f %-7.3f %-5d%s" %
                     (u["gt"], u["size"], u["recall"], u["precision"], u["f1"], u["n_fragments"],
                      "  SPLIT" if u["split"] else ""))
    if s["merges"]:
        L.append("  candidates merging multiple GT units:")
        for m in sorted(s["merges"], key=lambda m: -m["size"])[:top]:
            L.append("    cand %-6d size %-7d <- GT units %s" % (m["cand"], m["size"], m["gt_units"]))
    return "\n".join(L)


# ───────────────────────────── loaders / CLI ─────────────────────────────
def _load_clu(path):
    _, ids = nio.read_clu_file(path)
    return ids


def main():
    import argparse
    import json
    ap = argparse.ArgumentParser(
        description="Score a candidate .clu against a ground-truth .clu (ARI, V-measure, pairwise "
                    "precision/recall, per-unit and split/merge diagnostics).")
    ap.add_argument("--candidate", "--cand", required=True, help="candidate .clu file")
    ap.add_argument("--gt", "--ground-truth", required=True, help="ground-truth .clu file")
    ap.add_argument("--candidate-res", default=None, help=".res for the candidate (for timestamp alignment)")
    ap.add_argument("--gt-res", default=None, help=".res for the ground truth (for timestamp alignment)")
    ap.add_argument("--gt-noise", default="0,1", help="GT cluster ids to exclude from scope (default 0,1)")
    ap.add_argument("--cand-noise", default="", help="candidate cluster ids to drop from scope (default none)")
    ap.add_argument("--split-tol", type=float, default=0.2, help="largest-piece shortfall to call a unit split")
    ap.add_argument("--merge-frac", type=float, default=0.2, help="min share for a GT unit to count in a merge")
    ap.add_argument("--top", type=int, default=8, help="how many worst units / merges to list")
    ap.add_argument("--json", default=None, help="also write the full summary as JSON")
    args = ap.parse_args()

    def _ids(s):
        return tuple(int(x) for x in s.replace(",", " ").split()) if s.strip() else ()

    cand = _load_clu(args.candidate)
    gt = _load_clu(args.gt)
    if cand.size != gt.size:
        if not (args.candidate_res and args.gt_res):
            ap.error("candidate (%d) and ground-truth (%d) differ in length; pass --candidate-res and "
                     "--gt-res to align by timestamp" % (cand.size, gt.size))
        cres = nio.read_res_file(args.candidate_res)
        gres = nio.read_res_file(args.gt_res)
        cand, gt, ncommon = align_by_res(cand, cres, gt, gres)
        print("aligned by .res: %d common spikes" % ncommon)

    s = score(cand, gt, gt_noise=_ids(args.gt_noise), cand_noise=_ids(args.cand_noise),
              split_tol=args.split_tol, merge_frac=args.merge_frac)
    print(format_report(s, top=args.top))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(s, fh, indent=2)
        print("wrote %s" % args.json)


if __name__ == "__main__":
    main()
