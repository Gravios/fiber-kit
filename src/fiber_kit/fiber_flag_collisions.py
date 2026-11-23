#!/usr/bin/env python3
# ══════════════════════════════════════════════════════════════════════════════
#  fiber_flag_collisions.py — flag probable collisions BEFORE clustering, from
#  the energy a snippet holds AWAY from its own spike.  Adds them to the
#  per-spike exclusion mask fiber-session --exclude-clu reads.
#
#  WHY.  A window that holds two spikes is ambiguous to a clusterer: it sits
#  between its two units, pulls their clusters toward each other, and costs the
#  sort time it cannot turn into anything.  It is cheap to recognise first --
#  nothing but the .spk and the .res is needed -- and fiber-unmix explains it
#  afterwards, against units that were built without it.
#
#  THE SCORE (label-free, per spike):
#      p(s) = sum over channels of w[s, c]^2 / sigma_c^2
#  (sigma_c: the channel's robust noise, from the window's first samples).  A
#  spike has energy away from its peak too -- its own tail -- and more of it the
#  larger it is.  So the spikes are binned by their peak energy and, per bin and
#  sample, the median and the MAD of p(s) are taken; z(s) is how far a spike's
#  p(s) lies above its bin's median, in MADs.  The score is the largest z, over
#  the samples more than --guard from the peak, smoothed over 3 samples.
#
#  THE THRESHOLD is on the score itself (--score-min, in MADs), so it needs no
#  labels and does not depend on how many collisions a group holds.  (A quantile
#  of the score cannot do that: where one spike in ten is a collision, the top
#  2 % is a fifth of them -- the test fixture's first version showed it.)
#  --max-share caps the share of all spikes the score may flag, by raising the
#  threshold; it is off by default.
#
#  ALSO FLAGGED, whatever its score: a spike with another detection of the group
#  within --neighbour samples (default half a window) -- two spikes in one
#  window for certain.
#
#  MEASURED on sirotaA-jg-000005-20120312 group 8 (815,728 spikes, standard .spk,
#  2026-11-23; analysis/off-spike-energy).  Known two-spike windows: the 7,418
#  spikes with another detection 9-26 samples away.  "Kept" / "noise": what a
#  cleaning pass (fiber-clean) later made of the spike; "isolated": no other
#  detection within 80 samples.  The score alone (the neighbour rule off):
#      score >   of all   of the known pairs   of kept+isolated   of kept   of noise
#          8     10.7 %        81 %                3.5 %           4.9 %    16.5 %
#         10      7.8 %        73 %                2.3 %           3.2 %    12.3 %
#         12      5.8 %        66 %                1.5 %           2.2 %     9.4 %
#         15      3.9 %        55 %                0.9 %           1.3 %     6.5 %
#         20      2.2 %        41 %                0.4 %           0.6 %     3.8 %
#  AUC of the score, known pairs against kept+isolated: 0.969; its excess sits
#  within 3 samples of the neighbour for 77 % of the flagged pairs.
#  LIMITS.  Those positives are pairs in which BOTH spikes crossed the detection
#  threshold; a collision with a smaller, undetected partner is flagged less
#  often, and by how much was not measured (no ground truth for it here).  A
#  partner within --guard samples of the peak is not seen at all.  The negatives
#  "kept+isolated" spikes are not all clean either -- some hold an undetected
#  partner -- so the share of truly single spikes flagged is at most that column.
#
#  Output: the exclusion mask <base>.clu.<spk-method>.<group>.<out-stage> (the
#  one fiber-flag-artifacts writes; 0 = keep).  An existing mask is kept and
#  added to: its non-zero entries stay, collisions are written as 2.  Nothing is
#  removed from any file; with FK_SESSION_EXCLUDE=1 fiber-session leaves the
#  masked spikes out of every chunk and emits them as cluster 0, where
#  fiber-unmix (--noise-ids 0,1) picks them up.
# ══════════════════════════════════════════════════════════════════════════════
import argparse
import os
import sys

import numpy as np

try:
    from . import fiber_chunks as fch
    from . import neuro_io as nio
    from . import session_yaml as sy
except ImportError:
    import fiber_chunks as fch
    import neuro_io as nio
    import session_yaml as sy

COLLISION = 2          # the mask value of a flagged collision (1 = fiber-flag-artifacts' artifact)


def nearest_detection(res):
    """Per spike: samples to the nearest other detection.  The .res need not be sorted (a realign
    leaves a few rows out of order)."""
    res = np.asarray(res, np.int64)
    big = np.iinfo(np.int64).max // 2
    if res.size < 2:
        return np.full(res.size, big)
    o = np.argsort(res, kind="stable")
    d = np.diff(res[o])
    out = np.empty(res.size, np.int64)
    out[o] = np.minimum(np.r_[big, d], np.r_[d, big])
    return out


def off_spike_score(spk, peak, *, guard=8, bins=20, sample=40000, batch=50000, seed=0):
    """The score described in the header, for every row of spk (N, nSamples, nChannels).
    Returns (score, lag): lag = the sample offset from the peak at which the excess is largest."""
    n, ns, nc = spk.shape
    rng = np.random.default_rng(seed)
    sub = np.sort(rng.choice(n, min(n, sample), replace=False))
    head = np.asarray(spk[sub][:, :4, :], np.float32)
    sig = 1.4826 * np.median(np.abs(head - np.median(head, (0, 1))), (0, 1))
    sig = np.where(sig > 0, sig, 1.0).astype(np.float32)
    P = np.empty((n, ns), np.float32)
    for a in range(0, n, batch):
        P[a:a + batch] = ((np.asarray(spk[a:a + batch], np.float32) / sig) ** 2).sum(2)
    lo, hi = max(0, peak - 2), min(ns, peak + 3)
    pk = P[:, lo:hi].max(1)
    nb = int(max(1, min(bins, n // 200)))
    edges = np.quantile(pk[sub], np.linspace(0, 1, nb + 1)[1:-1]) if nb > 1 else np.zeros(0)
    which = np.searchsorted(edges, pk)
    Z = np.empty((n, ns), np.float32)
    for k in range(nb):
        m = which == k
        if not m.any():
            continue
        med = np.median(P[m], 0)
        mad = 1.4826 * np.median(np.abs(P[m] - med), 0) + 1e-6
        Z[m] = (P[m] - med) / mad
    Zs = (Z + np.roll(Z, 1, 1) + np.roll(Z, -1, 1)) / 3.0
    off = np.abs(np.arange(ns) - peak) > guard
    off[0] = off[-1] = False                               # the smoothing wraps there
    Zs[:, ~off] = -np.inf
    return Zs.max(1), Zs.argmax(1) - peak


def flag(spk, res, peak, *, score_min=12.0, max_share=0.0, guard=8, neighbour=None, bins=20, seed=0):
    """(flags bool, info dict) for every spike: the off-spike score above score_min (raised, with
    max_share > 0, until the score flags at most that share of the spikes), or another detection
    within `neighbour` samples (None = half a window, 0 = rule off)."""
    n, ns, _ = spk.shape
    near = nearest_detection(res)
    score, lag = off_spike_score(spk, peak, guard=guard, bins=bins, seed=seed)
    thr = float(score_min)
    if max_share > 0 and n:
        thr = max(thr, float(np.quantile(score, 1.0 - min(max_share, 1.0))))
    neighbour = ns // 2 if neighbour is None else int(neighbour)
    by_score = score > thr
    by_near = (near <= neighbour) if neighbour > 0 else np.zeros(n, bool)
    info = dict(threshold=thr, raised=bool(thr > score_min), by_score=int(by_score.sum()),
                by_neighbour=int(by_near.sum()), both=int((by_score & by_near).sum()), score=score, lag=lag,
                neighbour=neighbour, near_flagged=float(by_score[by_near].mean()) if by_near.any() else float("nan"))
    return by_score | by_near, info


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="fiber-flag-collisions",
        description="Flag probable collisions before clustering, from the energy a snippet holds away from "
                    "its own spike, and add them to the exclusion mask fiber-session --exclude-clu reads.")
    ap.add_argument("base"); ap.add_argument("elec", type=int)
    ap.add_argument("--spk-method", "--spk-variant", dest="spk_variant", default="standard",
                    help="waveform file the score is taken on (default standard)")
    ap.add_argument("--out-stage", "--out-tag", dest="out_tag", default="artifact",
                    help="stage tag of the mask .clu (default artifact: fiber-flag-artifacts' mask, which is "
                         "kept and added to)")
    ap.add_argument("--score-min", type=float, default=12.0,
                    help="a spike whose off-spike score exceeds this is flagged (default 12, in MADs of its "
                         "peak-energy bin; the module header has the measured rates for 8-20)")
    ap.add_argument("--max-share", type=float, default=0.0,
                    help="most of the spikes the score may flag: the threshold is raised to keep to it "
                         "(default 0 = no cap)")
    ap.add_argument("--guard", type=int, default=8, help="samples either side of the peak the score ignores")
    ap.add_argument("--neighbour", type=int, default=-1,
                    help="a spike with another detection within this many samples is flagged whatever its "
                         "score (default -1 = half a window; 0 = off)")
    ap.add_argument("--bins", type=int, default=20, help="peak-energy bins the expected tail is taken in")
    ap.add_argument("--dry-run", action="store_true", help="report the flagged counts, write nothing")
    a = ap.parse_args(argv)
    log = lambda s: print(s, file=sys.stderr, flush=True)  # noqa: E731
    if not 0.0 <= a.max_share <= 1.0:
        raise SystemExit("--max-share must be between 0 and 1")
    cfg = sy.resolve_session_params(a.base, a.elec, verbose=False)
    base = cfg["base"]; NS, NC = int(cfg["nsamp"]), int(cfg["nchan"])
    peak = int(cfg["peak"]) if cfg.get("peak") is not None else NS // 2
    res = np.asarray(nio.read_res(base, a.elec), np.int64); n = res.size
    spkp = nio.session_path(base, "spk", a.elec, variant=a.spk_variant)
    if not os.path.exists(spkp):
        raise SystemExit(f"fiber-flag-collisions needs {os.path.basename(spkp)}")
    spk = nio.open_spk_file(spkp, NS, NC)
    if spk.shape[0] != n:
        raise SystemExit(f"{os.path.basename(spkp)} holds {spk.shape[0]:,} spikes, the .res {n:,}")
    flags, info = flag(spk, res, peak, score_min=a.score_min, max_share=a.max_share, guard=a.guard,
                       neighbour=None if a.neighbour < 0 else a.neighbour, bins=a.bins)
    log(f"[flag-collisions] {os.path.basename(base)} group {a.elec}: {n:,} spikes, {os.path.basename(spkp)} | "
        f"score > {info['threshold']:.1f}"
        + (f" (raised from {a.score_min:g} to keep to --max-share {a.max_share:g})" if info["raised"] else ""))
    log(f"[flag-collisions] flagged {int(flags.sum()):,} ({flags.mean():.1%}): {info['by_score']:,} by the score, "
        f"{info['by_neighbour']:,} with another detection within {info['neighbour']} samples "
        f"({info['both']:,} both"
        + (f"; the score alone flags {info['near_flagged']:.0%} of those" if info["by_neighbour"] else "") + ")")
    out = nio.session_path(base, "clu", a.elec, variant=a.spk_variant, tag=a.out_tag)
    mask = np.zeros(n, np.int64)
    kept = 0
    if os.path.exists(out):
        try:
            _, old = nio.read_clu_file(out, n_spikes=n)
            mask = np.asarray(old, np.int64).copy()
            mask[mask == COLLISION] = 0                    # an earlier run's collisions are redone
            kept = int((mask != 0).sum())
        except ValueError as e:
            raise SystemExit(f"{os.path.basename(out)}: {e}") from None
    new = flags & (mask == 0)
    mask[new] = COLLISION
    if a.dry_run:
        log("[flag-collisions] dry run: nothing written")
        return 0
    nio.write_clu_file(out, mask.astype(np.int32))
    log(f"[flag-collisions] wrote {os.path.basename(out)}: {int(new.sum()):,} collisions as {COLLISION}"
        + (f", {kept:,} earlier entries kept (artifacts, chunks not evaluated)" if kept else "")
        + " -- 0 = keep; fiber-session reads it with --exclude-clu (FK_SESSION_EXCLUDE=1)")
    reserved = int(fch.is_reserved(mask).sum())
    if reserved:
        log(f"[flag-collisions] {reserved:,} spikes are in chunks the mask does not evaluate (reserved ids, kept)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
