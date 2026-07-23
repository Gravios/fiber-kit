#!/usr/bin/env python3
"""
hier_diag.py — diagnose a Klusters hierarchical triple (.clu/.clc/.clp).

Decides the two open questions in the handoff §6:
  (1) PROVENANCE — did Klusters or fiber-kit last write these files? (mtime)
  (2) CHARACTERIZATION — the full straddle picture, so the fix targets the
      right path instead of guessing.

Binary format (shared neurofileio::readCluBinary), confirmed against
data.cpp:  int32 nClusters header, then N x int32 ids in timestamp order.
The .clp is  int32 header, then nChildren x int32 parent-ids indexed by
(childId - 1).

Usage:
    python3 hier_diag.py <base> [suffix]
e.g.
    python3 hier_diag.py sirotaA-jg-000005-20120312 .stderiv_C5.5.fiber_stochastic
which reads  <base>.clu<suffix> / <base>.clc<suffix> / <base>.clp<suffix>.
"""
import sys, os, datetime
import numpy as np


def _read_ids(path):
    """int32 header (nClusters), then per-spike int32 ids."""
    a = np.fromfile(path, dtype=np.int32)
    if a.size < 1:
        raise ValueError(f"{path}: empty / not a binary clu file")
    return int(a[0]), a[1:]


def _mtime(path):
    return datetime.datetime.fromtimestamp(os.path.getmtime(path))


def main(base, suffix):
    p_clu = f"{base}.clu{suffix}"
    p_clc = f"{base}.clc{suffix}"
    p_clp = f"{base}.clp{suffix}"

    # ── (1) PROVENANCE ──────────────────────────────────────────────────────
    print("=" * 70)
    print("(1) PROVENANCE — who wrote these files last?")
    print("=" * 70)
    for p in (p_clu, p_clc, p_clp):
        if os.path.exists(p):
            print(f"  {_mtime(p)}   {os.path.basename(p)}")
        else:
            print(f"  MISSING                       {os.path.basename(p)}")
    print("""
  Interpret:
    * All three mtimes ~equal and LATER than the fiber_stochastic run
        -> Klusters wrote them. Fault is in the recluster/collapse path.
    * mtimes match the fiber_stochastic run (Klusters never re-saved)
        -> Fault is in fiber-kit's _write_cluster_triplet / refiberize.
    * .clp OLDER than .clu/.clc
        -> .clp is a stale cache; buildHierarchyMaps() rescans and warns,
           the straddle lives in the per-spike .clu/.clc themselves.
""")

    nclu_hdr, clu = _read_ids(p_clu)
    nclc_hdr, clc = _read_ids(p_clc)
    if clu.size != clc.size:
        print(f"!! .clu has {clu.size} spikes but .clc has {clc.size} — not aligned; abort")
        return 2
    nspk = clu.size
    parents = np.unique(clu[clu > 0])
    children = np.unique(clc[clc > 0])
    print(f"  {nspk} spikes | {parents.size} parents (fibers) | {children.size} children (atoms)")
    print(f"  clu header nClusters={nclu_hdr}  clc header nClusters={nclc_hdr}")

    # ── (2a) STRADDLE: children spanning >1 parent ──────────────────────────
    print("\n" + "=" * 70)
    print("(2a) NESTING VIOLATION — children whose spikes span >1 parent")
    print("=" * 70)
    bad = {}
    for c in children:
        ps, counts = np.unique(clu[clc == c], return_counts=True)
        ps_counts = [(int(p), int(n)) for p, n in zip(ps, counts) if p > 0]
        # a child sitting partly in noise/artifact (parent<=0) is not a
        # two-real-parent straddle; require >=2 distinct positive parents
        if sum(1 for p, _ in ps_counts if p > 0) > 1:
            bad[int(c)] = sorted(ps_counts, key=lambda t: -t[1])
    if not bad:
        print("  none — every child is nested in exactly one fiber. Invariant holds.")
    else:
        print(f"  {len(bad)} of {children.size} children straddle two+ fibers:\n")
        print(f"  {'child':>8}  {'parents (id:spikes, plurality first)'}")
        for c in sorted(bad):
            spread = "  ".join(f"{p}:{n}" for p, n in bad[c])
            # minority share tells collapse-vs-genuine-split apart
            tot = sum(n for _, n in bad[c])
            minority = sum(n for _, n in bad[c][1:])
            frac = minority / tot
            tag = "  <- tiny remainder (looks like an UN-collapsed straddler)" if frac < 0.10 else ""
            print(f"  {c:>8}  {spread}{tag}")

        # signature check: are the bad child ids clustered just above nParents?
        maxpar = int(parents.max())
        badids = np.array(sorted(bad))
        just_above = badids[(badids > maxpar) & (badids <= maxpar + 64)]
        print(f"\n  max parent id = {maxpar}")
        if just_above.size:
            print(f"  {just_above.size} bad child id(s) sit within +64 of it "
                  f"({just_above.min()}..{just_above.max()}) — the recluster/promote")
            print(f"  id-offset signature (new atom ids = target Data's highestClusterId()+k).")

    # ── (2b) COVERAGE: does each parent == union of its children? ────────────
    print("\n" + "=" * 70)
    print("(2b) COVERAGE — parents whose spike set exceeds their children's union")
    print("=" * 70)
    # first-seen owner (what rebuildHierarchyFromData derives): child -> parent
    owner = {}
    for c in children:
        rows = np.flatnonzero(clc == c)
        owner[int(c)] = int(clu[rows[0]])
    # parent -> its owned children
    from collections import defaultdict
    kids = defaultdict(list)
    for c, p in owner.items():
        kids[p].append(c)
    leaks = []
    for p in parents:
        p = int(p)
        pmask = clu == p
        covered = np.isin(clc, kids.get(p, [])) & pmask
        if int(pmask.sum()) != int(covered.sum()):
            leaks.append((p, int(pmask.sum()), int(covered.sum())))
    if not leaks:
        print("  none — every parent is exactly covered by its first-seen children.")
    else:
        print(f"  {len(leaks)} parent(s) reach spikes through a child owned elsewhere:\n")
        print(f"  {'parent':>8}  {'spikes':>8}  {'covered':>8}  {'leaked':>8}")
        for p, tot, cov in leaks:
            print(f"  {p:>8}  {tot:>8}  {cov:>8}  {tot - cov:>8}")

    # ── (2c) .clp cross-check ───────────────────────────────────────────────
    if os.path.exists(p_clp):
        print("\n" + "=" * 70)
        print("(2c) .clp CACHE vs per-spike truth")
        print("=" * 70)
        _, clp = _read_ids(p_clp)   # parent-id per child, indexed by childId-1
        stale = 0
        for c, p in owner.items():
            if 1 <= c <= clp.size and clp[c - 1] != p:
                stale += 1
        print(f"  {stale} of {len(owner)} children disagree between .clp and the per-spike .clu/.clc")
        if stale:
            print("  -> .clp is stale; buildHierarchyMaps() rescans and regenerates it on Save.")
    print()
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    base = sys.argv[1]
    suffix = sys.argv[2] if len(sys.argv) > 2 else ""
    sys.exit(main(base, suffix))
