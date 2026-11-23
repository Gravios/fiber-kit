#!/usr/bin/env python3
"""fiber-unmix-children -- a fiber-unmix result as a hierarchical session.

fiber-unmix names, for a spike the sort left in the noise, the one kept unit it
is a spike of (kind `one`).  As a flat .clu those spikes either sit apart from
their unit (--one-as cluster | units) or inside it, where they can no longer be
told from the unit's own (--one-as assign).  The .clu/.clc/.clp triple holds
both at once:

    .clu  parent layer   the one-unit spikes are IN the unit they were named for
    .clc  child layer    the unit's own spikes are its self child, the one-unit
                         spikes a second child of the same parent
    .clp  the parent of each child

so Klusters shows a unit with what unmix would add to it, and the addition can
be looked at, kept, cut or moved to another parent as one child.  Nothing is
decided here: the unit's own spikes stay one child, exactly as they were.

Read:   <base>.clu.<method>.<g>.<stage>            any --one-as form of the stage
        <base>.unmix.<method>.<g>.<stage>.tsv      the stage's manifest
Write:  <base>.{clu,clc,clp}.<method>.<g>.<out>    (default <stage>_h)
        and, beside them, links to the stage's .eap and .spkc files under the
        new tag, so Klusters shows the collision rows as it does for the stage.

Which rows move.  A `one` row goes under its unit (the manifest's k1) when the
stage's .clu still holds it where fiber-unmix put it: in a cluster made only of
one-unit rows (--one-as cluster | units), or in the unit itself (--one-as
assign).  A row that sits anywhere else was moved by hand since, and stays.
With --collisions the collision rows fiber-unmix did not hand to a unit (its
collision cluster) go under their first unit the same way, as a third child.

Numbering (neurosuite-3, hierarchyfiles.h): the self child of parent p is
p + 10000; every other child is 20000 and up.  Parent ids must be below 10000
(or a reserved chunk cluster).
"""
import argparse
import csv
import os
import shutil
import sys

import numpy as np

try:
    from . import neuro_io as nio
    from . import fiber_chunks as fch
except ImportError:                                   # run as a script
    import neuro_io as nio
    import fiber_chunks as fch

SELF_CHILD_BASE = 10000          # hierarchyfiles.h: kSelfChildBase
FREE_CHILD_BASE = 20000          # hierarchyfiles.h: kFreeChildBase


def is_reserved(ids):
    ids = np.asarray(ids)
    return (ids >= fch.RESERVED_LO) & (ids <= fch.RESERVED_HI)


def self_child(parent):
    """The self child of each parent id: p + 10000 for 2 <= p < 10000, else the id itself
    (the bins 0 and 1, a reserved chunk cluster)."""
    p = np.asarray(parent, np.int64)
    return np.where((p >= 2) & (p < SELF_CHILD_BASE), p + SELF_CHILD_BASE, p)


def read_manifest(path):
    """The manifest's rows as arrays: spike (row index), kind (str), k1 (-1 = none)."""
    spike, kind, k1 = [], [], []
    with open(path, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            spike.append(int(r["spike"])); kind.append(r["kind"])
            k1.append(int(r["k1"]) if r["k1"] not in ("", None) else -1)
    return np.asarray(spike, np.int64), np.asarray(kind), np.asarray(k1, np.int64)


def _movable(clu, rows, k1):
    """Of `rows` (each naming unit k1): those the .clu still holds where fiber-unmix put them --
    in their unit, or in a cluster all of whose spikes are among `rows`."""
    here = clu[rows]
    n_all = dict(zip(*np.unique(clu, return_counts=True)))
    ids, n_rows = np.unique(here, return_counts=True)
    made = {int(i) for i, n in zip(ids, n_rows) if int(n) == int(n_all[i]) and i > 1}
    return (here == k1) | np.isin(here, sorted(made))


def build(clu, spike, kind, k1, *, collisions=False):
    """(parent, child, clp, stats) for a stage .clu and its manifest columns.

    parent / child: (N,) int64.  clp: (highest child id,) int64, child c at entry c - 1 (0 = no such
    child).  stats: counts for the log, and `units`: {unit: child id} of the one-unit children."""
    clu = np.asarray(clu, np.int64)
    n = clu.size
    if spike.size and (spike.min() < 0 or spike.max() >= n):
        raise ValueError(f"the manifest names row {int(spike.max())}, the .clu holds {n:,} spikes")
    parent = clu.copy()
    child = None
    nxt = FREE_CHILD_BASE
    stats = dict(units={}, coll_units={}, one=0, one_left=0, coll=0, coll_left=0)
    groups = [("one", "units", "one", "one_left")]
    if collisions:
        groups.append(("collision", "coll_units", "coll", "coll_left"))
    moved = []                                    # (rows, unit of each, {unit: child id})
    for kname, ukey, nkey, lkey in groups:
        m = (kind == kname) & (k1 >= 2)
        rows, units = spike[m], k1[m]
        if kname == "collision":
            # only the rows fiber-unmix parked: one already in a unit is that unit's spike
            ok = _movable(clu, rows, units) & (clu[rows] != units)
        else:
            ok = _movable(clu, rows, units)
        stats[lkey] = int((~ok).sum())
        rows, units = rows[ok], units[ok]
        stats[nkey] = int(rows.size)
        ids = {}
        for u in np.unique(units):
            ids[int(u)] = nxt
            nxt += 1
        stats[ukey] = ids
        parent[rows] = units
        moved.append((rows, units, ids))
    bad = np.unique(parent[(parent >= SELF_CHILD_BASE) & ~is_reserved(parent)])
    if bad.size:
        raise ValueError(f"parent ids of {SELF_CHILD_BASE} and above cannot carry a child layer "
                         f"({bad.size} of them, from {int(bad[0])}); renumber the stage first")
    child = self_child(parent)
    for rows, units, ids in moved:
        if rows.size:
            child[rows] = np.asarray([ids[int(u)] for u in units], np.int64)
    # a unit all of whose own spikes are gone would leave a parent with no self child: fine, the
    # added child is then its only one.
    atoms, first = np.unique(child, return_index=True)
    top = int(atoms.max()) if atoms.size else 0
    clp = np.zeros(max(top, 0), np.int64)
    for a, i in zip(atoms, first):
        if a >= 1:
            clp[int(a) - 1] = parent[i]
    stats["parents"] = int(np.unique(parent).size)
    stats["children"] = int(atoms.size)
    return parent, child, clp, stats


def check_nesting(parent, child):
    """True when every child lies in one parent (the invariant Klusters reads the triple by)."""
    pairs = np.unique(np.stack([child, parent], 1), axis=0)
    return bool(np.unique(pairs[:, 0]).size == pairs.shape[0])


def link_collision_files(base, elec, stage, out, log):
    """The stage's .eap and .spkc files under the new tag (a hard link, a copy where links are not
    possible): the rows are the same, so the files are."""
    import glob
    made = []
    src = [nio.session_path(base, "eap", elec, tag=stage)]
    src += sorted(glob.glob(glob.escape(base) + ".spkc.*." + f"{elec}.{stage}"))
    for s in src:
        if not os.path.exists(s):
            continue
        d = s[: len(s) - len(stage)] + out
        if os.path.exists(d):
            os.remove(d)
        try:
            os.link(s, d)
        except OSError:
            shutil.copyfile(s, d)
        made.append(os.path.basename(d))
    if made:
        log("[unmix-children] collision files under the new tag: " + ", ".join(made))
    else:
        log(f"[unmix-children] stage '{stage}' has no .eap / .spkc of its own -- none linked")
    return made


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="fiber-unmix-children",
        description="Write a fiber-unmix stage as a hierarchical session (.clu + .clc + .clp): each "
                    "one-unit spike under the unit it was named for, as a child of its own beside "
                    "the unit's spikes.")
    ap.add_argument("base", help="session base (path without extension)")
    ap.add_argument("elec", type=int, help="electrode group")
    ap.add_argument("--clu-method", "--method", dest="clu_method", default="",
                    help="the stage's method token (stderiv_C5, standard, ...)")
    ap.add_argument("--clu-stage", "--stage", dest="clu_stage", required=True,
                    help="the fiber-unmix stage to read (its .clu and its .unmix...tsv)")
    ap.add_argument("--out-tag", default=None, help="output stage tag (default <clu-stage>_h)")
    ap.add_argument("--collisions", action="store_true",
                    help="also put the collision rows fiber-unmix parked in its collision cluster "
                         "under their first unit, as one more child")
    ap.add_argument("--no-collision-files", action="store_true",
                    help="do not link the stage's .eap / .spkc files under the new tag")
    a = ap.parse_args(argv)
    log = lambda s: print(s, file=sys.stderr, flush=True)  # noqa: E731

    base, stage = a.base, a.clu_stage
    out = a.out_tag or f"{stage}_h"
    if out == stage:
        raise SystemExit("fiber-unmix-children: --out-tag is the input stage; the stage's .clu would be overwritten")
    cpath = nio.session_path(base, "clu", a.elec, variant=a.clu_method, tag=stage)
    mpath = nio.session_path(base, "unmix", a.elec, variant=a.clu_method, tag=stage) + ".tsv"
    for p in (cpath, mpath):
        if not os.path.exists(p):
            raise SystemExit(f"fiber-unmix-children: {p} not found")
    clu = np.asarray(nio.read_clu_file(cpath)[1], np.int64)
    spike, kind, k1 = read_manifest(mpath)
    try:
        parent, child, clp, st = build(clu, spike, kind, k1, collisions=a.collisions)
    except ValueError as e:
        raise SystemExit(f"fiber-unmix-children: {e}") from None
    if not check_nesting(parent, child):
        raise SystemExit("fiber-unmix-children: internal error, a child lies in two parents; nothing written")
    paths = [nio.write_clu_file(nio.session_path(base, t, a.elec, variant=a.clu_method, tag=out), ids)
             for t, ids in (("clu", parent), ("clc", child), ("clp", clp))]
    log(f"[unmix-children] {os.path.basename(cpath)}: {clu.size:,} spikes, {np.unique(clu).size} clusters")
    log(f"[unmix-children] {st['one']:,} one-unit spikes -> {len(st['units'])} children "
        f"({FREE_CHILD_BASE}..) under their units"
        + (f"; {st['one_left']:,} left where they are (moved since the stage was written)" if st["one_left"] else ""))
    if a.collisions:
        log(f"[unmix-children] {st['coll']:,} parked collision spikes -> {len(st['coll_units'])} children under their first unit")
    log(f"[unmix-children] wrote {', '.join(os.path.basename(p) for p in paths)}: "
        f"{st['parents']} parents, {st['children']} children")
    if not a.no_collision_files:
        link_collision_files(base, a.elec, stage, out, log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
