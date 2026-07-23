#!/usr/bin/env python3
"""fiber_refiber.py — KlustaKwik-style re-fibering of microfiber atoms.

Given a per-spike atom layer (.clc) — e.g. the atoms produced by a "replace"
spike-recluster in Klusters, where the recluster output becomes the new atoms —
group those atoms into fibers and write the .clu/.clc/.clp triple.  The atom
layer (.clc) is the STABLE unit and is preserved verbatim; only the fiber layer
(.clu) and the child->parent map (.clp) are (re)computed.  Load the result in
Klusters and the atoms appear as children of the new fibers, ready to curate.

Algorithm (atom-respecting; never splits an atom):
  1. read the atoms (.clc) — the per-spike stable layer
  2. median RAW template per atom (.spk/.fil); localize each atom by its raw
     monopole footprint (fiber_localize) -> position {y0, z0, A, one_flank}
  3. co-gate position + template shape (fiber_merge.position_shape_merge):
     two atoms join a fiber iff they agree in depth/distance/amplitude AND their
     median templates are shape-consistent
  4. write the triple via FiberHierarchy (atoms unchanged; fibers = the groups)

This mirrors the existing KlustaKwik / fiber-localize interface: positional
<base> <group>, session params resolved from <session>.yaml (overridable), and
the merge gates exposed as flags.  It is the re-agglomeration counterpart to a
spike-level recluster: the recluster makes atoms, this makes fibers from them.
"""
import os
import shutil
import argparse

import numpy as np

try:
    from . import (neuro_io as nio, fiber_lib as fl, fiber_localize as floc,
                   fiber_merge as fm, session_yaml as sy)
    from .fiber_refiberize import FiberHierarchy
except ImportError:                                    # running as a flat script
    import neuro_io as nio
    import fiber_lib as fl
    import fiber_localize as floc
    import fiber_merge as fm
    import session_yaml as sy
    from fiber_refiberize import FiberHierarchy


def atom_templates(spk, atoms, ids, *, max_spikes=2000, rng=None):
    """Median RAW template (nsamp, nchan) per atom id."""
    if rng is None:
        rng = np.random.default_rng(0)
    out = {}
    for a in ids:
        idx = np.flatnonzero(atoms == a)
        if idx.size == 0:
            continue
        if idx.size > max_spikes:                      # subsample for the median
            idx = np.sort(rng.choice(idx, max_spikes, replace=False))
        out[int(a)] = np.median(np.asarray(spk[idx], np.float32), axis=0)
    return out


def groups_to_parent(ids, groups):
    """Map every atom id to a fiber id.  Multi-atom groups come first (largest
    first); atoms in no group keep their own singleton fiber so nothing is lost."""
    parent = {}
    for fib, grp in enumerate(sorted(groups, key=lambda g: -len(g)), start=1):
        for a in grp:
            parent[int(a)] = fib
    nxt = (max(parent.values()) + 1) if parent else 1
    for a in ids:
        if int(a) not in parent:
            parent[int(a)] = nxt
            nxt += 1
    return parent


def refiber(base, group, *, variant="stderiv", tag="microfiber",
            session=None, probe=None, channels=None, nsamp=None, nchan=None,
            dy_um=6.0, dlogA=0.25, dz_um=8.0, min_cos=0.85, clique=False,
            mask=None, max_spikes=2000, verbose=True):
    """Re-fiber the atoms of <base>.clc.<variant>.<group>.<tag>.

    Returns (atoms_per_spike, parent_map {atom: fiber}, groups)."""
    cfg = sy.resolve_session_params(session or base, group, require=()) or {}
    if isinstance(channels, str) and channels:
        channels = [int(c) for c in channels.split(",")]
    channels = channels or cfg.get("channels")
    probe = probe or cfg.get("probe")
    nsamp = nsamp if nsamp is not None else cfg.get("nsamp")
    nchan = nchan if nchan is not None else cfg.get("nchan")
    if not channels or not probe or nsamp is None or nchan is None:
        raise SystemExit("[refiber] need channels/probe/nsamp/nchan from "
                         "<session>.yaml or --channels/--probe/--nsamp/--nchan")
    mask = fl.MASK_FULL if mask is None else mask

    # 1. atoms (.clc) — the stable per-spike layer
    n, atoms = nio.read_clu_at(base, group, variant=variant, tag=tag)
    atoms = np.asarray(atoms, np.int64)

    # 2. per-atom RAW templates (localization MUST be on raw amplitudes)
    spk, _ = nio.open_spk_raw(base, group, nsamp, nchan)
    spk = np.asarray(spk)
    n = min(n, spk.shape[0])
    atoms = atoms[:n]
    ids = [int(a) for a in np.unique(atoms) if a >= 1]
    templates = atom_templates(spk, atoms, ids, max_spikes=max_spikes)

    # 3. per-atom localization (raw monopole) -> position dict
    xy = floc.load_geometry(probe, channels)
    clc_path = nio.session_path(base, "clc", group, variant=variant, tag=tag)
    rows = floc.localize(base, group, nsamp, nchan, xy, clu_path=clc_path,
                         verbose=verbose)
    pos = {int(r["unit"]): r for r in rows}

    # keep atoms that were both localized AND have a template
    keep = [a for a in ids if a in pos and a in templates]
    pos = {a: pos[a] for a in keep}
    templates = {a: templates[a] for a in keep}
    probe_y = ((float(xy[:, 1].min()), float(xy[:, 1].max()))
               if getattr(xy, "size", 0) else None)

    # 4. co-gate position + template shape -> fiber groups
    groups = fm.position_shape_merge(pos, templates, mask=mask, dy_um=dy_um,
                                     dlogA=dlogA, dz_um=dz_um, min_cos=min_cos,
                                     probe_y=probe_y, clique=clique)

    # 5. atom -> fiber map (ungrouped atoms keep singleton fibers)
    parent = groups_to_parent(ids, groups)
    if verbose:
        multi = sum(1 for g in groups if len(g) > 1)
        print(f"[refiber] {len(ids)} atoms -> {len(groups)} fibers "
              f"({multi} multi-atom, {len(groups) - multi} singletons)")
    return atoms, parent, groups


def write_triple(atoms, parent, base, group, *, variant="stderiv",
                 out_tag="microfiber", renumber=True, backup=True, dry_run=False,
                 verbose=True):
    """Write .clu/.clc/.clp for the (atoms, parent) hierarchy.  .clc reproduces
    the atoms verbatim; .clu/.clp carry the new fibering.  Overwrites the out_tag
    files in place (with .bak) unless dry_run."""
    h = FiberHierarchy(atoms, parent)
    clu, clc, clp = h.refiberize(renumber=renumber)
    out = {}
    for kind, arr in (("clu", clu), ("clc", clc), ("clp", clp)):
        path = nio.session_path(base, kind, group, variant=variant, tag=out_tag)
        out[kind] = path
        if dry_run:
            continue
        if backup and os.path.exists(path):
            shutil.copy2(path, path + ".bak")
        nio.write_clu_file(path, arr)
        if verbose:
            print(f"  wrote {os.path.basename(path)} ({arr.size} entries)")
    if dry_run and verbose:
        print("  --dry-run: nothing written")
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Re-fiber microfiber atoms (.clc) into fibers by raw-position "
                    "+ template-shape co-gating; write the .clu/.clc/.clp triple.")
    ap.add_argument("base")
    ap.add_argument("group", type=int)
    ap.add_argument("--method", "--variant", dest="variant", default="stderiv",
                    help="method the clu stems from: standard | stderiv | stderiv_C5 (default stderiv)")
    ap.add_argument("--stage", "--tag", dest="tag", default="microfiber",
                    help="post-fiber stage tag of the input atom layer")
    ap.add_argument("--out-stage", "--out-tag", dest="out_tag", default=None,
                    help="output tag (default: same as --tag, i.e. overwrite in place with .bak)")
    ap.add_argument("--session", default=None)
    ap.add_argument("--probe", nargs="+", default=None)
    ap.add_argument("--channels", default=None, help="comma-separated global channel ids")
    ap.add_argument("--nsamp", type=int, default=None)
    ap.add_argument("--nchan", type=int, default=None)
    # merge gates (mirrors fiber_merge.position_shape_merge defaults)
    ap.add_argument("--dy-um", type=float, default=6.0, help="max depth disagreement (um)")
    ap.add_argument("--dlogA", type=float, default=0.25, help="max log-amplitude disagreement")
    ap.add_argument("--dz-um", type=float, default=8.0, help="max distance disagreement (um)")
    ap.add_argument("--min-cos", type=float, default=0.85, help="min template cosine similarity")
    ap.add_argument("--clique", action="store_true",
                    help="require an atom to gate to ALL members of a fiber (stricter; no shape chaining)")
    ap.add_argument("--max-spikes", type=int, default=2000, help="cap per-atom spikes for templates")
    ap.add_argument("--no-renumber", action="store_true")
    ap.add_argument("--no-backup", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    atoms, parent, _ = refiber(
        a.base, a.group, variant=a.variant, tag=a.tag, session=a.session,
        probe=a.probe, channels=a.channels, nsamp=a.nsamp, nchan=a.nchan,
        dy_um=a.dy_um, dlogA=a.dlogA, dz_um=a.dz_um, min_cos=a.min_cos,
        clique=a.clique, max_spikes=a.max_spikes)
    write_triple(atoms, parent, a.base, a.group, variant=a.variant,
                 out_tag=a.out_tag or a.tag, renumber=not a.no_renumber,
                 backup=not a.no_backup, dry_run=a.dry_run)


if __name__ == "__main__":
    main()
