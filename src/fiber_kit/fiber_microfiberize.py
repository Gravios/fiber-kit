#!/usr/bin/env python3
"""fiber_microfiberize.py — lift a flat .clu fiber sort into the microfiber triple.

The fiber<-microfiber hierarchy (see fiber_refiberize) is three sibling files:

    .clc  per-SPIKE child (microfiber) label    -- the ATOM layer
    .clp  per-CHILD parent fiber id             -- the HIERARCHY (clp[c-1] = fiber of child c)
    .clu  per-SPIKE fiber = parent_of(child[i]) -- DERIVED

Klusters' hierarchical mode (and fiber_refiberize) LOAD .clc + .clp and re-derive
.clu.  A flat sort -- any ordinary .clu, e.g. the .refine_linked / .refine_relinked
output of fiber-link, or a hand-curated .clu -- has only the fiber layer and no
.clc/.clp siblings, so it cannot be opened in that mode.  This script synthesises
the two missing files for an existing .clu.

Two lifts:

  IDENTITY (default): each fiber becomes a single microfiber.  child[i] = clu[i],
    parent[fiber] = fiber.  The derived .clu is identical to the input (fiber
    renumbering aside); the operator opens the triple and starts splitting atoms.
    This is the only lift possible from a bare .clu -- no finer structure exists.

  ATOMS (--atoms TAG): use a FINER per-spike sort as the atom layer.  child[i] is
    read from <base>.clu.<variant>.<group>.<TAG> and every atom is parented to the
    fiber that holds the MAJORITY of its spikes (an atom straddling two fibers is
    assigned by majority and reported -- it is never split).  This attaches an
    existing over-clustered / per-chunk-fragment layer under a coarser fiber sort.

Both lifts go through FiberHierarchy, so the triple is correct by construction:
the .clp layout matches the loader (clp[c-1] = fiber of child c) and the nesting
invariant clu == parent_of(clc) holds with no separate purity check.

By default only .clc and .clp are written (the input .clu is left untouched); pass
--write-clu, or a differing --out-tag, to (re)emit the derived .clu as well.
"""
import argparse
import os
import shutil
import numpy as np
try:
    from . import neuro_io as nio
    from .fiber_refiberize import FiberHierarchy
except ImportError:                       # running as a loose script
    import neuro_io as nio
    from fiber_refiberize import FiberHierarchy


def lift_identity(clu):
    """Each fiber -> one microfiber.  child = clu; parent = identity over fibers >= 1.
    Reserve ids 0 (noise) and 1 (artefact) carry through unchanged."""
    clu = np.asarray(clu, dtype=np.int64)
    parent = {int(f): int(f) for f in np.unique(clu) if int(f) >= 1}
    return clu.copy(), parent


def lift_atoms(clu, atoms):
    """child = atoms; parent[atom] = the fiber holding the majority of that atom's
    spikes.  Returns (child, parent, n_straddle): n_straddle counts atoms whose
    spikes were not unanimous on a single fiber (assigned by majority, kept whole)."""
    clu = np.asarray(clu, dtype=np.int64)
    atoms = np.asarray(atoms, dtype=np.int64)
    if atoms.size != clu.size:
        raise SystemExit(f"[microfiberize] atom layer has {atoms.size} spikes, "
                         f"clu has {clu.size} -- they must index the same .res")
    parent = {}
    n_straddle = 0
    for a in np.unique(atoms):
        if int(a) < 1:
            continue
        fibs = clu[atoms == a]
        vals, cnts = np.unique(fibs, return_counts=True)
        parent[int(a)] = int(vals[int(cnts.argmax())])
        if vals.size > 1:
            n_straddle += 1
    return atoms.copy(), parent, n_straddle


def main():
    ap = argparse.ArgumentParser(
        description="Lift a flat .clu fiber sort into the microfiber triple (.clc atom "
                    "layer + .clp child->parent map) so it opens in the Klusters "
                    "hierarchical / fiber-refiberize machinery.")
    ap.add_argument("base", help="session base path (no extension)")
    ap.add_argument("group", help="electrode group id")
    ap.add_argument("--variant", default="stderiv", help="feature space (default stderiv)")
    ap.add_argument("--in-tag", default="refine_linked",
                    help="stage tag of the input .clu (default refine_linked)")
    ap.add_argument("--out-tag", default=None,
                    help="stage tag for the output triple (default: same as --in-tag)")
    ap.add_argument("--atoms", default=None,
                    help="stage tag of a FINER per-spike .clu to use as the atom (.clc) layer; "
                         "omit for the identity lift (each fiber = one microfiber)")
    ap.add_argument("--write-clu", action="store_true",
                    help="also (re-)write the derived .clu; by default only .clc/.clp are written "
                         "(the input .clu is left untouched unless --out-tag differs)")
    ap.add_argument("--no-renumber", action="store_true",
                    help="keep original fiber ids (leave gaps) instead of compacting to consecutive")
    ap.add_argument("--no-backup", action="store_true", help="do not write .bak copies of overwritten files")
    ap.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    a = ap.parse_args()

    out_tag = a.out_tag if a.out_tag is not None else a.in_tag
    clu_path = nio.session_path(a.base, "clu", a.group, variant=a.variant, tag=a.in_tag)
    _, clu = nio.read_clu_file(clu_path)
    print(f"  read {os.path.basename(clu_path)}  ({clu.size:,} spikes, "
          f"{np.unique(clu[clu > 1]).size} fibers)")

    if a.atoms:
        atoms_path = nio.session_path(a.base, "clu", a.group, variant=a.variant, tag=a.atoms)
        _, atoms = nio.read_clu_file(atoms_path)
        child, parent, n_straddle = lift_atoms(clu, atoms)
        print(f"  atom layer {os.path.basename(atoms_path)}  "
              f"({np.unique(child[child > 0]).size} atoms)")
        if n_straddle:
            print(f"  note: {n_straddle} atom(s) straddled >1 fiber -- parented by majority vote")
    else:
        child, parent = lift_identity(clu)
        print(f"  identity lift: {len(parent)} fibers -> {len(parent)} single-atom microfibers")

    h = FiberHierarchy(child, parent)
    clu_out, clc_out, clp_out = h.refiberize(renumber=not a.no_renumber)

    out = {k: nio.session_path(a.base, k, a.group, variant=a.variant, tag=out_tag)
           for k in ("clu", "clc", "clp")}
    write_clu = a.write_clu or (out_tag != a.in_tag)
    targets = [("clc", clc_out), ("clp", clp_out)] + ([("clu", clu_out)] if write_clu else [])
    nfib = np.unique(clu_out[clu_out > 1]).size
    print(f"  -> {nfib} fibers over {len(parent)} children ({child.size:,} spikes)")
    if a.dry_run:
        print("  --dry-run: nothing written")
        return
    for kind, arr in targets:
        path = out[kind]
        if not a.no_backup and os.path.exists(path):
            shutil.copy2(path, path + ".bak")
        nio.write_clu_file(path, arr)
        print(f"  wrote {os.path.basename(path)}  ({arr.size:,} entries)")


if __name__ == "__main__":
    main()
