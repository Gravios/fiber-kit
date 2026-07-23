#  fiber_refiberize.py — edit the fiber<-microfiber hierarchy, then regenerate
#  the aligned .clu / .clc / .clp triple from the edited child->parent map.
#
#  The three files written by the microfiber decomposition are NOT three
#  independent labelings — two of them are derived from one editable map:
#
#      .clc  per-SPIKE child (microfiber) label   -- the ATOM layer, stable
#      .clp  per-CHILD parent fiber id            -- the HIERARCHY, editable
#      .clu  per-SPIKE fiber label = parent_of(child[i])   -- DERIVED
#
#  So a fiber is just "the set of children that currently point at it", and
#  every structural edit is a change to child->parent.  Re-deriving .clu (and
#  re-emitting .clp) from .clc + the edited map is "refiberize".  Because .clu
#  is rebuilt as parent_of(child[i]), the nesting invariant (each child belongs
#  to exactly one fiber, every spike of a child shares that fiber) holds BY
#  CONSTRUCTION — there is no edit expressible here that can de-align the files
#  or split a child across two fibers.  That is what makes these merges "safe":
#  not a statistical purity test, but a structural guarantee.
#
#  Operations (all edit child->parent unless noted):
#    merge_parents(keep, *absorb)   fold absorbed fibers' children under `keep`
#                                   == the union of their children (your Q: yes)
#    promote_child(child)           detach a child from its parent and make it
#                                   its own fiber (your "remove child" -> a new
#                                   top-level cluster in .clu)
#    move_child(child, fiber)       re-parent a single child
#    split_parent(fiber, children)  promote a subset of a fiber's children to a
#                                   new fiber (the inverse of a merge)
#    dissolve_parent(fiber)         promote every child of a fiber (un-fiberize
#                                   a unit back into its microfibers)
#    merge_children(keep, *absorb)  ATOM-level: relabel .clc so two microfibers
#                                   become one child (this DOES change .clc)
#
#  Either drive it from Python (FiberHierarchy) or from the CLI with an --ops
#  script; with no ops it simply refiberizes (rebuild .clu from a hand-edited
#  .clp), which is the "change the relationships, refiberize later" path.

import argparse
import os
import shutil
import numpy as np

try:
    from . import neuro_io as nio
except ImportError:                       # running as a loose script
    import neuro_io as nio


# ── core model ───────────────────────────────────────────────────────────────

class FiberHierarchy:
    """Editable fiber<-microfiber hierarchy over a fixed per-spike child layer.

    child   : (nSpikes,) int64  per-spike microfiber id (.clc); 0 = noise/unassigned
    parent  : dict {child_id:int -> fiber_id:int}  child->fiber (.clp), child_id >= 1
    """

    def __init__(self, child, parent):
        self.child = np.asarray(child, dtype=np.int64)
        self.parent = {int(c): int(f) for c, f in parent.items() if int(c) >= 1}
        # next free fiber id for promotions; kept above every fiber AND child id
        # so a promoted child can never collide before the final renumber.
        fibers = set(self.parent.values())
        max_child = int(self.child.max()) if self.child.size else 0
        self._next = max([1, max_child, *fibers]) + 1

    # -- constructors ----------------------------------------------------------

    @classmethod
    def load(cls, base, group, *, variant="stderiv", tag="microfiber",
             clc=None, clp=None):
        """Load from the .clc + .clp siblings (paths derived from base/group/
        variant/tag unless given explicitly)."""
        clc_path = clc or nio.session_path(base, "clc", group, variant=variant, tag=tag)
        clp_path = clp or nio.session_path(base, "clp", group, variant=variant, tag=tag)
        _, child = nio.read_clu_file(clc_path)
        _, parent_arr = nio.read_clu_file(clp_path)        # parent_arr[c-1] = fiber of child c
        parent = {c + 1: int(parent_arr[c]) for c in range(parent_arr.size)
                  if int(parent_arr[c]) > 0}
        self = cls(child, parent)
        self._paths = dict(base=base, group=group, variant=variant, tag=tag,
                           clc=clc_path, clp=clp_path)
        return self

    # -- queries ---------------------------------------------------------------

    def fibers(self):
        return sorted(set(self.parent.values()))

    def children_of(self, fiber):
        return sorted(c for c, f in self.parent.items() if f == int(fiber))

    def parent_of(self, child):
        return self.parent.get(int(child), 0)

    def _fresh_fiber(self):
        f = self._next
        self._next += 1
        return f

    # -- hierarchy edits (change child->parent only) ---------------------------

    def merge_parents(self, keep, *absorb):
        """Fold the children of every `absorb` fiber under `keep`.  This is the
        union of their children: keep now owns keep's children PLUS all the
        absorbed children, and the absorbed fibers vanish on refiberize."""
        keep = int(keep)
        absorb = {int(a) for a in absorb} - {keep}
        moved = 0
        for c, f in self.parent.items():
            if f in absorb:
                self.parent[c] = keep
                moved += 1
        return moved

    def promote_child(self, child, new_fiber=None):
        """Detach `child` from its parent and make it its own fiber (a new
        top-level cluster in .clu).  Returns the new fiber id."""
        child = int(child)
        if child not in self.parent:
            raise KeyError(f"child {child} not in hierarchy")
        f = int(new_fiber) if new_fiber is not None else self._fresh_fiber()
        self.parent[child] = f
        return f

    def move_child(self, child, fiber):
        """Re-parent a single child onto an existing (or new) fiber."""
        child = int(child)
        if child not in self.parent:
            raise KeyError(f"child {child} not in hierarchy")
        self.parent[child] = int(fiber)

    def split_parent(self, fiber, children, into=None):
        """Promote a subset of `fiber`'s children to a separate fiber (inverse
        of a merge).  All listed children land on one new fiber unless `into`
        is given.  Returns the target fiber id."""
        target = int(into) if into is not None else self._fresh_fiber()
        owned = set(self.children_of(fiber))
        for c in children:
            c = int(c)
            if c not in owned:
                raise ValueError(f"child {c} is not under fiber {fiber}")
            self.parent[c] = target
        return target

    def dissolve_parent(self, fiber):
        """Promote every child of `fiber` to its own fiber (un-fiberize a unit
        back into its constituent microfibers).  Returns the new fiber ids."""
        out = []
        for c in self.children_of(fiber):
            out.append(self.promote_child(c))
        return out

    # -- atom edit (changes .clc) ----------------------------------------------

    def merge_children(self, keep, *absorb):
        """Collapse microfibers `absorb` into `keep` at the ATOM level: relabels
        .clc so those spikes carry `keep`, and drops the absorbed child ids from
        the map.  Unlike the others this rewrites .clc, so refiberize emits a
        changed .clc as well.  `keep` and `absorb` must share a parent (else the
        merge would move spikes across fibers — refused)."""
        keep = int(keep)
        absorb = [int(a) for a in absorb if int(a) != keep]
        pk = self.parent.get(keep)
        for a in absorb:
            if self.parent.get(a) != pk:
                raise ValueError(
                    f"child {a} (parent {self.parent.get(a)}) and {keep} "
                    f"(parent {pk}) differ — cross-fiber atom merge refused")
        if absorb:
            absorb_set = np.isin(self.child, absorb)
            self.child[absorb_set] = keep
            for a in absorb:
                self.parent.pop(a, None)
        return int(np.count_nonzero(absorb_set)) if absorb else 0

    # -- refiberize ------------------------------------------------------------

    def refiberize(self, renumber=True):
        """Rebuild (clu, clc, clp) from the current child layer + child->parent.

        clu[i] = parent_of(child[i]); child 0 -> fiber 0; a child with no parent
        entry (an orphan, e.g. left by an atom edit) -> fiber 0 with a warning.
        With `renumber`, fibers are compacted to consecutive ids (0,1 preserved
        as artefact/noise) so the output .clu has no gaps.  Returns numpy arrays
        (clu_per_spike, clc_per_spike, clp_per_child)."""
        max_child = int(self.child.max()) if self.child.size else 0
        lut = np.zeros(max_child + 1, dtype=np.int64)      # lut[child_id] = fiber (0 if unmapped)
        for c, f in self.parent.items():
            if c <= max_child:
                lut[c] = f
        clipped = np.clip(self.child, 0, max_child)
        clu = lut[clipped]
        clu[self.child <= 0] = 0

        orphan_children = sorted(int(c) for c in np.unique(self.child[self.child > 0])
                                 if int(c) not in self.parent)
        if orphan_children:
            print(f"  warning: {len(orphan_children)} child id(s) have no parent "
                  f"-> fiber 0 (e.g. {orphan_children[:5]})")

        parent_out = dict(self.parent)
        if renumber:
            survivors = sorted(set(int(x) for x in np.unique(clu)) - {0, 1})
            remap = {0: 0, 1: 1}
            for nxt, f in enumerate(survivors, start=2):
                remap[f] = nxt
            rlut = np.zeros(int(clu.max()) + 1, dtype=np.int64)
            for f, g in remap.items():
                if f <= clu.max():
                    rlut[f] = g
            clu = rlut[clu]
            parent_out = {c: remap.get(f, f) for c, f in self.parent.items()}

        clp = np.zeros(max_child, dtype=np.int64)          # clp[c-1] = fiber of child c
        for c, f in parent_out.items():
            if 1 <= c <= max_child:
                clp[c - 1] = f
        return clu, self.child.copy(), clp

    # -- validate + save -------------------------------------------------------

    def validate(self, clu=None):
        """Structural checks. Returns list of problem strings ([] == clean)."""
        problems = []
        clu = clu if clu is not None else self.refiberize(renumber=False)[0]
        if clu.size != self.child.size:
            problems.append(f"clu/clc length mismatch: {clu.size} vs {self.child.size}")
        # nesting: every spike of a child must share one fiber (guaranteed by
        # construction, but verify in case clc/clp were edited out of band)
        for c in np.unique(self.child[self.child > 0]):
            fibers_here = np.unique(clu[self.child == c])
            if fibers_here.size > 1:
                problems.append(f"child {int(c)} spans fibers {fibers_here.tolist()}")
        return problems

    def save(self, base=None, group=None, *, variant="stderiv", tag="microfiber",
             renumber=True, backup=True, dry_run=False):
        """Refiberize and write the .clu/.clc/.clp triple.  Defaults to the load
        paths; pass base/group/variant/tag (or a new tag) to redirect."""
        p = getattr(self, "_paths", {})
        base = base or p.get("base")
        group = group if group is not None else p.get("group")
        variant = variant or p.get("variant", "stderiv")
        clu, clc, clp = self.refiberize(renumber=renumber)
        out = {
            "clu": nio.session_path(base, "clu", group, variant=variant, tag=tag),
            "clc": nio.session_path(base, "clc", group, variant=variant, tag=tag),
            "clp": nio.session_path(base, "clp", group, variant=variant, tag=tag),
        }
        nfib = np.unique(clu[clu > 1]).size
        print(f"  refiberized: {nfib} fibers over {len(self.parent)} children "
              f"({self.child.size} spikes)")
        if dry_run:
            print("  --dry-run: nothing written")
            return out
        for kind, arr in (("clu", clu), ("clc", clc), ("clp", clp)):
            path = out[kind]
            if backup and os.path.exists(path):
                shutil.copy2(path, path + ".bak")
            nio.write_clu_file(path, arr)
            print(f"  wrote {os.path.basename(path)}  ({arr.size} entries)")
        return out


# ── ops-script driver ────────────────────────────────────────────────────────

def apply_ops(h, ops_path):
    """Apply a whitespace-tokenized edit script (one op per line, # comments).
        merge_parents   <keep> <absorb...>
        promote_child   <child> [new_fiber]
        move_child      <child> <fiber>
        split_parent    <fiber> <child...> [into <fiber>]
        dissolve_parent <fiber>
        merge_children  <keep> <absorb...>
    """
    n = 0
    with open(ops_path) as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            tok = line.split()
            op, args = tok[0], [int(x) for x in tok[1:] if x.lstrip("-").isdigit()]
            if op == "merge_parents":
                h.merge_parents(args[0], *args[1:])
            elif op == "promote_child":
                h.promote_child(args[0], args[1] if len(args) > 1 else None)
            elif op == "move_child":
                h.move_child(args[0], args[1])
            elif op == "split_parent":
                into = None
                if "into" in tok:
                    into = int(tok[tok.index("into") + 1])
                    args = [a for a in args if a != into]
                h.split_parent(args[0], args[1:], into=into)
            elif op == "dissolve_parent":
                h.dissolve_parent(args[0])
            elif op == "merge_children":
                h.merge_children(args[0], *args[1:])
            else:
                raise ValueError(f"{ops_path}:{lineno}: unknown op '{op}'")
            n += 1
    print(f"  applied {n} op(s) from {ops_path}")
    return n


def main():
    ap = argparse.ArgumentParser(
        prog="fiber-refiberize",
        description="Edit the fiber<-microfiber hierarchy and regenerate the "
                    "aligned .clu/.clc/.clp from the child->parent map.")
    ap.add_argument("base", help="session base path (no extension)")
    ap.add_argument("group", help="electrode group id")
    ap.add_argument("--method", "--variant", dest="variant", default="stderiv",
                    help="method the clu stems from: standard | stderiv | stderiv_C5 (default stderiv)")
    ap.add_argument("--stage", "--tag", dest="tag", default="microfiber",
                    help="post-fiber stage tag of the input triple (default microfiber)")
    ap.add_argument("--out-stage", "--out-tag", dest="out_tag", default=None,
                    help="post-fiber stage tag for the output triple (default: overwrite --stage)")
    ap.add_argument("--ops", default=None, help="edit-script file to apply before refiberizing")
    ap.add_argument("--no-renumber", action="store_true", help="keep original fiber ids (leave gaps)")
    ap.add_argument("--no-backup", action="store_true", help="do not write .bak copies")
    ap.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    a = ap.parse_args()

    print(f"\u25b8 fiber-refiberize  {os.path.basename(a.base)} group {a.group} "
          f"({a.variant}.{a.tag})")
    h = FiberHierarchy.load(a.base, a.group, variant=a.variant, tag=a.tag)
    print(f"  loaded {len(h.parent)} children under {len(h.fibers())} fibers "
          f"({h.child.size} spikes)")
    if a.ops:
        apply_ops(h, a.ops)
    problems = h.validate()
    if problems:
        print("  VALIDATION PROBLEMS:")
        for p in problems:
            print(f"    - {p}")
    h.save(variant=a.variant, tag=a.out_tag or a.tag,
           renumber=not a.no_renumber, backup=not a.no_backup, dry_run=a.dry_run)


if __name__ == "__main__":
    main()
