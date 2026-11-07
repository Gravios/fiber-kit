#!/usr/bin/env python3
"""fiber_subsession -- a SUB-SESSION: the spikes of chosen clusters of one group as a session of their own,
so that any fiber-kit plan runs on them unmodified (`fk run <group> --session <sub> --final-clu ...`).

This is the Klusters <-> fiber-kit recluster contract (docs/pipeline.md "Reclustering a selection"):
Klusters writes the same set of files from its in-memory selection (KlustersDoc::createReclusterSubsession)
and reads the final .clu back by spike order; this module is the reference implementation, the CLI for
scripting the same thing from a .clu on disk, and what the tests check the contract against.

The sub-session `<out>` lives in the session's directory (so relative paths in the yaml keep working):

    <out>.res[.<v>].<g>        the selected spikes' times -- a ROW SUBSET of every <base>.res[.<v>].<g>;
                               the times stay ABSOLUTE, so the real .fil is valid for them
    <out>.spk.<v>.<g>          a row subset of every <base>.spk.<v>.<g> on disk (every waveform variant)
    <out>.fet.<v>.<g>          a row subset of every <base>.fet.<v>.<g> on disk (every feature variant)
    <out>.clu.<v>.<g>[.<tag>]  the SEED: the selected spikes' current ids (a plan step with `in:` can
                               start from the curation); written for the clu variant given
    <out>.yaml / .yml / .xml   copies of the session's parameter files (probe geometry, rate, sdiffPairs)
    <out>.fil, .dat, .nst.*.<g>, .pca.*.<g>, .eap.*.<g>, .tcl.*.<g>
                               SYMLINKS to the session's -- nothing large is copied; realign --reextract,
                               deconv, the consensus envelope and refeaturize read the real recording

Row order is the spike-time order of the selection (ascending feature row), the same in every file, so the
sub-session's `.res` is sorted as the stages require and the final `.clu` maps back to the selection by
position.  Stages relabel spikes, they never drop one, so the final .clu has exactly len(rows) ids.

    fk subsession export <base> <group> --clu <path> --clusters 12 15 33 [--out <out>] [--variant V]
    fk subsession clean  <out> <group>              # remove everything the sub-session and its run wrote
"""
import argparse
import glob
import os
import re
import shutil
import sys

import numpy as np

try:
    from . import neuro_io as nio
except ImportError:
    import neuro_io as nio

LINKED = ("fil", "dat", "nst", "pca", "eap", "tcl")        # per-session / per-group files reached by symlink
COPIED_EXT = (".yaml", ".yml", ".xml")                      # parameter files: copied (small; edited by stages? no)


def _group_files(base, kind, group):
    """<base>.<kind>.<g> and <base>.<kind>.<variant>.<g> on disk (not stage-tagged), with their variant."""
    out = []
    g = str(int(group))
    for p in sorted(glob.glob(f"{glob.escape(base)}.{kind}.*")):
        tail = p[len(base) + len(kind) + 2:]               # after "<base>.<kind>."
        toks = tail.split(".")
        if toks[-1] != g:
            continue                                        # another group, or a stage tag after the group
        variant = ".".join(toks[:-1])
        if variant and variant.replace("_", "").replace("-", "").isdigit():
            continue
        out.append((p, variant))
    return out


def _rows_of(clu_path, clusters, n_spikes):
    _, ids = nio.read_clu_file(clu_path, n_spikes)
    ids = np.asarray(ids)
    want = np.isin(ids, np.asarray(list(clusters), dtype=ids.dtype))
    return np.flatnonzero(want), ids


def export(base, group, rows, out, seed_ids=None, seed_variant="", seed_tag="", links=True, log=None):
    """Write the sub-session `out` for the 0-based spike rows `rows` of group `group` of `base`.
    `seed_ids`: the rows' current cluster ids (len(rows)) for the seed .clu, named with `seed_variant` /
    `seed_tag`.  Returns the list of paths written or linked."""
    log = log or (lambda s: None)
    rows = np.asarray(rows, dtype=np.int64)
    if rows.size == 0:
        raise ValueError("no spikes selected")
    rows = np.unique(rows)                                  # sorted ascending = time order; no duplicates
    written = []
    n_ref = None

    def put(p):
        written.append(p); log(os.path.basename(p))

    # .res (every variant on disk)
    for p, variant in _group_files(base, "res", group):
        t = np.fromfile(p, dtype=nio.RES_DTYPE)
        n_ref = n_ref or t.size
        if t.size != n_ref:
            raise ValueError(f"{p}: {t.size} spikes, {n_ref} in the first .res -- the group's files disagree")
        if rows[-1] >= t.size:
            raise ValueError(f"row {int(rows[-1])} is beyond the {t.size} spikes of {os.path.basename(p)}")
        q = nio.session_path(out, "res", group, variant=variant)
        t[rows].astype(nio.RES_DTYPE).tofile(q); put(q)
    if n_ref is None:
        raise FileNotFoundError(f"no {base}.res[.<variant>].{group}")
    # .spk (every variant): rows of nsamp*nchan int16, the per-spike size from the file size
    for p, variant in _group_files(base, "spk", group):
        size = os.path.getsize(p)
        if size % (2 * n_ref):
            raise ValueError(f"{p}: {size} bytes is not {n_ref} spikes of int16 rows")
        per = size // (2 * n_ref)
        mm = np.memmap(p, dtype=nio.SPK_DTYPE, mode="r", shape=(n_ref, per))
        q = nio.session_path(out, "spk", group, variant=variant)
        np.ascontiguousarray(mm[rows]).tofile(q); put(q)
    # .fet (every variant): int32 header + int64 rows
    for p, variant in _group_files(base, "fet", group):
        f = nio.read_fet_file(p)
        if not f.ok or f.n_spikes != n_ref:
            raise ValueError(f"{p}: {f.n_spikes} rows, the .res has {n_ref}")
        q = nio.session_path(out, "fet", group, variant=variant)
        nio.write_fet_file(q, f.values[rows]); put(q)
    # the seed .clu
    if seed_ids is not None:
        seed = np.asarray(seed_ids, dtype=np.int64)
        if seed.size != rows.size:
            raise ValueError(f"seed ids: {seed.size} for {rows.size} rows")
        q = nio.session_path(out, "clu", group, variant=seed_variant)
        nio.write_clu_file(q, seed); put(q)
        if seed_tag:
            q = nio.session_path(out, "clu", group, variant=seed_variant, tag=seed_tag)
            nio.write_clu_file(q, seed); put(q)
    # parameter files: copies
    for ext in COPIED_EXT:
        if os.path.isfile(base + ext):
            shutil.copyfile(base + ext, out + ext); put(out + ext)
    # the recording and the per-group models: symlinks (relative, within the directory)
    if links:
        for kind in LINKED:
            cands = [base + "." + kind] if kind in ("fil", "dat") else [p for p, _ in _group_files(base, kind, group)]
            for p in cands:
                if not os.path.exists(p):
                    continue
                q = out + p[len(base):]
                if os.path.lexists(q):
                    os.remove(q)
                os.symlink(os.path.basename(p), q); put(q)
    return written


def clean(out, group=None):
    """Remove every file of the sub-session `out` and everything a run wrote under it: all entries of its
    directory whose name starts with `<out>.`.  Returns the number removed."""
    d, name = os.path.split(out)
    n = 0
    for p in glob.glob(os.path.join(glob.escape(d or "."), glob.escape(name) + ".*")):
        if os.path.islink(p) or os.path.isfile(p):
            os.remove(p); n += 1
        elif os.path.isdir(p):
            shutil.rmtree(p); n += 1
    return n


def default_out(base, clusters):
    ids = "-".join(str(c) for c in clusters)
    return f"{base}--{ids}--sub"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fk subsession",
                                 description="Export the spikes of chosen clusters as a sub-session a fiber-kit "
                                             "plan runs on unmodified, or remove one.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export", help="write <out>.res/.spk/.fet/.clu (+ parameter copies, links)")
    ex.add_argument("base", help="session base path (the file prefix)")
    ex.add_argument("group", type=int)
    ex.add_argument("--clu", required=True, help="the .clu the clusters are read from (aligned to the group's .res)")
    ex.add_argument("--clusters", type=int, nargs="+", required=True, help="cluster ids to export")
    ex.add_argument("--out", help="sub-session base (default <base>--<ids>--sub, in the session directory)")
    ex.add_argument("--variant", default="", help="variant token for the seed .clu (default: read off --clu's name)")
    ex.add_argument("--seed-tag", default="", help="also write the seed as <out>.clu.<v>.<g>.<tag>")
    ex.add_argument("--no-links", action="store_true", help="do not symlink .fil/.dat/.nst/.pca")
    ex.add_argument("--force", action="store_true", help="overwrite an existing sub-session of that name")
    ex.add_argument("-q", "--quiet", action="store_true")
    cl = sub.add_parser("clean", help="remove the sub-session and everything a run wrote under it")
    cl.add_argument("out")
    cl.add_argument("group", nargs="?", type=int)
    a = ap.parse_args(argv)
    if a.cmd == "clean":
        n = clean(a.out, a.group)
        print(f"fk subsession clean: removed {n} file(s) of {os.path.basename(a.out)}")
        return 0
    res = _group_files(a.base, "res", a.group)
    if not res:
        print(f"fk subsession: no {a.base}.res[.<variant>].{a.group}", file=sys.stderr); return 2
    n = np.fromfile(res[0][0], dtype=nio.RES_DTYPE).size
    rows, ids = _rows_of(a.clu, a.clusters, n)
    if rows.size == 0:
        print(f"fk subsession: no spike of clusters {a.clusters} in {a.clu}", file=sys.stderr); return 2
    variant = a.variant
    if not variant:
        m = re.search(r"\.clu\.(.+?)\." + str(a.group) + r"(\.|$)", os.path.basename(a.clu))
        variant = m.group(1) if m else ""
    out = a.out or default_out(a.base, a.clusters)
    if any(True for _ in glob.glob(glob.escape(out) + ".*")) and not a.force:
        print(f"fk subsession: {os.path.basename(out)}.* exists (--force overwrites, `fk subsession clean` removes)",
              file=sys.stderr); return 1
    if a.force:
        clean(out)
    log = (lambda s: None) if a.quiet else (lambda s: print("  " + s))
    try:
        files = export(a.base, a.group, rows, out, seed_ids=ids[rows], seed_variant=variant, seed_tag=a.seed_tag,
                       links=not a.no_links, log=log)
    except (ValueError, FileNotFoundError, OSError) as e:
        print(f"fk subsession: {e}", file=sys.stderr); return 2
    print(f"fk subsession: {rows.size:,} spikes of clusters {a.clusters} -> {out} ({len(files)} files).  "
          f"Run:  fk run {a.group} --session {out} --final-clu {out}.clu.{a.group}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
