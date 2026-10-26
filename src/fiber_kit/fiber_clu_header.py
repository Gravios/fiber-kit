#!/usr/bin/env python3
"""fiber-clu-header -- check or repair the int32 header of .clu / .clc / .clp files.

The header of a cluster-id file is the number of distinct ids in it; a .clp's is its number of
entries (neuro_io.clu_header, after the neurosuite-3 format pages clu.md, clc.md and clp.md).  Until
0.57.3 fiber-kit wrote the highest id + 1 instead.  For a gap-free sort the two are the same number.
A chunk run (FK_CHUNKS) is different: it parks the spikes of the chunks it did not compute in the
reserved clusters 16777116 + chunk, so its .clu files got headers near 16.8 million, and Klusters
refuses any header above 65536:

    Invalid or missing .clu header (nClusters=16777136)

This rewrites the 4 header bytes of each file in place.  The ids are not touched.

    fiber-clu-header FILE...            # fix every header that is off
    fiber-clu-header --check FILE...    # report only; exit 1 when a header is off

Skipped, with a message: names not of the form <base>.<clu|clc|clp>[.<variant>].<elec>[.<tag>]
(so a shell glob that also matches a stage's .res, .spk, .fet or .units.npz is safe), text .clu
files, and files whose size is not a whole number of int32 values.
"""
import argparse
import os
import sys

import numpy as np

from . import neuro_io as nio

KINDS = ("clu", "clc", "clp")


def file_kind(path):
    """'clu', 'clc' or 'clp' when `path` is named <base>.<kind>[.<variant>].<elec>[.<tag>], else None:
    after the kind token come at most one token before the electrode number and at most one after."""
    t = os.path.basename(os.fspath(path)).split(".")
    for k in range(1, len(t)):
        if t[k] not in KINDS:
            continue
        rest = t[k + 1:]
        e = next((i for i, x in enumerate(rest) if x.isdigit()), None)
        if e is not None and e <= 1 and len(rest) - e - 1 <= 1:
            return t[k]
    return None


def _is_text(head):
    """A text .clu: digits, signs and whitespace only.  A binary one holds NUL bytes, since every int32
    below 2**24 -- every header and id fiber-kit writes, the reserved clusters included -- has one."""
    return bool(head) and b"\0" not in head and not head.strip(b"0123456789-+ \t\r\n")


def inspect(path):
    """(kind, header on disk, header by the rule, number of ids) of a binary cluster-id file.
    Raises ValueError saying why the file is skipped."""
    kind = file_kind(path)
    if kind is None:
        raise ValueError("not named <base>.<clu|clc|clp>[.<variant>].<elec>[.<tag>]")
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        head = fh.read(4096)
    if _is_text(head):
        raise ValueError("a text .clu (fiber-kit writes binary ones); left as it is")
    if size < 4 or size % 4:
        raise ValueError(f"{size:,} bytes is not an int32 header plus int32 ids")
    raw = np.fromfile(path, dtype=nio.CLU_DTYPE)
    return kind, int(raw[0]), nio.clu_header(path, raw[1:], kind=kind), int(raw.size - 1)


def write_header(path, header):
    """Write `header` over the first 4 bytes of `path`; the ids after it are not rewritten."""
    with open(path, "r+b") as fh:
        fh.write(np.array([header], nio.CLU_DTYPE).tobytes())
        fh.flush()
        os.fsync(fh.fileno())


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="fiber-clu-header",
        description="Check or repair the int32 header of .clu/.clc/.clp files: the number of distinct "
                    "ids (a .clp: its number of entries).  fiber-kit before 0.57.3 wrote the highest id + 1, "
                    "which Klusters refuses for a chunk run's files.  Only the 4 header bytes are rewritten.")
    ap.add_argument("files", nargs="+", help=".clu / .clc / .clp files (other names are skipped)")
    ap.add_argument("--check", action="store_true",
                    help="report only, write nothing; exit 1 when a header is off")
    a = ap.parse_args(argv)
    off = fixed = skipped = 0
    for p in a.files:
        try:
            kind, have, want, n = inspect(p)
        except (OSError, ValueError) as e:
            print(f"  skip  {p}: {e}")
            skipped += 1
            continue
        what = "entries" if kind == "clp" else "distinct ids"
        if have == want:
            print(f"  ok    {p}: header {have:,} ({what}, {n:,} rows)")
            continue
        off += 1
        if a.check:
            print(f"  OFF   {p}: header {have:,}, should be {want:,} ({what}, {n:,} rows)")
            continue
        write_header(p, want)
        fixed += 1
        print(f"  fixed {p}: header {have:,} -> {want:,} ({what}, {n:,} rows)")
    verb = "off" if a.check else "fixed"
    print(f"fiber-clu-header: {off if a.check else fixed} {verb}, {skipped} skipped, "
          f"{len(a.files) - off - skipped} already right")
    return 1 if (a.check and off) else 0


if __name__ == "__main__":
    sys.exit(main())
