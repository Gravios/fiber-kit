"""fiber-relink maps fibers to clusters through the .fibers' clu_id, not gid + 1.

Fibers gid 0, 1, 2 written as clusters 2, 3, 5 (fibers start at 2; gid 2's cluster moved down when a
fiber vanished in the hierarchy save).  rewrite_clu must merge exactly the clusters of the gids the
relink joined; the pre-0.51 assumption (gid = cluster - 1) reads cluster 2 as gid 1 and merges the wrong
pair -- the negative control.  A .fibers without clu_id falls back to gid + 1 with a warning.
"""
import contextlib
import io
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from fiber_kit import fiber_relink as frk   # noqa: E402
from fiber_kit import neuro_io as nio       # noqa: E402


def main():
    ok = 0

    def check(name, cond):
        nonlocal ok
        assert cond, name
        ok += 1
        print(f"  ok  {name}")

    z = {"gid": np.array([0, 1, 2, 0]), "clu_id": np.array([2, 3, 5, 2])}
    g2c, c2g = frk.gid_clu_maps(z)
    check("gid_clu_maps reads clu_id", g2c == {0: 2, 1: 3, 2: 5} and c2g == {2: 0, 3: 1, 5: 2})
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        g2c_old, _ = frk.gid_clu_maps({"gid": np.array([0, 1, 2])})
    check("a .fibers without clu_id falls back to gid + 1, with a warning",
          g2c_old == {0: 1, 1: 2, 2: 3} and "pre-0.51" in buf.getvalue())
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "in.clu"); dst = os.path.join(td, "out.clu")
        ids = np.array([0, 2, 2, 3, 5, 5, 3, 0])
        nio.write_clu_file(src, ids)
        oldgid2unit = {0: 0, 1: 1, 2: 0}                   # the relink joined gid 0 and gid 2
        with contextlib.redirect_stdout(io.StringIO()):
            frk.rewrite_clu(src, dst, oldgid2unit, c2g)
        _, out = nio.read_clu_file(dst)
        out = np.asarray(out)
        check("rewrite_clu merges clusters 2 and 5 (gids 0 and 2) and keeps 3 apart",
              out[1] == out[4] and out[1] != out[3] and out[0] == 0)
        check("units are numbered from 2: none lands in the noise cluster 1", out[1:7].min() >= 2)
        src2 = os.path.join(td, "in2.clu"); dst2 = os.path.join(td, "out2.clu")
        nio.write_clu_file(src2, np.array([5, 1, 5, 7, 1, 0]))
        with contextlib.redirect_stdout(io.StringIO()):
            frk.rewrite_clu(src2, dst2, {})
        _, o2 = nio.read_clu_file(dst2)
        check("the noise (1) and artifact (0) bins pass through; units 5 and 7 become 2 and 3",
              list(np.asarray(o2)) == [2, 1, 2, 3, 1, 0])
        frk.rewrite_clu(src, dst, oldgid2unit)             # the pre-0.51 reading: gid = cluster - 1
        _, wrong = nio.read_clu_file(dst)
        wrong = np.asarray(wrong)
        check("NEGATIVE CONTROL: reading gid = cluster - 1 merges the wrong clusters",
              not (wrong[1] == wrong[4] and wrong[1] != wrong[3]))
    print(f"test_relink_ids: {ok} checks passed")


if __name__ == "__main__":
    main()
