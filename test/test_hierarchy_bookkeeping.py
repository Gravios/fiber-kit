"""Hierarchy and spike-set bookkeeping.

refiberize: a fiber that has no spikes left has no place among the compacted cluster ids, so its (empty)
children are parented to 0.  Keeping the old id -- what refiberize did until 0.51 -- aliases whichever
live fiber was renumbered onto it (the negative control below), and leaves .clp ids above the .clu's
largest (6 such on sirotaA-jg-000005-20120316 group 6).

apply_spike_keep (fiber-refine --dedup): the .clc is subset with its .clu (it was left at the old
count), the realign offsets follow the base spike set, and a deconvolution manifest and a realign pair
list -- rows indexed by spike -- are moved aside as .stalebkp; a later call that changes nothing leaves
a fresh manifest alone.  A .clc an earlier dedup left at the old count is stale on its own: it is
quarantined, and its stage's live .clu is deduped as before rather than quarantined with it.  Control:
a stage whose .clu itself is at a third count is still quarantined whole.
"""
import contextlib
import io
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from fiber_kit import neuro_io as nio                  # noqa: E402
from fiber_kit.fiber_refiberize import FiberHierarchy  # noqa: E402


def main():
    ok = 0

    def check(name, cond):
        nonlocal ok
        assert cond, name
        ok += 1
        print(f"  ok  {name}")

    # ── refiberize: fibers 2, 3, 4, 5; fiber 3's only child (3) holds no spikes ──
    child = np.array([0, 2, 2, 4, 4, 5, 5, 0])
    parent = {2: 2, 3: 3, 4: 4, 5: 5}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        clu, clc, clp = FiberHierarchy(child, parent).refiberize(renumber=True)
    live = set(np.unique(clu[clu > 1]).tolist())
    check("live fibers are compacted to 2, 3, 4", live == {2, 3, 4})
    check("the empty child of the vanished fiber is parented to 0", clp[3 - 1] == 0 and "without spikes" in buf.getvalue())
    check("every child that holds spikes maps to its spikes' cluster",
          all(clp[c - 1] == clu[np.flatnonzero(clc == c)[0]] for c in np.unique(clc[clc > 0])))
    old = {c: (2 if f == 2 else 3 if f == 4 else 4 if f == 5 else f) for c, f in parent.items()}
    check("NEGATIVE CONTROL: the old id of the vanished fiber (3) is a live cluster after renumbering",
          old[3] == 3 and 3 in live)

    # ── apply_spike_keep ──
    with tempfile.TemporaryDirectory() as td:
        base = os.path.join(td, "dd")
        n, nsamp, nchan = 10, 8, 2
        res = np.arange(n, dtype=np.int64) * 100
        nio.write_res_file(base + ".res.1", res)
        nio.write_spk_file(base + ".spk.stderiv.1", np.arange(n * nsamp * nchan, dtype=np.int16).reshape(n, nsamp, nchan))
        lab = np.array([2, 2, 3, 3, 2, 3, 0, 2, 3, 2])
        atom = np.where(lab == 0, 0, lab + 10)
        nio.write_clu_file(base + ".clu.stderiv.1.fs", lab)
        nio.write_clu_file(base + ".clc.stderiv.1.fs", atom)
        np.save(base + ".offsets.1.npy", np.arange(n, dtype=np.float32) / 10)
        man = base + ".deconv.stderiv.1.fs.tsv"
        with open(man, "w") as fh:
            fh.write("spike\tres\n" + "".join(f"{i}\t{res[i]}\n" for i in range(n)))
        pairs = base + ".realign.stderiv.1.fs.tsv"
        with open(pairs, "w") as fh:
            fh.write("kind\trow_a\trow_b\ttime\tunit_a\tunit_b\tparked\nsame_unit\t0\t1\t0\t2\t2\tTrue\n")
        keep = np.ones(n, bool); keep[[3, 7]] = False
        with contextlib.redirect_stdout(io.StringIO()):
            nio.apply_spike_keep(base, "1", keep, n, nsamp, nchan)
        _, clu2 = nio.read_clu_file(base + ".clu.stderiv.1.fs")
        _, clc2 = nio.read_clu_file(base + ".clc.stderiv.1.fs")
        check("the .clc is subset with its .clu", np.array_equal(clc2, atom[keep]) and np.array_equal(clu2, lab[keep]))
        check("the realign offsets follow the base spike set",
              np.allclose(np.load(base + ".offsets.1.npy"), (np.arange(n, dtype=np.float32) / 10)[keep]))
        check("the deconvolution manifest and the realign pair list are moved aside",
              not os.path.exists(man) and os.path.exists(man + ".stalebkp")
              and not os.path.exists(pairs) and os.path.exists(pairs + ".stalebkp"))
        with open(man, "w") as fh:                                  # a fresh manifest of the new set
            fh.write("spike\tres\n0\t0\n")
        with contextlib.redirect_stdout(io.StringIO()):
            nio.apply_spike_keep(base, "1", keep, n, nsamp, nchan)   # idempotent: nothing at n rows any more
        check("a call that changes nothing leaves a fresh manifest alone", os.path.exists(man))

    # a derived stage whose .clc an earlier (pre-0.51) dedup left at the old count
    for third in (False, True):
        with tempfile.TemporaryDirectory() as td:
            base = os.path.join(td, "st")
            n0, n1 = 120, 100                              # before the earlier dedup; now
            nio.write_res_file(base + ".res.1", np.arange(n1) * 10)
            nio.write_clu_file(base + ".clu.stderiv.1", np.full(n1, 2))
            nio.write_clu_file(base + ".clu.stderiv.1.fs", np.arange(n1 if not third else 77) % 5 + 2)
            nio.write_clu_file(base + ".clc.stderiv.1.fs", np.full(n0, 4))
            keep = np.arange(n1) % 10 != 0
            with contextlib.redirect_stdout(io.StringIO()):
                nio.apply_spike_keep(base, "1", keep, n1, 8, 2)
            live = os.path.exists(base + ".clu.stderiv.1.fs")
            gone = os.path.exists(base + ".clc.stderiv.1.fs.stalebkp")
            if not third:
                _, c = nio.read_clu_file(base + ".clu.stderiv.1.fs")
                check("a .clc left at an old count is quarantined on its own; its stage's .clu is deduped",
                      gone and live and np.array_equal(c, (np.arange(n1) % 5 + 2)[keep]))
            else:
                check("CONTROL: a stage whose .clu is at a third count is still quarantined whole",
                      gone and not live and os.path.exists(base + ".clu.stderiv.1.fs.stalebkp"))
    print(f"test_hierarchy_bookkeeping: {ok} checks passed")


if __name__ == "__main__":
    main()
