"""fiber-anchor-link keeps Klusters' reserve bins out of the link.

Cluster 1 is where fiber-consensus parks its quarantine (and intrachunk its reserve); it spans
the whole session.  Treated as an atom it (a) merged every chunk into one interval, so the chunk
partition fell back to fixed bins, (b) was chained onto real cells, carrying the quarantine back
into the merge, and (c) the linker numbered its largest chain 1, which every later stage
(xcorr-merge, backbone-link: units are `clu > 1`) then dropped to noise.  Reproduced on the
reference session g6 run: 14,190 quarantined spikes ended in exp_final cell 90 with three real
units, and a real 7,513-spike chain sat in cluster 1."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from fiber_kit import fiber_anchor_link as fal  # noqa: E402


def _toy():
    sr = 20000
    res, clu = [], []
    # three chunks of 12 min, two units per chunk (ids 2..7), plus the reserve bin 1 across all
    for c in range(3):
        t0 = c * 12 * 60 * sr
        for k, u in enumerate((2 + 2 * c, 3 + 2 * c)):
            ts = t0 + np.linspace(sr, 11 * 60 * sr, 200 + 50 * k).astype(np.int64)
            res.append(ts); clu.append(np.full(ts.size, u))
    tq = np.linspace(sr, 35 * 60 * sr, 500).astype(np.int64)
    res.append(tq); clu.append(np.ones(tq.size, int))
    res = np.concatenate(res); clu = np.concatenate(clu)
    o = np.argsort(res)
    return res[o], clu[o], sr


def test_reserve_does_not_collapse_chunks():
    res, clu, sr = _toy()
    chunk_of, rel = fal.atom_chunks_and_rates(clu, res, sr, chunk_min=None)
    assert 1 not in chunk_of and 0 not in chunk_of
    assert len(set(chunk_of.values())) == 3, chunk_of
    assert chunk_of[2] == chunk_of[3] == 0 and chunk_of[6] == chunk_of[7] == 2


def test_cells_never_numbered_into_the_reserve():
    # a chain that (wrongly) lists the reserve atom must not pull it in, and the biggest
    # chain must not be numbered 1
    parent = fal.cells_from_links([[2, 4, 6], [1, 3, 5], [7]], n_atoms=7)
    assert parent[0] == 1                       # atom 1 -> reserve cell 1
    assert len({parent[1], parent[3], parent[5]}) == 1
    assert len({parent[2], parent[4]}) == 1
    assert parent[1] != parent[2]
    assert parent[1:].min() >= 2                # no real atom lands in 0 or 1
    assert sorted(set(parent[1:].tolist())) == list(range(2, 2 + 3))


def test_anchor_pairs_translate_to_the_fragment_layer():
    # atoms 2..7 (fiber-session .clc); fragments merge atoms {2,3}->10, {4}->11, {5,6}->12; atom 7 is
    # split evenly between 11 and 12 (no majority); atom 8 is all noise
    atom = np.repeat([2, 3, 4, 5, 6, 7, 7, 8], 50)
    frag = np.repeat([10, 10, 11, 12, 12, 11, 12, 1], 50)
    pairs = [(2, 4), (3, 5), (2, 3), (7, 4), (6, 8), (3, 4)]
    out, st = fal.translate_anchor_pairs(pairs, atom, frag, min_frac=0.6)
    assert out == [(10, 11), (10, 12)], out
    assert st["same"] == 1 and st["unmapped"] == 2 and st["dup"] == 1 and not st["identity"]
    # same layer -> identity (ids untouched, even ones a translation would drop)
    out2, st2 = fal.translate_anchor_pairs(pairs, atom, atom)
    assert out2 == pairs and st2["identity"]


def test_wrong_layer_ids_are_not_read_as_fragments():
    # the bug: anchor atom ids that happen to exist as fragment ids name OTHER clusters
    atom = np.repeat([2, 3, 4, 5], 40)
    frag = np.repeat([5, 4, 3, 2], 40)              # same numbers, permuted meaning
    out, _ = fal.translate_anchor_pairs([(2, 3)], atom, frag)
    assert out == [(5, 4)]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("ok", fn.__name__)
    print(f"\n{len(fns)} passed")
