#!/usr/bin/env python3
"""fiber_provenance.py -- per-merge provenance for the atomic intrachunk+link hierarchy.

Both merge stages append one row per merge edge (parent <- child) recording WHY the
merge was admissible: the gate scores between the child and its parent's representative.
Written as a .merge.tsv sidecar alongside the .clu/.clc/.clp triple.

Diagnose overmerging: sort the table by weakest justification -- low `cosine` AND/OR
high `refrac_viol` is a prime overmerge suspect -- then unmerge the children the row
points to with FiberHierarchy.split_parent.  The hierarchy says WHERE a merge is; this
says WHY, so the two together make every merge reversible and attributable.
"""
import numpy as np

try:
    from . import fiber_geometry as fg
except ImportError:
    import fiber_geometry as fg

COLS = ["stage", "chunk", "parent", "child", "n_child",
        "cosine", "warp", "offset_rms", "depth_gap", "refrac_viol"]


def score_pair(ta, tb, *, sr=32552.0, off_a=None, off_b=None,
               pos_a=None, pos_b=None, times_a=None, times_b=None):
    """Recompute the gate scores between two unit templates (nsamp,nchan) and optional
    offsets / scalar depth / spike trains (samples).  Returns the COLS[5:] metrics
    (NaN where the input is unavailable)."""
    ta = np.asarray(ta, float); tb = np.asarray(tb, float)
    a = fg.mutual_center(ta).ravel(); b = fg.mutual_center(tb).ravel()
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    try:
        warp = float(fg.warp_correlation(fg.group_delay_profile(ta, sr),
                                         fg.group_delay_profile(tb, sr)))
    except Exception:
        warp = float("nan")
    if off_a is None:
        try: off_a = fg.interchannel_offsets(ta)
        except Exception: off_a = None
    if off_b is None:
        try: off_b = fg.interchannel_offsets(tb)
        except Exception: off_b = None
    if off_a is not None and off_b is not None:
        d = np.asarray(off_a, float) - np.asarray(off_b, float)
        offrms = float(np.sqrt(np.nanmean(d ** 2)))
    else:
        offrms = float("nan")
    depth = (float(abs(pos_a - pos_b)) if (pos_a is not None and pos_b is not None)
             else float("nan"))
    refr = float("nan")
    if times_a is not None and times_b is not None and len(times_a) and len(times_b):
        m = np.sort(np.concatenate([np.asarray(times_a, float), np.asarray(times_b, float)]))
        isi = np.diff(m)
        if isi.size:
            refr = float(np.mean(isi < 2.0e-3 * sr))    # frac of merged-train ISIs < 2 ms
    return dict(cosine=cos, warp=warp, offset_rms=offrms, depth_gap=depth, refrac_viol=refr)


class MergeLog:
    """Accumulates merge edges and writes the .merge.tsv sidecar."""

    def __init__(self):
        self.rows = []

    def edge(self, stage, chunk, parent, child, n_child, scores):
        row = dict(stage=str(stage),
                   chunk=int(chunk) if chunk is not None else -1,
                   parent=int(parent), child=int(child), n_child=int(n_child))
        row.update({k: float(scores.get(k, float("nan"))) for k in COLS[5:]})
        self.rows.append(row)

    def __len__(self):
        return len(self.rows)

    def write(self, path):
        with open(path, "w") as f:
            f.write("\t".join(COLS) + "\n")
            for r in self.rows:
                f.write("\t".join(_fmt(r[c]) for c in COLS) + "\n")
        return len(self.rows)


def _fmt(v):
    if isinstance(v, float):
        return "nan" if v != v else f"{v:.4f}"
    return str(v)
