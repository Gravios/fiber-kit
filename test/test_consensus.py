"""fiber_consensus on the synthetic recording of test_envelope: fragments of one unit are re-joined
into one cell through the resampled-t-SNE consensus + envelope merge, two units stay apart, and the
quarantine is enriched for spikes that carry a near collision.  Skips when openTSNE / networkx are
not installed (they are the optional `consensus` extra)."""
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import importlib.util  # noqa: E402

HAVE = all(importlib.util.find_spec(m) is not None for m in ("networkx", "openTSNE"))
import test_envelope as te                    # noqa: E402  (the synthetic recording)
from fiber_kit import fiber_consensus as fc   # noqa: E402
from fiber_kit import fiber_envelope as fe    # noqa: E402
from fiber_kit import neuro_io as nio         # noqa: E402

_C = {}


def _data():
    if "d" not in _C:
        sig, res, clu, ev, *_ = te._run()
        X = fe.extract(sig, res, -21, 21, "stderiv").reshape(len(res), -1)
        Xc = X - X.mean(0)
        _, _, Vt = np.linalg.svd(Xc[::4], full_matrices=False)
        feats = Xc @ Vt[:12].T                                 # a 12-d PCA stand-in for the .fet
        lab = clu.copy(); i2 = np.flatnonzero(clu == 2)
        lab[i2[1::3]] = 12; lab[i2[2::3]] = 22                 # unit 2 cut into three fragments
        _C["d"] = (sig, res, clu, ev, feats, lab)
    return _C["d"]


def test_perimeter_rank_puts_outliers_on_the_rim():
    """Pure function, no t-SNE: two Gaussian clouds, each with two points placed far out; those rank
    ~1 within their own cloud in every run, the bulk averages ~0.5."""
    rng = np.random.default_rng(0)
    n = 400
    lab = np.r_[np.zeros(n // 2, int), np.ones(n // 2, int)]
    runs = []
    for _ in range(5):
        Y = rng.normal(0, 1, (n, 2)); Y[lab == 1] += 20
        Y[[0, 1]] = [[8, 0], [0, -8]]; Y[[200, 201]] = [[28, 0], [20, 8]]
        s = np.union1d(np.sort(rng.choice(n, 320, replace=False)), [0, 1, 200, 201])
        runs.append((s, None, Y[s]))
    per = fc.perimeter_rank(runs, lab, min_n=50)
    assert np.all(per[[0, 1, 200, 201]] > 0.98), per[[0, 1, 200, 201]]
    assert 0.4 < np.nanmedian(per) < 0.6


def test_consensus_rejoins_fragments_and_quarantines_collisions():
    if not HAVE:
        print("skip: openTSNE/networkx not installed"); return
    sig, res, clu, ev, feats, lab = _data()
    r = fc.consensus_neighbourhood(sig, res, lab, [2, 12, 22, 3], feats, sr=te.FS, runs=8, k=15,
                                   n_jobs=1, log=lambda s: None)
    truth = clu[r["idx"]]
    for u in (2, 3):                                           # each true unit ends up mostly in ONE cell
        c, n = np.unique(r["cell"][truth == u], return_counts=True)
        assert n.max() / n.sum() > 0.85, (u, dict(zip(c, n)))
    c2 = np.unique(r["cell"][truth == 2], return_counts=True)
    c3 = np.unique(r["cell"][truth == 3], return_counts=True)
    assert c2[0][c2[1].argmax()] != c3[0][c3[1].argmax()]    # and the two units stay apart
    near, _, _ = te._near(res, clu, ev)
    nq = near[r["idx"]]; q = r["quarantine"]
    assert q.any() and nq[q].mean() > 3 * nq.mean(), (nq[q].mean(), nq.mean())


def test_cli_writes_stage_and_quarantine_table():
    if not HAVE:
        print("skip: openTSNE/networkx not installed"); return
    sig, res, clu, ev, feats, lab = _data()
    with tempfile.TemporaryDirectory() as d:
        b = os.path.join(d, "sim")
        nio.write_res(b, 6, res, variant="stderiv")
        nio.write_clu(b, 6, lab, variant="stderiv_C5_D34")
        nio.write_fet(b, 6, np.rint(feats).astype(np.int64), variant="stderiv_C5_D34")
        sigp = b + ".fil"
        np.clip(np.rint(sig), -32768, 32767).astype("<i2").tofile(sigp)
        fc.main([b, "6", "--signal", sigp, "--total-channels", str(te.C), "--channels", "0-7",
                 "--space", "fil", "--units", "2,12,22,3", "--runs", "6"])
        _, out = nio.read_clu_at(b, 6, variant="stderiv_C5_D34", tag="consensus")
        assert out.size == res.size
        assert not np.isin(out[np.isin(lab, [2, 12, 22, 3])], [2, 12, 22, 3]).any()   # relabelled
        assert np.array_equal(out[lab == 4], lab[lab == 4])                            # untouched
        with open(b + ".qrn.6.consensus") as fh:
            rows = fh.read().splitlines()
        assert rows[0].startswith("spike\ttime") and len(rows) > 1


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("ok", fn.__name__)
    print(f"\n{len(fns)} passed")
