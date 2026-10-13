"""fiber_consensus on the synthetic recording of test_envelope: fragments of one unit are re-joined
into one cell through the resampled-t-SNE consensus + envelope merge, two units stay apart, and the
quarantine is enriched for spikes that carry a near collision.  Skips when openTSNE / networkx are
not installed (they are the optional `consensus` extra, with scikit-image for the watershed)."""
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import importlib.util  # noqa: E402

HAVE = all(importlib.util.find_spec(m) is not None for m in ("networkx", "openTSNE", "skimage"))
HAVE_WS = importlib.util.find_spec("skimage") is not None
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


def test_categorise_basins_island_own_rim_other_cell():
    """Pure function on given basin labels: basin 1 owned by cell 0, basin 2 by cell 1, basin 3 mostly
    quarantined (an island), label 0 outside every basin (also an island)."""
    basin = np.r_[[1] * 10, [2] * 10, [3] * 4, 0]
    cell = np.r_[[0] * 10, [1] * 10, [0, 0, 1, 1], 0]
    q = np.zeros(basin.size, bool)
    q[[0, 1]] = True          # cell 0 in its own basin            -> own-rim
    q[12] = True; cell[12] = 0  # cell 0 in cell 1's basin          -> other-cell, destination 1
    q[[20, 21, 22]] = True    # 3 of 4 of basin 3                  -> island
    q[24] = True              # outside every basin                -> island
    code, dest = fc.categorise_basins(basin, cell, q)
    cats = [fc.WS_CATEGORIES[c] for c in code]
    assert cats == ["own-rim", "own-rim", "other-cell", "island", "island", "island", "island"], cats
    assert list(dest) == [0, 0, 1, -1, -1, -1, -1]


def test_watershed_categories_vote_over_given_maps():
    """Two Gaussian cells and a separate blob of quarantined spikes, on three given 2-D maps (no t-SNE):
    the blob is an island, a quarantined spike inside its own cell is own-rim, and one labelled cell 0
    sitting in cell 1's cloud is other-cell with destination 1."""
    if not HAVE_WS:
        print("skip: scikit-image not installed"); return
    rng = np.random.default_rng(1)
    n = 300
    cell = np.r_[np.zeros(n, int), np.ones(n, int), np.zeros(40, int)]
    q = np.zeros(cell.size, bool); q[2 * n:] = True; q[5] = True; q[n + 7] = True; cell[n + 7] = 0
    maps = []
    for _ in range(3):
        Y = np.r_[rng.normal(0, 1, (n, 2)), rng.normal(0, 1, (n, 2)) + [12, 0], rng.normal(0, 0.5, (40, 2)) + [6, 10]]
        maps.append(Y)
    ws = fc.watershed_categories(None, cell, q, maps=maps, grid=128)
    assert set(ws["category"][2 * n:]) == {"island"}
    assert ws["category"][5] == "own-rim" and ws["dest"][5] == 0
    assert ws["category"][n + 7] == "other-cell" and ws["dest"][n + 7] == 1
    assert ws["category"][0] == "" and np.isnan(ws["agree"][0])
    assert np.all(ws["agree"][q] == 1.0)


def test_cell_ids_keep_input_ids():
    """The largest cell takes its majority unit; a second cell whose majority unit is taken is new; a
    cell made of noise-labelled (<= 1) spikes is new; keep=False gives every cell a fresh id."""
    cell = np.r_[[0] * 50, [1] * 20, [2] * 10]
    unit = np.r_[[7] * 40, [9] * 10, [7] * 20, [1] * 10]
    q = np.zeros(cell.size, bool)
    cmap, nxt = fc.cell_ids(cell, unit, q, 100)
    assert cmap == {0: 7, 1: 100, 2: 101} and nxt == 102
    cmap, nxt = fc.cell_ids(cell, unit, q, 100, keep=False)
    assert sorted(cmap.values()) == [100, 101, 102]


def test_exp_yaml_wires_every_consensus_knob():
    """fiber-kit-exp.yaml section [C] and the stage's FK_CONSENSUS_* knobs name the same set (SIGNAL is
    read by fiber-pipeline, not the stage)."""
    import yaml
    with open(os.path.join(HERE, "..", "src", "fiber_kit", "fiber-kit-exp.yaml")) as fh:
        d = yaml.safe_load(fh)
    keys = {k for k in d if k.startswith("FK_CONSENSUS_")}
    assert keys - {"FK_CONSENSUS_SIGNAL"} == set(fc._KNOBS), keys ^ set(fc._KNOBS)
    for k, (_, typ, _fb, _) in fc._KNOBS.items():         # the yaml value parses as the knob's type
        if d[k] not in (None, ""):
            typ(d[k])


class _E:
    def __init__(self, n, template_ext=None):
        self.n = n; self.template_ext = template_ext


def test_neighbourhoods_cap_recuts_a_chained_group():
    """Six units in a d' chain (each 3 from the next, all < 12 apart on average linkage) form ONE group
    uncapped; with a spike cap the group is re-cut at a lower d' into pieces that fit."""
    units = {u: _E(1000) for u in range(2, 8)}
    pos = {u: 3.0 * i for i, u in enumerate(units)}
    real = fc.fe.separation
    try:
        fc.fe.separation = lambda a, b: (abs(pos[_id[id(a)]] - pos[_id[id(b)]]), 0.0)
        _id = {id(e): u for u, e in units.items()}
        one = fc.neighbourhoods(units, 12.0)
        capped = fc.neighbourhoods(units, 12.0, max_spikes=2500)
    finally:
        fc.fe.separation = real
    assert len(one) == 1 and len(one[0]) == 6, one
    assert all(len(g) * 1000 <= 2500 for g in capped) and sum(map(len, capped)) == 6, capped


def test_find_partners_with_no_other_unit():
    """A neighbourhood that merges into ONE cell leaves the partner dictionary empty: no crash, no
    partners (was: ValueError from np.stack)."""
    Lx, C = 60, 8
    T = np.random.default_rng(0).normal(0, 1, (Lx, C))
    env = _E(100, T)
    p = fc.fe.find_partners(np.zeros((3, Lx, C)), 2, env, {2: env}, np.ones(C), core_in_ext=slice(10, 50))
    assert list(p["partner"]) == [-1, -1, -1] and not p["explained"].any()
    atoms, keys, _ = fc.fe.partner_dictionary({2: env}, Lx, exclude=2)
    assert atoms.shape == (0, Lx, C) and keys == []


def test_parallel_runs_identical_to_serial():
    """workers only changes speed: subsets and seeds are drawn up front."""
    if not HAVE:
        print("skip: openTSNE/networkx not installed"); return
    X = np.random.default_rng(0).normal(size=(300, 8)); X[:150] += 3
    a = fc.tsne_runs(X, runs=3, workers=1)
    b = fc.tsne_runs(X, runs=3, workers=2)
    for x, y in zip(a, b):
        assert all(np.array_equal(p, q) for p, q in zip(x, y))
    q = np.arange(300) % 13 == 0; cell = (np.arange(300) < 150).astype(int)
    w1 = fc.watershed_categories(X, cell, q, runs=2, workers=1)
    w2 = fc.watershed_categories(X, cell, q, runs=2, workers=2)
    assert (w1["category"] == w2["category"]).all() and (w1["dest"] == w2["dest"]).all()


def test_knobs_env_over_global_yaml_over_default():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "fk.yaml")
        with open(p, "w") as fh:
            fh.write("FK_CONSENSUS_RUNS: 12\nFK_CONSENSUS_RULES: collision\n")
        old = {k: os.environ.get(k) for k in ("FK_CONFIG", "FK_CONSENSUS_RUNS", "FK_CONSENSUS_RULES")}
        try:
            os.environ["FK_CONFIG"] = p
            os.environ.pop("FK_CONSENSUS_RULES", None); os.environ["FK_CONSENSUS_RUNS"] = "9"
            from fiber_kit import config as cfgmod
            g = cfgmod.load_global_config()
            assert fc._knob_default("FK_CONSENSUS_RUNS", int, 40, g) == 9            # env beats the yaml
            assert fc._knob_default("FK_CONSENSUS_RULES", str, "x", g) == "collision"  # yaml beats default
            assert fc._knob_default("FK_CONSENSUS_WS_RUNS", int, 5, g) == 5           # default
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


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
                 "--space", "fil", "--units", "2,12,22,3", "--runs", "6", "--ws-runs", "2"])
        _, out = nio.read_clu_at(b, 6, variant="stderiv_C5_D34", tag="consensus")
        assert out.size == res.size
        # --keep-ids (default): the cell that IS unit 3 keeps id 3; unit 2's fragments join one cell that
        # keeps one of their ids
        def major(m):
            v, n = np.unique(out[m], return_counts=True); return v[n.argmax()]
        assert major(clu == 3) == 3
        assert major(clu == 2) in (2, 12, 22)
        assert np.array_equal(out[lab == 4], lab[lab == 4])                            # untouched
        with open(b + ".qrn.6.consensus") as fh:
            rows = fh.read().splitlines()
        assert rows[0].startswith("spike\ttime") and len(rows) > 1
        hdr = rows[0].split("\t"); col = {h: i for i, h in enumerate(hdr)}
        recs = [r.split("\t") for r in rows[1:]]
        assert hdr[-3:] == ["category", "ws_agree", "ws_dest"]
        assert {r[col["category"]] for r in recs} <= set(fc.WS_CATEGORIES)
        # islands and rim go to separate clusters: no output cluster holds both
        by_out = {}
        for r in recs:
            by_out.setdefault(r[col["out"]], set()).add(r[col["category"]] == "island")
        assert all(len(v) == 1 for v in by_out.values()), by_out


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("ok", fn.__name__)
    print(f"\n{len(fns)} passed")
