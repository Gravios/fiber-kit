"""fiber_decollide core: raw-space mutual subtraction, recovered times, the
.decollided stage write (through the staged readers/writers), and the manifest.

The whitened-space decomposition is fiber_collision's job (validated there and
needing the .fil whitener); these tests cover the data-shaping core, which takes
decomposition results + raw templates and is independent of the whitener.
"""
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from fiber_kit import neuro_io as nio          # noqa: E402
from fiber_kit import fiber_decollide as fd    # noqa: E402

NS, NCH = 32, 4


def _pulse(center, amp, ch, width=3.0):
    """A clean (nsamp, nch) template: a Gaussian trough on one channel, ~0 at edges."""
    t = np.arange(NS)
    w = np.zeros((NS, NCH))
    w[:, ch] = -amp * np.exp(-0.5 * ((t - center) / width) ** 2)
    return w


def test_subtraction_recovers_both_constituents():
    T1 = _pulse(16, 1000, ch=0)
    T2 = _pulse(16, 600, ch=1)
    a1, a2, tau2 = 1.0, 1.0, 3
    coll = a1 * T1 + a2 * fd._roll0(T2, tau2)          # a clean 2-cell collision
    raw = np.stack([coll])
    decomp = dict(k1=[10], a1=[a1], tau1=[0], k2=[11], a2=[a2], tau2=[tau2], gain=[0.9])
    nt, nc, nw, man = fd.decollide_spikes(
        raw, res_times=[1000], clu_ids=[10], sel=[0], decomp=decomp,
        raw_templates={10: T1, 11: T2})
    assert len(nt) == 2 and set(nc.tolist()) == {10, 11}
    # recover each constituent in the central region (edges carry _roll0 zero-fill)
    w_by = {int(c): nw[i].astype(float) for i, c in enumerate(nc)}
    core = slice(8, 25)
    assert np.allclose(w_by[10][core], (a1 * T1)[core], atol=2.0)      # k2 removed
    assert np.allclose(w_by[11][core], (a2 * T2)[core], atol=2.0)      # k1 removed, recentred


def test_recovered_times_and_recentring():
    T1 = _pulse(16, 1000, ch=0); T2 = _pulse(16, 800, ch=1)
    tau1, tau2 = 0, 4
    coll = T1 + fd._roll0(T2, tau2)
    nt, nc, nw, man = fd.decollide_spikes(
        [coll], [5000], [10], [0],
        dict(k1=[10], a1=[1.0], tau1=[tau1], k2=[11], a2=[1.0], tau2=[tau2], gain=[0.9]),
        {10: T1, 11: T2})
    tmap = {int(c): int(t) for c, t in zip(nc, nt)}
    assert tmap[10] == 5000 + tau1 and tmap[11] == 5000 + tau2      # recovered times
    # k2's trough is back at the window centre (recentred on its own peak)
    w11 = nw[np.flatnonzero(nc == 11)[0]].astype(float)
    assert abs(int(np.argmin(w11[:, 1])) - 16) <= 1


def test_net_plus_one_and_passthrough_when_template_missing():
    T1 = _pulse(16, 900, 0); T2 = _pulse(16, 500, 1)
    N = 6
    raw = np.stack([T1 + fd._roll0(T2, 2)] * N)
    res = np.arange(N) * 100 + 10
    clu = np.full(N, 10)
    # resolve spikes 1 and 4; spike 4's partner template is MISSING -> passthrough
    decomp = dict(k1=[10, 10], a1=[1.0, 1.0], tau1=[0, 0],
                  k2=[11, 99], a2=[1.0, 1.0], tau2=[2, 2], gain=[0.9, 0.9])
    nt, nc, nw, man = fd.decollide_spikes(raw, res, clu, [1, 4], decomp, {10: T1, 11: T2})
    # spike 1 -> 2 spikes; spike 4 -> left as 1 (missing template); 4 kept + 2 + 1 = 7
    assert len(nt) == N + 1
    assert len(man) == 1 and man[0]["source_idx"] == 1
    assert np.all(np.diff(nt) >= 0)                                 # sorted by time


def test_stage_write_reads_back_consistently():
    T1 = _pulse(16, 1000, 0); T2 = _pulse(16, 700, 1)
    raw = np.stack([T1 + fd._roll0(T2, 3), T1 * 0.9])
    res = np.array([200, 900]); clu = np.array([10, 10])
    decomp = dict(k1=[10], a1=[1.0], tau1=[0], k2=[11], a2=[1.0], tau2=[3], gain=[0.9])
    nt, nc, nw, man = fd.decollide_spikes(raw, res, clu, [0], decomp, {10: T1, 11: T2})
    with tempfile.TemporaryDirectory() as d:
        b = os.path.join(d, "sess")
        paths = fd.write_decollided_stage(
            b, 6, nt, nc, nw, man, tag="decollided",
            res_variant="stderiv", clu_variant="stderiv_C5_D34", spk_variant="standard")
        # read the staged set back through the patch-1 pinned readers
        rr = nio.read_res_at(b, 6, variant="stderiv", tag="decollided")
        _, ids = nio.read_clu_at(b, 6, variant="stderiv_C5_D34", tag="decollided")
        spk = nio.open_spk_at(b, 6, NS, NCH, variant="standard", tag="decollided")
        n = 1 + 2                                                   # one kept + one resolved->2
        assert rr.size == ids.size == spk.shape[0] == n
        # matched pair reader agrees (lengths in lockstep)
        cr = nio.read_cluster_res_at(b, 6, clu_variant="stderiv_C5_D34",
                                     res_variant="stderiv", tag="decollided")
        assert cr.ok
        assert os.path.exists(paths["manifest"])


def test_manifest_roundtrip_and_reconstruction():
    T1 = _pulse(16, 1000, 0); T2 = _pulse(16, 600, 1)
    a1, a2, tau2 = 1.0, 1.0, 3
    coll = a1 * T1 + a2 * fd._roll0(T2, tau2)
    nt, nc, nw, man = fd.decollide_spikes(
        [coll], [1000], [10], [0],
        dict(k1=[10], a1=[a1], tau1=[0], k2=[11], a2=[a2], tau2=[tau2], gain=[0.9]),
        {10: T1, 11: T2})
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.tsv")
        fd.write_manifest(p, man)
        back = fd.read_manifest(p)
        assert len(back) == 1 and back[0]["k2"] == 11 and back[0]["tau2"] == tau2
        # the audit inverse reconstructs the collision from the templates
        recon = fd.reconstruct_collision(back[0], {10: T1, 11: T2}, NS, NCH)
        assert np.allclose(recon[8:25], coll[8:25], atol=1e-6)


def test_from_decomposition_end_to_end():
    rng = np.random.default_rng(0)
    T1 = _pulse(16, 1000, 0); T2 = _pulse(16, 700, 1)
    waves, res, clu = [], [], []
    for i in range(5):
        waves.append(T1 + rng.standard_normal((NS, NCH)) * 2); res.append(100 + i * 50); clu.append(10)
    for i in range(5):
        waves.append(T2 + rng.standard_normal((NS, NCH)) * 2); res.append(1000 + i * 50); clu.append(11)
    waves.append(T1 + fd._roll0(T2, 3)); res.append(2000); clu.append(10)      # collision, idx 10
    waves = np.rint(np.stack(waves)).astype(nio.SPK_DTYPE)
    res = np.array(res, np.int64); clu = np.array(clu, np.int64)
    with tempfile.TemporaryDirectory() as d:
        b = os.path.join(d, "sess")
        nio.write_res(b, 6, res, variant="stderiv")                          # parent: no tag
        nio.write_clu(b, 6, clu, variant="stderiv_C5_D34")
        nio.write_spk(b, 6, waves, variant="standard")
        decomp = dict(k1=[10], a1=[1.0], tau1=[0], k2=[11], a2=[1.0], tau2=[3], gain=[0.9])
        fd.decollide_from_decomposition(
            b, 6, nsamp=NS, nch=NCH, sel=[10], decomp=decomp, tag="decollided",
            clu_variant="stderiv_C5_D34", res_variant="stderiv", min_tmpl=2)
        cr = nio.read_cluster_res_at(b, 6, clu_variant="stderiv_C5_D34",
                                     res_variant="stderiv", tag="decollided")
        assert cr.ok and cr.times.size == 12                                 # 11 - 1 + 2
        tc = {int(t): int(c) for t, c in zip(cr.times, cr.ids)}
        assert tc.get(2000) == 10 and tc.get(2003) == 11                     # the two recovered spikes
        spk = nio.open_spk_at(b, 6, NS, NCH, variant="standard", tag="decollided")
        assert spk.shape[0] == 12


def test_all_variants_spk_per_variant():
    """decollide_all_variants writes new units' .spk in EVERY variant, each
    subtracted in its OWN space (so the two variants' recovered constituents
    differ), with shared res/clu/manifest.  .fet is gated on a PCA basis, so this
    core test runs write_fet=False (the .fet path is validated on a real basis)."""
    rng = np.random.default_rng(0)
    # two distinct per-variant template sets for the SAME cells (10, 11)
    T10s = _pulse(16, 1000, 0); T11s = _pulse(16, 700, 1)           # standard space
    T10d = _pulse(16, 600, 2);  T11d = _pulse(16, 900, 3)           # stderiv space (different)
    waves_s, waves_d, res, clu = [], [], [], []
    for i in range(5):
        waves_s.append(T10s + rng.standard_normal((NS, NCH)) * 2); waves_d.append(T10d + rng.standard_normal((NS, NCH)) * 2)
        res.append(100 + i * 50); clu.append(10)
    for i in range(5):
        waves_s.append(T11s + rng.standard_normal((NS, NCH)) * 2); waves_d.append(T11d + rng.standard_normal((NS, NCH)) * 2)
        res.append(1000 + i * 50); clu.append(11)
    # a collision of 10+11 in BOTH spaces at idx 10 (same shift)
    waves_s.append(T10s + fd._roll0(T11s, 3)); waves_d.append(T10d + fd._roll0(T11d, 3))
    res.append(2000); clu.append(10)
    ws = np.rint(np.stack(waves_s)).astype(nio.SPK_DTYPE)
    wd = np.rint(np.stack(waves_d)).astype(nio.SPK_DTYPE)
    res = np.array(res, np.int64); clu = np.array(clu, np.int64)

    with tempfile.TemporaryDirectory() as d:
        b = os.path.join(d, "sess")
        nio.write_res(b, 6, res, variant="stderiv")
        nio.write_clu(b, 6, clu, variant="stderiv_C5_D34")
        nio.write_spk(b, 6, ws, variant="standard")
        nio.write_spk(b, 6, wd, variant="stderiv_C5_D34")
        decomp = dict(k1=[10], a1=[1.0], tau1=[0], k2=[11], a2=[1.0], tau2=[3], gain=[0.9])
        paths = fd.decollide_all_variants(
            b, 6, nsamp=NS, nch=NCH, sel=[10], decomp=decomp, tag="decollided",
            clu_variant="stderiv_C5_D34", res_variant="stderiv",
            spk_variants=("standard", "stderiv_C5_D34"), min_tmpl=2, write_fet=False)

        # both variants' .spk written; shared res/clu/manifest; net +1 (12 rows)
        assert "spk.standard" in paths and "spk.stderiv_C5_D34" in paths
        assert "fet.standard" not in paths                        # write_fet=False
        for v in ("standard", "stderiv_C5_D34"):
            spk = nio.open_spk_at(b, 6, NS, NCH, variant=v, tag="decollided")
            assert spk.shape[0] == 12
        cr = nio.read_cluster_res_at(b, 6, clu_variant="stderiv_C5_D34",
                                     res_variant="stderiv", tag="decollided")
        assert cr.ok and cr.times.size == 12
        # each variant recovered its OWN constituents: cell 10's recovered spike
        # (time 2000) peaks on ch0 in standard but ch2 in stderiv
        ss = nio.open_spk_at(b, 6, NS, NCH, variant="standard", tag="decollided")
        sd = nio.open_spk_at(b, 6, NS, NCH, variant="stderiv_C5_D34", tag="decollided")
        r2000 = int(np.flatnonzero(cr.times == 2000)[0])
        assert int(np.unravel_index(np.argmin(ss[r2000]), (NS, NCH))[1]) == 0
        assert int(np.unravel_index(np.argmin(sd[r2000]), (NS, NCH))[1]) == 2
        assert os.path.exists(paths["manifest"])


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("ok", fn.__name__)
    print(f"\n{len(fns)} passed")
