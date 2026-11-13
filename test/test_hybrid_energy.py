"""energy_half / refit_basis / fit_hybrid_gain(fit=) -- the energy-aware hybrid knobs."""
import numpy as np
from fiber_kit import fiber_pca as fp


def _basis(nCh=2, nComp=3, T=25):
    rng = np.random.default_rng(1)
    ev = np.linalg.qr(rng.normal(size=(T, T)))[0][:nComp].T          # (T, nComp) orthonormal columns
    evec = np.stack([ev.T for _ in range(nCh)])                      # (nCh, nComp, T)
    return dict(means=np.zeros((nCh, T)), evec=evec, recShift=0, data2use=T, centered=1)


def test_energy_half_splits_at_median():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(1000, 20, 3)) * np.linspace(0.5, 2.0, 1000)[:, None, None]
    lo, hi = fp.energy_half(w, "low"), fp.energy_half(w, "high")
    assert lo.sum() + hi.sum() == 1000 and abs(lo.sum() - hi.sum()) <= 1
    assert lo[:300].all() and hi[-300:].all()
    assert fp.energy_half(w, "all").all()


def test_refit_basis_recovers_planted_components():
    rng = np.random.default_rng(2)
    b = _basis()
    T, nCh = 25, 2
    Q = np.linalg.qr(rng.normal(size=(T, T)))[0][:, :3]
    X = np.stack([rng.normal(size=(4000, 3)) * [5, 3, 1] @ Q.T for _ in range(nCh)], 2)
    nb = fp.refit_basis(b, X)
    assert nb["evec"].shape == b["evec"].shape
    for ch in range(nCh):
        s = np.linalg.svd(nb["evec"][ch] @ Q, compute_uv=False)
        assert s.min() > 0.99                                        # same 3-D subspace
        assert np.allclose(nb["evec"][ch] @ nb["evec"][ch].T, np.eye(3), atol=1e-8)
    assert b["evec"] is not nb["evec"]                               # the session basis is untouched


def test_gain_fit_half_differs_from_all():
    rng = np.random.default_rng(3)
    T, nCh, N = 25, 2, 2000
    b = _basis(nCh, 3, T)
    amp = np.exp(rng.normal(size=N))[:, None, None]
    wm = rng.normal(size=(N, T, nCh)) * amp
    wr = rng.normal(size=(N, T, nCh)) * amp ** 2                    # raw grows faster with energy
    b["_hyb"] = fp.hybrid_basis(b, pcs=3, nraw=nCh)
    g_all = fp.fit_hybrid_gain(wm, wr, b, fit="all")
    g_low = fp.fit_hybrid_gain(wm, wr, b, fit="low")
    g_high = fp.fit_hybrid_gain(wm, wr, b, fit="high")
    assert g_low > g_all > g_high > 0
