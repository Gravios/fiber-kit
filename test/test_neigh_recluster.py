"""Block N: neighbourhood re-cluster (rkk over a random set of neighbouring clusters, seeded)."""
import numpy as np
from fiber_kit import fiber_session as fs
from fiber_kit.klustakwik import klustakwik_refine


def _spikes(rng, n, amp, width, T=52, C=4, pk=26):
    t = np.arange(T)
    shape = -np.exp(-0.5 * ((t - pk) / width) ** 2) + 0.4 * np.exp(-0.5 * ((t - pk - 2 * width) / (1.6 * width)) ** 2)
    prof = np.array(amp)                                            # per-channel amplitude
    w = shape[None, :, None] * prof[None, None, :] * 1000.0
    return w + rng.normal(scale=60.0, size=(n, T, C))


def test_refine_fixes_planted_boundary_errors():
    rng = np.random.default_rng(0)
    A = _spikes(rng, 400, [1.0, 0.6, 0.2, 0.0], 2.0)
    B = _spikes(rng, 400, [0.7, 1.0, 0.5, 0.1], 2.6)
    waves = np.concatenate([A, B]); truth = np.r_[np.zeros(400, int), np.ones(400, int)]
    fine = truth.copy()
    bad = rng.choice(800, 160, replace=False); fine[bad] ^= 1    # 20 % of the spikes on the wrong side
    mask = np.arange(10, 42)
    out, nacc = fs._neigh_recluster(fine, waves, None, None, mask, 4, 20, k=3, cos_thr=0.5, min_n=20, seed=1)
    assert nacc == 1
    # labels keep the old ids (majority inheritance) and agree with the truth far better than the input
    agree = max(np.mean(out == truth), np.mean(out == 1 - truth))
    assert agree > 0.97 and np.mean(fine == truth) < 0.85
    assert set(np.unique(out)) <= {0, 1}


def test_clean_partition_is_left_alone():
    rng = np.random.default_rng(2)
    A = _spikes(rng, 300, [1.0, 0.6, 0.2, 0.0], 2.0)
    B = _spikes(rng, 300, [0.0, 0.3, 1.0, 0.6], 2.6)
    waves = np.concatenate([A, B]); fine = np.r_[np.zeros(300, int), np.ones(300, int)]
    mask = np.arange(10, 42)
    out, nacc = fs._neigh_recluster(fine, waves, None, None, mask, 4, 20, k=3, cos_thr=0.0, min_n=20, seed=1)
    assert (out == fine).all()


def test_refine_scores_are_on_the_same_data():
    rng = np.random.default_rng(3)
    X = np.vstack([rng.normal(size=(200, 3)), rng.normal(size=(200, 3)) + 5])
    truth = np.r_[np.zeros(200, int), np.ones(200, int)]
    lab, s0, s1 = klustakwik_refine(X, truth)
    assert s1 <= s0 + 1e-6 and np.unique(lab).size == 2
