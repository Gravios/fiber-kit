"""The rkk starting-K die-out and its retry (FK_SESSION_RKK_PTS_PER_START)."""
import numpy as np
from fiber_kit import fiber_session as fs


def _two_blobs(n, rng):
    X = np.vstack([rng.normal(size=(n, 6)), rng.normal(size=(n, 6)) + [6, 0, 0, 0, 0, 0]])
    return X, np.r_[np.zeros(n, int), np.ones(n, int)]


def test_legacy_start_loses_a_small_group(monkeypatch):
    rng = np.random.default_rng(0)
    X, _ = _two_blobs(40, rng)                        # 80 points over 50 starting clusters
    monkeypatch.setattr(fs, "_RKK_PTS_PER_START", 0)
    lab = fs._rkk_labels(X, 50, 7, seed=42)
    assert (lab < 0).all()                            # the documented defect: everything in the reject cluster


def test_retry_recovers_it_and_leaves_big_groups_alone(monkeypatch):
    rng = np.random.default_rng(0)
    X, truth = _two_blobs(40, rng)
    monkeypatch.setattr(fs, "_RKK_PTS_PER_START", 21)
    lab = fs._rkk_labels(X, 50, 7, seed=42)
    assert (lab >= 0).mean() > 0.8
    for u in (0, 1):                                  # each blob is a pure cluster
        l = lab[(truth == u) & (lab >= 0)]
        assert np.bincount(l).max() / l.size > 0.95
    Xb, _ = _two_blobs(600, rng)                      # big group: the legacy start survives -> identical result
    monkeypatch.setattr(fs, "_RKK_PTS_PER_START", 0); a = fs._rkk_labels(Xb, 50, 7, seed=42)
    monkeypatch.setattr(fs, "_RKK_PTS_PER_START", 21); b = fs._rkk_labels(Xb, 50, 7, seed=42)
    assert (a == b).all()
