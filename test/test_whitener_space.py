"""The chunk whitener builds its noise model in the waveform space it is given.

A noise model in another space than the spikes is not white for them: all-pairs stderiv noise has no
per-sample common mode, so whitening an order-5 (stderiv_C5) .spk with it inflates the common-mode
directions C5 keeps.  On the reference session that put 80% of the whitened noise and ~93% of each
spike's whitened energy in 15 directions.  These checks run on synthetic .fil noise with a common-mode
source, spatial leakage and temporal colour:

  * C5 noise is white under the C5 whitener and NOT under the historical all-pairs one (the negative
    control: the defect reproduces with the old model);
  * raw noise is white under the raw (standard) whitener;
  * the default is still the historical all-pairs model bit for bit, and space_transform('stderiv')
    gives the same model, so all-pairs sessions are unchanged;
  * a snippet at the very start of the span (no earlier sample) is handled as before.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from fiber_kit import fiber_lib as fl  # noqa: E402
from fiber_kit import fiber_stderiv as fst  # noqa: E402

SETS8 = "0-2+4,1-3+5,2-4+0+6,3-5+1+7,4-6+2+7,5-7+3+6,6-7+3+5,7-5+6"   # a real 8-ch order-5 pattern
C, WIN = 8, 32


def _noise(T, seed):
    """.fil-like integer noise: a common-mode source, leakage between neighbouring channels and
    temporally coloured per-channel noise -- the structure the spatial references differ on."""
    rng = np.random.default_rng(seed)
    k = np.array([1.0, 0.7, 0.35, 0.1])
    own = np.stack([np.convolve(rng.standard_normal(T + 3), k, "valid") for _ in range(C)], 1) * 60
    common = np.convolve(rng.standard_normal(T + 3), k, "valid")[:, None] * 90
    mix = np.eye(C) + 0.35 * (np.eye(C, k=1) + np.eye(C, k=-1))
    return np.round(own @ mix + common)


def _spikes(T, seed, n=40):
    return np.sort(np.random.default_rng(seed).choice(np.arange(100, T - 100), n, replace=False))


def _windows(trace, n, seed, transform):
    rng = np.random.default_rng(seed)
    st = rng.integers(1, trace.shape[0] - WIN - 1, n)
    raw = np.stack([trace[s - 1:s + WIN] for s in st])                 # one leading sample
    return raw[:, 1:] if transform is None else transform(raw)


def _spread(W, nmean, X):
    """Largest eigenvalue / median eigenvalue of the whitened covariance of masked windows X."""
    Z = (X[:, fl.MASK_FULL, :].reshape(len(X), -1).astype(np.float64) - nmean) @ W
    ev = np.linalg.eigvalsh(np.cov(Z.T))
    return float(ev.max() / np.median(ev))


def _legacy_baseline(read_window, T, spike_rel, n_base, guard, seed, win):
    """The pre-change _collect_baseline, verbatim in effect: per snippet fil_to_spkD_space."""
    sp = np.sort(np.asarray(spike_rel, dtype=np.int64))
    rng = np.random.default_rng(seed)
    base = []; tries = 0; maxtries = 50 * n_base
    hi_start = max(1, T - win)
    while len(base) < n_base and tries < maxtries:
        tries += 1
        start = int(rng.integers(0, hi_start))
        i = np.searchsorted(sp, start - guard, "left")
        j = np.searchsorted(sp, start + win + guard, "right")
        if j > i:
            continue
        seg = np.asarray(read_window(max(0, start - 1), start + win), dtype=np.float32)
        if seg.shape[0] < win:
            continue
        base.append(np.asarray(fl.fil_to_spkD_space(seg)[-win:], dtype=np.float32))
    return base


TRACE = _noise(400000, 1)
TEST = _noise(200000, 2)                                               # independent noise for checking
SPK = _spikes(TRACE.shape[0], 3)


def test_c5_noise_is_white_only_under_the_c5_whitener():
    tf = fst.space_transform("stderiv_C5", SETS8)
    X = _windows(TEST, 4000, 4, tf)
    Wc, mc, _ = fl.chunk_whitener(TRACE, SPK, mask=fl.MASK_FULL, win=WIN, transform=tf)
    Wa, ma, _ = fl.chunk_whitener(TRACE, SPK, mask=fl.MASK_FULL, win=WIN)              # historical
    sc, sa = _spread(Wc, mc, X), _spread(Wa, ma, X)
    assert sc < 2.0, sc                                # white up to sampling error
    assert sa > 10 * sc, (sa, sc)                      # negative control: the defect


def test_raw_noise_is_white_under_the_raw_whitener():
    X = _windows(TEST, 4000, 5, None)
    Wr, mr, _ = fl.chunk_whitener(TRACE, SPK, mask=fl.MASK_FULL, win=WIN, transform=None)
    assert _spread(Wr, mr, X) < 2.0
    Wa, ma, _ = fl.chunk_whitener(TRACE, SPK, mask=fl.MASK_FULL, win=WIN)
    assert _spread(Wa, ma, X) > 10.0                    # stderiv noise model on raw windows


def test_default_is_the_historical_model_bit_for_bit():
    Wa, ma, n = fl.chunk_whitener(TRACE, SPK, mask=fl.MASK_FULL, win=WIN)
    base = _legacy_baseline(lambda a, b: TRACE[a:b], TRACE.shape[0], SPK, 6000, 24, 0, WIN)
    Wl, ml, nl = fl._fit_whitener(base, fl.MASK_FULL)
    assert n == nl
    np.testing.assert_array_equal(Wa, Wl)
    np.testing.assert_array_equal(ma, ml)
    # the memmap path and the stage adapter give the same model
    Wm, mm_, _ = fl.chunk_whitener_mm(TRACE, np.arange(C), 0, TRACE.shape[0], SPK, mask=fl.MASK_FULL, win=WIN)
    np.testing.assert_array_equal(Wa, Wm)
    np.testing.assert_array_equal(ma, mm_)


def test_stderiv_token_keeps_allpairs_sessions_unchanged():
    Wa, ma, _ = fl.chunk_whitener(TRACE, SPK, mask=fl.MASK_FULL, win=WIN)
    Ws, ms, _ = fl.chunk_whitener(TRACE, SPK, mask=fl.MASK_FULL, win=WIN,
                                  transform=fst.space_transform("stderiv"))
    np.testing.assert_array_equal(Wa, Ws)
    np.testing.assert_array_equal(ma, ms)


def test_snippet_at_the_start_of_the_span():
    tr = TRACE[:WIN + 6]                                # starts 0..5: start 0 has no earlier sample
    new = fl._collect_baseline(lambda a, b: tr[a:b], tr.shape[0], [], 60, 24, 7, win=WIN)
    old = _legacy_baseline(lambda a, b: tr[a:b], tr.shape[0], [], 60, 24, 7, WIN)
    np.testing.assert_array_equal(np.asarray(new), np.asarray(old))
    assert np.any(np.all(np.asarray(new)[:, 0] == 0, axis=1))   # a start-0 snippet was drawn


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("ok", fn.__name__)
    print(f"\n{len(fns)} passed")
