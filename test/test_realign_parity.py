"""fiber_realign 'klusters' is at parity with Klusters' realignSpikes.

The golden shifts below were produced by the COMPILED Klusters code from neurosuite-3 origin/main
(libklustersshared realign_xcorr_omp.cpp + realign_center.cpp) driven by a verbatim port of the GUI loop
in klustersdoc_realign.cpp (int16 truncated mean template, circular pre-roll to peakSampleIndex - 1,
cumulative rolls, threshold 0.70, 2 iterations; mode 2 adds the RMS circular group recenter).  On the
g6 36-47 min sort (19,543 spikes, 115 units) the same comparison agreed on every spike in both the
standard and the stderiv_C5_D34 .spk."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from fiber_kit import fiber_realign as fr   # noqa: E402


def golden_cluster():
    """Deterministic synthetic cluster (no RNG): 40 spikes x 32 samples x 4 channels, int16.  Trough at
    13+c and a large late lobe (energy centroid well after the trough), jitters -6..+6 (beyond a
    max_shift of 4), and 4 noise-only spikes that score below 0.70."""
    T, C, n = 32, 4, 40
    t = np.arange(T)
    tmpl = np.stack([-900 * np.exp(-0.5 * ((t - 13 - c) / 1.6) ** 2)
                     + 700 * np.exp(-0.5 * ((t - 22 - c) / 2.5) ** 2) for c in range(C)], 1)
    i = np.arange(n)[:, None, None]; tt = t[None, :, None]; cc = np.arange(C)[None, None, :]
    noise = 70 * np.sin(i * 12.9898 + tt * 78.233 + cc * 37.719) * np.cos(i * 4.1414 + tt * 1.618)
    jit = ((np.arange(n) * 7) % 13) - 6
    amp = 0.75 + 0.5 * ((np.arange(n) * 5) % 11) / 10.0
    W = np.stack([np.roll(tmpl * amp[k], jit[k], axis=0) for k in range(n)]) + noise
    W[:4] = 250 * np.sin(np.arange(4)[:, None, None] * 3.3 + tt * 2.7 + cc * 1.9)
    return np.clip(np.rint(W), -32768, 32767).astype(np.int16)



PEAK0, MAX_SHIFT = 18, 4
K_DEFAULT = [0, 0, 0, 0, 3, 0, 4, 0, 4, 0, 0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 4, 0, 0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 4, 0, 0, 0, 0, 1]
K_RMS_RMIN_02 = [-2, -2, -2, -2, 1, -2, 2, -2, 2, -2, -2, -2, -2, -1, -2, 0, -2, 1, -2, 2, -2, 2, -2, -2, -2, -2, -1, -2, 0, -2, 1, -2, 2, -2, 2, -2, -2, -2, -2, -1]          # group recenter applied (R = 0.32 >= 0.2)
K_RMS_THR_010 = [-2, -2, -2, -2, 1, -2, 2, -2, 2, -1, 2, 0, -2, -1, -2, 0, -2, 1, -2, 2, -2, 2, -1, 2, 0, -2, -1, -2, 0, -2, 1, -2, 2, -2, 2, -1, 2, 0, -2, -1]          # threshold 0.10 (the curator's Klusters setting), rmin 0.2
K_RMS_RMIN_04 = [0, 0, 0, 0, 3, 0, 4, 0, 4, 0, 0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 4, 0, 0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 4, 0, 0, 0, 0, 1]          # skipped: R = 0.32 < 0.4 -> same as default


def _lab(n):
    return np.full(n, 2)


def test_default_mode_matches_klusters():
    W = golden_cluster()
    _, ioff = fr.klusters_offsets(W, _lab(len(W)), PEAK0, max_shift=MAX_SHIFT, min_n=1)
    assert ioff.tolist() == K_DEFAULT


def test_rms_mode_matches_klusters():
    W = golden_cluster()
    _, a = fr.klusters_offsets(W, _lab(len(W)), PEAK0, max_shift=MAX_SHIFT, min_n=1, recenter=True, rmin=0.2)
    _, b = fr.klusters_offsets(W, _lab(len(W)), PEAK0, max_shift=MAX_SHIFT, min_n=1, recenter=True, rmin=0.4)
    assert a.tolist() == K_RMS_RMIN_02 and b.tolist() == K_RMS_RMIN_04
    assert set((np.array(K_RMS_RMIN_02) - np.array(K_DEFAULT)).tolist()) == {-2}


def test_threshold_010_matches_klusters():
    W = golden_cluster()
    _, a = fr.klusters_offsets(W, _lab(len(W)), PEAK0, max_shift=MAX_SHIFT, min_n=1, min_score=0.10,
                               recenter=True, rmin=0.2)
    assert a.tolist() == K_RMS_THR_010


def test_defaults_are_klusters():
    """threshold 0.70, 2 iterations, max_shift = max(1, peak // 2) clamped to T - 1."""
    W = golden_cluster()
    a = fr.klusters_offsets(W, _lab(len(W)), PEAK0, min_n=1)[1]
    b = fr.klusters_offsets(W, _lab(len(W)), PEAK0, max_shift=max(1, PEAK0 // 2), iters=2, min_score=0.70,
                            min_n=1)[1]
    assert np.array_equal(a, b)


def test_rms_recenter_shift():
    e = np.zeros(32); e[20:23] = [1, 4, 1]          # single-lobed, centroid 21
    assert fr.rms_recenter_shift(e, 18)[0] == 3
    assert fr.rms_recenter_shift(np.ones(32), 18)[3] is False      # flat: R = 0 -> skipped


def test_waveforms_of_a_lag_token():
    """custody contract: a _D feature-space token's .spk is its WAVEFORM token's (no _D waveform set);
    an exact _D .spk still wins; RAW waveforms are never aligned in place of a stderiv request."""
    import tempfile
    from fiber_kit import neuro_io as nio
    with tempfile.TemporaryDirectory() as d:
        b = os.path.join(d, "s")
        w = np.zeros((3, 4, 2), np.int16)
        nio.write_spk(b, 6, w, variant="stderiv_C5")
        nio.write_spk(b, 6, w + 1, variant="standard")
        _, r = fr.open_waveforms(b, 6, 4, 2, "stderiv_C5_D34")
        assert os.path.basename(r.path) == "s.spk.stderiv_C5.6"
        assert fr._resolve_variant_token(b, 6, "stderiv_C5_D34") == "stderiv_C5"
        nio.write_spk(b, 6, w + 2, variant="stderiv_C5_D34")
        assert os.path.basename(fr.open_waveforms(b, 6, 4, 2, "stderiv_C5_D34")[1].path) == "s.spk.stderiv_C5_D34.6"
    with tempfile.TemporaryDirectory() as d:
        b = os.path.join(d, "s")
        nio.write_spk(b, 6, np.zeros((3, 4, 2), np.int16), variant="standard")
        try:
            fr.open_waveforms(b, 6, 4, 2, "stderiv_C5_D34")
            raise AssertionError("raw waveforms were accepted for a stderiv token")
        except FileNotFoundError as e:
            assert "refusing" in str(e)


def test_plain_orders_match_computeSDiff():
    """_S1/_S2/_S3 re-extraction == the extractor's computeSDiff (FIRST / LAPLACIAN / ALLPAIRS), ported
    literally per channel, then round-half-away, int16 clamp, temporal first-difference, clamp."""
    from fiber_kit import fiber_stderiv as fst
    rng = np.random.default_rng(3)
    raw = rng.integers(-9000, 9000, (5, 13, 8)).astype(np.float64)   # large enough to hit the clamp

    def compute_sdiff(x, idx, n, order):
        v = x[idx]
        if order == 1:
            return v - x[idx + 1] if idx < n - 1 else v - x[idx - 1]
        if order == 2:
            if idx == 0:
                return v - x[1]
            if idx == n - 1:
                return v - x[n - 2]
            return v - 0.5 * (x[idx - 1] + x[idx + 1])
        return n * v - x.sum()

    for order in (1, 2, 3):
        sd = np.array([[[compute_sdiff(raw[k, t], i, 8, order) for i in range(8)] for t in range(13)]
                       for k in range(5)])
        sd = np.clip(np.sign(sd) * np.floor(np.abs(sd) + 0.5), -32768, 32767)
        want = np.clip(sd[:, 1:] - sd[:, :-1], -32768, 32767).astype(np.int16)
        partner, sets = fst.plain_order_spec(order, 8)
        got = fst.apply_stderiv_transform(raw, partner=partner, sets=sets)
        assert np.array_equal(got, want), order


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("ok", fn.__name__)
    print(f"\n{len(fns)} passed")
