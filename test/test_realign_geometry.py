"""A realign commit cuts its windows where neurosuite-3 cuts them -- the timestamp at peakSampleIndex - 1,
which is also the sample the klusters algorithm aligns each template's peak to.

On synthetic sessions extracted the way neurosuite-3's extractor does it, `fiber-realign --reextract`
(the commit fiber-session runs at the end of a sort) must give windows that

  * equal a fresh neurosuite-3 cut at the committed times,
  * hold each template's peak on the alignment target (peakSampleIndex - 1),
  * are a fixed point: a second realign on them moves nothing,
  * and committed times that are the true spike times (the detection jitter is gone).

The window geometry is the YAML's, per group, so three are tried, one with the peak placed far from the
middle of the window (data sets differ in where it sits).  Negative control: the cut fiber-realign used
until 0.51 (at peakSampleIndex, one sample late) fails the window, target and fixed-point checks.
"""
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from fiber_kit import fiber_realign as frl   # noqa: E402
from fiber_kit import neuro_io as nio        # noqa: E402

SR, NCH, NTOTAL = 20000, 3, 4
GEOMETRIES = [(42, 21), (32, 16), (42, 12)]          # (nSamples, peakSampleIndex); 12 = peak far from centre


def _session(td, nsamp, pk, rng):
    """One group, one unit: a .fil, jittered detections, neurosuite-3-geometry .spk.standard, a .clu."""
    base = os.path.join(td, "geo")
    n_t = 300000
    fil = rng.normal(0, 20, (n_t, NTOTAL))
    t = np.arange(-8, 9)
    wav = -3000 * np.exp(-0.5 * (t / 1.5) ** 2) + 900 * np.exp(-0.5 * ((t - 5) / 2.5) ** 2)
    true = np.sort(rng.choice(np.arange(1000, n_t - 1000, 70), 300, replace=False))
    for s in true:
        fil[s + t, :NCH] += wav[:, None] * np.array([1.0, 0.6, 0.3])[None, :]
    fil = fil.round().clip(-32768, 32767).astype(np.int16)
    fil.tofile(base + ".fil")
    det = (true + rng.integers(-2, 3, true.size)).astype(np.int64)
    det.tofile(base + ".res.1")
    c0 = frl.timestamp_index(pk)                                  # neurosuite-3's window start offset
    spk = np.stack([fil[d - c0:d - c0 + nsamp, :NCH] for d in det])
    nio.write_spk(base, 1, spk, variant="standard")
    nio.write_clu_file(base + ".clu.standard.1", np.full(det.size, 2))
    with open(base + ".yaml", "w") as fh:
        fh.write(f"acquisitionSystem: {{nBits: 16, nChannels: {NTOTAL}, samplingRate: {SR}}}\n"
                 "spikeDetection:\n  channelGroups:\n    - channels: [0, 1, 2]\n"
                 f"      nSamples: {nsamp}\n      peakSampleIndex: {pk}\n")
    return base, fil, true


def _commit(base):
    frl.main([base, "1", "--clu", base + ".clu.standard.1", "--align-algo", "klusters", "--reextract",
              "--variants", "standard", "--variant", "standard", "--no-emit-clu", "--min-score", "0.1"])
    res = nio.read_res_file(base + ".res.1")
    off = np.load(base + ".offsets.1.npy")
    return res, off


def _measure(base, fil, true, nsamp, pk):
    res1, _ = _commit(base)
    spk1 = np.array(nio.open_spk_file(base + ".spk.standard.1", nsamp, NCH))   # a copy: pass 2 rewrites the file
    c0 = frl.timestamp_index(pk)
    fresh = np.stack([fil[r - c0:r - c0 + nsamp, :NCH] for r in res1])
    window = bool(np.array_equal(spk1, fresh))
    peak_at = int(np.argmax(np.abs(spk1.astype(float).mean(0)).sum(1)))
    _, off2 = _commit(base)                                          # a second pass on the committed set
    return dict(window=window, peak_at=peak_at,
                moved2=float(np.mean(np.rint(off2) != 0)), true=float(np.mean(res1 == true)))


def main():
    ok = 0

    def check(name, cond):
        nonlocal ok
        assert cond, name
        ok += 1
        print(f"  ok  {name}")

    rng = np.random.default_rng(5)
    for nsamp, pk in GEOMETRIES:
        with tempfile.TemporaryDirectory() as td:
            base, fil, true = _session(td, nsamp, pk, rng)
            m = _measure(base, fil, true, nsamp, pk)
        tag = f"{nsamp} samples, peakSampleIndex {pk}"
        check(f"{tag}: committed windows = a neurosuite-3 cut at the committed times", m["window"])
        check(f"{tag}: template peak on the target, index {pk - 1} (got {m['peak_at']})", m["peak_at"] == pk - 1)
        check(f"{tag}: a second realign moves nothing ({100 * m['moved2']:.1f}%)", m["moved2"] <= 0.01)
        check(f"{tag}: committed times are the true times ({100 * m['true']:.1f}%)", m["true"] >= 0.95)

    # negative control: the pre-0.51 cut (one sample after the target) is caught by these checks
    orig = frl.reextract_from_fil
    frl.reextract_from_fil = lambda filmm, gch, res, nsamp, peak, batch=50000: orig(filmm, gch, res, nsamp,
                                                                                     peak + 1, batch)
    try:
        nsamp, pk = GEOMETRIES[0]
        with tempfile.TemporaryDirectory() as td:
            base, fil, true = _session(td, nsamp, pk, rng)
            m = _measure(base, fil, true, nsamp, pk)
    finally:
        frl.reextract_from_fil = orig
    check(f"NEGATIVE CONTROL: the old cut fails the window check, puts the peak at {m['peak_at']} and a second "
          f"pass moves {100 * m['moved2']:.0f}%",
          not m["window"] and m["peak_at"] == pk and m["moved2"] > 0.5)
    print(f"test_realign_geometry: {ok} checks passed")


if __name__ == "__main__":
    main()
