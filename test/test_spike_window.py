"""Non-32-sample spike windows: the chunk whitener's baseline snippets must be as long as the spike
window, so a mask built for it (build_masks) indexes inside them.  Regression for nSamples 64 /
peak 33 (Plexon session): IndexError 'index 32 is out of bounds for axis 1 with size 32'."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from fiber_kit import fiber_lib as fl  # noqa: E402


def test_whitener_on_a_64_sample_window():
    rng = np.random.default_rng(0)
    fil = rng.standard_normal((200000, 8)).astype(np.float32)
    spikes = np.sort(rng.choice(np.arange(100, 199000), 300, replace=False))
    m = fl.build_masks(64, 33)
    assert m.full.max() >= 32                       # the case that used to crash
    W, nmean, nb = fl.fil_chunk_whitener(fil, np.arange(8), 0, fil.shape[0], spikes, 64, m.full)
    d = len(m.full) * 8
    assert W.shape == (d, d) and nmean.shape == (d,) and nb > 1000


def test_32_sample_window_unchanged():
    rng = np.random.default_rng(1)
    fil = rng.standard_normal((100000, 8)).astype(np.float32)
    spikes = np.sort(rng.choice(np.arange(100, 99000), 100, replace=False))
    a = fl.chunk_whitener_mm(fil, np.arange(8), 0, fil.shape[0], spikes, mask=fl.MASK_FULL, n_base=500)
    b = fl.fil_chunk_whitener(fil, np.arange(8), 0, fil.shape[0], spikes, 32, fl.MASK_FULL)
    assert fl._baseline_win(fl.MASK_FULL) == 32 and fl._baseline_win(fl.MASK_FULL, 32) == 32
    assert a[0].shape == b[0].shape


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("ok", fn.__name__)
    print(f"\n{len(fns)} passed")
