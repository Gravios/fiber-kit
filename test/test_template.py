"""fiber_template: LINKED per-unit median templates (drift + adapt) written one
.wtf per variant, plus the .wtf/.wtfinfo IO.  Synthetic session (res/clu/spk in
two variants) with a unit whose amplitude varies — so the drift series (per time
chunk) and the adapt series (per energy bin) both resolve the variation, the
point of "templates like fibers"."""
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from fiber_kit import neuro_io as nio          # noqa: E402
from fiber_kit import fiber_template as ft      # noqa: E402

NS, NCH, SR = 32, 4, 1000.0


def _trough(center, amp, ch):
    w = np.zeros((NS, NCH), np.float32)
    t = np.arange(NS)
    w[:, ch] = -amp * np.exp(-0.5 * ((t - center) / 3.0) ** 2)
    return w


def _make_session(d):
    b = os.path.join(d, "sess")
    rng = np.random.default_rng(0)
    times, clu, w_std, w_sd = [], [], [], []

    def add(t, u, amp_std, amp_sd, ch):
        times.append(t); clu.append(u)
        w_std.append(_trough(16, amp_std, ch) + rng.standard_normal((NS, NCH)) * 5)
        w_sd.append(_trough(16, amp_sd, ch) + rng.standard_normal((NS, NCH)) * 5)

    # unit 2: amplitude 1000 in chunk 0 (t<10s) -> 600 in chunk 1 (drift AND a
    # bimodal energy distribution, so the adapt bins separate it too).
    for i in range(60):
        add(int((i / 60) * 10 * SR), 2, 1000, 700, ch=1)
    for i in range(60):
        add(int((10 + (i / 60) * 10) * SR), 2, 600, 500, ch=1)
    for i in range(40):                      # unit 3, chunk 0 only, channel 2
        add(int((i / 40) * 9 * SR), 3, 900, 650, ch=2)

    order = np.argsort(times)
    times = np.asarray(times, np.int64)[order]
    clu = np.asarray(clu, np.int64)[order]
    w_std = np.rint(np.stack(w_std)[order]).astype(nio.SPK_DTYPE)
    w_sd = np.rint(np.stack(w_sd)[order]).astype(nio.SPK_DTYPE)

    nio.write_res(b, 6, times, variant="stderiv")
    nio.write_clu(b, 6, clu, variant="stderiv_C5_D34", tag="lab_units")
    nio.write_spk(b, 6, w_std, variant="standard")
    nio.write_spk(b, 6, w_sd, variant="stderiv_C5_D34")
    return b


def test_wtf_io_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        b = os.path.join(d, "s")
        arr = (np.random.default_rng(1).standard_normal((5, NS, NCH)) * 100).astype(nio.SPK_DTYPE)
        nio.write_wtf(b, 6, arr, variant="standard", tag="t")
        back = nio.read_wtf(b, 6, NS, NCH, variant="standard", tag="t")
        assert back.shape == (5, NS, NCH)
        assert np.array_equal(back.astype(nio.SPK_DTYPE), arr)       # byte-identical, .spk-style
        rows = [dict(unit=2, link="drift", bin=0, lo=0.0, hi=1.0, nspk=10,
                     src_clu_variant="stderiv_C5_D34", src_clu_tag="lab_units")]
        nio.write_wtf_info(b, 6, rows, tag="t")
        info = nio.read_wtf_info(b, 6, tag="t")
        assert len(info) == 1 and info[0]["unit"] == 2 and info[0]["row"] == 0
        assert info[0]["link"] == "drift" and info[0]["bin"] == 0


def test_drift_and_adapt_series():
    with tempfile.TemporaryDirectory() as d:
        b = _make_session(d)
        rows, paths = ft.generate(
            b, 6, nsamp=NS, nchan=NCH, sr=SR,
            variants=["standard", "stderiv_C5_D34"],
            clu_variant="stderiv_C5_D34", clu_tag="lab_units",
            out_tag="wtf", links=("drift", "adapt"), n_chunks=2, n_energy=2)

        T_std = nio.read_wtf(b, 6, NS, NCH, variant="standard", tag="wtf")
        T_sd = nio.read_wtf(b, 6, NS, NCH, variant="stderiv_C5_D34", tag="wtf")
        info = nio.read_wtf_info(b, 6, tag="wtf")
        assert T_std.shape[0] == T_sd.shape[0] == len(info) == len(rows)

        idx = {(r["unit"], r["link"], r["bin"]): i for i, r in enumerate(info)}
        # DRIFT: unit 2 present in both chunks; amplitude larger in chunk 0 (1000) than 1 (600)
        assert (2, "drift", 0) in idx and (2, "drift", 1) in idx
        a0 = np.abs(T_std[idx[(2, "drift", 0)]]).max()
        a1 = np.abs(T_std[idx[(2, "drift", 1)]]).max()
        assert a0 > a1 + 100, (a0, a1)
        # ADAPT: two energy bins for unit 2; the high-E bin has the larger template
        assert (2, "adapt", 0) in idx and (2, "adapt", 1) in idx
        lo = np.abs(T_std[idx[(2, "adapt", 0)]]).max()
        hi = np.abs(T_std[idx[(2, "adapt", 1)]]).max()
        assert hi > lo + 100, (lo, hi)                 # energy-ordered bins
        # the two variants differ for the same (unit,link,bin)
        assert np.abs(T_std[idx[(2, "drift", 0)]]).max() > np.abs(T_sd[idx[(2, "drift", 0)]]).max()
        # unit 3 only in chunk 0, on channel 2
        assert (3, "drift", 0) in idx and (3, "drift", 1) not in idx
        assert int(np.unravel_index(np.argmin(T_std[idx[(3, "drift", 0)]]), (NS, NCH))[1]) == 2
        assert os.path.exists(paths["info"])


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("ok", fn.__name__)
    print(f"\n{len(fns)} passed")
