"""fiber_template: LINKED per-unit median templates (drift + adapt) written one
.wtf per variant, plus the .wtf/.wti IO.  Synthetic session (res/clu/spk in
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
        nio.write_wti(b, 6, rows, NS, NCH, sr=SR, tag="t")
        wti = nio.read_wti(b, 6, tag="t")
        assert wti["version"] == 1 and wti["nSamples"] == NS and wti["nChannels"] == NCH
        info = wti["rows"]
        assert len(info) == 1 and info[0]["unit"] == 2 and info[0]["row"] == 0
        assert info[0]["link"] == "drift" and info[0]["bin"] == 0
        assert info[0]["a"] == 0.0 and info[0]["b"] == 1.0 and info[0]["nSpikes"] == 10
        assert info[0]["src_clu_variant"] == "stderiv_C5_D34"   # provenance round-trips


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
        info = nio.read_wti(b, 6, tag="wtf")["rows"]
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


def test_align_to_res():
    nsamp, nchan = 20, 2
    w = np.zeros((1, nsamp, nchan), np.float32); w[0, 10, 0] = 5.0   # peak at sample 10
    # aligned[t] = w[t + offset]: a +3 offset moves the peak 10 -> 7, a -3 moves it -> 13.
    assert np.argmax(ft._align_to_res(w, np.array([3]), nsamp)[0, :, 0]) == 7
    assert np.argmax(ft._align_to_res(w, np.array([-3]), nsamp)[0, :, 0]) == 13
    assert np.array_equal(ft._align_to_res(w, np.array([0]), nsamp), w)   # offset 0 == identity
    # samples rolled in from outside the window are zero-filled
    assert ft._align_to_res(w, np.array([3]), nsamp)[0, 18, 0] == 0.0


def test_eap_tcl_io_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        b = os.path.join(d, "s")
        N, T = 5, 4
        cells = np.full((N, T), nio.EAP_ABSENT, np.int8)
        cells[0, 1] = 0            # offset 0 is PRESENT (not the -128 sentinel)
        cells[1, 0] = -5; cells[1, 2] = 7   # spike 1 is a collision (classes 0 and 2)
        cells[3, 2] = -127
        nio.write_eap(b, 6, cells, group=6, tag="st")
        e = nio.read_eap(b, 6, tag="st")
        assert e["ok"] and e["nSpikes"] == N and e["nClasses"] == T and e["group"] == 6
        assert np.array_equal(e["cells"], cells)
        row1 = list(np.flatnonzero(e["cells"][1] != nio.EAP_ABSENT))
        assert row1 == [0, 2] and e["cells"][1, 0] == -5 and e["cells"][1, 2] == 7
        assert not nio.read_eap(b, 6, tag="nope")["ok"]

        entries = [dict(col=0, status="active", label="CA1 pyr a", provenance_clu=23,
                        provenance_stage="gt", created="2026-10-02"),
                   dict(col=1, status="merged", merged_into=0),
                   dict(col=2, status="tomb", provenance_clu=41),
                   dict(col=3, status="free")]
        nio.write_tcl(b, 6, entries, n_classes=4)
        r = nio.read_tcl(b, 6)
        assert r["ok"] and r["nClasses"] == 4 and len(r["entries"]) == 4
        assert (r["entries"][0]["status"] == "active" and r["entries"][0]["label"] == "CA1 pyr a"
                and r["entries"][0]["provenance_clu"] == 23 and r["entries"][0]["provenance_stage"] == "gt")
        assert r["entries"][1]["status"] == "merged" and r["entries"][1]["merged_into"] == 0
        assert r["entries"][2]["status"] == "tomb" and r["entries"][2]["provenance_clu"] == 41
        assert r["entries"][3]["status"] == "free" and r["entries"][3]["label"] == ""


def test_eap_mode_matches_clu_on_clean_members():
    """EAP-mode templating selects members from .eap columns and stamps the class
    id into .wti; with all-offset-0 (clean) membership mapped from the clu, each
    class's template is byte-identical to the clu-mode template of its source unit
    — proving membership-from-eap + class-id stamping + offset-0 == raw."""
    with tempfile.TemporaryDirectory() as d:
        b = _make_session(d)
        _, clu = nio.read_clu_at(b, 6, variant="stderiv_C5_D34", tag="lab_units")
        N, T = clu.size, 8
        colmap = {2: 5, 3: 6}                    # class id (column) != clu id, deliberately
        cells = np.full((N, T), nio.EAP_ABSENT, np.int8)
        for i in range(N):
            c = colmap.get(int(clu[i]))
            if c is not None:
                cells[i, c] = 0                  # clean: offset 0
        nio.write_eap(b, 6, cells, group=6, tag="lab_units")     # tag == clu_tag/stage
        nio.write_tcl(b, 6, [dict(col=j, status=("active" if j in (5, 6) else "free"),
                                  label={5: "u2", 6: "u3"}.get(j, ""),
                                  provenance_clu={5: 2, 6: 3}.get(j, -1)) for j in range(T)],
                      n_classes=T)

        vs = ["standard", "stderiv_C5_D34"]
        rows_e, _ = ft.generate(b, 6, nsamp=NS, nchan=NCH, sr=SR, variants=vs,
                                clu_variant="stderiv_C5_D34", clu_tag="lab_units",
                                out_tag="eapstage", links=("drift",), n_chunks=2, eap=True)
        # units defaulted from the .tcl active set -> classes 5 and 6, stamped as `class`.
        assert sorted({r["class"] for r in rows_e}) == [5, 6]

        rows_c, _ = ft.generate(b, 6, nsamp=NS, nchan=NCH, sr=SR, variants=vs,
                                clu_variant="stderiv_C5_D34", clu_tag="lab_units",
                                out_tag="clustage", links=("drift",), n_chunks=2,
                                eap=False, units=[2, 3])

        Te = nio.read_wtf(b, 6, NS, NCH, variant="standard", tag="eapstage")
        Tc = nio.read_wtf(b, 6, NS, NCH, variant="standard", tag="clustage")
        ie = {(r["unit"], r["link"], r["bin"]): k for k, r in enumerate(nio.read_wti(b, 6, tag="eapstage")["rows"])}
        ic = {(r["unit"], r["link"], r["bin"]): k for k, r in enumerate(nio.read_wti(b, 6, tag="clustage")["rows"])}
        # class 5 (eap) == clu unit 2; class 6 == clu unit 3. Same members, offset 0 -> identical median.
        assert np.array_equal(Te[ie[(5, "drift", 0)]], Tc[ic[(2, "drift", 0)]])
        assert np.array_equal(Te[ie[(6, "drift", 0)]], Tc[ic[(3, "drift", 0)]])


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("ok", fn.__name__)
    print(f"\n{len(fns)} passed")
