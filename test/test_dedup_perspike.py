"""apply_spike_keep (fiber-refine --dedup) and the per-spike files of a group.

  * a super-sampled space: <base>.spk.standard_U4.<g> holds 4 x nSamples samples per channel.  It is
    subset with its .fet / .clu / tick .res, row for row (was: read at nSamples, 4 x the spike count,
    quarantined as stale while the rest of its space was subset);
  * the EAP membership table <base>.eap.<g> (one row per spike) is subset too (was: left at the old
    count, so every row after the first dropped spike named another spike);
  * a Klusters working copy (<file>.pending) refuses the dedup and leaves every file as it was;
  * a binary group file of an unknown type: a warning by default, a refusal with stale='error';
    known non-per-spike files (.pca, .nst, .clp) and text sidecars raise nothing.
"""
import contextlib
import io
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from fiber_kit import neuro_io as nio              # noqa: E402

N, T, C, F = 100, 8, 4, 4
fails = 0


def check(ok, what):
    global fails
    print(("  ok:   " if ok else "  FAIL: ") + what); fails += not ok


def session(d):
    """A group with the standard space, a x4 super-sampled space and an EAP table."""
    rng = np.random.default_rng(0)
    b = os.path.join(d, "s")
    made = dict(
        res=np.sort(rng.integers(0, 10**6, N)).astype(np.int64),
        clu=rng.integers(2, 5, N).astype(np.int64),
        spk=rng.integers(-100, 100, (N, T, C)).astype(np.int16),
        spk_u=rng.integers(-100, 100, (N, T * F, C)).astype(np.int16),
        phase=rng.integers(0, F, N).astype(np.int64),
        eap=rng.integers(-3, 4, (N, 3)).astype(np.int8),
    )
    made["fet"] = np.c_[rng.integers(0, 9, (N, 3)), made["res"]].astype(np.int64)
    made["fet_u"] = np.c_[rng.integers(0, 9, (N, 3)), made["phase"], made["res"]].astype(np.int64)
    nio.write_res_file(f"{b}.res.standard.1", made["res"])
    nio.write_res_file(f"{b}.res.standard_U4.1", made["res"] * F + made["phase"])
    nio.write_clu_file(f"{b}.clu.standard.1", made["clu"])
    nio.write_clu_file(f"{b}.clu.standard_U4.1", made["clu"])
    nio.write_spk_file(f"{b}.spk.standard.1", made["spk"])
    nio.write_spk_file(f"{b}.spk.standard_U4.1", made["spk_u"])
    nio.write_fet_file(f"{b}.fet.standard.1", made["fet"])
    nio.write_fet_file(f"{b}.fet.standard_U4.1", made["fet_u"])
    nio.write_eap(b, 1, made["eap"], group=1)
    return b, made


def snapshot(d):
    out = {}
    for f in sorted(os.listdir(d)):
        with open(os.path.join(d, f), "rb") as fh:
            out[f] = fh.read()
    return out


def stub(d, name):
    """A 16-byte file standing in for a group file of some other type."""
    with open(os.path.join(d, name), "wb") as fh:
        fh.write(b"\x00" * 16)


def main():
    keep = np.ones(N, bool); keep[::10] = False
    k = int(keep.sum())

    print("a super-sampled space and an EAP table")
    with tempfile.TemporaryDirectory() as d:
        b, m = session(d)
        nio.apply_spike_keep(b, 1, keep, N, T, C, verbose=False)
        names = sorted(os.listdir(d))
        check(not any("stalebkp" in f or f.endswith(".tmp") for f in names),
              f"nothing quarantined, no temp left ({names})")
        spk_u = nio.open_spk_file(f"{b}.spk.standard_U4.1", T * F, C)
        check(spk_u.shape == (k, T * F, C), f"the _U4 .spk holds {k} rows of {T * F} samples (shape {spk_u.shape})")
        check(np.array_equal(np.asarray(spk_u), m["spk_u"][keep]), "... the kept rows, byte for byte")
        check(np.array_equal(np.asarray(nio.open_spk_file(f"{b}.spk.standard.1", T, C)), m["spk"][keep]),
              "the base-rate .spk is the kept rows")
        check(np.array_equal(nio.read_fet_file(f"{b}.fet.standard_U4.1").values, m["fet_u"][keep]),
              "the _U4 .fet (phase and time columns included) is the kept rows")
        check(np.array_equal(nio.read_res_file(f"{b}.res.standard_U4.1"), (m["res"] * F + m["phase"])[keep]),
              "the tick .res is the kept rows")
        check(np.array_equal(nio.read_clu_file(f"{b}.clu.standard_U4.1", n_spikes=k)[1], m["clu"][keep]),
              "the _U4 .clu is the kept rows")
        e = nio.read_eap(b, 1)
        check(e["ok"] and e["nSpikes"] == k and e["group"] == 1 and np.array_equal(e["cells"], m["eap"][keep]),
              f"the .eap holds the kept rows and its header says {k} (nSpikes {e['nSpikes']})")
        before = snapshot(d)
        nio.apply_spike_keep(b, 1, keep, N, T, C, verbose=False)
        check(snapshot(d) == before, "a second run changes nothing")

    print("a Klusters working copy")
    with tempfile.TemporaryDirectory() as d:
        b, m = session(d)
        nio.write_spk_file(f"{b}.spk.standard.1.pending", m["spk"])
        before = snapshot(d)
        try:
            nio.apply_spike_keep(b, 1, keep, N, T, C, verbose=False); msg = ""
        except RuntimeError as e:
            msg = str(e)
        check("working cop" in msg and "s.spk.standard.1.pending" in msg, "the dedup refuses and names the file")
        check(snapshot(d) == before, "every file as it was")

    print("group files of other types")
    with tempfile.TemporaryDirectory() as d:
        b, m = session(d)
        for name in ("s.pca.standard.1", "s.nst.standard.1", "s.clp.standard.1", "s.notes.1.tsv",
                     "s.probe.1.probe", "s.spk.standard.2"):
            stub(d, name)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            nio.apply_spike_keep(b, 1, keep, N, T, C, verbose=False)
        check(err.getvalue() == "", "known non-per-spike files, text sidecars and another group: no warning")
        check(all(os.path.getsize(os.path.join(d, n)) == 16 for n in ("s.pca.standard.1", "s.spk.standard.2")),
              "... and they are untouched")
    with tempfile.TemporaryDirectory() as d:
        b, m = session(d)
        stub(d, "s.zzz.standard.1")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            nio.apply_spike_keep(b, 1, keep, N, T, C, verbose=False)
        check("s.zzz.standard.1" in err.getvalue() and "WARNING" in err.getvalue(),
              "an unknown binary type is named in a warning")
        check(nio.read_res_file(f"{b}.res.standard.1").size == k, "... and the group is still deduped")
    with tempfile.TemporaryDirectory() as d:
        b, m = session(d)
        stub(d, "s.zzz.standard.1")
        before = snapshot(d)
        try:
            nio.apply_spike_keep(b, 1, keep, N, T, C, verbose=False, stale="error"); msg = ""
        except RuntimeError as e:
            msg = str(e)
        check("s.zzz.standard.1" in msg and snapshot(d) == before,
              "with stale='error' it is refused and every file is as it was")

    print(f"test_dedup_perspike: {'PASS' if not fails else 'FAIL'} ({fails} failed)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
