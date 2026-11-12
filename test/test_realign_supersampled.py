#!/usr/bin/env python3
"""A commit (--reextract / --shift-spk, --variants all) leaves a SUPER-SAMPLED space (_U<factor>) alone and
says so: it cannot rebuild one (factor x nSamples ticks of an interpolated cut, a tick .res, a phase column,
the interpolator it was built with), and deriving it as its base-rate variant -- what it did until 0.67.6 --
wrote a 52-sample window under the _U4 name and left the tick .res behind.  Aligning IN a _U variant is
refused with the base-rate token named."""
import contextlib, io, os, sys, tempfile
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
from fiber_kit import fiber_realign as frl, neuro_io as nio   # noqa: E402
import test_realign_geometry as G                                # noqa: E402

fails = 0


def check(ok, what):
    global fails
    print(("  ok:   " if ok else "  FAIL: ") + what); fails += not ok


def main():
    rng = np.random.default_rng(3)
    nsamp, pk = G.GEOMETRIES[0]
    with tempfile.TemporaryDirectory() as td:
        base, fil, true = G._session(td, nsamp, pk, rng)
        # a super-sampled space beside the base one, as Klusters / ndm_extractspikes --supersample write it
        n = nio.read_res_file(base + ".res.1").size
        u4 = rng.integers(-500, 500, (n, nsamp * 4, G.NCH)).astype(np.int16)
        nio.write_spk(base, 1, u4, variant="standard_U4")
        tick = (nio.read_res_file(base + ".res.1") * 4 + 1).astype(np.int64)
        with open(base + ".res.standard_U4.1", "w") as fh:
            fh.write("\n".join(map(str, tick)) + "\n")
        check(frl.supersampled_variants_on_disk(base, 1) == ["standard_U4"], "the _U4 space is listed as super-sampled")
        check("standard_U4" not in frl._variants_on_disk(base, 1), "and not among the variants a commit writes")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            frl.main([base, "1", "--clu", base + ".clu.standard.1", "--align-algo", "klusters", "--reextract",
                      "--variants", "all", "--variant", "standard", "--no-emit-clu", "--min-score", "0.1", "--refeaturize"])
        res1 = nio.read_res_file(base + ".res.1")
        u4_after = np.fromfile(base + ".spk.standard_U4.1", np.int16)
        check(u4_after.size == u4.size and np.array_equal(u4_after.reshape(u4.shape), u4),
              f"the _U4 .spk is untouched ({u4_after.size} int16 = {u4.size})")
        check(np.array_equal(np.loadtxt(base + ".res.standard_U4.1", dtype=np.int64), tick), "the tick .res is untouched")
        check(not os.path.exists(base + ".fet.standard_U4.1"), "no .fet was invented for it")
        check("standard_U4" in out.getvalue() and "super-sampled" in out.getvalue(),
              "the log says the super-sampled space was left alone and how to rebuild it")
        spk1 = np.array(nio.open_spk_file(base + ".spk.standard.1", nsamp, G.NCH))
        c0 = frl.timestamp_index(pk)
        fresh = np.stack([fil[r - c0:r - c0 + nsamp, :G.NCH] for r in res1])
        check(np.array_equal(spk1, fresh), "the base-rate space was committed as before (a fresh cut at the new times)")
        # aligning IN the _U variant is refused, naming the base-rate token
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                frl.main([base, "1", "--clu", base + ".clu.standard.1", "--variant", "standard_U4", "--check"])
            refused = ""
        except SystemExit as e:
            refused = str(e)
        check("super-sampled" in refused and "standard)" in refused, f"a realign in standard_U4 is refused, base token named ({refused[:80]}...)")
    print(f"test_realign_supersampled: {'PASS' if not fails else 'FAIL'} ({fails} failed)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
