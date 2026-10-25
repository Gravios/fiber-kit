"""Shared-file safety: atomic writes, two-phase dedup, length checks, realign commit order.

  * neuro_io.atomic_write: a block that raises leaves the old file and no temp file;
    every neuro_io writer goes through it (a failing write keeps the old .res/.clu/.spk/.fet);
  * read_clu_file(n_spikes=) raises on a .clu of another length (was: truncate / return short);
  * an empty .spk reads as 0 spikes (was: np.memmap raised);
  * apply_spike_keep: a rewrite that fails (full disk) raises and leaves the group untouched -- no file
    subset, none quarantined, no temp left (was: the file was truncated, renamed .stalebkp, success);
  * fiber-realign refuses a clu whose length differs from the .res (was: cut .res/.spk to the clu);
  * fiber-realign writes the .res last: a failure while writing windows leaves the .res as it was.
"""
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from fiber_kit import fiber_realign as frl         # noqa: E402
from fiber_kit import neuro_io as nio              # noqa: E402

T, C = 32, 4
fails = 0


def check(ok, what):
    global fails
    print(("  ok:   " if ok else "  FAIL: ") + what); fails += not ok


def session(d, n=200, seed=0):
    rng = np.random.default_rng(seed)
    res = np.sort(rng.choice(np.arange(1000, 10 ** 6), n, replace=False)).astype(np.int64)
    t = np.arange(T)
    tmpl = (-3000 * np.exp(-0.5 * ((t - 15) / 2.0) ** 2))[:, None] * np.ones(C)
    spk = np.stack([np.roll(tmpl, rng.integers(-2, 3), 0) + rng.normal(0, 50, (T, C))
                    for _ in range(n)]).astype(np.int16)
    b = os.path.join(d, "S")
    nio.write_res_file(f"{b}.res.1", res)
    nio.write_spk_file(f"{b}.spk.standard.1", spk)
    nio.write_clu_file(f"{b}.clu.standard.1", np.full(n, 2))
    return b, res, spk


def read_bytes(p):
    with open(p, "rb") as f:
        return f.read()


def files(d):
    return sorted(os.listdir(d))


def realign_args(b, clu):
    return [b, "1", "--clu", clu, "--channels", "0,1,2,3", "--nchan", str(C), "--nsamp", str(T),
            "--peak", "16", "--sr", "20000", "--shift-spk", "--no-emit-clu", "--variants", "all",
            "--variant", "standard"]


def main():
    d = tempfile.mkdtemp(prefix="fk_atomic_")
    try:
        print("atomic_write")
        p = os.path.join(d, "x.res.1")
        nio.write_res_file(p, np.arange(10))
        try:
            with nio.atomic_write(p) as f:
                f.write(b"partial")
                raise OSError("disk full")
        except (OSError, AttributeError):
            pass
        check(hasattr(nio, "atomic_write"), "neuro_io has atomic_write")
        check(np.array_equal(nio.read_res_file(p), np.arange(10)), "a failed block leaves the old file")
        check(files(d) == ["x.res.1"], f"and no temp file ({files(d)})")
        os.remove(p)

        print("read_clu_file length")
        p = os.path.join(d, "x.clu.1")
        nio.write_clu_file(p, np.full(12, 2))
        for n in (10, 14):
            try:
                nio.read_clu_file(p, n_spikes=n); ok = False
            except ValueError:
                ok = True
            check(ok, f"12 labels read for {n} spikes raises")
        check(nio.read_clu_file(p, n_spikes=12)[1].size == 12, "the matching length reads")
        os.remove(p)

        print("empty .spk")
        p = os.path.join(d, "x.spk.1")
        open(p, "wb").close()
        try:
            empty_ok = nio.open_spk_file(p, T, C).shape == (0, T, C)
        except ValueError:
            empty_ok = False
        check(empty_ok, "a 0-byte .spk reads as 0 spikes")
        os.remove(p)

        print("apply_spike_keep: a failed rewrite")
        b, res, spk = session(d)
        keep = np.ones(res.size, bool); keep[::7] = False
        before = {f: read_bytes(os.path.join(d, f)) for f in files(d)}
        real = nio.write_spk_file
        nio.write_spk_file = lambda path, waves: (_ for _ in ()).throw(OSError(28, "No space left on device"))
        try:
            nio.apply_spike_keep(b, 1, keep, res.size, T, C, verbose=False); raised = False
        except RuntimeError:
            raised = True
        finally:
            nio.write_spk_file = real
        after = {f: read_bytes(os.path.join(d, f)) for f in files(d)}
        check(raised, "raises")
        check(after == before, f"every file unchanged, none quarantined, no temp left ({files(d)})")
        real_replace = os.replace
        def flaky(src, dst):                               # the .spk cannot be moved in (Windows: mapped)
            if str(src).endswith(".dedup.tmp") and str(dst).endswith(".spk.standard.1"):
                raise PermissionError(13, "The process cannot access the file")
            return real_replace(src, dst)
        nio.os.replace = flaky
        try:
            nio.apply_spike_keep(b, 1, keep, res.size, T, C, verbose=False); raised = False
        except RuntimeError:
            raised = True
        finally:
            nio.os.replace = real_replace
        after = {f: read_bytes(os.path.join(d, f)) for f in files(d)}
        check(raised and after == before,
              f"a rename that fails mid-commit is rolled back: every file as it was, nothing left over ({files(d)})")
        nio.apply_spike_keep(b, 1, keep, res.size, T, C, verbose=False)
        check(nio.read_res_file(f"{b}.res.1").size == keep.sum()
              and nio.open_spk_file(f"{b}.spk.standard.1", T, C).shape[0] == keep.sum()
              and nio.read_clu_file(f"{b}.clu.standard.1")[1].size == keep.sum(),
              "a run that can write subsets .res, .spk and .clu together")
        check(not any(f.endswith(".tmp") or "stalebkp" in f for f in files(d)), "and leaves no temp/backup")
        for f in files(d):
            os.remove(os.path.join(d, f))

        print("fiber-realign: clu of another length")
        b, res, spk = session(d)
        short = f"{b}.clu.standard.1.stage"
        nio.write_clu_file(short, np.full(150, 2))
        try:
            frl.main(realign_args(b, short)); refused = False
        except SystemExit as e:
            refused = bool(e.code) and e.code != 0
        check(refused, "refused")
        check(np.array_equal(nio.read_res_file(f"{b}.res.1"), res)
              and np.array_equal(np.asarray(nio.open_spk_file(f"{b}.spk.standard.1", T, C)), spk),
              ".res and .spk untouched")
        os.remove(short)

        print("fiber-realign: the .res is committed last")
        clu = f"{b}.clu.standard.1"
        real = nio.write_spk
        nio.write_spk = lambda *a, **k: (_ for _ in ()).throw(OSError(28, "No space left on device"))
        try:
            frl.main(realign_args(b, clu)); died = False
        except OSError:
            died = True
        finally:
            nio.write_spk = real
        check(died, "a failure writing the windows propagates")
        check(np.array_equal(nio.read_res_file(f"{b}.res.1"), res), "and the .res was not shifted")
        frl.main(realign_args(b, clu))
        moved = int(np.count_nonzero(nio.read_res_file(f"{b}.res.1") != res))
        check(moved > 0, f"the full run then moves spikes ({moved})")
        res1 = nio.read_res_file(f"{b}.res.1")
        frl.main(realign_args(b, clu))
        again = int(np.count_nonzero(nio.read_res_file(f"{b}.res.1") != res1))
        check(again == 0, f"and a second run moves none ({again})")
        print("fiber-realign: an interrupted --shift-spk commit is finished from its journal")
        for f in files(d):
            os.remove(os.path.join(d, f))
        b, res, spk = session(d)
        nio.write_spk_file(f"{b}.spk.stderiv.1", spk // 2)                # a second variant to roll
        ref = os.path.join(d, "ref"); os.mkdir(ref)
        for f in ("S.res.1", "S.spk.standard.1", "S.spk.stderiv.1", "S.clu.standard.1"):
            shutil.copy(os.path.join(d, f), ref)
        rb = os.path.join(ref, "S")
        frl.main(realign_args(rb, rb + ".clu.standard.1"))                 # the uninterrupted commit
        calls = {"n": 0}
        real = nio.write_spk

        def second_fails(*a, **k):
            calls["n"] += 1
            if calls["n"] == 2:
                raise OSError(28, "No space left on device")
            return real(*a, **k)
        nio.write_spk = second_fails
        try:
            frl.main(realign_args(b, b + ".clu.standard.1")); died = False
        except OSError:
            died = True
        finally:
            nio.write_spk = real
        jn = [f for f in files(d) if f.endswith(".pending.npz")]
        check(died and len(jn) == 1, f"the interrupted commit leaves its journal ({jn})")
        check(np.array_equal(nio.read_res_file(f"{b}.res.1"), res), "and the .res is not written")
        frl.main(realign_args(b, b + ".clu.standard.1"))                  # finishes it, then aligns again
        same = (np.array_equal(nio.read_res_file(f"{b}.res.1"), nio.read_res_file(rb + ".res.1"))
                and all(np.array_equal(np.asarray(nio.open_spk_file(f"{b}.spk.{v}.1", T, C)),
                                       np.asarray(nio.open_spk_file(f"{rb}.spk.{v}.1", T, C)))
                        for v in ("standard", "stderiv")))
        check(same, "the re-run ends where the uninterrupted commit did: same .res, same windows in both variants")
        check(not any(f.endswith(".pending.npz") for f in files(d)), "and removes the journal")
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print(f"test_atomic_io: {'PASS' if not fails else 'FAIL'} ({fails} failed)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
