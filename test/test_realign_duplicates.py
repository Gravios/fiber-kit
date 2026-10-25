"""Two detections of one spike that a realign stacks on one sample.

A unit where 40 spikes were detected twice (3 samples apart, the second crossing of one spike).  The
klusters alignment puts both detections of each on the spike's true time: 40 same-unit pairs on one
sample with identical windows -- the spike counted twice.

  * --park-duplicates moves the later of each pair to a new duplicates cluster (the next free id, never
    0: only a derivative outlier goes to the artifact cluster), as ONE child atom under it in the .clc/.clp; every other label is unchanged, no row is removed, the hierarchy reproduces its .clu and
    reloads without orphans; <base>.realign.<variant>.<group>.tsv lists the 40 pairs as parked;
  * without it the labels are untouched and the pairs are only listed;
  * a clu named without a variant token (<base>.clu.<group>.<stage>) is parked with its own .clc/.clp,
    and no other sort is touched;
  * a side-by-side commit (--out-tag) leaves the canonical .res and the input clu alone: the parking
    is in the clu it emits under the tag, and the pair list is named after that tag;
  * fiber-session's own commit always asks for the parking, and its --out-variant alias (a copy of
    the sort, written before the commit) carries it too.
Negative control: the same unit without double detections leaves nothing to park and writes no list.
"""
import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from fiber_kit import fiber_pca as fpca               # noqa: E402
from fiber_kit import fiber_realign as frl            # noqa: E402
from fiber_kit import fiber_session as fs             # noqa: E402
from fiber_kit import fiber_stderiv as fst            # noqa: E402
from fiber_kit import neuro_io as nio                 # noqa: E402
from fiber_kit.fiber_refiberize import FiberHierarchy  # noqa: E402

SR, NCH, NTOTAL, NSAMP, PK = 20000, 3, 4, 42, 21


def _session(td, doubles):
    rng = np.random.default_rng(17)
    base = os.path.join(td, "dup")
    n_t = 300000
    fil = rng.normal(0, 20, (n_t, NTOTAL))
    t = np.arange(-8, 9)
    wav = -3000 * np.exp(-0.5 * (t / 1.5) ** 2) + 900 * np.exp(-0.5 * ((t - 5) / 2.5) ** 2)
    true = np.sort(rng.choice(np.arange(1000, n_t - 1000, 150), 400, replace=False))
    for s in true:
        fil[s + t, :NCH] += wav[:, None] * np.array([1.0, 0.6, 0.3])[None, :]
    fil = fil.round().clip(-32768, 32767).astype(np.int16)
    fil.tofile(base + ".fil")
    det = true + rng.integers(-2, 3, true.size)
    if doubles:
        det = np.concatenate([det, true[::10] + 3])          # 40 second detections of one spike
    det = np.sort(det).astype(np.int64)
    det.tofile(base + ".res.1")
    c0 = frl.timestamp_index(PK)
    nio.write_spk(base, 1, np.stack([fil[d - c0:d - c0 + NSAMP, :NCH] for d in det]), variant="standard")
    child = np.where(np.arange(det.size) % 2 == 0, 2, 3)   # two atoms of fiber 2
    FiberHierarchy(child, {2: 2, 3: 2}).save(base, 1, variant="standard", tag="st", renumber=False,
                                             backup=False)
    with open(base + ".yaml", "w") as fh:
        fh.write(f"acquisitionSystem: {{nBits: 16, nChannels: {NTOTAL}, samplingRate: {SR}}}\n"
                 "spikeDetection:\n  channelGroups:\n    - channels: [0, 1, 2]\n"
                 f"      nSamples: {NSAMP}\n      peakSampleIndex: {PK}\n")
    return base, det


def _alias_session(td):
    """A small fiber-session input with double detections: three units, every sixth spike detected twice."""
    rng = np.random.default_rng(3)
    base = os.path.join(td, "mini")
    sr, nsamp, pk, n_t = 20000, 32, 15, 60 * 20000
    fil = rng.normal(0, 25, (n_t, NTOTAL))
    t = np.arange(-8, 9)
    true = []
    for amp, sh in zip((2600.0, 1800.0, 2200.0), ([1.0, 0.5, 0.2], [0.3, 1.0, 0.6], [0.2, 0.4, 1.0])):
        tt = np.sort(rng.choice(np.arange(sr, n_t - sr, 37), 260, replace=False))
        wav = -amp * np.exp(-0.5 * (t / 1.6) ** 2) + 0.35 * amp * np.exp(-0.5 * ((t - 5) / 2.5) ** 2)
        for s_ in tt:
            fil[s_ + t, :NCH] += wav[:, None] * np.array(sh)[None, :]
        true.append(tt)
    fil.round().clip(-32768, 32767).astype(np.int16).tofile(base + ".fil")
    true = np.sort(np.concatenate(true))
    det = np.sort(np.concatenate([true + rng.integers(-1, 2, true.size), true[::6] + 2])).astype(np.int64)
    det.tofile(base + ".res.1")
    raw = frl.reextract_from_fil(nio.open_signal(base + ".fil", NTOTAL), [0, 1, 2], det, nsamp + 1,
                                 frl.timestamp_index(pk) + 1)
    nio.write_spk(base, 1, raw[:, 1:, :], variant="standard")
    nio.write_spk(base, 1, fst.apply_stderiv_transform(raw), variant="stderiv")
    for v in ("standard", "stderiv"):
        x = np.fromfile(f"{base}.spk.{v}.1", np.int16).reshape(-1, nsamp, NCH).astype(float)
        means, evec = fpca.fit_basis(x[:, 8:24, :], 3, centered=False)
        fpca.write_pcad(f"{base}.pca.{v}.1", means, evec, 8, centered=0)
    with open(base + ".yaml", "w") as fh:
        fh.write(f"acquisitionSystem: {{nBits: 16, nChannels: {NTOTAL}, samplingRate: {sr}}}\n"
                 "spikeDetection:\n  channelGroups:\n    - channels: [0, 1, 2]\n"
                 f"      nSamples: {nsamp}\n      peakSampleIndex: {pk}\n")
    return base


def _bytes(path):
    with open(path, "rb") as fh:
        return fh.read()


def _commit(base, *extra):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        frl.main([base, "1", "--clu", base + ".clu.standard.1.st", "--align-algo", "klusters", "--reextract",
                  "--variants", "standard", "--variant", "standard", "--no-emit-clu", "--min-score", "0.1",
                  *extra])
    return buf.getvalue()


def _tsv(base):
    p = base + ".realign.standard.1.st.tsv"
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return [ln.split("\t") for ln in fh.read().splitlines()[1:]]


def main():
    ok = 0

    def check(name, cond, out=""):
        nonlocal ok
        if not cond and out:
            print(out[-3000:])
        assert cond, name
        ok += 1
        print(f"  ok  {name}")

    with tempfile.TemporaryDirectory() as td:
        base, det = _session(td, doubles=True)
        _, lab0 = nio.read_clu_file(base + ".clu.standard.1.st")
        out = _commit(base, "--park-duplicates")
        res = nio.read_res_file(base + ".res.1")
        same, cross = frl.time_coincidences(res, lab0)
        check("the commit stacks the 40 double detections on one sample each", len(same) == 40, out)
        _, lab = nio.read_clu_file(base + ".clu.standard.1.st")
        moved = np.flatnonzero(lab != lab0)
        dup = frl.duplicates_cluster(lab0)
        check(f"--park-duplicates: exactly the later spike of each pair goes to the duplicates cluster {dup}",
              np.array_equal(np.sort(moved), np.sort(same[:, 1])) and (lab[moved] == dup).all()
              and dup > int(lab0.max()), out)
        check("no row removed", lab.size == det.size and res.size == det.size)
        h = FiberHierarchy.load(base, 1, variant="standard", tag="st")
        new = np.unique(h.child[moved])
        check("the parked spikes are one new child atom under the duplicates cluster",
              new.size == 1 and h.parent.get(int(new[0])) == dup and int(new[0]) not in (2, 3))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rebuilt = h.refiberize(renumber=False)[0]
        check("the hierarchy reproduces its .clu and reloads without orphans",
              np.array_equal(rebuilt, lab) and "no parent" not in buf.getvalue(), buf.getvalue())
        rows = _tsv(base)
        check("the pair list has the 40 same-unit pairs, parked, each under its unit's own id",
              rows is not None and sum(r[0] == "same_unit" and r[6] == "True" and r[4] == r[5] == "2"
                                       for r in rows) == 40)

    with tempfile.TemporaryDirectory() as td:
        base, det = _session(td, doubles=True)
        _, lab0 = nio.read_clu_file(base + ".clu.standard.1.st")
        out = _commit(base)
        _, lab = nio.read_clu_file(base + ".clu.standard.1.st")
        rows = _tsv(base)
        check("without --park-duplicates the labels are untouched", np.array_equal(lab, lab0), out)
        check("... and the 40 pairs are listed, not parked",
              rows is not None and sum(r[0] == "same_unit" and r[6] == "False" for r in rows) == 40
              and "listed, not parked" in out, out)

    with tempfile.TemporaryDirectory() as td:
        base, det = _session(td, doubles=False)
        _, lab0 = nio.read_clu_file(base + ".clu.standard.1.st")
        out = _commit(base, "--park-duplicates")
        _, lab = nio.read_clu_file(base + ".clu.standard.1.st")
        check("NEGATIVE CONTROL: no double detections -> nothing parked, no list",
              np.array_equal(lab, lab0) and _tsv(base) is None, out)

    # a clu without a variant token, with its own .clc/.clp, beside an unrelated sort of the same stage
    with tempfile.TemporaryDirectory() as td:
        base, det = _session(td, doubles=True)
        for k in ("clu", "clc", "clp"):
            shutil.copyfile(f"{base}.{k}.standard.1.st", f"{base}.{k}.1.st")
        nio.write_clu_file(base + ".clu.stderiv.1.st", np.full(det.size, 7))
        sentinel = _bytes(base + ".clu.stderiv.1.st")
        _, lab0 = nio.read_clu_file(base + ".clu.1.st")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            frl.main([base, "1", "--clu", base + ".clu.1.st", "--align-algo", "klusters", "--reextract",
                      "--variants", "standard", "--variant", "standard", "--no-emit-clu", "--min-score", "0.1",
                      "--park-duplicates"])
        _, lab = nio.read_clu_file(base + ".clu.1.st")
        h = FiberHierarchy.load(base, 1, clc=base + ".clc.1.st", clp=base + ".clp.1.st")
        new = np.unique(h.child[lab != lab0])
        check("a variant-less clu is parked with its own .clc/.clp (40 rows, one new atom under the duplicates "
              "cluster)", int((lab != lab0).sum()) == 40 and new.size == 1
              and h.parent.get(int(new[0])) == frl.duplicates_cluster(lab0), buf.getvalue())
        check("... and no other sort is touched or created",
              _bytes(base + ".clu.stderiv.1.st") == sentinel
              and not os.path.exists(base + ".clc.stderiv.1.st"), buf.getvalue())

    # a side-by-side commit
    with tempfile.TemporaryDirectory() as td:
        base, det = _session(td, doubles=True)
        res0 = _bytes(base + ".res.1")
        clu0 = _bytes(base + ".clu.standard.1.st")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            frl.main([base, "1", "--clu", base + ".clu.standard.1.st", "--align-algo", "klusters", "--reextract",
                      "--variants", "standard", "--variant", "standard", "--min-score", "0.1",
                      "--park-duplicates", "--out-tag", "side"])
        out = buf.getvalue()
        check("--out-tag: the canonical .res and the input clu are left as they are",
              _bytes(base + ".res.1") == res0 and _bytes(base + ".clu.standard.1.st") == clu0,
              out)
        _, lab_side = nio.read_clu_file(base + ".clu.standard.1.side")
        p = base + ".realign.standard.1.side.tsv"
        with open(p) as fh:
            rows = [ln.split("\t") for ln in fh.read().splitlines()[1:]]
        check("... the clu emitted under the tag carries the 40 parked spikes, and the pair list is named "
              "after the tag", int((lab_side == frl.duplicates_cluster(nio.read_clu_file(base + ".clu.standard.1.st")[1]))
                                .sum()) == 40 and sum(r[6] == "True" for r in rows) == 40, out)

    # fiber-session --out-variant: the alias is a copy of the sort, written before the commit
    env = {k: v for k, v in os.environ.items() if not k.startswith("FK_")}
    env.update(FK_CONFIG=os.devnull, FK_REALIGN_MIN_SCORE="0.1",
               PYTHONPATH=os.pathsep.join([os.path.join(HERE, "..", "src"), env.get("PYTHONPATH", "")]))
    with tempfile.TemporaryDirectory() as td:
        base = _alias_session(td)
        r = subprocess.run([sys.executable, "-m", "fiber_kit.fiber_session", base, "1", "--method", "stderiv",
                            "--no-link", "--no-fine", "--chunk-min", "0.5", "--overlap-min", "0", "--min-group",
                            "40", "--clu-stage", "fs", "--out-variant", "stderivalias"],
                           capture_output=True, text=True, env=env, timeout=1200)
        out = r.stdout + r.stderr
        same = all(_bytes(f"{base}.{k}.stderiv.1.fs") == _bytes(f"{base}.{k}.stderivalias.1.fs")
                   for k in ("clu", "clc", "clp")) if r.returncode == 0 else False
        check("fiber-session --out-variant: the commit parks duplicates and the alias .clu/.clc/.clp carry "
              "the same parking", r.returncode == 0 and "parked" in out and same, out)

    seen = {}
    orig = frl.main
    frl.main = lambda argv: seen.setdefault("argv", list(argv))
    try:
        class A:                                            # the fields _commit_realign reads
            session, group, realign, realign_algo, realign_fil, method = "s", 1, "shift-spk", "klusters", None, "x"
            base = os.path.join(os.sep, "nonexistent", "s")
            channels = ntotal = nchan = nsamp = sr = None
        with contextlib.redirect_stdout(io.StringIO()):
            fs._commit_realign(A(), "c.clu", "x", None, 21, lambda *_: None, lambda *_: None)
    finally:
        frl.main = orig
    check("fiber-session's commit asks for the parking", "--park-duplicates" in seen.get("argv", []))
    print(f"test_realign_duplicates: {ok} checks passed")


if __name__ == "__main__":
    main()
