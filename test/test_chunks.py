"""Chunk runs (FK_CHUNKS / --chunks) and the reserved cluster ids (fiber_chunks).

  * constants: the 100 reserved ids are the top of Klusters' exact (float32) range, 2**24-100 .. 2**24-1;
    next_free_id skips them and refuses to enter them; parse / fmt / chunk_index (clamped like fiber-session);
  * refiberize keeps reserved ids while compacting the rest (negative control: the old compaction moved them);
  * fiber-session --chunks 1 on a two-chunk synthetic session: chunk 0's spikes all in cluster 16777116,
    chunk 1 partitioned exactly as in the full run, and the .clc/.clp reproduce the .clu;
  * fiber-flag-artifacts --chunks 1: chunk 0 marked reserved (not evaluated), and fiber-session reads such
    a mask without excluding those spikes;
  * fiber-xcorr-merge on the chunk run: the reserved cluster is never merged and keeps its id;
  * fiber-realign: spikes in reserved clusters get offset 0 while the others move;
  * fiber-rdr skips the reserved cluster;
  * fiber-pipeline refuses a stage that changes the spike set while FK_CHUNKS is set, and SHIELDS the
    relabelling stages: backbone-link and anchor-link run on the chunk run, the other chunk's spikes
    come back in its reserved cluster (in the .clc/.clp too), and the input is restored byte for byte.
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from fiber_kit import fiber_chunks as fch          # noqa: E402
from fiber_kit import neuro_io as nio              # noqa: E402
from fiber_kit.fiber_refiberize import FiberHierarchy   # noqa: E402

SR = 20000
NSAMP, NCH, NTOTAL, PEAK = 32, 3, 4, 15
DUR_S = 60
fails = 0


def check(ok, what):
    global fails
    print(("  ok:   " if ok else "  FAIL: ") + what); fails += not ok


def _env(extra=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("FK_")}
    env["FK_CONFIG"] = os.devnull
    env["PYTHONPATH"] = os.pathsep.join([os.path.join(HERE, "..", "src"), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    env.update(extra or {})
    return env


def _build(td, rng):
    """Three units over 60 s (two 30-s chunks), standard + stderiv .spk extracted from a synthetic .fil."""
    from fiber_kit import fiber_stderiv as fst
    from fiber_kit.fiber_realign import reextract_from_fil
    base = os.path.join(td, "mini")
    n_t = DUR_S * SR
    fil = rng.normal(0, 25, (n_t, NTOTAL))
    t = np.arange(-8, 9)
    true = []
    for amp, sh in zip((2600.0, 1800.0, 2200.0), (np.array([1.0, 0.5, 0.2]), np.array([0.3, 1.0, 0.6]),
                                                 np.array([0.2, 0.4, 1.0]))):
        tt = np.sort(rng.choice(np.arange(SR, n_t - SR, 7), 300, replace=False))
        wav = -amp * np.exp(-0.5 * (t / 1.6) ** 2) + 0.35 * amp * np.exp(-0.5 * ((t - 5) / 2.5) ** 2)
        for s in tt:
            fil[s + t, :NCH] += wav[:, None] * sh[None, :]
        true.append(tt)
    fil.round().clip(-32768, 32767).astype(np.int16).tofile(base + ".fil")
    true = np.sort(np.concatenate(true))
    true.astype(np.int64).tofile(base + ".res.1")
    raw = reextract_from_fil(nio.open_signal(base + ".fil", NTOTAL), [0, 1, 2], true, NSAMP + 1, PEAK + 1)
    nio.write_spk(base, 1, raw[:, 1:, :], variant="standard")
    nio.write_spk(base, 1, fst.apply_stderiv_transform(raw), variant="stderiv")
    with open(base + ".yaml", "w") as fh:
        fh.write(f"acquisitionSystem: {{nBits: 16, nChannels: {NTOTAL}, samplingRate: {SR}}}\n"
                 "spikeDetection:\n  channelGroups:\n    - channels: [0, 1, 2]\n"
                 f"      nSamples: {NSAMP}\n      peakSampleIndex: {PEAK}\n")
    return base, true


def run(mod, *args, extra=None):
    r = subprocess.run([sys.executable, "-m", f"fiber_kit.{mod}", *args], capture_output=True, text=True,
                       env=_env(extra), timeout=900)
    return r.returncode, r.stdout + r.stderr


def session(base, stage, *extra):
    return run("fiber_session", base, "1", "--method", "stderiv", "--chunk-min", "0.5", "--overlap-min", "0",
               "--min-group", "40", "--fine-min-group", "40", "--clu-stage", stage, "--realign", "off",
               "--whiten-source", "spk", *extra)


def same_partition(a, b):
    """a and b group the spikes identically (a bijection between their labels)."""
    pairs = set(zip(a.tolist(), b.tolist()))
    return len(pairs) == len(set(a.tolist())) == len(set(b.tolist()))


def main():
    # ── constants and helpers ──
    check(fch.RESERVED_HI == 2 ** 24 - 1 and fch.RESERVED_LO == 2 ** 24 - 100 and
          float(np.float32(fch.RESERVED_HI)) == fch.RESERVED_HI,
          f"reserved block {fch.RESERVED_LO}..{fch.RESERVED_HI}, exact in a float32")
    check(int(fch.reserved_id(3)) == fch.RESERVED_LO + 3 and fch.is_reserved(fch.RESERVED_LO)
          and not fch.is_reserved(fch.RESERVED_LO - 1), "reserved_id / is_reserved")
    check(fch.next_free_id(np.array([0, 2, 7, fch.RESERVED_LO, fch.RESERVED_HI])) == 8,
          "next_free_id skips the reserved block")
    try:
        fch.next_free_id(np.array([fch.RESERVED_LO - 1])); refused = False
    except ValueError:
        refused = True
    check(refused, "next_free_id refuses to enter the block")
    check(fch.parse("") is None and fch.parse("all") is None and fch.parse("3,5-7") == [3, 5, 6, 7]
          and fch.fmt([3, 5, 6, 7]) == "3,5-7", "parse / fmt")
    L = 0.5 * 60 * SR
    res = np.array([100, 100 + L - 1, 100 + L, 100 + 2 * L])     # the last spike sits exactly on the end
    check(fch.chunk_index(res, SR, 0.5).tolist() == [0, 0, 1, 1], "chunk_index clamps the last spike like fiber-session")
    m = fch.mark_inactive(np.array([5, 5, 6, 6]), res, SR, [1], 0.5)
    check(m.tolist() == [fch.RESERVED_LO, fch.RESERVED_LO, 6, 6], "mark_inactive: chunk 0 -> its reserved id")

    # ── refiberize keeps reserved ids ──
    child = np.array([1, 1, 2, 3, 3]); parent = {1: 10, 2: 40, 3: int(fch.RESERVED_LO + 2)}
    clu, _, _ = FiberHierarchy(child, parent).refiberize(renumber=True)
    check(clu.tolist() == [2, 2, 3, fch.RESERVED_LO + 2, fch.RESERVED_LO + 2],
          f"refiberize compacts 10, 40 -> 2, 3 and keeps the reserved id ({clu.tolist()})")
    ids = np.unique([10, 40, fch.RESERVED_LO + 2])                 # negative control: the old compaction
    old = {f: i for i, f in enumerate(ids, start=2)}
    check(old[fch.RESERVED_LO + 2] != fch.RESERVED_LO + 2, "negative control: plain compaction renumbers it")

    with tempfile.TemporaryDirectory() as td:
        rng = np.random.default_rng(3)
        base, true = _build(td, rng)
        # ── fiber-session ──
        rc, out = session(base, "full")
        check(rc == 0, "full fiber-session run" + ("" if rc == 0 else "\n" + out[-3000:]))
        rc, out = session(base, "c1", "--chunks", "1")
        check(rc == 0 and "chunks 1 of 0.5 min" in out, "fiber-session --chunks 1" + ("" if rc == 0 else "\n" + out[-3000:]))
        full = nio.read_clu_file(base + ".clu.stderiv.1.full")[1]
        c1 = nio.read_clu_file(base + ".clu.stderiv.1.c1")[1]
        ci = fch.chunk_index(true, SR, 0.5)
        check(np.all(c1[ci == 0] == fch.RESERVED_LO) and not fch.is_reserved(c1[ci == 1]).any(),
              f"chunk 0 -> cluster {fch.RESERVED_LO}; chunk 1 sorted ({len(np.unique(c1[ci == 1]))} labels)")
        check(same_partition(full[ci == 1], c1[ci == 1]), "chunk 1 is partitioned exactly as in the full run")
        h = FiberHierarchy.load(base, 1, variant="stderiv", tag="c1")
        clu_h, _, _ = h.refiberize(renumber=False)
        check(np.array_equal(clu_h, c1) and len(h.parent) < 10 ** 6,
              f"the .clc/.clp reproduce the .clu ({len(h.parent)} atoms, the reserved one included)")
        # ── fiber-flag-artifacts ──
        rc, out = run("fiber_flag_artifacts", base, "1", "--chunks", "1", "--chunk-length", "0.5")
        mask = nio.read_clu_file(base + ".clu.standard.1.artifact")[1]
        check(rc == 0 and np.all(mask[ci == 0] == fch.RESERVED_LO) and set(np.unique(mask[ci == 1])) <= {0, 1},
              "flag-artifacts --chunks 1: chunk 0 marked not evaluated" + ("" if rc == 0 else "\n" + out[-2000:]))
        rc, out = session(base, "c1m", "--chunks", "1", "--exclude-clu", base + ".clu.standard.1.artifact")
        c1m = nio.read_clu_file(base + ".clu.stderiv.1.c1m")[1]
        check(rc == 0 and np.all(c1m[ci == 0] == fch.RESERVED_LO) and same_partition(c1m[ci == 1], c1[ci == 1]),
              "fiber-session reads a chunk mask: reserved = not excluded" + ("" if rc == 0 else "\n" + out[-2000:]))
        # ── fiber-xcorr-merge ──
        rc, out = run("fiber_xcorr_merge", base, "1", "--clu-method", "stderiv", "--clu-stage", "c1",
                      "--out-stage", "c1x", "--cos-thr", "0.5", "--min-n", "15", "--emit-drift", "0")
        x = nio.read_clu_file(base + ".clu.stderiv.1.c1x")[1]
        check(rc == 0 and np.all(x[ci == 0] == fch.RESERVED_LO) and not fch.is_reserved(x[ci == 1]).any(),
              "xcorr-merge never merges the reserved cluster, even at cos 0.5" + ("" if rc == 0 else "\n" + out[-2000:]))
        rc, out = run("fiber_xcorr_merge", base, "1", "--clu-method", "stderiv", "--clu-stage", "full",
                      "--out-stage", "fx", "--cos-thr", "0.99", "--min-n", "15", "--emit-drift", "0",
                      "--chunks", "1", "--chunk-length", "0.5")
        fx = nio.read_clu_file(base + ".clu.stderiv.1.fx")[1]
        check(rc == 0 and np.all(fx[ci == 0] == fch.RESERVED_LO), "xcorr-merge --chunks 1 on a full sort: chunk 0 -> reserved")
        # ── fiber-realign: reserved spikes are frozen ──
        from fiber_kit import fiber_realign as frl
        spk = np.asarray(nio.open_spk_file(base + ".spk.standard.1", NSAMP, NCH)).copy()
        jit = rng.integers(-2, 3, spk.shape[0])
        nio.write_spk_file(base + ".spk.standard.1", np.stack([np.roll(w, j, axis=0) for w, j in zip(spk, jit)]))
        nio.write_clu_file(base + ".clu.standard.1.c1", c1)
        _, _, ioff, _, _, _, _ = frl.realign(base, 1, NSAMP, NCH, base + ".clu.standard.1.c1", method="klusters",
                                             peak=PEAK, min_score=0.1, variant="standard", verbose=False)
        check(np.all(ioff[ci == 0] == 0) and np.count_nonzero(ioff[ci == 1]) > 0.3 * np.sum(ci == 1),
              f"realign: reserved spikes keep offset 0, {np.count_nonzero(ioff[ci == 1])} others move")
        # ── fiber-rdr ──
        rc, out = run("fiber_rdr", base, "1", "--clu-method", "stderiv", "--clu-stage", "c1", "--min-n", "15")
        tab = ""
        if rc == 0:
            with open(base + ".rdr.stderiv.1.c1.tsv") as fh:
                tab = fh.read()
        check(rc == 0 and str(fch.RESERVED_LO) not in tab, "fiber-rdr does not score the reserved cluster")
        # ── the runner guard ──
        cfgp = os.path.join(td, "fk.yaml")
        with open(cfgp, "w") as fh:
            fh.write('FK_CHUNKS: "1"\n')
        r = subprocess.run(["bash", os.path.join(HERE, "..", "scripts", "fiber-pipeline"), "1", "fiber-refine", "--dry-run"],
                           capture_output=True, text=True, cwd=td,
                           env=_env({"FK_CONFIG": cfgp, "FK_DIR": td, "FK_SESS": "mini"}), timeout=300)
        check(r.returncode != 0 and "does not support chunk runs" in (r.stdout + r.stderr),
              "fiber-pipeline refuses fiber-refine (it changes the spike set) while FK_CHUNKS is set")
        # ── shielded stages: backbone-link and anchor-link on the chunk run, through the runner ──
        plan = os.path.join(td, "plan.yaml")
        with open(plan, "w") as fh:
            fh.write("pipeline:\n"
                     "  - {stage: fiber-backbone-link, in: c1, out: c1bb}\n"
                     "  - {stage: fiber-anchor-link,   in: c1, out: c1al}\n")
        before = {k: open(f"{base}.{k}.stderiv.1.c1", "rb").read() for k in ("clu", "clc", "clp")}
        penv = _env({"FK_CONFIG": cfgp, "FK_DIR": td, "FK_SESS": "mini", "FK_SESSION_CHUNK_MIN": "0.5"})
        r = subprocess.run(["bash", os.path.join(HERE, "..", "scripts", "fiber-pipeline"), "1", "--plan", plan,
                            "--dry-run"], capture_output=True, text=True, cwd=td, env=penv, timeout=300)
        check(r.returncode == 0 and (r.stdout + r.stderr).count("chunk shield") == 2,
              "dry run: both stages are shielded" + ("" if r.returncode == 0 else "\n" + (r.stdout + r.stderr)[-2000:]))
        r = subprocess.run(["bash", os.path.join(HERE, "..", "scripts", "fiber-pipeline"), "1", "--plan", plan],
                           capture_output=True, text=True, cwd=td, env=penv, timeout=1800)
        pout = r.stdout + r.stderr
        check(r.returncode == 0, "the shielded plan runs" + ("" if r.returncode == 0 else "\n" + pout[-4000:]))
        if r.returncode == 0:
            for tag in ("c1bb", "c1al"):
                o = nio.read_clu_file(f"{base}.clu.stderiv.1.{tag}")[1]
                check(np.all(o[ci == 0] == fch.RESERVED_LO) and not fch.is_reserved(o[ci == 1]).any()
                      and (o[ci == 1] > 1).mean() > 0.9,
                      f".{tag}: chunk 0 back in cluster {fch.RESERVED_LO}, chunk 1 in real clusters")
                if os.path.exists(f"{base}.clc.stderiv.1.{tag}"):
                    hh = FiberHierarchy.load(base, 1, variant="stderiv", tag=tag)
                    check(np.array_equal(hh.refiberize(renumber=False)[0], o),
                          f".{tag}: its .clc/.clp reproduce the .clu, reserved atom included")
            after = {k: open(f"{base}.{k}.stderiv.1.c1", "rb").read() for k in ("clu", "clc", "clp")}
            check(after == before, "the input .c1 triple is restored byte for byte")
            left = [f for f in os.listdir(td) if "chunkshield" in f]
            check(not left, f"no shield state or backup left ({left})")
            bad = os.path.join(td, "bad.yaml")                 # a shielded stage that fails
            with open(bad, "w") as fh:
                fh.write("pipeline:\n  - {stage: fiber-backbone-link, in: c1, out: c1x, params: {no-such-flag: 1}}\n")
            r = subprocess.run(["bash", os.path.join(HERE, "..", "scripts", "fiber-pipeline"), "1", "--plan", bad],
                               capture_output=True, text=True, cwd=td, env=penv, timeout=600)
            after = {k: open(f"{base}.{k}.stderiv.1.c1", "rb").read() for k in ("clu", "clc", "clp")}
            check(r.returncode != 0 and after == before and not [f for f in os.listdir(td) if "chunkshield" in f],
                  "a shielded stage that fails: the step fails, the input is restored, nothing left over")
            st = os.path.join(td, "mini.chunkshield.1.json")      # a run killed between shield and unshield
            fch.shield(os.path.join(td, "mini"), 1, "stderiv", "c1", "c1y", st, [1], 0.5, log=lambda m: None)
            masked = open(f"{base}.clu.stderiv.1.c1", "rb").read()
            fch.shield(os.path.join(td, "mini"), 1, "stderiv", "c1", "c1y", st, [1], 0.5, log=lambda m: None)
            fch.unshield(st, log=lambda m: None)
            after = {k: open(f"{base}.{k}.stderiv.1.c1", "rb").read() for k in ("clu", "clc", "clp")}
            check(masked != before["clu"] and after == before,
                  "a shield left by a killed run is undone by the next one (the input is not shielded twice)")
        r = subprocess.run(["bash", os.path.join(HERE, "..", "scripts", "fiber-pipeline"), "1", "fiber-rdr", "--dry-run"],
                           capture_output=True, text=True, cwd=td,
                           env=_env({"FK_CONFIG": cfgp, "FK_DIR": td, "FK_SESS": "mini"}), timeout=300)
        check("does not support chunk runs" not in (r.stdout + r.stderr), "... and lets fiber-rdr through")
        shutil.rmtree(td, ignore_errors=True)
    print(f"\ntest_chunks: {'FAIL' if fails else 'PASS'} ({fails} failed)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
