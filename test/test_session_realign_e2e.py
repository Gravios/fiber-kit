"""fiber-session --realign END TO END: the shifted .res, every variant's .spk and .fet.

A synthetic one-group session built from a .fil: three units whose spikes are detected with
+-2-sample jitter, a .spk.standard extracted from the .fil at the jittered times, a .spk.stderiv
(the allpairs transform of the same windows), an opaque .spk.standard_ADLP that no .fil
transform derives, and a .pca for standard and stderiv.

  * --realign off leaves .res/.spk/.fet exactly as extracted (the negative control);
  * the default (auto, .fil present) commits -- after locating the windows on disk in the .fil (the
    spike time at peakSampleIndex - 1) -- the .res moves, and the result is BYTE-IDENTICAL
    to `--realign off` followed by the plan step `fiber-realign --reextract --refeaturize
    --variants all --park-duplicates` (the shipped profiles set FK_REALIGN_PARK_DUPS) -- the two
    paths cannot drift apart;
  * every variant on disk is rebuilt: standard and stderiv re-extracted from the .fil (standard
    equals a fresh .fil read at the committed times), the opaque one rolled by the offsets rather
    than skipped, and a .fet per variant whose time column is the committed .res;
  * the jitter is what was removed: the committed times sit closer to the true times;
  * without a .fil, auto falls back to shift-spk and commits the same .res, and the rolled windows equal
    the re-extracted ones away from the wrap (both keep neurosuite-3's window: the timestamp at
    peakSampleIndex - 1, where the klusters algorithm aligns each template's peak);
  * --emit-fet writes its .fet from the committed waveforms, last column = committed .res.
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from fiber_kit import fiber_pca as fpca       # noqa: E402
from fiber_kit import fiber_stderiv as fst    # noqa: E402
from fiber_kit import neuro_io as nio         # noqa: E402

SR = 20000
NSAMP, NCH, NTOTAL, PEAK = 32, 3, 4, 15
DUR_S = 60


def _build(td, rng):
    base = os.path.join(td, "mini")
    n_t = DUR_S * SR
    fil = rng.normal(0, 25, (n_t, NTOTAL))
    t = np.arange(-8, 9)
    shapes = [np.array([1.0, 0.5, 0.2]), np.array([0.3, 1.0, 0.6]), np.array([0.2, 0.4, 1.0])]
    true = []
    for amp, sh in zip((2600.0, 1800.0, 2200.0), shapes):
        tt = np.sort(rng.choice(np.arange(SR, n_t - SR, 7), 260, replace=False))
        wav = -amp * np.exp(-0.5 * (t / 1.6) ** 2) + 0.35 * amp * np.exp(-0.5 * ((t - 5) / 2.5) ** 2)
        for s in tt:
            fil[s + t, :NCH] += wav[:, None] * sh[None, :]
        true.append(tt)
    fil.round().clip(-32768, 32767).astype(np.int16).tofile(base + ".fil")
    true = np.sort(np.concatenate(true))
    det = (true + rng.integers(-2, 3, true.size)).astype(np.int64)        # detection jitter
    det.tofile(base + ".res.1")
    filmm = nio.open_signal(base + ".fil", NTOTAL)
    from fiber_kit.fiber_realign import reextract_from_fil, timestamp_index
    # what neurosuite-3's extractor writes: the timestamp at peakSampleIndex - 1 (+1 leading sample)
    raw = reextract_from_fil(filmm, [0, 1, 2], det, NSAMP + 1, timestamp_index(PEAK) + 1)
    nio.write_spk(base, 1, raw[:, 1:, :], variant="standard")
    nio.write_spk(base, 1, fst.apply_stderiv_transform(raw), variant="stderiv")
    shutil.copy(base + ".spk.standard.1", base + ".spk.standard_ADLP.1")   # opaque: no .fil transform
    for v in ("standard", "stderiv"):
        x = np.fromfile(f"{base}.spk.{v}.1", np.int16).reshape(-1, NSAMP, NCH).astype(float)
        means, evec = fpca.fit_basis(x[:, 8:24, :], 3, centered=False)
        fpca.write_pcad(f"{base}.pca.{v}.1", means, evec, 8, centered=0)
    with open(base + ".yaml", "w") as fh:
        fh.write(f"acquisitionSystem: {{nBits: 16, nChannels: {NTOTAL}, samplingRate: {SR}}}\n"
                 "spikeDetection:\n  channelGroups:\n    - channels: [0, 1, 2]\n"
                 f"      nSamples: {NSAMP}\n      peakSampleIndex: {PEAK}\n")
    return base, true


def _env():
    env = {k: v for k, v in os.environ.items() if not k.startswith("FK_")}
    env["FK_CONFIG"] = os.devnull
    env["PYTHONPATH"] = os.pathsep.join([os.path.join(HERE, "..", "src"),
                                         env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    return env


def session(base, *extra):
    r = subprocess.run([sys.executable, "-m", "fiber_kit.fiber_session", base, "1",
                        "--method", "stderiv", "--no-link", "--no-fine", "--chunk-min", "0.5",
                        "--overlap-min", "0", "--min-group", "40", "--clu-stage", "fs"] + list(extra),
                       capture_output=True, text=True, env=_env(), timeout=900)
    return r.returncode, r.stdout + r.stderr


def realign_step(base):
    r = subprocess.run([sys.executable, "-m", "fiber_kit.fiber_realign", base, "1",
                        "--clu", f"{base}.clu.stderiv.1.fs", "--align-algo", "klusters",
                        "--reextract", "--refeaturize", "--no-emit-clu", "--variants", "all",
                        "--variant", "stderiv", "--park-duplicates"],
                       capture_output=True, text=True, env=_env(), timeout=900)
    return r.returncode, r.stdout + r.stderr


SET = ["res.1", "spk.standard.1", "spk.stderiv.1", "spk.standard_ADLP.1", "fet.standard.1",
       "fet.stderiv.1", "clu.stderiv.1.fs", "clc.stderiv.1.fs", "clp.stderiv.1.fs", "offsets.1.npy",
       "realign.stderiv.1.fs.tsv"]


def blob(base, ext):
    p = f"{base}.{ext}"
    return open(p, "rb").read() if os.path.exists(p) else None


def main():
    ok = 0

    def check(name, cond, out=""):
        nonlocal ok
        if not cond and out:
            print(out[-4000:])
        assert cond, name
        ok += 1
        print(f"  ok  {name}")

    rng = np.random.default_rng(3)
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "src"); os.mkdir(src)
        base0, true = _build(src, rng)
        orig = {e: blob(base0, e) for e in SET}

        def fresh(name):
            d = os.path.join(td, name); shutil.copytree(src, d)
            return os.path.join(d, "mini")

        # negative control: --realign off touches nothing it did not write
        bo = fresh("off")
        rc, out = session(bo, "--realign", "off")
        check("--realign off exits 0", rc == 0, out)
        check("--realign off: .res/.spk unchanged",
              all(blob(bo, e) == orig[e] for e in ("res.1", "spk.standard.1", "spk.stderiv.1",
                                                    "spk.standard_ADLP.1")), out)
        check("--realign off: no offsets, no .fet", blob(bo, "offsets.1.npy") is None
              and blob(bo, "fet.stderiv.1") is None)
        rc, out = realign_step(bo)
        check("then the plan's fiber-realign step exits 0", rc == 0, out)

        # the default: auto -> reextract (the .fil is there)
        ba = fresh("auto")
        rc, out = session(ba)
        check("default run exits 0", rc == 0, out)
        check("default with a .fil commits by reextract", "realign: reextract" in out, out)
        check("the commit locates the stderiv windows on disk at the group's index (re-cut from the .fil)",
              f"the spike time at index {PEAK - 1}, the group's geometry" in out, out)
        res_a = nio.read_res_file(ba + ".res.1")
        check("the .res moved", blob(ba, "res.1") != orig["res.1"])
        diff = [e for e in SET if blob(ba, e) != blob(bo, e)]
        check("byte-identical to --realign off + fiber-realign --reextract --refeaturize --variants all "
              "--park-duplicates",
              not diff, f"differs: {diff}\n{out}")

        filmm = nio.open_signal(ba + ".fil", NTOTAL)
        from fiber_kit.fiber_realign import reextract_from_fil, roll_spikes, timestamp_index
        raw = reextract_from_fil(filmm, [0, 1, 2], res_a, NSAMP + 1, timestamp_index(PEAK) + 1)
        spk_std = np.fromfile(ba + ".spk.standard.1", np.int16).reshape(-1, NSAMP, NCH)
        check(".spk.standard = a fresh .fil read at the committed times", np.array_equal(spk_std, raw[:, 1:, :]))
        spk_sd = np.fromfile(ba + ".spk.stderiv.1", np.int16).reshape(-1, NSAMP, NCH)
        check(".spk.stderiv = the stderiv transform of it", np.array_equal(spk_sd, fst.apply_stderiv_transform(raw)))
        ioff = np.rint(np.load(ba + ".offsets.1.npy")).astype(int)
        adlp0 = np.frombuffer(orig["spk.standard_ADLP.1"], np.int16).reshape(-1, NSAMP, NCH)
        adlp = np.fromfile(ba + ".spk.standard_ADLP.1", np.int16).reshape(-1, NSAMP, NCH)
        check("the opaque variant is rolled by the offsets, not left stale",
              np.array_equal(adlp, roll_spikes(adlp0, ioff)) and np.count_nonzero(ioff) > 0, out)
        for v in ("standard", "stderiv"):
            fet = nio.read_fet_file(f"{ba}.fet.{v}.1").values
            check(f".fet.{v}: one row per spike, time column = committed .res",
                  fet.shape[0] == res_a.size and np.array_equal(fet[:, -1], res_a))
        res0 = np.frombuffer(orig["res.1"], "<i8")
        check(f"jitter removed: spread of t - true "
              f"{np.std(res_a - true):.2f} (from {np.std(res0 - true):.2f})",
              np.std(res_a - true) < 0.75 * np.std(res0 - true), out)

        # no .fil: auto falls back to shift-spk, same offsets / .res
        bn = fresh("nofil")
        os.remove(bn + ".fil")
        rc, out = session(bn, "--whiten-source", "spk")
        check("no .fil: auto commits by shift-spk", rc == 0 and "realign: shift-spk" in out, out)
        rc2, out2 = session(fresh("nofil_ref") , "--whiten-source", "spk", "--realign", "reextract")
        check("no .fil: same .res as the reextract commit of the same sort",
              np.array_equal(nio.read_res_file(bn + ".res.1"),
                             nio.read_res_file(os.path.join(td, "nofil_ref", "mini.res.1"))), out + out2)
        # The roll keeps the extractor's window and the re-extraction must cut the same one: away from
        # the wrap the two commits hold identical samples (a re-cut at peakSampleIndex, one sample late,
        # fails here).
        bref = os.path.join(td, "nofil_ref", "mini")
        ioff_n = np.rint(np.load(bn + ".offsets.1.npy")).astype(int)
        m = int(np.abs(ioff_n).max()) + 1
        rolled = np.fromfile(bn + ".spk.standard.1", np.int16).reshape(-1, NSAMP, NCH)
        recut = np.fromfile(bref + ".spk.standard.1", np.int16).reshape(-1, NSAMP, NCH)
        check(f"shift-spk and reextract write the same windows away from the wrap (+-{m} samples)",
              np.count_nonzero(ioff_n) > 0 and np.array_equal(rolled[:, m:NSAMP - m], recut[:, m:NSAMP - m]),
              out + out2)
        rc, out = session(fresh("nofil2"), "--realign", "reextract", "--realign-fil", "/nonexistent.fil",
                          "--whiten-source", "spk")
        check("--realign reextract without the .fil stops with a message",
              rc != 0 and "no filtered signal" in out and "Traceback" not in out, out)

        # --emit-fet: written from the committed set
        be = fresh("emit")
        rc, out = session(be, "--emit-fet", "--emit-pca", "--out-variant", "stderiv_D13")
        check("--emit-fet run exits 0", rc == 0, out)
        fe = nio.read_fet_file(be + ".fet.stderiv_D13.1").values
        check("--emit-fet .fet: last column = committed .res",
              np.array_equal(fe[:, -1], nio.read_res_file(be + ".res.1")), out)

    print(f"test_session_realign_e2e: {ok} checks passed")


if __name__ == "__main__":
    main()
