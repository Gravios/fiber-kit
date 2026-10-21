"""fiber-session's end-of-run commit cuts at the group's peak -- the YAML's, or a measured one.

Sessions whose YAML has no peakSampleIndex, extracted with the spike time at sample 10 of a 42-sample
window (far from build_masks' 15/32 guess of 20):

  * with the .fil: the commit locates the windows on disk in it (index 10) and behaves as with
    peakSampleIndex 11 in the YAML: every committed unit template peaks on 10 in the aligned (stderiv)
    space, and the committed windows keep the extractor's geometry (a .fil cut at 10 at the committed
    times).  Negative control: the template-peak measure in the stderiv space (where the derivative
    puts the peak a few samples later) re-cuts the windows at another index;
  * without a .fil (a --shift-spk commit, which keeps the windows' geometry): the median template peak
    of the largest units, so the units' committed times stay on the true times.  Negative control:
    the same commit at the guess (what fiber-session passed until 0.51) moves them by several samples.

With a peakSampleIndex in the YAML the commit uses it -- Klusters' geometry: each template's peak goes
to peakSampleIndex - 1 in the aligned space, wherever the data had it -- and measures nothing.  The
commits run at the curator's threshold (FK_REALIGN_MIN_SCORE 0.10), so every unit spike may move.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from fiber_kit import fiber_session as fs        # noqa: E402
from fiber_kit import fiber_stderiv as fst       # noqa: E402
from fiber_kit import neuro_io as nio            # noqa: E402

SR = 20000
NSAMP, NCH, NTOTAL, AT = 42, 3, 4, 10              # spikes peak at sample AT of the window
DUR_S = 60


def _build(td, rng, yaml_peak=None):
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
    fil = fil.round().clip(-32768, 32767).astype(np.int16)
    fil.tofile(base + ".fil")
    true = np.sort(np.concatenate(true))
    det = (true + rng.integers(-2, 3, true.size)).astype(np.int64)
    det.tofile(base + ".res.1")
    raw = np.stack([fil[d - AT - 1:d - AT + NSAMP, :NCH] for d in det])     # + 1 leading sample
    nio.write_spk(base, 1, raw[:, 1:, :], variant="standard")
    nio.write_spk(base, 1, fst.apply_stderiv_transform(raw), variant="stderiv")
    pk = f"      peakSampleIndex: {yaml_peak}\n" if yaml_peak is not None else ""
    with open(base + ".yaml", "w") as fh:
        fh.write(f"acquisitionSystem: {{nBits: 16, nChannels: {NTOTAL}, samplingRate: {SR}}}\n"
                 "spikeDetection:\n  channelGroups:\n    - channels: [0, 1, 2]\n"
                 f"      nSamples: {NSAMP}\n" + pk)
    return base, true


def _env():
    env = {k: v for k, v in os.environ.items() if not k.startswith("FK_")}
    env["FK_CONFIG"] = os.devnull
    env["FK_REALIGN_MIN_SCORE"] = "0.1"
    env["PYTHONPATH"] = os.pathsep.join([os.path.join(HERE, "..", "src"),
                                         env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    return env


def _run(mod, base, *args):
    r = subprocess.run([sys.executable, "-m", mod, base, "1"] + list(args),
                       capture_output=True, text=True, env=_env(), timeout=900)
    return r.returncode, r.stdout + r.stderr


def _session(base, *extra):
    return _run("fiber_kit.fiber_session", base, "--method", "stderiv", "--no-link", "--no-fine",
                "--chunk-min", "0.5", "--overlap-min", "0", "--min-group", "40", "--clu-stage", "fs",
                "--whiten-source", "fil", *extra)


def _templates(base, lab, variant="stderiv"):
    sd = np.asarray(nio.open_spk_file(f"{base}.spk.{variant}.1", NSAMP, NCH), float)
    return [int(np.argmax(np.abs(sd[lab == u].mean(0)).sum(1))) for u in np.unique(lab[lab > 0])
            if (lab == u).sum() >= 20]


def _cut_at(base, k):
    """Do the committed .spk.standard windows equal a .fil cut with the spike time at index k?"""
    fil = nio.open_signal(base + ".fil", NTOTAL)
    res = nio.read_res_file(base + ".res.1")
    spk = np.asarray(nio.open_spk_file(base + ".spk.standard.1", NSAMP, NCH))
    ok = (res >= NSAMP) & (res < fil.shape[0] - NSAMP)
    cut = np.stack([fil[r - k:r - k + NSAMP, :NCH] for r in res[ok]])
    return bool(np.array_equal(spk[ok], cut))


def main():
    ok = 0

    def check(name, cond, out=""):
        nonlocal ok
        if not cond and out:
            print(out[-4000:])
        assert cond, name
        ok += 1
        print(f"  ok  {name}")

    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "src"); os.mkdir(src)
        _, true = _build(src, np.random.default_rng(3))

        def fresh(name, fil=True):
            d = os.path.join(td, name); shutil.copytree(src, d)
            b = os.path.join(d, "mini")
            if not fil:
                os.remove(b + ".fil")
            return b

        # ── with the .fil: the windows on disk are located in it ──
        b = fresh("located")
        rc, out = _session(b)
        check("no peakSampleIndex, a .fil: fiber-session exits 0", rc == 0, out)
        check(f"the commit locates the windows on disk at index {AT}",
              f"windows hold the spike time at index {AT}" in out, out)
        _, lab = nio.read_clu_file(b + ".clu.stderiv.1.fs")
        at = _templates(b, lab)
        check(f"... every committed unit template peaks on index {AT}, as with peakSampleIndex {AT + 1} ({at})",
              len(at) >= 1 and all(x == AT for x in at), out)
        check(f"... and the committed windows keep the extractor's geometry (a .fil cut at index {AT})",
              _cut_at(b, AT), out)
        # the measurement is the aligned variant's own: one the .fil cannot derive is not located, even
        # with a .spk.standard beside it that is
        shutil.copyfile(b + ".spk.stderiv.1", b + ".spk.stderiv_C5.1")
        ns = types.SimpleNamespace(realign_fil=None, base=b, ntotal=NTOTAL, elec=1, nsamp=NSAMP, nchan=NCH,
                                   method="stderiv")
        rb = nio.read_res_file(b + ".res.1")
        own = fs._locate_windows(ns, [0, 1, 2], None, rb, "stderiv")
        c5 = fs._locate_windows(ns, [0, 1, 2], None, rb, "stderiv_C5")
        check(f"_locate_windows measures the aligned variant only (stderiv {own.index if own else None}; an "
              f"underivable stderiv_C5 -> {c5})", own is not None and own.index == AT and c5 is None)

        # negative control: the template-peak measure, in the stderiv space, cuts elsewhere
        bt = fresh("template_peak")
        rc, out = _session(bt, "--realign", "off")
        check("--realign off exits 0", rc == 0, out)
        _, lab_t = nio.read_clu_file(f"{bt}.clu.stderiv.1.fs")
        spk_t = nio.open_spk_file(bt + ".spk.stderiv.1", NSAMP, NCH)
        pk_t, how = fs._commit_peak(None, spk_t, np.asarray(lab_t))
        rc, out = _run("fiber_kit.fiber_realign", bt, "--clu", f"{bt}.clu.stderiv.1.fs", "--align-algo",
                       "klusters", "--reextract", "--refeaturize", "--no-emit-clu", "--variants", "all",
                       "--variant", "stderiv", "--peak", str(pk_t))
        check(f"NEGATIVE CONTROL: the stderiv template peaks measure {pk_t} ({how}) and that commit re-cuts the "
              f"windows at index {pk_t - 1}, not the extractor's {AT}",
              rc == 0 and pk_t - 1 != AT and _cut_at(bt, pk_t - 1) and not _cut_at(bt, AT), out)

        # ── without a .fil: a --shift-spk commit at the median template peak ──
        bn = fresh("nofil", fil=False)
        rc, out = _session(bn, "--whiten-source", "spk")
        check("no peakSampleIndex, no .fil: fiber-session exits 0 and commits by shift-spk",
              rc == 0 and "realign: shift-spk" in out, out)
        check("the commit measures the peak on the units' templates", "measured on the template peaks" in out, out)
        res = nio.read_res_file(bn + ".res.1")
        _, lab = nio.read_clu_file(bn + ".clu.stderiv.1.fs")
        unit = lab > 0
        err = np.median(np.abs(res - true)[unit])
        check(f"the units' committed times stay on the true times (median |error| {err:.1f} samples)",
              unit.sum() >= 100 and err <= 1.0, out)

        # negative control: the commit at the 15/32 guess (fiber-session's argument until 0.51)
        bg = fresh("guess", fil=False)
        rc, out = _session(bg, "--whiten-source", "spk", "--realign", "off")
        check("--realign off exits 0", rc == 0, out)
        guess = int(round(NSAMP * 15 / 32))
        rc, out = _run("fiber_kit.fiber_realign", bg, "--clu", f"{bg}.clu.stderiv.1.fs", "--align-algo",
                       "klusters", "--shift-spk", "--refeaturize", "--no-emit-clu", "--variants", "all",
                       "--variant", "stderiv", "--peak", str(guess))
        check("the commit at the guessed peak exits 0", rc == 0, out)
        _, lab_g = nio.read_clu_file(f"{bg}.clu.stderiv.1.fs")
        err_g = np.median(np.abs(nio.read_res_file(bg + ".res.1") - true)[lab_g > 0])
        check(f"NEGATIVE CONTROL: at the guess ({guess}) the units' times move by {err_g:.0f} samples",
              err_g >= 4, out)

    with tempfile.TemporaryDirectory() as td:
        base, true = _build(td, np.random.default_rng(3), yaml_peak=AT + 1)
        rc, out = _session(base)
        check("with peakSampleIndex in the YAML: exits 0 and does not measure",
              rc == 0 and "measured" not in out, out)
        _, lab_y = nio.read_clu_file(base + ".clu.stderiv.1.fs")
        at = _templates(base, lab_y)
        check(f"... and every committed unit template peaks on the YAML's target, index {AT} ({at})",
              len(at) >= 1 and all(x == AT for x in at), out)
    print(f"test_session_commit_peak: {ok} checks passed")


if __name__ == "__main__":
    main()
