#!/usr/bin/env python3
"""The Klusters <-> fiber-kit recluster contract: a sub-session of chosen clusters, a plan run on it,
the final .clu read back by position.

On a synthetic three-unit session (as test_session_ids):
  * `fk subsession export` writes a row subset of every .res/.spk/.fet variant (same order, time-sorted),
    the seed .clu with the selected ids, the yaml copy, and symlinks for .fil/.nst; the rows are exactly the
    selected clusters' spikes and the subset's waveforms equal the originals' rows;
  * `fiber-pipeline <g> --session <sub> --final-clu <path> fiber-session` runs on it and delivers a final .clu
    with one id per selected spike (nothing dropped); the real .fil is reached through the link
    (fiber-session whitens from it); a plan with an `out:` step and the dry run deliver too;
  * `fk run --session --final-clu` forwards both; `fk subsession clean` removes everything;
  * refusals: a row beyond the .res, a seed of the wrong length, an existing sub-session without --force.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from fiber_kit import fiber_stderiv as fst, fiber_subsession as fss, neuro_io as nio   # noqa: E402

SR = 20000
NSAMP, NCH, NTOTAL, PK = 32, 3, 4, 16
DUR_S = 60
fails = 0


def check(ok, what, out=""):
    global fails
    print(("  ok:   " if ok else "  FAIL: ") + what)
    if not ok and out:
        print(out[-3000:])
    fails += not ok


def session(td):
    rng = np.random.default_rng(23)
    base = os.path.join(td, "sub")
    n_t = DUR_S * SR
    fil = rng.normal(0, 25, (n_t, NTOTAL))
    t = np.arange(-8, 9)
    shapes = [np.array([1.0, 0.5, 0.2]), np.array([0.3, 1.0, 0.6]), np.array([0.2, 0.4, 1.0])]
    det, lab = [], []
    for u, (amp, sh) in enumerate(zip((2600.0, 1800.0, 2200.0), shapes), 2):
        tt = np.sort(rng.choice(np.arange(SR, n_t - SR, 7), 300, replace=False))
        wav = -amp * np.exp(-0.5 * (t / 1.6) ** 2) + 0.35 * amp * np.exp(-0.5 * ((t - 5) / 2.5) ** 2)
        for s in tt:
            fil[s + t, :NCH] += wav[:, None] * sh[None, :]
        det.append(tt + rng.integers(-1, 2, tt.size)); lab.append(np.full(tt.size, u))
    fil = fil.round().clip(-32768, 32767).astype(np.int16)
    fil.tofile(base + ".fil")
    det = np.concatenate(det); lab = np.concatenate(lab)
    o = np.argsort(det, kind="stable"); det, lab = det[o].astype(np.int64), lab[o]
    det.tofile(base + ".res.1")
    c0 = PK - 1
    raw = np.stack([fil[d - c0 - 1:d - c0 + NSAMP, :NCH] for d in det])
    nio.write_spk(base, 1, raw[:, 1:, :], variant="standard")
    nio.write_spk(base, 1, fst.apply_stderiv_transform(raw), variant="stderiv")
    fet = np.column_stack([rng.integers(-1000, 1000, (det.size, 6)), det]).astype(np.int64)
    nio.write_fet(base, 1, fet, variant="stderiv")
    nio.write_clu(base, 1, lab, variant="stderiv", tag="cur")
    with open(base + ".yaml", "w") as fh:
        fh.write(f"acquisitionSystem: {{nBits: 16, nChannels: {NTOTAL}, samplingRate: {SR}}}\n"
                 "spikeDetection:\n  channelGroups:\n    - channels: [0, 1, 2]\n"
                 f"      nSamples: {NSAMP}\n      peakSampleIndex: {PK}\n")
    D = NSAMP * NCH                                            # a valid noise-statistics file (white, 25 LSB)
    nio.write_nst(base + ".nst.stderiv.1", dict(nsamp=NSAMP, nchan=NCH, peak=PK, sigma=np.full(NCH, 25.0),
                                                mean=np.zeros((NSAMP, NCH)), cov=np.eye(D) * 625.0))
    return base, det, lab


def fk(*args, cwd, env):
    return subprocess.run([sys.executable, "-c", "from fiber_kit.plan_steps import main; import sys; "
                           "sys.exit(main(sys.argv[1:]))", *args], cwd=cwd, env=env, capture_output=True, text=True)


def main():
    env = {k: v for k, v in os.environ.items() if not k.startswith("FK_")}
    env["FK_CONFIG"] = os.devnull
    env["PYTHONPATH"] = os.pathsep.join([os.path.join(ROOT, "src"), env.get("PYTHONPATH", "")])
    env["PATH"] = os.path.join(ROOT, "scripts") + os.pathsep + env.get("PATH", "")
    pipe = os.path.join(ROOT, "scripts", "fiber-pipeline")
    with tempfile.TemporaryDirectory() as td:
        base, res, lab = session(td)
        clu = base + ".clu.stderiv.1.cur"
        # ── export ──
        r = fk("subsession", "export", base, "1", "--clu", clu, "--clusters", "2", "4", "--out", base + "--2-4--t",
               cwd=td, env=env)
        out = base + "--2-4--t"
        check(r.returncode == 0, "fk subsession export exits 0", r.stdout + r.stderr)
        want = np.flatnonzero(np.isin(lab, [2, 4]))
        sres = np.fromfile(out + ".res.1", dtype=nio.RES_DTYPE)
        check(sres.size == want.size and np.array_equal(sres, res[want]), f"the sub .res is the selected spikes' times ({sres.size})")
        check(np.all(np.diff(sres) >= 0), "the sub .res is time-sorted")
        for v in ("standard", "stderiv"):
            full = nio.open_spk_file(f"{base}.spk.{v}.1", NSAMP, NCH)
            sub_ = nio.open_spk_file(f"{out}.spk.{v}.1", NSAMP, NCH)
            check(sub_.shape[0] == want.size and np.array_equal(np.asarray(sub_), np.asarray(full)[want]),
                  f"the sub .spk.{v} rows equal the originals' selected rows")
        f_full = nio.read_fet_file(f"{base}.fet.stderiv.1"); f_sub = nio.read_fet_file(f"{out}.fet.stderiv.1")
        check(f_sub.ok and np.array_equal(f_sub.values, f_full.values[want]), "the sub .fet rows equal the originals'")
        _, seed = nio.read_clu_file(f"{out}.clu.stderiv.1")
        check(np.array_equal(np.asarray(seed), lab[want]), "the seed .clu carries the selected ids (variant read off the clu name)")
        check(os.path.islink(out + ".fil") and os.path.realpath(out + ".fil") == os.path.realpath(base + ".fil"), ".fil is a symlink to the session's")
        check(os.path.islink(out + ".nst.stderiv.1"), ".nst.<v>.<g> is linked")
        check(os.path.isfile(out + ".yaml"), "the yaml is copied")
        r = fk("subsession", "export", base, "1", "--clu", clu, "--clusters", "2", "--out", out, cwd=td, env=env)
        check(r.returncode == 1 and "exists" in r.stderr, "an existing sub-session is not overwritten without --force")
        # refusals of the library
        try:
            fss.export(base, 1, [res.size + 5], out + "x"); check(False, "row beyond the .res refused")
        except ValueError as e:
            check("beyond" in str(e), f"row beyond the .res refused: {e}")
        try:
            fss.export(base, 1, [0, 1], out + "x", seed_ids=[2]); check(False, "wrong seed length refused")
        except ValueError as e:
            check("seed ids" in str(e), f"wrong seed length refused: {e}")
        fss.clean(out + "x")

        # ── a stage on the sub-session, with --final-clu ──
        final = out + ".clu.1"
        r = subprocess.run([pipe, "1", "--session", out, "--final-clu", final, "fiber-session", "--method", "stderiv",
                            "--chunk-min", "0.5", "--overlap-min", "0.1", "--min-group", "40", "--realign", "off",
                            "--whiten-source", "fil"],
                           cwd=td, env=env, capture_output=True, text=True, timeout=900)
        o = r.stdout + r.stderr
        check(r.returncode == 0, "fiber-pipeline --session <sub> fiber-session exits 0", o)
        check(os.path.isfile(final), "--final-clu delivered the final .clu", o)
        if os.path.isfile(final):
            _, ids = nio.read_clu_file(final)
            ids = np.asarray(ids)
            check(ids.size == want.size, f"the final .clu has one id per selected spike ({ids.size} == {want.size})")
            check(np.unique(ids[ids > 1]).size >= 2, f"the sub-session was sorted into >= 2 units ({np.unique(ids).tolist()[:8]})")
        check(r.returncode == 0 and "recording" in o and "samples" in o, "fiber-session read the real recording (.fil) through the link", o)
        # a plan with an out: step, dry run and real
        plan = os.path.join(td, "plan.yaml")
        with open(plan, "w") as fh:
            fh.write("pipeline:\n  - {stage: fiber-session, name: session, out: fs, params: {method: stderiv, chunk-min: 0.5, "
                     "overlap-min: 0.1, min-group: 40, realign: off}}\n")
        r = subprocess.run([pipe, "1", "--session", out, "--plan", plan, "--final-clu", out + ".clu.1.p", "--dry-run"],
                           cwd=td, env=env, capture_output=True, text=True)
        check(r.returncode == 0 and "would copy" in r.stdout + r.stderr and ".clu.stderiv.1.fs" in r.stdout + r.stderr,
              "a plan dry run names the final clu it would deliver", r.stdout + r.stderr)
        r = subprocess.run([pipe, "1", "--session", out, "--plan", plan, "--final-clu", out + ".clu.1.p"],
                           cwd=td, env=env, capture_output=True, text=True, timeout=900)
        check(r.returncode == 0 and os.path.isfile(out + ".clu.1.p"), "a plan run delivers its last out: clu", r.stdout + r.stderr)
        # fk run forwards
        r = fk("run", "1", "--plan", plan, "--session", out, "--final-clu", out + ".clu.1.q", "--dry-run", cwd=td, env=env)
        check(r.returncode == 0 and "--final-clu" not in r.stderr and "would copy" in r.stdout + r.stderr,
              "fk run --session --final-clu forwards to fiber-pipeline", r.stdout + r.stderr)
        # clean
        r = fk("subsession", "clean", out, cwd=td, env=env)
        left = [f for f in os.listdir(td) if f.startswith(os.path.basename(out) + ".")]
        check(r.returncode == 0 and not left, f"fk subsession clean removed everything under the sub-session ({r.stdout.strip()})")
        check(os.path.isfile(base + ".fil") and os.path.isfile(base + ".res.1"), "the session's own files are untouched")
    print(f"test_subsession: {'PASS' if not fails else 'FAIL'} ({fails} failed)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
