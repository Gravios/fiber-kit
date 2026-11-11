#!/usr/bin/env python3
"""fiber-manifold: the off-spike-channel field manifold of a unit, in sliding windows, as children and as
a .fet.manifold feature space.

On a synthetic 6-channel group (one unit on channels 0-2, 2400 spikes over 60 min, white noise; 30% of the
spikes carry a planted field on channels 3-5 (peak on channel 4, 0.8 ms AFTER the spike, decaying over the
neighbours); a second small unit that is below --min-n):
  * the stage exits 0 and writes .clu/.clc/.clp at the out tag, .fet.manifold.<g> (10 columns, the last the
    .res) and the .mft table;
  * the unit's children number <= --max-children and cover exactly the unit's spikes; the other unit keeps
    one atom; the .clu is unchanged (nothing moves between units);
  * the planted field is found: field spikes reach SNR >= 4 far more often than plain ones, their lag is
    positive (after) and their position near channel 4; the 'after' classes hold mostly field spikes;
  * --spk-variant runs the fit in another .spk space (the temporal derivative) and still finds the field;
  * --channels-set full fits the field over every channel and still finds the planted one;
  * --units with --jobs 2 gives the same per-spike classes (the parallel path is bit-identical);
  * an ISI pair planted 1 ms apart gets a flag (the farther-from-median member);
  * fiber-pipeline runs it as a step (`fiber-manifold` with FK_MANIFOLD_* from the env) and as the
    `klusters-manifold` plan on a sub-session (clu-stage '' -> tag manifold).
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from fiber_kit import neuro_io as nio   # noqa: E402

SR = 20000
NSAMP, NCH, PK = 40, 6, 16
fails = 0


def check(ok, what, out=""):
    global fails
    print(("  ok:   " if ok else "  FAIL: ") + what)
    if not ok and out:
        print(out[-3000:])
    fails += not ok


def session(td):
    rng = np.random.default_rng(5)
    os.makedirs(os.path.join(td, "man"), exist_ok=True)
    base = os.path.join(td, "man", "man")                        # fiber-pipeline names the session after its directory
    dur = 60 * 60 * SR
    n1, n2 = 2400, 200
    t1 = np.sort(rng.choice(np.arange(SR, dur - SR, 50), n1, replace=False))
    t2 = np.sort(rng.choice(np.arange(SR + 25, dur - SR, 50), n2, replace=False))   # never the same time as unit 2
    t1[1] = t1[0] + 20                                         # one ISI pair 1 ms apart
    s = np.arange(NSAMP) - (PK - 1)
    unit = -2500 * np.exp(-0.5 * (s / 2.0) ** 2) + 700 * np.exp(-0.5 * ((s - 6) / 3.0) ** 2)
    sh1 = np.array([1.0, 0.7, 0.35, 0.05, 0.02, 0.0])
    sh2 = np.array([0.0, 0.1, 0.4, 1.0, 0.5, 0.2])
    has_field = rng.random(n1) < 0.3
    lag = int(round(0.8e-3 * SR))                              # 16 samples after the spike
    field = -900 * np.exp(-0.5 * ((s - lag) / 2.0) ** 2)
    fsp = np.exp(-np.abs(np.arange(NCH) - 4.0) / 1.0) * (np.arange(NCH) >= 3)
    sig = 25.0
    X1 = rng.normal(0, sig, (n1, NSAMP, NCH)) + unit[None, :, None] * sh1[None, None, :]
    X1[has_field] += field[None, :, None] * fsp[None, None, :]
    X1 *= (1 + 0.1 * np.sin(t1 / dur * 2 * np.pi))[:, None, None]           # slow amplitude drift
    X2 = rng.normal(0, sig, (n2, NSAMP, NCH)) + 1800 * (unit / 2500)[None, :, None] * sh2[None, None, :]
    res = np.concatenate([t1, t2]); lab = np.concatenate([np.full(n1, 2), np.full(n2, 3)])
    X = np.concatenate([X1, X2])
    o = np.argsort(res, kind="stable")
    res, lab, X = res[o].astype(np.int64), lab[o], X[o]
    hf = np.zeros(res.size, bool); hf[o < n1] = has_field[o[o < n1]]
    res.tofile(base + ".res.1")
    nio.write_spk(base, 1, X.round().clip(-32768, 32767).astype(np.int16), variant="standard")
    nio.write_clu(base, 1, lab, variant="stderiv", tag="cur")
    with open(base + ".yaml", "w") as fh:
        fh.write(f"acquisitionSystem: {{nBits: 16, nChannels: {NCH}, samplingRate: {SR}}}\n"
                 "spikeDetection:\n  channelGroups:\n    - channels: [0, 1, 2, 3, 4, 5]\n"
                 f"      nSamples: {NSAMP}\n      peakSampleIndex: {PK}\n")
    D = NSAMP * NCH
    nio.write_nst(base + ".nst.standard.1", dict(nsamp=NSAMP, nchan=NCH, peak=PK, sigma=np.full(NCH, sig),
                                                  mean=np.zeros((NSAMP, NCH)), cov=np.eye(D) * sig ** 2))
    return base, res, lab, hf


def run(args, env, cwd, timeout=600):
    return subprocess.run([sys.executable, "-c", "from fiber_kit.fiber_manifold import main; import sys; "
                           "sys.exit(main(sys.argv[1:]))", *args], cwd=cwd, env=env, capture_output=True, text=True,
                          timeout=timeout)


def main():
    env = {k: v for k, v in os.environ.items() if not k.startswith("FK_")}
    env["FK_CONFIG"] = os.devnull
    env["PYTHONPATH"] = os.pathsep.join([os.path.join(ROOT, "src"), env.get("PYTHONPATH", "")])
    with tempfile.TemporaryDirectory() as td:
        # the stage's console script, as the package installs it (the repo's fiber-pipeline calls it by name)
        os.makedirs(os.path.join(td, "bin"))
        with open(os.path.join(td, "bin", "fiber-manifold"), "w") as fh:
            fh.write(f"#!/bin/sh\nexec {sys.executable} -m fiber_kit.fiber_manifold \"$@\"\n")
        os.chmod(os.path.join(td, "bin", "fiber-manifold"), 0o755)
        env["PATH"] = os.pathsep.join([os.path.join(td, "bin"), os.path.join(ROOT, "scripts"), env.get("PATH", "")])
        base, res, lab, hf = session(td)
        common = ["--clu-method", "stderiv", "--clu-stage", "cur", "--window-min", "20", "--step-min", "10",
                  "--min-n", "1000", "--min-window-n", "200", "--noise-n", "3000", "--time-bins", "4", "--off-channels", "3,4,5", "--jobs", "1"]
        sd = os.path.dirname(base)
        r = run([base, "1"] + common, env, sd)
        o = r.stdout + r.stderr
        check(r.returncode == 0, "fiber-manifold exits 0", o)
        for ext in ("clu", "clc", "clp"):
            check(os.path.isfile(f"{base}.{ext}.stderiv.1.cur_manifold"), f"wrote .{ext} at the out tag cur_manifold")
        check(os.path.isfile(base + ".fet.manifold.1") and os.path.isfile(base + ".mft.stderiv.1.cur_manifold.tsv"),
              "wrote .fet.manifold.1 and the .mft table")
        check("examining 1 unit(s): [2]" in o, "unit 3 (200 spikes) is below --min-n: only unit 2 is examined", o)
        _, clu2 = nio.read_clu_file(f"{base}.clu.stderiv.1.cur_manifold")
        check(np.array_equal(np.asarray(clu2), lab), "the .clu is unchanged (no spike moved between units)")
        _, clc = nio.read_clu_file(f"{base}.clc.stderiv.1.cur_manifold")
        clc = np.asarray(clc)
        clp = np.fromfile(f"{base}.clp.stderiv.1.cur_manifold", dtype=np.int32)[1:]
        parent = lambda atom: int(clp[atom - 1])   # noqa: E731
        kids = np.unique(clc[lab == 2]); others = np.unique(clc[lab == 3])
        check(1 < kids.size <= 200 and all(parent(k) == 2 for k in kids), f"unit 2 has {kids.size} children (<= 200), all mapped to 2")
        check(others.size == 1 and parent(int(others[0])) == 3, "the unexamined unit keeps one atom, mapped to itself")
        check(not np.intersect1d(kids, others).size, "no child is shared between units")
        f = nio.read_fet_file(base + ".fet.manifold.1")
        check(f.ok and f.values.shape == (res.size, 10) and np.array_equal(f.values[:, -1], res),
              f"the feature space has 10 columns, the last the .res ({f.values.shape if f.ok else f})")
        snr, lag_us, pos, cls = f.values[:, 0] / 100, f.values[:, 1], f.values[:, 2] / 100, f.values[:, 6]
        u2 = lab == 2
        ff = (snr[u2 & hf] >= 4).mean(); fp = (snr[u2 & ~hf] >= 4).mean()
        check(ff > 0.8 and fp < 0.15, f"the planted field is found: SNR>=4 for {100*ff:.0f}% of field spikes, {100*fp:.0f}% of plain ones")
        det = u2 & hf & (snr >= 4)
        check(np.median(lag_us[det]) > 500 and abs(np.median(pos[det]) - 4) < 0.6,
              f"its lag is after the spike (median {np.median(lag_us[det]):.0f} us) at channel ~4 (median {np.median(pos[det]):.2f})")
        after = u2 & (cls > 0) & ((cls - 1) % 3 == 2)
        check(after.sum() > 0.7 * (u2 & hf).sum() and hf[after].mean() > 0.9,
              f"the 'after' classes hold the field spikes ({after.sum()} spikes, {100*hf[after].mean():.0f}% planted)")
        check((f.values[u2, 7] == 1).sum() >= 1, "the 1-ms ISI pair got a flag")
        # --channels-set full: the field over every channel of the group; the planted one is still found there
        rf = run([base, "1"] + common + ["--channels-set", "full", "--out-tag", "full"], env, sd)
        of = rf.stdout + rf.stderr
        ffv = nio.read_fet_file(base + ".fet.manifold.1").values
        s2, l2, p2 = ffv[:, 0] / 100, ffv[:, 1], ffv[:, 2] / 100
        d2 = u2 & hf & (s2 >= 4)
        check(rf.returncode == 0 and "all channels [0, 1, 2, 3, 4, 5]" in of, "--channels-set full fits on every channel", of)
        check((s2[u2 & hf] >= 4).mean() > 0.8 and (s2[u2 & ~hf] >= 4).mean() < 0.15 and np.median(l2[d2]) > 500
              and abs(np.median(p2[d2]) - 4) < 0.6, "... and still finds the planted field at its lag and channel")
        # --spk-variant: any .spk space the session holds (here the temporal derivative, with no .nst of its own)
        spk_std = nio.open_spk_file(base + ".spk.standard.1", NSAMP, NCH)
        d = np.diff(np.asarray(spk_std).astype(np.int32), axis=1, prepend=np.asarray(spk_std)[:, :1].astype(np.int32))
        nio.write_spk(base, 1, d.clip(-32768, 32767).astype(np.int16), variant="stderiv")
        rs = run([base, "1"] + common + ["--spk-variant", "stderiv", "--out-tag", "deriv"], env, sd)
        os_ = rs.stdout + rs.stderr
        fv = nio.read_fet_file(base + ".fet.manifold.1").values
        s3, l3, p3 = fv[:, 0] / 100, fv[:, 1], fv[:, 2] / 100
        d3 = u2 & hf & (s3 >= 4)
        check(rs.returncode == 0 and "noise covariance from" not in os_, "--spk-variant stderiv: fits in that space (no .nst for it: white noise)", os_)
        check((s3[u2 & hf] >= 4).mean() > 0.8 and (s3[u2 & ~hf] >= 4).mean() < 0.15 and np.median(l3[d3]) > 500
              and abs(np.median(p3[d3]) - 4) < 0.6, f"... and finds the planted field there ({100*(s3[u2 & hf] >= 4).mean():.0f}% / {100*(s3[u2 & ~hf] >= 4).mean():.0f}%, lag {np.median(l3[d3]):.0f} us, ch {np.median(p3[d3]):.2f})")
        rs = run([base, "1"] + common + ["--spk-variant", "nosuch"], env, sd)
        check(rs.returncode != 0 and "no .spk for 'nosuch'" in rs.stderr, "an absent .spk variant is refused by name")
        rf = run([base, "1"] + common + ["--channels-set", "strong"], env, sd)
        check(rf.returncode == 2 and "invalid choice" in rf.stderr, "--channels-set takes off | full only")
        # the other units' atoms when the input HAS a hierarchy: build one and rerun to a new tag
        # (the parallel path and a pinned unit / off-channel set give the same classes)
        r2 = run([base, "1"] + common[:-2] + ["--jobs", "2", "--units", "2,3", "--min-n", "10", "--out-tag", "par"], env, sd)
        o2 = r2.stdout + r2.stderr
        check(r2.returncode == 0 and "examining 2 unit(s)" in o2, "--units pins the units (unit 3 examined too), --jobs 2 runs", o2)
        f2 = nio.read_fet_file(base + ".fet.manifold.1")
        check(f2.ok and np.array_equal(f2.values[u2, 6], cls[u2]), "the parallel run gives unit 2 the same classes as the serial one")
        check(np.unique(np.asarray(nio.read_clu_file(f"{base}.clc.stderiv.1.par")[1])[lab == 3]).size > 1, "unit 3 got children too")
        # as a fiber-pipeline step (the knobs from the env) and as the klusters-manifold plan on a sub-session
        pipe = os.path.join(ROOT, "scripts", "fiber-pipeline")
        env2 = dict(env, FK_MANIFOLD_WINDOW_MIN="20", FK_MANIFOLD_STEP_MIN="10", FK_MANIFOLD_MIN_N="1000",
                    FK_MANIFOLD_MIN_WINDOW_N="200", FK_MANIFOLD_NOISE_N="3000", FK_MANIFOLD_JOBS="1", FK_MANIFOLD_TIME_BINS="4")
        r3 = subprocess.run([pipe, "1", "--method", "stderiv", "fiber-manifold"], cwd=sd, env=dict(env2, STEP_IN="cur"),
                            capture_output=True, text=True, timeout=600)
        o3 = r3.stdout + r3.stderr
        check(r3.returncode == 0 and os.path.isfile(f"{base}.clc.stderiv.1.cur_manifold") and "--window-min 20" not in o3,
              "fiber-pipeline fiber-manifold (STEP_IN=cur) runs the stage, the knobs from the env", o3)
        # the klusters-manifold profile on a sub-session: an export of unit 2, in: '' -> tag manifold, --final-clu
        fk = lambda *a: subprocess.run([sys.executable, "-c", "from fiber_kit.plan_steps import main; import sys; "  # noqa: E731
                                        "sys.exit(main(sys.argv[1:]))", *a], cwd=sd, env=env2, capture_output=True, text=True, timeout=600)
        sub = base + "--2--t"
        r4 = fk("subsession", "export", base, "1", "--clu", f"{base}.clu.stderiv.1.cur", "--clusters", "2", "--out", sub)
        check(r4.returncode == 0, "fk subsession export of unit 2", r4.stdout + r4.stderr)
        final = sub + ".clu.1"
        r5 = fk("run", "1", "--profile", "klusters-manifold", "--session", sub, "--final-clu", final, "--method", "stderiv",
                "--FK_MANIFOLD_MIN_N=100")
        o5 = r5.stdout + r5.stderr
        check(r5.returncode == 0, "fk run --profile klusters-manifold --session <sub> exits 0", o5)
        check(os.path.isfile(final) and os.path.isfile(sub + ".fet.manifold.1") and os.path.isfile(sub + ".clc.stderiv.1.manifold"),
              "the sub-session got its final .clu, .clc at tag manifold and .fet.manifold.1 (for Klusters to carry back)")
        if os.path.isfile(final):
            _, ids = nio.read_clu_file(final)
            check(np.asarray(ids).size == int((lab == 2).sum()) and np.all(np.asarray(ids) == 2), "the final .clu: one id per selected spike, unchanged")
        # the REFINE contract (Klusters 0948): the sub-session holds the atoms of units 2 and 3 (seed .clu = the
        # parents, seed .clc/.clp = the atoms); the final .clc replaces the examined units' atoms by children
        # and keeps the others; the .clp maps every child to its parent
        sub2 = base + "--2-3--t"
        r6 = fk("subsession", "export", base, "1", "--clu", f"{base}.clu.stderiv.1.cur", "--clusters", "2", "3", "--out", sub2)
        check(r6.returncode == 0, "fk subsession export of units 2 and 3", r6.stdout + r6.stderr)
        sres = np.fromfile(sub2 + ".res.1", dtype=nio.RES_DTYPE)
        slab = lab[np.searchsorted(res, sres)]
        atoms = np.where(slab == 2, np.where(np.arange(sres.size) % 2 == 0, 10, 11), 12).astype(np.int64)   # 2 -> {10, 11}, 3 -> {12}
        np.r_[np.int32(3), atoms.astype(np.int32)].astype(np.int32).tofile(sub2 + ".clc.stderiv.1")
        clp = np.zeros(12, np.int32); clp[9] = clp[10] = 2; clp[11] = 3
        np.r_[np.int32(12), clp].astype(np.int32).tofile(sub2 + ".clp.stderiv.1")
        final2 = sub2 + ".clu.klusters.1"            # what Klusters asks for: its .clc/.clp twins come along
        r7 = fk("run", "1", "--profile", "klusters-manifold", "--session", sub2, "--final-clu", final2, "--method", "stderiv",
                "--FK_MANIFOLD_MIN_N=1000", "--FK_MANIFOLD_OFF_FRAC=0.3")
        o7 = r7.stdout + r7.stderr
        check(r7.returncode == 0 and os.path.isfile(sub2 + ".clc.klusters.1") and os.path.isfile(sub2 + ".clp.klusters.1"),
              "--final-clu delivered the .clc and .clp beside the final .clu", o7)
        if os.path.isfile(sub2 + ".clc.klusters.1"):
            _, c2 = nio.read_clu_file(sub2 + ".clc.klusters.1"); c2 = np.asarray(c2)
            p2 = np.fromfile(sub2 + ".clp.klusters.1", np.int32)[1:]
            new = np.unique(c2[slab == 2])
            check(np.all(c2[slab == 3] == 12), "the unexamined unit's atom (12) is kept in the final .clc")
            check(new.size > 1 and not np.isin(new, [10, 11, 12]).any(), f"the examined unit's atoms are replaced by {new.size} children")
            check(all(p2[a - 1] == 2 for a in new) and p2[11] == 3, "the final .clp maps the children to parent 2 and atom 12 to 3")
    print(f"test_manifold: {'PASS' if not fails else 'FAIL'} ({fails} failed)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
