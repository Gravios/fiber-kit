"""fiber-clean: the core cut (a numpy port of Klusters' corecut.h), the quality value, and the stage.

PARITY with the C++ header was measured on real data, which this test cannot carry: on
sirotaA-jg-000005-20120312 group 8 (2026-11-23) fiber_corecut.analyse gave the header's keep mask
spike for spike on 40 fs_polish clusters of 200-5000 spikes (36,214 spikes), on 8 of 6,000-22,000
spikes (121,689) and, ordering a subset of 2,000, on 4 of 6,400-10,400 (36,320 spikes); fiber_clean.clean
relabelled 643 clusters (279,964 spikes) exactly as the analysis script that drives the header did.
No cluster above 40,000 spikes was compared.

Here, on synthetic clusters:
  * a unit with a quarter of its spikes from a look-alike cell (same primary channel, a different
    weak channel): the cut removes most of the intruder and keeps most of the unit;
  * NEGATIVE CONTROL -- a cluster of one unit keeps at least half its spikes (the cut errs on the
    side of splitting; it does not empty a clean cluster);
  * too few spikes: nothing is cut and the reason says so;
  * the quality value: about 0 for one unit, clearly higher for two units of different size in one
    cluster, NaN under 30 spikes; it equals the definition computed the slow way;
  * the stage: cut-off spikes, a dirty cluster, a small cluster and a junk cluster go to the noise;
    cluster 0, the noise and a reserved cluster are untouched; the report says what happened.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from fiber_kit import fiber_chunks as fch         # noqa: E402
from fiber_kit import fiber_corecut as fcc        # noqa: E402
from fiber_kit import fiber_clean as fcl          # noqa: E402
from fiber_kit import neuro_io as nio             # noqa: E402

SR, NS, NC, PEAK = 32552.0, 52, 8, 26


def template(depth, amp, width=1.0):
    t = (np.arange(NS) - PEAK) / SR * 1e3
    w = -np.exp(-0.5 * (t / (0.12 * width)) ** 2) + 0.35 * np.exp(-0.5 * ((t - 0.45 * width) / (0.25 * width)) ** 2)
    foot = np.exp(-0.5 * ((np.arange(NC) * 20.0 - depth) / 28.0) ** 2) + 0.05
    return amp * w[:, None] * foot[None, :]


def spikes(rng, T, n, sd=18.0, amp_sd=0.05):
    return (T[None] * rng.normal(1.0, amp_sd, (n, 1, 1)) + rng.normal(0, sd, (n, NS, NC))).astype(np.float32)


def main():
    ok = 0

    def check(name, cond):
        nonlocal ok
        assert cond, name
        ok += 1
        print(f"  ok  {name}")

    rng = np.random.default_rng(0)
    TA = template(60, 420)
    TB = TA.copy(); TB[:, 6:] += template(140, 260)[:, 6:]        # the same cell on its channels + a far channel
    A = spikes(rng, TA, 1500); Bx = spikes(rng, TB, 500)
    W = np.concatenate([A, Bx]); isb = np.r_[np.zeros(1500, bool), np.ones(500, bool)]
    o = rng.permutation(len(W)); W, isb = W[o], isb[o]
    r = fcc.analyse(W, SR)
    cutb = float((~r["keep"])[isb].mean()); keepa = float(r["keep"][~isb].mean())
    print(f"      mixture: {cutb:.1%} of the intruder cut off, {keepa:.1%} of the unit kept "
          f"(cut at {r['cut']} of {r['ordered']}, channel {r['cut_channel']})")
    check("core cut ran and cut", r["ok"] and r["cut"] < r["ordered"] and r["primary_channel"] == 3)
    check("core cut: >= 80 % of the intruder is cut off", cutb >= 0.80)
    check("core cut: >= 80 % of the unit is kept", keepa >= 0.80)
    check("core cut: the order is a permutation and keep marks the tail",
          sorted(r["order"]) == list(range(len(W))) and not r["keep"][r["order"][r["cut"]:]].any()
          and r["keep"][r["order"][:r["cut"]]].all())
    rp = fcc.analyse(spikes(rng, TA, 1500), SR)
    print(f"      one unit: {rp['keep'].mean():.1%} kept ({rp['reason'] or 'cut'})")
    check("NEGATIVE CONTROL: a cluster of one unit keeps >= half its spikes", rp["ok"] and rp["keep"].mean() >= 0.5)
    rs = fcc.analyse(A[:150], SR)
    check("too few spikes: not run, everything kept", not rs["ok"] and rs["reason"] == "too few spikes" and rs["keep"].all())
    rsub = fcc.analyse(W, SR, max_ordered=800)
    check("above max_ordered: a subset is ordered and every spike still gets a verdict",
          rsub["ordered"] == 800 and rsub["keep"].shape == (2000,)
          and (~rsub["keep"])[isb].mean() >= 0.6 and rsub["keep"][~isb].mean() >= 0.7)

    TQ = template(60, 420, width=0.5)                 # narrow: nothing of the spike in the window's edges
    q1 = fcc.residual_variance(spikes(rng, TQ, 800, amp_sd=0.0))
    two = np.concatenate([spikes(rng, TQ, 400, amp_sd=0.0), spikes(rng, 0.45 * TQ, 400, amp_sd=0.0)])
    q2 = fcc.residual_variance(two)
    print(f"      quality: one unit {q1:.3f}, two sizes in one cluster {q2:.3f}")
    check("quality: one unit reads about 0", abs(q1) < 0.03)
    check("quality: two sizes in one cluster read much higher", q2 > q1 + 0.1)
    check("quality: NaN under 30 spikes", np.isnan(fcc.residual_variance(A[:29])))
    Wq = two.astype(np.float64); e = 5
    med = np.median(Wq, 0); num = 0.0; den = 0.0
    for c in range(NC):
        noise = np.mean(np.r_[Wq[:, :e, c].ravel(), Wq[:, NS - e:, c].ravel()] ** 2)
        for s in range(e, NS - e):
            num += np.mean((Wq[:, s, c] - med[s, c]) ** 2) - noise; den += med[s, c] ** 2
    check("quality: equals the definition computed sample by sample", abs(q2 - num / den) < 1e-9)

    # the stage
    with tempfile.TemporaryDirectory() as td:
        base = os.path.join(td, "syn")
        resv = int(fch.RESERVED_LO)
        parts = [(W, 5), (spikes(rng, template(120, 380), 900), 6), (two[::5], 7),   # 160: under --cc-min, so not cut first
                 (spikes(rng, template(20, 300), 40), 8), (spikes(rng, TA, 300), 9),
                 (spikes(rng, TA, 60), 0), (spikes(rng, TA, 80), 1), (spikes(rng, TA, 70), resv)]
        wav = np.concatenate([p for p, _ in parts]); lab = np.concatenate([np.full(len(p), k) for p, k in parts])
        o = rng.permutation(len(lab)); wav, lab = wav[o], lab[o]
        n = len(lab)
        (np.arange(n, dtype=np.int64) * 200 + 1000).tofile(f"{base}.res.1")
        np.clip(np.rint(wav), -32768, 32767).astype(np.int16).tofile(f"{base}.spk.standard.1")
        nio.write_clu(base, 1, lab, variant="standard", tag="in")
        with open(f"{base}.yaml", "w") as f:
            f.write("acquisitionSystem: {nChannels: 8, samplingRate: 32552, nBits: 16}\n"
                    "spikeDetection:\n  channelGroups:\n"
                    "    - channels: [0,1,2,3,4,5,6,7]\n      nSamples: 52\n      peakSampleIndex: 26\n"
                    "      nFeatures: 3\n")
        env = {k: v for k, v in os.environ.items() if not k.startswith("FK_")}
        env["PYTHONPATH"] = os.pathsep.join([os.path.join(HERE, "..", "src"), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
        r1 = subprocess.run([sys.executable, "-m", "fiber_kit.fiber_clean", base, "1", "--clu-method", "standard",
                             "--clu-stage", "in", "--junk", "9", "--q-dirty", "0.12"],
                            capture_output=True, text=True, env=env, timeout=900)
        if r1.returncode != 0:
            print((r1.stdout + r1.stderr)[-3000:])
        check("fiber-clean exits 0", r1.returncode == 0)
        _, got = nio.read_clu_file(f"{base}.clu.standard.1.in_clean")
        with open(f"{base}.clean.standard.1.in_clean.tsv") as fh:
            rep = {int(x[0]): x for x in (ln.rstrip("\n").split("\t") for ln in fh.readlines()[1:])}
        check("the report has one row per input cluster above 1 that is not reserved", sorted(rep) == [5, 6, 7, 8, 9])
        check("cluster 0, the noise and the reserved cluster are untouched",
              (got[lab == 0] == 0).all() and (got[lab == 1] == 1).all() and (got[lab == resv] == resv).all())
        check("a spike only ever stays or goes to the noise", ((got == lab) | (got == 1)).all())
        check("the mixed cluster is kept, less its cut-off spikes",
              rep[5][5] == "kept" and 0.5 * 2000 < int(rep[5][2]) < 2000 and (got == 5).sum() == int(rep[5][2]))
        check("the clean cluster is kept", rep[6][5] == "kept" and (got == 6).sum() == int(rep[6][2]) >= 450)
        check("the two-size cluster (too small for the cut) is dirty and goes to the noise whole", rep[7][5] == "dirty" and (got == 7).sum() == 0)
        check("the 40-spike cluster is small and goes to the noise", rep[8][5] == "small" and (got == 8).sum() == 0)
        check("the junk cluster goes to the noise", rep[9][5] == "junk" and (got == 9).sum() == 0)
        r2 = subprocess.run([sys.executable, "-m", "fiber_kit.fiber_clean", base, "1", "--clu-method", "standard",
                             "--clu-stage", "in", "--core-cut", "0", "--q-dirty", "0", "--min-keep", "1",
                             "--out-tag", "noop"], capture_output=True, text=True, env=env, timeout=900)
        _, same = nio.read_clu_file(f"{base}.clu.standard.1.noop")
        check("with the cut and the quality test off, nothing moves", r2.returncode == 0 and np.array_equal(same, lab))
        # --junk residue: the id fiber-polish recorded for the input stage
        common = [sys.executable, "-m", "fiber_kit.fiber_clean", base, "1", "--clu-method", "standard", "--clu-stage", "in",
                  "--core-cut", "0", "--q-dirty", "0", "--min-keep", "1"]
        r3 = subprocess.run(common + ["--junk", "residue", "--out-tag", "nores"], capture_output=True, text=True, env=env, timeout=900)
        check("--junk residue on a stage without a polish record stops and names the file",
              r3.returncode != 0 and "polish.standard.1.in.tsv" in (r3.stdout + r3.stderr) and not os.path.exists(f"{base}.clu.standard.1.nores"))
        rec = fcl.write_residue_record(base, 1, "standard", "in", 9, int((lab == 9).sum()), source="x")
        check("the record is <base>.polish.<method>.<group>.<stage>.tsv and reads back",
              rec.endswith(".polish.standard.1.in.tsv") and fcl.read_residue_record(base, 1, "standard", "in") == 9)
        r4 = subprocess.run(common + ["--junk", "residue", "--out-tag", "res"], capture_output=True, text=True, env=env, timeout=900)
        _, g4 = nio.read_clu_file(f"{base}.clu.standard.1.res")
        check("--junk residue sends the recorded residue cluster to the noise and nothing else",
              r4.returncode == 0 and (g4[lab == 9] == 1).all() and np.array_equal(g4[lab != 9], lab[lab != 9]))
        r5 = subprocess.run(common + ["--junk", "residue,6", "--out-tag", "res6"], capture_output=True, text=True, env=env, timeout=900)
        _, g5 = nio.read_clu_file(f"{base}.clu.standard.1.res6")
        check("the word and ids can be given together", r5.returncode == 0 and (g5[np.isin(lab, (6, 9))] == 1).all())
        fcl.write_residue_record(base, 1, "standard", "in", None, 0)
        r6 = subprocess.run(common + ["--junk", "residue", "--out-tag", "res0"], capture_output=True, text=True, env=env, timeout=900)
        _, g6 = nio.read_clu_file(f"{base}.clu.standard.1.res0")
        check("a polish that made no residue cluster: nothing is junked", r6.returncode == 0 and np.array_equal(g6, lab))

    print(f"test_clean: {ok} checks passed")


if __name__ == "__main__":
    main()
