"""fiber-bench: config hashes, plan rewriting, truth matching / scoring, and score + report end to end.

  * hashes: a session knob changes the session hash; a merge knob does not; a flag knob changes the
    flag hash AND the session's (the mask is its input); the merge step reads the hashed session tag;
    FK_SESSION_REALIGN is forced off (in-place commit), and an in-place fiber-realign step is refused;
  * --grid / --set / sweep files expand to the expected override list;
  * match_gt matches one-to-one within the tolerance, nearest offset first, and drops non-truth units;
  * truth_metrics: accuracy / recall / precision / frag80 / merges on a hand-built example, typed;
  * score + report on a synthetic session: metrics.tsv, units.tsv, clusters.tsv and an HTML report.
"""
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from fiber_kit import fiber_bench as fb    # noqa: E402
from fiber_kit import neuro_io as nio      # noqa: E402

fails = 0


def check(ok, what):
    global fails
    print(("  ok:   " if ok else "  FAIL: ") + what); fails += not ok


def main():
    cfg = fb.load_config(os.path.join(HERE, "..", "src", "fiber_kit", "fiber-kit-exp.yaml"))
    c0, s0, h0 = fb.build_run(cfg)
    check([s["stage"] for s in s0] == ["fiber-flag-artifacts", "fiber-session"] and h0 == s0[-1]["hash"],
          f"default run = the plan up to fiber-session ({[s.get('out_tag') for s in s0]})")
    check(c0["FK_SESSION_REALIGN"] == "off" and c0["FK_FLAG_STAGE"] == s0[0]["out_tag"],
          "session realign forced off; the flag stage tag is hashed and set in the run config")
    _, s1, h1 = fb.build_run(dict(cfg, FK_SESSION_COARSE_DR=0.21))
    _, s2, h2 = fb.build_run(dict(cfg, FK_XCM_COS_THR=0.5))
    _, s3, h3 = fb.build_run(dict(cfg, FK_FLAG_HF_FRAC=0.5))
    check(h1 != h0 and s1[0]["hash"] == s0[0]["hash"], "a session knob changes the session hash, not the flag's")
    check(h2 == h0, "a merge knob leaves the session hash alone")
    check(s3[0]["hash"] != s0[0]["hash"] and h3 != h0, "a flag knob changes the flag hash and the session's")
    _, s4, _ = fb.build_run(cfg, upto="merge")
    check(s4[-1]["in"] == s4[1]["out_tag"] and s4[-1]["out_tag"].startswith("exp_xcm_"),
          f"the merge step reads the hashed session tag ({s4[-1]['in']} -> {s4[-1]['out_tag']})")
    try:
        fb.build_run(cfg, upto="realign-final"); refused = False
    except SystemExit:
        refused = True
    check(refused, "an in-place fiber-realign step is refused without --allow-inplace")
    check(fb.build_run(cfg)[2] == h0, "the hash is deterministic")
    _, s5, h5 = fb.build_run(dict(cfg, FK_CHUNKS="14-16"))
    check(h5 != h0 and s5[0]["hash"] != s0[0]["hash"], "a chunk selection (FK_CHUNKS) changes every step's hash")

    ov = fb.parse_overrides(["FK_A=1"], ["FK_B=1,2", "FK_C=x,y"], None)
    check(len(ov) == 4 and all(o["FK_A"] == 1 for o in ov) and {(o["FK_B"], o["FK_C"]) for o in ov} ==
          {(1, "x"), (1, "y"), (2, "x"), (2, "y")}, "--set + two --grid axes -> 4 variants")
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "sw.yaml")
        with open(p, "w") as fh:
            fh.write("- {FK_D: 1}\n- {FK_D: 2, FK_E: 0}\n")
        ov2 = fb.parse_overrides([], [], p)
        check(ov2 == [{"FK_D": 1}, {"FK_D": 2, "FK_E": 0}], "a sweep file lists its variants")

    # truth matching: offsets 0, +2, -3, +6 (beyond tol 5); unit 9 is not a truth unit
    res = np.array([100, 200, 300, 400, 500])
    g_res = np.array([100, 202, 297, 406, 500]); g_lab = np.array([2, 2, 3, 3, 9])
    gt = fb.match_gt(res, g_res, g_lab, [2, 3], tol=5)
    check(gt.tolist() == [2, 2, 3, -1, -1], f"match_gt within +-5, unit 9 dropped ({gt.tolist()})")

    # truth metrics: unit 2 = 60 spikes (50 in cluster 10, 10 in cluster 11); unit 3 = 40 in cluster 11
    gt = np.array([2] * 60 + [3] * 40 + [-1] * 5)
    clu = np.array([10] * 50 + [11] * 10 + [11] * 40 + [0] * 5)
    S, rows = fb.truth_metrics(clu, gt, [2, 3], {2: "pyr", 3: "int"}, merge_min=5, merge_frac=0.2)
    r2 = next(r for r in rows if r["unit"] == 2); r3 = next(r for r in rows if r["unit"] == 3)
    check(abs(r2["acc"] - 50 / 60) < 1e-9 and abs(r3["acc"] - 40 / 50) < 1e-9 and r2["frag80"] == 1,
          f"accuracy n_uc/(n_u+g_c-n_uc): unit 2 {r2['acc']:.3f}, unit 3 {r3['acc']:.3f}")
    check(S["merges"] == 1 and S["merges_pyr_pyr"] == 0 and S["merges_with_int"] == 1 and abs(S["pyr_acc"] - 50 / 60) < 1e-9,
          f"typed summaries and merges ({S['merge_list']})")

    # score + report end to end on a synthetic session
    with tempfile.TemporaryDirectory() as td:
        base = os.path.join(td, "mini"); rng = np.random.default_rng(1)
        NS, NC, PK = 32, 4, 15
        t = np.arange(NS)
        W, lab = [], []
        for u, amp in enumerate((600, 900, 1300)):
            sh = -amp * np.exp(-0.5 * ((t - PK + 1) / 1.4) ** 2)
            for _ in range(200):
                W.append(sh[:, None] * np.array([1, 0.6, 0.3, 0.1])[None, :] * rng.normal(1, 0.04) + rng.normal(0, 60, (NS, NC)))
                lab.append(u + 2)
        W = np.clip(np.rint(np.array(W)), -32768, 32767).astype(np.int16); lab = np.array(lab)
        perm = rng.permutation(lab.size); W, lab = W[perm], lab[perm]
        res = np.sort(rng.choice(np.arange(20000 * 600), lab.size, replace=False))
        nio.write_spk(base, 1, W, variant="standard"); nio.write_res_file(base + ".res.1", res)
        sortlab = lab.copy(); sortlab[lab == 4] = 3                      # the sort merges units 3 and 4
        nio.write_clu_file(base + ".clu.stderiv.1.sortA", sortlab)
        nio.write_clu_file(base + ".gt.clu", lab); nio.write_res_file(base + ".gt.res", res)
        with open(base + ".yaml", "w") as fh:
            fh.write("acquisitionSystem: {nBits: 16, nChannels: 4, samplingRate: 20000}\n"
                     f"spikeDetection:\n  channelGroups:\n    - channels: [0, 1, 2, 3]\n      nSamples: {NS}\n"
                     f"      peakSampleIndex: {PK}\n")
        old = {k: v for k, v in os.environ.items() if k.startswith("FK_")}
        for k in old:
            os.environ.pop(k)
        os.environ["FK_CONFIG"] = os.devnull
        try:
            rc = fb.main(["score", base, "1", "--clu-method", "stderiv", "--stages", "sortA",
                          "--gt-clu", base + ".gt.clu", "--gt-res", base + ".gt.res", "--interneurons", "4"])
            bd = os.path.join(td, "fk-bench")
            M = fb._tsv_read(os.path.join(bd, "metrics.tsv"))
            check(rc == 0 and len(M) == 1 and M[0]["tag"] == "sortA", "score writes one metrics row for the baseline")
            m = M[0]
            # pyramidal units 2 (alone: accuracy 1) and 3 (shares its cluster with interneuron 4: 200/400 = 0.5)
            check(m["merges"] == 1 and m["merge_list"] in ("3+4", "4+3") and abs(m["pyr_acc"] - 0.75) < 1e-9
                  and m["merges_pyr_pyr"] == 0 and m["merges_with_int"] == 1,
                  f"the 3+4 merge is found and typed, pyramidal accuracy (1 + 0.5) / 2 ({m['merge_list']}, {m['pyr_acc']})")
            check(np.isfinite(m.get("rdr_median", np.nan)) and m["n_clusters"] == 2,
                  f"truth-free measures computed (rdr median {m.get('rdr_median')}, {int(m['n_clusters'])} clusters)")
            check(os.path.exists(os.path.join(bd, "units.tsv")) and os.path.exists(os.path.join(bd, "clusters.tsv")),
                  "units.tsv and clusters.tsv written")
            rc = fb.main(["report", base, "1"])
            with open(os.path.join(bd, "report.html")) as fh:
                html = fh.read()
            check(rc == 0 and "<table" in html and html.count("data:image/png") >= 2, "report.html with table and figures")
            # a chunk run: spikes in a reserved cluster are not scored (truth or truth-free)
            from fiber_kit import fiber_chunks as fch
            part = sortlab.copy(); part[:100] = fch.RESERVED_LO + 3
            nio.write_clu_file(base + ".clu.stderiv.1.sortC", part)
            fb.main(["score", base, "1", "--clu-method", "stderiv", "--stages", "sortC",
                     "--gt-clu", base + ".gt.clu", "--gt-res", base + ".gt.res", "--interneurons", "4"])
            mc = next(r for r in fb._tsv_read(os.path.join(bd, "metrics.tsv")) if r["tag"] == "sortC")
            check(mc["n_spikes"] == 500 and mc["chunks_left_out"] == 1 and mc["n_clusters"] == 2,
                  f"a chunk run scores only its computed spikes ({int(mc['n_spikes'])} of 600, "
                  f"{int(mc['chunks_left_out'])} chunk left out)")
            # two swept runs (as `run` would register them): the report gains the parameter-effect figure
            rows = [r for r in fb._tsv_read(os.path.join(bd, "metrics.tsv")) if r["tag"] == "sortA"]
            extra = [dict(rows[0], tag=f"sortA_{i}", hash=f"0000000{i}", overrides=json.dumps({"FK_SESSION_COARSE_DR": v}),
                          pyr_acc=0.5 + 0.1 * i) for i, v in enumerate((0.1, 0.2))]
            fb._tsv_write(os.path.join(bd, "metrics.tsv"), rows + extra)
            rc = fb.main(["report", base, "1"])
            with open(os.path.join(bd, "report.html")) as fh:
                html2 = fh.read()
            check(rc == 0 and "Parameter effects" in html2 and html2.count("data:image/png") >= 3
                  and "<td>0.1</td>" in html2, "with swept runs the report shows the parameter effects and their values")
            with open(os.path.join(bd, "runs.jsonl"), "w") as fh:
                fh.write(json.dumps(dict(hash="abc12345", ok=True, final_tag="sortA", overrides={"FK_X": 1}, seconds=1)) + "\n")
            rc = fb.main(["list", base, "1"])
            check(rc == 0, "list reads the registry")
            # a cached run with no successful record is registered on the next `run`
            open(os.path.join(bd, "runs.jsonl"), "w").close()
            cfgp = os.path.join(td, "cfg.yaml")
            with open(cfgp, "w") as fh:
                fh.write("FK_REALIGN_VARIANT: stderiv\nFK_SESSION_EXCLUDE: 0\npipeline:\n"
                         "  - {stage: fiber-session, name: session, out: fs}\n")
            _, st_, h_ = fb.build_run(fb.load_config(cfgp))
            nio.write_clu_file(f"{base}.clu.stderiv.1.{st_[-1]['out_tag']}", sortlab)
            rc = fb.main(["run", base, "1", "--config", cfgp])
            reg = fb.read_registry(base)
            check(rc == 0 and len(reg) == 1 and reg[0]["hash"] == h_ and reg[0]["ok"],
                  "a run whose output exists but was never registered is registered as cached")
        finally:
            os.environ.pop("FK_CONFIG", None); os.environ.update(old)
    print(f"\ntest_bench: {'FAIL' if fails else 'PASS'} ({fails} failed)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
