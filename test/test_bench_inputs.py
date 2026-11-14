#!/usr/bin/env python3
"""fiber-bench's step hashes cover the session files a run reads (input_fingerprint): a rewritten .spk/.res/.pca
gives every step a new tag, so a cached sort made from other inputs is never reused (g6 0316, 2026-11-12:
the .spk files were realigned between two fiber-tune runs and the second reused the first's sorts)."""
import os, sys, tempfile, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from fiber_kit import fiber_bench as fb   # noqa: E402

fails = 0


def check(ok, what):
    global fails
    print(("  ok:   " if ok else "  FAIL: ") + what); fails += not ok


cfg = {"pipeline": [{"stage": "fiber-session", "out": "fiber_session"},
                    {"stage": "fiber-xcorr-merge", "in": "fiber_session", "out": "xcm"}], "FK_XCM_COS_THR": 0.94}
with tempfile.TemporaryDirectory() as d:
    base = os.path.join(d, "rat")
    names = ["rat.spk.stderiv_C5.6", "rat.spk.standard.6", "rat.pca.stderiv_C5.6", "rat.nst.stderiv_C5.6",
             "rat.res.stderiv_C5.6", "rat.xml", "rat.yaml",
             "rat.fet.stderiv_C5.6",                       # derived: not an input
             "rat.spk.stderiv_C5.7",                       # another group
             "rat.clu.stderiv_C5.6.fiber_session_0123abcd",  # a stage output
             "rat.res.stderiv_C5.6.psame_89abcdef"]        # a stage output
    for n in names:
        with open(os.path.join(d, n), "wb") as fh:
            fh.write(b"x" * 10)
    fp = fb.input_fingerprint(base, 6)
    check(set(fp) == {"rat.spk.stderiv_C5.6", "rat.spk.standard.6", "rat.pca.stderiv_C5.6", "rat.nst.stderiv_C5.6",
                      "rat.res.stderiv_C5.6", "rat.xml", "rat.yaml"},
          f"the fingerprint lists the group's spk/pca/nst/res and the xml/yaml, not .fet, other groups or stage outputs ({sorted(fp)})")
    _, s0, h0 = fb.build_run(cfg, upto="fiber-xcorr-merge", inputs=fp)
    check(fb.build_run(cfg, upto="fiber-xcorr-merge", inputs=fb.input_fingerprint(base, 6))[2] == h0, "deterministic for unchanged files")
    _, sN, hN = fb.build_run(cfg, upto="fiber-xcorr-merge")
    check(hN != h0, "without a fingerprint (--no-input-hash) the tags are the pre-0.67.5 ones")
    time.sleep(0.01)
    with open(os.path.join(d, "rat.spk.standard.6"), "wb") as fh:
        fh.write(b"y" * 10)                                   # same size, new mtime: a realign's rewrite
    _, s1, h1 = fb.build_run(cfg, upto="fiber-xcorr-merge", inputs=fb.input_fingerprint(base, 6))
    check(h1 != h0 and s1[0]["hash"] != s0[0]["hash"], "a rewritten .spk (same size) changes the session step's tag and every later one")
    with open(os.path.join(d, "rat.fet.stderiv_C5.6"), "wb") as fh:
        fh.write(b"z" * 99)
    check(fb.build_run(cfg, upto="fiber-xcorr-merge", inputs=fb.input_fingerprint(base, 6))[2] == h1, "a rewritten .fet (emitted by a run) does not")
    with open(os.path.join(d, "rat.clu.stderiv_C5.6.fiber_session_0123abcd"), "wb") as fh:
        fh.write(b"z" * 99)
    check(fb.build_run(cfg, upto="fiber-xcorr-merge", inputs=fb.input_fingerprint(base, 6))[2] == h1, "nor does a stage output")
# FK_SESSION_JOBS / JOB_MEM_GIB change only the runtime, so a profile may gain or change them without
# invalidating a single cached artifact.
_st = {"stage": "fiber-session", "params": {}}
_cf = {"FK_SESSION_DIP_ALPHA": 0.03, "FK_SESSION_FEAT_HYBRID": 1}
_h0 = fb.step_hash(_st, _cf, [])
check(all(fb.step_hash(_st, {**_cf, **x}, []) == _h0
          for x in ({"FK_SESSION_JOBS": 0}, {"FK_SESSION_JOBS": 64, "FK_SESSION_JOB_MEM_GIB": 6.0})),
      "the performance knobs are outside the step hash")
check(fb.step_hash(_st, {**_cf, "FK_SESSION_DIP_ALPHA": 0.02}, []) != _h0, "a real knob still moves the hash")
check("FK_SESSION_JOBS" not in fb.stage_params("fiber-session", {**_cf, "FK_SESSION_JOBS": 8}),
      "stage_params drops the performance knobs")

print(f"test_bench_inputs: {'PASS' if not fails else 'FAIL'} ({fails} failed)")
sys.exit(1 if fails else 0)
