#!/usr/bin/env python3
"""
test_session_cfg.py — SessionCfg must actually CONSTRUCT.

Why this exists.  resolve_session_params() ends in SessionCfg(**cfg): a dataclass
built by keyword splat from a dict assembled elsewhere in the module.  Add a key
to that dict without adding the matching field and every stage dies at startup
with

    TypeError: SessionCfg.__init__() got an unexpected keyword argument '...'

which is what shipped when sdiff_pairs was added for fiber_realign's stderiv_C4/_C5
re-extraction.  compile() and pyflakes BOTH pass on that code -- the mismatch only
exists at call time -- so no static check in this repo could have caught it.  It
needs one execution, which is all this is.

The dual access path is checked too: SessionCfg deliberately supports cfg.field
AND cfg["field"], the second gated on a _SESSION_FIELDS tuple.  A field added to
the dataclass but missing from that tuple works via attribute access and raises
KeyError via mapping access -- a half-broken state that only shows up in whichever
caller happens to use the other form.

Run:  python3 test/test_session_cfg.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fiber_kit import session_yaml as sy   # noqa: E402

YAML = """\
acquisitionSystem: {nChannels: 96, samplingRate: 32552, nBits: 16}
spikeDetection:
  channelGroups:
    - channels: [0,1,2,3,4,5,6,7]
      nSamples: 42
      peakSampleIndex: 21
      nFeatures: 3
    - channels: [32,33,34,35,36,37,38,39]
      nSamples: 42
      peakSampleIndex: 21
      nFeatures: 3
      sdiffPairs: "0-1+2,1-0+2,2-0+1,3-4+5,4-3+5,5-3+4,6-7+0,7-6+1"
"""

PATTERN = "0-1+2,1-0+2,2-0+1,3-4+5,4-3+5,5-3+4,6-7+0,7-6+1"


def check(cond, what):
    print(f"  {'ok  ' if cond else 'FAIL'} {what}")
    return 0 if cond else 1


def main():
    bad = 0
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "sess.yaml")
        open(path, "w").write(YAML)

        # 1. the construction that broke: a group WITH a custom pattern
        cfg = sy.resolve_session_params(path, 2, require=("ntotal", "sr"), verbose=False)
        bad += check(cfg.sdiff_pairs == PATTERN, "group with sdiffPairs: attribute access")
        bad += check(cfg["sdiff_pairs"] == PATTERN, "group with sdiffPairs: mapping access")
        bad += check(cfg.get("sdiff_pairs") == PATTERN, "group with sdiffPairs: .get()")
        bad += check("sdiff_pairs" in cfg.keys(), "field listed in keys()")

        # 2. a group WITHOUT one resolves to None rather than raising
        c1 = sy.resolve_session_params(path, 1, require=("ntotal", "sr"), verbose=False)
        bad += check(c1.sdiff_pairs is None, "group without sdiffPairs -> None")
        bad += check(c1.nsamp == 42 and c1.peak == 21, "other fields still resolve")

        # 3. the no-YAML fallback builds a SessionCfg too, from a different dict
        c2 = sy.resolve_session_params(os.path.join(d, "base_only"), 5,
                                       channels="1,2,3", ntotal=96, sr=32552.0,
                                       verbose=False)
        bad += check(c2.sdiff_pairs is None, "no-YAML fallback constructs")

        # 4. every dataclass field must be reachable by BOTH access paths
        import dataclasses
        names = [f.name for f in dataclasses.fields(sy.SessionCfg)]
        missing = [n for n in names if n not in cfg.keys()]
        bad += check(not missing,
                     f"all {len(names)} dataclass fields exposed to mapping access"
                     + (f" (missing {missing})" if missing else ""))

        # 5. and an unknown key must still fail loudly -- the reason the type exists
        try:
            cfg["samplingRate"]
            bad += check(False, "unknown key raises KeyError")
        except KeyError:
            bad += check(True, "unknown key raises KeyError")

    print(f"\n{'FAIL' if bad else 'PASS'}: {bad} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
