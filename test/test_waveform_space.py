"""The waveform space a .spk variant token names: fiber_stderiv.space_transform (raw .fil windows ->
that space) and neuro_io.spk_variant (the token of the file a stage actually opened).  The noise
whitener is built from these, so a whitener in the wrong space is the failure they exist to prevent."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from fiber_kit import fiber_envelope as fe  # noqa: E402
from fiber_kit import fiber_stderiv as fst  # noqa: E402
from fiber_kit import neuro_io as nio  # noqa: E402

SETS8 = "0-2+4,1-3+5,2-4+0+6,3-5+1+7,4-6+2+7,5-7+3+6,6-7+3+5,7-5+6"    # a real 8-ch order-5 pattern
PAIRS8 = "0-1,1-2,2-3,3-4,4-5,5-6,6-7"                                  # order 4, root = last (7)
EXPR8 = "c0-c1,c1-c2,c2-c3,c3-c4,c4-c5,c5-c6,c6-c7,c7-c6"               # order 6, same operator as PAIRS8


def _raw(n=6, L=21, C=8, seed=0):
    return np.random.default_rng(seed).normal(0, 300, (n, L + 1, C)).round()


def test_untransformed_spaces():
    for tok in ("", None, "standard", "fil"):
        assert fst.space_transform(tok) is None, tok


def test_allpairs_and_legacy_spkD():
    r = _raw()
    want = fst.apply_stderiv_transform(r)
    for tok in ("stderiv", "D", "stderiv_S3", "stderiv_D34"):
        got = fst.space_transform(tok)(r)
        assert got.dtype == np.float32 and got.shape == want.shape, tok
        np.testing.assert_array_equal(got, want)


def test_plain_orders():
    r = _raw(seed=1)
    for order in (1, 2):
        partner, sets = fst.plain_order_spec(order, r.shape[2])
        np.testing.assert_array_equal(fst.space_transform(f"stderiv_S{order}")(r),
                                      fst.apply_stderiv_transform(r, partner=partner, sets=sets))


def test_session_patterns():
    r = _raw(seed=2)
    np.testing.assert_array_equal(fst.space_transform("stderiv_C5", SETS8)(r),
                                  fst.apply_stderiv_transform(r, sets=fst.parse_sdiff_sets(SETS8)))
    np.testing.assert_array_equal(fst.space_transform("stderiv_C5_D34", SETS8)(r),      # _D ignored
                                  fst.space_transform("stderiv_C5", SETS8)(r))
    partner, _ = fst.parse_sdiff_pairs(PAIRS8)
    np.testing.assert_array_equal(fst.space_transform("stderiv_C4", PAIRS8)(r),
                                  fst.apply_stderiv_transform(r, partner=partner))
    np.testing.assert_array_equal(fst.space_transform("stderiv_C6", EXPR8)(r),
                                  fst.space_transform("stderiv_C4", PAIRS8)(r))
    np.testing.assert_array_equal(fst.space_transform("stderiv_C6", PAIRS8)(r),           # older form, read
                                  fst.space_transform("stderiv_C4", PAIRS8)(r))           # as expressions


def test_refusals():
    for bad in (("stderiv_C5", None), ("stderiv_C5", ""), ("stderiv_C4", SETS8), ("stderiv_C5", PAIRS8),
                ("stderiv_S4", None), ("pca", None), ("sdiff", None)):
        try:
            fst.space_transform(*bad)
        except ValueError:
            continue
        raise AssertionError(f"space_transform{bad} should refuse")


def test_envelope_uses_the_shared_definition():
    r = _raw(seed=3)
    assert fe.space_transform("standard") is None
    np.testing.assert_array_equal(fe.space_transform("stderiv_C5", SETS8)(r),
                                  fst.space_transform("stderiv_C5", SETS8)(r))
    np.testing.assert_array_equal(fe.space_transform("stderiv_S2")(r), fst.space_transform("stderiv_S2")(r))


def test_labels():
    assert fst.space_label("standard") == "raw .fil (standard)"
    assert fst.space_label("stderiv") == "stderiv, all-pairs"
    assert "order 5" in fst.space_label("stderiv_C5_D34", SETS8) and SETS8 in fst.space_label("stderiv_C5", SETS8)
    assert fst.space_label("stderiv_S2") == "stderiv, plain spatial order 2"


def test_spk_variant_of_a_resolved_path():
    b = os.path.join("x", "sess")
    assert nio.spk_variant(f"{b}.spk.stderiv_C5.7", b, 7) == "stderiv_C5"
    assert nio.spk_variant(f"{b}.spk.stderiv_C5_D34.7", b, "7") == "stderiv_C5_D34"
    assert nio.spk_variant(f"{b}.spk.standard.12", b, 12) == "standard"
    assert nio.spk_variant(f"{b}.spkD.3", b, 3) == "D"
    assert nio.spk_variant(f"{b}.spk.3", b, 3) == ""
    for bad in (f"{b}.fet.stderiv.3", f"{b}.spk.stderiv.4", f"{b}x.spk.3"):
        try:
            nio.spk_variant(bad, b, 3)
        except ValueError:
            continue
        raise AssertionError(f"spk_variant({bad!r}) should refuse")


def test_spk_variant_with_open_spkD():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        base = os.path.join(d, "s")
        np.zeros((4, 32, 8), np.int16).tofile(f"{base}.spk.stderiv_C5.2")
        _, path = nio.open_spkD(base, 2, 32, 8, prefer="stderiv_C5_D34")     # a _D token reads its waveforms
        assert nio.spk_variant(path, base, 2) == "stderiv_C5"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("ok", fn.__name__)
    print(f"\n{len(fns)} passed")
