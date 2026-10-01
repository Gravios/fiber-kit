"""Method-pinned staged readers: read_res_at / read_fet_at / open_spk_at.

Covers the de-collision staging contract: a derived stage carries the clu's
STAGE on the `tag` axis while every file keeps its own method on the `variant`
axis (res holds its detection method, spk/fet/clu hold theirs).  The staged res
must be reachable ONLY by its exact (variant, tag) -- never via read_res's
shared resolve_any -- so a staged res that deviates from the original is not
conflated with it.
"""
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from fiber_kit import neuro_io as nio  # noqa: E402


def _base(d):
    return os.path.join(d, "sess")


def test_res_at_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        b = _base(d)
        t = np.array([10, 20, 30, 40], np.int64)
        nio.write_res(b, 6, t, variant="stderiv", tag="decollided")
        got = nio.read_res_at(b, 6, variant="stderiv", tag="decollided")
        assert np.array_equal(got, t)
        # the physical name is exactly the contract path
        assert os.path.exists(f"{b}.res.stderiv.6.decollided")


def test_res_at_not_conflated_with_shared():
    # The shared original and a deviating staged res coexist; each resolves to
    # its own content -- read_res (resolve_any) must NOT return the staged copy.
    with tempfile.TemporaryDirectory() as d:
        b = _base(d)
        orig = np.array([1, 2, 3], np.int64)          # detection res (shared)
        deco = np.array([1, 2, 3, 4, 5], np.int64)    # de-collided: MORE spikes
        nio.write_res(b, 6, orig, variant="stderiv")                 # <base>.res.stderiv.6
        nio.write_res(b, 6, deco, variant="stderiv", tag="decollided")
        assert np.array_equal(nio.read_res(b, 6), orig)              # shared -> original
        assert np.array_equal(
            nio.read_res_at(b, 6, variant="stderiv", tag="decollided"), deco)


def test_fet_at_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        b = _base(d)
        vals = np.arange(12, dtype=np.int64).reshape(4, 3)
        nio.write_fet(b, 6, vals, variant="stderiv_C5_D34", tag="decollided")
        fb = nio.read_fet_at(b, 6, variant="stderiv_C5_D34", tag="decollided")
        assert fb.ok and fb.n_spikes == 4 and fb.n_features == 3
        assert np.array_equal(fb.values, vals)


def _write_spk(path, arr):
    np.asarray(arr, dtype=nio.SPK_DTYPE).tofile(path)


def test_spk_at_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        b = _base(d)
        nsamp, nchan = 4, 2
        arr = np.arange(3 * nsamp * nchan, dtype=nio.SPK_DTYPE).reshape(3, nsamp, nchan)
        _write_spk(f"{b}.spk.standard.6.decollided", arr)
        mm = nio.open_spk_at(b, 6, nsamp, nchan, variant="standard", tag="decollided")
        assert mm.shape == (3, nsamp, nchan)
        assert np.array_equal(np.asarray(mm), arr)


def test_spk_at_does_not_fall_back_across_variants():
    # A DIFFERENT variant's staged file present must not satisfy a pinned read:
    # substituting a transformed waveform for raw would break the amplitude law.
    with tempfile.TemporaryDirectory() as d:
        b = _base(d)
        nsamp, nchan = 4, 2
        arr = np.ones((2, nsamp, nchan), dtype=nio.SPK_DTYPE)
        _write_spk(f"{b}.spk.stderiv.6.decollided", arr)   # only the stderiv copy exists
        try:
            nio.open_spk_at(b, 6, nsamp, nchan, variant="standard", tag="decollided")
            raise AssertionError("expected FileNotFoundError (no fall-back to another variant)")
        except FileNotFoundError:
            pass


def test_per_type_variant_shared_tag():
    # The staging rule: tag unified to the clu's stage; each file keeps its own
    # variant.  res = detection method, clu = feature variant, same tag.
    with tempfile.TemporaryDirectory() as d:
        b = _base(d)
        nio.write_res(b, 6, np.array([5, 6, 7], np.int64), variant="stderiv", tag="dc")
        nio.write_clu(b, 6, np.array([2, 2, 3], np.int64), variant="stderiv_C5_D34", tag="dc")
        assert np.array_equal(
            nio.read_res_at(b, 6, variant="stderiv", tag="dc"), [5, 6, 7])
        n_clu, ids = nio.read_clu_at(b, 6, variant="stderiv_C5_D34", tag="dc")
        assert np.array_equal(ids, [2, 2, 3])


def test_tag_dot_is_sanitized_consistently():
    # A multi-part stage joins with '_' on BOTH write and read, so it round-trips.
    with tempfile.TemporaryDirectory() as d:
        b = _base(d)
        t = np.array([9, 9], np.int64)
        nio.write_res(b, 6, t, variant="stderiv", tag="pair.decollided")
        assert os.path.exists(f"{b}.res.stderiv.6.pair_decollided")
        assert np.array_equal(
            nio.read_res_at(b, 6, variant="stderiv", tag="pair.decollided"), t)


def test_pinned_miss_names_tokens_on_disk():
    with tempfile.TemporaryDirectory() as d:
        b = _base(d)
        # a sibling token exists for the stage, but not the one asked for
        nio.write_res(b, 6, np.array([1], np.int64), variant="stderiv_C5", tag="dc")
        try:
            nio.read_res_at(b, 6, variant="stderiv_C5_D34", tag="dc")
            raise AssertionError("expected FileNotFoundError")
        except FileNotFoundError as e:
            s = str(e)
            assert "tag='dc'" in s and "stderiv_C5" in s   # names what IS there


def test_cluster_res_at_pair_different_variants():
    # The matched staged pair: res on its detection variant, clu on its feature
    # variant, same tag.  A single `prefer` could not name both.
    with tempfile.TemporaryDirectory() as d:
        b = _base(d)
        nio.write_res(b, 6, np.array([10, 20, 30], np.int64), variant="stderiv", tag="dc")
        nio.write_clu(b, 6, np.array([2, 3, 2], np.int64), variant="stderiv_C5_D34", tag="dc")
        cr = nio.read_cluster_res_at(b, 6, clu_variant="stderiv_C5_D34",
                                     res_variant="stderiv", tag="dc")
        assert cr.binary and cr.ok
        assert np.array_equal(cr.times, [10, 20, 30])
        assert np.array_equal(cr.ids, [2, 3, 2])


def test_cluster_res_at_length_mismatch_is_not_ok():
    # res and clu that grew out of lockstep -> ok=False, no raise (the guard).
    with tempfile.TemporaryDirectory() as d:
        b = _base(d)
        nio.write_res(b, 6, np.array([1, 2, 3], np.int64), variant="stderiv", tag="dc")
        nio.write_clu(b, 6, np.array([2, 2, 2, 2], np.int64), variant="stderiv_C5_D34", tag="dc")
        cr = nio.read_cluster_res_at(b, 6, clu_variant="stderiv_C5_D34",
                                     res_variant="stderiv", tag="dc")
        assert cr.binary and not cr.ok


def test_cluster_res_at_missing_raises():
    with tempfile.TemporaryDirectory() as d:
        b = _base(d)
        nio.write_res(b, 6, np.array([1], np.int64), variant="stderiv", tag="dc")
        try:  # clu of the pair is absent
            nio.read_cluster_res_at(b, 6, clu_variant="stderiv_C5_D34",
                                    res_variant="stderiv", tag="dc")
            raise AssertionError("expected FileNotFoundError for the missing clu")
        except FileNotFoundError as e:
            assert "clu" in str(e) and "tag='dc'" in str(e)


def test_full_staged_set_roundtrips_at_one_tag():
    # The whole per-spike set moves to one stage, each file on its own variant.
    with tempfile.TemporaryDirectory() as d:
        b = _base(d)
        n, nsamp, nchan, nfeat = 5, 4, 2, 3
        tag = "decollided"
        nio.write_res(b, 6, np.arange(n, dtype=np.int64), variant="stderiv", tag=tag)
        nio.write_clu(b, 6, np.full(n, 2, np.int64), variant="stderiv_C5_D34", tag=tag)
        nio.write_fet(b, 6, np.zeros((n, nfeat), np.int64), variant="stderiv_C5_D34", tag=tag)
        nio.write_spk(b, 6, np.zeros((n, nsamp, nchan), nio.SPK_DTYPE),
                      variant="standard", tag=tag)
        res = nio.read_res_at(b, 6, variant="stderiv", tag=tag)
        _, ids = nio.read_clu_at(b, 6, variant="stderiv_C5_D34", tag=tag)
        fet = nio.read_fet_at(b, 6, variant="stderiv_C5_D34", tag=tag)
        spk = nio.open_spk_at(b, 6, nsamp, nchan, variant="standard", tag=tag)
        # every per-spike file reports the same N -> the stage is internally consistent
        assert res.size == ids.size == fet.n_spikes == spk.shape[0] == n


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print(f"\n{len(fns)} passed")
