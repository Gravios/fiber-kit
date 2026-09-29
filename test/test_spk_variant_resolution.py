"""Working-.spk resolution honours the caller's method token (open_spkD prefer=...).

The regression this pins down: a session directory holding TWO stderiv-family .spk files
-- the extractor's .spk.stderiv_C5.<g> next to a hand-materialised feature-space-alias
copy .spk.stderiv_C5_D34.<g> (written so Klusters can display waveforms under the alias
name) -- made every bare open_spkD raise resolve_input's family-ambiguity ValueError,
because the stages threw away the very --method / --clu-method token the runner passed
them and fell back to the bare 'stderiv' family walk.  open_spkD now takes prefer=
(the caller's token, tried first) with $FK_SPK_VARIANT as the no-flag fallback, and the
historical prefer_derived() walk after it.
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import fiber_kit.neuro_io as nio  # noqa: E402

NSAMP, NCH, NSPK = 4, 2, 3


def _write_spk(path, fill):
    np.full(NSPK * NSAMP * NCH, fill, dtype=np.int16).tofile(path)


def _open(base, prefer=None):
    return nio.open_spkD(base, 1, NSAMP, NCH, prefer=prefer)


def main():
    ok = 0

    def check(name, cond):
        nonlocal ok
        assert cond, name
        ok += 1
        print(f"  ok  {name}")

    env_had = os.environ.pop("FK_SPK_VARIANT", None)
    try:
        with tempfile.TemporaryDirectory() as td:
            base = os.path.join(td, "sess")
            _write_spk(f"{base}.spk.stderiv_C5.1", 5)
            _write_spk(f"{base}.spk.stderiv_C5_D34.1", 34)

            # 1. bare open in a two-variant dir keeps the explicit ambiguity error + guidance
            try:
                _open(base)
                raised = False
            except ValueError as e:
                raised = "several 'stderiv' variants" in str(e)
            check("bare open in ambiguous dir raises the family-ambiguity ValueError", raised)

            # 2/3. an exact caller token resolves that file, both variants
            mm, path = _open(base, prefer="stderiv_C5")
            check("prefer='stderiv_C5' opens .spk.stderiv_C5",
                  path.endswith(".spk.stderiv_C5.1") and int(mm[0, 0, 0]) == 5)
            mm, path = _open(base, prefer="stderiv_C5_D34")
            check("prefer='stderiv_C5_D34' opens the alias .spk",
                  path.endswith(".spk.stderiv_C5_D34.1") and int(mm[0, 0, 0]) == 34)

            # 4. $FK_SPK_VARIANT (the runner's exported settled token) fills an empty prefer
            os.environ["FK_SPK_VARIANT"] = "stderiv_C5_D34"
            mm, path = _open(base)
            check("FK_SPK_VARIANT fallback resolves the alias when the caller passed nothing",
                  path.endswith(".spk.stderiv_C5_D34.1"))

            # 5. an explicit caller token beats the env fallback
            mm, path = _open(base, prefer="stderiv_C5")
            check("caller token wins over FK_SPK_VARIANT",
                  path.endswith(".spk.stderiv_C5.1"))
            del os.environ["FK_SPK_VARIANT"]

            # 6. an exact token with no file degrades to the historical family walk,
            #    which in an ambiguous dir still raises the guidance error
            try:
                _open(base, prefer="stderiv_C9")
                raised = False
            except ValueError as e:
                raised = "several 'stderiv' variants" in str(e)
            check("absent exact token falls back to the family walk (still ambiguous here)", raised)

        with tempfile.TemporaryDirectory() as td:
            base = os.path.join(td, "sess")
            _write_spk(f"{base}.spk.stderiv_C5.1", 5)

            # 7. single-variant dir: bare open keeps working (historical behaviour)
            mm, path = _open(base)
            check("bare open in a single-variant dir resolves it",
                  path.endswith(".spk.stderiv_C5.1"))

            # 8. absent exact token in a single-variant dir degrades to that variant
            #    (the _D feature-space-alias case: alias token, no alias .spk on disk)
            mm, path = _open(base, prefer="stderiv_C5_D34")
            check("alias token with no alias .spk degrades to the family's one member",
                  path.endswith(".spk.stderiv_C5.1"))

        # 9/10. the chunk-worker path (the reported crash site): fiber_session's
        #    _init_chunk_worker used open_spk(prefer=prefer_derived()) directly, so it
        #    family-walked PAST main()'s already-token-resolved open and raised in the
        #    two-variant dir.  It now threads cfg['method'].
        import fiber_kit.fiber_session as fs
        with tempfile.TemporaryDirectory() as td:
            base = os.path.join(td, "sess")
            _write_spk(f"{base}.spk.stderiv_C5.1", 5)
            _write_spk(f"{base}.spk.stderiv_C5_D34.1", 34)
            import numpy as _np
            _np.zeros(100 * 2, dtype=_np.int16).tofile(f"{base}.fil")
            cfg = dict(base=base, elec=1, nsamp=NSAMP, nchan=NCH,
                       fil=f"{base}.fil", ntotal=2, cf={}, method="stderiv_C5_D34")
            fs._init_chunk_worker(cfg)
            check("chunk worker opens the exact-token spk from cfg['method']",
                  int(fs._CTX["spk"][0, 0, 0]) == 34)
            try:
                fs._init_chunk_worker(dict(cfg, method=None))
                raised = False
            except ValueError as e:
                raised = "several 'stderiv' variants" in str(e)
            check("chunk worker without a token keeps the guidance error", raised)
    finally:
        if env_had is not None:
            os.environ["FK_SPK_VARIANT"] = env_had
        else:
            os.environ.pop("FK_SPK_VARIANT", None)

    print(f"test_spk_variant_resolution: {ok}/10 checks passed")


if __name__ == "__main__":
    main()
