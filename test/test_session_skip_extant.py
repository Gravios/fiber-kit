"""fiber-session --skip-extant (FK_SESSION_SKIP_EXTANT): resume past an extant sort.

With the knob on, fiber-session checks the EXACT destination .clu this invocation
would write (--out, else <base>.clu.<out-variant or method>.<elec>[.<clu-stage>]) and,
when it exists, exits before any compute -- so a plan re-run resumes past its
fiber-session step instead of regenerating hours of sort over the top of an extant
(possibly hand-curated) one.  These checks drive main() with the session resolve and
the first compute step monkeypatched, so 'entered the compute path' is observable as
a sentinel exception and 'skipped' as a clean return.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import fiber_kit.fiber_session as fs  # noqa: E402


class _Compute(Exception):
    """Raised by the patched read_res: main() got past the guard into real work."""


def _run(base, argv):
    def _fake_resolve(session, group, **kw):
        return dict(base=base, group=group, ntotal=1000, nchan=2, nsamp=32,
                    sr=20000.0, channels=[0, 1], peak=16)

    def _boom(*a, **kw):
        raise _Compute

    old_resolve, old_read = fs.sy.resolve_session_params, fs.read_res
    old_argv = sys.argv
    fs.sy.resolve_session_params, fs.read_res = _fake_resolve, _boom
    sys.argv = ["fiber-session", "sess", "1"] + argv
    try:
        fs.main()
        return "skipped"
    except _Compute:
        return "computed"
    finally:
        fs.sy.resolve_session_params, fs.read_res = old_resolve, old_read
        sys.argv = old_argv


def main():
    ok = 0

    def check(name, cond):
        nonlocal ok
        assert cond, name
        ok += 1
        print(f"  ok  {name}")

    env_had = os.environ.pop("FK_SESSION_SKIP_EXTANT", None)
    try:
        with tempfile.TemporaryDirectory() as td:
            base = os.path.join(td, "sess")
            dst = f"{base}.clu.stderiv_C5_D34.1.fiber_session"

            check("no destination, knob on -> computes",
                  _run(base, ["--method", "stderiv_C5_D34", "--skip-extant", "1"]) == "computed")

            with open(dst, "w") as fh:
                fh.write("1\n")
            check("destination extant, knob on -> skips before any compute",
                  _run(base, ["--method", "stderiv_C5_D34", "--skip-extant", "1"]) == "skipped")

            check("destination extant, knob off (default) -> rebuilds as before",
                  _run(base, ["--method", "stderiv_C5_D34"]) == "computed")

            os.environ["FK_SESSION_SKIP_EXTANT"] = "1"
            check("FK_SESSION_SKIP_EXTANT=1 arms the guard without the flag",
                  _run(base, ["--method", "stderiv_C5_D34"]) == "skipped")
            del os.environ["FK_SESSION_SKIP_EXTANT"]

            # the check follows the ALIAS naming when --out-variant is set: the alias clu
            # (the generated product a curator works on) is the file that must gate
            check("--out-variant names the checked destination",
                  _run(base, ["--method", "stderiv_C5", "--out-variant", "stderiv_C5_D34",
                              "--skip-extant", "1"]) == "skipped")
            check("--out-variant absent on disk -> computes",
                  _run(base, ["--method", "stderiv_C5", "--out-variant", "stderiv_C5_D9",
                              "--skip-extant", "1"]) == "computed")

            # --clu-stage changes the destination, so a different stage tag rebuilds
            check("different --clu-stage -> different destination -> computes",
                  _run(base, ["--method", "stderiv_C5_D34", "--skip-extant", "1",
                              "--clu-stage", "resort"]) == "computed")

            # --out overrides the composed path entirely
            outp = os.path.join(td, "custom.clu")
            with open(outp, "w") as fh:
                fh.write("1\n")
            check("--out extant -> skips on that exact path",
                  _run(base, ["--method", "stderiv_C5_D34", "--skip-extant", "1",
                              "--out", outp]) == "skipped")
    finally:
        if env_had is not None:
            os.environ["FK_SESSION_SKIP_EXTANT"] = env_had

    print(f"test_session_skip_extant: {ok}/8 checks passed")


if __name__ == "__main__":
    main()
