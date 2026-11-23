#!/usr/bin/env python3
"""fiber-pipeline reads the session's method token off its .spk files; a super-sampled copy
(<method>_U<f>, written by Klusters) beside the base-rate file is the same method and must not
make the token ambiguous.

Before 0.71.5 a group holding .spk.stderiv_C5.<g> AND .spk.stderiv_C5_U4.<g> gave "cannot pin ONE
method token ... naming falls back to 'stderiv'", and fiber-session then stopped on "several
'stderiv' variants exist" (seen on sirotaA-jg-000005-20120312 group 8).

  - base-rate + _U4: the token is the base-rate one, and fiber-session is handed it;
  - NEGATIVE CONTROL: two different base-rate methods (stderiv_C5, stderiv_C6) still cannot be
    pinned, and the launcher still says so;
  - a _U copy alone names no method (nothing to sort on at base rate).
"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
fails = 0


def check(ok, what, extra=""):
    global fails
    print(("  ok:   " if ok else "  FAIL: ") + what + (("\n        " + extra) if (not ok and extra) else ""))
    fails += not ok


def dry(tokens):
    with tempfile.TemporaryDirectory() as d:
        sess = os.path.join(d, "s")
        os.mkdir(sess)
        for t in tokens:
            open(os.path.join(sess, f"s.spk.{t}.8"), "wb").close()
        plan = os.path.join(d, "p.yaml")
        with open(plan, "w") as fh:
            fh.write("pipeline:\n  - {stage: fiber-session, name: session, out: x}\n")
        env = dict(os.environ, FK_DIR=sess, FK_SESS="s", PYTHONPATH=os.path.join(ROOT, "src"))
        for k in ("FK_REALIGN_VARIANT", "FK_CONFIG", "FK_ELEC"):
            env.pop(k, None)
        r = subprocess.run(["bash", os.path.join(ROOT, "scripts", "fiber-pipeline"), "8", "--plan", plan, "--dry-run"],
                           cwd=sess, env=env, capture_output=True, text=True)
        return r.stdout + r.stderr


def main():
    out = dry(["standard", "stderiv_C5", "stderiv_C5_U4"])
    check("method token 'stderiv_C5' read off" in out and "cannot pin" not in out,
          "base-rate file + its _U4 copy: the token is stderiv_C5", out[-600:])
    check("--method stderiv_C5 " in out + " " and "s.clu.stderiv_C5.8.x" in out,
          "... and fiber-session is handed it, with the .clu named after it", out[-600:])
    out = dry(["standard", "stderiv_C5", "stderiv_C6"])
    check("cannot pin ONE method token" in out, "NEGATIVE CONTROL: two base-rate methods are still not pinned", out[-400:])
    out = dry(["standard", "stderiv_C5_U4"])
    check("method token 'stderiv_C5_U4'" not in out, "a _U copy alone does not name the method", out[-400:])
    print(f"test_pin_method_supersampled: {'PASS' if not fails else 'FAIL'} ({fails} failed)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
