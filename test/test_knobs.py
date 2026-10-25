"""FK_* knobs: one precedence, YAML 1.2 booleans, validated values.

  * config.yaml_load keeps a bare off/on/yes/no as the word written (YAML 1.1 made them booleans,
    so `FK_SESSION_REALIGN: off` reached fiber-session as "0"/"False" and the realign RAN);
  * fiber-pipeline exports `off` from the yaml as "off" (seen in a dry run's fiber-session command);
  * a direct fiber-session call with `FK_SESSION_REALIGN: off` in $FK_CONFIG defaults --realign to off,
    and a legacy "0" in the environment is read as off too;
  * a value outside a knob's choices stops the stage with the knob's name (was: ran the wrong mode);
  * an FK_* environment variable wins over the yaml in every stage reader (was inverted in
    fiber-xcorr-merge, fiber-rdr, fiber-session's chunk/overlap/whiten knobs and fiber_chunks);
  * a value that is not a number is an error, not silently the default (fiber-consolidate) or a
    truncated integer (yaml 6.7 for an int knob);
  * a $FK_CONFIG that cannot be parsed is an error, not an empty config.
"""
import argparse
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from fiber_kit import config as cfgmod                  # noqa: E402
from fiber_kit import fiber_chunks as fch               # noqa: E402
from fiber_kit import fiber_consolidate as fcons        # noqa: E402
from fiber_kit import fiber_rdr as frdr                 # noqa: E402
from fiber_kit import fiber_session as fs               # noqa: E402
from fiber_kit import fiber_xcorr_merge as fxcm         # noqa: E402

PIPE = os.path.join(ROOT, "scripts", "fiber-pipeline")
fails = 0


def check(ok, what):
    global fails
    print(("  ok:   " if ok else "  FAIL: ") + what); fails += not ok


def raises(fn, needle=""):
    try:
        fn()
    except SystemExit as e:
        return needle in str(e.code)
    return False


class Env:
    """Set FK_* variables (and FK_CONFIG) for a block; restore afterwards."""

    def __init__(self, **kv):
        self.kv = kv

    def __enter__(self):
        self.old = {k: os.environ.get(k) for k in self.kv}
        for k, v in self.kv.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def __exit__(self, *exc):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def write(d, name, text):
    p = os.path.join(d, name)
    with open(p, "w") as fh:
        fh.write(text)
    return p


def realign_default():
    ap = argparse.ArgumentParser()
    fs.add_realign_arguments(ap)
    cfgmod.check_choice_defaults(ap)
    return ap.parse_args([]).realign


def main():
    for k in [k for k in os.environ if k.startswith("FK_")]:
        os.environ.pop(k)
    d = tempfile.mkdtemp(prefix="fk_knobs_")

    print("YAML 1.2 booleans")
    y = cfgmod.yaml_load("a: off\nb: on\nc: yes\nd: true\ne: False\nf: 0.5\ng: ''\n")
    check(y == {"a": "off", "b": "on", "c": "yes", "d": True, "e": False, "f": 0.5, "g": ""}, f"{y}")

    print("FK_SESSION_REALIGN: off")
    cfg_off = write(d, "off.yaml", "FK_SESSION_REALIGN: off\n")
    with Env(FK_CONFIG=cfg_off):
        check(realign_default() == "off", "a direct fiber-session call reads --realign off from $FK_CONFIG")
    with Env(FK_CONFIG=os.path.join(d, "none.yaml"), FK_SESSION_REALIGN="0"):
        check(realign_default() == "off", "a legacy FK_SESSION_REALIGN=0 in the environment means off")
    with Env(FK_CONFIG=os.path.join(d, "none.yaml"), FK_SESSION_REALIGN="sideways"):
        check(raises(realign_default, "FK_SESSION_REALIGN"), "a value outside the choices stops the stage")
    e = {k: v for k, v in os.environ.items() if not k.startswith("FK_")}
    e["PYTHONPATH"] = os.path.join(ROOT, "src")
    e["FK_CONFIG"] = cfg_off
    r = subprocess.run(["bash", PIPE, "5", "fiber-session", "--dry-run"], cwd=d, capture_output=True,
                       text=True, env=e, timeout=300)
    out = r.stdout + r.stderr
    check("--realign off" in out, "fiber-pipeline passes --realign off from the yaml"
          + ("" if "--realign off" in out else f"  [{[ln for ln in out.splitlines() if '--realign' in ln][:1]}]"))

    print("fiber-pipeline 0/1 switches accept on/off words")
    sw = write(d, "sw.yaml", "FK_SESSION_LINK: off\nFK_SESSION_CFIBER_GATE: yes\n")
    e["FK_CONFIG"] = sw
    r = subprocess.run(["bash", PIPE, "5", "fiber-session", "--dry-run"], cwd=d, capture_output=True,
                       text=True, env=e, timeout=300)
    out = r.stdout + r.stderr
    check(r.returncode == 0 and "--no-link" in out and "--cfiber-gate" in out,
          "FK_SESSION_LINK: off -> --no-link, FK_SESSION_CFIBER_GATE: yes -> --cfiber-gate")
    e["FK_CONFIG"] = write(d, "bad_sw.yaml", "FK_SESSION_LINK: maybe\n")
    r = subprocess.run(["bash", PIPE, "5", "fiber-session", "--dry-run"], cwd=d, capture_output=True,
                       text=True, env=e, timeout=300)
    check(r.returncode != 0 and "not a 0/1 switch" in r.stdout + r.stderr,
          "FK_SESSION_LINK: maybe stops the run")

    print("environment over yaml")
    cfg = write(d, "k.yaml", "FK_XCM_COS_THR: 0.96\nFK_RDR_THR: 2.0\nFK_SESSION_OVERLAP_MIN: 4\n"
                             "FK_SESSION_WHITEN_SOURCE: nst\nFK_CHUNKS: '1'\n")
    g = cfgmod.load_global_config(cfg)
    with Env(FK_XCM_COS_THR="0.995", FK_RDR_THR="7", FK_SESSION_OVERLAP_MIN="2",
             FK_SESSION_WHITEN_SOURCE="spk", FK_CHUNKS="3"):
        check(fxcm._knob_default("FK_XCM_COS_THR", float, 0.5, g) == 0.995, "fiber-xcorr-merge")
        check(frdr._knob_default("FK_RDR_THR", float, 0.5, g) == 7.0, "fiber-rdr")
        check(fch.config_value(g, "FK_CHUNKS") == "3", "fiber_chunks (FK_CHUNKS)")
        with Env(FK_CONFIG=cfg):
            ap = argparse.ArgumentParser()
            fs.add_core_arguments(ap)
            a = ap.parse_args([])
            check(a.overlap_min == 2.0, f"fiber-session --overlap-min ({a.overlap_min})")
            check(a.whiten_source == "spk", f"fiber-session --whiten-source ({a.whiten_source})")
    check(fxcm._knob_default("FK_XCM_COS_THR", float, 0.5, g) == 0.96, "and the yaml when the env is unset")

    print("bad values are errors")
    with Env(FK_CONS_MAX_DIST="0,4"):
        check(raises(lambda: fcons._knob_default("FK_CONS_MAX_DIST", float, 0.5, {}), "FK_CONS_MAX_DIST"),
              "FK_CONS_MAX_DIST=0,4 (was: silently 0.5)")
    check(raises(lambda: cfgmod.knob_value("FK_X_N", int, 3, {"FK_X_N": 6.7}), "FK_X_N"),
          "an int knob of 6.7 in the yaml (was: truncated to 6)")
    check(cfgmod.knob_value("FK_X_N", int, 3, {"FK_X_N": "6.0"}) == 6, "6.0 is still the integer 6")
    check(raises(lambda: cfgmod.coerce_knob("FK_INTRA_GATE", "Band", str,
                                            ("cosine", "mmd", "kcov", "cfiber", "band")), "FK_INTRA_GATE"),
          "FK_INTRA_GATE=Band (was: silently the cosine gate)")
    check(cfgmod.coerce_knob("FK_INTRA_RECUT", "off", str, ("", "all", "dip", "giqr")) == "",
          "FK_INTRA_RECUT: off means '' (was: the 'all' recut)")
    check(cfgmod.coerce_knob("FK_INTRA_RECUT", False, str, ("", "all", "dip", "giqr")) == "",
          "and so does a YAML false")
    bad = write(d, "bad.yaml", "FK_A: [unclosed\n")
    check(raises(lambda: cfgmod.load_global_config(bad), "cannot read the config"),
          "a $FK_CONFIG that does not parse is an error (was: {})")

    print(f"test_knobs: {'PASS' if not fails else 'FAIL'} ({fails} failed)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
