#!/usr/bin/env python3
"""The shipped Klusters plugin descriptors against the Klusters side of the contract
(neurosuite-3 src/klusters/docs/PLUGIN_API.md, PluginRegistry::parseDescriptor / buildArgv):

  * every src/fiber_kit/klusters/*.xml is WELL-FORMED XML (Klusters' QXmlStreamReader refuses a file that
    is not -- a `--` inside a comment was enough to make both descriptors vanish from the Plugins menu);
  * the schema Klusters reads: a program <n> before <parameters>, every parameter with <n> and <value>,
    a declared <type> among bool|int|real|path|choice|text, a `choice` with <choices>, <min> <= default
    <= <max> for int/real; the <klusters> block with kind recluster, input subsession, a <command> that
    places %params;
  * every FK_* parameter of the manifold descriptor is a knob the stage reads (fiber_manifold._KNOBS) and
    the choices of FK_MANIFOLD_CHANNELS are the stage's;
  * the argv Klusters assembles (its paramArgs rule: "--name value", %tokens from the context, the
    installed profile as `config`), run as `fk run ... --dry-run` after `fk install klusters-plugin
    --dir`, is accepted: exit 0, the stage in the plan, the FK_* parameters exported for the run.
"""
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from fiber_kit import fiber_manifold as fm   # noqa: E402

XML_DIR = os.path.join(ROOT, "src", "fiber_kit", "klusters")
TYPES = {"bool", "int", "real", "path", "choice", "text"}
fails = 0


def check(ok, what, out=""):
    global fails
    print(("  ok:   " if ok else "  FAIL: ") + what)
    if not ok and out:
        print(out[-3000:])
    fails += not ok


def klusters_argv(prog, params, ctx):
    """PluginRegistry::buildArgv for a <command> template, with paramArgs' rules."""
    pargs = []
    for p in prog.find("parameters").findall("parameter"):
        name, typ = p.findtext("n"), (p.findtext("type") or "").strip().lower()
        v = params.get(name, (p.findtext("value") or "")).strip()
        for k, val in ctx.items():
            v = v.replace("%" + k, val)
        if typ == "bool":
            if v.lower() in ("true", "on", "yes", "1"):
                pargs.append("--" + name)
            continue
        if not v and (p.findtext("status") or "").strip().lower() != "mandatory":
            continue
        pargs += ["--" + name] + v.split()
    argv = []
    for w in prog.find("klusters").findtext("command").split():
        if w == "%params":
            argv += pargs
        else:
            for k, val in ctx.items():
                w = w.replace("%" + k, val)
            argv.append(w)
    return argv


def main():
    files = sorted(f for f in os.listdir(XML_DIR) if f.endswith(".xml"))
    check(len(files) >= 2, f"shipped descriptors: {files}")
    progs = {}
    for f in files:
        path = os.path.join(XML_DIR, f)
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as e:
            check(False, f"{f} is well-formed XML: {e}"); continue
        check(root.tag == "program" and root.findtext("n"), f"{f}: well-formed, a <program> with its <n> ({root.findtext('n')})")
        progs[f] = root
        ok = True
        for p in root.find("parameters").findall("parameter"):
            n, typ = p.findtext("n"), (p.findtext("type") or "").strip().lower()
            if not n or p.find("value") is None:
                ok = False; print(f"    {f}: a parameter without <n>/<value>")
            if typ and typ not in TYPES:
                ok = False; print(f"    {f}: {n}: unknown type {typ!r}")
            if typ == "choice" and not (p.findtext("choices") or "").strip():
                ok = False; print(f"    {f}: {n}: a choice without <choices>")
            if typ in ("int", "real"):
                v = float(p.findtext("value"))
                lo, hi = p.findtext("min"), p.findtext("max")
                if (lo and v < float(lo)) or (hi and v > float(hi)):
                    ok = False; print(f"    {f}: {n}: default {v} outside [{lo}, {hi}]")
        check(ok, f"{f}: every parameter has <n>/<value>, a known <type>, choices for a choice, the default within min/max")
        k = root.find("klusters")
        check(k is not None and k.findtext("kind") == "recluster" and k.findtext("input") == "subsession"
              and "%params" in (k.findtext("command") or "") and "%final" in k.findtext("command"),
              f"{f}: a recluster plugin on the sub-session whose <command> places %params and %final")
    man = progs.get("fiber-kit-manifold.xml")
    if man is not None:
        names = [p.findtext("n") for p in man.find("parameters").findall("parameter")]
        fk_names = [n for n in names if n.startswith("FK_")]
        check(all(n in fm._KNOBS for n in fk_names), f"every FK_* parameter is a knob the stage reads ({[n for n in fk_names if n not in fm._KNOBS]})")
        ch = next(p for p in man.find("parameters").findall("parameter") if p.findtext("n") == "FK_MANIFOLD_CHANNELS")
        check(ch.findtext("type") == "choice" and ch.findtext("choices") == "off|full" and ch.findtext("value") == "off",
              "FK_MANIFOLD_CHANNELS is a choice off|full, default off (a combo box in Preferences > Plugins and the run dialog)")
        check("config" in names and "method" in names, "the descriptor carries config and method")
        check(man.find("klusters").findtext("integration") == "refine",
              "the manifold plugin integrates as `refine` (children of the same parents; Klusters 0948)")

    # the argv Klusters would run, after fk install, as a dry run
    env = {k: v for k, v in os.environ.items() if not k.startswith("FK_")}
    env["FK_CONFIG"] = os.devnull
    env["PYTHONPATH"] = os.pathsep.join([os.path.join(ROOT, "src"), env.get("PYTHONPATH", "")])
    env["PATH"] = os.path.join(ROOT, "scripts") + os.pathsep + env.get("PATH", "")
    fk = [sys.executable, "-c", "from fiber_kit.plan_steps import main; import sys; sys.exit(main(sys.argv[1:]))"]
    with tempfile.TemporaryDirectory() as td:
        plug = os.path.join(td, "plug")
        r = subprocess.run(fk + ["install", "klusters-plugin", "--dir", plug], cwd=td, env=env, capture_output=True, text=True)
        check(r.returncode == 0, "fk install klusters-plugin --dir", r.stdout + r.stderr)
        sess = os.path.join(td, "rat"); os.makedirs(sess)
        ctx = {"group": "6", "base": os.path.join(sess, "rat--5--t"), "final": os.path.join(sess, "rat--5--t.clu.klusters.6"),
               "variant": "stderiv", "tag": "cur", "dir": sess}
        for f in files:
            root = ET.parse(os.path.join(plug, f)).getroot()          # the INSTALLED copy: config points at the profile
            params = {}
            if f == "fiber-kit-manifold.xml":
                params = {"FK_MANIFOLD_CHANNELS": "full", "FK_MANIFOLD_WINDOW_MIN": "30"}   # what the run dialog holds
            argv = klusters_argv(root, params, ctx)
            check(argv[:2] == ["fk", "run"], f"{f}: the command starts with fk run ({' '.join(argv[:6])})")
            r = subprocess.run(fk + argv[1:] + ["--dry-run"], cwd=sess, env=env, capture_output=True, text=True, timeout=300)
            o = r.stdout + r.stderr
            check(r.returncode == 0 and "[plan 1/" in r.stdout, f"{f}: Klusters' argv is accepted by fk run (dry run)", o)
            if f == "fiber-kit-manifold.xml":
                check("fiber-manifold" in r.stdout and "FK_MANIFOLD_CHANNELS=full" in r.stderr and "FK_MANIFOLD_WINDOW_MIN=30" in r.stderr,
                      "... the manifold stage runs with the dialog's FK_MANIFOLD_CHANNELS=full and window exported", o)
    print(f"test_klusters_contract: {'PASS' if not fails else 'FAIL'} ({fails} failed)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
