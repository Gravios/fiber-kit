#!/usr/bin/env python3
"""ndm_fiber-kit -- fiber-kit as an ndmanager plugin (ndm plugin protocol v1).

Run:        ndm_fiber-kit <session>.yaml [<group>] [stages...]
            With <group>: run that group.  Without one (the ndmanager convention -- programs are invoked
            with just the session parameter file): run every spikeDetection.channelGroup in the session.
            Translates to fiber-pipeline (FK_DIR/FK_SESS + group); each stage reads its knobs from the
            session yaml programs[ndm_fiber-kit] entry.
Protocol:   ndm_fiber-kit --ndm-describe [--format xml]   NDManager program: schema (YAML; the source of truth)
            ndm_fiber-kit --ndm-programs                   the session.yaml `programs:` entry to paste in
            ndm_fiber-kit --ndm-version                    protocol + plugin version
"""
import os
import subprocess
import sys

try:
    from .config import FiberKitPlugin
    from . import session_yaml
except ImportError:
    from config import FiberKitPlugin
    import session_yaml

PROTOCOL = "1"


def _version():
    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version("fiber-kit")
        except PackageNotFoundError:
            return "0+unknown"
    except Exception:
        return "0+unknown"


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--ndm-describe":
        fmt = "yaml"
        if "--format" in argv:
            i = argv.index("--format")
            fmt = argv[i + 1] if i + 1 < len(argv) else "yaml"
        if fmt != "yaml":
            sys.stderr.write("ndm_fiber-kit: only --format yaml is supported\n")
            return 2
        sys.stdout.write(FiberKitPlugin.describe())
        return 0
    if argv and argv[0] == "--ndm-programs":
        sys.stdout.write(FiberKitPlugin.programs_entry())
        return 0
    if argv and argv[0] == "--ndm-version":
        sys.stdout.write("ndm-plugin-protocol %s\nfiber-kit %s\n" % (PROTOCOL, _version()))
        return 0
    if not argv or argv[0] in ("-h", "--help"):
        sys.stderr.write(__doc__)
        return 0 if argv[:1] in (["-h"], ["--help"]) else 2

    session = argv[0]
    rest = argv[1:]
    p = os.path.abspath(session)
    d = os.path.dirname(p) or "."
    base = os.path.basename(p)
    for ext in (".yaml", ".yml"):
        if base.endswith(ext):
            base = base[:-len(ext)]
            break
    env = dict(os.environ, FK_DIR=d, FK_SESS=base)

    # explicit group (leading numeric arg): hand straight to fiber-pipeline (exec, single process).
    if rest and rest[0].isdigit():
        try:
            os.execvpe("fiber-pipeline", ["fiber-pipeline", *rest], env)
        except FileNotFoundError:
            sys.stderr.write("ndm_fiber-kit: 'fiber-pipeline' not found on PATH (is fiber-kit installed?)\n")
            return 127

    # ndmanager convention: just the session file (no group) -> run every spike group.  `rest`
    # (an optional stage list) is forwarded to each group.
    groups = session_yaml.spike_groups(session)
    if not groups:
        sys.stderr.write("ndm_fiber-kit: no spikeDetection.channelGroups in %s; nothing to run "
                         "(give an explicit group: ndm_fiber-kit <session>.yaml <group>)\n"
                         % os.path.basename(session))
        return 2
    rc = 0
    failed = []
    for g in groups:
        sys.stderr.write("== ndm_fiber-kit: group %d/%d ==\n" % (g, len(groups)))
        try:
            r = subprocess.call(["fiber-pipeline", str(g), *rest], env=env)
        except FileNotFoundError:
            sys.stderr.write("ndm_fiber-kit: 'fiber-pipeline' not found on PATH (is fiber-kit installed?)\n")
            return 127
        if r != 0:
            failed.append((g, r)); rc = r
    if failed:
        sys.stderr.write("ndm_fiber-kit: %d of %d group(s) failed: %s\n"
                         % (len(failed), len(groups), ", ".join("g%d(exit %d)" % (g, r) for g, r in failed)))
    return rc


if __name__ == "__main__":
    sys.exit(main())
