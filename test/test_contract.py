#!/usr/bin/env python3
# test_contract.py — contract.yaml (repository root) states what the code does.
#
# Checks that contract, plugin_protocol, session_schema and every formats entry
# equal the constants in ndm_fiber_kit, session_yaml, neuro_io and fiber_pca;
# that each writer writes the version the contract names and each reader
# rejects the version after the one it claims; and that `ndm_fiber-kit
# --ndm-version` prints the protocol line first and the same format lines.
# The vectors block is test_shared_contract.py's.  neurosuite-3's
# core_test_contract is the mirror on the C++ side.
import os
import re
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from fiber_kit import neuro_io as nio          # noqa: E402
from fiber_kit import fiber_pca                 # noqa: E402
from fiber_kit import session_yaml              # noqa: E402
from fiber_kit import ndm_fiber_kit as plugin   # noqa: E402

fails = 0
ran = 0


def check(ok, what):
    global fails, ran
    ran += 1
    print(("  ok:   " if ok else "  FAIL: ") + what)
    if not ok:
        fails += 1


def read_contract(path):
    """The flat layout contract.yaml documents: top-level scalars and
    formats {name: (read, write)}."""
    out = {"formats": {}}
    section = None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            m = re.match(r"^([A-Za-z_]+):\s*(.*?)\s*$", line)
            if m:
                section = m.group(1)
                if m.group(2):
                    out[section] = m.group(2).strip("'\"")
                continue
            m = re.match(r"^\s+(\w+):\s*\{\s*read:\s*(\d+)\s*,\s*write:\s*(\d+)\s*\}\s*$", line)
            if section == "formats" and m:
                out["formats"][m.group(1)] = (int(m.group(2)), int(m.group(3)))
    return out


def first_line(path):
    with open(path) as fh:
        return fh.readline().rstrip("\n")


def read_bytes(path):
    with open(path, "rb") as fh:
        return bytearray(fh.read())


def write_bytes(path, data):
    with open(path, "wb") as fh:
        fh.write(bytes(data))


def main():
    c = read_contract(os.path.join(ROOT, "contract.yaml"))

    print("contract.yaml against the code:")
    check(c.get("contract") == str(plugin.CONTRACT), f"contract == ndm_fiber_kit.CONTRACT ({plugin.CONTRACT})")
    check(c.get("plugin_protocol") == plugin.PROTOCOL,
          f"plugin_protocol == ndm_fiber_kit.PROTOCOL ({plugin.PROTOCOL})")
    check(c.get("session_schema") == session_yaml.SESSION_SCHEMA,
          f"session_schema == session_yaml.SESSION_SCHEMA ({session_yaml.SESSION_SCHEMA})")
    code = {name: (r, w) for name, r, w in plugin.formats()}
    check(c["formats"] == code, f"formats {c['formats']} == the code's {code}")

    print("\nThe writers write, and the readers stop at, those versions:")
    with tempfile.TemporaryDirectory() as d:
        base = os.path.join(d, "s")

        r, w = nio.WTI_VERSION
        nio.write_wti(base, 1, [{"row": 0, "unit": 1, "link": "drift", "bin": 0, "a": 0, "b": 1,
                                 "nSpikes": 1},
                                {"row": 1, "unit": 1, "link": "drift", "bin": 1, "a": 1, "b": 2,
                                 "nSpikes": 1, "parent": 0}], 4, 1)
        p = nio.session_path(base, "wti", 1)
        check(first_line(p) == f"wti {w}", f"write_wti writes wti {w} when a row has a parent")
        check(nio.read_wti(base, 1)["version"] == w, "read_wti reads it")
        with open(p, "w") as fh:
            fh.write(f"wti {r + 1}\nnSamples 4\nnChannels 1\n")
        check(nio.read_wti(base, 1)["version"] == 0, f"read_wti rejects wti {r + 1}")

        r, w = nio.WTL_VERSION
        nio.write_wtl(base, 1, [{"node": 0, "class": 0, "kind": "drift-root", "parent": -1,
                                 "a": 0.0, "b": 1.0, "spikes": [0, 1]}])
        p = nio.session_path(base, "wtl", 1)
        check(first_line(p) == f"wtl {w}", f"write_wtl writes wtl {w}")
        check(nio.read_wtl(base, 1)["version"] == w, "read_wtl reads it")
        with open(p, "w") as fh:
            fh.write(f"wtl {r + 1}\nnNodes 0\n")
        check(nio.read_wtl(base, 1)["version"] == 0, f"read_wtl rejects wtl {r + 1}")

        r, w = nio.TCL_VERSION
        nio.write_tcl(base, 1, [{"col": 0, "status": "free"}], 1)
        p = nio.session_path(base, "tcl", 1)
        check(first_line(p) == f"tcl {w}", f"write_tcl writes tcl {w}")
        check(nio.read_tcl(base, 1)["ok"], "read_tcl reads it")
        with open(p, "w") as fh:
            fh.write(f"tcl {r + 1}\nnClasses 0\n")
        check(not nio.read_tcl(base, 1)["ok"], f"read_tcl rejects tcl {r + 1}")

        r, w = nio.EAP_VERSION
        nio.write_eap(base, 1, np.full((2, 2), nio.EAP_ABSENT, np.int8))
        p = nio.session_path(base, "eap", 1)
        raw = read_bytes(p)
        check(raw[:3] == b"EAP" and raw[3] == w, f"write_eap writes version byte {w}")
        check(nio.read_eap(base, 1)["ok"], "read_eap reads it")
        raw[3] = r + 1
        write_bytes(p, raw)
        check(not nio.read_eap(base, 1)["ok"], f"read_eap rejects version byte {r + 1}")

        v = fiber_pca.PCAE_VERSION
        p = os.path.join(d, "s.pca.1")
        fiber_pca.write_pcad(p, np.zeros((1, 2)), np.ones((1, 1, 2)), 0)
        raw = read_bytes(p)
        check(int.from_bytes(raw[4:8], "little") == v, f"write_pcad writes PCAE version {v}")
        check(fiber_pca.read_pcad(p)["nCh"] == 1, "read_pcad reads it")
        raw[4:8] = (v + 1).to_bytes(4, "little")
        write_bytes(p, raw)
        try:
            fiber_pca.read_pcad(p)
            rejected = False
        except ValueError:
            rejected = True
        check(rejected, f"read_pcad rejects PCAE version {v + 1}")

    print("\n--ndm-version:")
    lines = plugin.ndm_version().splitlines()
    check(lines[:1] == [f"ndm-plugin-protocol {plugin.PROTOCOL}"], "the protocol line comes first")
    fmt = {}
    for ln in lines[1:]:
        parts = ln.split()
        if parts[:1] == ["format"] and len(parts) == 4:
            fmt[parts[1]] = (int(parts[2]), int(parts[3]))
    check(fmt == c["formats"], f"its format lines {fmt} == contract.yaml formats")

    print(f"\ntest_contract: {ran} checks, {fails} failed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
