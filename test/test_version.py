#!/usr/bin/env python3
"""test_version.py — fiber_kit.__version__ and pyproject.toml agree.

The version is written in two places: pyproject.toml (what pip installs) and
fiber_kit/__init__.py (what `fiber_kit.__version__` reports).  They had drifted
apart (0.40.6 vs 0.25.0).  The newest CHANGELOG.md heading must name the same
version, so a release is one bump in three places, checked here.
Exits non-zero on a mismatch.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def grab(path, pattern):
    with open(os.path.join(ROOT, path), encoding="utf-8") as fh:
        m = re.search(pattern, fh.read(), re.M)
    return m.group(1) if m else None


def main():
    found = {
        "pyproject.toml": grab("pyproject.toml", r'^version\s*=\s*"([^"]+)"'),
        "src/fiber_kit/__init__.py": grab(os.path.join("src", "fiber_kit", "__init__.py"),
                                          r'^__version__\s*=\s*"([^"]+)"'),
        "CHANGELOG.md (newest entry)": grab("CHANGELOG.md", r'^## \[([^\]]+)\]'),
    }
    for where, v in found.items():
        print(f"  {v or '(not found)':10s} {where}")
    ok = None not in found.values() and len(set(found.values())) == 1
    print(("PASS" if ok else "FAIL") + ": the version is " + ("consistent" if ok else "not consistent"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
