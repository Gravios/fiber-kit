"""README.md does not drift from the package.

  * the version badge names the version pyproject.toml / fiber_kit.__version__ carry (it said 0.21.0
    at 0.54.0);
  * every console script of pyproject.toml [project.scripts] is in the README's command table (41 of
    51 were missing at 0.54.0).
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(*p):
    with open(os.path.join(ROOT, *p), encoding="utf-8") as fh:
        return fh.read()


def main():
    bad = 0
    readme = read("README.md")
    pp = read("pyproject.toml")
    ver = re.search(r'^version\s*=\s*"([^"]+)"', pp, re.M).group(1)
    badge = re.search(r"badge/version-([0-9.]+)-", readme)
    ok = badge is not None and badge.group(1) == ver
    print(f"  {'ok  ' if ok else 'FAIL'} version badge {badge.group(1) if badge else None} == {ver}")
    bad += not ok
    block = pp.split("[project.scripts]", 1)[1].split("\n[", 1)[0]
    scripts = re.findall(r"^([A-Za-z0-9_-]+)\s*=", block, re.M)
    missing = [s for s in scripts if f"| `{s}` |" not in readme]
    ok = not missing
    print(f"  {'ok  ' if ok else 'FAIL'} all {len(scripts)} console scripts are in the README command table"
          + (f" (missing: {', '.join(missing)})" if missing else ""))
    bad += not ok
    print(f"test_readme: {'PASS' if not bad else 'FAIL'} ({bad} failed)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
