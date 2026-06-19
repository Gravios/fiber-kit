#!/usr/bin/env python3
"""fiber-kit-init — drop a template ``fiber-kit.yaml`` into the current directory.

``fiber-kit.yaml`` is the flat ``FK_*`` tuning config that ``fiber-pipeline`` reads from the
working directory it is run in.  This command copies the packaged template there so you can edit
it instead of hand-writing the knobs (or exporting ``FK_*`` env vars).  The template ships every
knob at its default; ``fiber-pipeline`` ignores any key left empty and falls back to the built-in,
and an exported ``FK_*`` still overrides the file.
"""
import argparse
import sys
from pathlib import Path

try:
    from importlib.resources import files
except ImportError:                                   # py<3.9 fallback (requires-python is >=3.9)
    from importlib_resources import files             # type: ignore

TEMPLATE = "fiber-kit.yaml"


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="fiber-kit-init",
        description="Copy the template fiber-kit.yaml (fiber-pipeline tuning config) into the current directory.")
    ap.add_argument("-o", "--output", default=TEMPLATE,
                    help="destination path (default: ./fiber-kit.yaml)")
    ap.add_argument("-f", "--force", action="store_true",
                    help="overwrite the destination if it already exists")
    a = ap.parse_args(argv)

    dst = Path(a.output)
    if dst.exists() and not a.force:
        sys.stderr.write(f"fiber-kit-init: {dst} already exists (use --force to overwrite)\n")
        return 1
    try:
        text = (files("fiber_kit") / TEMPLATE).read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, OSError) as e:
        sys.stderr.write(f"fiber-kit-init: packaged template not found: {e}\n")
        return 2
    dst.write_text(text, encoding="utf-8")
    print(f"fiber-kit-init: wrote {dst} ({len(text.splitlines())} lines). "
          f"Edit it, then run fiber-pipeline from this directory.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
