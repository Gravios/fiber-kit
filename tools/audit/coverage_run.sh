#!/usr/bin/env bash
# Line coverage of src/fiber_kit by the whole test suite.  The tests are scripts that start further
# processes, so coverage is started in every python process through a .pth file, which this script
# writes into site-packages and removes again.  Needs: pip install coverage.
#   bash tools/audit/coverage_run.sh [outdir]        (about half an hour on two cores)
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; out="${1:-$root/.coverage-audit}"
mkdir -p "$out"; rm -f "$out"/.coverage*
cat > "$out/covrc" <<RC
[run]
source = $root/src/fiber_kit
parallel = True
data_file = $out/.coverage
concurrency = multiprocessing
RC
pth="$(python3 -c 'import site; print(site.getsitepackages()[0])')/zz_fiber_kit_cov.pth"
echo "import coverage; coverage.process_startup()" > "$pth"
trap 'rm -f "$pth"' EXIT
(cd "$root" && COVERAGE_PROCESS_START="$out/covrc" python3 test/run_all.py > "$out/run_all.log" 2>&1) || true
rm -f "$pth"
(cd "$out" && python3 -m coverage combine --rcfile=covrc >/dev/null && python3 -m coverage report --rcfile=covrc --sort=cover | sed "s#$root/src/fiber_kit/##" > report.txt)
tail -1 "$out/report.txt"; echo "report: $out/report.txt"
