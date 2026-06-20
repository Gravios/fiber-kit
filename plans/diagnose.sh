#!/usr/bin/env bash
# ============================================================================
#  diagnose.sh -- compare the diagnostic pipeline branches written by the plans
#  in this directory.  Reports, per branch: cluster counts (refine + linked),
#  contamination flags, a QC summary, and -- if a ground-truth .clu is given --
#  fiber-score ARI / pairwise precision+recall / split / merge.
#
#  usage:
#     plans/diagnose.sh <session-base-path> <elec> [--gt <clu> --gt-res <res>]
#  e.g.
#     plans/diagnose.sh /data/sirotaA-jg-000005-20120312 5 \
#         --gt /data/curated_head.clu --gt-res /data/curated_head.res
# ============================================================================
set -uo pipefail

BASE="${1:?session base path required, e.g. /data/sirotaA-jg-000005-20120312}"
ELEC="${2:?electrode group required}"
shift 2
GT=""; GTRES=""
while [ $# -gt 0 ]; do
    case "$1" in
        --gt)     GT="$2"; shift 2 ;;
        --gt-res) GTRES="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

PY="${FK_PYTHON:-python3}"
RES="$BASE.res.stderiv.$ELEC"
NS="bl mg sp c6 c24 lc"

nclu() {   # count real clusters (id > 1) in a .clu file; prints '-' if absent
    local f="$1"
    [ -f "$f" ] || { echo "-"; return; }
    "$PY" - "$f" <<'PY'
import sys, numpy as np
raw = np.fromfile(sys.argv[1], dtype="<i4")
print(int((np.unique(raw[1:]) > 1).sum()) if raw.size > 1 else 0)
PY
}

echo "================ cluster counts per branch ================"
printf "%-6s %-10s %-10s\n" "ns" "refine" "linked"
for ns in $NS; do
    printf "%-6s %-10s %-10s\n" "$ns" \
        "$(nclu "$BASE.clu.stderiv.$ELEC.${ns}_refine")" \
        "$(nclu "$BASE.clu.stderiv.$ELEC.${ns}_linked")"
done

if [ -n "$GT" ]; then
    echo ""
    echo "================ ground-truth score (end product '<ns>_linked' vs GT) ================"
    for ns in $NS; do
        cand="$BASE.clu.stderiv.$ELEC.${ns}_linked"
        [ -f "$cand" ] || { printf "%-6s (no linked clu)\n" "$ns"; continue; }
        args=(--candidate "$cand" --gt "$GT")
        [ -n "$GTRES" ] && args+=(--candidate-res "$RES" --gt-res "$GTRES")
        echo "--- $ns ---"
        fiber-score "${args[@]}" 2>&1 | grep -E "ARI|pairwise|diagnosis|GT units split"
    done
fi

echo ""
echo "================ contamination (fiber-contam on each '<ns>_refine') ================"
for ns in $NS; do
    f="$BASE.clu.stderiv.$ELEC.${ns}_refine"
    [ -f "$f" ] || continue
    echo "--- $ns ---"
    fiber-contam "$BASE.yaml" "$ELEC" --variant "${ns}_refine" 2>&1 | grep -iE "two-cell|burst|flagged|contam" | head -4
done

echo ""
echo "================ QC report per branch (writes <base>.qc.<elec>.<ns>_linked.{csv,html}) ================"
for ns in $NS; do
    f="$BASE.clu.stderiv.$ELEC.${ns}_linked"
    [ -f "$f" ] || continue
    echo "--- $ns ---"
    fiber-qc "$BASE.yaml" "$ELEC" --variant "${ns}_linked" 2>&1 | grep -iE "qc:|wrote" | head -3
done

echo ""
echo "done.  Read precision-down -> over-merge, recall-down -> over-split, merged-up -> inspect those fusions."
