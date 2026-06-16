#!/usr/bin/env bash
# fiber-kit pipeline for one electrode group.  Args verified against main HEAD (4075890),
# which already contains all of: 0074-0076 (intrachunk cfiber/dynamic/ms), 0077 (robust
# interchannel_offsets), 0078 (--fold-off-thr), 0079 (--cfiber-gate), 0080 (--refrac-ceiling),
# 0081 (--split-var-mult).  No patches to apply -- they are upstream.
#
# Fixes vs the previous (broken) script:
#   * fiber-session  uses --fine-method (NOT --method) for the algorithm.
#   * fiber-realign  has NO --clu-method and NO --align-method xcorr.  Valid align methods are
#                    {klusters,template,centroid}.  Input clu is --clu <path>.  CRUCIAL: do NOT
#                    pass --out-variant realign -- realign reads the INPUT .spk from that variant,
#                    so it would look for .spk.realign.N and fail.  Run it as an in-place commit
#                    (the realign IS the commit): variant is inferred from the clu name (stderiv).
#   * fiber-cpos     localizes from the RAW/standard spk (never .spkD); --spk-method standard.
#   * fiber-refine   input is --in-clu; output stage is --out-variant (NOT --out-stage).
#   * cpos is run AFTER refine, on the refine clu, so its positions are keyed to the SAME clu
#     fiber-intrachunk/-link read (fixes the stale-position mismatch).
set -euo pipefail

# ============================ session config ============================
SESS=sirotaA-jg-000005-20120312
ELEC=5
DIR=/data/testing/kke/fiber-0.27/$SESS
cd "$DIR"

# ===================== the NEW knobs (all upstream) =====================
CFIBER_Q=0.90        # 0079 fiber-session: within-fiber cfiber-null quantile for the shape veto
FOLD_OFF_THR=0.22    # 0078 fiber-refine : timing veto on the contaminant-fold (samples; 0.20-0.25)
SPLIT_VAR_MULT=1.5   # 0081 fiber-refine : only split clusters with top-3 feat-var > this x median
REFRAC_CEILING=1.0   # 0080 fiber-intrachunk: reject a merge whose combined 2ms-ISI viol exceeds this %

# ====================== 1. initial over-clustering ======================
# Produces <SESS>.clu.stderiv.<ELEC> (+ .spk.stderiv.<ELEC>).  --method is the extraction/feature
# tag (stderiv); --fine-method is the clustering algorithm.
fiber-session "$SESS" "$ELEC" \
    --method stderiv --fine-method rkk \
    --chunk-min 12 --merge-method sliding --merge-corr 0.90 \
    --cfiber-gate --cfiber-q "$CFIBER_Q" \
    --dipsplit

# ====================== 2. realign (in-place commit) ====================
# Reads .spk.stderiv.<ELEC> (variant inferred from the clu name), Klusters-style per-spike align,
# re-extracts the raw + stderiv .spk/.fet from .fil, commits aligned .res/.clu/.spk in place.
# NO --out-variant / --out-tag  ->  overwrites the canonical files (the realign is the commit).
fiber-realign "$SESS" "$ELEC" \
    --clu "$SESS.clu.stderiv.$ELEC" \
    --align-method klusters --reextract --refeaturize \
    --variants standard,stderiv

# ====================== 3. refine (split + fold gates) ==================
# --in-clu is the aligned canonical clu; --out-variant names the output stage (-> .clu.stderiv.<ELEC>.refine).
# --split-var-mult (curator variance trigger) + --fold-off-thr (timing veto on the contaminant fold).
fiber-refine "$SESS" "$ELEC" \
    --in-clu "$SESS.clu.stderiv.$ELEC" \
    --out-method stderiv --out-variant refine \
    --split-min-corr 0.93 --split-var-mult "$SPLIT_VAR_MULT" \
    --fold-off-thr "$FOLD_OFF_THR" \
    --chunk-minutes 12

# ====================== 4. positions ON THE REFINE CLU ==================
# Run cpos AFTER refine so positions key to the clu the merge tools read (no stale-cpos mismatch).
# cpos localizes from the RAW/standard .spk (re-extracted in step 2), never the stderiv .spkD.
fiber-cpos "$SESS" "$ELEC" \
    --clu-method stderiv --clu-stage refine \
    --spk-method standard \
    --out-method stderiv --out-stage refine

# ====================== 5. within-chunk merge ===========================
# --refrac-ceiling rejects merges whose combined train inflates the 2ms refractory violation
# (backstop for high-rate over-merges).  Emits the .units.npz for linking.
fiber-intrachunk "$SESS" "$ELEC" \
    --clu-method stderiv --clu-stage refine \
    --cpos-method stderiv --cpos-stage refine \
    --cos-thr 0.85 --off-thr 1.0 --depth-gate 35 \
    --refrac-ceiling "$REFRAC_CEILING" \
    --sig-cap 8000 --emit-units \
    --out-stage refine_intrachunk

# ====================== 6. cross-chunk link =============================
UNITS="$SESS.clu.stderiv.$ELEC.refine_intrachunk.units.npz"
fiber-link "$SESS" "$ELEC" \
    --clu-method stderiv --clu-stage refine \
    --cpos-method stderiv --cpos-stage refine \
    --from-units "$UNITS" \
    --cos-thr 0.85 --pos-thr 1.5 --off-thr 1.0 --max-gap 2 \
    --refine-trajectory \
    --out-stage refine_linked

echo "[done] linked clu: $SESS.clu.stderiv.$ELEC.refine_linked"
echo "       count units spanning many chunks and compare against the 50-100 target."
