#!/usr/bin/env bash
# fiber-kit pipeline for one electrode group, with the three new patches wired in:
#   0077  fiber_geometry  robust xcorr-lag interchannel_offsets (used internally by 0078)
#   0078  fiber-refine    --fold-off-thr : inter-channel TIMING veto on the contaminant-fold
#                         (keeps small timing-distinct cells like 294/295 from being shattered)
#   0079  fiber-session   --cfiber-gate  : affine-invariant cfiber shape veto on fragment merges
#
# Apply the patches first (on a clean clone):
#   git am 0077-fiber_geometry-robust-xcorr-offsets.patch \
#          0078-fiber-refine-fold-timing-veto.patch \
#          0079-fiber-session-cfiber-gate.patch
set -euo pipefail

# ============================ session config ============================
SESS=sirotaA-jg-000005-20120312     # base name; expects $SESS.yaml/.fil/.spk.* in $DIR
ELEC=5                              # electrode / shank group
DIR=/data/testing/kke/fiber-0.27.1/$SESS
cd "$DIR"

# ===================== the important NEW knobs ==========================
FOLD_OFF_THR=0.22   # 0078 (samples): same-cell ~0.11, distinct co-located cells ~0.26,
                    #   so 0.20-0.25 vetoes the 294/295 fold. Empty string = off (old behaviour).
CFIBER_Q=0.90       # 0079: quantile of the within-fiber cfiber null used as the veto threshold

# ====================== 1. initial over-clustering ======================
# --cfiber-gate adds the shape veto to the Block-A fragment merges (precision).
fiber-session "$SESS" "$ELEC" \
    --method rkk --chunk-min 12 --merge-method sliding --merge-corr 0.90 \
    --cfiber-gate --cfiber-q "$CFIBER_Q" \
    --dipsplit --out stderiv

# ====================== 2. realign spikes ===============================
# rigid channel-summed xcorr realign; re-extracts the .spk/.spkD in the aligned frame.
fiber-realign "$SESS" "$ELEC" \
    --clu-method stderiv --align-method xcorr --reextract --refeaturize \
    --out-variant realign

# ====================== 3. positions (cpos) =============================
# IMPORTANT: cpos must be keyed to the SAME clu the merge tools read (avoids the
# stale-position mismatch). Point --clu-stage at the stage you just produced.
fiber-cpos "$SESS" "$ELEC" \
    --spk-method stderiv --clu-method stderiv --clu-stage realign \
    --out-method stderiv --out-stage refine

# ====================== 4. refine  (TIMING VETO ON) =====================
# --fold-off-thr is the fix: a small group is NOT folded into an amplitude/shape
# look-alike if their robust inter-channel offset profiles differ by > this.
fiber-refine "$SESS" "$ELEC" \
    --fold-off-thr "$FOLD_OFF_THR" \
    --split-min-corr 0.93 --chunk-minutes 12 \
    --out-method stderiv --out-variant refine

# ====================== 5. within-chunk merge ===========================
# emits the per-chunk unit signatures (.units.npz) that fiber-link consumes.
fiber-intrachunk "$SESS" "$ELEC" \
    --clu-stage refine --cpos-stage refine \
    --cos-thr 0.85 --off-thr 1.0 --depth-gate 35 \
    --sig-cap 8000 --emit-units \
    --out-stage refine_intrachunk

# ====================== 6. cross-chunk link =============================
# collapses per-chunk units across chunks -> the ~50-100 tracked neurons.
UNITS="$SESS.clu.stderiv.$ELEC.refine_intrachunk.units.npz"
fiber-link "$SESS" "$ELEC" \
    --from-units "$UNITS" \
    --cos-thr 0.85 --pos-thr 1.5 --off-thr 1.0 --max-gap 2 \
    --refine-trajectory \
    --out-stage refine_linked

echo "[done] linked clu: $SESS.clu.stderiv.$ELEC.refine_linked"
echo "       count tracked neurons (units spanning many chunks) to compare against the 50-100 target."
