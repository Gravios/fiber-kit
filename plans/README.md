# Diagnostic pipeline plans

A set of `fiber-pipeline --plan` files that run the pipeline several different ways and write every
intermediate and end product to a **unique tag namespace**, so all variants coexist on disk and can be
compared directly.  Use them to diagnose where the sort loses quality (over-merge, over-split, drift
tracking, cross-chunk linking) rather than guessing at parameters.

## How the namespacing works

Every branch starts from the same shared base over-cluster (produced once by `00-prep`) and writes its
own `.clu.stderiv.<elec>.<ns>_*` and `.units.npz` files under a per-branch prefix:

| plan | namespace | end product | what it probes |
|------|-----------|-------------|----------------|
| `00-prep` | *(base)* | `.clu.stderiv.<elec>` | over-cluster + realign (shared; run once) |
| `01-baseline` | `bl_` | `bl_linked` | reference: default refine/intrachunk/link |
| `02-merge-aggressive` | `mg_` | `mg_linked` | relaxed refine merge — does precision drop? (over-merge) |
| `03-split-conservative` | `sp_` | `sp_linked` | stricter merge / easier split — more, purer fragments (over-split) |
| `04-chunk-fine` | `c6_` | `c6_linked` | 6-min chunks — finer drift tracking |
| `05-chunk-coarse` | `c24_` | `c24_linked` | 24-min chunks — steadier templates, coarser drift |
| `06-link-cfiber` | `lc_` | `lc_linked` | cfiber co-gate + stricter cosine on the cross-chunk link |
| `07-curated-refit` | `cur_` | `cur_refit.units.npz` | post manual-curation: re-localize + refit (template) |

No two clu-producing or units-producing stages share a tag, so nothing overwrites anything (verified).

## Running

```sh
ELEC=5
fiber-pipeline $ELEC --plan plans/00-prep.yaml          # once: builds the shared base
for p in 01-baseline 02-merge-aggressive 03-split-conservative \
         04-chunk-fine 05-chunk-coarse 06-link-cfiber; do
    fiber-pipeline $ELEC --plan plans/$p.yaml
done
# 07 only after you have manually curated .clu.stderiv.$ELEC.curated in Klusters
```

Add `--dry-run` to any of these to preview the exact commands without running them.

## Diagnosing

`diagnose.sh` compares the branches.  Point it at the session base path and electrode; optionally give a
curated ground-truth `.clu` (e.g. the first-36-min curated head) to score against:

```sh
plans/diagnose.sh /path/to/sirotaA-jg-000005-20120312 5 \
    --gt /path/to/curated_head.clu --gt-res /path/to/curated_head.res
```

It prints, per branch: cluster count at `refine` and `linked`, contamination flags (`fiber-contam`), a QC
summary (`fiber-qc`), and — when a ground truth is given — `fiber-score` ARI / pairwise precision+recall /
split / merge.  Read it as: pairwise **precision** falling = that branch over-merges distinct cells;
pairwise **recall** falling = it over-splits; `merged candidates` rising = specific false fusions to inspect.
