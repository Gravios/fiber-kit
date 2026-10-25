# fiber-bench — compare parameter variants under a config hash

`fiber-bench` runs the plan of a config (e.g. `fiber-kit-exp.yaml`) for one or many parameter variants,
writes every clustering step's output under a tag that carries a **hash of the parameters that step
reads**, scores every run with the metrics used to tune the pipeline so far, and renders an HTML report
to compare them.  Its first job is tuning **fiber-session**.

```bash
cd <session dir>
# 1. run variants (default: the plan up to and including fiber-session)
fiber-bench run sirotaA-jg-000005-20120312 7 --config fiber-kit-exp.yaml \
    --sweep plans/sweeps/session-first-pass.yaml           # or --grid K=v1,v2 / --set K=V
# 2. score them (and any existing sorts as baselines) against a curation
fiber-bench score sirotaA-jg-000005-20120312 7 --gt-clu old/....clu.standard.7 --gt-res old/....res.standard.7 \
    --gt-units 2-32 --gt-exclude 7,18 --stages fiber_session
# 3. compare
fiber-bench report sirotaA-jg-000005-20120312 7          # -> fk-bench/report.html
fiber-bench list   sirotaA-jg-000005-20120312 7
```

## Tags and hashes

A run's outputs are ordinary stage files whose tag ends in the hash, e.g.
`<base>.clu.stderiv_C5_D34.7.fiber_session_c45730da` and the artifact mask
`<base>.clu.standard.7.artifact_677d157e`, so Klusters, `fiber-score` and every later stage read them
like any other stage.  The hash (8 hex digits of a SHA-1) covers:

| step | knobs in its hash |
|---|---|
| fiber-session | every `FK_*` key not owned by another stage (so a new session knob is never missed), plus the artifact step's hash when `FK_SESSION_EXCLUDE` is on |
| fiber-flag-artifacts | `FK_FLAG_*` |
| fiber-xcorr-merge, fiber-rdr, ... | their own prefix (`FK_XCM_*`, `FK_RDR_*`, ...), plus the hash of the step that made their `in:` |
| every step | its plan `params:`, its stage name, `FK_REALIGN_VARIANT` |

So the same parameters always give the same tag, a changed merge knob reuses the cached session output,
and a changed session knob gives every later step a new tag.  A step whose output already exists is
skipped: a sweep can be stopped and resumed.  The hash covers parameters, not code; the fiber-kit
version is recorded with each run (`fk-bench/runs.jsonl`).

## Isolation

* Each run has its own config, `fk-bench/runs/<hash>/fiber-kit.yaml` (the base config, the overrides, the
  plan with hashed tags), and its log beside it.
* fiber-pipeline runs with every `FK_*` variable removed from the environment: an exported variable
  would override the yaml and the hash would not describe the run.
* **fiber-session's end-of-run realign is forced off** (`FK_SESSION_REALIGN: off`, recorded in the hash):
  it rewrites the shared `.res`/`.spk` in place, so every later run would start from data an earlier
  run had moved.  `--keep-realign` restores it for a single deliberate run.  In-place `fiber-realign`
  steps are refused for the same reason unless `--allow-inplace`.

## Metrics

Against a ground truth (`--gt-clu/--gt-res`, spikes matched one-to-one within `--gt-tol` samples):

| column | meaning |
|---|---|
| `pyr_acc`, `int_acc`, `all_acc` | mean over truth units of the best single cluster's accuracy n_uc / (n_u + g_c − n_uc) |
| `*_rec`, `*_prec` | that cluster's recall and precision |
| `*_frag80` | clusters needed to hold 80% of a unit (over-splitting) |
| `*_n80` | units with accuracy ≥ 0.8 |
| `merges_pyr_pyr`, `merges_with_int`, `merge_list` | clusters holding ≥ 20% and ≥ 50 spikes of a second truth unit |

Units are typed **pyramidal / interneuron** by firing rate (`--int-rate`, 5 Hz; or `--interneurons`).

Without truth (every run): `n_clusters`, `frac_c0`/`frac_c1` (shares in clusters 0 and 1),
`median_size`, `frac_small`, refractory violations (`isi_viol_median`, `frac_isi_over_1pct`),
`rdr_median`/`rdr_flagged` (fiber-rdr score, lower-envelope line), `gain_iqr_median`, `rvu_median`,
`rvu_over_05`.  With truth, `auc_rdr`, `auc_gain_iqr`, `auc_rvu_max`, `auc_isi_viol` give how well each
truth-free measure ranks the clusters below 90% purity: the basis for a truth-free objective on
sessions without a curation.

Files in `fk-bench/`: `metrics.tsv` (one row per sort), `units.tsv` (per truth unit), `clusters.tsv`
(per cluster: size, ISI, rdr, gain IQR, RVU and, with truth, purity), `report.html`.

## Report

A sortable table (overrides and metrics per run), pyramidal accuracy against false pyramidal merges, the
effect of each swept parameter on accuracy / recall / precision / merges / cluster count, a per-unit
accuracy heatmap (runs × units, interneurons starred), and the truth-free measures against cluster purity.
