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
| fiber-session | every `FK_*` key not owned by another stage (so a new session knob is never missed), fiber-consolidate's `FK_CONS_*` among them (its in-stage consolidation pass), plus the artifact step's hash when `FK_SESSION_EXCLUDE` is on |
| fiber-flag-artifacts | `FK_FLAG_*` |
| fiber-xcorr-merge, fiber-rdr, ... | their own prefix (`FK_XCM_*`, `FK_RDR_*`, ...), plus the hash of the step that made their `in:` |
| fiber-intrachunk, fiber-anchor-link | their own prefix and `FK_CONS_*` (they host the consolidation pass too); fiber-consensus has `FK_CONSENSUS_*` only |
| every step | its plan `params:`, its stage name, `FK_REALIGN_VARIANT`, `FK_CHUNKS`, `FK_SESSION_CHUNK_MIN` |

A step that **annotates** its input rather than writing a new `.clu` -- `fiber-cpos` with `out:` equal to its
`in:`, as the exp plan's `cpos-clean` -- keeps its input's tag: its `.cpos` table is named for the
labelling it describes, the next step reading that name still gets the labelling, and the step counts as
done when the `.cpos` table exists.

So the same parameters always give the same tag, a changed merge knob reuses the cached session output,
and a changed session knob gives every later step a new tag.  A step whose output already exists is
skipped: a sweep can be stopped and resumed.  The hash covers parameters, not code; the fiber-kit
version is recorded with each run (`fk-bench/runs.jsonl`).

## Chunk runs

`fiber-bench run --chunks 14-16` sets `FK_CHUNKS` for every variant (or set it per variant with `--set`):
each stage sorts only those chunks, and the others' spikes go to the reserved clusters
16777116+chunk (pipeline.md, Chunk runs).  A parameter search on a few chunks costs only those chunks' sorting
time.  `FK_CHUNKS` is in every step's hash, so chunk runs never share a tag with
full runs.  `score` leaves the reserved spikes out: truth spikes in a chunk that was not run count for
no unit.  Units with no truth spikes in the chunks run are dropped, and `units_scored` and
`chunks_left_out` record that.  The free metrics ignore the reserved clusters.

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

## fiber-tune

`fiber-tune` drives fiber-bench to **search** a plan's knobs against a ground truth on a few chunks, then
**confirms** the best configurations with the full plan.

```bash
cd <session dir>
fiber-tune search  <session> <elec> --spec plans/tune/exp-search.yaml      # resumable; --max-trials N
fiber-tune confirm <session> <elec> --spec plans/tune/exp-search.yaml --top 3 [--chunks all]
fiber-tune show    <session> <elec> --spec plans/tune/exp-search.yaml
FK_CONFIG=fk-bench/tune/exp-search/best.yaml fiber-pipeline <elec>       # use the winner
```

**Spec** (yaml; `plans/tune/exp-search.yaml` is a worked example over the exp plan): `config` (base config
with a plan), `truth` (`clu`, `res`, optional `units`, `exclude`, `tol`, `int_rate`), `chunks` (the
search's `FK_CHUNKS`), `score_at` (the plan step scored; default the last step with `out:`), `objective`,
`set` (fixed overrides), `start` (where the search starts), `passes`, `min_gain`, `search_realign`, and `stages`: knob
groups, each a mapping of knob -> list of values, searched in order (upstream first).  Every knob must
be one the base config lists, so a typo stops the run instead of doing nothing.

**Search.**  Coordinate ascent: each knob in turn is tried at each listed value with every other knob at
the current best, and its best value is kept when the objective rises by at least `min_gain`; the knob
list is walked `passes` times, stopping early when a pass changes nothing.  Each trial is one
`fiber-bench run` with the trial's overrides, so steps whose parameters did not change are re-used: a
merge knob re-runs the merges, not fiber-session.  Interactions between knobs are reached only through
later passes; this is not a global search.

**In-place realign.**  The search leaves out every `fiber-realign` step without `out:` (they rewrite the
shared `.res`/`.spk`, so each trial would start from data the previous one moved), and fiber-bench turns
fiber-session's own end-of-run realign off.  `confirm` re-runs the start configuration and the `--top`
best with the full plan, realign steps included, each in a private workspace
(`fk-bench/tune/<name>/ws/<hash>/`): the `.res`/`.spk`/`.fet`/offset files are copied (reflinked where the
file system allows), everything else is symlinked, stage outputs are not carried over.  The copies are
removed after scoring unless `--keep-ws`; the workspace's stage outputs stay for inspection.

The realign matters: on g6 0316 (one 6-min chunk, identical parameters) the search scored ARI 0.675 /
accuracy 0.376 and the workspace run ARI 0.503 / accuracy 0.417, because fiber-session's end-of-run realign
and the realign steps change the spikes every later stage sees.  Treat the search's ranking as a shortlist
and let `confirm` decide; or set `search_realign: true` in the spec to run every trial that way (faithful,
but each trial re-sorts the chunks, since nothing can be shared between workspaces).

**Metrics** per trial (truth spikes in chunks not run are left out): fiber-bench's `all_acc`, `pyr_acc`,
`int_acc`, `*_rec`, `*_prec`, `*_n80`, `*_frag80`, `merges`, `merges_pyr_pyr`, `merges_with_int`, plus
`ari` (sorted truth spikes), `purity` (of the sorted truth spikes), `completeness` (share of truth spikes
in their unit's best single cluster), `unsorted` and `n_clusters`.  The `objective` is an arithmetic
expression over these names (`+ - * /`, numbers, `min`, `max`, `abs`); a missing name or NaN scores
`-inf`.

**Output** in `fk-bench/tune/<name>/`: `trials.jsonl` (every trial with its overrides, metrics, objective
and phase), `trials.tsv`, `summary.md` (best first, search and confirm) and `best.yaml` (the base config
with the best overrides and the plan as given, realign steps included, no `FK_CHUNKS`).

**Choosing chunks.**  At least two consecutive chunks, so the linker and the merges see a chunk boundary,
and chunks the truth covers well.  `FK_SESSION_CHUNK_MIN` cannot be searched on a chunk subset (the chunk
indices would move); set it under `set:` or search it with `chunks: all`.  A winner on a few chunks is a
candidate, not a result: `confirm --chunks all` scores it on the whole session.
