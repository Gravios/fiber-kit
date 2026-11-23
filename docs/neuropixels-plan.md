# Plan: fiber-kit on high-density probes (Neuropixels 1.0 / 2.0)

2026-11-22. Written against fiber-kit `b50d20e` (0.67.37). **A design plan, not a description of
what fiber-kit does**: nothing here has been built or run, and there is no Neuropixels recording
in the project yet.

Revised the same day with Justin, in this order: the filtered file is transient and the kept
file sparse (6a); a spike is sorted in one tile, chosen from its spatial profile (3.2); a drift
field from the signal's depth profile (5.4); detection on a wider set than the sorting set (3);
all tile sizes derived from the row pitch and the measured cell size (3.1).

## 1. Summary

Keep fiber-kit's unit of work -- one channel group, cut into time chunks -- and make the group a
**tile**: a short stretch of one shank, sized from the probe's row pitch and the measured cell
size. Each tile is an ordinary `spikeDetection.channelGroups` entry, so every existing stage and
Klusters read it unchanged. Events are detected and located on a set of channels wider than the
one the tile is sorted on, and a spike is sorted in **one** tile, the one it is located in, so
that it is never sorted on a partial waveform. Units cut by a tile border are put back together by the linker that already works across
chunks, seeded by a thin band of spikes two neighbours share. A new last stage gives each unit one
home tile and moves its strays there.

What is new in fiber-kit is small and sits at the two ends:

| | new | changed | untouched |
|---|---|---|---|
| layout | `fiber_probe.py` (site table, tiles) | `session_yaml.py` reads tile fields | |
| per tile | | recalibrated knobs, a tile profile | fiber-session, realign, consensus, refine, cpos, merges |
| across tiles | `fiber-tile-anchors`, `fiber-tile-reconcile`, `fiber-drift-field` | `fiber_anchor_link.py`, `fiber_link_core.py` take (tile, chunk) nodes | |
| driver | `fiber-pipeline --tiles`, job pool | | plans, per-stage CLIs |

Everything that turns a recording into `.dat`/`.fil`/`.spk`/`.res` per group stays in neurosuite-3
and ndmanager-plugins (section 7).

## 2. The probes

| | NP 1.0 | NP 2.0 |
|---|---|---|
| sites / recorded at once | 960 / 384 | 1280 per shank / 384 |
| layout | 2 sites per row, checkerboard over 4 columns, rows ~20 um apart | 2 aligned columns 32 um apart, rows ~15 um apart |
| 384 contiguous channels span | 192 rows, ~3.84 mm | 192 rows, ~2.88 mm (one shank) |
| bands | spike band 30 kHz + LFP 2.5 kHz | one wideband stream, 30 kHz (to confirm) |
| shanks | 1 | 1 or 4 (to confirm spacing; 250 um as I recall) |

Site counts, layouts and the 1.0 sampling rates are from the Allen description and Steinmetz et
al. 2021 (sources at the end). Not verified here, and to be checked against the probe manual
before any code depends on them: the 2.0 bit depth, the 4-shank spacing, and the per-channel ADC
sampling offsets (section 7).

Two properties decide the design:

* **No natural groups.** The recorded sites are a user-chosen subset (the IMRO table) and need not
  be contiguous, or even on one shank. The site table must come from the recording's metadata.
* **A unit covers a few rows of a long column.** A pyramidal footprint is roughly 60-100 um, i.e.
  4-7 rows, on 8-14 of 384 channels. The existing octrode (8 sites, 20 um pitch) is the same scale.

## 3. Tiles

A tile is a contiguous run of rows on one shank, with three nested extents:

* the **detection set**: the channels the transient pass (6a) looks at to detect an event and to
  decide where it is;
* the **sorting set**, inside it: the only channels of the tile's `.spk`/`.fet`, the ones it is
  clustered on;
* the **assignment zone**, inside that: a spike is sorted in the tile whose zone contains it. Zones
  of successive tiles abut, so the zone height is the stride.

### 3.1 Sizes follow from the row pitch and the cell size

Nothing is a fixed channel count. All sizes are in um and derive from:

| quantity | from | meaning |
|---|---|---|
| `p`, sites per row | the site table (`fiber_probe.py`) | row pitch along the shank |
| `F` footprint extent | measured on the recording class: along depth, the span over which a unit's template stays above the noise (monopole: FWHM ~ 3.46 z0, `fiber_localize`); the 90th percentile over units | the cell size |
| `m` flank margin | choice, default `F / 4` | flank kept inside the sorting set for a cell at the zone's edge |
| `h` halo | at least one row, default `F / 2` | what localization needs beyond the sorting set to see both flanks |

    sorting rows   = ceil(F / p) + 1
    detection rows = sorting rows + 2 * ceil(h / p)
    zone rows      = sorting rows - 2 * ceil(m / p)        (at least 1; the stride)

| | p | F = 60 um | F = 100 um |
|---|---|---|---|
| NP 1.0 (2 per row) | 20 um | sort 4 rows / 8 ch, detect 8 rows / 16 ch, zone 2 rows (96 tiles per 192 rows) | sort 6 / 12, detect 12 / 24, zone 2 |
| NP 2.0 (2 per row) | 15 um | sort 5 / 10, detect 9 / 18, zone 3 (64 tiles) | sort 8 / 16, detect 16 / 32, zone 4 (48 tiles) |
| Buzsaki64L (1 per row) | 20 um | the shank (8 sites) is one detection set: one group, as today | the same |

"Detect on 12, sort on the central 8" (Justin's formulation) is the NP 1.0, F = 60 um row with a
halo of one row instead of two, and the zone equal to the sorting set (48 tiles).

`F` is a property of the cells, so it belongs to the recording-class profile (region, cell type)
beside the sorting knobs, and is measured, not assumed: a first pass with a generous `F` on a
stretch of the shank gives the units' footprints, and the tiling is then set from their
distribution. The plan sizes on the pyramidal footprint and accepts that the largest cells (some
interneurons) are sorted with a clipped flank.

When the rule lands on 8 sorting channels, the sorting dimension is the octrode's: the knobs of
section 4, Klusters' views and KiloKlustaKwik carry over, and recalibration reduces to the
sampling rate and the row geometry. On NP 2.0, or for larger cells, it does not land on 8.

### 3.2 A spike is sorted in one tile, chosen by where it is

* In the transient pass each event is located on its detection set: first by the
  amplitude-weighted centroid of its profile over the channels above the noise, which is cheap and
  uses the tile and its two neighbours (the reason the pass holds three tiles). `fiber_localize`
  constrains depth well only when both flanks of the footprint are sampled and flags a peak on a
  terminal channel as degenerate; the halo is what gives every spike in the sorting set both
  flanks. A spike that peaks on an edge channel of the sorting set but is centred outside shows it
  by its span into the halo and goes to the neighbour.
* The spike is sorted in the tile whose zone contains that depth (in tissue coordinates once the
  drift field of 5.4 exists). It is never sorted on a waveform that lacks its peak.
* **Border band.** A spike within `FK_TILE_BORDER_UM` (default half a row pitch) of a zone
  boundary is written to both tiles: the same physical events in two tiles, the seeds for linking
  (5.1) and the positive class for its calibration.
* **A cell on a boundary is still split by the per-spike assignment.** `fiber_localize` is
  explicit that per-spike amplitude noise is large and position is a per-unit quantity, and the
  centroid can move with amplitude within one cell (the later spikes of a complex-spike burst).
  So the assignment is two passes: (i) per spike, as above, and sort; (ii) per **unit**: localize
  each cluster's template on its detection set (1-3 um for interior units) and re-extract the whole
  unit in the tile where it is central, together with the fragment the linker (5.2) found for it
  next door. The sparse `.fil` windows are centred on the spike (6a), so pass (ii) is a re-cut.
* A non-contiguous site selection breaks a shank into **runs**; tiles never span a gap larger than
  one row pitch. Runs and shanks are independent: nothing is linked across them.

To be measured in phase 0, not assumed: the zone equal to the sorting set (a cell at the zone's
edge row keeps its peak but has one flank outside the sorting channels -- the situation of a cell
at the end of an octrode shank today) against the narrower zone of the rule above.

### 3.3 In the session YAML and in code

Tiles are written into the session YAML by ndmanager, on request, as channel groups with extra
keys:

```yaml
spikeDetection:
  channelGroups:
    - channels: [2, 3, ..., 9]          # the sorting set: global channel ids, as today
      nSamples: 61
      peakSampleIndex: 20
      tile: {probe: 0, shank: 0, run: 0, index: 0,
             detect: [0, 1, ..., 11],   # the detection set
             zone_um: [40.0, 80.0]}     # depth range of the assignment zone
```

`session_yaml.load_session` already returns `channels` and the probe files for a group;
`fiber_localize.load_geometry` already indexes the site table by global channel id, so positions
(`.cpos`) are in probe coordinates without a change. A group without `tile:` is what it is today.

**New module `fiber_probe.py`** -- the only place that knows site positions:

* `site_table(session)` -> per channel: probe, shank, x, depth, row, run. From the `.probe`
  files (one loader: `fiber_localize.load_geometry` moves here and keeps its name as an alias).
  The reversed-geometry correction (`FIBER_PROBE_FLIP_Y`) lives here too.
* `tiles(site_table, footprint_um, margin_um, halo_um)` -> the tile list by the rule of 3.1;
  `tile_of_group(session, g)`, `neighbours(g)`, `shared_channels(g, h)`.
* `fiber-probe footprint <session> <elec>` measures `F` from a sorted stretch;
  `fiber-probe tiles <session> [--write-yaml]` prints the tiling, or hands it to ndmanager's
  writer. fiber-kit does not write the YAML by itself.

## 4. Per-tile sorting

No stage changes. What does change is calibration, and none of it should be copied from the
octrode profile:

| knob | why it moves |
|---|---|
| `FK_SESSION_RKK_DIMS` (6) | set for 8 channels; unchanged where the tile rule gives 8 sorting channels, to re-derive where it gives more |
| lag of the `_D34` feature space (+-3 samples at 32,552 Hz) | 30 kHz; re-derive, as planned for any new session |
| `MergeThresh` = chi2(nSpatialDims, 0.9999) | follows the dimension count by construction; check it |
| chunk / overlap minutes | depend on drift rate in um/min, unknown until measured |
| consensus neighbourhood d' (12), min spikes | many more units per chunk are confusable in a column |
| expected parents per group (<= ~100) | per tile, a few to a few tens depending on the zone height; a border unit possibly as two fragments until pass (ii) of 3.2 |
| stderiv pattern | `sdiffPairs` per group, generated from the tile's geometry (nearest rows), since the all-pairs form over a larger sorting set mixes distant sites |

Deliverable: a profile `fiber-kit-np1.yaml` / `fiber-kit-np2.yaml` beside `fiber-kit-exp.yaml`,
each knob with the measurement that set it.

To check early, only where the rule gives more than 8 sorting channels: Klusters and KiloKlustaKwik with 10-16 channels per group (feature count
3 x channels = 30-48; waveform and matrix views).

## 5. Linking across tiles

### 5.1 Spatial anchors -- `fiber-tile-anchors`

Two neighbouring tiles hold the same physical spikes in their border band (section 3). The
transient pass detects each event once and gives it an id, stored with the window origins of the
sparse `.fil` (6a), so the identity is exact and survives each tile's own realign. For a pair
(tile g, tile h), per chunk, two fragments are anchored when each holds the majority of the
other's border spikes (the mutual-majority rule fiber-session uses for the chunk overlap).

Output: `<base>.tanchors.<method>.<g>-<h>.<stage>.npz` with `(chunk, atom_g, atom_h, n_shared,
frac_g, frac_h)`, in the `.clc` labels of the stage it read, like `drift_anchor_pairs`.

A border band holds few spikes of any one cell, fewer than a chunk overlap does. Anchors will be
sparse; the scored links of 5.2 carry more of the load across tiles than they do across chunks.

### 5.2 One linker in space and time

`fiber_anchor_link.anchor_link` works on fragments keyed by chunk (`byc`, `chunk_of`). The change
is to key them by **(tile, chunk)**:

* seeds: temporal anchors (as now) and spatial anchors (5.1), both applied before any scored link;
* candidates: same tile, chunk gap <= `max_gap` (as now); neighbouring tile, same chunk or
  adjacent chunk (a unit drifting across the tile border);
* score: `flc.ci_overlap` x template cosine, as now within a tile. Across tiles the sorting sets
  may share no channel (3.1), so a cross-tile pair is scored on templates built from the
  spike-centred windows of the sparse `.fil` (6a), re-cut for both fragments on one common set of
  rows around the pair; `pair_channels` takes the two channel lists and returns index pairs. Both
  fragments then have their whole footprint in the comparison, not an edge view against a centre
  view. A pair with fewer than `FK_TILE_MIN_SHARED` (4) common primary channels is not a candidate;
* null for calibration: two fragments of the same (tile, chunk), as now. Positives: both kinds of
  anchor, reported separately, since spatial anchors should score higher than temporal ones;
* the rule "a cell never holds two co-temporal fragments" becomes "... two in the same tile";
* rate gate: the relative rate is taken within the tile, and across tiles it compares only spikes
  with the peak on shared channels;
* the warp veto finally has the channel count it needs (>= 3 channels with finite delay), so
  `FK_ALINK_WARP_THR` is worth calibrating here.

Output: cell ids that are global per run of a shank, written as each tile's `.clu` + `.clc/.clp`
(the cell's fragments in that tile), plus `<base>.tcells.<method>.<stage>.npz`: cell -> tiles,
chunks, fragments.

### 5.3 Reconcile -- `fiber-tile-reconcile`

For each cell: the home tile is the one whose core contains the energy-weighted depth of its
template, taken over the whole session. Its spikes sorted in a neighbouring tile (the strays of a
cell near a border) are **moved** to the home tile: their waveforms are re-cut on the home tile's
channels from the sparse `.fil`, whose windows are centred on the spike and wide enough for
either tile (6a). Border-band duplicates are dropped from the tile that is not home. The decision
is made per stage for `.res`/`.clu`/`.clc`/`.clp`/`.spk`/`.fet` together, as dedup does.

Session-level output, one per shank run: `<base>.res.<method>.s<shank>r<run>.<stage>` and the
matching `.clu` (each physical spike once, global cell ids) with a `.cli` sidecar that names the
home tile. This is what downstream analysis reads; tiles remain the curation view.

A cell that drifts further than half a tile over the session has no single good home. Reported,
and left in the tile holding most of its spikes; whether to split such a cell's storage by time is
an open question (section 9).

### 5.4 Drift from the signal itself -- `fiber-drift-field` (added 2026-11-22, Justin's proposal)

A tile fixed on the probe does not follow a neuron: when the probe moves against the tissue, a
cell near a core boundary changes tile. `fiber_drift.py` estimates drift from the depths of
tracked units, and says of itself that drift faster than the link gate breaks the tracks it
depends on and "is better estimated by raster registration". It is also circular here: it needs
the units that the tiling is supposed to produce. So estimate the displacement before sorting,
from the depth profile of the signal.

**Estimate.** In the transient pass, per time bin (default 2 s) and per row: the power in a few
bands -- the spike band (300 Hz up; a sharp peak at a cell layer and available at all times) and,
from the LFP stream, the ripple band and gamma (layer landmarks in hippocampus, but present only
in some states). Each bin's depth profile is registered against the others by cross-correlation
along depth, per band, sub-row by interpolating the correlation peak, and solved decentrally for
one displacement curve; in windows along the shank for a non-rigid field `d(t, depth)`. This is
the approach of DREDge (Windolf et al.), which registers either spike rasters or LFP; the
per-event centroids and amplitudes of the transient pass give the spike raster for free, as a
second estimate. Output: `<base>.drift.s<shank>r<run>.npz` (`t`, `depth`, `d`, per-band
estimates and their disagreement).

What it measures is the motion of the probe against the tissue, common to all cells at a depth.
It does not see one cell changing on its own. The spectral profile also changes with brain state
(ripples in rest, theta and gamma in running), which shifts power between layers without any
motion: bins are registered within a state, or on the spike band only, and the disagreement
between bands is the check.

**Use, in order of how much it changes:**

1. *Assignment in tissue coordinates.* A spike's tile is chosen from `centroid - d(t, depth)`,
   so a cell keeps its tile while the probe moves. The tile's channel set then follows the
   tissue: per chunk it is shifted by a whole number of rows (the sparse `.fil` windows are
   centred on the spike, so the `.spk` view is a re-cut). Whole rows only, no interpolation of
   waveforms between sites: the residual sub-row displacement stays in the data as a waveform
   change, which the chunk linker is built for. On NP 1.0 the checkerboard repeats every 2 rows,
   so the step is 40 um there and 15 um on NP 2.0.
2. *A prior for the linker.* `flc.link` already takes `drift=` (transforms fitted from the
   anchors by `fit_drift_transforms`). The field gives the expected depth change between two
   chunks without anchors: a candidate pair whose `.cpos` depths differ from it by more than a
   tolerance is not a candidate. This is where sparse spatial anchors (5.1) need the help.
3. *The home tile* (5.3) is taken in tissue coordinates, which answers the case of a cell
   drifting more than half a tile: it has one home, and the home moves on the probe.
4. *Chunk length* from the measured um/min instead of a default.

**Checks.** The field against the anchor-fitted drift and `fiber-drift`'s unit-based curve on the
same session (three estimates with different inputs); on an imposed-motion recording against the
known motion; negative control: time bins shuffled within a state give no displacement beyond
the estimator's noise, which is the resolution to quote.

**Expectation, to be measured.** In a chronic recording the displacement within a session may
stay under one row step; then use 1 does nothing and uses 2-4 remain. On the octrodes (8 rows)
only the estimate and use 2 apply.

## 6. Driver and resources

* `fiber-pipeline --tiles <spec> [--jobs N] <plan>` runs a plan per tile in a job pool, then the
  three cross-tile steps once.
* Per-tile `.spk` storage with each spike once (plus the border band) is about half of the first
  draft's. A tile is an `<elec>`, so resuming, single-stage runs and plans work
  as they do now.
* Tiles are independent until 5.1: memory is one tile's, wall time is tiles / jobs.
* Disk, per waveform variant: spikes x ~1.25 (border band) x channels x samples x 2 bytes. At
  16 x 61 samples a spike costs ~2 kB, so 50 million detected spikes are ~120 GB per variant
  (see 6a for not storing them at all). Measures, in
  order: keep `standard` and one sort variant on disk and derive the rest on demand; cut
  `nSamples` to what 30 kHz needs; raise the detection threshold for tiles outside the layers of
  interest. A shank-wide sparse waveform store would halve it but is a new format on both sides of
  the contract, so it is not planned.
* GPU: `backend.py` (CuPy) already covers realign and whitening; nothing new is required for
  tiling. The ordering and t-SNE costs are per tile and do not grow with the probe.

## 6a. The filtered file: transient, then sparse (added 2026-11-22, Justin's proposal)

A full `.fil` of a 384-channel probe is as large as the `.dat` (~83 GB per hour at 30 kHz) and
is only ever read near spikes, plus a sample of spike-free noise. Proposal: never keep it.

**Transient pass.** The extractor walks the shank with a window of three overlapping regions (a
tile and its two neighbours -- what is needed to decide which tile holds an event's peak). It
filters the window from the `.dat`, detects, and writes for the middle tile, then moves on. The
filtered samples exist only in memory or a scratch file. The pass is reproducible from the `.dat`
and the recorded filter parameters, so anything that later needs continuous data (re-detection on
a residual after decollision or deconvolution, a re-cut beyond the margin) re-runs it for the
tiles concerned.

**What the pass must leave behind**, because 30 fiber-kit modules and Klusters' realign read the
`.fil` today:

| kept | replaces | used by |
|---|---|---|
| sparse `.fil`: one padded window per spike, **centred on the spike's own row** (centroid row +- half the detection rows of 3.1 plus one zone, clipped at the run's ends; +- 6 rows in the NP 1.0, F = 60 um case), `nSamples + 2 x margin` samples | `.fil` near spikes, and `.spk.standard` (a view of it at `.res - origin`, on the tile's channels) | realign `--reextract`, the session commit, cpos/localize, Klusters realign and re-featurize, moving a spike to a neighbouring tile (5.3) |
| window origins, `.res` order, fixed at detection: sample, first row, event id | | the `.res` moves on realign; the windows do not. The event id is the identity of 5.1 |
| spike-free snippets per tile and chunk (`FK_LOWMIX_NOISE_N`-sized sample) and the per-chunk channel covariance | the rest of the `.fil` | whiteners, the envelope noise floor, lowmix, manifold, `tools/fil_noise_*` |
| the artifact mask | | fiber-flag-artifacts (unwhitened derivative), run inside the pass |

**Why per-spike windows and not a time-sparse file of all channels.** On a dense probe some
channel is spiking almost all the time: at 1.4 kHz of detected events (10 million in 2 h) and
3 ms windows, the union of windows covers ~99 % of the recording, so a file sparse in time only
saves nothing. Sparsity has to be in space as well, and a window per spike on the rows around it
is that, with O(1) access by spike row and no index of merged intervals. The price is redundancy
where neighbouring spikes fall within a window of each other.

The window is centred on the spike rather than cut on its tile's channels so that the spike has
its whole footprint whichever tile ends up holding it: +-6 rows cover the spike's own tile, and
the neighbouring tile for a spike near the border.

**Margin.** Realign shifts are bounded (Klusters: `max(1, peak // 2)` per pass, accumulating over
commits); super-sampling and the temporal derivative need a few samples more. A margin of 12
samples (0.4 ms) either side is the starting value; a spike whose accumulated shift would leave
the window is reported and needs the transient pass for its tile.

**Disk**, NP 1.0 (13 rows = 26 channels, 61 + 24 samples, int16, each spike once): 4.4 kB per
physical spike. The right-hand column is what today's layout would cost with one copy per spike
on 16 channels, border duplicates (~25 %) not counted on either side.

| detected in 2 h | sparse `.fil` | full `.fil` + `.spk.standard` + one sort variant |
|---|---|---|
| 10 million | ~44 GB | 166 + 20 + 20 GB |
| 50 million | ~221 GB | 166 + 98 + 98 GB |

The sort variant (`stderiv*`) is a fixed transform of the padded window (spatial pattern inside
the tile, temporal derivative inside the margin), so its `.spk` can be derived on demand and only
its `.fet` kept. The gain is large at low rates (about 4.5x at 10 million) and under 2x at high
ones; at 50 million the sparse file alone is bigger than the full `.fil`, and the saving comes
from dropping the `.spk` copies.

**Cost.** A new format on both sides of the contract: `neuro_io` (one reader that serves "window
of spike i" from either a full or a sparse `.fil`, so the 30 callers change in one place) and
Klusters' realign / feature-space code. An octrode session keeps its full `.fil`; the reader hides
the difference.

**Check in phase 0.** Build the sparse file for g6 from its existing `.fil`, point the reader at
it, and require realign `--reextract`, the session commit and cpos to give byte-identical files
(the test pattern of `test_session_realign_e2e.py`), and consensus / lowmix results unchanged
with the kept noise snippets.

## 7. Upstream (neurosuite-3 / ndmanager-plugins), not fiber-kit

1. Converter from SpikeGLX (`.bin` + `.meta`) and Open Ephys to `.dat`, with the IMRO site
   selection written as a `.probe` file and `probes:` entry (the pattern of `ndm_plx2dat`).
2. Tile generator in ndmanager: calls the tiling rule, writes `channelGroups` on request.
3. NP 1.0: the channels of one ADC are sampled in turn, so channels are offset by a fraction of a
   sample. Any cross-channel derivative (stderiv) and the group-delay term need this corrected
   before the high-pass. To verify for 2.0.
4. Detection once per event in the transient pass, with the centroid, the tile assignment and
   the border band of section 3.
5. `contract.yaml`: `tile:` keys in the session schema; `.tanchors`, `.tcells`, the
   sparse `.fil` and its origins with the event id (6a).
6. The extractor as a transient pass over three overlapping regions, writing the sparse `.fil`,
   the noise snippets and covariance, and the artifact mask (6a); Klusters reading the sparse
   `.fil` in realign and re-featurize.

## 8. Phases

Each phase ends in a measured result and a negative control. A phase that fails its criterion
stops the plan there.

| # | work | passes when | negative control |
|---|---|---|---|
| 0 | **Tiling on the reference octrode, no new data.** Split g6 (29 ground-truth cells) into two tiles with a detection set wider than the sorting set (3a, scaled to 8 sites: detect on channels 0-5 and 2-7, sort on 0-3 and 4-7), then with overlapping sorting sets for comparison; assign each spike by its centroid; run the exp plan per tile; build 5.1 and 5.3. | the reconciled sort matches the whole-group sort on the GT cells (purity, completeness within run-to-run spread); reported: how many GT cells are cut by the border, the share of their spikes on the wrong side, and whether that share depends on amplitude | anchors with tile h's event ids permuted: no anchor survives |
| 1 | `fiber_probe.py`, `tile:` in `session_yaml`, `fiber-probe tiles`; tests on synthetic NP 1.0 / 2.0 site tables incl. a gapped selection and 4 shanks | every site in two tiles (ends: one), cores abut, no tile spans a gap | |
| 2 | Upstream items 1-4 on one real recording; one tile through the existing pipeline; the tile profile (section 4) | the tile opens in Klusters; knobs each have a measurement | same tile with channel order shuffled within rows: positions change, unit count does not |
| 3 | (tile, chunk) linker (5.2) on three neighbouring tiles | calibration AUC of anchors vs null reported for both kinds; false links against hand-curated tiles | linking with tile h replaced by a tile 1 mm away: no links |
| 4 | Driver, full shank, reconcile, shank-level `.res`/`.clu` | each physical spike once; unit yield and refractory statistics per depth | a second run with tiles offset by half a stride gives the same cells (the tiling is not the answer) |
| 5 | Drift field (5.4): estimate, tissue-coordinate assignment, linker prior | the field agrees with the anchor-fitted and unit-based drift; cells followed across a tile border on a recording with known imposed motion | bins shuffled within a state: no displacement beyond noise |

Phase 0 is the cheap test of the idea: if two overlapping half-octrodes cannot be put back
together, a probe will not work either.

Data for phases 2-5 (candidates, not yet checked for format or licence): the imposed-motion
recordings of Steinmetz et al. 2021; the paired juxtacellular + Neuropixels recordings of
Marques-Smith et al. for single-cell ground truth; an own recording from the lab.

## 9. Open questions

1. Is the final product the shank-level `.clu` (curation in tiles feeds it), or do the tiles stay
   authoritative and the shank-level files are exports? The plan assumes the first.
2. Curation after reconcile: a split or merge made in one tile changes cells that have fragments
   in the neighbour. Re-run 5.1-5.3 from the curated tiles, or propagate the edit?
3. A cell drifting more than half a tile: with the drift field (5.4) it has one home in tissue
   coordinates. Open only if the field proves too coarse.
4. Detection threshold and stderiv pattern per tile vs per shank (a common reference along the
   shank removes distant sources that a tile cannot see as common).
5. NP 2.0 four-shank selections spread over shanks give short runs; below about one tile height a
   run is a single group and the plan reduces to today's pipeline.

## Sources

* Windolf et al., DREDge: robust motion correction for high-density extracellular recordings across species: https://www.biorxiv.org/content/10.1101/2023.10.24.563768.full.pdf (Nature Methods 2025, https://www.nature.com/nature-index/article/10.1038/s41592-025-02614-5)

* Neuropixels probes, Allen Institute: https://allenswdb.github.io/background/neuropixels-description.html
* Steinmetz et al. 2021, Neuropixels 2.0: https://pmc.ncbi.nlm.nih.gov/articles/8244810
* fiber-kit `docs/pipeline.md`, `src/fiber_kit/{session_yaml,fiber_localize,fiber_anchor_link,fiber_link_core,fiber_cpos,backend}.py` at `b50d20e`
