"""stages.py — the one table of the stages `fiber-pipeline` can run.

A stage used to be listed in the runner (five arrays, their summaries, two dispatch tables and
the plan parser's known set), in the plan editor's catalogue and in its module map, and the lists
had drifted: the editor and `fiber-plan-lint` did not know four stages the shipped profiles run,
`fiber-pipeline --list` stopped on a summary array shorter than its stage array, and a plan step
`fiber-stochastic` was dispatched by the runner but refused by its parser.

Everything that needs the list reads it here:
  * `fiber-pipeline --list` prints `listing()`; its plan parser takes `names()` as the known set;
  * the plan editor / linter derive their module map from `module()` (their per-stage tag and
    parameter catalogue is checked against `names()` by test_stage_table);
  * the runner's own arrays and dispatch tables stay in bash and are compared with this table by
    test_stage_table, so a stage added in one place and not the other fails the suite.

Dependency-free (no numpy, no yaml): the runner imports it at start-up.

    python -m fiber_kit.stages list            # what `fiber-pipeline --list` prints
    python -m fiber_kit.stages names [group]   # one name per line
"""
import sys
from collections import namedtuple

Stage = namedtuple("Stage", "name module group summary")

# group -> heading of `fiber-pipeline --list`, in the order printed.  `core` is what `all` runs, in
# this order; the others are plannable and never part of `all`.
GROUPS = (
    ("core", "fiber-kit pipeline stages (canonical order; 'all' runs these in sequence):"),
    ("curated", "post-curation (recognized + plannable, NOT part of 'all'):"),
    ("pre", "preprocessing, before fiber-session (recognized + plannable, NOT part of 'all'):"),
    ("alt", "alternative linkers (recognized + plannable, NOT part of 'all'):"),
    ("post", "post-sort review (recognized + plannable, NOT part of 'all'):"),
)

STAGES = (
    Stage("fiber-session", "fiber_session", "core",
          "initial over-clustering -> deliberately over-split clean fragments (.clu.stderiv + .spk)"),
    Stage("fiber-realign", "fiber_realign", "core",
          "Klusters-style per-spike realign, in-place commit (re-extract raw+stderiv from .fil)"),
    Stage("fiber-refine", "fiber_refine", "core",
          "per-chunk split + fold gates + merge-back -> .refine"),
    Stage("fiber-peel", "fiber_peel", "core",
          "footprint + refractory consolidation of the refine fragments (in-place relabel on .refine)"),
    Stage("fiber-cpos", "fiber_cpos", "core",
          "per-chunk positions/templates on the refine clu, from the raw/standard .spk"),
    Stage("fiber-intrachunk", "fiber_intrachunk", "core",
          "within-chunk precision merge (depth/shape/warp/offset/refractory) + emit .units.npz"),
    Stage("fiber-link", "fiber_link", "core",
          "cross-chunk link across drift (union-find bundles) -> .refine_linked"),
    Stage("fiber-refit", "fiber_refit", "curated",
          "post-curation: rebuild the per-unit/per-chunk model from a curated .clu (no regrouping)"),
    Stage("fiber-flag-artifacts", "fiber_flag_artifacts", "pre",
          "flag acquisition artifacts from the unwhitened derivative -> a per-spike exclusion mask"),
    Stage("fiber-flag-collisions", "fiber_flag_collisions", "pre",
          "probable two-spike windows -> the same exclusion mask (collisions as 2); after fiber-flag-artifacts"),
    Stage("fiber-backbone-link", "fiber_backbone_link", "alt",
          "link fiber-session fragments on the invariant backbone (CI-overlap + warp veto); alt to intrachunk+link"),
    Stage("fiber-anchor-link", "fiber_anchor_link", "alt",
          "link fiber-session fragments seeded by the overlap anchors, on a within-session calibrated null; "
          "alt to backbone-link"),
    Stage("fiber-xcorr-merge", "fiber_xcorr_merge", "alt",
          "fuse near-identical clusters by Klusters roll-shift cosine + refractory veto; realign after each merge"),
    Stage("fiber-residual-merge", "fiber_residual_merge", "alt",
          "fuse clusters whose residual strips keep each other's spikes + refractory veto; re-template after "
          "each merge"),
    Stage("fiber-psame-merge", "fiber_psame", "alt",
          "merge clusters by a calibrated P(same cell) from history-conditioned cell models (refit + cannot-link)"),
    Stage("fiber-stochastic", "fiber_stochastic", "alt",
          "DIAGNOSTIC: ensemble/consensus fibering over resampled draws -> <SESS>.fiberens.<ELEC>.npz"),
    Stage("fiber-consensus", "fiber_consensus", "post",
          "resampled t-SNE consensus over similar clusters -> cells + island/rim quarantine clusters + .qrn "
          "review table"),
    Stage("fiber-lowmix", "fiber_lowmix", "post",
          "split low-amplitude mixture units after noise whitening -> copies returned, novel groups quarantined "
          "+ .lmx table"),
    Stage("fiber-rdr", "fiber_rdr", "post",
          "residual-derivative contamination score per cluster -> .rdr table (nothing moves)"),
    Stage("fiber-manifold", "fiber_manifold", "post",
          "off-spike-channel field manifold per unit, in sliding windows -> children (.clc/.clp) + "
          ".fet.manifold.<g> + .mft table"),
    Stage("fiber-polish", "fiber_polish", "post",
          "residual strip + pooled full-dim rkk, small-leaf t-SNE islands, reverse-strip merge -> .<in>_polish "
          "(flat .clu)"),
    Stage("fiber-unmix", "fiber_unmix", "post",
          "explain the noise cluster's spikes as kept units and their sums, on the .fil -> .<in>_unmix"),
    Stage("fiber-clean", "fiber_clean", "post",
          "core cut per cluster + quality value: cut-off spikes, dirty and too-small clusters -> noise -> "
          ".<in>_clean"),
)
_BY_NAME = {s.name: s for s in STAGES}


# ─────────────── one spelling for a stage's input, output and method: --in / --out / --method ───────────────
# A plan says `in:` and `out:`; on the command line the same two were --clu-stage or --in-stage, and
# --out-stage or --out-tag, and the method token --clu-method or --method, depending on the stage.
# IO names, per stage, the flag each of the three is an alias of.  A stage is absent from a row when it
# has no such thing (fiber-realign reads no clustering) or when the short name already means something
# else there: fiber-session, fiber-stochastic and fiber-rdr have an --out that names a file, and in the
# first two --clu-stage is the OUTPUT, so they get --in only.  The longer names keep working.
IO = {
    "fiber-session": {"--in": "--in-stage"},
    "fiber-stochastic": {"--in": "--in-stage"},
    "fiber-refine": {"--out": "--out-stage"},
    "fiber-realign": {"--out": "--out-stage"},
    "fiber-flag-artifacts": {"--out": "--out-stage"},
    "fiber-flag-collisions": {"--out": "--out-stage"},
    "fiber-rdr": {"--in": "--clu-stage", "--method": "--clu-method"},
}
for _n in ("fiber-peel", "fiber-cpos", "fiber-intrachunk", "fiber-link", "fiber-refit", "fiber-backbone-link",
           "fiber-anchor-link", "fiber-xcorr-merge", "fiber-residual-merge", "fiber-psame-merge"):
    IO[_n] = {"--in": "--clu-stage", "--out": "--out-stage", "--method": "--clu-method"}
for _n in ("fiber-consensus", "fiber-lowmix", "fiber-manifold", "fiber-polish", "fiber-unmix", "fiber-clean"):
    IO[_n] = {"--in": "--clu-stage", "--out": "--out-tag", "--method": "--clu-method"}
del _n


def add_io_aliases(ap, stage):
    """Give the parser `ap` of `stage` the short names of IO, as further spellings of the flags they
    stand for.  Call it after the arguments are added and before parse_args."""
    acts = ap._option_string_actions
    for short, full in IO.get(stage, {}).items():
        act = acts.get(full)
        if act is None:
            raise KeyError(f"{stage}: stages.IO names {full}, which the stage does not take")
        if short in acts:
            if acts[short] is not act:
                raise KeyError(f"{stage}: {short} already means something else")
            continue
        act.option_strings.append(short)
        acts[short] = act


def names(group=None):
    """Stage names in table order; one group's when `group` is given."""
    if group is not None and group not in dict(GROUPS):
        raise KeyError(f"no stage group {group!r} (groups: {', '.join(g for g, _ in GROUPS)})")
    return [s.name for s in STAGES if group is None or s.group == group]


def module(name):
    """The importable module of a stage, e.g. 'fiber_kit.fiber_psame' for fiber-psame-merge."""
    return "fiber_kit." + _BY_NAME[name].module


def listing():
    """The text of `fiber-pipeline --list`."""
    out = []
    for g, head in GROUPS:
        out.append(head)
        out += [f"  {s.name:<22s} {s.summary}" for s in STAGES if s.group == g]
    return "\n".join(out)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["list"] and len(argv) == 1:
        print(listing())
        return 0
    if argv[:1] == ["names"] and len(argv) <= 2:
        try:
            print("\n".join(names(*argv[1:])))
        except KeyError as e:
            print(f"stages: {e.args[0]}", file=sys.stderr)
            return 2
        return 0
    print("usage: python -m fiber_kit.stages list | names [group]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
