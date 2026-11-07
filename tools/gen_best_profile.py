#!/usr/bin/env python3
"""Generate src/fiber_kit/fiber-kit-best.yaml -- the tuned profile -- from fiber-kit-exp.yaml.

fiber-kit-best.yaml is fiber-kit-exp.yaml with the fiber-tune exp-search winner applied (g6 0316,
2026-11-07: see the WINNER table below and docs/bench.md) and its knobs REORDERED BY RELEVANCE instead of
by plan position:

  [T1] TUNED     the knobs the search moved, at their tuned values (the exp value noted beside each)
  [T2] SEARCHED  the knobs the search tried that changed nothing at the winner (kept at the exp value)
  [T3] RUN       what is set per session or per machine: chunks, variant, resume, chunking, realign modes
  then the stages the plan RUNS, in plan order, each section's knobs in their exp order
  then the stages the plan does not run (kept for direct CLI runs and other plans)

The knob lines and their comments are copied verbatim from fiber-kit-exp.yaml (every knob exactly once),
so the two files document the same knobs; only the plan, the T1 values and the order differ.  Edit
fiber-kit-exp.yaml, then run this tool; run_all.py checks fiber-kit-best.yaml is current (--check).

    python3 tools/gen_best_profile.py           # rewrite fiber-kit-best.yaml and fiber-kit-klusters.yaml
    python3 tools/gen_best_profile.py --check   # exit 1 when either is stale

fiber-kit-klusters.yaml is the best profile made session-agnostic for Klusters' recluster sub-sessions
(KLUSTERS_OVERRIDES, KLUSTERS_DROP_STEPS below); `fk install klusters-plugin` installs it beside the
plugin descriptor and the descriptor passes it at run time.
"""
import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
EXP = os.path.join(ROOT, "src", "fiber_kit", "fiber-kit-exp.yaml")
OUT = os.path.join(ROOT, "src", "fiber_kit", "fiber-kit-best.yaml")
OUT_KLUSTERS = os.path.join(ROOT, "src", "fiber_kit", "fiber-kit-klusters.yaml")

# ── the KLUSTERS profile: best, made session-agnostic for a recluster SUB-SESSION ──
# Klusters' Recluster with fiber-kit runs a plan on the selected clusters' spikes, exported as a
# session of their own, in ANY session folder.  So nothing in the profile may name a session: the
# three RUN knobs that do are neutralised, and the QC-report step that writes files nobody reads in a
# recluster is dropped.  Installed beside the plugin descriptor by `fk install klusters-plugin` and
# passed at run time (`--config`), so it is the one place to tune what Klusters runs.
KLUSTERS_OVERRIDES = {
    "FK_CHUNKS":              ('""', "every chunk of the sub-session: a selection is reclustered whole"),
    "FK_SESSION_SKIP_EXTANT": ("0",  "never resume: the sub-session is fresh every run"),
    "FK_REALIGN_VARIANT":     ('""', "no fixed variant: the plugin's `method` (the open clustering's) decides"),
}
KLUSTERS_DROP_STEPS = ("rdr",)      # QC table only

# ── the winner (fiber-tune exp-search, g6 0316, 2026-11-07; objective all_acc - 0.005*merges) ──
# knob -> (tuned value, exp value, what it did).  `@plan` and the psame step params live in the plan.
WINNER = {
    "FK_SESSION_CFIBER_Q":     ("0.85", "0.90", "looser within-fiber shape veto: acc 0.328 -> 0.343 with psame-chunk theta 0.7"),
    "FK_SESSION_INCL_ASSIGN":  ("1",    "0",    "spikes beyond the inclusion radius are ASSIGNED, not stripped"),
    "FK_SESSION_DIP_ALPHA":    ("0.03", "0.02", "the dip test bisects more readily"),
    "FK_CONSENSUS_THETA_CELL": ("5.0",  "7.0",  "a lower consensus cell threshold: more clusters are cells, fewer go to quarantine"),
    "FK_XCM_COS_THR":          ("0.94", "0.96", "the roll-shift-cosine merge accepts slightly less similar pairs"),
}
PLAN_WINNER = {   # step params of the plan (fiber-tune `<step>:<param>` knobs)
    "psame:theta": ("0.9", "the final P(same) merge threshold; the largest single gain: acc 0.343 -> 0.375"),
    "psame:chunk-aware": ("1", "chunk-aware history-conditioned cell models"),
    "@plan": ("no-psame-chunk", "the per-chunk P(same) merge before the linker is DROPPED (acc 0.375 -> 0.409 "
                                "once psame:theta was 0.9; the two P(same) merges interact)"),
}
# tried by the search, inert at the winner (kept at the exp value)
SEARCHED_INERT = ["FK_SESSION_INCLUSION_K", "FK_SESSION_DECONV_ASSIGN", "FK_ALINK_AMP_THR", "FK_ALINK_MAX_GAP"]
# what a user sets per session / machine, in the order worth reading
RUN_CONTROL = ["FK_CHUNKS", "FK_REALIGN_VARIANT", "FK_SESSION_SKIP_EXTANT", "FK_SESSION_CHUNK_MIN",
               "FK_SESSION_OVERLAP_MIN", "FK_SESSION_REALIGN", "FK_REALIGN_FINAL_MODE", "FK_REALIGN_AFTER_MODE",
               "FK_REALIGN_AFTER", "FK_SESSION_WHITEN_SOURCE", "FK_REFINE_CHUNK_JOBS"]
# the plan's stages, in plan order, as (section id, subsection name or None)
IN_PLAN = [("B", "fiber-flag-artifacts"), ("1", None), ("6", None), ("2", None), ("C", None), ("3", None),
           ("5", None), ("B", "fiber-xcorr-merge"), ("F", None), ("E", None)]
NOT_IN_PLAN = [("4", None), ("A", None), ("B", "fiber-refine"), ("B", "fiber-peel"), ("B", "fiber-backbone-link"),
               ("B", "fiber-stochastic"), ("D", None)]

BANNER_OPEN = re.compile(r"^# =+\s*$")
BANNER = re.compile(r"^#\s+\[([A-Z0-9])\]\s+(.*\S)\s*$")
SUB = re.compile(r"^#\s*-{3,}\s+(fiber-[a-z-]+|intrachunk|overlap-backbone)\b")
KNOB = re.compile(r"^([A-Z_][A-Z0-9_]*):(\s*)(\"(?:[^\"\\]|\\.)*\"|'[^']*'|[^\s#]+)(\s*)(#.*)?$")
CONT = re.compile(r"^\s+#")


class Knob:
    def __init__(self, name, lines, preamble, section, sub, substage):
        self.name, self.lines, self.preamble, self.section = name, lines, preamble, section
        self.sub, self.substage = sub, substage        # the sub-rule line it sits under, and that rule's stage


def parse(path):
    """The exp file's header (up to and incl. the pipeline block), and its knobs with their section /
    subsection and the free comment paragraph that precedes each (section banners and subsection rules
    are kept as text keyed by (section, sub))."""
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    # the first section banner after `pipeline:` ends the header
    i_pipe = next(i for i, ln in enumerate(lines) if ln.startswith("pipeline:"))
    i_sec = next(i for i in range(i_pipe, len(lines)) if BANNER_OPEN.match(lines[i]) and BANNER.match(lines[i + 1]))
    header = lines[:i_pipe]
    plan = lines[i_pipe:i_sec]
    knobs, banners, subs = [], {}, {}
    section = sub = substage = None
    preamble = []
    i = i_sec
    while i < len(lines):
        ln = lines[i]
        if BANNER_OPEN.match(ln) and i + 1 < len(lines) and BANNER.match(lines[i + 1]):
            j = i + 1
            while j < len(lines) and not BANNER_OPEN.match(lines[j]):
                j += 1
            section, sub, substage = BANNER.match(lines[i + 1]).group(1), None, None
            banners[section] = lines[i:j + 1]
            preamble = []
            i = j + 1
            continue
        m = SUB.match(ln)
        if m:
            if m.group(1) in ("intrachunk", "overlap-backbone"):      # intrachunk's own sub-rules stay in [4]
                preamble.append(ln)
            else:
                sub, substage = ln, m.group(1)
                subs[(section, sub)] = [ln]
                preamble = []
            i += 1
            continue
        m = KNOB.match(ln)
        if m:
            block = [ln]
            j = i + 1
            while j < len(lines) and CONT.match(lines[j]):
                block.append(lines[j]); j += 1
            knobs.append(Knob(m.group(1), block, preamble, section, sub, substage))
            preamble = []
            i = j
            continue
        if ln.startswith("#"):
            preamble.append(ln)
        elif not ln.strip():
            if preamble and preamble[-1] != "":
                preamble.append("")
        else:
            raise SystemExit(f"{path}:{i + 1}: unexpected line: {ln!r}")
        i += 1
    names = [k.name for k in knobs]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise SystemExit(f"{path}: knob listed twice: {', '.join(dup)}")
    return header, plan, knobs, banners, subs


def retune(knob, value, exp_value, why):
    """The knob block with its value replaced, the comment column kept, and a TUNED line before it."""
    first = knob.lines[0]
    m = KNOB.match(first)
    name, pad, old, gap, comment = m.groups()
    comment = comment or ""
    width = len(pad) + len(old) + len(gap)
    body = f"{pad}{value}"
    body += " " * max(1, width - len(body)) if comment else ""
    note = f"# TUNED {exp_value or old} -> {value}: {why}"
    return [note, f"{name}:{body}{comment}"] + knob.lines[1:]


def plan_block(plan_lines, drop_steps=()):
    """The exp plan with the winner applied: psame-chunk dropped, the final psame step's params set;
    `drop_steps` (by name) commented out for the klusters profile."""
    out = []
    for ln in plan_lines:
        if any(f"name: {d}," in ln for d in drop_steps) and ln.lstrip().startswith("- {"):
            out.append("  # " + ln.strip() + "   # DROPPED in the klusters profile: a QC table nobody reads in a recluster")
            continue
        if "name: psame-chunk," in ln:
            out.append("  # psame-chunk (fiber-psame-merge per chunk before the linker) is DROPPED here: the exp-search's")
            out.append("  # second pass took it out once psame:theta was 0.9 (acc 0.375 -> 0.409; the two P(same) merges")
            out.append("  # interact).  To restore it, re-add the step and point `link` at fs_psc:")
            out.append("  # " + ln.strip())
            continue
        if "name: link," in ln and "in: fs_psc" in ln:
            out.append(ln.replace("in: fs_psc, ", "in: fs_clean, "))
            continue
        if "name: psame," in ln and "out: exp_psame}" in ln:
            ln = ln.replace("out: exp_psame}", "out: exp_psame, params: {theta: 0.9, chunk-aware: 1}}")
            out.append(ln.replace("# section [F]", "# section [F]; TUNED theta 0.9, chunk-aware 1 (exp: the stage's defaults)"))
            continue
        if ln.startswith("  # without the per-chunk P(same) merge (the 0.58.3 plan): drop psame-chunk and"):
            continue
        if ln.startswith("  # - {stage: fiber-anchor-link,    name: link,            in: fs_clean,"):
            continue
        out.append(ln)
    return out


HEADER = """# ============================================================================
#  fiber-kit-best.yaml  —  TUNED fiber-kit config: the fiber-tune exp-search winner.
#
#  GENERATED by tools/gen_best_profile.py from fiber-kit-exp.yaml: the same knobs, the same comments,
#  with the winner of `fiber-tune search` on sirotaA-jg-000005-20120316 group 6 (2026-11-07; objective
#  all_acc - 0.005*merges, scored at the final psame step, chunks 2-3, 109 trials in 2 passes; truth = the
#  curated .clu.stderiv_C6_D34.6, 33 units) applied, and the knobs ordered by RELEVANCE rather than by
#  plan position.  Do not edit this file by hand: edit fiber-kit-exp.yaml (or this tool) and regenerate.
#
#  WINNER vs fiber-kit-exp.yaml (accuracy 0.328 -> 0.436, pyramidal 0.324 -> 0.437, purity 0.724 ->
#  0.780; interneuron accuracy 0.468 -> 0.389 and completeness 0.725 -> 0.675 -- the sort is cleaner and
#  less complete, see docs/bench.md "Clean-core metrics" for the objective that asks for cores):
#      @plan                    no-psame-chunk   the per-chunk P(same) step before the linker is dropped
#      psame:theta              0.9              (plan step `psame`, params)
#      psame:chunk-aware        1
#      FK_SESSION_CFIBER_Q      0.90 -> 0.85
#      FK_SESSION_INCL_ASSIGN   0    -> 1
#      FK_SESSION_DIP_ALPHA     0.02 -> 0.03
#      FK_CONSENSUS_THETA_CELL  7.0  -> 5.0
#      FK_XCM_COS_THR           0.96 -> 0.94
#  Tuned on two 18-min chunks (~36 min); steps under ~0.005 are within chunk noise.  Not yet confirmed
#  with `fiber-tune confirm --chunks all` (the full plan with realign, every chunk).
#
#  USE.  In a session folder:   fk set config --profile best     (writes ./fiber-kit.yaml, this file)
#        then                   fiber-pipeline <elec>   or   fk run <elec>
#        fk list config         lists the shipped profiles and which one ./fiber-kit.yaml is.
#  Or point FK_CONFIG at this file.  PRECEDENCE (highest first):
#      explicit CLI flag  >  exported FK_* env var  >  THIS FILE  >  <session>.yaml
#      fiber_kit.<stage> section  >  built-in default
#
#  LAYOUT -- by relevance:
#    [T1] TUNED      the knobs the search moved (value, and the exp value it replaced)
#    [T2] SEARCHED   tried by the search, inert at the winner (exp values)
#    [T3] RUN        set per session / machine: chunks, variant, resume, chunking, realign modes
#    IN THE PLAN, in plan order:  flag [B] -> session [1] + in-stage curator [6] -> realign [2]
#        -> consensus [C] -> cpos [3] -> anchor-link [5] -> xcorr-merge [B] -> psame [F] -> rdr [E]
#    NOT IN THE PLAN (direct CLI runs, other plans):  intrachunk [4], fiber-link [A], refine / peel /
#        backbone-link / stochastic [B], lowmix [D]
#  The section tags [1]..[F] are fiber-kit-exp.yaml's, so docs/config.md's sections apply to both files.
#  Each comment gives:  DEFAULT | useful RANGE | the reasoning.
# ============================================================================


# ============================================================================
#  PRIMARY PLAN (embedded) -- fiber-kit-exp.yaml's plan WITHOUT the per-chunk P(same) step
#  (`@plan no-psame-chunk`), the final `psame` step at theta 0.9, chunk-aware.  `fiber-pipeline <elec>`
#  runs this by default when this file is the active config; `fk list` names the steps.
#    flag -> session -> realign-1 -> clean (consensus, quarantine-only) -> realign-2 -> cpos-clean
#      -> link (anchor-link, units) -> realign-3 -> merge (xcorr) -> psame -> realign-final -> rdr
# ============================================================================
"""


def tier_banner(tag, title, body):
    rule = "# " + "=" * 76
    out = [rule, f"#  [{tag}] {title}"]
    out += ["#  " + b if b else "#" for b in body]
    out.append(rule)
    return out


KLUSTERS_HEADER = """# ============================================================================
#  fiber-kit-klusters.yaml  \u2014  the KLUSTERS RECLUSTER profile: best, made session-agnostic.
#
#  GENERATED by tools/gen_best_profile.py from fiber-kit-exp.yaml: fiber-kit-best.yaml (the fiber-tune
#  exp-search winner, knobs ordered by relevance -- see that file's header for the winner and the layout)
#  with three RUN knobs neutralised and the QC-report step dropped, because this profile runs on a
#  SUB-SESSION: the clusters Klusters' curator selected, exported as a session of their own, in ANY
#  session folder (Klusters' Recluster with fiber-kit / the fiber-kit-recluster plugin; fiber-kit's
#  docs/pipeline.md "Reclustering a selection").  Nothing here may name a session:
#      FK_CHUNKS               ""     every chunk of the sub-session (a selection is reclustered whole)
#      FK_SESSION_SKIP_EXTANT  0      never resume: the sub-session is fresh every run
#      FK_REALIGN_VARIANT      ""     the plugin's `method` (the open clustering's variant) decides
#      plan                    fiber-rdr dropped (a QC table nobody reads in a recluster)
#
#  WHERE IT LIVES.  `fk install klusters-plugin` copies this file next to the plugin descriptor in
#  Klusters' plugin directory (~/.local/share/neurosuite-3/klusters/plugins/descriptions/) and points the
#  descriptor's `config` parameter at that copy, so Klusters passes it at run time (`fk run --config`)
#  whatever session is open.  EDIT THAT COPY to change what Klusters runs on a selection; the session
#  folder's fiber-kit.yaml is not read by the plugin.  `fk set config --profile klusters` writes it as a
#  session config too, and `fk run --profile klusters` runs it directly.
#  PRECEDENCE (highest first):  explicit CLI flag / plan params  >  exported FK_* env var  >  THIS FILE
#      >  <session>.yaml  fiber_kit.<stage> section  >  built-in default
# ============================================================================


# ============================================================================
#  PRIMARY PLAN (embedded) -- the best profile's plan without fiber-rdr:
#    flag -> session -> realign-1 -> clean (consensus, quarantine-only) -> realign-2 -> cpos-clean
#      -> link (anchor-link, units) -> realign-3 -> merge (xcorr) -> psame -> realign-final
#  The final .clu (and .clc/.clp) is what Klusters reads back (--final-clu).
# ============================================================================
"""


def render(target="best"):
    header, plan, knobs, banners, subs = parse(EXP)
    by_name = {k.name: k for k in knobs}
    for n in list(WINNER) + SEARCHED_INERT + RUN_CONTROL + list(KLUSTERS_OVERRIDES):
        if n not in by_name:
            raise SystemExit(f"gen_best_profile: {n} is not a knob of fiber-kit-exp.yaml")
    klusters = (target == "klusters")
    used = set()
    out = (KLUSTERS_HEADER if klusters else HEADER).rstrip("\n").split("\n")
    out += plan_block(plan, KLUSTERS_DROP_STEPS if klusters else ())
    while out and not out[-1].strip():                      # the plan, without its trailing blanks
        out.pop()
    out += ["", ""]

    def emit(knob, tuned=None):
        used.add(knob.name)
        block = []
        if knob.preamble:
            block += list(knob.preamble)
        if tuned:
            block += retune(knob, *tuned)
        else:
            block += knob.lines
        return block

    # T1
    out += tier_banner("T1", "TUNED -- the knobs the exp-search moved (fiber-tune, g6 0316, 2026-11-07)", [
        "Each line carries the exp value it replaced and what the move did to the accuracy objective.  The",
        "plan-level part of the winner (no psame-chunk step; psame theta 0.9, chunk-aware 1) is in the plan above."])
    for n, (v, ev, why) in WINNER.items():
        k = by_name[n]
        out += retune(k, v, ev, why); used.add(n)
    out.append("")
    # T2
    out += tier_banner("T2", "SEARCHED -- tried by the exp-search, no effect at the winner (exp values kept)", [
        "FK_SESSION_INCLUSION_K and DECONV_ASSIGN are inert with INCL_ASSIGN 1 (nothing is stripped for them to",
        "act on); the anchor-link amplitude threshold and max gap did not move the objective.  The psame step's",
        "knn (4/8) and cannot-link (0.2/0.3) params were inert too; they are the stage's defaults in the plan."])
    for n in SEARCHED_INERT:
        out += emit(by_name[n])
    out.append("")
    # T3
    out += tier_banner("T3", "RUN -- set per session / machine", [
        "Which chunks run, the waveform variant the sort realigns and writes, whether an existing fiber-session",
        "output is resumed, the chunk grid the drift model assumes, and the realign modes.  Everything a run",
        "of THIS profile on a new session may need changed; nothing below this section usually does."])
    for n in RUN_CONTROL:
        if klusters and n in KLUSTERS_OVERRIDES:
            v, why = KLUSTERS_OVERRIDES[n]
            k = by_name[n]
            block = retune(k, v, None, why)
            m = KNOB.match(k.lines[0])
            block[0] = (f"# KLUSTERS kept {v}: {why}" if m.group(3) == v
                        else block[0].replace("# TUNED ", "# KLUSTERS ", 1))
            out += block; used.add(n)
        else:
            out += emit(by_name[n])
    out.append("")

    def section(sid, sub, title_note):
        """One output section: a whole exp section (sub None: every knob of it, its own sub-rules kept) or
        one named subsection of [B]."""
        whole = sub is None
        rows = [k for k in knobs if k.section == sid and (whole or k.substage == sub) and k.name not in used]
        if not rows and sid != "3":
            return []
        blk = ["", ""]
        if whole:
            ban = list(banners[sid])
            ban.insert(2, f"#  {title_note}")
            blk += ban
        else:
            blk += [f"# ----- [{sid}] {sub} : {title_note} " + "-" * max(3, 60 - len(sub) - len(title_note))]
        moved = [k.name for k in knobs if k.section == sid and (whole or k.substage == sub) and k.name in used]
        if moved:
            blk.append(f"# (listed above, in [T1]-[T3]: {', '.join(moved)})")
        last_sub = None
        for k in rows:
            if whole and k.sub != last_sub:
                if k.sub is not None:
                    blk += [""] + subs[(sid, k.sub)]
                last_sub = k.sub
            blk += emit(k)
        return blk

    out += ["", "# " + "#" * 76, "#  STAGES THE PLAN RUNS, in plan order", "# " + "#" * 76]
    notes = {("B", "fiber-flag-artifacts"): "plan step `flag`", ("1", None): "plan step `session`",
             ("6", None): "in-stage curator of `session`", ("2", None): "plan steps `realign-1..3`, `realign-final`",
             ("C", None): "plan step `clean`", ("3", None): "plan step `cpos-clean`", ("5", None): "plan step `link`",
             ("B", "fiber-xcorr-merge"): "plan step `merge`",
             ("F", None): "plan step `psame`; its params {theta: 0.9, chunk-aware: 1} in the plan OVERRIDE FK_PSAME_THETA / _CHUNK_AWARE below",
             ("E", None): "plan step `rdr`"}
    for sid, sub in IN_PLAN:
        out += section(sid, sub, "IN THE PLAN: " + notes[(sid, sub)])
    out += ["", "", "# " + "#" * 76, "#  STAGES NOT IN THE PLAN (direct CLI runs and other plans)", "# " + "#" * 76]
    for sid, sub in NOT_IN_PLAN:
        out += section(sid, sub, "NOT IN THE PLAN")
    # anything the two lists did not place (a new section in the exp file) goes last, verbatim
    rest = [k for k in knobs if k.name not in used]
    if rest:
        out += ["", "", "# ----- other knobs of fiber-kit-exp.yaml (not placed by tools/gen_best_profile.py) -----"]
        for k in rest:
            out += emit(k)
    text = "\n".join(out).rstrip("\n") + "\n"
    return text, knobs


def main(argv=None):
    ap = argparse.ArgumentParser(prog="gen_best_profile")
    ap.add_argument("--check", action="store_true", help="exit 1 when fiber-kit-best.yaml / -klusters.yaml are stale")
    a = ap.parse_args(argv)
    rc = 0
    for target, out_path in (("best", OUT), ("klusters", OUT_KLUSTERS)):
        text, knobs = render(target)
        name = os.path.relpath(out_path, ROOT)
        if a.check:
            cur = ""
            if os.path.exists(out_path):
                with open(out_path, encoding="utf-8") as fh:
                    cur = fh.read()
            if cur != text:
                print(f"DRIFT: {name} does not match fiber-kit-exp.yaml -- run tools/gen_best_profile.py")
                rc = 1
            else:
                print(f"OK: {name} is current ({len(knobs)} knobs)")
            continue
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"wrote {name} ({len(knobs)} knobs, {len(text.splitlines())} lines)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
