#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
#  config.py — typed, metadata-driven stage configs for the fiber-kit pipeline.
#
#  One dataclass per stage holds every tunable knob with its default, help, unit,
#  valid choices and env-var name as FIELD METADATA — a single source of truth that
#  drives (a) argparse generation, (b) the layered resolution, and (c) the YAML
#  template emission.  No more 60-parameter signatures or hand-maintained YAML
#  comments that drift from the code.
#
#  Resolution precedence (highest first):
#      explicit CLI flag  >  FK_* env var  >  GLOBAL fiber-kit.yaml  >  <session>.yaml fiber_kit.<stage>  >  field default
#  The global fiber-kit.yaml (a flat FK_* map at $FK_CONFIG or ./fiber-kit.yaml) OVERRIDES the
#  per-session tuning when present -- and resolve() auto-loads it, so the override holds on a
#  direct stage call, not only when run through scripts/fiber-pipeline (which exports it as env).
#  A session's own tuning still lives in its <session>.yaml (the ndmanager-plugins convention: a
#  program's parameters travel with the session) for when no global file is present; env/CLI stay
#  available for one-off sweeps.
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import os
import re
from dataclasses import dataclass, field, fields as _dc_fields


_UNSET = object()


# ── reading FK_* knobs ───────────────────────────────────────────────────────
# One precedence everywhere:  CLI flag / plan params  >  FK_* environment variable  >  $FK_CONFIG yaml
# >  built-in default.  knob_value() is the single implementation; the stages' _knob_default helpers
# delegate to it.
#
# YAML 1.1 (PyYAML's default) reads a bare  off / on / yes / no  as a boolean, so `FK_SESSION_REALIGN:
# off` used to arrive as False -> "0" (pipeline) or "False" (direct call), neither of which is "off", and
# the realign ran.  yaml_load() reads FK configs with YAML 1.2 booleans: only true/false are booleans,
# off/on/yes/no stay the words that were written.

_OFF_WORDS = ("off", "0", "false", "no", "none")
_ON_WORDS = ("on", "1", "true", "yes")


class KnobError(SystemExit):
    """A knob value that cannot be used (not one of the choices, not a number).  A SystemExit so a stage
    stops with the message instead of a traceback, before it reads or writes anything."""

    def __init__(self, msg):
        super().__init__(f"fiber-kit: {msg}")


def yaml_loader():
    """A SafeLoader with YAML 1.2 booleans (true/false only)."""
    import yaml

    class _FKLoader(yaml.SafeLoader):
        pass
    _FKLoader.yaml_implicit_resolvers = {
        k: [(tag, rx) for tag, rx in v if tag != "tag:yaml.org,2002:bool"]
        for k, v in yaml.SafeLoader.yaml_implicit_resolvers.items()}
    _FKLoader.add_implicit_resolver("tag:yaml.org,2002:bool",
                                    re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"), list("tTfF"))
    return _FKLoader


def yaml_load(stream):
    """yaml.safe_load with YAML 1.2 booleans -- for every FK config, plan and --set value."""
    import yaml
    return yaml.load(stream, Loader=yaml_loader())  # noqa: S506 -- a SafeLoader subclass


def normalize_choice(v, choices):
    """Map an off/on spelling (False, "0", "no", ...) onto the choice that means it: "off" when the
    choices have it, else "" (several knobs spell off as the empty string), and "on" for on-words.
    Other values are returned as strings, unchanged."""
    if isinstance(v, bool):
        v = "true" if v else "false"
    s = str(v).strip()
    if s in choices:
        return s
    low = s.lower()
    if low in _OFF_WORDS:
        if "off" in choices:
            return "off"
        if "" in choices:
            return ""
    if low in _ON_WORDS and "on" in choices:
        return "on"
    return s


def coerce_knob(name, v, typ=str, choices=None, where=""):
    """Convert one knob value to `typ`, validating it.  Raises KnobError naming the knob, the value and
    where it came from, instead of a stage silently running something else."""
    src = f" (from {where})" if where else ""
    if choices is not None:
        s = normalize_choice(v, choices)
        if s not in choices:
            raise KnobError(f"{name}={v!r}{src} is not one of: " + ", ".join(repr(c) for c in choices))
        return s
    if isinstance(v, bool):
        v = int(v)
    if typ is int:
        try:
            f = float(v)
        except (TypeError, ValueError):
            raise KnobError(f"{name}={v!r}{src} is not an integer") from None
        if f != f or f in (float("inf"), float("-inf")) or f != int(f):   # nan, inf, 6.7
            raise KnobError(f"{name}={v!r}{src} is not an integer")
        return int(f)
    if typ is str:
        return str(v)
    try:
        return typ(v)
    except (TypeError, ValueError):
        tn = getattr(typ, "__name__", str(typ))
        raise KnobError(f"{name}={v!r}{src} cannot be read as {tn}") from None


def _blank(v):
    return v is None or (isinstance(v, str) and v.strip() == "")


def knob_value(name, typ=str, fallback=None, gcfg=None, choices=None, env=None):
    """An FK_* knob's value: the environment variable, then the $FK_CONFIG yaml (`gcfg`; loaded when
    None), then `fallback`.  A blank value means "not set" at that level.  The value is converted and
    validated by coerce_knob; `fallback` is returned as given."""
    env = os.environ if env is None else env
    v = env.get(name)
    if not _blank(v):
        return coerce_knob(name, v, typ, choices, "the environment")
    gcfg = load_global_config() if gcfg is None else gcfg
    v = (gcfg or {}).get(name)
    if not _blank(v):
        return coerce_knob(name, v, typ, choices, "$FK_CONFIG")
    return fallback


def check_choice_defaults(ap):
    """argparse never checks a `default=` against `choices=`, and these defaults come from FK_* knobs.
    Normalise an off/on spelling and refuse any other value outside the choices, before parsing."""
    for act in ap._actions:
        if not act.choices or act.default in (None, argparse.SUPPRESS):
            continue
        d = normalize_choice(act.default, tuple(act.choices))
        if d not in act.choices:
            ap.error(f"{'/'.join(act.option_strings)}: default {act.default!r} (from an FK_* knob) is not one "
                     "of: " + ", ".join(repr(c) for c in act.choices))
        act.default = d


# ───────────────── knobs the runner passes as flags: a stage run by hand reads them too ─────────────────
# scripts/fiber-pipeline reads these knobs into shell variables and hands them to the stage as flags
# (the older route; the newer stages read their FK_* knobs themselves).  A stage started by hand never
# saw them: `fiber-session <base> 8` beside a profile ran on its argparse defaults.  The table says,
# per stage, which flag a knob becomes:
#     "value"  --flag <the knob's value>
#     "on"     --flag when the knob is on            "off"   --flag when the knob is off
# apply_profile_knobs() makes those the parser's DEFAULTS, so a flag on the command line -- the
# runner's included -- still wins, and a knob that is set nowhere leaves the stage's own default.
# test_runner_knobs keeps the table equal to what the runner's stage functions pass.
RUNNER_KNOBS = {
    "fiber-session": (
        ("--eband-width", "FK_SESSION_EBAND_WIDTH", "value"),
        ("--eband-overlap", "FK_SESSION_EBAND_OVERLAP", "value"),
        ("--eband-confound", "FK_SESSION_EBAND_CONFOUND", "value"),
        ("--eband-min-span", "FK_SESSION_EBAND_MIN_SPAN", "value"),
        ("--eband-low-assign", "FK_SESSION_EBAND_LOW_ASSIGN", "value"),
        ("--cfiber-q", "FK_SESSION_CFIBER_Q", "value"),
        ("--dip-dim", "FK_SESSION_DIP_DIM", "value"),
        ("--dip-alpha", "FK_SESSION_DIP_ALPHA", "value"),
        ("--dip-min", "FK_SESSION_DIP_MIN", "value"),
        ("--keep-small-min", "FK_SESSION_KEEP_SMALL_MIN", "value"),
        ("--fine-algo", "FK_SESSION_FINE_METHOD", "value"),
        ("--chunk-minutes", "FK_SESSION_CHUNK_MIN", "value"),
        ("--merge-algo", "FK_SESSION_MERGE_METHOD", "value"),
        ("--merge-corr", "FK_SESSION_MERGE_CORR", "value"),
        ("--coarse-dr", "FK_SESSION_COARSE_DR", "value"),
        ("--seed-density", "FK_SESSION_SEED_DENSITY", "value"),
        ("--reseed-residual-thr", "FK_SESSION_RESEED_RESIDUAL_THR", "value"),
        ("--reseed-min-reduction", "FK_SESSION_RESEED_MIN_REDUCTION", "value"),
        ("--refit-iters", "FK_SESSION_REFIT_ITERS", "value"),
        ("--resplit-passes", "FK_SESSION_RESPLIT_PASSES", "value"),
        ("--resplit-residual-thr", "FK_SESSION_RESPLIT_RESIDUAL_THR", "value"),
        ("--resplit-topch", "FK_SESSION_RESPLIT_TOPCH", "value"),
        ("--resplit-min-reduction", "FK_SESSION_RESPLIT_MIN_REDUCTION", "value"),
        ("--resplit-merge-corr", "FK_SESSION_RESPLIT_MERGE_CORR", "value"),
        ("--resplit-detrend-win", "FK_SESSION_RESPLIT_DETREND_WIN", "value"),
        ("--resplit-detrend-min-n", "FK_SESSION_RESPLIT_DETREND_MIN_N", "value"),
        ("--feature-align", "FK_SESSION_FEATURE_ALIGN", "value"),
        ("--inclusion-k", "FK_SESSION_INCLUSION_K", "value"),
        ("--realign", "FK_SESSION_REALIGN", "value"),
        ("--realign-algo", "FK_REALIGN_ALGO", "value"),
        ("--energy-band", "FK_EBAND", "on"),
        ("--cfiber-gate", "FK_SESSION_CFIBER_GATE", "on"),
        ("--dipsplit", "FK_SESSION_DIPSPLIT", "on"),
        ("--incl-assign-rejected", "FK_SESSION_INCL_ASSIGN", "on"),
        ("--no-noise", "FK_SESSION_NO_NOISE", "on"),
        ("--keep-small-groups", "FK_SESSION_KEEP_SMALL", "on"),
        ("--assign-unsorted", "FK_SESSION_ASSIGN_UNSORTED", "on"),
        ("--deconv", "FK_SESSION_DECONV", "on"),
        ("--deconv-assign", "FK_SESSION_DECONV_ASSIGN", "on"),
        ("--no-rkk-delete", "FK_SESSION_RKK_DELETE", "off"),
        ("--no-link", "FK_SESSION_LINK", "off"),
        ("--no-dipsplit", "FK_SESSION_DIPSPLIT", "off"),
    ),
    "fiber-realign": (
        ("--variant", "FK_REALIGN_VARIANT", "value"),
        ("--align-algo", "FK_REALIGN_ALGO", "value"),
    ),
    "fiber-refine": (
        ("--merge-warp-recall", "FK_MERGE_WARP_RECALL", "value"),
        ("--merge-amp-thr", "FK_MERGE_AMP_THR", "value"),
        ("--merge-warp-thr", "FK_MERGE_WARP_THR", "value"),
        ("--merge-warp-resid-thr", "FK_MERGE_WARP_RESID_THR", "value"),
        ("--merge-warp-resid-thr-int", "FK_MERGE_WARP_RESID_THR_INT", "value"),
        ("--merge-warp-resid-thr-pyr", "FK_MERGE_WARP_RESID_THR_PYR", "value"),
        ("--ab-distinct", "FK_REFINE_AB_DISTINCT", "value"),
        ("--ab-abs", "FK_REFINE_AB_ABS", "value"),
        ("--ab-margin", "FK_REFINE_AB_MARGIN", "value"),
        ("--ab-min", "FK_REFINE_AB_MIN", "value"),
        ("--ab-sigcap", "FK_REFINE_AB_SIGCAP", "value"),
        ("--ab-jobs", "FK_REFINE_AB_JOBS", "value"),
        ("--large", "FK_REFINE_LARGE", "value"),
        ("--min-group", "FK_REFINE_MIN_GROUP", "value"),
        ("--drop-min", "FK_REFINE_DROP_MIN", "value"),
        ("--feature-align", "FK_SESSION_FEATURE_ALIGN", "value"),
        ("--merge-min-sim", "FK_REFINE_MERGE_MIN_SIM", "value"),
        ("--merge-budget", "FK_REFINE_MERGE_BUDGET", "value"),
        ("--split-min-corr", "FK_REFINE_SPLIT_MIN_CORR", "value"),
        ("--split-var-mult", "FK_REFINE_SPLIT_VAR_MULT", "value"),
        ("--fold-off-thr", "FK_REFINE_FOLD_OFF_THR", "value"),
        ("--chunk-minutes", "FK_REFINE_CHUNK_MINUTES", "value"),
        ("--chunk-jobs", "FK_REFINE_CHUNK_JOBS", "value"),
        ("--dedup", "FK_REFINE_DEDUP", "on"),
        ("--ab-reclaim", "FK_REFINE_AB_RECLAIM", "on"),
        ("--no-rkk-delete", "FK_REFINE_RKK_DELETE", "off"),
    ),
    "fiber-peel": (
        ("--foot-hi", "FK_PEEL_FOOT_HI", "value"),
        ("--foot-lo", "FK_PEEL_FOOT_LO", "value"),
        ("--refrac-ms", "FK_PEEL_REFRAC_MS", "value"),
        ("--refrac-thr", "FK_PEEL_REFRAC_THR", "value"),
        ("--refrac-min-exp", "FK_PEEL_REFRAC_MIN_EXP", "value"),
        ("--refrac-censor-ms", "FK_PEEL_REFRAC_CENSOR_MS", "value"),
        ("--min-n", "FK_PEEL_MIN_N", "value"),
        ("--burst-foot-lo", "FK_PEEL_BURST_FOOT_LO", "value"),
    ),
    "fiber-intrachunk": (
        ("--backbone-std-cos", "FK_INTRA_BACKBONE_STD_COS", "value"),
        ("--backbone-warp", "FK_INTRA_BACKBONE_WARP", "value"),
    ),
    "fiber-link": (
        ("--warp-thr", "FK_LINK_WARP_THR", "value"),
        ("--warp-amp-thr", "FK_LINK_WARP_AMP_THR", "value"),
        ("--warp-resid-thr", "FK_LINK_WARP_RESID_THR", "value"),
        ("--cfiber-q", "FK_LINK_CFIBER_Q", "value"),
        ("--tan-thr", "FK_LINK_TAN_THR", "value"),
        ("--bundle", "FK_LINK_BUNDLE", "value"),
        ("--var-scale", "FK_LINK_VAR_SCALE", "value"),
        ("--n-pc", "FK_LINK_N_PC", "value"),
        ("--var-allow", "FK_LINK_VAR_ALLOW", "value"),
        ("--dr-k", "FK_LINK_DR_K", "value"),
        ("--dr-thr", "FK_LINK_DR_THR", "value"),
        ("--cos-thr", "FK_LINK_COS_THR", "value"),
        ("--pos-thr", "FK_LINK_POS_THR", "value"),
        ("--off-thr", "FK_LINK_OFF_THR", "value"),
        ("--max-gap", "FK_LINK_MAX_GAP", "value"),
        ("--amp-gate", "FK_LINK_AMP_GATE", "value"),
        ("--min-a", "FK_LINK_MIN_A", "value"),
        ("--amp-span", "FK_LINK_AMP_SPAN", "value"),
        ("--align-lag", "FK_LINK_ALIGN_LAG", "value"),
        ("--align-upsample", "FK_LINK_ALIGN_UPSAMPLE", "value"),
        ("--primary-amp-frac", "FK_LINK_PRIMARY_AMP_FRAC", "value"),
    ),
    "fiber-flag-artifacts": (
        ("--spk-method", "FK_FLAG_SPK", "value"),
        ("--out-stage", "FK_FLAG_STAGE", "value"),
    ),
    "fiber-flag-collisions": (
        ("--spk-method", "FK_FLAG_SPK", "value"),
        ("--out-stage", "FK_FLAG_STAGE", "value"),
    ),
    "fiber-stochastic": (
        ("--fine-algo", "FK_SESSION_FINE_METHOD", "value"),
        ("--chunk-minutes", "FK_SESSION_CHUNK_MIN", "value"),
        ("--merge-algo", "FK_SESSION_MERGE_METHOD", "value"),
        ("--merge-corr", "FK_SESSION_MERGE_CORR", "value"),
        ("--coarse-dr", "FK_SESSION_COARSE_DR", "value"),
        ("--seed-density", "FK_SESSION_SEED_DENSITY", "value"),
        ("--reseed-residual-thr", "FK_SESSION_RESEED_RESIDUAL_THR", "value"),
        ("--reseed-min-reduction", "FK_SESSION_RESEED_MIN_REDUCTION", "value"),
        ("--refit-iters", "FK_SESSION_REFIT_ITERS", "value"),
        ("--resplit-passes", "FK_SESSION_RESPLIT_PASSES", "value"),
        ("--resplit-residual-thr", "FK_SESSION_RESPLIT_RESIDUAL_THR", "value"),
        ("--resplit-topch", "FK_SESSION_RESPLIT_TOPCH", "value"),
        ("--resplit-min-reduction", "FK_SESSION_RESPLIT_MIN_REDUCTION", "value"),
        ("--resplit-merge-corr", "FK_SESSION_RESPLIT_MERGE_CORR", "value"),
        ("--feature-align", "FK_SESSION_FEATURE_ALIGN", "value"),
        ("--inclusion-k", "FK_SESSION_INCLUSION_K", "value"),
        ("--stochastic-draws", "FK_STOCH_DRAWS", "value"),
        ("--stochastic-frac", "FK_STOCH_FRAC", "value"),
        ("--stochastic-match-corr", "FK_STOCH_MATCH_CORR", "value"),
        ("--stochastic-stable-freq", "FK_STOCH_STABLE_FREQ", "value"),
        ("--stochastic-peel-rounds", "FK_STOCH_PEEL_ROUNDS", "value"),
        ("--stochastic-link", "FK_STOCH_LINK", "value"),
        ("--stochastic-jobs", "FK_STOCH_JOBS", "value"),
        ("--stochastic-seed", "FK_STOCH_SEED", "value"),
    ),
}


def knob_is_on(name, v, where=""):
    """A 0/1 switch as the runner's _on reads it: 1/on/yes/true and 0/off/no/false, any case."""
    if isinstance(v, bool):
        return v
    low = str(v).strip().lower()
    if low in ("1", "on", "yes", "true"):
        return True
    if low in ("0", "off", "no", "false"):
        return False
    raise KnobError(f"{name}={v!r}" + (f" (from {where})" if where else "") + " is not a switch: use 0/1")


def apply_profile_knobs(ap, stage, gcfg=None, env=None):
    """Make the knobs of RUNNER_KNOBS[stage] the defaults of the parser `ap` (see the table above).
    Call it after the arguments are added and before parse_args.  Returns the knobs applied, as
    {flag: (knob, value, where)}."""
    env = os.environ if env is None else env
    rows = RUNNER_KNOBS.get(stage, ())
    if not rows:
        return {}
    gcfg = load_global_config() if gcfg is None else gcfg
    acts = {s: a for a in ap._actions for s in a.option_strings}
    applied = {}
    for flag, name, kind in rows:
        act = acts.get(flag)
        if act is None:
            continue
        v, where = env.get(name), "the environment"
        if _blank(v):
            v, where = (gcfg or {}).get(name), "$FK_CONFIG"
        if _blank(v):
            continue
        if kind == "value":
            if isinstance(v, bool):
                v = int(v)
            if act.choices is not None:
                val = normalize_choice(v, tuple(act.choices))
                if val not in act.choices:
                    raise KnobError(f"{name}={v!r} (from {where}) is not one of: "
                                    + ", ".join(repr(c) for c in act.choices))
            elif act.type is None:
                val = str(v)
            else:
                try:
                    val = act.type(str(v))
                except (TypeError, ValueError, argparse.ArgumentTypeError):
                    tn = getattr(act.type, "__name__", str(act.type))
                    raise KnobError(f"{name}={v!r} (from {where}) cannot be read as {tn}") from None
        else:
            if knob_is_on(name, v, where) != (kind == "on"):
                continue                                     # the runner passes no flag in this state
            if isinstance(act, argparse.BooleanOptionalAction):
                val = not flag.startswith("--no-")
            elif act.nargs == 0:
                val = act.const                              # store_true / store_false / store_const
            else:
                continue                                     # a flag that takes a path: the runner's to build
        ap.set_defaults(**{act.dest: val})
        applied[flag] = (name, val, where)
    return applied


def load_global_config(path=None):
    """Load the global fiber-kit.yaml (a FLAT FK_* mapping) that OVERRIDES per-session
    tuning when present.  Search order: explicit `path`, then $FK_CONFIG, then
    ./fiber-kit.yaml in the working directory (the same file scripts/fiber-pipeline reads).
    Returns {} when there is no file.  A file that exists but cannot be parsed is an error
    (KnobError): ignoring it would run every stage on built-in defaults without a word."""
    p = path or os.environ.get("FK_CONFIG") or os.path.join(os.getcwd(), "fiber-kit.yaml")
    if not (p and os.path.isfile(p)):
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    try:
        with open(p) as fh:
            d = yaml_load(fh) or {}
    except (OSError, yaml.YAMLError) as e:
        raise KnobError(f"cannot read the config {p}: {e}") from None
    return d if isinstance(d, dict) else {}


def knob(default, help="", *, env=None, choices=None, type=float, cli=None, recommended=_UNSET):
    """A config field: `default` plus metadata. `recommended` is the value the 'pipeline' profile uses
    when nothing higher-priority is set (defaults to `default` when not given)."""
    rec = default if recommended is _UNSET else recommended
    return field(default=default, metadata=dict(help=help, env=env, choices=choices, type=type,
                                                cli=cli, recommended=rec))


class StageConfig:
    """Mixin: argparse generation + layered resolution + YAML emission, all from field metadata."""

    @classmethod
    def add_arguments(cls, ap):
        """Add one argparse flag per field, default=SUPPRESS so 'provided on the CLI' is detectable."""
        for f in _dc_fields(cls):
            m = f.metadata
            flag = "--" + (m.get("cli") or f.name.replace("_", "-"))
            kw = dict(dest=f.name, default=argparse.SUPPRESS, help=m.get("help", ""))
            if m.get("choices"):
                kw["choices"] = tuple(m["choices"])
            else:
                kw["type"] = m.get("type", float)
            ap.add_argument(flag, **kw)

    @staticmethod
    def _coerce(f, v):
        if v is None or v == "":
            return f.default                       # empty in env/yaml => fall back to the built-in default
        ch = f.metadata.get("choices")
        return coerce_knob(f.metadata.get("env") or f.name, v, f.metadata.get("type", float),
                           tuple(ch) if ch else None)

    @classmethod
    def resolve(cls, args=None, env=None, section=None, profile="default", global_cfg=None):
        """Build an instance with precedence CLI > env > GLOBAL fiber-kit.yaml > session-section > default.

        args       : the parsed argparse namespace (flags use default=SUPPRESS, so an attribute is
                     present iff the user gave it).  None => skip the CLI layer.
        env        : os.environ-like mapping; a field's env metadata names its FK_* var.
        section    : the session.yaml fiber_kit.<stage> dict; keys may be the field name OR its FK_* env name.
        global_cfg : the global fiber-kit.yaml flat FK_* map; OVERRIDES the session section when a key is set.
                     None => auto-load via load_global_config() ($FK_CONFIG or ./fiber-kit.yaml), so the
                     override holds on a direct stage call, not only when run through scripts/fiber-pipeline.
        profile    : final fallback when a knob is unset everywhere -- "recommended" uses the pipeline
                     profile baked into the field metadata, anything else uses the plain field default.
        """
        env = env or {}
        section = section or {}
        gcfg = load_global_config() if global_cfg is None else (global_cfg or {})
        vals = {}
        for f in _dc_fields(cls):
            ev = f.metadata.get("env")
            if args is not None and hasattr(args, f.name):
                vals[f.name] = getattr(args, f.name)
            elif ev and env.get(ev, "") != "":
                vals[f.name] = cls._coerce(f, env[ev])
            elif ev and gcfg.get(ev, "") not in (None, ""):        # global fiber-kit.yaml overrides the session
                vals[f.name] = cls._coerce(f, gcfg[ev])
            elif f.name in section and section[f.name] not in (None, ""):
                vals[f.name] = cls._coerce(f, section[f.name])
            elif ev and section.get(ev, "") not in (None, ""):     # tolerate FK_* keys in the yaml section
                vals[f.name] = cls._coerce(f, section[ev])
            else:
                vals[f.name] = f.metadata.get("recommended", f.default) if profile == "recommended" else f.default
        return cls(**vals)

    def apply_to(self, args):
        """Write the resolved values back onto an argparse namespace (so existing `a.x` reads work)."""
        for f in _dc_fields(type(self)):
            setattr(args, f.name, getattr(self, f.name))
        return args

    @classmethod
    def to_yaml(cls, prefix="", profile="default"):
        """Emit a commented YAML block, generated from metadata. profile='recommended' emits the pipeline
        profile values, otherwise the plain field defaults."""
        out = []
        for f in _dc_fields(cls):
            v = f.metadata.get("recommended", f.default) if profile == "recommended" else f.default
            sv = '""' if v is None else v
            out.append(f"{prefix}{f.name}: {sv}".ljust(30) + f"  # {f.metadata.get('help', '')}")
        return "\n".join(out)

@dataclass
class IntrachunkConfig(StageConfig):
    """Tunable knobs for fiber-intrachunk (the within-chunk merge). Structural data-flow flags
    (--cpos-*, --clu-*, --out-stage, --emit-units) are NOT here — they are the file-naming contract."""
    gate: str = knob("band", "shape gate: cosine|mmd|kcov|cfiber|band (default band: energy-scaled median+/-sigma overlap)",
                     env="FK_INTRA_GATE", choices=("cosine", "mmd", "kcov", "cfiber", "band"), type=str, recommended="band")
    cos_thr: float = knob(0.85, "cosine recall prefilter", env="FK_INTRA_COS_THR")
    off_thr: float = knob(1.0, "inter-channel offset RMS gate (samples)", env="FK_INTRA_OFF_THR")
    depth_gate: float = knob(35.0, "depth gate (um)", env="FK_INTRA_DEPTH_GATE")
    amp_gate: float = knob(0.0, "absolute log-amplitude (energy) gate, natural log; ln(3)=1.1 -> 3x (0=off)",
                           env="FK_INTRA_AMP_GATE", recommended=1.10)
    amp_band_gap: float = knob(0.0, "post-linkage amplitude-band partition (0=off): split a unit's atoms where "
                               "the sorted log2 CEILING gap exceeds this.  Ceilings = median best-channel p2p of "
                               "each atom's top AMP_BAND_TOP fraction on the raw .spk.standard -- the un-adapted "
                               "top of the adaptation ladder -- so the same-cell tolerance is tight (~1.3x = "
                               "0.33 log2 measured) where whole-cluster amplitude is adaptation-confounded (the "
                               "3x amp_gate).  Covers EVERY linkage incl. 'ms', which carries no pairwise "
                               "amplitude gate at all.  0.4 measured best on g6 (+2.6pp GT purity, completeness "
                               "unchanged).  Inert without .spk.standard.",
                               env="FK_INTRA_AMP_BAND_GAP", recommended=0.4)
    amp_band_top: float = knob(0.2, "top fraction of an atom's spikes (by best-channel p2p) defining its ceiling",
                               env="FK_INTRA_AMP_BAND_TOP")
    amp_band_min: int = knob(25, "min atom spikes to ANCHOR a band; smaller atoms attach to the nearest band",
                             env="FK_INTRA_AMP_BAND_MIN", type=int)
    recut_gate: str = knob("", "post-linkage median-residual KlustaKwik denoise of every unit: '' off | all | "
                           "dip | giqr.  'all' recuts every unit >= RECUT_MIN_SPK (surveyed g6: 40 units "
                           "improved GT purity >2pp, none hurt -- kk's model selection + the component floor "
                           "are the quality control at UNIT level, but chain-scored 'all' over-splits "
                           "drifting cells: ARI 0.188 -> 0.144).  'giqr' (own-gain IQR) is the measured "
                           "sweet spot -- it recuts only amplitude-impure units: +4.3pp GT purity with ARI "
                           "UP (0.190), vs 'dip' +2.5pp/0.189 and 'all' +5.7pp at heavy ARI/completeness "
                           "cost (terminal pre-curation use only)", env="FK_INTRA_RECUT",
                           choices=("", "all", "dip", "giqr"), type=str, recommended="giqr")
    recut_dims: int = knob(4, "residual-PCA features fed to the recut KlustaKwik round",
                           env="FK_INTRA_RECUT_DIMS", type=int)
    recut_rounds: int = knob(2, "max recut passes per unit lineage (products are re-tested -- the curator's "
                             "double round)", env="FK_INTRA_RECUT_ROUNDS", type=int)
    recut_min_n: int = knob(40, "min spikes for a recut component to become a unit; smaller components and "
                            "kk's noise bucket shed to reserve", env="FK_INTRA_RECUT_MIN_N", type=int)
    recut_min_spk: int = knob(60, "min unit spikes to attempt a recut", env="FK_INTRA_RECUT_MIN_SPK", type=int)
    recut_alpha: float = knob(0.01, "gate 'dip': Hartigan dip p on the residual PCs below this = recut",
                              env="FK_INTRA_RECUT_ALPHA")
    recut_giqr: float = knob(0.4, "gate 'giqr': own-gain IQR above this = recut",
                             env="FK_INTRA_RECUT_GIQR")
    refrac_ceiling: float = knob(None, "reject merge if combined 2ms-ISI violation > this percent (empty=off)",
                                 env="FK_INTRA_REFRAC_CEILING", recommended=1.0)
    pre_merge_cos: float = knob(0.0, "pre-collapse obvious mutual-NN pairs at cosine>=this (0=off)",
                                env="FK_INTRA_PRE_MERGE_COS", recommended=0.97)
    n_iter: int = knob(1, "iterate group->re-estimate->regroup this many passes (1=single pass); >1 keeps the tight "
                       "gate but re-merges DENOISED units across passes, consolidating over-split fragments a single "
                       "pass leaves. Early-converges when a pass merges nothing (g5: 5 -> ~1124). Left at 1 in "
                       "production; the exp config opts in (FK_INTRA_ITER).",
                       env="FK_INTRA_ITER", type=int, cli="iter")
    linkage: str = knob("complete", "complete|dynamic|ms|kk",
                        env="FK_INTRA_LINKAGE", choices=("complete", "dynamic", "ms", "kk"), type=str)
    kk_dims: int = knob(5, "kk linkage (MW-cluster + median-residual KlustaKwik): PCs of the median-residual "
                        "waveform used to split within each MW group (0 or >=ncol = all). Reduced dims keep the "
                        "split sensitive to the dominant between-cell deviation.", env="FK_INTRA_KK_DIMS", type=int)
    kk_mw_thr: float = knob(0.10, "kk linkage: median-waveform grouping cut = 1 - cosine; atoms whose windowed "
                            "median waveforms are within this cosine distance group together (complete-linkage, so "
                            "EVERY pair must be within it). 0.10 -> group only atoms with median-waveform cosine "
                            ">= 0.90. Lower = stricter grouping (less merging); this wall is what prevents pooling "
                            "distinct cells.", env="FK_INTRA_KK_MW_THR")
    kk_win: int = knob(10, "kk linkage: half-window (samples each side of the peak) the waveform is cropped to "
                       "BEFORE the median-waveform cosine and the median-residual SVD -- focuses both on the peak "
                       "region and drops the flat baseline tails (validated: cleaner refractory than the full "
                       "42-sample window).", env="FK_INTRA_KK_WIN", type=int)
    kk_min_size: int = knob(20, "kk linkage: KlustaKwik min cluster size for the within-group split.",
                            env="FK_INTRA_KK_MIN_SIZE", type=int)
    kk_big: int = knob(20, "kk linkage: atoms with >= this many spikes are 'big' (kept as grouping anchors); "
                       "smaller atoms are folded into the nearest big atom by gated median-waveform cosine, or "
                       "reserved if none matches. fiber_stochastic emits mostly tiny atoms, so this fold is what "
                       "brings the singles into the fold and prevents the over-split.", env="FK_INTRA_KK_BIG", type=int)
    kk_fold_thr: float = knob(0.90, "kk linkage: a small atom folds into the nearest big atom only if their "
                              "windowed median-waveform cosine >= this; else it goes to the reserve (id 1). This "
                              "gate is the reject option a Mahalanobis assignment lacks -- a tiny fragment of a "
                              "real cell folds in, a tiny distinct/noise atom stays out (holds zero false-merge).",
                              env="FK_INTRA_KK_FOLD_THR")
    kk_strip: int = knob(1, "kk linkage: per-channel polynomial-coefficient contamination strip run "
                          "after the split (0 = off, 1 = on; ON by default). Fits each spike's per-channel waveform with a "
                          "Legendre polynomial (smooths per-sample stderiv noise into stable coeffs) and moves "
                          "robust-z outlier spikes to the reserve. Cleans most contaminated units while keeping "
                          "clean units usable (raw per-sample strips cannot -- stderiv self-correlation ~0.6 "
                          "drowns the signal).", env="FK_INTRA_KK_STRIP", type=int)
    kk_strip_deg: int = knob(4, "kk linkage: Legendre polynomial degree for the per-channel strip (validated: "
                             "4 best; 3 under-cleans, 5 over-strips).", env="FK_INTRA_KK_STRIP_DEG", type=int)
    kk_strip_z: float = knob(4.0, "kk linkage: robust-z (MAD-scaled) threshold on the per-channel polynomial "
                             "coefficients; a spike is a per-channel outlier if any coefficient exceeds this.",
                             env="FK_INTRA_KK_STRIP_Z")
    kk_strip_maxbad: int = knob(1, "kk linkage: a spike is stripped to the reserve if it is a polynomial-coeff "
                                "outlier on MORE than this many signal channels (0 = strictest).",
                                env="FK_INTRA_KK_STRIP_MAXBAD", type=int)
    align_lag: int = knob(6, "merge-time best-lag half-window, NATIVE samples (0=off)",
                          env="FK_INTRA_ALIGN_LAG", type=int)
    align_upsample: int = knob(1, "cubic-spline upsampling factor for the align-lag search",
                               env="FK_INTRA_ALIGN_UPSAMPLE", type=int)
    cfiber_q: float = knob(0.90, "cfiber self-calibration quantile", env="FK_INTRA_CFIBER_Q")
    cfiber_null: str = knob("order", "cfiber split-half null basis: order|energy",
                            env="FK_INTRA_CFIBER_NULL", choices=("order", "energy"), type=str)
    band_thr: float = knob(None, "gate='band': min energy-scaled median+/-sigma band-overlap IoU to merge (empty -> 0.5)",
                           env="FK_INTRA_BAND_THR")
    cfiber_thr_floor: float = knob(0.0, "absolute floor on the self-calibrated cfiber threshold (0=off)",
                                   env="FK_INTRA_CFIBER_THR_FLOOR")
    sig_cap: int = knob(None, "per-fragment spikes for the mean template (empty = no cap)", env="FK_INTRA_SIG_CAP", type=int, recommended=8000)
    warp_thr: float = knob(None, "group-delay WARP coherence gate (Omlor-Giese): merge only if the cross-channel "
                           "correlation of the two fragments' per-channel group-delay profiles >= this. Same-neuron "
                           "warps cohere; co-located different cells anti-correlate. Group-delay is noisy at low spike "
                           "count -> use LOW (~0.3). empty=off.", env="FK_INTRA_WARP_THR")
    warp_resid_thr: float = knob(None, "single-channel warp-incongruity SUB-GATE (layers on warp_thr): among already-"
                                 "coherent pairs (corr>=0.85), veto if any ONE centroid-range channel's group-delay "
                                 "residual (Theil-Sen line) > this many samples -- a strong-channel-masked different "
                                 "source. g5 knee ~1.0. empty=off.", env="FK_INTRA_WARP_RESID_THR")
    off_thr_int: float = knob(None, "DUAL gate: offset RMS threshold for suspected INTERNEURON pairs (narrow trough-to-"
                              "peak). Fast cells have stable offsets (~0.23) so off_thr=1.0 is inert; tighten to ~0.5. "
                              "Needs raw .spk for cell-typing. empty=off (use off_thr).", env="FK_INTRA_OFF_THR_INT")
    off_thr_pyr: float = knob(None, "DUAL gate: offset RMS threshold for suspected PYRAMIDAL pairs (wide trough-to-peak); "
                              "~1.0. Set BOTH off_thr_int and off_thr_pyr to enable the dual gate; mixed pairs use the "
                              "stricter. empty=off.", env="FK_INTRA_OFF_THR_PYR")


class Plugin:
    """An ndmanager plugin: an ordered list of (stage, StageConfig) pairs whose union of knobs forms one
    `ndm_<name>` program.  The typed config is the single source of truth for the CLI, the `--ndm-describe`
    schema, the session.yaml `programs:` entry, and the readback (session_yaml.pipeline_section)."""
    name = None            # the ndm command, e.g. "ndm_fiber-kit"
    stages = ()            # ((stage_key, StageConfig_subclass), ...)
    help = ""

    @classmethod
    def _items(cls, profile):
        for stage, sc in cls.stages:
            for f in _dc_fields(sc):
                v = f.metadata.get("recommended", f.default) if profile == "recommended" else f.default
                yield stage, f, v

    @classmethod
    def describe(cls, profile="recommended"):
        """The NDManager program description, as the YAML `program:` mapping that DescriptionYamlReader
        parses (name / help / parameters:[{name,value,status}]).  This is the --ndm-describe payload and
        the single source of truth for the GUI editor.  Knobs flatten <stage>.<knob>."""
        import yaml
        params = [{"name": "%s.%s" % (stage, f.name),
                   "value": ("" if v is None else v),
                   "status": "Optional"}
                  for stage, f, v in cls._items(profile)]
        doc = {"program": {"name": cls.name, "help": cls.help, "parameters": params}}
        return yaml.safe_dump(doc, sort_keys=False, default_flow_style=False)

    @classmethod
    def programs_entry(cls, profile="recommended"):
        """The session.yaml `programs:` list entry to paste in (flattened <stage>.<knob> parameters)."""
        rows = []
        for stage, f, v in cls._items(profile):
            sv = "''" if v is None else v
            rows.append("  - {name: %s.%s, value: %s, status: Optional}" % (stage, f.name, sv))
        return "- name: %s\n  parameters:\n%s\n" % (cls.name, "\n".join(rows))


class FiberKitPlugin(Plugin):
    name = "ndm_fiber-kit"
    stages = (("intrachunk", IntrachunkConfig),)
    help = ("fiber-kit drift-stable spike-sorting pipeline (standalone CLI: fiber-pipeline). Parameters are\n"
            "flattened as <stage>.<knob>; only the intrachunk stage is typed so far -- the others use\n"
            "fiber-pipeline's --profile/env defaults until they are converted to StageConfig.")
