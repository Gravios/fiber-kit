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
from dataclasses import dataclass, field, fields as _dc_fields


_UNSET = object()


def load_global_config(path=None):
    """Load the global fiber-kit.yaml (a FLAT FK_* mapping) that OVERRIDES per-session
    tuning when present.  Search order: explicit `path`, then $FK_CONFIG, then
    ./fiber-kit.yaml in the working directory (the same file scripts/fiber-pipeline reads).
    Returns {} when absent or unreadable, so it is a no-op unless a file is there."""
    import os
    p = path or os.environ.get("FK_CONFIG") or os.path.join(os.getcwd(), "fiber-kit.yaml")
    if not (p and os.path.isfile(p)):
        return {}
    try:
        import yaml
        with open(p) as fh:
            d = yaml.safe_load(fh) or {}
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


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
        if f.metadata.get("choices"):
            return str(v)
        t = f.metadata.get("type", float)
        return t(v)

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
    refrac_ceiling: float = knob(None, "reject merge if combined 2ms-ISI violation > this percent (empty=off)",
                                 env="FK_REFRAC_CEILING", recommended=1.0)
    pre_merge_cos: float = knob(0.0, "pre-collapse obvious mutual-NN pairs at cosine>=this (0=off)",
                                env="FK_PRE_MERGE_COS", recommended=0.97)
    n_iter: int = knob(1, "iterate group->re-estimate->regroup this many passes (1=single pass); >1 keeps the tight "
                       "gate but re-merges DENOISED units across passes, consolidating over-split fragments a single "
                       "pass leaves. Early-converges when a pass merges nothing (g5: 5 -> ~1124). Left at 1 in "
                       "production; the exp config opts in (FK_INTRA_ITER).",
                       env="FK_INTRA_ITER", type=int, cli="iter")
    linkage: str = knob("complete", "complete|dynamic|ms",
                        env="FK_INTRA_LINKAGE", choices=("complete", "dynamic", "ms"), type=str)
    align_lag: int = knob(6, "merge-time best-lag half-window, NATIVE samples (0=off)",
                          env="FK_ALIGN_LAG", type=int)
    align_upsample: int = knob(1, "cubic-spline upsampling factor for the align-lag search",
                               env="FK_ALIGN_UPSAMPLE", type=int)
    cfiber_q: float = knob(0.90, "cfiber self-calibration quantile", env="FK_INTRA_CFIBER_Q")
    cfiber_null: str = knob("order", "cfiber split-half null basis: order|energy",
                            env="FK_CFIBER_NULL", choices=("order", "energy"), type=str)
    band_thr: float = knob(None, "gate='band': min energy-scaled median+/-sigma band-overlap IoU to merge (empty -> 0.5)",
                           env="FK_INTRA_BAND_THR")
    cfiber_thr_floor: float = knob(0.0, "absolute floor on the self-calibrated cfiber threshold (0=off)",
                                   env="FK_CFIBER_THR_FLOOR")
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
