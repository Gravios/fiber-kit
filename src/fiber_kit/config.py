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
#      explicit CLI flag  >  FK_* env var  >  <session>.yaml fiber_kit.<stage>  >  field default
#  so a session's tuning lives in its own <session>.yaml (the ndmanager-plugins
#  convention: a program's parameters travel with the session), while env/CLI stay
#  available for one-off sweeps.
# ─────────────────────────────────────────────────────────────────────────────
import argparse
from dataclasses import dataclass, field, fields as _dc_fields


def knob(default, help="", *, env=None, choices=None, type=float, cli=None):
    """A config field: `default` plus metadata (help/env/choices/type/cli-flag-name)."""
    return field(default=default, metadata=dict(help=help, env=env, choices=choices, type=type, cli=cli))


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
    def resolve(cls, args=None, env=None, section=None):
        """Build an instance with precedence CLI > env > session-section > default.

        args    : the parsed argparse namespace (flags use default=SUPPRESS, so an attribute is
                  present iff the user gave it).  None => skip the CLI layer.
        env     : os.environ-like mapping; a field's env metadata names its FK_* var.
        section : the session.yaml fiber_kit.<stage> dict; keys may be the field name OR its FK_* env name.
        """
        env = env or {}
        section = section or {}
        vals = {}
        for f in _dc_fields(cls):
            ev = f.metadata.get("env")
            if args is not None and hasattr(args, f.name):
                vals[f.name] = getattr(args, f.name)
            elif ev and env.get(ev, "") != "":
                vals[f.name] = cls._coerce(f, env[ev])
            elif f.name in section and section[f.name] not in (None, ""):
                vals[f.name] = cls._coerce(f, section[f.name])
            elif ev and section.get(ev, "") not in (None, ""):     # tolerate FK_* keys in the yaml section
                vals[f.name] = cls._coerce(f, section[ev])
            else:
                vals[f.name] = f.default
        return cls(**vals)

    def apply_to(self, args):
        """Write the resolved values back onto an argparse namespace (so existing `a.x` reads work)."""
        for f in _dc_fields(type(self)):
            setattr(args, f.name, getattr(self, f.name))
        return args

    @classmethod
    def to_yaml(cls, prefix=""):
        """Emit a commented YAML block of the defaults — the YAML template, generated from metadata."""
        out = []
        for f in _dc_fields(cls):
            v = f.default
            sv = '""' if v is None else v
            out.append(f"{prefix}{f.name}: {sv}".ljust(30) + f"  # {f.metadata.get('help', '')}")
        return "\n".join(out)


@dataclass
class IntrachunkConfig(StageConfig):
    """Tunable knobs for fiber-intrachunk (the within-chunk merge). Structural data-flow flags
    (--cpos-*, --clu-*, --out-stage, --emit-units) are NOT here — they are the file-naming contract."""
    gate: str = knob("cosine", "shape gate: cosine|mmd|kcov|cfiber",
                     env="FK_INTRA_GATE", choices=("cosine", "mmd", "kcov", "cfiber"), type=str)
    cos_thr: float = knob(0.85, "cosine recall prefilter", env="FK_INTRA_COS_THR")
    off_thr: float = knob(1.0, "inter-channel offset RMS gate (samples)", env="FK_INTRA_OFF_THR")
    depth_gate: float = knob(35.0, "depth gate (um)", env="FK_INTRA_DEPTH_GATE")
    amp_gate: float = knob(0.0, "absolute log-amplitude (energy) gate, natural log; ln(3)=1.1 -> 3x (0=off)",
                           env="FK_INTRA_AMP_GATE")
    refrac_ceiling: float = knob(None, "reject merge if combined 2ms-ISI violation > this percent (empty=off)",
                                 env="FK_REFRAC_CEILING")
    pre_merge_cos: float = knob(0.0, "pre-collapse obvious mutual-NN pairs at cosine>=this (0=off)",
                                env="FK_PRE_MERGE_COS")
    linkage: str = knob("complete", "complete|dynamic|ms",
                        env="FK_INTRA_LINKAGE", choices=("complete", "dynamic", "ms"), type=str)
    align_lag: int = knob(6, "merge-time best-lag half-window, NATIVE samples (0=off)",
                          env="FK_ALIGN_LAG", type=int)
    align_upsample: int = knob(1, "cubic-spline upsampling factor for the align-lag search",
                               env="FK_ALIGN_UPSAMPLE", type=int)
    cfiber_q: float = knob(0.90, "cfiber self-calibration quantile", env="FK_INTRA_CFIBER_Q")
    cfiber_null: str = knob("order", "cfiber split-half null basis: order|energy",
                            env="FK_CFIBER_NULL", choices=("order", "energy"), type=str)
    cfiber_thr_floor: float = knob(0.0, "absolute floor on the self-calibrated cfiber threshold (0=off)",
                                   env="FK_CFIBER_THR_FLOOR")
    sig_cap: int = knob(None, "per-fragment spikes for the mean template (empty = no cap)", env="FK_INTRA_SIG_CAP", type=int)
