# ════════════════════════════════════════════════════════════════════════════
#  session_yaml.py  —  read a neurosuite-3 SESSION.yaml so the CLIs only need
#  <script> <session> <group>.  Mirrors the ndmanager-plugins convention:
#  group is 1-based on the command line and indexes spikeDetection.channelGroups
#  at [group-1]; acquisitionSystem gives nChannels / samplingRate.
#
#  Search order for <session>.yaml (per the project layout SESSION/SESSION.yaml):
#    1. <session>.yaml / .yml                 (working directory)
#    2. <session>/<basename>.yaml / .yml      (session folder)
#    3. <session> itself, if it ends in .yaml/.yml
#  The binary files (.res/.spkD/.fil) are taken from the YAML's own location:
#  base = <yaml path without extension>, so files resolve as base.res.<group> etc.
# ════════════════════════════════════════════════════════════════════════════
import os
from dataclasses import dataclass

_LP = "\u25b8 session"
def _log(m=""): print(f"{_LP} \u00b7 {m}" if m else _LP)
def _det(k, v, w=9): print(f"{' ' * (len(_LP) + 3)}{k:<{w}} {v}")


def find_session_yaml(session):
    """Return the path to <session>.yaml (or .yml), or None if not found."""
    s = session.rstrip("/")
    if s.endswith((".yaml", ".yml")) and os.path.isfile(s):
        return s
    stem = s[:-5] if s.endswith(".yaml") else (s[:-4] if s.endswith(".yml") else s)
    name = os.path.basename(stem)
    cands = [f"{stem}.yaml", f"{stem}.yml",
             os.path.join(stem, f"{name}.yaml"), os.path.join(stem, f"{name}.yml")]
    for c in cands:
        if os.path.isfile(c):
            return c
    return None


def _safe_load(path):
    try:
        import yaml
    except ImportError as e:
        raise SystemExit("session YAML support needs PyYAML — `pip install pyyaml` "
                         "(or pass --channels/--ntotal/--sr manually).") from e
    with open(path) as f:
        return yaml.safe_load(f)


_PROBE_KEYS = ("probeFile", "probeFiles", "probe", "probeFileName", "probefile", "probes")


def probe_files(doc, yaml_dir, group=None):
    """Resolve the NeuroSuite .probe file path(s) named by a session YAML.

    Searches, in order: top-level probeFile/probeFiles/probe/...; the 'files',
    'general', 'spikeDetection' and 'acquisitionSystem' sub-sections; and the
    per-group entry spikeDetection.channelGroups[group-1].probe.  A value may be a
    single path or a list; relative paths resolve against the YAML's directory.
    Returns a (de-duplicated, order-preserving) list of paths, or [] if none."""
    found = []

    def collect(v):
        if v is None:
            return
        for item in (v if isinstance(v, (list, tuple)) else [v]):
            if isinstance(item, str) and item:
                found.append(item)

    for k in _PROBE_KEYS:
        collect(doc.get(k))
    for sect in ("files", "general", "spikeDetection", "acquisitionSystem"):
        d = doc.get(sect) or {}
        if isinstance(d, dict):
            for k in _PROBE_KEYS:
                collect(d.get(k))
    if group is not None:
        groups = (doc.get("spikeDetection", {}) or {}).get("channelGroups", []) or []
        if 1 <= group <= len(groups):
            for k in _PROBE_KEYS:
                collect((groups[group - 1] or {}).get(k))
    # the canonical ndmanager 'probes:' list: one entry per probe, each with a
    # probeFile; concatenate in channelOffset order so load_geometry's global table
    # lines up with global channel ids (probe 0 @ offset 0, probe 1 @ offset 64, ...)
    probes = doc.get("probes")
    if isinstance(probes, list):
        entries = [p for p in probes if isinstance(p, dict) and p.get("probeFile")]
        entries.sort(key=lambda p: p.get("channelOffset", 0))
        for p in entries:
            collect(p["probeFile"])

    roots = [yaml_dir]
    plp = doc.get("probeLibraryPath")
    if isinstance(plp, str) and plp:
        roots.append(os.path.expanduser(plp))                # ndmanager probe library
    out, seen = [], set()
    for p in found:
        if os.path.isabs(p):
            rp = p
        else:
            rp = next((c for c in (os.path.normpath(os.path.join(r, p)) for r in roots)
                       if os.path.exists(c)), os.path.normpath(os.path.join(yaml_dir, p)))
        if rp not in seen:
            seen.add(rp); out.append(rp)
    return out


def load_session(session, group, path=None):
    """Parse <session>.yaml for `group` (1-based). Returns a dict with base,
    yaml, group, ntotal, sr, nbits, nchan, channels, nsamp, peak, nfeatures."""
    path = path or find_session_yaml(session)
    if path is None:
        raise FileNotFoundError(f"no <session>.yaml found for '{session}' "
                                f"(looked for {session}.yaml and {session}/{os.path.basename(session)}.yaml)")
    doc = _safe_load(path) or {}
    acq = doc.get("acquisitionSystem", {}) or {}
    groups = (doc.get("spikeDetection", {}) or {}).get("channelGroups", []) or []
    if not (1 <= group <= len(groups)):
        raise ValueError(f"group {group} out of range 1..{len(groups)} in {path}")
    g = groups[group - 1]
    channels = [int(c) for c in g.get("channels", [])]
    if not channels:
        raise ValueError(f"group {group} in {path} has no 'channels'")
    base = path[:-5] if path.endswith(".yaml") else path[:-4]
    probe = probe_files(doc, os.path.dirname(os.path.abspath(path)), group)
    return dict(
        base=base, yaml=path, group=group,
        ntotal=int(acq["nChannels"]) if acq.get("nChannels") is not None else None,
        sr=float(acq["samplingRate"]) if acq.get("samplingRate") is not None else None,
        nbits=int(acq.get("nBits", 16)),
        nchan=len(channels), channels=channels,
        nsamp=int(g["nSamples"]) if g.get("nSamples") is not None else None,
        peak=int(g["peakSampleIndex"]) if g.get("peakSampleIndex") is not None else None,
        nfeatures=int(g["nFeatures"]) if g.get("nFeatures") is not None else None,
        # Optional per-group custom spatial-derivative pattern for the stderiv
        # pipeline: "a-b,c-d,..." (order 4, single partner) or "a-b+c+d,..."
        # (order 5, reference sets), in group-local 0-based channel positions.
        # Absent on a session that uses a plain spatial-derivative order.
        sdiff_pairs=(str(g["sdiffPairs"]).strip()
                     if g.get("sdiffPairs") not in (None, "") else None),
        probe=probe,
    )


def refractory_period_samples(session, group, sr=None, path=None, default=16):
    """Imposed DETECTION refractory for `group`, in samples, with its source.

    The detector enforces a minimum inter-event interval; sub-floor ISIs are
    duplicate detections, not contamination.  Looks for, in order:
      spikeDetection.channelGroups[group-1].refractoryPeriod  (samples)
      spikeDetection.refractoryPeriod                         (samples)
      channelGroups[group-1].refractoryMs / spikeDetection.refractoryMs (ms -> samples, needs sr)
    falling back to `default` (16 samples ~= 0.49 ms at 32552 Hz).  Returns
    (samples, source) where source is a short provenance string."""
    path = path or find_session_yaml(session)
    if path is None:
        return int(default), "default (no yaml)"
    doc = _safe_load(path) or {}
    sd = doc.get("spikeDetection", {}) or {}
    groups = sd.get("channelGroups", []) or []
    g = groups[group - 1] if (isinstance(groups, list) and 1 <= group <= len(groups)) else {}
    if isinstance(g, dict) and g.get("refractoryPeriod") is not None:
        return int(g["refractoryPeriod"]), "yaml channelGroup.refractoryPeriod"
    if sd.get("refractoryPeriod") is not None:
        return int(sd["refractoryPeriod"]), "yaml spikeDetection.refractoryPeriod"
    ms = None
    if isinstance(g, dict) and g.get("refractoryMs") is not None:
        ms = float(g["refractoryMs"])
    elif sd.get("refractoryMs") is not None:
        ms = float(sd["refractoryMs"])
    if ms is not None and sr:
        return max(1, int(round(ms * sr / 1000.0))), f"yaml refractoryMs={ms}"
    return int(default), "default (not in yaml)"


_SESSION_FIELDS = ("base", "yaml", "group", "channels", "ntotal", "nchan",
                   "nsamp", "sr", "peak", "probe")


@dataclass
class SessionCfg:
    """Resolved session parameters.  Supports attribute access (cfg.sr) AND the legacy mapping
    access (cfg["sr"], cfg.get("sr")) so existing callers keep working unchanged — but an unknown
    key now raises KeyError instead of silently returning None.  That is the class of bug that let
    fiber_intrachunk read cfg["samplingRate"] / cfg.get("nSamples", 32) undetected (fixed in 0032):
    with this type those mistakes fail loudly at the read."""
    base: object
    yaml: object
    group: object
    channels: object
    ntotal: object
    nchan: object
    nsamp: object
    sr: object
    peak: object
    probe: object

    def __getitem__(self, k):
        if k not in _SESSION_FIELDS:
            raise KeyError(f"SessionCfg has no field {k!r}; valid: {_SESSION_FIELDS}")
        return getattr(self, k)

    def get(self, k, default=None):
        if k not in _SESSION_FIELDS:
            raise KeyError(f"SessionCfg has no field {k!r}; valid: {_SESSION_FIELDS}")
        return getattr(self, k)

    def __contains__(self, k):
        return k in _SESSION_FIELDS

    def keys(self):
        return _SESSION_FIELDS


def add_session_args(ap, *, positional=True, channels=True, ntotal=True, nsamp=True,
                     nchan=True, sr=True, probe=False, peak=False, nsamp_default=None):
    """Register the standard <session>.yaml CLI arguments on `ap` so the ~15 tools that resolve a
    session stop hand-rolling (and occasionally mistyping) them.  Pair with resolve_session_params,
    which returns a SessionCfg.  Keyword flags select which overrides to expose; nsamp_default /
    peak / probe cover the documented per-tool variations."""
    if positional:
        ap.add_argument("session", help="session basename or folder (finds <session>.yaml)")
        ap.add_argument("group", type=int, help="1-based spike group")
    if channels:
        ap.add_argument("--channels", default=None, help="override: comma-separated physical channels")
    if ntotal:
        ap.add_argument("--ntotal", type=int, default=None, help="override: total channels in the recording")
    if nchan:
        ap.add_argument("--nchan", "--nch", dest="nchan", type=int, default=None,
                        help="override: channels in this group")
    if nsamp:
        ap.add_argument("--nsamp", type=int, default=nsamp_default,
                        help="override: samples per spike (default from YAML)")
    if sr:
        ap.add_argument("--sr", type=float, default=None, help="override: sampling rate")
    if peak:
        ap.add_argument("--peak", type=int, default=16, help="peak sample index within the window")
    if probe:
        ap.add_argument("--probe", nargs="*", default=None, help="probe file(s) for geometry")
    return ap


def spike_groups(session):
    """The 1-based spike-group ids in <session>.yaml -- one per spikeDetection.channelGroups entry,
    matching the CLI (group g indexes channelGroups[g-1]).  [] if the session has none."""
    path = find_session_yaml(session)
    if not path:
        return []
    doc = _safe_load(path) or {}
    groups = (doc.get("spikeDetection", {}) or {}).get("channelGroups", []) or []
    return list(range(1, len(groups) + 1))


def program_parameters(session, program):
    """Flatten <session>.yaml programs[program].parameters into {name: value}, or {} if absent.
    The ndmanager convention: a program's parameters live in the session parameter file's `programs:` list,
    each entry `{name: <prog>, parameters: [{name, value, status}, ...]}`."""
    path = find_session_yaml(session)
    if not path:
        return {}
    doc = _safe_load(path) or {}
    for entry in (doc.get("programs") or []):
        if isinstance(entry, dict) and entry.get("name") == program:
            out = {}
            for p in (entry.get("parameters") or []):
                if isinstance(p, dict) and "name" in p:
                    out[p["name"]] = p.get("value")
            return out
    return {}


def pipeline_section(session, stage, program="ndm_fiber-kit"):
    """The <stage>.* slice of programs[program] (stage prefix stripped) -- one pipeline stage's per-session
    knobs.  e.g. programs[ndm_fiber-kit] {intrachunk.gate: cfiber} -> {gate: cfiber}.  {} if absent."""
    flat = program_parameters(session, program)
    pre = stage + "."
    return {k[len(pre):]: v for k, v in flat.items() if isinstance(k, str) and k.startswith(pre)}


def resolve_session_params(session, group, channels=None, ntotal=None, nchan=None,
                           nsamp=None, sr=None, require=("channels", "ntotal"), verbose=True):
    """Resolve run parameters from <session>.yaml with CLI overrides taking
    precedence.  If no YAML is found, falls back to treating `session` as the
    file base and relies entirely on the passed overrides.  `require` lists the
    keys that MUST end up set (raises SystemExit otherwise)."""
    yp = find_session_yaml(session)
    if yp is not None:
        info = load_session(session, group, path=yp)
        cfg = dict(base=info["base"], yaml=yp, group=group, channels=info["channels"],
                   ntotal=info["ntotal"], nchan=info["nchan"], nsamp=info["nsamp"],
                   sr=info["sr"], peak=info["peak"], probe=info.get("probe") or None,
                   sdiff_pairs=info.get("sdiff_pairs"))
        if verbose:
            _log(yp)
            _det("channels", f"{info['ntotal']} \u00b7 sr {float(info['sr']):,.0f} Hz")
            _det("group", f"{group} \u2192 {info['nchan']} ch {info['channels']}")
            _det("samples", f"{info['nsamp']} \u00b7 peak {info['peak']}")
            for pr in (info.get("probe") or []):
                _det("probe", pr)
    else:
        base = session.rstrip("/")
        for ext in (".yaml", ".yml"):
            if base.endswith(ext):
                base = base[:-len(ext)]
        cfg = dict(base=base, yaml=None, group=group, channels=None, ntotal=None,
                   nchan=None, nsamp=32, sr=32552.0, peak=None, probe=None,
                   sdiff_pairs=None)
        if verbose:
            _log(f"no YAML for '{session}'; using base='{base}' + explicit flags")

    # CLI overrides win
    if channels is not None:
        ch = [int(x) for x in (channels.split(",") if isinstance(channels, str) else channels)]
        cfg["channels"] = ch; cfg["nchan"] = len(ch)
    if ntotal is not None: cfg["ntotal"] = ntotal
    if nchan is not None: cfg["nchan"] = nchan
    if nsamp is not None: cfg["nsamp"] = nsamp
    if sr is not None: cfg["sr"] = sr
    if cfg.get("channels") and cfg.get("nchan") is None:
        cfg["nchan"] = len(cfg["channels"])

    bad = [k for k in require if k not in _SESSION_FIELDS]
    if bad:
        raise ValueError(f"resolve_session_params: unknown require keys {bad}; "
                         f"valid: {_SESSION_FIELDS}")
    missing = [k for k in require if cfg.get(k) in (None, [])]
    if missing:
        raise SystemExit(f"[session] missing {missing}: no usable <session>.yaml and no "
                         f"--{'/--'.join(missing)} given.")
    if cfg.get("channels") is not None and cfg.get("nchan") is not None \
            and len(cfg["channels"]) != cfg["nchan"]:
        raise SystemExit(f"[session] --channels has {len(cfg['channels'])} entries != nchan={cfg['nchan']}")
    # gentle warning when the mask/offset calibration may not fit
    if verbose and cfg.get("nsamp") not in (None, 32):
        _det("note", f"nSamples {cfg['nsamp']} \u2260 32 \u2192 masks/realign window rebuilt "
                     f"peak-relative (peak={cfg['peak']})")
    return SessionCfg(**cfg)
