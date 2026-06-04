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
    return dict(
        base=base, yaml=path, group=group,
        ntotal=int(acq["nChannels"]) if acq.get("nChannels") is not None else None,
        sr=float(acq["samplingRate"]) if acq.get("samplingRate") is not None else None,
        nbits=int(acq.get("nBits", 16)),
        nchan=len(channels), channels=channels,
        nsamp=int(g["nSamples"]) if g.get("nSamples") is not None else None,
        peak=int(g["peakSampleIndex"]) if g.get("peakSampleIndex") is not None else None,
        nfeatures=int(g["nFeatures"]) if g.get("nFeatures") is not None else None,
    )


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
                   sr=info["sr"], peak=info["peak"])
        if verbose:
            print(f"[session] {yp}: nChannels={info['ntotal']} sr={info['sr']} "
                  f"group {group} -> {info['nchan']} ch {info['channels']} "
                  f"nSamples={info['nsamp']} peak={info['peak']}")
    else:
        base = session.rstrip("/")
        for ext in (".yaml", ".yml"):
            if base.endswith(ext):
                base = base[:-len(ext)]
        cfg = dict(base=base, yaml=None, group=group, channels=None, ntotal=None,
                   nchan=None, nsamp=32, sr=32552.0, peak=None)
        if verbose:
            print(f"[session] no YAML for '{session}'; using base='{base}' + explicit flags")

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

    missing = [k for k in require if cfg.get(k) in (None, [])]
    if missing:
        raise SystemExit(f"[session] missing {missing}: no usable <session>.yaml and no "
                         f"--{'/--'.join(missing)} given.")
    if cfg.get("channels") is not None and cfg.get("nchan") is not None \
            and len(cfg["channels"]) != cfg["nchan"]:
        raise SystemExit(f"[session] --channels has {len(cfg['channels'])} entries != nchan={cfg['nchan']}")
    # gentle warning when the mask/offset calibration may not fit
    if verbose and cfg.get("nsamp") not in (None, 32):
        print(f"[session] note: fiber_lib MASK_FULL/EXTRACT_OFFSET are calibrated for 32-sample "
              f"windows (peak ~15); this group has nSamples={cfg['nsamp']} — verify masking.")
    return cfg
