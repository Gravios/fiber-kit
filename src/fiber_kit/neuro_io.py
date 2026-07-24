# ════════════════════════════════════════════════════════════════════════════
#  neuro_io.py — standardized neurosuite-3 on-disk I/O for fiber-kit.
#
#  Single source of truth for the NeuroSuite binary formats fiber-kit reads and
#  writes, so the naming convention and variant resolution match the C++
#  toolchain exactly instead of being re-implemented (with hardcoded extension
#  lists) in every script.  All artifacts are binary (see below).
#
#  Mirrors:
#    - src/libneurosuite-core/src/neurosuite/core/neurofileio.{h,cpp}
#        readClu / writeClu / readRes / writeRes / readFetBinary /
#        readClusterRes / resolveInput / preferDerived / preferCanonical
#    - src/kiloklustakwik/src/KK_io.cpp
#        LoadData / LoadClu  (the pickInputPath -> resolveInput .fet/.fetD fallback)
#
#  ── Variant-aware naming (NeuroSuite convention) ──────────────────────────────
#  A per-group typed file (res/clu/fet/spk/pca …) may exist in several
#  representation "variants" — most notably the stderiv-derived features.  The
#  group number is ALWAYS the trailing token; the variant, when present, sits
#  between the type and the group:
#
#      <base>.<type>.<group>            canonical (no variant)        e.g. .spk.5
#      <base>.<type>.<variant>.<group>  dotted variant (preferred)   e.g. .spk.stderiv.5
#      <base>.<type><variant>.<group>   legacy glued form (READ-only) e.g. .spkD.5
#
#  The glued form (a single letter glued onto the type token, as the stderiv
#  pipeline historically wrote .fetD/.spkD/.pcaD) is recognised on READ only,
#  for backward compatibility; new writers emit the dotted form.  resolve_input()
#  walks `prefer_variants` in order and returns the first file that exists; ""
#  denotes the canonical (no-variant) form.  fiber-kit operates in stderiv space
#  so it defaults to prefer_derived()  ({"stderiv","D",""}); the .fet reader
#  defaults to prefer_canonical() to match KlustaKwik's pickInputPath.
#
#  ── All artifacts are binary ──────────────────────────────────────────────────
#  fiber-kit reads and writes .res (int64), .clu (int32 nClusters header + int32
#  ids) and .fet (int32 header + int64 body) as raw binary only.  The legacy text
#  format and the first-byte binary/text heuristic have been removed: that
#  heuristic misclassified a binary file whose header low byte lands in 0x30-0x39
#  as text (e.g. a 1332-cluster .clu -> first header byte 0x34 = '4'), which broke
#  reads outright -- it was never the "non-issue in practice" it was assumed to be.
# ════════════════════════════════════════════════════════════════════════════
import glob
import os
from collections import namedtuple

import numpy as np

__all__ = [
    "ResolvedInput", "resolve_input", "resolve_any", "prefer_derived", "prefer_canonical",
    "VariantSpec", "parse_variant_token", "variant_family", "is_stderiv_variant",
    "read_res_file", "write_res_file", "read_res", "write_res",
    "read_clu_file", "write_clu_file", "read_clu", "write_clu",
    "read_cluster_res",
    "read_fet_file", "read_fet", "write_fet_file", "write_fet",
    "open_spk_file", "open_spk", "open_spkD", "write_spk_file", "write_spk",
    "open_signal",
    "fibers_path",
    "add_clu_args",
    "resolve_clu",
]

# ── dtypes (little-endian, NeuroSuite on-disk) ──────────────────────────────
RES_DTYPE = np.dtype("<i8")   # .res    int64 timestamps, no header
CLU_DTYPE = np.dtype("<i4")   # .clu    int32 header + int32 ids
FET_DTYPE = np.dtype("<i8")   # .fet    int32 header + int64 values, row-major
SPK_DTYPE = np.dtype("<i2")   # .spk    int16 sample-major, no header
FET_HDR_DTYPE = np.dtype("<i4")


# ── variant-aware input resolution (mirrors neurofileio::resolveInput) ───────
ResolvedInput = namedtuple("ResolvedInput", ["path", "variant", "dotted", "found"])


VariantSpec = namedtuple("VariantSpec", "family kind order")


def parse_variant_token(variant):
    """Decompose a variant token <family>[_<kind><order>], e.g. 'stderiv_C4'.

    Returns VariantSpec(family, kind, order): kind is 'S' (plain spatial derivative)
    or 'C' (the session's custom sdiffPairs pattern), order is the spatial-derivative
    order actually applied; both are None on a bare token.

    Mirrors the neurosuite-3 token grammar as implemented in custody.hpp
    (MethodSpec/parseMethodToken), ndm_custody (ndm_parse_method) and
    ndm_resolve_io (parse_method_token).  A token that does not match the suffix
    grammar is opaque -- the whole string is the family -- so an unknown token never
    masquerades as a known one.  Kept honest by test/custody_vectors.tsv, an
    identical copy of the canonical table those three are checked against.
    """
    cut = variant.rfind("_")
    if cut != -1:
        suffix = variant[cut + 1:]
        if len(suffix) >= 2 and suffix[0] in ("S", "C") and suffix[1:].isdigit():
            return VariantSpec(variant[:cut], suffix[0], int(suffix[1:]))
    return VariantSpec(variant, None, None)


def variant_family(variant):
    """Family of a variant token -- the part before an _S<order>/_C<order> suffix.

    'stderiv_C4' -> 'stderiv';  'stderiv' -> 'stderiv';  'standard' -> 'standard'.
    """
    return parse_variant_token(variant).family


def is_stderiv_variant(variant):
    """True for the stderiv family, including suffixed tokens (stderiv_S3, stderiv_C4).

    Deliberately not a 'stderiv' prefix test: 'stderivfoo' is a different family.
    """
    return variant_family(variant) == "stderiv"


def prefer_derived():
    """Preference order for stderiv-space inputs: derived first, then a raw fallback.
    Mirrors neurofileio::preferDerived().  Under the dotted naming convention raw is the
    'standard' variant (<base>.spk.standard.N), so the raw fallback is 'standard' (with the
    legacy canonical <base>.spk.N last)."""
    return ["stderiv", "D", "standard", ""]


def prefer_standard():
    """Preference order for RAW (standard) inputs: <base>.spk.standard.N first, then the
    legacy canonical <base>.spk.N.  Deliberately does NOT fall back to stderiv -- callers
    that need raw amplitudes (localization, position, raw-PCA) must fail rather than
    silently localize on the stderiv transform (which breaks the amplitude-distance law)."""
    return ["standard", ""]


def prefer_canonical():
    """Preference order for canonical inputs first.  Mirrors
    neurofileio::preferCanonical()  ->  {"", "stderiv", "D"}."""
    return ["", "stderiv", "D"]


def resolve_input(base, type_, group, prefer_variants):
    """Return the first existing variant of <base>.<type>[.<variant>].<group>.

    Walks `prefer_variants` in order; "" denotes the canonical (no-variant)
    form.  For each non-empty variant the dotted form (<base>.<type>.<variant>.
    <group>) is probed first, then the legacy glued form (<base>.<type><variant>.
    <group>).  If nothing exists, `found` is False and `path` is the canonical
    path so the caller can emit a sensible "missing input" error.

    Faithful port of neurofileio::resolveInput / KK_io's pickInputPath.
    """
    g = str(group)
    canonical = f"{base}.{type_}.{g}"
    for v in prefer_variants:
        if v == "":
            if os.path.exists(canonical):
                return ResolvedInput(canonical, "", False, True)
            continue
        dotted = f"{base}.{type_}.{v}.{g}"
        if os.path.exists(dotted):
            return ResolvedInput(dotted, v, True, True)
        glued = f"{base}.{type_}{v}.{g}"
        if os.path.exists(glued):
            return ResolvedInput(glued, v, False, True)
        # A session produced with a SUFFIXED token (stderiv_C4) has no plain
        # <type>.stderiv.<group>.  Match the family here rather than falling
        # through to the next preference: for prefer_derived() the next entry is
        # "standard", so falling through would silently hand back RAW waveforms
        # for a stderiv request.
        if "_" not in v:
            prefix, tail = f"{base}.{type_}.{v}_", f".{g}"
            hits = [m for m in sorted(glob.glob(f"{prefix}*{tail}"))
                    if variant_family(m[len(f"{base}.{type_}."):-len(tail)]) == v]
            if len(hits) == 1:
                token = hits[0][len(f"{base}.{type_}."):-len(tail)]
                return ResolvedInput(hits[0], token, True, True)
            if len(hits) > 1:
                tokens = [h[len(f"{base}.{type_}."):-len(tail)] for h in hits]
                raise ValueError(
                    f"{base}.{type_}.*.{g}: several '{v}' variants exist "
                    f"({', '.join(tokens)}); pass the exact token in prefer_variants "
                    f"rather than the family name so the choice is explicit")
    return ResolvedInput(canonical, "", False, False)


# ── .res.N ───────────────────────────────────────────────────────────────────
def read_res_file(path):
    """Read a binary .res file (int64 spike timestamps; mirrors
    neurofileio::writeRes).  Returns an int64 ndarray."""
    return np.fromfile(path, dtype=RES_DTYPE).astype(np.int64)


def write_res_file(path, times):
    """Write spike timestamps as canonical binary .res (little-endian int64,
    no header).  Mirrors neurofileio::writeRes' binary counterpart."""
    np.asarray(times, dtype=RES_DTYPE).tofile(path)
    return path


def read_res(base, elec, prefer=None):
    """Resolve and read <base>.res.<elec> (variant-aware).  Defaults to the
    canonical form first; pass prefer_derived() to prefer a derived .res."""
    r = resolve_input(base, "res", elec, prefer or prefer_canonical())
    if not r.found:
        raise FileNotFoundError(f"no .res for {base} elec {elec}")
    return read_res_file(r.path)


def session_path(base, type_, group, variant="", tag=""):
    """Canonical neurosuite-3 path:  <base>.<type>[.<variant>].<group>[.<tag>].

    `variant` (the feature space / method: standard|stderiv|sdiff|...) precedes the
    group; `tag` (the post-group fiber/pipeline STAGE: refine|realigned|...) follows
    it.  Empty slots are omitted.  The tag axis is method-pinned (built directly,
    never probed by resolve_input), so e.g. the refined stderiv sort of group 5 is
        <base>.clu.stderiv.5.refine
    -- variant `stderiv` before the group, fiber stage `refine` after it."""
    p = f"{base}.{type_}"
    if variant:
        p += f".{variant}"
    p += f".{group}"
    if tag:
        p += f".{str(tag).replace('.', '_')}"   # Klusters parses '.' as a field separator, so a
    return p                                     # multi-part STAGE must join with '_' (refine_linked)


def write_res(base, elec, times, variant="", tag=None):
    """Write <base>.res[.<variant>].<elec>[.<tag>] as binary int64.  `tag` is the
    post-group fiber-stage / curation axis (e.g. 'refine' -> <base>.res.<variant>.<elec>.refine,
    'realigned' -> ...res.<elec>.realigned), a separate axis from `variant` and never
    touched by resolve_input."""
    return write_res_file(session_path(base, "res", elec, variant=variant, tag=tag or ""), times)


# ── .clu.N ───────────────────────────────────────────────────────────────────
def read_clu_file(path, n_spikes=None):
    """Read a binary .clu -> (nClusters, ids:int64 ndarray): int32 nClusters
    header + int32 ids (mirrors neurofileio::writeClu / KK::LoadClu binary path).
    `n_spikes` truncates the id array for symmetry with the C++ reader; the file
    size otherwise determines it."""
    raw = np.fromfile(path, dtype=CLU_DTYPE)
    n_clu = int(raw[0]) if raw.size else 0
    ids = raw[1:].astype(np.int64)
    if n_spikes is not None and ids.size != n_spikes:
        ids = ids[:n_spikes]
    return n_clu, ids


def write_clu_file(path, ids, n_clusters=None):
    """Write canonical binary .clu (int32 nClusters header + int32 ids).
    `n_clusters` defaults to max(id)+1 (NeuroSuite convention: ids are the
    header count, 0 = noise).  Mirrors neurofileio::writeClu's binary form."""
    ids = np.asarray(ids, dtype=CLU_DTYPE)
    if n_clusters is None:
        n_clusters = int(ids.max()) + 1 if ids.size else 0
    with open(path, "wb") as f:
        np.array([n_clusters], CLU_DTYPE).tofile(f)
        ids.tofile(f)
    return path


def read_clu(base, elec, n_spikes=None, prefer=None):
    """Resolve and read <base>.clu.<elec> -> (nClusters, ids)."""
    r = resolve_input(base, "clu", elec, prefer or prefer_canonical())
    if not r.found:
        raise FileNotFoundError(f"no .clu for {base} elec {elec}")
    return read_clu_file(r.path, n_spikes=n_spikes)



def resolve_any(base, type_, group, preferred=""):
    """Find the ONE physical copy of a Shared artifact (.res, .spk), whatever
    method token it happens to carry.

    Mirror of custody.hpp::resolveAny and ndm_custody's ndm_resolve_any; the three
    must stay in step, which test/custody_vectors.tsv now enforces.

    resolve_input() is the wrong tool for a Shared artifact: it walks a FIXED
    preference list, so a copy written under a token that list does not name --
    any suffixed token, which no fixed list can enumerate -- is missed, and the
    caller concludes the file is absent when it is sitting right there.

    Order, matching the C++ exactly:
      1. `preferred`, when given
      2. standard, stderiv, sdiff (skipping whichever was preferred)
      3. ANY other method-tagged copy in the directory.  Scanning is what makes a
         suffixed token work without hard-coding it.
      4. the untagged legacy name <base>.<type>.<group>
      5. otherwise not found, with path set to where the preferred copy WOULD be,
         so the caller can name it in an error.

    @return ResolvedInput(path, variant, dotted, found)
    """
    order = [preferred] if preferred else []
    order += [m for m in ("standard", "stderiv", "sdiff") if m != preferred]
    for m in order:
        cand = f"{base}.{type_}.{m}.{group}"
        if os.path.exists(cand):
            return ResolvedInput(cand, m, True, True)

    head = f"{os.path.basename(base)}.{type_}."
    tail = f".{group}"
    d = os.path.dirname(os.path.abspath(base)) or "."
    try:
        names = sorted(os.listdir(d))
    except OSError:
        names = []
    for name in names:
        if not (name.startswith(head) and name.endswith(tail)):
            continue
        tok = name[len(head):len(name) - len(tail)]
        if not tok or "." in tok:            # a longer stage tag, not a bare token
            continue
        return ResolvedInput(os.path.join(d, name), tok, True, True)

    untagged = f"{base}.{type_}.{group}"
    if os.path.exists(untagged):
        return ResolvedInput(untagged, "", False, True)

    m = preferred or "standard"
    return ResolvedInput(f"{base}.{type_}.{m}.{group}", preferred, True, False)

def sibling_variants(base, type_, group, tag="", family=None):
    """Method tokens that actually exist on disk for <base>.<type>.*.<group>[.<tag>].

    The tag axis is deliberately method-pinned -- session_path/read_clu_at build the
    path and never probe -- so a stale --clu-method just misses.  This does not add
    a fallback (silently substituting a different method is exactly what
    PROJECT-INSTRUCTIONS warns against); it only lets the FAILURE say what is there.

    Every module still defaults its method flag to the bare 'stderiv', which
    predates the <family>_<kind><order> token grammar: a session extracted under a
    custom sdiffPairs pattern has .clu.stderiv_C5.5, no .clu.stderiv.5, so the
    default misses on every one of them.  Naming the real token in the error turns
    a bare 'file not found' into an instruction.

    @param family  if set, only tokens of that family are returned.
    """
    g = str(group)
    suffix = f".{str(tag).replace('.', '_')}" if tag else ""
    pat = f"{base}.{type_}.*.{g}{suffix}"
    head = f"{base}.{type_}."
    tail = f".{g}{suffix}"
    out = []
    for m in sorted(glob.glob(pat)):
        if not (m.startswith(head) and m.endswith(tail)):
            continue
        token = m[len(head):-len(tail)]
        if "." in token:                       # a longer tag, not a method token
            continue
        if family and variant_family(token) != family:
            continue
        out.append(token)
    return out


def _pinned_miss_message(kind, path, base, type_, group, variant, tag):
    """Uniform 'method-pinned read missed' error that names the tokens on disk."""
    msg = (f"no .{kind} at {path} (variant={variant!r}, tag={tag!r})")
    same = sibling_variants(base, type_, group, tag=tag, family=variant_family(variant))
    other = [t for t in sibling_variants(base, type_, group, tag=tag) if t not in same]
    if same:
        msg += (f"; this session has {', '.join(same)} for that stage -- "
                f"pass the exact token (e.g. --clu-method {same[0]})")
    elif other:
        msg += f"; tokens present for that stage: {', '.join(other)}"
    else:
        msg += "; no method token exists for that stage at all"
    return msg


def read_clu_at(base, elec, variant="", tag="", n_spikes=None):
    """Read a method-pinned staged .clu directly (no resolve_input probing):
    <base>.clu[.<variant>].<elec>[.<tag>], e.g. variant='stderiv', tag='refine'
    -> <base>.clu.stderiv.<elec>.refine.  Raises with the expected path if absent."""
    path = session_path(base, "clu", elec, variant=variant, tag=tag)
    if not os.path.exists(path):
        raise FileNotFoundError(
            _pinned_miss_message("clu", path, base, "clu", elec, variant, tag))
    return read_clu_file(path, n_spikes=n_spikes)


def write_clu(base, elec, ids, n_clusters=None, variant="", tag=""):
    """Write <base>.clu[.<variant>].<elec>[.<tag>].  `variant` = method (before group),
    `tag` = fiber stage (after group), e.g. variant='stderiv', tag='refine' ->
    <base>.clu.stderiv.<elec>.refine."""
    return write_clu_file(session_path(base, "clu", elec, variant=variant, tag=tag),
                          ids, n_clusters=n_clusters)


# ── matched .clu + .res pair (mirrors neurofileio::readClusterRes) ───────────
ClusterResData = namedtuple(
    "ClusterResData", ["n_clusters", "ids", "times", "binary", "ok"])


def read_cluster_res(base, elec, prefer=None):
    """Read the matched binary .clu/.res pair.  Returns ClusterResData with ok=False
    on any open/parse/length mismatch (never raises), mirroring the C++ helper."""
    prefer = prefer or prefer_canonical()
    rr = resolve_input(base, "res", elec, prefer)
    rc = resolve_input(base, "clu", elec, prefer)
    if not (rr.found and rc.found):
        return ClusterResData(0, np.empty(0, np.int64), np.empty(0, np.int64), False, False)
    times = read_res_file(rr.path)
    n_clu, ids = read_clu_file(rc.path)
    ok = ids.size == times.size
    return ClusterResData(n_clu, ids, times, True, ok)


# ── .fet.N (binary; mirrors KK::LoadData binary path / readFetBinary) ────────
FetBinaryFile = namedtuple("FetBinaryFile", ["n_features", "n_spikes", "values", "ok"])


def read_fet_file(path):
    """Read a binary .fet: int32 feature-count header, then nSpikes × nFeatures
    int64 values row-major.  Returns FetBinaryFile (values shaped (nSpikes,
    nFeatures)).  Mirrors neurofileio::readFetBinary / KK::LoadData."""
    with open(path, "rb") as f:
        hdr = np.fromfile(f, dtype=FET_HDR_DTYPE, count=1)
        if hdr.size != 1 or int(hdr[0]) < 1:
            return FetBinaryFile(0, 0, np.empty((0, 0), np.int64), False)
        n_feat = int(hdr[0])
        body = np.fromfile(f, dtype=FET_DTYPE)
    if body.size % n_feat != 0:
        return FetBinaryFile(n_feat, 0, np.empty((0, n_feat), np.int64), False)
    n_spk = body.size // n_feat
    return FetBinaryFile(n_feat, n_spk, body.reshape(n_spk, n_feat), True)


def read_fet(base, elec, prefer=None):
    """Resolve and read <base>.fet.<elec> (binary), defaulting to canonical-first
    to match KlustaKwik's pickInputPath (.fet then .fetD)."""
    r = resolve_input(base, "fet", elec, prefer or prefer_canonical())
    if not r.found:
        raise FileNotFoundError(f"no .fet for {base} elec {elec}")
    return read_fet_file(r.path)


def write_fet_file(path, values):
    """Write a binary .fet: int32 feature-count header + nSpikes x nFeatures
    int64 values row-major (inverse of read_fet_file)."""
    values = np.asarray(values, dtype=FET_DTYPE)
    if values.ndim != 2:
        raise ValueError("write_fet_file expects (nSpikes, nFeatures)")
    with open(path, "wb") as f:
        np.array([values.shape[1]], FET_HDR_DTYPE).tofile(f)
        values.tofile(f)
    return path


def write_fet(base, elec, values, variant="", tag=""):
    """Write <base>.fet[.<variant>].<elec>[.<tag>] (binary), mirroring write_res/write_clu.
    `variant` = method (before group), `tag` = fiber stage (after group)."""
    return write_fet_file(session_path(base, "fet", elec, variant=variant, tag=tag), values)


# ── .spk.N / .spkD.N (int16, sample-major, no header) ────────────────────────
def open_spk_file(path, nsamp, nchan, mode="r"):
    """Memmap a .spk/.spkD file as (nSpikes, nsamp, nchan) int16, sample-major.
    Only whole spikes are exposed (trailing partial spike, if any, is dropped)."""
    mm = np.memmap(path, dtype=SPK_DTYPE, mode=mode)
    n = mm.size // (nsamp * nchan)
    return mm[:n * nsamp * nchan].reshape(n, nsamp, nchan)


def open_spk(base, elec, nsamp, nchan, prefer=None, mode="r"):
    """Resolve and memmap waveforms for (base, elec).  Defaults to prefer_derived()
    (stderiv/D before canonical), so a stderiv run picks up .spk.stderiv.N /
    .spk.D.N / legacy .spkD.N before the raw .spk.N — preserving fiber-kit's
    historical '.spkD then .spk' preference while also honouring the dotted form.
    Returns (memmap (n,nsamp,nchan), ResolvedInput)."""
    r = resolve_input(base, "spk", elec, prefer or prefer_derived())
    if not r.found:
        raise FileNotFoundError(f"no .spkD/.spk for {base} elec {elec}")
    return open_spk_file(r.path, nsamp, nchan, mode=mode), r


def open_spkD(base, elec, nsamp, nch, mode="r"):
    """Back-compat shim: same contract as the historical fiber_session.open_spkD
    — returns (memmap, path), preferring the derived (stderiv) representation."""
    mm, r = open_spk(base, elec, nsamp, nch, prefer=prefer_derived(), mode=mode)
    return mm, r.path


def open_spk_raw(base, elec, nsamp, nchan, mode="r"):
    """Resolve and memmap the RAW (standard) waveforms: <base>.spk.standard.N (then legacy
    <base>.spk.N).  For position/amplitude work — never returns the stderiv .spk.  Returns
    (memmap (n,nsamp,nchan), ResolvedInput); raises if the resolved file is a stderiv form."""
    mm, r = open_spk(base, elec, nsamp, nchan, prefer=prefer_standard(), mode=mode)
    if r.variant in ("stderiv", "D"):
        raise FileNotFoundError(f"no raw/standard .spk for {base} elec {elec} "
                                f"(resolved {r.path}; refusing stderiv for amplitude work)")
    return mm, r


def write_spk_file(path, waves):
    """Write a .spk/.spkD: int16, sample-major, no header (inverse of
    open_spk_file).  `waves` is (nSpikes, nsamp, nchan)."""
    np.asarray(waves, dtype=SPK_DTYPE).tofile(path)
    return path


def write_spk(base, elec, waves, variant="", tag=""):
    """Write <base>.spk[.<variant>].<elec>[.<tag>] (int16, sample-major).  `variant` =
    method (before group), `tag` = fiber stage (after group)."""
    return write_spk_file(session_path(base, "spk", elec, variant=variant, tag=tag), waves)


# ── group-wide spike-count edit (dedup propagation) ──────────────────────────
# Stage tokens (the dotted segments AFTER the electrode) that mark a file as NOT a
# live per-spike artefact and so must never be edited by a dedup propagation:
#   * a backup snapshot ('..._bkp')
#   * a byte-split fragment ('.part.aa', ...)
#   * a sidecar with its own row semantics ('.units.npz', ...)
# Dated snapshots ('.06.14.2026.12.35') are caught separately by the all-digit test:
# a frozen backup of the pre-dedup state has exactly n_orig rows, so the row-count
# guard alone CANNOT distinguish it from a live file -- the name must.
_PERSPIKE_SIDECAR = {"npz", "npy", "gz", "zip", "bak", "tmp", "swp", "units"}


def _is_live_perspike(stage):
    """True iff the stage tokens (everything after the electrode) name a live per-spike
    file -- not a dated/explicit backup, byte-split fragment, or non-binary sidecar."""
    for tok in stage:
        if tok.isdigit():                 # dated-snapshot component, e.g. 06.14.2026.12.35
            return False
        if tok == "part":                 # byte-split fragment (.part.aa)
            return False
        if tok.endswith("bkp"):           # explicit backup (refine_linked_bkp)
            return False
        if tok in _PERSPIKE_SIDECAR:      # .units.npz and friends
            return False
    return True


def apply_spike_keep(base, elec, keep, n_orig, nsamp, nchan,
                     types=("res", "clu", "spk", "spkD", "fet", "fetD"), verbose=True, strict=True,
                     stale=None):
    """Subset EVERY live per-spike file of one spike group to `keep`, in place, so the
    .res / .clu / .spk(/.spkD) / .fet(/.fetD) of the group stay row-aligned after a dedup --
    across all variants (standard|stderiv|...) and post-group stages.

    All per-spike files are BINARY; .clu/.res are read here directly with their fixed binary
    dtype (CLU_DTYPE int32 header+ids, RES_DTYPE int64), never via a text/binary heuristic, so a
    binary file whose header byte happens to be an ASCII digit (e.g. nClusters=1590 -> leading
    byte 0x36 '6') is never mis-parsed as text.

    The unit of work is the STAGE (the tokens after the electrode: '' = the base over-cluster,
    'refine' = the refined stage, ...), not the individual file -- a stage's res/clu/spk/fet are one
    spike set and must move together.  The base stage is the set `keep` is for: its files at `n_orig`
    are subset to `nkeep`, files already at `nkeep` are left (idempotent).  A STAGED stage that is
    internally consistent at `n_orig` is likewise subset (so a derived stage like 'refine' gets
    deduped too, not just the base).  A staged stage at any other count -- or internally inconsistent
    (e.g. clu at n_orig but spk at a third count, a stale leftover from an earlier extraction) -- is
    a different/broken spike set that CANNOT be subset by `keep`; the WHOLE stage is handled together
    so it is never left half-deduped.  By DEFAULT such a stale stage is quarantined aside as
    <file>.stalebkp (non-destructive; the live group stays consistent and the owning stage regenerates
    it).  Pass stale='error' to hard-fail, or stale='skip'/--no-dedup-strict to leave it.  `keep` is a
    boolean mask or index array over the original n_orig spikes.
    Returns (rewritten_paths, skipped:[(path, rows)])."""
    import glob
    import os
    from collections import OrderedDict
    keep = np.asarray(keep)
    nkeep = int(keep.sum()) if keep.dtype == bool else len(keep)
    policy = stale if stale in ("error", "skip", "quarantine") else ("quarantine" if strict else "skip")
    elec = str(elec)
    bname = os.path.basename(base)

    def _rows(path, t):                                    # spike-axis length, fixed binary dtype per type
        if t == "res":
            return int(np.fromfile(path, dtype=RES_DTYPE).size)
        if t == "clu":
            return int(max(np.fromfile(path, dtype=CLU_DTYPE).size - 1, 0))
        if t in ("spk", "spkD"):
            return int(open_spk_file(path, nsamp, nchan).shape[0])
        if t in ("fet", "fetD"):
            f = read_fet_file(path)
            return int(f.n_spikes) if f.ok else None
        return None

    def _subset(path, t):                                  # rewrite path in place, keeping `keep` rows
        if t == "res":
            write_res_file(path, np.fromfile(path, dtype=RES_DTYPE)[keep])
        elif t == "clu":
            raw = np.fromfile(path, dtype=CLU_DTYPE)
            write_clu_file(path, raw[1:][keep].astype(np.int64), n_clusters=int(raw[0]))
        elif t in ("spk", "spkD"):
            write_spk_file(path, np.asarray(open_spk_file(path, nsamp, nchan)[keep]))   # materialise before truncate
        elif t in ("fet", "fetD"):
            write_fet_file(path, read_fet_file(path).values[keep])

    # ── group live per-spike files by stage (variant before the electrode is ignored: standard
    #    and stderiv are the same spike set per stage) ──
    stages = OrderedDict()                                 # stage_tuple -> [(path, type)]
    excluded = 0
    seen = set()
    for t in types:
        for path in sorted(glob.glob(f"{glob.escape(base)}.{t}.*")):
            tail = os.path.basename(path)[len(bname) + 1:].split(".")  # <type>[.<variant>].<elec>[.<stage>]
            if not tail or tail[0] != t:
                continue
            # the electrode is the first all-digit token (variants are non-numeric); everything
            # after it is the stage.  Require an EXACT group match so group 5 never picks up 15.
            ei = next((i for i in range(1, len(tail)) if tail[i].isdigit()), None)
            if ei is None or tail[ei] != elec:
                continue
            stage = tail[ei + 1:]
            if not _is_live_perspike(stage):
                excluded += 1                              # backup / fragment / sidecar -- never touch
                continue
            if path in seen:
                continue
            seen.add(path)
            stages.setdefault(tuple(stage), []).append((path, t))

    rewritten = []
    done = []                                              # already at the deduped count (idempotent)
    orphans = []                                           # whole stale/inconsistent stages, file by file
    for stage, files in stages.items():
        counts = {}
        for path, t in files:
            try:
                counts[(path, t)] = _rows(path, t)
            except Exception as e:  # noqa: BLE001 -- a single unreadable file marks its stage stale, not aborts
                counts[(path, t)] = f"error: {e}"
        vals = set(counts.values())
        is_base = (len(stage) == 0)
        if is_base:
            # the base stage is the set `keep` is for; subset its n_orig files, leave nkeep, orphan the rest
            for (path, t), c in counts.items():
                if c == n_orig:
                    try:
                        _subset(path, t); rewritten.append(path)
                    except Exception as e:  # noqa: BLE001
                        orphans.append((path, f"error: {e}"))
                elif c == nkeep:
                    done.append((path, c))
                else:
                    orphans.append((path, c))
        elif vals == {n_orig}:
            for (path, t), c in counts.items():            # consistent derived stage -> dedup it too
                try:
                    _subset(path, t); rewritten.append(path)
                except Exception as e:  # noqa: BLE001
                    orphans.append((path, f"error: {e}"))
        elif vals == {nkeep}:
            done.extend((path, c) for (path, t), c in counts.items())
        else:                                              # stale / inconsistent stage -> whole stage as a unit
            orphans.extend((path, c) for (path, t), c in counts.items())

    if verbose:
        if rewritten:
            print(f"dedup: rewrote {len(rewritten)} group file(s) to {nkeep} spikes: "
                  + ", ".join(os.path.basename(p) for p in rewritten))
        for p, c in done:
            print(f"dedup: already deduped, left {os.path.basename(p)} (rows={c})")
        if excluded:
            print(f"dedup: left {excluded} backup/fragment/sidecar file(s) untouched")
    # stale stages: their rows do not correspond 1:1 to this spike set, so they cannot be subset by `keep`.
    if orphans and policy == "quarantine":
        moved = []
        for p, c in orphans:
            dst = p + ".stalebkp"                 # excluded from _is_live_perspike(): never re-globbed live
            try:
                os.replace(p, dst)
            except OSError as e:
                raise RuntimeError(f"dedup: could not quarantine stale {os.path.basename(p)}: {e} "
                                   f"(would otherwise leave group {elec} half-deduped)")
            moved.append((dst, c))
        if verbose:
            for dst, c in moved:
                print(f"dedup: quarantined stale {os.path.basename(dst)[:-9]} (rows={c}) "
                      f"-> {os.path.basename(dst)}")
        orphans = []
    elif verbose:
        for p, c in orphans:
            print(f"dedup: STALE/misaligned {os.path.basename(p)} (rows={c}, expected {n_orig} or {nkeep})")
    if orphans and policy == "error":
        names = ", ".join(os.path.basename(p) for p, _ in orphans)
        raise RuntimeError(
            f"dedup: {len(orphans)} live per-spike file(s) of group {elec} are misaligned "
            f"(rows != {n_orig} and != {nkeep}) and cannot be subset by the keep-mask: {names}. "
            f"These are stale from an earlier run -- regenerate, remove, or quarantine them and re-run "
            f"(--dedup-stale quarantine moves them aside as <file>.stalebkp; --no-dedup-strict skips them).")
    return rewritten, done + orphans


# ── .dat / .fil / .lfp (interleaved int16) ───────────────────────────────────
def open_signal(path, nchan, mode="r"):
    """Memmap an interleaved int16 wideband/LFP file as (nSamples, nchan).
    Used for .fil/.dat baseline reads (neurofileio::readDatWindow analogue)."""
    return np.memmap(path, dtype=SPK_DTYPE, mode=mode).reshape(-1, nchan)


# ── .fibers.<method>.<elec> (dotted-variant npz) ─────────────────────────────

# ── the clu-resolution preamble, once ────────────────────────────────────────
def add_clu_args(ap, *, method_default="stderiv", stage_default="refine",
                 stage_dest="variant", stage_alias="--variant", in_clu=True,
                 method_help=None, stage_help=None, in_clu_help=None):
    """Declare the three flags every clu-consuming stage needs.

    Nineteen stages were declaring --clu-method / --clu-stage / --in-clu by hand
    and then repeating the same four-line resolution below.  That duplication is
    what let the two --variant conventions diverge in the first place: with the
    names, aliases and dests fixed in one place they could not have disagreed.
    Mirrors session_yaml.add_session_args, which already does this for the
    session flags.

    Note --variant aliases the STAGE here, not the method: that is the legacy
    spelling these stages shipped with, and moving it would break every existing
    invocation.  New stages should pass stage_alias=None and use --clu-stage.

    @param stage_dest  attribute the stage tag lands on; "variant" for the
                       existing stages, which read a.variant throughout.
    """
    ap.add_argument("--clu-method", default=method_default,
                    help=method_help or "method the clu stems from: "
                                        "standard | stderiv | stderiv_C5")
    stage_flags = ["--clu-stage"] + ([stage_alias] if stage_alias else [])
    ap.add_argument(*stage_flags, dest=stage_dest, default=stage_default,
                    help=stage_help or "post-fiber stage tag at the end of the .clu name")
    if in_clu:
        ap.add_argument("--in-clu", default=None,
                        help=in_clu_help or "explicit .clu path, overriding "
                                            "--clu-method/--clu-stage")


def resolve_clu(a, base, elec, n_spikes=None, *, stage_dest="variant"):
    """Read the clu a stage was pointed at: --in-clu if given, else the
    method/stage pair.  Returns (n_clusters, labels).

    The explicit path wins because it is the escape hatch for a file that does
    not follow the naming; everything else goes through the method-pinned
    read_clu_at, which refuses rather than guessing and now names the tokens
    present when it misses.
    """
    in_clu = getattr(a, "in_clu", None)
    if in_clu:
        return read_clu_file(in_clu, n_spikes=n_spikes)
    return read_clu_at(base, elec, variant=a.clu_method,
                       tag=getattr(a, stage_dest), n_spikes=n_spikes)

def fibers_path(base, method, elec, stage=""):
    """Path of the per-(chunk,fiber) geometry table, in the canonical form
    <base>.fibers.<method>.<group>[.<stage>].

    `method` is the operation the table's clusters stem from (standard | stderiv |
    stderiv_C5) and occupies the slot BEFORE the group; `stage` is the post-fiber
    tag and follows the group.  Built through session_path so this file obeys the
    same two-slot rule as .clu/.fet/.spk rather than a private one.

    The stage slot exists because both writers of this file were previously
    squeezing two different things into `method`: fiber_session passed the real
    method ('stderiv'), while fiber_stats passed its clu STAGE ('refine') and so
    produced <base>.fibers.refine.<group> -- a stage sitting in the method slot,
    and a name fiber_session's could never be matched against.
    """
    return session_path(base, "fibers", elec, variant=method, tag=stage)
