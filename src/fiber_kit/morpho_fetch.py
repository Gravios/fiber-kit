#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════════════════════
#  morpho_fetch.py — acquire reconstructions, validate them, record provenance.
#
#  Adding a morphology to this project is not "download a file".  Three things
#  have gone wrong already in ways that only showed up much later:
#
#    * a dentate basket cell was used where a CA1 one was needed, and the region
#      substitution had to be carried as a caveat through five patches;
#    * four of twelve NeuroMorpho files tested here have NO AXON, and for this
#      work a dendrite-only interneuron is useless -- the axon carries 46-109%
#      of the off-peak extracellular field;
#    * a reconstruction loaded as 977 disconnected roots because of a parser
#      gap, and only failed loudly because morpho_geom refuses that.
#
#  So every acquisition goes through a validation GATE and lands in a manifest
#  that records what it is and where it came from.  A morphology that is not in
#  the manifest is not a morphology this project uses, and a result naming a
#  cell can be traced to its archive, species and region without asking anyone.
#
#  ═══ NETWORK CODE HERE IS UNVERIFIED ═══
#  neuromorpho.org is unreachable from the environment this was written in --
#  the egress proxy blocks the host and its robots.txt disallows automated
#  access -- so search() and download() have never been executed against the
#  live service.  Their URL construction follows the documented API and an
#  observed file URL, and is unit-tested against those; the HTTP round trip is
#  not.  Everything downstream of a file existing on disk IS tested, which is
#  the path that matters if you fetch by hand.
# ════════════════════════════════════════════════════════════════════════════
import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request

BASE = "https://neuromorpho.org"
API = BASE + "/api/neuron"
MANIFEST = "morphologies.tsv"
UA = "fiber-kit/morpho_fetch (research use; contact via repository)"

# Fields kept in the manifest.  Chosen so a caveat can be checked rather than
# remembered: region and species catch the dentate-for-CA1 substitution, and
# the axonal length catches a dendrite-only reconstruction.
FIELDS = ("neuron_name", "archive", "species", "brain_region", "cell_type",
          "note", "axon_um", "dend_um", "n_sections", "n_points", "sha1", "url")


def _get(url, timeout=30, retries=3, pause=1.0):
    """GET with a retry and a declared user agent.

    NeuroMorpho is a shared academic service; the pause and the identifying
    agent are the minimum courtesy for a script that may issue hundreds of
    requests, and the retry is because a shared service returns transient 5xx.
    """
    last = None
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as fh:
                return fh.read()
        except Exception as e:                        # noqa: BLE001
            last = e
            time.sleep(pause * (k + 1))
    raise RuntimeError(f"GET failed after {retries} tries: {url}\n  {last}")


def search(query=None, filters=None, size=100, max_pages=20, pause=0.5):
    """Query the NeuroMorpho REST API, following pagination.

    filters is a list of "field:value" strings passed as repeated fq
    parameters, e.g. ["brain_region:CA1", "cell_type:interneuron"].  Returned
    records are the API's own dicts, unmodified -- this does not reshape them,
    so a field that later matters is still there.
    """
    out = []
    for page in range(int(max_pages)):
        params = [("q", query or "*:*"), ("size", str(int(size))), ("page", str(page))]
        for f in (filters or []):
            params.append(("fq", f))
        url = f"{API}/select?" + urllib.parse.urlencode(params)
        data = json.loads(_get(url))
        emb = data.get("_embedded", {}).get("neuronResources", [])
        out.extend(emb)
        pg = data.get("page", {})
        if not emb or page + 1 >= int(pg.get("totalPages", 1)):
            break
        time.sleep(pause)
    return out


def swc_urls(rec):
    """Candidate download URLs for a record, standardised version first.

    Two candidates because the archive path segment is the archive name
    lower-cased with spaces removed, which is inferred from one observed URL
    (`dableFiles/hamad/CNG version/int27_3_1.CNG.swc`) and not from
    documentation.  If a multi-word archive breaks that rule the second form
    -- the raw archive string -- is tried, and if both fail the caller should
    fall back to the neuron's own page, which carries the link explicitly.
    """
    name = rec.get("neuron_name")
    arch = rec.get("archive") or ""
    if not name or not arch:
        raise ValueError(f"record lacks neuron_name/archive: {sorted(rec)[:6]}")
    outs = []
    for slug in (arch.lower().replace(" ", ""), arch):
        for sub, suffix in (("CNG version", ".CNG.swc"), ("Source-Version", ".swc")):
            outs.append(f"{BASE}/dableFiles/{urllib.parse.quote(slug)}/"
                        f"{urllib.parse.quote(sub)}/{urllib.parse.quote(name)}{suffix}")
    seen, uniq = set(), []
    for u in outs:
        if u not in seen:
            seen.add(u); uniq.append(u)
    return uniq


def download(rec, dest, overwrite=False):
    """Fetch one reconstruction, trying each candidate URL.  Returns the path."""
    name = rec["neuron_name"]
    path = os.path.join(dest, f"{name}.CNG.swc")
    if os.path.exists(path) and not overwrite:
        return path, None
    errs = []
    for url in swc_urls(rec):
        try:
            body = _get(url)
        except Exception as e:                        # noqa: BLE001
            errs.append(f"{url}: {e}"); continue
        if b"#" not in body[:4096] and b" " not in body[:4096]:
            errs.append(f"{url}: does not look like SWC"); continue
        os.makedirs(dest, exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(body)
        return path, url
    raise RuntimeError("no candidate URL worked:\n  " + "\n  ".join(errs))


# ── validation ──────────────────────────────────────────────────────────────
def validate(path, min_axon_um=0.0, max_comp=4000, d_lambda=0.25):
    """Load a reconstruction and report what it is; raise if unusable.

    The two hard failures are the ones experience has produced: a morphology
    that does not form a single tree (which morpho_geom refuses rather than
    simulating as fragments), and one whose axon is missing when the caller
    said the axon matters.  Everything else is reported for the manifest and
    left to the caller's judgement.
    """
    try:
        from . import morpho_geom as mg
    except ImportError:
        import morpho_geom as mg
    secs = mg.load(path)
    c = mg.compartmentalize(secs, d_lambda=d_lambda, max_comp=max_comp)
    axon = float(c.L[c.type == mg.AXON].sum())
    dend = float(c.L[(c.type == mg.BASAL) | (c.type == mg.APICAL)].sum())
    if min_axon_um > 0 and axon < min_axon_um:
        raise ValueError(f"{os.path.basename(path)}: axon is {axon:.0f} um, below the "
                         f"{min_axon_um:.0f} um required — a dendrite-only reconstruction "
                         f"cannot carry the off-peak extracellular field")
    npts = sum(1 for ln in open(path, errors="ignore")
               if ln.strip() and not ln.lstrip().startswith("#"))
    return dict(n_sections=len(secs), n_comps=len(c), axon_um=axon, dend_um=dend,
                n_points=npts, total_um=float(c.L.sum()),
                y_min=float(c.mid[:, 1].min()), y_max=float(c.mid[:, 1].max()),
                roots=int((c.parent < 0).sum()))


def sha1_of(path, nbytes=1 << 20):
    import hashlib
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(nbytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ── manifest ────────────────────────────────────────────────────────────────
def read_manifest(path):
    if not os.path.exists(path):
        return []
    rows, hdr = [], None
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            f = line.split("\t")
            if hdr is None:
                hdr = f; continue
            rows.append(dict(zip(hdr, f)))
    return rows


def write_manifest(path, rows):
    """Rewrite the manifest, sorted and de-duplicated by neuron_name.

    Sorted so a diff between two runs shows what changed rather than how the
    API happened to order its response.
    """
    by = {}
    for r in rows:
        by[r.get("neuron_name", "")] = r
    with open(path, "w") as fh:
        fh.write("# morphologies used by fiber-morpho — written by morpho_fetch.\n")
        fh.write("# axon_um is the gate: a dendrite-only reconstruction cannot carry\n")
        fh.write("# the off-peak extracellular field and must not be used for one.\n")
        fh.write("\t".join(FIELDS) + "\n")
        for k in sorted(by):
            fh.write("\t".join(str(by[k].get(c, "")) for c in FIELDS) + "\n")
    return len(by)


def adopt(paths, manifest, meta=None, min_axon_um=0.0):
    """Validate local files and add them to the manifest.

    This is the path that does not need the network, and the one to use for
    files fetched by hand.  Metadata that only the archive knows -- species,
    region, cell type -- is taken from `meta` if supplied and left blank
    otherwise, blank being honest rather than guessed.
    """
    rows = read_manifest(manifest)
    added, failed = [], []
    for p in paths:
        try:
            v = validate(p, min_axon_um=min_axon_um)
        except Exception as e:                        # noqa: BLE001
            failed.append((p, str(e))); continue
        name = os.path.basename(p).replace(".CNG.swc", "").replace(".swc", "")
        r = dict((meta or {}).get(name, {}))
        r.update(neuron_name=name, axon_um="%.0f" % v["axon_um"],
                 dend_um="%.0f" % v["dend_um"], n_sections=v["n_sections"],
                 n_points=v["n_points"], sha1=sha1_of(p))
        r.setdefault("url", "local")
        rows.append(r); added.append((p, v))
    write_manifest(manifest, rows)
    return added, failed


# ── CLI ─────────────────────────────────────────────────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="fiber-morpho-fetch",
        description="Acquire, validate and record neuronal reconstructions.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search", help="query NeuroMorpho and list matches (no download)")
    s.add_argument("--q", default="*:*")
    s.add_argument("--fq", action="append", default=[],
                   help="repeatable filter, e.g. --fq brain_region:CA1")
    s.add_argument("--size", type=int, default=100)
    s.add_argument("--max-pages", type=int, default=5, dest="max_pages")
    s.add_argument("--json", default=None, help="write the raw records here")

    d = sub.add_parser("fetch", help="download matches into a directory")
    d.add_argument("--q", default="*:*")
    d.add_argument("--fq", action="append", default=[])
    d.add_argument("--size", type=int, default=100)
    d.add_argument("--max-pages", type=int, default=5, dest="max_pages")
    d.add_argument("--dest", required=True)
    d.add_argument("--manifest", default=None)
    d.add_argument("--min-axon", type=float, default=0.0, dest="min_axon")
    d.add_argument("--limit", type=int, default=0)

    a = sub.add_parser("adopt", help="validate local .swc files and record them")
    a.add_argument("files", nargs="+")
    a.add_argument("--manifest", required=True)
    a.add_argument("--min-axon", type=float, default=0.0, dest="min_axon")
    a.add_argument("--meta", default=None, help="json mapping neuron_name -> fields")

    v = sub.add_parser("verify", help="re-validate everything in a manifest")
    v.add_argument("--manifest", required=True)
    v.add_argument("--dir", required=True)

    args = ap.parse_args(argv)

    if args.cmd in ("search", "fetch"):
        recs = search(args.q, args.fq, size=args.size, max_pages=args.max_pages)
        print(f"{len(recs)} records")
        for r in recs[:40]:
            print("  %-28s %-12s %-8s %-22s %s" % (
                r.get("neuron_name", "?")[:28], (r.get("archive") or "")[:12],
                (r.get("species") or "")[:8],
                ",".join(r.get("brain_region") or [])[:22],
                ",".join(r.get("cell_type") or [])[:34]))
        if args.cmd == "search":
            if getattr(args, "json", None):
                json.dump(recs, open(args.json, "w"), indent=1)
                print(f"wrote {args.json}")
            return 0
        if args.limit:
            recs = recs[:args.limit]
        man = args.manifest or os.path.join(args.dest, MANIFEST)
        rows = read_manifest(man)
        okn = bad = 0
        for r in recs:
            try:
                path, url = download(r, args.dest)
                v_ = validate(path, min_axon_um=args.min_axon)
            except Exception as e:                    # noqa: BLE001
                print("  SKIP %-26s %s" % (r.get("neuron_name", "?")[:26], str(e)[:70]))
                bad += 1; continue
            rows.append(dict(neuron_name=r.get("neuron_name"), archive=r.get("archive"),
                             species=r.get("species"),
                             brain_region=",".join(r.get("brain_region") or []),
                             cell_type=",".join(r.get("cell_type") or []),
                             note=r.get("note", ""), axon_um="%.0f" % v_["axon_um"],
                             dend_um="%.0f" % v_["dend_um"], n_sections=v_["n_sections"],
                             n_points=v_["n_points"], sha1=sha1_of(path), url=url or ""))
            print("  ok   %-26s axon %8.0f um  dend %8.0f um  %d sections"
                  % (r.get("neuron_name", "?")[:26], v_["axon_um"], v_["dend_um"],
                     v_["n_sections"]))
            okn += 1
        n = write_manifest(man, rows)
        print(f"\n{okn} added, {bad} skipped; manifest now lists {n} ({man})")
        return 0

    if args.cmd == "adopt":
        meta = json.load(open(args.meta)) if args.meta else None
        added, failed = adopt(args.files, args.manifest, meta, args.min_axon)
        for p, v_ in added:
            print("  ok   %-30s %5d sections  axon %8.0f um  dend %8.0f um"
                  % (os.path.basename(p)[:30], v_["n_sections"], v_["axon_um"], v_["dend_um"]))
        for p, e in failed:
            print("  FAIL %-30s %s" % (os.path.basename(p)[:30], e[:70]))
        print(f"\n{len(added)} adopted, {len(failed)} rejected -> {args.manifest}")
        return 1 if failed and not added else 0

    rows = read_manifest(args.manifest)
    bad = 0
    for r in rows:
        p = os.path.join(args.dir, r["neuron_name"] + ".CNG.swc")
        if not os.path.exists(p):
            p = os.path.join(args.dir, r["neuron_name"] + ".swc")
        if not os.path.exists(p):
            print("  MISSING %s" % r["neuron_name"]); bad += 1; continue
        h = sha1_of(p)
        if r.get("sha1") and h != r["sha1"]:
            print("  CHANGED %-28s manifest %s, file %s"
                  % (r["neuron_name"][:28], r["sha1"][:10], h[:10])); bad += 1; continue
        print("  ok      %-28s axon %8s um" % (r["neuron_name"][:28], r.get("axon_um", "?")))
    print(f"\n{len(rows) - bad}/{len(rows)} verified")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
