#!/usr/bin/env python3
# test_pca_basis_resolution.py — read_cluster_basis must survive the _D alias.
#
# fiber-session --out-variant <method>_D<lag><dims> --emit-pca writes the lag
# feature-space basis BESIDE the extraction-method basis, so a lag-alias session's
# normal on-disk layout is two same-family .pca files ({stderiv_C5,
# stderiv_C5_D34}).  resolve_input's family scan of a bare token deliberately
# raises on that ("pass the exact token"), and read_cluster_basis used to walk the
# bare token FIRST -- so the raise aborted the walk before the exact tokens it had
# itself discovered were ever tried, and every family request came back None: the
# fine split silently fell back to a local SVD, --emit-fet skipped, and
# anchor-link's --feat-lag scored the raw template.  These checks pin the repaired
# resolution order and, as the negative control, that resolve_input's own
# ambiguity contract is still intact underneath it.
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

try:
    from fiber_kit import fiber_pca as fp
    from fiber_kit import neuro_io as nio
except ImportError:
    sys.path.insert(0, os.path.join(HERE, "..", "src", "fiber_kit"))
    import fiber_pca as fp
    import neuro_io as nio

fails = 0
ran = 0


def check(ok, what):
    global fails, ran
    ran += 1
    if not ok:
        fails += 1
        print(f"FAIL  {what}")
    else:
        print(f"ok    {what}")


def _write_basis(base, tok, elec=6, nch=7, ncomp=2, d2u=25, rs=8):
    means = np.zeros((nch, d2u))
    evec = np.random.default_rng(0).normal(size=(nch, ncomp, d2u))
    fp.write_pcad(nio.session_path(base, "pca", elec, variant=tok), means, evec, rs)


def _write_alias(base, from_tok, lag=3, elec=6):
    b = fp.lag_basis(fp.read_pcad(nio.session_path(base, "pca", elec, variant=from_tok)), lag)
    fp.write_pcad(nio.session_path(base, "pca", elec, variant=f"{from_tok}_D{lag}4"),
                  b["means"], b["evec"], b["recShift"], centered=0)


# ── 1. the lag-alias session layout: method basis + its _D alias ─────────────
with tempfile.TemporaryDirectory() as d:
    base = os.path.join(d, "sess")
    _write_basis(base, "stderiv_C5")
    _write_alias(base, "stderiv_C5")
    r = fp.read_cluster_basis(base, 6, "stderiv")
    check(r is not None and r["_path"].endswith(".pca.stderiv_C5.6"),
          "family request beside the alias resolves the METHOD basis "
          f"(got {os.path.basename(r['_path']) if r else None})")
    # the exact-token request keeps working too
    r = fp.read_cluster_basis(base, 6, "stderiv_C5_D34")
    check(r is not None and r["_path"].endswith(".pca.stderiv_C5_D34.6"),
          "the exact alias token still resolves the alias")

# ── 2. an exact family file outranks every suffixed token ────────────────────
with tempfile.TemporaryDirectory() as d:
    base = os.path.join(d, "sess")
    _write_basis(base, "stderiv")
    _write_alias(base, "stderiv")
    r = fp.read_cluster_basis(base, 6, "stderiv")
    check(r is not None and r["_path"].endswith(".pca.stderiv.6"),
          "exact .pca.stderiv.<g> wins over the alias")

# ── 3. only the alias on disk: reachable, and self-describing for the caller ─
with tempfile.TemporaryDirectory() as d:
    base = os.path.join(d, "sess")
    _write_basis(base, "stderiv_C5")
    _write_alias(base, "stderiv_C5")
    os.remove(nio.session_path(base, "pca", 6, variant="stderiv_C5"))
    r = fp.read_cluster_basis(base, 6, "stderiv")
    check(r is not None and r.get("_variant") == "stderiv_C5_D34",
          "alias-only family request returns the alias, _variant carries its token")

# ── 4. negative control: resolve_input's ambiguity contract is intact ────────
# The fix must live in read_cluster_basis's list construction, NOT in weakening
# resolve_input: a bare family token over several members still raises there.
with tempfile.TemporaryDirectory() as d:
    base = os.path.join(d, "sess")
    _write_basis(base, "stderiv_C5")
    _write_alias(base, "stderiv_C5")
    try:
        nio.resolve_input(base, "pca", 6, ["stderiv"])
        check(False, "resolve_input bare-family over two members raises")
    except ValueError:
        check(True, "resolve_input bare-family over two members raises")

# ── 5. the scan orders method tokens before _D aliases ───────────────────────
with tempfile.TemporaryDirectory() as d:
    base = os.path.join(d, "sess")
    _write_basis(base, "stderiv_C5")
    _write_alias(base, "stderiv_C5")
    _write_basis(base, "stderiv_S3")
    toks = fp._same_family_variants(base, 6, "stderiv")
    check(toks and nio.parse_variant_token(toks[0]).lag == 0 and toks[-1] == "stderiv_C5_D34",
          f"_same_family_variants lists method tokens first ({toks})")

print(f"\n{ran - fails}/{ran} passed")
sys.exit(1 if fails else 0)
