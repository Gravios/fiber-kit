#!/usr/bin/env python3
# ══════════════════════════════════════════════════════════════════════════════
#  fiber_spkc.py  —  collision rows with their partners subtracted (the .spkc)
#
#  A spike row whose .eap row names two (or more) template classes holds two
#  units' spikes.  What a reader should see of it is the record less the
#  partners:
#        corrected = spk[i] - sum over partners of a_p * T_p(tau_p)
#  This module is the numpy mirror of neurosuite-3's core/decollide_fit.hpp and
#  of Klusters' rebuildCollisionRows (klustersdoc_spkc.cpp), so a .spkc written
#  by the pipeline holds what Klusters would build from the same files.
#
#    * the .spk is NOT changed: it stays the raw cut of the recording.  The
#      corrected rows go to the sparse <base>.spkc.<method>.<group>[.<stage>]
#      (neuro_io.write_spkc): a row index and those rows' records;
#    * the .eap says which partners and about where.  The amplitude and the exact
#      shift are FITTED here, jointly with the row's own template, within
#      `radius` samples of the stored offset and to a fraction of a sample; a
#      partner is subtracted only for a plausible amplitude (amp_min..amp_max)
#      that removes at least min_gain of the residual energy;
#    * which cluster a class is, is read off the membership (home_clusters): the
#      cluster holding most of the rows that name only that class;
#    * a unit's template is the mean record of its rows that are NOT collisions,
#      the 300 of them nearest in time to the row.
#
#  The subtraction is linear, so it is run per waveform variant with that
#  variant's .spk and templates.
#
#  Parity with the C++ (2026-11-23, measured outside the test suite, which
#  cannot carry a C++ build): on 2,000 random cases fit_partner made the same
#  accept decision and found the same offset as decollide_fit::fitPartner, the
#  shift within 2e-10 sample and the amplitude within 1e-13, and the corrected
#  int16 records were identical; and on a synthetic session with 40 planted
#  collisions `fiber-spkc` wrote the .spkc Klusters wrote, row for row and
#  record for record.
# ══════════════════════════════════════════════════════════════════════════════
import argparse
import os
import sys

import numpy as np

try:
    from . import neuro_io as nio
    from . import session_yaml as sy
except ImportError:
    import neuro_io as nio
    import session_yaml as sy

RADIUS, AMP_MIN, AMP_MAX, MIN_GAIN = 4, 0.3, 2.5, 0.05
TEMPLATE_SPIKES, TEMPLATE_MIN = 300, 30


def shifted(t, shift):
    """t (nSamples, nChannels) moved `shift` samples later: out[s] = t[s - shift], cubic
    (Catmull-Rom) between samples, zero where the source lies outside the window.  float32,
    as decollide_fit::shifted."""
    t = np.asarray(t, np.float32)
    ns = t.shape[0]
    fl = np.floor(shift)
    whole = int(fl)
    u = np.float32(shift - fl)
    pad = np.zeros((ns + 4, t.shape[1]), np.float32)          # pad[j + 2] = t[j], zero outside
    pad[2:ns + 2] = t
    i = np.arange(ns) - whole                                 # source position i - u
    ok = (i >= -1) & (i <= ns + 1)
    ic = np.clip(i, -1, ns + 1)
    out = np.zeros_like(t)
    if u == 0:
        j = np.clip(ic, -2, ns + 1) + 2
        v = pad[np.clip(j, 0, ns + 3)]
    else:
        f = np.float32(1.0) - u
        p0, p1, p2, p3 = pad[ic], pad[ic + 1], pad[ic + 2], pad[np.minimum(ic + 3, ns + 3)]
        h, c2, c5, c4, c3 = np.float32(0.5), np.float32(2), np.float32(5), np.float32(4), np.float32(3)
        v = p1 + h * f * (p2 - p0 + f * (c2 * p0 - c5 * p1 + c4 * p2 - p3 + f * (c3 * (p1 - p2) + p3 - p0)))
    out[ok] = v[ok]
    return out


def _joint(w, own, q, ww, oo, wo):
    """Least squares of w on (own, q); returns (residual energy, a_own, a_q)."""
    qq = float(np.dot(q.ravel().astype(np.float64), q.ravel().astype(np.float64)))
    wq = float(np.dot(w, q.ravel().astype(np.float64)))
    if not qq > 0.0:
        return np.inf, 0.0, 0.0
    if own is None or not oo > 0.0:
        a = wq / qq
        return ww - a * wq, 0.0, a
    oq = float(np.dot(own, q.ravel().astype(np.float64)))
    det = oo * qq - oq * oq
    if not det > 1e-9 * oo * qq:
        return np.inf, wo / oo, 0.0
    a_own = (wo * qq - wq * oq) / det
    a_q = (wq * oo - wo * oq) / det
    return ww - a_own * wo - a_q * wq, a_own, a_q


def fit_partner(w, own, partner, tau, radius=RADIUS, amp_min=AMP_MIN, amp_max=AMP_MAX, min_gain=MIN_GAIN):
    """Fit the template `partner` into the record `w` (both (nSamples, nChannels)) around the stored
    offset `tau`, jointly with `own` (the row's own unit's template, or None).  Returns a dict:
    accepted, tau (nearest sample), shift (samples), amp, amp_own, resid_before, resid_after."""
    wd = np.asarray(w, np.float32).ravel().astype(np.float64)
    od = None if own is None else np.asarray(own, np.float32).ravel().astype(np.float64)
    ww = float(wd @ wd)
    oo = 0.0 if od is None else float(od @ od)
    wo = 0.0 if od is None else float(wd @ od)
    before = ww - wo * wo / oo if (od is not None and oo > 0.0) else ww
    f = dict(accepted=False, tau=int(tau), shift=float(tau), amp=0.0, amp_own=0.0,
             resid_before=before, resid_after=0.0)
    radius = max(0, int(radius))
    resid = np.array([_joint(wd, od, shifted(partner, tau + k), ww, oo, wo)[0]
                      for k in range(-radius, radius + 1)])
    best = int(np.argmin(resid)) - radius                     # the earlier shift on a tie
    if not np.isfinite(resid[best + radius]):
        return f
    frac = 0.0
    if -radius < best < radius:
        a, b, c = resid[best + radius - 1], resid[best + radius], resid[best + radius + 1]
        den = a - 2.0 * b + c
        if np.isfinite(a) and np.isfinite(c) and den > 0.0:
            frac = float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
    shift = tau + best + frac
    r, a_own, a_q = _joint(wd, od, shifted(partner, shift), ww, oo, wo)
    if not r <= resid[best + radius]:
        shift = float(tau + best)
        r, a_own, a_q = _joint(wd, od, shifted(partner, shift), ww, oo, wo)
    f.update(shift=float(shift), tau=int(np.floor(shift + 0.5)) if shift >= 0 else -int(np.floor(-shift + 0.5)),
             amp=float(a_q), amp_own=float(a_own), resid_after=float(r))
    f["accepted"] = bool(np.isfinite(r) and amp_min <= a_q <= amp_max and before > 0.0
                         and (before - r) >= min_gain * before)
    return f


def home_clusters(cells, clu):
    """Per class (an .eap column): the cluster holding most of the rows in which that class is the
    only one, or -1.  Clusters <= 1 are nobody's home; the lower id wins a tie."""
    cells = np.asarray(cells); clu = np.asarray(clu)
    present = cells != nio.EAP_ABSENT
    single = present.sum(1) == 1
    home = np.full(cells.shape[1], -1, np.int64)
    for j in range(cells.shape[1]):
        c = clu[single & present[:, j] & (clu > 1)]
        if c.size:
            ids, cnt = np.unique(c, return_counts=True)
            home[j] = int(ids[np.argmax(cnt)])
    return home


def prefer_provenance(home, cells, clu, provenance, min_rows=TEMPLATE_MIN):
    """The home of a class is the cluster its .tcl entry says it was made from, when that cluster
    still holds at least min_rows rows naming only that class -- whatever cluster holds MOST of
    them (fiber-unmix's cluster of one-unit spikes holds every unit's: the largest holder is then a
    mixture).  provenance: per class, the .tcl's provenance cluster or -1.  Returns home, changed in
    place.  Mirrors decollide_fit::preferProvenance."""
    cells = np.asarray(cells); clu = np.asarray(clu)
    present = cells != nio.EAP_ABSENT
    single = present.sum(1) == 1
    for j in range(min(len(home), len(provenance))):
        p = int(provenance[j])
        if p > 1 and int((single & present[:, j] & (clu == p)).sum()) >= min_rows:
            home[j] = p
    return home


def split_row(row_cells, cluster, home):
    """The classes of one .eap row as (own, partners): own = (class, offset) of the class whose home
    cluster is the row's cluster, else of the class nearest offset 0 (the lower id on a tie);
    partners = the other (class, offset) pairs, by class.  own is None for a row without a class."""
    idx = np.flatnonzero(np.asarray(row_cells) != nio.EAP_ABSENT)
    if idx.size == 0:
        return None, []
    members = [(int(j), int(row_cells[j])) for j in idx]
    pick = None
    for k, (j, off) in enumerate(members):
        if cluster > 1 and home[j] == cluster and (pick is None or abs(off) < abs(members[pick][1])):
            pick = k
    if pick is None:
        pick = 0
        for k in range(1, len(members)):
            if abs(members[k][1]) < abs(members[pick][1]):
                pick = k
    return members[pick], [m for k, m in enumerate(members) if k != pick]


def correct_row(w, own, partners, templates, **fit_kw):
    """One collision record corrected: each partner with a template is fitted (in class order, each on
    what the ones before left) and, when accepted, subtracted.  templates: {class: (nSamples,
    nChannels)}.  Returns (record float32, fits) with one fit dict per partner (not accepted, at its
    stored offset, for a partner without a template).  Without the OWN unit's template nothing
    is subtracted: a partner fitted alone takes whatever of the row's own spike it can explain."""
    w = np.asarray(w, np.float32).copy()
    own_t = templates.get(own[0]) if own is not None else None
    if own_t is None:
        return w, [dict(accepted=False, tau=int(off), shift=float(off), amp=0.0, amp_own=0.0,
                        resid_before=0.0, resid_after=0.0) for _, off in partners]
    if own[1] != 0:
        own_t = shifted(own_t, own[1])
    fits = []
    for cls, off in partners:
        t = templates.get(cls)
        if t is None:
            fits.append(dict(accepted=False, tau=int(off), shift=float(off), amp=0.0, amp_own=0.0,
                             resid_before=0.0, resid_after=0.0))
            continue
        f = fit_partner(w, own_t, t, off, **fit_kw)
        if f["accepted"]:
            w -= np.float32(f["amp"]) * shifted(t, f["shift"])
        fits.append(f)
    return w, fits


def to_int16(w):
    w = np.nan_to_num(np.asarray(w, np.float64), nan=0.0, posinf=32767, neginf=-32768)
    r = np.where(w >= 0, np.floor(w + 0.5), -np.floor(-w + 0.5))     # half away from zero, as lround
    return np.clip(r, -32768, 32767).astype(np.int16)


def build(spk, cells, clu, *, tcl_entries=None, write_offsets=False, log=None, **fit_kw):
    """The corrected collision rows of one waveform file.

    spk: (N, nSamples, nChannels) int16 (a memmap is fine); cells: the .eap matrix (N, T) int8;
    clu: (N,) cluster ids.  A class with no row of its own falls back on the cluster its .tcl entry
    was made from (tcl_entries, as neuro_io.read_tcl gives them) while that cluster exists.
    Returns dict(rows, waves, stats, cells): the corrected rows (ascending), their int16 records,
    counts, and -- with write_offsets -- a copy of `cells` holding the fitted whole-sample offsets
    of the accepted partners (else the input)."""
    log = log or (lambda s: None)
    cells = np.asarray(cells); clu = np.asarray(clu, np.int64)
    present = cells != nio.EAP_ABSENT
    is_coll = present.sum(1) >= 2
    coll = np.flatnonzero(is_coll)
    home = home_clusters(cells, clu)
    if tcl_entries:
        have = set(np.unique(clu[clu > 1]).tolist())
        prov = np.full(home.size, -1, np.int64)
        for e in tcl_entries:
            j = int(e.get("col", -1))
            if 0 <= j < home.size and int(e.get("provenance_clu", -1)) > 1 and int(e["provenance_clu"]) in have:
                prov[j] = int(e["provenance_clu"])
        prefer_provenance(home, cells, clu, prov)       # the unit a class was made from, while it holds its rows
        home = np.where(home < 0, prov, home)           # ... and for a class with no row of its own
    clean_rows, cache = {}, {}

    def template(cls, row):
        if cls not in clean_rows:
            h = home[cls] if 0 <= cls < home.size else -1
            clean_rows[cls] = np.flatnonzero((clu == h) & ~is_coll) if h > 1 else np.zeros(0, np.int64)
        rows = clean_rows[cls]
        if rows.size < TEMPLATE_MIN:
            return None
        step = TEMPLATE_SPIKES // 2
        stretch = int(np.searchsorted(rows, row)) // step
        if (cls, stretch) not in cache:
            lo = max(0, stretch * step + step // 2 - TEMPLATE_SPIKES // 2)
            hi = min(rows.size, lo + TEMPLATE_SPIKES)
            lo = max(0, hi - TEMPLATE_SPIKES)
            cache[(cls, stretch)] = np.asarray(spk[rows[lo:hi]], np.float32).mean(0, dtype=np.float32)
        return cache[(cls, stretch)]

    out_rows, out_waves = [], []
    new_cells = cells.copy() if write_offsets else cells
    st = dict(collision_rows=int(coll.size), corrected=0, partners_fitted=0, partners_subtracted=0,
              partners_without_template=0, rows_without_own_template=0, offsets_differ=0)
    for i in coll:
        own, partners = split_row(cells[i], int(clu[i]), home)
        if not partners:
            continue
        templates = {}
        t = template(own[0], int(i))
        if t is None:                                   # nothing is subtracted without it (correct_row)
            st["rows_without_own_template"] += 1
            continue
        templates[own[0]] = t
        for cls, _ in partners:
            t = template(cls, int(i))
            if t is not None:
                templates[cls] = t
            else:
                st["partners_without_template"] += 1
        w, fits = correct_row(np.asarray(spk[i], np.float32), own, partners, templates, **fit_kw)
        done = 0
        for (cls, off), f in zip(partners, fits):
            st["partners_fitted"] += cls in templates
            if f["accepted"]:
                done += 1
                st["partners_subtracted"] += 1
                st["offsets_differ"] += f["tau"] != off
                if write_offsets:
                    new_cells[i, cls] = int(np.clip(f["tau"], -127, 127))
        if done:
            out_rows.append(int(i)); out_waves.append(to_int16(w))
            st["corrected"] += 1
    ns, nc = spk.shape[1], spk.shape[2]
    waves = np.stack(out_waves) if out_waves else np.zeros((0, ns, nc), np.int16)
    return dict(rows=np.asarray(out_rows, np.int64), waves=waves, stats=st, cells=new_cells)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Write the sparse .spkc of a stage: every collision row (two or more classes in its "
                    ".eap row) with its partners' fitted templates subtracted.  The .spk is not changed.")
    ap.add_argument("base"); ap.add_argument("elec", type=int)
    nio.add_clu_args(ap, method_default="stderiv", stage_default="fiber_session", stage_dest="clu_stage",
                     stage_alias=None)
    ap.add_argument("--variants", nargs="+", default=None,
                    help="waveform variants to write a .spkc for (default: standard and the clu method)")
    ap.add_argument("--write-offsets", action="store_true",
                    help="write the fitted whole-sample offsets of the subtracted partners back to the stage's "
                         ".eap (from the FIRST variant listed)")
    ap.add_argument("--radius", type=int, default=RADIUS, help="samples searched either side of the stored offset")
    ap.add_argument("--amp-min", type=float, default=AMP_MIN); ap.add_argument("--amp-max", type=float, default=AMP_MAX)
    ap.add_argument("--min-gain", type=float, default=MIN_GAIN,
                    help="least share of the residual energy a partner must remove to be subtracted")
    a = ap.parse_args(argv)
    log = lambda s: print(s, file=sys.stderr, flush=True)  # noqa: E731
    cfg = sy.resolve_session_params(a.base, a.elec, verbose=False)
    base = cfg["base"]; NS, NC = int(cfg["nsamp"]), int(cfg["nchan"])
    n = len(nio.read_res(base, a.elec))
    _, clu = nio.resolve_clu(a, base, a.elec, n_spikes=n, stage_dest="clu_stage")
    tag = a.clu_stage or ""
    e = nio.read_eap(base, a.elec, tag=tag)
    if not e["ok"] and tag:
        e = nio.read_eap(base, a.elec)                    # a stage without its own .eap reads the base one
    if not e["ok"]:
        raise SystemExit(f"fiber-spkc: no .eap for group {a.elec}" + (f" (stage '{tag}' or base)" if tag else ""))
    if e["nSpikes"] != n:
        raise SystemExit(f"fiber-spkc: the .eap holds {e['nSpikes']:,} rows, the .res {n:,}")
    tcl = nio.read_tcl(base, a.elec)
    variants = a.variants or list(dict.fromkeys(["standard", a.clu_method]))
    kw = dict(radius=a.radius, amp_min=a.amp_min, amp_max=a.amp_max, min_gain=a.min_gain)
    wrote = 0
    for k, v in enumerate(variants):
        r = nio.resolve(base, "spk", a.elec, v)
        if not (r.found and r.variant == v):
            log(f"[spkc] no .spk of variant {v}: skipped")
            continue
        spk = nio.open_spk_file(r.path, NS, NC)
        if spk.shape[0] != n:
            raise SystemExit(f"{os.path.basename(r.path)} holds {spk.shape[0]:,} spikes, the .res {n:,}")
        b = build(spk, e["cells"], clu, tcl_entries=tcl["entries"] if tcl["ok"] else None,
                  write_offsets=a.write_offsets and k == 0, **kw)
        path = nio.write_spkc(base, a.elec, b["rows"], b["waves"], n, variant=v, tag=tag)
        s = b["stats"]
        log(f"[spkc] {os.path.basename(path)}: {s['collision_rows']:,} rows with two or more classes, "
            f"{s['corrected']:,} corrected ({s['partners_subtracted']:,} of {s['partners_fitted']:,} partners "
            f"subtracted, {s['partners_without_template']:,} without a template; "
            f"{s['rows_without_own_template']:,} rows left as they are, their own unit having none), "
            f"{s['offsets_differ']:,} fitted offsets differ from the .eap's")
        if a.write_offsets and k == 0 and s["offsets_differ"]:
            p = nio.write_eap(base, a.elec, b["cells"], group=int(a.elec), tag=tag)
            log(f"[spkc] wrote the fitted offsets to {os.path.basename(p)}")
        wrote += 1
    if not wrote:
        raise SystemExit("fiber-spkc: no waveform file to correct")
    return 0


if __name__ == "__main__":
    sys.exit(main())
