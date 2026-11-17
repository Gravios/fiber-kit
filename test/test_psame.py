#!/usr/bin/env python3
"""fiber-psame-merge: history-conditioned cell models and the P(same) merge.

  1. ewma_state equals fiber_adapt.ewma (the reference recurrence) on a random train; the default
     10-tau horizon is within 1e-3 of it.
  2. A bursting cell (later burst spikes smaller AND differently shaped, the amplitude set by the cell's
     own recent firing; noise 15 against a 100 peak, SNR ~7 as on g6) cut in time into two clusters:
     one history-conditioned model explains the union (dbic > 0) and psame_merge joins them.
  3. A second cell with A's AVERAGE waveform but tonic firing (no bursts, no history-driven amplitude)
     is NOT joined to either half (dbic < 0) -- the negative control: roll-cosine alone ranks it as
     close as A's true other half, so the cell model is what separates them.
  Not covered, because it fails: a cell cut by a hard AMPLITUDE threshold.  Each half is then narrower
  than the cell along the noisy amplitude axis (selection), the union always fits worse, and dbic < 0
  (synthetic: -0.53 for the true pair, -0.06 for a tonic look-alike).  See the module docstring.
  4. A distinct cell stays alone; the input labels are not modified in place.
  5. Chunk modes: --per-chunk never merges clusters of different chunks (and still merges within one);
     --chunk-aware joins A's time halves across ADJACENT chunks (scored on those chunks), keeps the
     look-alike B apart, and with max-gap 0 refuses halves that share no chunk.
  6. The scale-residual veto: scale_residual is zero for a scaled copy and for a scaled, SHIFTED copy,
     and large for another cell at matched amplitude; the veto leaves a correct merge alone, refuses a
     different-shaped cluster that an all-accepting score would swallow, never fires below scale_min_n,
     and reports the residual on every merge.  Its documented blind spot is also asserted: cell B here
     is a different cell whose template really is a scaled copy of A's, so no shape test can refuse it.
Exits nonzero on failure.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from fiber_kit import fiber_psame as ps, fiber_adapt as fa  # noqa: E402

fails = 0


def check(name, ok, detail=""):
    global fails
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail and not ok else ""))
    fails += 0 if ok else 1


rng = np.random.default_rng(3)
t = np.sort(rng.uniform(0, 50, 400))
check("ewma_state = fiber_adapt.ewma (full horizon)", np.allclose(ps.ewma_state(t, 0.3, horizon=60.0), fa.ewma(t, 0.3), atol=1e-9))
check("ewma_state, default 10-tau horizon, within 1e-3", np.allclose(ps.ewma_state(t, 0.3), fa.ewma(t, 0.3), atol=1e-3))

NS, NC = 32, 8
tt = np.arange(NS)


def template(c0, w):
    T = np.zeros((NS, NC))
    for c in range(NC):
        g = np.exp(-0.5 * ((c - c0) / 1.5) ** 2)
        T[:, c] = -g * np.exp(-0.5 * ((tt - 12) / w) ** 2) + 0.4 * g * np.exp(-0.5 * ((tt - 17) / (2 * w)) ** 2)
    return T * 100


TA = template(3.0, 1.2)
SA = template(3.6, 2.0) - template(3.0, 1.2)        # shape change per unit amplitude loss
TC = template(6.5, 1.0)
spk, times, lab = [], [], []
# cell A: bursts of 4 at 5 ms; the amplitude follows the cell's own recent firing (an EWMA adaptation
# state, tau 30 ms) and the shape follows S as the amplitude drops
tA = np.sort(np.concatenate([t0 + 0.005 * np.arange(4) for t0 in rng.uniform(0, 300, 300)]))
aA = np.clip(1.0 - 0.25 * fa.ewma(tA, 0.03), 0.3, 1.0)
for t_, a in zip(tA, aA):
    spk.append(a * TA + (1 - a) * SA + rng.normal(0, 15, (NS, NC))); times.append(t_); lab.append(0)
# cell B: A's AVERAGE waveform, but tonic (Poisson): no burst structure, no history-driven amplitude
TB = aA.mean() * TA + (1 - aA.mean()) * SA
for t0 in np.sort(rng.uniform(0, 300, 900)):
    spk.append(TB * rng.uniform(0.9, 1.1) + rng.normal(0, 15, (NS, NC))); times.append(t0); lab.append(1)
# cell C: elsewhere
for t0 in np.sort(rng.uniform(0, 300, 600)):
    spk.append(TC + rng.normal(0, 15, (NS, NC))); times.append(t0); lab.append(2)
o = np.argsort(times)
spk = np.asarray(spk, np.float32)[o]; times = np.asarray(times)[o]; lab = np.asarray(lab)[o]
amp = np.array([float((w * TA).sum() / (TA * TA).sum()) for w in spk])
clu = np.where(lab == 0, np.where(times < 150, 2, 3), np.where(lab == 1, 4, 5))   # A cut in time (drift-like)
check("setup: A's two fragments both populated", (clu == 2).sum() > 100 and (clu == 3).sum() > 100,
      f"{(clu == 2).sum()} / {(clu == 3).sum()}")
mu, V = ps.pca_basis(spk, n=12, nfit=len(spk))
M = {u: ps.CellModel(ps.project(spk, np.flatnonzero(clu == u), mu, V), times[clu == u]) for u in (2, 3, 4, 5)}
d_true = ps.dbic(M[2], M[3]); d_b3 = ps.dbic(M[4], M[3]); d_b2 = ps.dbic(M[2], M[4])
rc_true = ps.rollcos(spk[clu == 2].mean(0), spk[clu == 3].mean(0))
rc_b = ps.rollcos(spk[clu == 4].mean(0), spk[clu == 3].mean(0))
check("history model: A's fragments are one cell (dbic > 0)", d_true > 0, f"{d_true:.3f}")
check("negative control: B is not one of A's halves (dbic < 0)", d_b3 < 0 and d_b2 < 0, f"{d_b3:.3f}, {d_b2:.3f}")
check("roll-cosine alone cannot tell them apart (B as close to A's second half)", rc_b >= rc_true - 0.02,
      f"rc B-A2 {rc_b:.3f} vs A1-A2 {rc_true:.3f}")
check("A's halves score above B (dbic)", d_true > max(d_b3, d_b2) + 0.5, f"{d_true:.3f} vs {d_b3:.3f}, {d_b2:.3f}")
clu_in = clu.copy()
new, merges = ps.psame_merge(spk, times, clu, theta=0.5, cannot_link=0.3, knn=3, cos_floor=0.5, min_n=50,
                             w_rc=0.0, w_dbic=4.0, bias=0.0, log=lambda *_: None)
check("A's fragments merged", len(np.unique(new[lab == 0])) == 1, str(merges))
check("B stays apart from A", not set(np.unique(new[lab == 1])) & set(np.unique(new[lab == 0])))
check("C stays alone", len(np.unique(new[lab == 2])) == 1 and not set(np.unique(new[lab == 2])) & set(np.unique(new[lab != 2])))
check("input labels untouched", (clu == clu_in).all())

# pair alignment.  A's second fragment realigned to its own median at a 2-sample offset (what an in-place
# per-cluster realign leaves between two fragments of one unit): the union model sees a shape difference
# and dbic collapses; with align_pairs the windows are rolled back by the roll-cos shift and dbic stands.
spk_off = spk.copy(); spk_off[clu == 3] = np.roll(spk_off[clu == 3], 2, axis=1)
M3o = ps.CellModel(ps.project(spk_off, np.flatnonzero(clu == 3), mu, V), times[clu == 3])
d_off = ps.dbic(M[2], M3o)
rc_o, s_o = ps.rollcos_shift(spk_off[clu == 2].mean(0), spk_off[clu == 3].mean(0))
M3a = ps.CellModel(ps.project(spk_off, np.flatnonzero(clu == 3), mu, V, roll=s_o), times[clu == 3])
d_al = ps.dbic(M[2], M3a)
check("an offset between A's fragments collapses dbic on raw windows", d_off < d_true - 0.5, f"{d_off:.3f} vs {d_true:.3f}")
check("roll-cos finds the offset", s_o == -2 and rc_o >= rc_true - 0.02, f"shift {s_o}, rc {rc_o:.3f}")
check("dbic on pair-aligned windows recovers", d_al > d_true - 0.5, f"{d_al:.3f} vs {d_true:.3f}")
# inside psame_merge the PCA basis is fit on the data as given (the shifted fragment included), which costs
# the aligned pair a few tenths of dbic against the unshifted run (0.31 -> -0.19 here; raw windows -3.0), so
# the merge is tested at theta 0.3 (p = 0.32 for the aligned pair, ~0 for the raw one)
new_raw, m_raw = ps.psame_merge(spk_off, times, clu, theta=0.3, cannot_link=0.0, knn=3, cos_floor=0.5, min_n=50,
                                w_rc=0.0, w_dbic=4.0, bias=0.0, log=lambda *_: None, align_pairs=False)
new_al, m_al = ps.psame_merge(spk_off, times, clu, theta=0.3, cannot_link=0.0, knn=3, cos_floor=0.5, min_n=50,
                              w_rc=0.0, w_dbic=4.0, bias=0.0, log=lambda *_: None, align_pairs=True)
check("align_pairs 0: the offset fragments are not merged", len(np.unique(new_raw[lab == 0])) == 2, str(m_raw))
check("align_pairs 1: the offset fragments merge, B and C stay apart", len(np.unique(new_al[lab == 0])) == 1
      and not set(np.unique(new_al[lab == 1])) & set(np.unique(new_al[lab == 0]))
      and len(np.unique(new_al[lab == 2])) == 1, str(m_al))

# chunk modes.  A's two fragments are cut at 150 s (A1 earlier, A2 later); the session has four 75-s chunks.
kw = dict(theta=0.5, cannot_link=0.3, knn=3, cos_floor=0.5, min_n=50, w_rc=0.0, w_dbic=4.0, bias=0.0, log=lambda *_: None)
grp = {2: 0, 3: 1, 4: 0, 5: 0}                                    # per-chunk: A1 and A2 in different chunks
new_pc, m_pc = ps.psame_merge(spk, times, clu, group_of=grp, **kw)
check("per-chunk: fragments of different chunks are never merged", len(np.unique(new_pc[lab == 0])) == 2, str(m_pc))
grp2 = {2: 0, 3: 0, 4: 0, 5: 0}
new_pc2, _ = ps.psame_merge(spk, times, clu, group_of=grp2, **kw)
check("per-chunk: fragments of one chunk still merge", len(np.unique(new_pc2[lab == 0])) == 1)
chunk = np.floor(times / 75.0).astype(np.int64)
new_ca, m_ca = ps.psame_merge(spk, times, clu, chunk_of=chunk, min_chunk_n=30, max_gap=1, **kw)
check("chunk-aware: A's halves (adjacent chunks 1|2) merge", len(np.unique(new_ca[lab == 0])) == 1, str(m_ca))
check("chunk-aware: the merge is scored across the adjacent chunks", any(m.get("how") == "adjacent" for m in m_ca), str(m_ca))
check("chunk-aware: B stays apart from A", not set(np.unique(new_ca[lab == 1])) & set(np.unique(new_ca[lab == 0])))
new_g0, _ = ps.psame_merge(spk, times, clu, chunk_of=chunk, min_chunk_n=30, max_gap=0, **kw)
check("chunk-aware, max-gap 0: halves with no shared chunk are not merged", len(np.unique(new_g0[lab == 0])) == 2)
# ── the scale-residual veto (0.67.24) ──────────────────────────────────────
# Distance attenuates a waveform without changing its shape, so a cell's energy bands differ by a
# scalar while two cells at matched amplitude do not.  Measured AUC 0.943 on g6 0316's own pre-psame
# clusters at purity >= 0.85 / n >= 100 (analysis/scale-residual-veto-g6-0316/).
r_scaled = ps.scale_residual(TA, 0.4 * TA)[0]
r_rolled = ps.scale_residual(TA, 0.4 * np.roll(TA, 2, axis=0))[0]
r_other = ps.scale_residual(TA, TC * (abs(TA).max() / abs(TC).max()))[0]
check("scale_residual: a scaled copy leaves no residual", r_scaled < 1e-9, f"{r_scaled:.2e}")
check("scale_residual: roll-invariant (a scaled, shifted copy too)", r_rolled < 1e-9, f"{r_rolled:.2e}")
check("scale_residual: a different cell at matched amplitude does not scale onto it",
      r_other > 0.5, f"{r_other:.3f}")

vkw = dict(theta=0.5, cannot_link=0.3, knn=3, cos_floor=0.5, min_n=50, w_rc=0.0, w_dbic=4.0,
           bias=0.0, log=lambda *_: None)
new_v, m_v = ps.psame_merge(spk, times, clu, scale_veto=True, scale_thr=0.20, scale_min_n=100, **vkw)
check("veto does not block a correct merge: A's fragments still join",
      len(np.unique(new_v[lab == 0])) == 1, str(m_v))
check("the pair's scale residual is reported on every merge",
      all("sresid" in m for m in m_v) and all(m["sresid"] == m["sresid"] for m in m_v), str(m_v))

# force the score to accept everything, so only the veto can refuse: p = sigmoid(bias) ~ 1 for all pairs
akw = dict(theta=0.5, cannot_link=0.3, knn=3, cos_floor=0.0, min_n=50, w_rc=0.0, w_dbic=0.0,
           bias=10.0, log=lambda *_: None)
new_all, _m_all_ref = ps.psame_merge(spk, times, clu, **akw)
new_veto, m_veto = ps.psame_merge(spk, times, clu, scale_veto=True, scale_thr=0.20, scale_min_n=100, **akw)
check("without the veto an all-accepting score swallows C into A",
      set(np.unique(new_all[lab == 2])) & set(np.unique(new_all[lab == 0])) != set(),
      str(sorted(np.unique(new_all))))
check("the veto refuses it: C's template is not a scaled copy of A's",
      not set(np.unique(new_veto[lab == 2])) & set(np.unique(new_veto[lab == 0])),
      str(sorted(np.unique(new_veto))))
new_guard, _ = ps.psame_merge(spk, times, clu, scale_veto=True, scale_thr=0.20,
                              scale_min_n=10 ** 6, **akw)
# ── the residual as a SCORE TERM over the pair's own noise floor (0.67.28) ─
# A fixed threshold cannot work at low spike counts: the residual a SINGLE cell scores against
# itself rises as 1/m, so below ~100 spikes the floor approaches scale_thr and the veto had to be
# switched off (scale_min_n).  split_half_residual measures that floor per cluster.
W5 = spk[clu == 5]
f_full, f_small = ps.split_half_residual(W5), ps.split_half_residual(W5[:60])
check("split_half_residual: the noise floor falls as spikes are added",
      f_small > 4 * f_full, f"{f_small:.4f} at 60 vs {f_full:.4f} at {len(W5)}")
check("split_half_residual: NaN when a half would be too thin",
      ps.split_half_residual(W5[:10]) != ps.split_half_residual(W5[:10]))
# the reference predicts what a SAME-cell pair scores: two disjoint subsets of cluster 5
for m_lo, m_hi in ((100, 400), (60, 400)):
    _lo, _hi = W5[:m_lo].mean(0), W5[m_lo:m_lo + m_hi].mean(0)
    _sr, _lag, _sc = ps.scale_residual(_hi, _lo)
    _ref = ps.residual_reference(ps.split_half_residual(W5[:m_lo]), m_lo, m_hi, _sc)
    check(f"residual_reference predicts a same-cell cross residual (m_lo {m_lo})",
          0.7 < _sr / _ref < 1.6, f"cross {_sr:.4f} vs reference {_ref:.4f} ({_sr / _ref:.2f}x)")
check("residual_reference: NaN floor in, NaN out",
      ps.residual_reference(float("nan"), 100, 400, 0.5) != ps.residual_reference(float("nan"), 100, 400, 0.5))
# the excess separates same-cell from different-cell BELOW scale_min_n, where the veto is off.
# Measured across spike counts in analysis/sresid-reference/: 74x at 50 spikes, 13x at 20.
_hiW = spk[clu == 4]
for _m in (150, 60):
    _same = ps.scale_residual(_hiW.mean(0), W5[:_m].mean(0))
    _exc_d = _same[0] - ps.residual_reference(ps.split_half_residual(W5[:_m]), _m, len(_hiW), _same[2])
    _lo2 = W5[:_m]
    _s2 = ps.scale_residual(W5[_m:].mean(0), _lo2.mean(0))
    _exc_s = _s2[0] - ps.residual_reference(ps.split_half_residual(_lo2), _m, len(W5) - _m, _s2[2])
    check(f"excess separates two cells from one at m={_m} (below scale_min_n 100)",
          _exc_d > 10 * max(_exc_s, 1e-6), f"different {_exc_d:+.4f} vs same {_exc_s:+.4f}")

new_w0, m_w0 = ps.psame_merge(spk, times, clu, w_sresid=0.0, **akw)
check("w_sresid 0 changes nothing: the same merges as the untouched score",
      [(m["into"], m["merged"]) for m in m_w0] == [(m["into"], m["merged"]) for m in _m_all_ref],
      f"{[(m['into'], m['merged']) for m in m_w0]}")
new_w, m_w = ps.psame_merge(spk, times, clu, w_sresid=30.0, **akw)
check("the score term refuses C into A without a veto",
      not set(np.unique(new_w[lab == 2])) & set(np.unique(new_w[lab == 0])),
      str(sorted(np.unique(new_w))))
check("the score term does not block A's own fragments", len(np.unique(new_w[lab == 0])) == 1, str(m_w))
check("every merge records its floor and excess",
      all(m.get("sresid_ref") == m.get("sresid_ref") and "sresid_exc" in m for m in m_w), str(m_w))
_af = [m for m in m_w if {m["into"], m["merged"]} == {2, 3}]
check("A's own fragments pay no penalty: their residual sits BELOW their floor",
      len(_af) == 1 and _af[0]["sresid_exc"] < 0 and _af[0]["sresid_pen"] == 0.0,
      str([(m["into"], m["merged"], round(m["sresid_exc"], 5), round(m["sresid_pen"], 5)) for m in m_w]))
check("documented limitation carries over: B is built from A's OWN mean shape, so it is a scaled "
      "copy and the term cannot refuse it either (only dbic can)",
      any({m["into"], m["merged"]} == {2, 4} for m in m_w),
      str([(m["into"], m["merged"]) for m in m_w]))
# sresid_ref fixed reproduces the constant threshold as the reference
new_f, m_f = ps.psame_merge(spk, times, clu, w_sresid=30.0, sresid_ref="fixed", scale_thr=0.20, **akw)
check("sresid_ref fixed uses scale_thr as the floor",
      all(abs(m["sresid_ref"] - 0.20) < 1e-9 for m in m_f), str([m["sresid_ref"] for m in m_f]))
# the no-power refusal: a floor above sresid_ref_max means the shape test cannot tell them apart
new_np, m_np = ps.psame_merge(spk, times, clu, sresid_ref_max=1e-9, **akw)
check("sresid_ref_max refuses every merge whose floor exceeds it", len(m_np) == 0, str(m_np))
check("...and it does not reroute: no cluster merges into another instead",
      len(np.unique(new_np)) == len(np.unique(clu)), str(sorted(np.unique(new_np))))
new_np2, m_np2 = ps.psame_merge(spk, times, clu, sresid_ref_max=10.0, **akw)
check("sresid_ref_max above every floor refuses nothing", len(m_np2) == len(_m_all_ref), str(m_np2))

# the merge table carries the residual terms: that is how a session's SRESID_REF_MAX gets chosen
_rows = ps.merge_table_rows(m_w + [dict(into=9, merged=8, p=0.9, rc=0.8, dbic=0.1, geo_z=0.0)])
check("merge table header names every term", _rows[0].split("\t") == list(ps.MERGE_TABLE_COLUMNS),
      _rows[0])
check("merge table keeps every row rectangular",
      all(len(r.split("\t")) == len(ps.MERGE_TABLE_COLUMNS) for r in _rows),
      str([len(r.split("\t")) for r in _rows]))
check("merge table writes a term the run did not compute as an EMPTY cell, not 'nan'",
      _rows[-1].split("\t")[-4:] == ["", "", "", ""] and "nan" not in _rows[-1], _rows[-1])
check("merge table reports the floor of every accepted merge",
      all(r.split("\t")[8] for r in _rows[1:-1]), str(_rows[1:-1]))

check("scale_min_n guards it: below the count the veto never fires",
      set(np.unique(new_guard[lab == 2])) & set(np.unique(new_guard[lab == 0])) != set(),
      str(sorted(np.unique(new_guard))))
# B is A's AVERAGE waveform: a genuinely different cell that IS a scaled copy, so the veto cannot see
# it -- the documented limitation (the 10 % of over-merges it misses are exactly these).
rb = ps.scale_residual(spk[clu == 2].mean(0), spk[clu == 4].mean(0))[0]
check("documented limitation: a look-alike cell that really is a scaled copy slips the veto",
      rb < 0.20, f"{rb:.3f}")

print("OK (0 failures)" if not fails else f"FAILED ({fails})")
sys.exit(1 if fails else 0)
