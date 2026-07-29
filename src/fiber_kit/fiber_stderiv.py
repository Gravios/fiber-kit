"""fiber_stderiv -- the stderiv (SDIFF) spatial+temporal derivative transform and the sdiffPairs
grammar, kept as a SINGLE shared definition.

This is the fiber-kit mirror of neurosuite-3's C++ ground truth:
  - the transform: libneurosuite-core/src/neurosuite/core/stderiv_transform.hpp
    (== process_extractspikes_stderiv's fill_sdiff_buffer + computeSDiff)
  - the grammar:   libklustersshared/src/klustersshared/sdiff_pairs.h
i.e. the exact math that produced the .spk on disk.  fiber-realign (re-extraction) and any future
consumer import from HERE instead of carrying their own copy -- a second copy is precisely the
"shared policy re-expressed locally" drift that this codebase keeps getting bitten by.
"""
import numpy as np


def _round_half_away(x):
    """C round(): halfway cases go AWAY from zero.  np.round is half-to-EVEN, which
    disagrees on exactly-.5 values -- and order 5 produces them routinely (x minus
    the mean of an even-sized integer set)."""
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def parse_sdiff_pairs(spec):
    """Order 4 partner map: "a-b,c-d,..." -> partner[a] = b (group-local 0-based).

    Port of parseSdiffPairs in neurosuite-3 libklustersshared/sdiff_pairs.h.  Output
    channel a becomes x[a] - x[partner[a]].  The pattern must form a spanning tree
    with exactly one root (a position never used as a source), and the root must be
    the LAST position -- its output is redundant and is what SDIFF_PASS drops.
    Raises ValueError on anything malformed rather than guessing.
    """
    partner, maxpos = {}, 0
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" not in tok:
            raise ValueError(f"bad sdiffPairs token {tok!r} (want a-b)")
        lhs, rhs = tok.split("-", 1)
        try:
            a, b = int(lhs), int(rhs)
        except ValueError:
            raise ValueError(f"bad sdiffPairs token {tok!r} (want integers)")
        if a < 0 or b < 0 or a == b:
            raise ValueError(f"bad sdiffPairs token {tok!r}")
        if a in partner:
            raise ValueError(f"sdiffPairs channel {a} specified twice")
        partner[a] = b
        maxpos = max(maxpos, a + 1, b + 1)
    if not partner:
        raise ValueError("empty sdiffPairs")
    roots = [i for i in range(maxpos) if i not in partner]
    if len(roots) != 1:
        raise ValueError(f"sdiffPairs must have exactly one root, found {roots}")
    if roots[0] != maxpos - 1:
        raise ValueError(f"sdiffPairs root must be the last position "
                         f"({maxpos - 1}), found {roots[0]}")
    return [partner.get(i, i) for i in range(maxpos)], roots[0]


def parse_sdiff_sets(spec):
    """Order 5 reference sets: "a-b+c+d,..." -> sets[a] = [b,c,d] (group-local 0-based).

    Port of parseSdiffSets in neurosuite-3 libklustersshared/sdiff_pairs.h.  Output
    channel a becomes x[a] - mean(x[sets[a]]).  Every channel must carry a non-empty
    set and no channel may reference itself; there is no root requirement.
    """
    toks, maxpos = [], 0
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" not in tok:
            raise ValueError(f"bad sdiffSets token {tok!r} (want a-b[+c...])")
        lhs, rhs = tok.split("-", 1)
        try:
            a = int(lhs)
        except ValueError:
            raise ValueError(f"bad sdiffSets source in {tok!r}")
        if a < 0:
            raise ValueError(f"bad sdiffSets source in {tok!r}")
        members = []
        for m in rhs.split("+"):
            m = m.strip()
            try:
                b = int(m)
            except ValueError:
                raise ValueError(f"bad sdiffSets target in {tok!r}")
            if b < 0 or b == a:
                raise ValueError(f"bad sdiffSets target in {tok!r}")
            members.append(b)
            maxpos = max(maxpos, b + 1)
        maxpos = max(maxpos, a + 1)
        toks.append((a, members))
    if not toks:
        raise ValueError("empty sdiffSets")
    sets = [None] * maxpos
    for a, members in toks:
        if sets[a] is not None:
            raise ValueError(f"sdiffSets channel {a} specified twice")
        sets[a] = members
    missing = [i for i, v in enumerate(sets) if v is None]
    if missing:
        raise ValueError(f"sdiffSets channels {missing} have no reference set "
                         f"(all channels must be specified)")
    return sets


def sdiff_spec_uses_sets(spec):
    """True iff the pattern uses order-5 SET syntax (any '+'), else order-4."""
    return "+" in str(spec or "")


def apply_stderiv_transform(raw_ext, partner=None, sets=None):
    """ns3 stderiv transform, unified over the in-use orders (allpairs / custom / custom-CAR).

    Ported from stderiv_transform.hpp / process_extractspikes_stderiv (fill_sdiff_buffer +
    computeSDiff).  The spatial derivative (double) is selected by argument:

        order 4 (partner): sd[t,a] = x[t,a] - x[t, partner[a]]
        order 5 (sets):    sd[t,a] = x[t,a] - mean(x[t, sets[a]])
        order 3 (neither): sd[t,a] = nChanGrp * x[t,a] - Sum_j x[t,j]      (SDIFF_ALLPAIRS)

    Then, for EVERY order, round-half-away + clamp to int16 (the C++ holds sd in a short buffer --
    clampToInt16 in stderiv_transform.hpp / fill_sdiff_buffer Step 1), THEN the temporal
    first-difference:

        stderiv[t,a] = sd[t,a] - sd[t-1,a]      clamped to int16

    The intermediate clamp is applied UNIFORMLY, and it matters for allpairs too: sd = nChanGrp*x - Sum
    is an exact integer, so the ROUNDING is a no-op, but a large spike can push it past int16, and the
    C++ saturates BEFORE the temporal diff.  Orders 4/5 always relied on this; allpairs historically did
    not, diverging from the extractor on the ~0.07% of order-3 spikes whose spatial deriv overflowed.
    (Note orders 4/5 are plain differences -- NO nChanGrp scaling, which would inflate every amplitude
    by the channel count.)

    `raw_ext` is (N, nsamp+1, C): the window PLUS one preceding .fil sample, so the t=0 temporal diff
    uses the TRUE previous sample (the continuous g_prev_sdiff of the original extraction, matching the
    re-extractor's "need t-1").  Returns (N, nsamp, C) int16, aligned 1:1 with the standard window
    (raw_ext[:, 1:, :]).
    """
    r = np.asarray(raw_ext, np.float64)
    C = r.shape[2]
    if partner is not None:                                  # order 4
        if len(partner) != C:
            raise ValueError(f"sdiffPairs covers {len(partner)} channels but the group has {C}")
        sd = r - r[:, :, partner]
    elif sets is not None:                                   # order 5
        if len(sets) != C:
            raise ValueError(f"sdiffSets covers {len(sets)} channels but the group has {C}")
        ref = np.stack([r[:, :, s].mean(axis=2) for s in sets], axis=2)
        sd = r - ref
    else:                                                    # order 3 (allpairs)
        sd = C * r - r.sum(2, keepdims=True)
    sd = np.clip(_round_half_away(sd), -32768.0, 32767.0)
    st = sd[:, 1:, :] - sd[:, :-1, :]
    return np.clip(st, -32768.0, 32767.0).astype(np.int16)
