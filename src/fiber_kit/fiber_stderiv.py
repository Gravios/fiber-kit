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
    root = roots[0]
    if root != maxpos - 1:
        raise ValueError(f"sdiffPairs root must be the last position "
                         f"({maxpos - 1}), found {root}")
    # A tree, not just one root: every partner chain must reach the root ("0-1,1-0,2-3" has one
    # root, 3, but 0 and 1 form a cycle).  As parseSdiffPairs refuses it.
    for i in range(maxpos):
        cur, steps = i, 0
        while cur != root and steps <= maxpos:
            cur, steps = partner[cur], steps + 1
        if cur != root:
            raise ValueError(f"sdiffPairs channel {i} is on a cycle that never reaches the root")
    # The root's own output is x[root] - x[pred], pred its first predecessor in position order --
    # exactly what parseSdiffPairs wires and process_extractspikes_stderiv writes.  (It used to be
    # x[root] - x[root] = 0 here, so a re-extracted .spk differed from the extractor's on that
    # channel.)
    preds = [j for j in range(maxpos) if partner.get(j) == root]
    if not preds:
        raise ValueError("sdiffPairs root is disconnected")
    out = [partner.get(i, i) for i in range(maxpos)]
    out[root] = preds[0]
    return out, root


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
    """True iff the pattern uses order-5 SET syntax (any '+'), else order-4.  False for the
    expression form (order 6), whose '+' is arithmetic: see sdiff_spec_order."""
    if sdiff_spec_is_expression(spec):
        return False
    return "+" in str(spec or "")


# ── order 6: expressions ──────────────────────────────────────────────────────────
#
# Port of neurosuite-3 libneurosuite-core/src/neurosuite/core/channel_expr.hpp: one arithmetic
# expression per output channel, in channel order, over the group's channels c0, c1, ...:
#     "c0 - (c1 + c2)/2, c1 - c0, c2 - 0.5*c3, (c3 - c2)^2"
# + - * / ^ and parentheses with the usual precedence (^ right-associative and binding tighter
# than a unary minus); numbers are constants; any arithmetic.  A non-finite result is 0 and a
# result beyond +-1e9 is held there.  Both implementations run the same vectors
# (test/sdiff_expr_vectors.tsv, an identical copy of neurosuite-3's).

EXPR_LIMIT = 1e9
_EXPR_MAX_DEPTH = 64     # evaluation stack (as ChannelExpr::kMaxDepth)
_EXPR_MAX_NEST = 64      # nesting of ( ) and signs (as ChannelExpr::kMaxNest)
_BLANKS = " \t\n\r\f\v"


def _is_digit(c):
    return "0" <= c <= "9"           # ASCII only, as the C++ grammar


def _shown(c):
    return f"'{c}'" if " " <= c <= "~" else "a non-ASCII or control character"


def sdiff_spec_is_expression(spec):
    """True when a pattern is written as expressions (it names a channel cN)."""
    s = str(spec or "")
    return "c" in s or "C" in s


def sdiff_spec_order(spec):
    """The spatial order a pattern selects: 6 for expressions, 5 for reference sets, else 4."""
    if sdiff_spec_is_expression(spec):
        return 6
    return 5 if sdiff_spec_uses_sets(spec) else 4


class ChannelExpr:
    """One output channel's expression, compiled to postfix ("num", v) / ("chan", i) / op."""

    def __init__(self, text):
        self.text = text.strip(_BLANKS)
        self.code = []
        self.max_chan = -1
        self._depth = 0
        self._sp = 0
        self._p = 0
        self._nest = 0
        if not self.text:
            raise ValueError("an empty expression")
        self._parse()

    # parsing -------------------------------------------------------------------------------
    def _err(self, msg, col=True):
        where = f" (column {self._p + 1})" if col else ""
        shown = self.text if len(self.text) <= 60 else self.text[:57] + "..."
        return ValueError(f"'{shown}' {msg}{where}")

    def _skip(self):
        while self._p < len(self.text) and self.text[self._p] in _BLANKS:
            self._p += 1

    def _peek(self, c):
        self._skip()
        return self._p < len(self.text) and self.text[self._p] == c

    def _emit(self, op, arg=None):
        self.code.append((op, arg))
        if op in ("num", "chan"):
            self._sp += 1
            self._depth = max(self._depth, self._sp)
        elif op != "neg":
            self._sp -= 1

    def _parse(self):
        self._expr()
        self._skip()
        if self._p != len(self.text):
            c = self.text[self._p]
            raise self._err("has a ')' without its '('" if c == ")"
                            else f"has {_shown(c)} where an operator is expected")
        if self.max_chan < 0:
            raise self._err("uses no channel (channels are written c0, c1, ...)", col=False)
        if self._depth > _EXPR_MAX_DEPTH:
            raise self._err("is nested too deeply", col=False)

    def _expr(self):
        self._term()
        while self._peek("+") or self._peek("-"):
            op = self.text[self._p]
            self._p += 1
            self._term()
            self._emit("add" if op == "+" else "sub")

    def _term(self):
        self._unary()
        while self._peek("*") or self._peek("/"):
            op = self.text[self._p]
            self._p += 1
            self._unary()
            self._emit("mul" if op == "*" else "div")

    def _unary(self):
        # Every recursion of the descent passes through here, so this bounds it (no
        # RecursionError on deep input) -- the same limit as the C++.
        self._nest += 1
        try:
            if self._nest > _EXPR_MAX_NEST:
                raise self._err("is nested too deeply", col=False)
            self._unary_body()
        finally:
            self._nest -= 1

    def _unary_body(self):
        if self._peek("-"):
            self._p += 1
            self._unary()
            self._emit("neg")
            return
        if self._peek("+"):
            self._p += 1
            self._unary()
            return
        self._power()

    def _power(self):
        self._primary()
        if self._peek("^"):
            self._p += 1
            self._unary()                          # right-associative: 2^3^2 = 2^(3^2)
            self._emit("pow")

    def _primary(self):
        self._skip()
        t = self.text
        if self._p >= len(t):
            raise self._err("ends where a channel, number or '(' is expected")
        c = t[self._p]
        if c == "(":
            self._p += 1
            self._expr()
            if not self._peek(")"):
                raise self._err("is missing a ')'")
            self._p += 1
            return
        if c in "cC":
            self._p += 1
            start = self._p
            ch = 0
            while self._p < len(t) and _is_digit(t[self._p]):
                ch = ch * 10 + (ord(t[self._p]) - 48)
                if ch > 1000000:
                    raise self._err("has a channel number that is too large")
                self._p += 1
            if self._p == start:
                self._p = start - 1
                raise self._err("has a 'c' without a channel number")
            self.max_chan = max(self.max_chan, ch)
            self._emit("chan", ch)
            return
        if _is_digit(c) or c == ".":
            start = self._p
            while self._p < len(t) and _is_digit(t[self._p]):
                self._p += 1
            if self._p < len(t) and t[self._p] == ".":
                self._p += 1
                while self._p < len(t) and _is_digit(t[self._p]):
                    self._p += 1
            if self._p - start == 1 and c == ".":
                self._p = start
                raise self._err("has a '.' that is not a number")
            if self._p < len(t) and t[self._p] in "eE":
                q = self._p + 1
                if q < len(t) and t[q] in "+-":
                    q += 1
                if q < len(t) and _is_digit(t[q]):
                    self._p = q
                    while self._p < len(t) and _is_digit(t[self._p]):
                        self._p += 1
            self._emit("num", float(t[start:self._p]))
            self._skip()
            if self._p < len(t) and t[self._p] in "cC(":
                raise self._err("needs a '*' between a number and what follows it")
            return
        raise self._err(f"has {_shown(c)} where a channel, number or '(' is expected")

    # evaluation ----------------------------------------------------------------------------
    def eval(self, chan):
        """Value for channel values chan(i) (scalars or same-shape numpy arrays), float64."""
        st = []
        with np.errstate(all="ignore"):
            for op, arg in self.code:
                if op == "num":
                    st.append(np.float64(arg))
                elif op == "chan":
                    st.append(np.asarray(chan(arg), np.float64))
                elif op == "neg":
                    st[-1] = -st[-1]
                else:
                    b = st.pop()
                    a = st.pop()
                    st.append(a + b if op == "add" else a - b if op == "sub" else
                              a * b if op == "mul" else a / b if op == "div" else np.power(a, b))
            v = np.asarray(st[0], np.float64) if st else np.float64(0.0)
            v = np.where(np.isfinite(v), v, 0.0)
        return np.clip(v, -EXPR_LIMIT, EXPR_LIMIT)


def parse_sdiff_exprs(spec):
    """Compile an expression pattern: one ChannelExpr per output channel (comma-separated).
    ValueError naming the element and the problem when it does not follow the grammar.  A group
    written in an older form (no channel names) is converted to the expressions it computes."""
    s = str(spec or "")
    if not sdiff_spec_is_expression(s):
        s = sdiff_pattern_as_exprs(s)
    out = []
    for k, el in enumerate(s.split(",")):
        try:
            out.append(ChannelExpr(el))
        except ValueError as e:
            raise ValueError(f"sdiffPairs expression {k}: {e}") from None
    return out


def sdiff_pattern_as_exprs(spec):
    """An order-4 or order-5 pattern written as the expressions it computes (bit-identical:
    0-1 -> c0-c1, 0-1+2 -> c0-(c1+c2)/2).  Mirrors sdiffPairsAsExpr / sdiffSetsAsExpr."""
    if sdiff_spec_uses_sets(spec):
        sets = parse_sdiff_sets(spec)
        return ",".join(f"c{k}-(" + "+".join(f"c{m}" for m in st) + f")/{len(st)}"
                        for k, st in enumerate(sets))
    partner, _ = parse_sdiff_pairs(spec)
    return ",".join(f"c{k}-c{p}" for k, p in enumerate(partner))


def check_exprs_fit(exprs, n_chan):
    """ValueError unless the expressions fit a group of n_chan channels (one per channel, no
    channel beyond the group's)."""
    if len(exprs) != n_chan:
        raise ValueError(f"{len(exprs)} expressions for {n_chan} channels "
                         f"(one per channel, in channel order)")
    for k, e in enumerate(exprs):
        if e.max_chan >= n_chan:
            raise ValueError(f"expression {k} ('{e.text}') uses c{e.max_chan} but the group has "
                             f"c0..c{n_chan - 1}")


def plain_order_spec(order, n_chan):
    """(partner, sets) reproducing neurosuite-3's plain spatial-derivative orders (computeSDiff in
    process_extractspikes_stderiv) through apply_stderiv_transform's partner / sets forms:
      1 SDIFF_FIRST      x[i] - x[i+1]; the last channel x[n-1] - x[n-2]       -> partner map
      2 SDIFF_LAPLACIAN  x[i] - (x[i-1] + x[i+1]) / 2; the ends minus their one neighbour -> sets
      3 SDIFF_ALLPAIRS   n x[i] - Sum_j x[j]                                      -> (None, None)
    Group-local 0-based channel positions.  ValueError for any other order."""
    n = int(n_chan)
    if order == 3:
        return None, None
    if n < 2:
        raise ValueError("a spatial derivative needs at least two channels")
    if order == 1:
        return [i + 1 for i in range(n - 1)] + [n - 2], None
    if order == 2:
        return None, [[1]] + [[i - 1, i + 1] for i in range(1, n - 1)] + [[n - 2]]
    raise ValueError(f"plain spatial-derivative order {order} does not exist (1-3)")


def apply_stderiv_transform(raw_ext, partner=None, sets=None, exprs=None):
    """ns3 stderiv transform, unified over the in-use orders (allpairs / custom / custom-CAR).

    Ported from stderiv_transform.hpp / process_extractspikes_stderiv (fill_sdiff_buffer +
    computeSDiff).  The spatial derivative (double) is selected by argument:

        order 4 (partner): sd[t,a] = x[t,a] - x[t, partner[a]]
        order 5 (sets):    sd[t,a] = x[t,a] - mean(x[t, sets[a]])
        order 6 (exprs):   sd[t,a] = exprs[a] over x[t, 0..C)  (parse_sdiff_exprs)
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
    if exprs is not None:                                    # order 6
        check_exprs_fit(exprs, C)
        sd = np.stack([e.eval(lambda i: r[:, :, i]) * np.ones(r.shape[:2]) for e in exprs], axis=2)
    elif partner is not None:                                # order 4
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
