#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════════════════════
#  morpho_geom.py — morphology I/O and compartmentalization.
#
#  Reads a reconstruction (SWC, or a NEURON "geometry" .hoc built from
#  create/connect/pt3dadd) and turns it into the flat compartment arrays the
#  cable solver and the line-source extracellular model both need: endpoints,
#  length, diameter, membrane area, and a PARENT INDEX with parent[i] < i.
#
#  Why a parent index and not a section graph: every downstream step is either
#  a tree sweep (Hines elimination) or a per-compartment reduction (line-source
#  sum, laminar profile).  Both want contiguous arrays and a topological order;
#  keeping the section structure around would mean re-deriving that order at
#  each use.  Section identity survives as .sec / .type labels.
#
#  Discretization is NEURON's d_lambda rule, so a cell discretized here has the
#  same spatial resolution as the published model it came from.  This matters:
#  under-discretizing a thin distal dendrite changes the bAP amplitude there by
#  more than any of the channel densities do.
# ════════════════════════════════════════════════════════════════════════════
import os, re
import numpy as np

# SWC structure identifiers (columns 2), extended by convention past 4.
SOMA, AXON, BASAL, APICAL = 1, 2, 3, 4
TYPE_NAME = {SOMA: "soma", AXON: "axon", BASAL: "basal", APICAL: "apical"}

# hoc section-name stem -> SWC type.  The geometry files exported by the
# Duke-Southampton / Neurolucida chain use these names; "user5" is what that
# exporter emits for the apical tuft, and mis-typing it as basal would put the
# tuft dendrites on the wrong side of the soma in the laminar model.
_HOC_TYPE = [("axon", AXON), ("soma", SOMA), ("apical", APICAL), ("user5", APICAL),
             ("dend", BASAL), ("basal", BASAL)]


class Section:
    """One unbranched cable: a 3-D point list plus its attachment.

    points: (M, 4) float [x, y, z, diam] in um.  parent is an index into the
    owning list (-1 for the root), parent_x the 0..1 position on the parent
    where this section's 0-end attaches.
    """

    __slots__ = ("name", "points", "parent", "parent_x", "type")

    def __init__(self, name, type_=BASAL):
        self.name = name; self.points = []; self.parent = -1
        self.parent_x = 1.0; self.type = type_

    def arr(self):
        return np.asarray(self.points, float).reshape(-1, 4)

    def length(self):
        p = self.arr()
        return 0.0 if len(p) < 2 else float(np.linalg.norm(np.diff(p[:, :3], axis=0), axis=1).sum())


def _type_of(name):
    low = name.lower()
    for stem, t in _HOC_TYPE:
        if low.startswith(stem):
            return t
    return BASAL


# ── SWC ─────────────────────────────────────────────────────────────────────
def load_swc(path):
    """Read an SWC reconstruction into Sections.

    SWC is a per-POINT format; sections are recovered by walking each unbranched
    run between branch points / type changes.  Multi-point somata (the usual
    three-point or contour form) collapse to one section, which is what NEURON's
    import3d does and what keeps the soma a single compartment below.
    """
    idx, xyz, rad, typ, par = [], [], [], [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            f = line.split()
            if len(f) < 7:
                continue
            idx.append(int(f[0])); typ.append(int(float(f[1])))
            xyz.append([float(f[2]), float(f[3]), float(f[4])])
            rad.append(float(f[5])); par.append(int(float(f[6])))
    if not idx:
        raise ValueError(f"{path}: no SWC records")
    remap = {v: i for i, v in enumerate(idx)}
    xyz = np.asarray(xyz, float); rad = np.asarray(rad, float)
    typ = np.asarray(typ, int)
    parent = np.array([remap.get(p, -1) for p in par], int)

    nchild = np.zeros(len(idx), int)
    for p in parent:
        if p >= 0:
            nchild[p] += 1
    root = int(np.flatnonzero(parent < 0)[0])

    secs, sec_of_point, order = [], {}, []
    # A new section starts at the root, at any child of a branch point, and at
    # any type change (so the soma never fuses with the dendrite it feeds).
    starts = [root] + [i for i in range(len(idx)) if parent[i] >= 0 and
                       (nchild[parent[i]] > 1 or typ[i] != typ[parent[i]])]
    for s in starts:
        sec = Section(f"{TYPE_NAME.get(int(typ[s]), 'dend')}[{len(secs)}]", int(typ[s]))
        chain, cur = [s], s
        while nchild[cur] == 1:
            nxt = next(i for i in range(len(idx)) if parent[i] == cur)
            if typ[nxt] != typ[s]:
                break
            chain.append(nxt); cur = nxt
        if parent[s] >= 0:                       # start at the parent point so
            sec.points.append([*xyz[parent[s]], 2 * rad[parent[s]]])   # cables abut
        for c in chain:
            sec.points.append([*xyz[c], 2 * rad[c]])
        sec_of_point[s] = len(secs)
        for c in chain:
            order.append((c, len(secs)))
        secs.append(sec)
    point_sec = dict(order)
    for si, s in enumerate(starts):
        if parent[s] >= 0:
            secs[si].parent = point_sec[parent[s]]
            secs[si].parent_x = 1.0
    if len(secs[0].points) == 1:                 # single-point soma -> sphere as
        x, y, z, d = secs[0].points[0]           # an equivalent cylinder L = d
        secs[0].points = [[x, y - d / 2, z, d], [x, y + d / 2, z, d]]
    return secs


# ── NEURON geometry hoc ─────────────────────────────────────────────────────
_RE_CREATE = re.compile(r"\bcreate\s+(.+)")
_RE_ACCESS = re.compile(r"\baccess\s+([A-Za-z_]\w*(?:\[\s*[\d.]+\s*\])?)")
_RE_PT3D = re.compile(r"\bpt3dadd\s*\(([^)]*)\)")
_RE_CONNECT = re.compile(
    r"(?:([A-Za-z_]\w*(?:\[\s*[\d.]+\s*\])?)\s+)?connect\s+"
    r"([A-Za-z_]\w*(?:\[\s*[\d.]+\s*\])?)\s*\(\s*([\d.]+)\s*\)\s*,\s*(.+)")
_RE_BARE = re.compile(r"^\s*[A-Za-z_]\w*(?:\[\s*[\d.]+\s*\])?\s*$")
_RE_SECREF = re.compile(r"([A-Za-z_]\w*(?:\[\s*[\d.]+\s*\])?)\s*\(\s*([\d.]+)\s*\)")


_RE_FOR = re.compile(r"\bfor\s+([A-Za-z_]\w*)\s*=\s*(-?\d+)\s*,\s*(-?\d+)\s*\{([^{}]*)\}", re.S)
_RE_IDXEXPR = re.compile(r"\[([^\]]+)\]")
_SAFE_IDX = re.compile(r"^[\d\s+\-*/()]+$")


def _expand_for(txt, max_iter=8):
    """Unroll `for i=a,b { ... }` bodies (no nesting) with the index substituted.

    Reduced-morphology cell templates chain their dendrites in a loop
    (`for i=0,3 { connect dend[i+1](0), dend[i](1) }`).  Without unrolling, the
    connect statements are simply not seen and the cell loads as a pile of
    disconnected stubs -- which is worse than failing, so the alternative to
    this ~20 lines is to reject those files outright.
    """
    for _ in range(max_iter):
        m = _RE_FOR.search(txt)
        if not m:
            break
        var, a, b, body = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
        out = []
        for k in range(a, b + 1):
            piece = re.sub(rf"\b{re.escape(var)}\b", str(k), body)
            out.append(_RE_IDXEXPR.sub(_eval_idx, piece))
        txt = txt[:m.start()] + "\n".join(out) + txt[m.end():]
    return txt


def _eval_idx(m):
    e = m.group(1)
    if not _SAFE_IDX.match(e):
        return m.group(0)
    try:
        return "[" + str(int(eval(e, {"__builtins__": {}}, {}))) + "]"   # noqa: S307
    except Exception:
        return m.group(0)


def _norm(nm):
    """Canonical section name.  Some archive geometry files write the index as a
    float ("soma[1.]", "apical_dendrite[0.]"); NEURON accepts it, so a parser
    that does not will silently build a disconnected tree rather than fail."""
    nm = nm.replace(" ", "")
    m = re.match(r"([A-Za-z_]\w*)(?:\[\s*([\d.]+)\s*\])?$", nm)
    if not m:
        return nm
    return f"{m.group(1)}[{int(float(m.group(2) or 0))}]"


def load_hoc(path):
    """Read a NEURON geometry .hoc (create / connect / pt3dadd) into Sections.

    Deliberately NOT a hoc interpreter: it recognizes the four statements that
    a Neurolucida-exported geometry file consists of.  Anything else (proc
    bodies, biophysics, GUI) is skipped.  A cell whose morphology is built by
    procedural code with L= / diam= assignments in loops is NOT handled --
    it will raise rather than silently return a cell with no 3-D points, because
    a morphology that is silently wrong is far worse here than a missing one.
    """
    with open(path, errors="ignore") as fh:
        raw = fh.read()
    txt = re.sub(r"/\*.*?\*/", " ", raw, flags=re.S)
    txt = re.sub(r"//[^\n]*", " ", txt)
    # array sizes are often symbolic ("create soma[NumSoma], dend[NumApical]");
    # resolve the simple integer assignments so the sections still get created.
    ints = {m.group(1): m.group(2) for m in
            re.finditer(r"^\s*([A-Za-z_]\w*)\s*=\s*(\d+)\s*$", txt, flags=re.M)}
    if ints:
        txt = re.sub(r"\[\s*([A-Za-z_]\w*)\s*\]",
                     lambda m: "[" + ints.get(m.group(1), m.group(1)) + "]", txt)
    txt = _expand_for(txt)
    txt = txt.replace("{", "\n").replace("}", "\n").replace(";", "\n")

    order, byname = [], {}

    def get(nm):
        nm = _norm(nm)
        if nm not in byname:
            s = Section(nm, _type_of(nm)); byname[nm] = s; order.append(nm)
        return byname[nm]

    current = None
    for line in txt.split("\n"):
        line = line.strip()
        if not line:
            continue
        m = _RE_CREATE.search(line)
        if m and "connect" not in line:
            for decl in m.group(1).split(","):
                decl = decl.strip()
                mm = re.match(r"([A-Za-z_]\w*)\s*(?:\[\s*([\d.]+)\s*\])?", decl)
                if not mm:
                    continue
                n = int(float(mm.group(2) or 1))
                for k in range(n):
                    get(f"{mm.group(1)}[{k}]")
            continue
        m = _RE_CONNECT.search(line)
        if m:
            pre, child, cx, rest = m.groups()
            ch = get(child)
            mr = _RE_SECREF.search(rest)
            if mr:                                   # connect a(0), b(1)
                par, px = mr.group(1), float(mr.group(2))
            else:                                    # <sec> connect a(0), 1
                par, px = pre or current, float(rest.strip().split()[0])
            if par is None:
                continue
            ch.parent = order.index(_norm(par)); ch.parent_x = px
            if float(cx) > 0.5:                      # 1-end attachment: store the
                ch.parent_x = px                     # points reversed at build time
            continue
        m = _RE_ACCESS.search(line)
        if m:
            current = m.group(1); get(current)
            continue
        # "soma[0] { pt3dclear() ... }" -- after brace splitting the section name
        # is left alone on its own line.  Only accept it if the section already
        # exists (i.e. was created), so a stray identifier cannot invent one.
        if _RE_BARE.match(line) and _norm(line) in byname:
            current = line.strip()
            continue
        for m in _RE_PT3D.finditer(line):
            f = [float(v) for v in m.group(1).split(",")[:4]]
            if len(f) == 4 and current is not None:
                get(current).points.append(f)

    secs = [byname[n] for n in order]
    secs = [s for s in secs if len(s.points) >= 1]
    keep = {byname[n].name: i for i, n in enumerate(order) if len(byname[n].points) >= 1}
    remap = {order.index(n): keep[byname[n].name] for n in order if byname[n].name in keep}
    for s in secs:
        s.parent = remap.get(s.parent, -1)
    if not secs or sum(len(s.points) for s in secs) < 2:
        raise ValueError(f"{path}: no pt3d geometry found (procedural morphology?)")
    for s in secs:
        if len(s.points) == 1:
            x, y, z, d = s.points[0]
            s.points = [[x, y - d / 2, z, d], [x, y + d / 2, z, d]]
    return secs


def load(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".swc":
        return load_swc(path)
    if ext in (".hoc", ".ses"):
        return load_hoc(path)
    raise ValueError(f"{path}: unsupported morphology format '{ext}'")


# ── compartments ────────────────────────────────────────────────────────────
class Compartments:
    """Flat compartment arrays, topologically ordered (parent[i] < i, root 0).

    p0/p1 (N,3) endpoints and mid (N,3) centres in um; L, diam in um; area in
    um^2; parent (N,) int; sec (N,) section index; type (N,) SWC type;
    pathdist (N,) path distance in um from the soma centre.
    """

    __slots__ = ("p0", "p1", "mid", "L", "diam", "area", "parent", "sec", "type",
                 "pathdist", "name")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __len__(self):
        return len(self.L)

    @property
    def is_soma(self):
        return self.type == SOMA

    def summary(self):
        by = {}
        for t in np.unique(self.type):
            m = self.type == t
            by[TYPE_NAME.get(int(t), str(t))] = (int(m.sum()), float(self.L[m].sum()))
        return by


def lambda_f(diam, Ra, cm, f=100.0):
    """AC length constant (um) at frequency f -- NEURON's lambda_f."""
    return 1e5 * np.sqrt(diam / (4 * np.pi * f * Ra * cm))


def _nseg_dlambda(L, diam, Ra, cm, d_lambda=0.1, f=100.0):
    if L <= 0 or diam <= 0:
        return 1
    return int((L / (d_lambda * lambda_f(diam, Ra, cm, f)) + 0.9) / 2) * 2 + 1


def compartmentalize(secs, d_lambda=0.1, Ra=150.0, cm=1.0, f=100.0, max_comp=4000,
                     require_connected=True):
    """Discretize sections by the d_lambda rule into ordered compartments.

    max_comp caps total compartments by coarsening d_lambda uniformly; a
    reconstruction with 3000 sections would otherwise produce a solve that is
    accurate to no useful purpose and slow enough to discourage running it.
    """
    for _ in range(12):
        nseg = [_nseg_dlambda(s.length(), float(np.mean(s.arr()[:, 3])), Ra, cm, d_lambda, f)
                for s in secs]
        if sum(nseg) <= max_comp:
            break
        d_lambda *= 1.35
    nseg = [1 if secs[i].type == SOMA else n for i, n in enumerate(nseg)]

    # per-section compartment slabs, then a BFS renumber to get parent[i] < i
    segs = []            # (sec_index, k, p0, p1, L, diam, area)
    for si, s in enumerate(secs):
        p = s.arr()
        if len(p) < 2:
            continue
        seg_len = np.linalg.norm(np.diff(p[:, :3], axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg_len)])
        total = cum[-1]
        if total <= 0:
            continue
        n = nseg[si]
        edges = np.linspace(0.0, total, n + 1)

        def at(dist):
            j = int(np.clip(np.searchsorted(cum, dist) - 1, 0, len(seg_len) - 1))
            u = 0.0 if seg_len[j] <= 0 else (dist - cum[j]) / seg_len[j]
            pos = p[j, :3] + u * (p[j + 1, :3] - p[j, :3])
            dia = p[j, 3] + u * (p[j + 1, 3] - p[j, 3])
            return pos, dia

        for k in range(n):
            a, b = edges[k], edges[k + 1]
            pa, da = at(a); pb, db = at(b)
            # lateral area of the frustum chain inside [a, b]
            area, cuts = 0.0, [a] + [c for c in cum if a < c < b] + [b]
            for u0, u1 in zip(cuts[:-1], cuts[1:]):
                q0, e0 = at(u0); q1, e1 = at(u1)
                area += np.pi * (e0 + e1) / 2.0 * float(np.linalg.norm(q1 - q0))
            segs.append([si, k, pa, pb, b - a, (da + db) / 2.0, area])

    if not segs:
        raise ValueError("morphology produced no compartments")

    # index of (section, k)
    key = {(g[0], g[1]): i for i, g in enumerate(segs)}
    nseg_of = {}
    for g in segs:
        nseg_of[g[0]] = max(nseg_of.get(g[0], 0), g[1] + 1)

    raw_parent = np.full(len(segs), -1, int)
    for (si, k), i in key.items():
        if k > 0:
            raw_parent[i] = key[(si, k - 1)]
        else:
            ps = secs[si].parent
            if ps >= 0 and ps in nseg_of:
                x = float(np.clip(secs[si].parent_x, 0.0, 1.0))
                pk = int(np.clip(int(x * nseg_of[ps]), 0, nseg_of[ps] - 1))
                raw_parent[i] = key[(ps, pk)]

    roots = np.flatnonzero(raw_parent < 0)
    soma_i = [i for i in roots if secs[segs[i][0]].type == SOMA]
    root = int(soma_i[0]) if soma_i else int(roots[0])
    if len(roots) > 1 and require_connected:
        # A detached fragment still contributes membrane current but carries no
        # axial current, so it silently adds a spurious current source to the
        # extracellular field and changes the cell's input resistance.  This is
        # nearly always a morphology the loader could not fully connect, not a
        # real cell, so refuse rather than produce a plausible wrong answer.
        bad = [secs[segs[i][0]].name for i in roots if i != root][:6]
        raise ValueError(
            f"morphology is disconnected: {len(roots)} roots "
            f"(e.g. {', '.join(bad)}) — the loader did not resolve every "
            f"connect statement; pass require_connected=False to simulate the "
            f"soma-connected part only")

    children = {}
    for i, p in enumerate(raw_parent):
        if p >= 0:
            children.setdefault(int(p), []).append(i)
    order, stack = [], [root]
    while stack:
        i = stack.pop(0); order.append(i); stack.extend(children.get(i, []))
    if not require_connected:                    # explicit opt-in: keep only the
        order = list(order)                      # soma-connected component
    new = {o: i for i, o in enumerate(order)}

    N = len(order)
    p0 = np.zeros((N, 3)); p1 = np.zeros((N, 3))
    L = np.zeros(N); diam = np.zeros(N); area = np.zeros(N)
    parent = np.full(N, -1, int); sec = np.zeros(N, int); typ = np.zeros(N, int)
    for o, i in new.items():
        si, k, pa, pb, ln, dd, ar = segs[o]
        p0[i] = pa; p1[i] = pb; L[i] = ln; diam[i] = dd; area[i] = ar
        sec[i] = si; typ[i] = secs[si].type
        parent[i] = new.get(int(raw_parent[o]), -1) if raw_parent[o] >= 0 else -1

    mid = (p0 + p1) / 2.0
    pathdist = np.zeros(N)
    for i in range(1, N):
        p = parent[i]
        pathdist[i] = (pathdist[p] + (L[p] + L[i]) / 2.0) if p >= 0 else 0.0
    return Compartments(p0=p0, p1=p1, mid=mid, L=L, diam=diam, area=area,
                        parent=parent, sec=sec, type=typ, pathdist=pathdist, name="")


def orient(cmp_, axis=None, centre=True):
    """Rotate so the somato-apical axis lies along +y and (optionally) put the
    soma centre at the origin.  +y is the probe's depth axis in fiber-kit's
    .probe geometry, so this is what makes a simulated cell comparable to a
    recorded one; without it, a cell's footprint depends on how the
    reconstruction happened to be stored.

    The axis is the soma -> apical/dendritic centre-of-mass vector (weighted by
    membrane area), not the first principal axis: for a pyramidal cell the two
    agree, but for a symmetric multipolar cell the principal axis is arbitrary
    while the soma->dendrite vector is at least reproducible.
    """
    soma = cmp_.mid[cmp_.type == SOMA]
    origin = soma.mean(0) if len(soma) else cmp_.mid[0]
    if axis is None:
        m = cmp_.type == APICAL
        if not m.any():
            m = (cmp_.type == BASAL) | (cmp_.type == APICAL)
        if not m.any():
            axis = np.array([0.0, 1.0, 0.0])
        else:
            w = cmp_.area[m]
            axis = (cmp_.mid[m] - origin).T @ w / max(w.sum(), 1e-12)
    axis = np.asarray(axis, float)
    n = np.linalg.norm(axis)
    R = np.eye(3) if n < 1e-9 else _rot_to_y(axis / n)
    out = Compartments(**{k: getattr(cmp_, k) for k in Compartments.__slots__})
    for k in ("p0", "p1", "mid"):
        v = getattr(cmp_, k) - (origin if centre else 0.0)
        setattr(out, k, v @ R.T)
    return out


def _rot_to_y(u):
    """Rotation taking unit vector u to +y (Rodrigues; handles the antipode)."""
    t = np.array([0.0, 1.0, 0.0])
    v = np.cross(u, t); c = float(np.dot(u, t))
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * (1.0 / (1.0 + c))


def rotate_z(cmp_, deg):
    """Rotate about the probe's depth axis (y).  Used to sample the cell's
    orientation relative to the shank plane, which is unobserved in a real
    recording and is therefore a nuisance dimension of waveform variance."""
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    out = Compartments(**{k: getattr(cmp_, k) for k in Compartments.__slots__})
    for k in ("p0", "p1", "mid"):
        setattr(out, k, getattr(cmp_, k) @ R.T)
    return out


def translate(cmp_, dx):
    out = Compartments(**{k: getattr(cmp_, k) for k in Compartments.__slots__})
    for k in ("p0", "p1", "mid"):
        setattr(out, k, getattr(cmp_, k) + np.asarray(dx, float))
    return out


# ── cable templates (L/diam, no 3-D points) ─────────────────────────────────
# Laminar direction hints from NEURON section-name stems.  Reduced CA1 templates
# name their sections after the layer the compartment lies in -- radT/radM/radt
# for radiatum thick/medium/thin, lm* for lacunosum-moleculare, ori* for oriens
# -- so the names carry real anatomy that L and diam alone do not.  A layout that
# ignored them would put oriens dendrites above the soma.
_LAMINAR = [
    ("lm", (0.0, 1.0, 0.0), APICAL),        # lacunosum-moleculare: distal apical
    ("rad", (0.0, 1.0, 0.0), APICAL),       # radiatum: apical
    ("ori", (0.0, -1.0, 0.0), BASAL),       # oriens: basal
    ("axon", (0.0, -1.0, 0.0), AXON),
    ("soma", (0.0, 1.0, 0.0), SOMA),
    ("apic", (0.0, 1.0, 0.0), APICAL),
    ("dend", (1.0, 0.0, 0.0), BASAL),       # unqualified: horizontal (OLM-like)
]


def _laminar(name):
    low = name.lower()
    for stem, d, t in _LAMINAR:
        if low.startswith(stem):
            return np.array(d, float), t
    return np.array([1.0, 0.0, 0.0]), BASAL


_RE_TEMPLATE = re.compile(r"begintemplate\s+(\w+)(.*?)endtemplate", re.S)
_RE_LDIAM = re.compile(r"([A-Za-z_]\w*(?:\[\s*\d+\s*\])?)\s*\{[^{}]*?\bL\s*=\s*([\d.eE+-]+)"
                       r"[^{}]*?\bdiam\s*=\s*([\d.eE+-]+)", re.S)


def load_cable_template(path, template=None, spread_deg=25.0):
    """Load a NEURON cell template that has L/diam but NO pt3dadd.

    Reduced multi-compartment models -- the Santhakumar / Cutsuridis / Bezaire
    CA1 interneuron lineage among them -- specify geometry as lengths and
    diameters plus a connect topology, and leave 3-D placement to NEURON's
    define_shape().  load_hoc() refuses these on purpose, because a morphology
    that silently loads with no geometry is worse than one that fails.  This is
    the deliberate path for them.

    The layout is ALREADY in laminar coordinates -- +y is apical by construction,
    because that is what the name hints encode -- so callers must NOT re-orient
    one of these with orient()'s default axis search.  Doing so rotates a cell
    whose dendritic mass is symmetric (an OLM's two opposed horizontal dendrites,
    for instance) onto an arbitrary axis and stands it upright.  Use
    orient(c, axis=(0, 1, 0)) to centre on the soma without rotating.

    The result is a STYLIZED layout, not a reconstruction: cable dimensions and
    topology are the published ones, but the 3-D positions are laid out here
    from the section names' laminar hints.  Sibling branches are fanned by
    spread_deg so they do not superimpose, which is a drawing choice with no
    anatomical content.  Never report one of these as a reconstruction, and
    expect its extracellular footprint to be smoother than a real cell's --
    it has no branch-point clutter.
    """
    with open(path, errors="ignore") as fh:
        raw = fh.read()
    txt = re.sub(r"/\*.*?\*/", " ", raw, flags=re.S)
    txt = re.sub(r"//[^\n]*", " ", txt)
    tmpls = _RE_TEMPLATE.findall(txt)
    if tmpls:
        body = None
        for nm, b in tmpls:
            if template is None or nm == template:
                body = b; template = nm; break
        if body is None:
            raise ValueError(f"{path}: template {template!r} not found; "
                             f"have {[n for n, _ in tmpls]}")
        txt = body

    dims = {}
    for m in _RE_LDIAM.finditer(txt):
        dims.setdefault(_norm(m.group(1)), (float(m.group(2)), float(m.group(3))))
    if not dims:
        raise ValueError(f"{path}: no 'sec {{ L=.. diam=.. }}' assignments found")

    flat = _expand_for(txt).replace("{", "\n").replace("}", "\n").replace(";", "\n")
    order = []
    for line in flat.split("\n"):
        m = _RE_CREATE.search(line)
        if m and "connect" not in line:
            for decl in m.group(1).split(","):
                mm = re.match(r"\s*([A-Za-z_]\w*)\s*(?:\[\s*([\d.]+)\s*\])?", decl)
                if mm and mm.group(1):
                    for k in range(int(float(mm.group(2) or 1))):
                        n = _norm(f"{mm.group(1)}[{k}]")
                        if n not in order:
                            order.append(n)
    order = [n for n in order if n in dims]
    if not order:
        raise ValueError(f"{path}: create statements and L/diam names do not intersect")
    idx = {n: i for i, n in enumerate(order)}

    parent = [-1] * len(order)
    parent_x = [1.0] * len(order)
    for line in flat.split("\n"):
        m = _RE_CONNECT.search(line)
        if not m:
            continue
        _, child, cx, rest = m.groups()
        mr = _RE_SECREF.search(rest)
        if not mr:
            continue
        c, p = _norm(child), _norm(mr.group(1))
        if c in idx and p in idx:
            parent[idx[c]] = idx[p]
            parent_x[idx[c]] = float(mr.group(2))

    # place: walk parents-first, each section leaving its parent's attachment end
    secs = [None] * len(order)
    nchild = {}
    for i, p in enumerate(parent):
        if p >= 0:
            nchild[p] = nchild.get(p, 0) + 1
    seen = {}
    remaining = list(range(len(order)))
    guard = 0
    while remaining and guard < 10 * len(order):
        guard += 1
        i = remaining.pop(0)
        p = parent[i]
        if p >= 0 and secs[p] is None:
            remaining.append(i); continue
        L, d = dims[order[i]]
        u, typ = _laminar(order[i])
        if p < 0:
            start = np.zeros(3)
        else:
            pp = np.asarray(secs[p].points, float)
            start = pp[-1, :3] if parent_x[i] > 0.5 else pp[0, :3]
            k = seen.get(p, 0); seen[p] = k + 1
            if nchild.get(p, 1) > 1:                      # fan siblings apart
                a = np.radians(spread_deg) * (k - (nchild[p] - 1) / 2.0)
                c, s = np.cos(a), np.sin(a)
                u = np.array([u[0] * c - u[1] * s, u[0] * s + u[1] * c, u[2]])
        sec = Section(order[i], typ)
        if typ == SOMA:
            sec.points = [[start[0], start[1] - L / 2, start[2], d],
                          [start[0], start[1] + L / 2, start[2], d]]
        else:
            end = start + u * L
            sec.points = [[*start, d], [*end, d]]
        sec.parent = p; sec.parent_x = parent_x[i]
        secs[i] = sec
    if any(s is None for s in secs):
        raise ValueError(f"{path}: could not place every section (cyclic connect?)")
    return secs
