#  fiber_config_editor.py — Qt6 (>= 6.8) form editor for a fiber-kit.yaml knob file.
#
#  `fiber-config-edit [fiber-kit.yaml]` opens the config ($FK_CONFIG, else ./fiber-kit.yaml) as a form:
#     LEFT    the file's own sections ("[1] FIBER-SESSION ...", the "-----" sub-headings, the embedded
#             pipeline plan), each a drop-down.  One row per FK_* knob, typed from its comment:
#               `0 | 0/1`, `0 = off, 1 = on`      -> a switch
#               `auto | nst | fil | spk`          -> a choice list
#               anything else                     -> a text field (numbers right-aligned; amber when the
#                                                    value leaves the documented `lo-hi` range)
#             A knob that is OFF is greyed, and so is every knob it gates (an off FK_<NAME> greys every FK_<NAME>_*).
#             A value that differs from the opened file is marked; one that differs from the documented
#             default shows that default, with a button to go back to it.
#             Plan steps are rows with a tick: unticking comments the step out, ticking brings it back.
#     RIGHT   the help pane: whatever the pointer is over (or the keyboard focus is in) shows its full
#             comment there -- a knob's explanation, a section's header text, a note between knobs.
#
#  LOSSLESS.  The file is never re-serialised: the editor rewrites only the value token of a line it
#  changed (padding adjusted so the comment column stays put where it can) and the leading "# " of a
#  plan step it toggled.  Every comment, blank line and column of an untouched line is written back
#  byte for byte, and saving an unedited file reproduces it exactly.
#
#  The model below (parse, typing, off/gating, set/toggle, text) is Qt-free and unit-tested
#  (test/test_config_editor.py); the PySide6 view is built only when the window is opened.
#  `fiber-config-edit --check FILE` prints what the model reads from a file, without Qt.

import difflib
import os
import re
import sys
from dataclasses import dataclass, field

_KNOB = re.compile(r"^(FK_[A-Z0-9_]+):([ \t]*)(.*)$")
_BANNER = re.compile(r"^# ?={8,}\s*$")
_SUB = re.compile(r"^# ?-{3,}\s*(.*?)\s*-*\s*$")
_STEP = re.compile(r"^(\s+)(#\s?)?(-\s*\{.*\})(\s*#.*)?\s*$")
_NUM = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")
_RANGE = re.compile(r"^\s*([-+]?\d+\.?\d*)\s*-\s*([-+]?\d+\.?\d*)(?![\w.])")
_WORD = re.compile(r'^(""|[A-Za-z][\w.\-]*)$')
_OFF_HINT = re.compile(r'(?:\b0(?:\.0)?|"")\s*=\s*(?:off|OFF|disabled|no\b|none|term off)|\b0 disables|> ?0 enables'
                       r'|\(0 = [^)]*off\)|0 = (?:single|not|never)')
_PLACEHOLDER = {"path", "N", "n", "name", "file", "list", "stage", "tag", "token", "variant"}
_PLAIN_UNSAFE = re.compile(r'^[\[\]{}&*!|>\'"%@`,#?-]|: | #|^\s|\s$')


def is_number(s):
    return bool(_NUM.match(str(s).strip()))


def split_value(rest):
    """`rest` is what follows the knob name, its colon and its padding.  -> (value token, gap, comment incl. '#', quote char)."""
    if rest[:1] in ('"', "'"):
        q = rest[0]
        j = rest.find(q, 1)
        if j > 0:
            tail = rest[j + 1:]
            m = re.match(r"^(\s*)(#.*)?$", tail)
            if m:
                return rest[:j + 1], m.group(1) or "", m.group(2) or "", q
    m = re.match(r"^(.*?)(\s+)(#.*)$", rest)
    if m:
        return m.group(1), m.group(2), m.group(3), ""
    if rest.startswith("#"):
        return "", "", rest, ""
    return rest.rstrip(), rest[len(rest.rstrip()):], "", ""


@dataclass
class Knob:
    line: int                    # index of the knob's own line
    name: str
    pad: str                     # whitespace between the colon and the value
    token: str                   # the value exactly as written (quotes included)
    gap: str                     # whitespace between the value and the comment
    comment: str                 # the rest of the line, from '#'
    quote: str                   # '"', "'" or ""
    help: str = ""               # the comment and its continuation lines, de-commented
    kind: str = "text"           # bool | enum | number | text
    choices: tuple = ()
    default: object = None       # documented default (text), or None when the comment gives none
    range: str = ""              # the documented range, as written
    lo: object = None
    hi: object = None
    orig_token: str = ""         # the token in the file as opened
    section: object = None

    @property
    def value(self):
        t = self.token
        return t[1:-1] if self.quote and len(t) >= 2 else t

    @property
    def orig_value(self):
        t = self.orig_token
        return t[1:-1] if t[:1] in ('"', "'") and len(t) >= 2 and t[-1] == t[0] else t

    @property
    def changed(self):
        return self.token != self.orig_token

    @property
    def off_default(self):
        """Differs from the documented default (False when the comment documents none)."""
        if self.default is None:
            return False
        a, b = self.value, str(self.default)
        if is_number(a) and is_number(b):
            return float(a) != float(b)
        return a != b

    @property
    def out_of_range(self):
        if self.lo is None or not is_number(self.value):
            return False
        v = float(self.value)
        if self.default is not None and is_number(self.default) and v == float(self.default):
            return False                                  # the default is legal even outside the tuning range
        return not (self.lo <= v <= self.hi)

    def is_off(self):
        """The knob's own value switches something off."""
        v = self.value.strip().lower()
        if self.kind == "bool":
            return v in ("0", "false", "off", "no", "")
        if v in ("off", "none"):
            return True
        if v == "" or (is_number(v) and float(v) == 0.0):
            return bool(_OFF_HINT.search(self.help)) or (v == "" and self.kind != "text")
        return False


@dataclass
class Step:
    line: int
    indent: str
    body: str                    # "- {stage: ..., ...}"
    tail: str                    # trailing "   # comment" or ""
    enabled: bool
    orig_enabled: bool
    fields: dict = field(default_factory=dict)
    section: object = None

    @property
    def changed(self):
        return self.enabled != self.orig_enabled


@dataclass
class Note:
    line: int                    # first line
    text: str
    sub: bool = False            # a "# ----- title -----" sub-heading
    section: object = None


@dataclass
class Section:
    line: int
    title: str
    header: str = ""             # the banner block's text
    items: list = field(default_factory=list)      # Knob | Step | Note, in file order

    @property
    def knobs(self):
        return [i for i in self.items if isinstance(i, Knob)]

    @property
    def steps(self):
        return [i for i in self.items if isinstance(i, Step)]


def _decomment(s):
    s = s.strip()
    if s.startswith("#"):
        s = s[1:]
        if s.startswith("   "):
            s = s[3:]
        elif s.startswith(" "):
            s = s[1:]
    return s.rstrip()


def _literal(tok):
    """The leading literal of a spec part ('12', '""', 'auto', '12.0 (min)' -> '12.0'), else None."""
    tok = tok.strip()
    if not tok:
        return None
    head = tok.split()[0]
    if head == '""':
        return ""
    if is_number(head):
        return head
    rest = tok[len(head):].strip()
    if _WORD.match(head) and (not rest or rest.startswith(("=", "("))):
        return head
    return None


def classify(k):
    """Type a knob from the lead of its comment: `default | range.  explanation`."""
    m = re.search(r"\.(?:[ \t]{2,}|\n|$)", k.help)
    lead = k.help[:m.start()] if m else k.help
    lead = re.sub(r"\s*\n\s*", " ", lead) if len(lead) <= 220 else ""
    k.kind, k.choices, k.default, k.range, k.lo, k.hi = "text", (), None, "", None, None
    val = k.value
    if re.match(r"^\s*0\s*=\s*off\s*,\s*1\s*=\s*on\b", lead) and val in ("0", "1"):
        k.kind = "bool"
        return k
    parts = [p.strip() for p in lead.split(" | ")] if " | " in lead else []
    if parts:
        words = [_literal(p) for p in parts]
        if len(parts) >= 2 and all(w is not None and not is_number(w) for w in words) \
                and all(_WORD.match(p.split()[0]) for p in parts) and not any(w in _PLACEHOLDER for w in words):
            # a choice list: `auto | nst | fil | spk`, `"" | accumulated | global`
            k.kind, k.choices = "enum", tuple(words)
            return k
        k.default = _literal(parts[0])
        k.range = " | ".join(parts[1:])
        rng = parts[1]
        if (re.match(r"^0\s*/\s*1\b", rng) or (rng in ("0", "1") and k.default in ("0", "1"))) and val in ("0", "1"):
            k.kind = "bool"
            return k
        if "/" in rng and not is_number(k.default or "x") and k.default is not None:
            opts = [o.strip() for o in rng.split("/")]
            if all(_WORD.match(o) for o in opts):                 # `off | off/strip/knn/both`
                k.kind, k.choices = "enum", tuple(dict.fromkeys([k.default] + opts))
                return k
        r = _RANGE.match(rng)
        if r:
            k.lo, k.hi = float(r.group(1)), float(r.group(2))
    elif lead:
        k.default = _literal(lead) if is_number((lead.split() or [""])[0]) else None
    if is_number(val) or (k.default is not None and is_number(k.default)):
        k.kind = "number"
    return k


class ConfigDoc:
    """A fiber-kit.yaml as lines plus the editable items found in them."""

    def __init__(self, text=""):
        self.newline = "\r\n" if "\r\n" in text else "\n"
        self.trailing_nl = text.endswith("\n")
        self.lines = text.split("\n")
        if self.trailing_nl:
            self.lines.pop()
        if self.newline == "\r\n":
            self.lines = [ln[:-1] if ln.endswith("\r") else ln for ln in self.lines]
        self.orig_lines = list(self.lines)
        self.sections = []
        self.knobs = {}
        self._parse()

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8", newline="") as f:
            return cls(f.read())

    # ── parse ────────────────────────────────────────────────────────────────
    def _banner_block(self, i):
        """If line i opens a `====` banner block, the index of its closing `====` line, else None."""
        L = self.lines
        if not _BANNER.match(L[i]) or i + 1 >= len(L) or not L[i + 1].startswith("#") or _BANNER.match(L[i + 1]):
            return None
        for j in range(i + 2, min(i + 80, len(L))):
            if _BANNER.match(L[j]):
                return j
            if not L[j].startswith("#"):
                return None
        return None

    def _parse(self):
        L = self.lines
        cur = Section(0, "File header")
        self.sections = [cur]
        in_plan = False
        i, n = 0, len(L)
        note = None

        def flush():
            nonlocal note
            if note is not None and note.text.strip():
                note.section = cur
                cur.items.append(note)
            note = None

        while i < n:
            ln = L[i]
            j = self._banner_block(i)
            if j is not None:
                flush()
                in_plan = False
                head = [_decomment(x) for x in L[i + 1:j]]
                title = re.sub(r"\s+", " ", head[0]).strip()
                cur = Section(i, title, "\n".join(head))
                self.sections.append(cur)
                # a long note block may share its closing rule with the next block's opening one
                i = j if (j - i > 4 and self._banner_block(j) is not None) else j + 1
                continue
            m = _KNOB.match(ln)
            if m:
                flush()
                in_plan = False
                token, gap, comment, quote = split_value(m.group(3))
                k = Knob(i, m.group(1), m.group(2), token, gap, comment, quote, orig_token=token, section=cur)
                hl = [_decomment(comment)] if comment else []
                i += 1
                while i < n and re.match(r"^\s+#", L[i]) and not _STEP.match(L[i]):
                    hl.append(_decomment(L[i]))
                    i += 1
                k.help = "\n".join(hl)
                classify(k)
                cur.items.append(k)
                self.knobs[k.name] = k
                continue
            if re.match(r"^pipeline\s*:", ln):
                flush()
                in_plan = True
                cur = Section(i, "PIPELINE PLAN -- the steps `fiber-pipeline <elec>` runs (tick = on)", cur.header)
                self.sections.append(cur)
                i += 1
                continue
            s = _STEP.match(ln) if in_plan else None
            if s:
                flush()
                en = not s.group(2)
                st = Step(i, s.group(1), s.group(3), s.group(4) or "", en, en, section=cur)
                for key in ("stage", "name", "in", "out"):
                    fm = re.search(r"[{,]\s*%s:\s*([^,}]+)" % key, st.body)
                    if fm:
                        st.fields[key] = fm.group(1).strip()
                cur.items.append(st)
                i += 1
                continue
            if ln.strip().startswith("#"):
                sm = _SUB.match(ln)
                if sm and sm.group(1):
                    flush()
                    sub = Note(i, sm.group(1), sub=True, section=cur)
                    cur.items.append(sub)
                else:
                    if note is None:
                        note = Note(i, "")
                    note.text += ("\n" if note.text else "") + _decomment(ln)
            else:
                flush()
                if ln.strip():
                    in_plan = in_plan and ln[:1] in (" ", "\t")
            i += 1
        flush()
        self.sections = [s for s in self.sections if s.items or s.header]
        self._gates = None

    # ── off / gating ─────────────────────────────────────────────────────────
    def gate_of(self, k):
        """The knob that switches `k` off (an off FK_<NAME> gates FK_<NAME>_*), or None."""
        name = k.name
        while "_" in name[3:]:
            name = name.rsplit("_", 1)[0]
            g = self.knobs.get(name)
            if g is not None and g is not k and g.is_off():
                return g
        return None

    def is_dimmed(self, k):
        return k.is_off() or self.gate_of(k) is not None

    # ── edit ─────────────────────────────────────────────────────────────────
    @staticmethod
    def encode(value, quote):
        """The token to write for a value typed by the user (quotes only where YAML needs them)."""
        v = str(value)
        if v == "":
            return (quote or '"') * 2
        if is_number(v):
            return v
        if quote or _PLAIN_UNSAFE.search(v):
            q = quote or '"'
            if q in v:
                q = "'" if q == '"' else '"'
            return q + v + q
        return v

    def set(self, name, value):
        k = self.knobs[name]
        token = self.encode(value, k.quote)
        if token == k.token:
            return False
        # keep the comment where it was: the padding absorbs the change in the value's width
        gap = k.gap
        if k.comment:
            width = len(k.token) + len(k.gap)
            gap = " " * max(2, width - len(token))
        k.token, k.gap = token, gap
        k.quote = token[0] if token[:1] in ('"', "'") else ""
        self.lines[k.line] = f"{k.name}:{k.pad}{k.token}{k.gap}{k.comment}"
        return True

    def reset(self, name):
        k = self.knobs[name]
        if k.changed:
            self.lines[k.line] = self.orig_lines[k.line]
            m = _KNOB.match(self.lines[k.line])
            k.token, k.gap, k.comment, k.quote = split_value(m.group(3))

    def set_step(self, step, enabled):
        if enabled == step.enabled:
            return False
        step.enabled = bool(enabled)
        if step.enabled == step.orig_enabled:
            self.lines[step.line] = self.orig_lines[step.line]
        else:
            self.lines[step.line] = f"{step.indent}{'' if enabled else '# '}{step.body}{step.tail}"
        return True

    # ── out ──────────────────────────────────────────────────────────────────
    def text(self):
        return self.newline.join(self.lines) + (self.newline if self.trailing_nl else "")

    def orig_text(self):
        return self.newline.join(self.orig_lines) + (self.newline if self.trailing_nl else "")

    def changes(self):
        out = [(k.name, k.orig_value, k.value) for k in self.knobs.values() if k.changed]
        for s in self.sections:
            for st in s.steps:
                if st.changed:
                    out.append((f"plan step {st.fields.get('name', st.fields.get('stage', '?'))}",
                                "on" if st.orig_enabled else "off", "on" if st.enabled else "off"))
        return out

    def diff(self, name="fiber-kit.yaml"):
        return "".join(difflib.unified_diff([x + "\n" for x in self.orig_lines], [x + "\n" for x in self.lines],
                                            name, name + " (edited)", n=0))

    def validate(self):
        """YAML-parse the edited text.  -> error text, or '' (also '' when no YAML parser is installed)."""
        try:
            try:
                from .config import yaml_load
            except ImportError:
                from config import yaml_load
        except Exception:
            return ""
        try:
            d = yaml_load(self.text())
        except Exception as e:                                   # noqa: BLE001 - shown to the user verbatim
            return str(e)
        if not isinstance(d, dict):
            return "the file no longer parses to a mapping"
        missing = [n for n in self.knobs if n not in d]
        return f"knob(s) lost by the edit: {', '.join(missing[:5])}" if missing else ""

    def save(self, path):
        err = self.validate()
        if err:
            raise ValueError(err)
        tmp = path + ".tmp~"
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            f.write(self.text())
        os.replace(tmp, path)
        self.orig_lines = list(self.lines)
        for k in self.knobs.values():
            k.orig_token = k.token
        for s in self.sections:
            for st in s.steps:
                st.orig_enabled = st.enabled


def reflow(text):
    """Help text -> [(kind, text)] blocks for a narrow pane: prose lines are joined into paragraphs
    ('p'); lines that carry aligned columns, and the lines around them, stay preformatted ('pre')."""
    blocks = []
    for ln in text.split("\n"):
        body = ln.strip()
        if not body:
            blocks.append(("gap", ""))
            continue
        table = bool(re.search(r"\S {3,}\S", body)) or body[:1] in "|+"
        indent = len(ln) - len(ln.lstrip())
        bullet = bool(re.match(r"^([*\-•]|\d+[.)]|\[\w+\])\s", body))
        if table:
            if blocks and blocks[-1][0] == "pre":
                blocks[-1] = ("pre", blocks[-1][1] + "\n" + ln.rstrip())
            else:
                blocks.append(("pre", ln.rstrip()))
        elif blocks and blocks[-1][0] == "p" and not bullet and indent <= blocks[-1][2]:
            blocks[-1] = ("p", blocks[-1][1] + " " + body, blocks[-1][2])
        else:
            blocks.append(("p", body, indent))
    return [(b[0], b[1]) for b in blocks if b[0] != "gap"]


def default_path():
    p = os.environ.get("FK_CONFIG", "")
    return p if p else "fiber-kit.yaml"


def check_main(path):
    doc = ConfigDoc.load(path)
    with open(path, encoding="utf-8", newline="") as f:
        same = doc.text() == f.read()
    kinds = {}
    for k in doc.knobs.values():
        kinds[k.kind] = kinds.get(k.kind, 0) + 1
    print(f"{path}: {len(doc.sections)} sections, {len(doc.knobs)} knobs "
          f"({', '.join(f'{v} {n}' for n, v in sorted(kinds.items()))}), "
          f"{sum(len(s.steps) for s in doc.sections)} plan steps "
          f"({sum(st.enabled for s in doc.sections for st in s.steps)} on); round trip {'exact' if same else 'DIFFERS'}")
    off = [k for k in doc.knobs.values() if k.is_off()]
    gated = [k for k in doc.knobs.values() if not k.is_off() and doc.gate_of(k) is not None]
    print(f"  off: {len(off)} knobs; gated off by another knob: {len(gated)}; "
          f"differ from the documented default: {sum(k.off_default for k in doc.knobs.values())}")
    for k in doc.knobs.values():
        if k.out_of_range:
            print(f"  outside the documented range: {k.name} = {k.value}  (range {k.range})")
    err = doc.validate()
    if err:
        print(f"  YAML: {err}")
    return 0 if same and not err else 1


# ── Qt view ──────────────────────────────────────────────────────────────────
def _build_window(path):
    from PySide6 import QtCore, QtGui, QtWidgets
    Qt = QtCore.Qt

    mono = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)

    class NoWheelCombo(QtWidgets.QComboBox):
        def wheelEvent(self, e):                                  # the page scrolls; the value does not
            e.ignore()

    class Row(QtWidgets.QFrame):
        """One line of the form; reports hover / focus to the window so the help pane follows."""

        def __init__(self, win, item):
            super().__init__()
            self.win, self.item = win, item
            self.setObjectName("row")
            self.setAttribute(Qt.WA_Hover, True)
            self.lay = QtWidgets.QHBoxLayout(self)
            self.lay.setContentsMargins(10, 2, 8, 2)
            self.lay.setSpacing(10)

        def enterEvent(self, e):
            self.win.show_help(self.item)
            super().enterEvent(e)

        def eventFilter(self, obj, ev):
            if ev.type() == QtCore.QEvent.FocusIn:
                self.win.show_help(self.item)
            return False

    class KnobRow(Row):
        def __init__(self, win, k):
            super().__init__(win, k)
            self.k = k
            self.mark = QtWidgets.QLabel()
            self.mark.setFixedWidth(4)
            self.name = QtWidgets.QLabel(k.name)
            self.name.setFont(mono)
            self.name.setMinimumWidth(310)
            self.name.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.lay.addWidget(self.mark)
            self.lay.addWidget(self.name)
            if k.kind == "bool":
                self.ed = QtWidgets.QCheckBox()
                self.ed.toggled.connect(lambda on: self.commit("1" if on else "0"))
            elif k.kind == "enum":
                self.ed = NoWheelCombo()
                self.ed.setFocusPolicy(Qt.StrongFocus)
                self.ed.activated.connect(lambda _i: self.commit(self.ed.currentData()))
            else:
                self.ed = QtWidgets.QLineEdit()
                self.ed.setFont(mono)
                if k.kind == "number":
                    self.ed.setAlignment(Qt.AlignRight)
                self.ed.editingFinished.connect(lambda: self.commit(self.ed.text()))
            self.ed.setFixedWidth(64 if k.kind == "bool" else 190)
            self.ed.installEventFilter(self)
            self.lay.addWidget(self.ed)
            self.hint = QtWidgets.QLabel()
            self.hint.setObjectName("hint")
            self.lay.addWidget(self.hint, 1)
            self.undo = QtWidgets.QToolButton()
            self.undo.setText("default")
            self.undo.setToolTip("Set the documented default")
            self.undo.setAutoRaise(True)
            self.undo.clicked.connect(lambda: self.commit(str(k.default)))
            self.lay.addWidget(self.undo)
            self.back = QtWidgets.QToolButton()
            self.back.setText("revert")
            self.back.setToolTip("Back to the value in the file as opened")
            self.back.setAutoRaise(True)
            self.back.clicked.connect(self.revert)
            self.lay.addWidget(self.back)
            self.refresh()

        def commit(self, value):
            if self.win.doc.set(self.k.name, value):
                classify_keep_kind(self.k)
                self.win.changed(self.k)

        def revert(self):
            self.win.doc.reset(self.k.name)
            self.win.changed(self.k)

        def refresh(self):
            k, doc = self.k, self.win.doc
            b = QtCore.QSignalBlocker(self.ed)
            if k.kind == "bool":
                on = not k.is_off()
                self.ed.setChecked(on)
                self.ed.setText("on" if on else "off")
            elif k.kind == "enum":
                self.ed.clear()
                opts = list(k.choices)
                if k.value not in opts:
                    opts.append(k.value)
                for o in opts:
                    self.ed.addItem(o if o != "" else '""  (unset)', o)
                self.ed.setCurrentIndex(opts.index(k.value))
            else:
                self.ed.setText(k.value)
                self.ed.setPlaceholderText('""')
            del b
            gate = doc.gate_of(k)
            dim = k.is_off() or gate is not None
            bits = []
            if gate is not None:
                bits.append(f"inactive: {gate.name} is off")
            if k.out_of_range:
                bits.append(f"outside {k.lo:g}-{k.hi:g}")
            if k.off_default:
                bits.append(f"default {k.default if k.default != '' else chr(34) * 2}")
            elif k.range and k.kind == "number" and not bits:
                bits.append(k.range)
            if k.changed:
                bits.append(f"was {k.orig_value if k.orig_value != '' else chr(34) * 2}")
            self.hint.setText("   ·   ".join(bits))
            self.undo.setVisible(k.off_default)
            self.back.setVisible(k.changed)
            for w, key, val in ((self, "dim", dim), (self, "changed", k.changed), (self.ed, "warn", k.out_of_range),
                                (self.name, "dim", dim), (self.ed, "dim", dim), (self.mark, "changed", k.changed)):
                if w.property(key) != val:
                    w.setProperty(key, val)
                    w.style().unpolish(w)
                    w.style().polish(w)

        def matches(self, needle):
            return needle in self.k.name.lower() or needle in self.k.help.lower() or needle in self.k.value.lower()

    def classify_keep_kind(k):
        """Re-read range/default after an edit, but keep the widget's kind (a switch stays a switch)."""
        kind, choices = k.kind, k.choices
        classify(k)
        k.kind, k.choices = kind, choices

    class StepRow(Row):
        def __init__(self, win, st):
            super().__init__(win, st)
            self.st = st
            self.mark = QtWidgets.QLabel()
            self.mark.setFixedWidth(4)
            self.lay.addWidget(self.mark)
            self.cb = QtWidgets.QCheckBox()
            self.cb.toggled.connect(self.toggle)
            self.cb.installEventFilter(self)
            self.lay.addWidget(self.cb)
            f = st.fields
            self.name = QtWidgets.QLabel(f.get("name", ""))
            self.name.setFont(mono)
            self.name.setMinimumWidth(130)
            self.stage = QtWidgets.QLabel(f.get("stage", st.body))
            self.stage.setFont(mono)
            self.stage.setMinimumWidth(190)
            flow = "  →  ".join(x for x in (f.get("in", ""), f.get("out", "")) if x) if ("in" in f or "out" in f) else ""
            if "in" in f and "out" not in f:
                flow = f["in"] + "  →"
            elif "out" in f and "in" not in f:
                flow = "→  " + f["out"]
            self.flow = QtWidgets.QLabel(flow)
            self.flow.setObjectName("hint")
            for w in (self.name, self.stage):
                self.lay.addWidget(w)
            self.lay.addWidget(self.flow, 1)
            self.refresh()

        def toggle(self, on):
            if self.win.doc.set_step(self.st, on):
                self.win.changed(self.st)

        def refresh(self):
            b = QtCore.QSignalBlocker(self.cb)
            self.cb.setChecked(self.st.enabled)
            del b
            for w, key, val in ((self.name, "dim", not self.st.enabled), (self.stage, "dim", not self.st.enabled),
                                (self.mark, "changed", self.st.changed)):
                if w.property(key) != val:
                    w.setProperty(key, val)
                    w.style().unpolish(w)
                    w.style().polish(w)

        def matches(self, needle):
            return needle in self.st.body.lower()

    class NoteRow(Row):
        def __init__(self, win, note):
            super().__init__(win, note)
            first = next((x for x in note.text.split("\n") if x.strip()), "")
            lab = QtWidgets.QLabel(first.strip() if note.sub else "¶  " + first.strip()[:110])
            lab.setObjectName("sub" if note.sub else "note")
            lab.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)   # never widens the page
            self.lay.addWidget(lab, 1)

        def refresh(self):
            pass

        def matches(self, needle):
            return needle in self.item.text.lower()

    class SectionBox(QtWidgets.QWidget):
        def __init__(self, win, sec):
            super().__init__()
            self.win, self.sec = win, sec
            v = QtWidgets.QVBoxLayout(self)
            v.setContentsMargins(0, 0, 0, 6)
            v.setSpacing(0)
            self.head = QtWidgets.QToolButton()
            self.head.setObjectName("sechead")
            self.head.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            self.head.setCheckable(True)
            self.head.setArrowType(Qt.RightArrow)
            self.head.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            self.head.toggled.connect(self.set_open)
            self.head.setAttribute(Qt.WA_Hover, True)
            self.head.installEventFilter(self)
            v.addWidget(self.head)
            self.body = QtWidgets.QWidget()
            self.bl = QtWidgets.QVBoxLayout(self.body)
            self.bl.setContentsMargins(14, 2, 0, 6)
            self.bl.setSpacing(0)
            self.rows = []
            for it in sec.items:
                row = KnobRow(win, it) if isinstance(it, Knob) else StepRow(win, it) if isinstance(it, Step) \
                    else NoteRow(win, it)
                self.rows.append(row)
                self.bl.addWidget(row)
            if not sec.items:
                lab = QtWidgets.QLabel(sec.header)
                lab.setFont(mono)
                lab.setWordWrap(True)
                lab.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
                lab.setObjectName("note")
                lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
                self.bl.addWidget(lab)
            self.body.setVisible(False)
            v.addWidget(self.body)
            self.head.setProperty("notes", not (sec.knobs or sec.steps))
            self.retitle()

        def eventFilter(self, obj, ev):
            if ev.type() in (QtCore.QEvent.HoverEnter, QtCore.QEvent.FocusIn):
                self.win.show_help(self.sec)
            return False

        def set_open(self, on):
            self.head.setArrowType(Qt.DownArrow if on else Qt.RightArrow)
            self.body.setVisible(on)

        def retitle(self):
            s = self.sec
            bits = []
            if s.knobs:
                nch = sum(k.changed for k in s.knobs)
                noff = sum(self.win.doc.is_dimmed(k) for k in s.knobs)
                bits.append(f"{len(s.knobs)} knobs" + (f", {noff} off" if noff else "") + (f", {nch} edited" if nch else ""))
            if s.steps:
                nch = sum(st.changed for st in s.steps)
                bits.append(f"{sum(st.enabled for st in s.steps)} of {len(s.steps)} steps on" + (f", {nch} edited" if nch else ""))
            title = s.title if len(s.title) <= 96 else s.title[:95] + "…"
            self.head.setText(f"  {title}" + (f"      ({'; '.join(bits)})" if bits else ""))

        def apply_filter(self, needle, changed_only, hide_off):
            shown = 0
            active = bool(needle) or changed_only or hide_off
            for r in self.rows:
                it = r.item
                ok = (not needle) or r.matches(needle)
                if changed_only:
                    ok = ok and not isinstance(it, Note) and it.changed
                if hide_off and isinstance(it, Knob):
                    ok = ok and not self.win.doc.is_dimmed(it)
                if hide_off and isinstance(it, Step):
                    ok = ok and it.enabled
                if active and isinstance(it, Note) and not (needle and r.matches(needle)):
                    ok = False
                r.setVisible(ok)
                shown += ok and not isinstance(it, Note)
            if active:
                self.setVisible(shown > 0)
                if shown and (needle or changed_only):
                    self.head.setChecked(True)
            else:
                self.setVisible(True)

    class Window(QtWidgets.QMainWindow):
        def __init__(self, path):
            super().__init__()
            self.path = path
            self.doc = ConfigDoc.load(path) if path and os.path.exists(path) else ConfigDoc("")
            self.resize(1500, 900)
            tb = self.addToolBar("main")
            tb.setMovable(False)
            self.act_open = tb.addAction("Open…", self.open_file)
            self.act_open.setShortcut(QtGui.QKeySequence.Open)
            self.act_save = tb.addAction("Save", self.save)
            self.act_save.setShortcut(QtGui.QKeySequence.Save)
            tb.addAction("Save as…", self.save_as)
            self.act_review = tb.addAction("Review changes…", self.review)
            self.act_revert = tb.addAction("Revert all", self.revert_all)
            tb.addSeparator()
            tb.addAction("Expand all", lambda: self.expand(True))
            tb.addAction("Collapse all", lambda: self.expand(False))
            tb.addSeparator()
            self.search = QtWidgets.QLineEdit()
            self.search.setPlaceholderText("Filter knobs by name, value or help text")
            self.search.setClearButtonEnabled(True)
            self.search.setMaximumWidth(380)
            self.search.textChanged.connect(self.refilter)
            tb.addWidget(self.search)
            self.only_changed = QtWidgets.QCheckBox("Edited only")
            self.only_changed.toggled.connect(self.refilter)
            tb.addWidget(self.only_changed)
            self.hide_off = QtWidgets.QCheckBox("Hide off")
            self.hide_off.toggled.connect(self.refilter)
            tb.addWidget(self.hide_off)
            QtGui.QShortcut(QtGui.QKeySequence.Find, self, activated=self.search.setFocus)

            split = QtWidgets.QSplitter()
            self.scroll = QtWidgets.QScrollArea()
            self.scroll.setWidgetResizable(True)
            self.scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
            self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            split.addWidget(self.scroll)
            self.help = QtWidgets.QTextBrowser()
            self.help.setObjectName("help")
            self.help.setOpenExternalLinks(False)
            self.help.setMinimumWidth(360)
            split.addWidget(self.help)
            split.setStretchFactor(0, 3)
            split.setStretchFactor(1, 2)
            split.setSizes([940, 560])
            self.setCentralWidget(split)
            self.status = QtWidgets.QLabel()
            self.statusBar().addPermanentWidget(self.status)
            self.apply_style()
            self.populate()

        def apply_style(self):
            pal = self.palette()
            dim = pal.color(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText).name()
            acc = pal.color(QtGui.QPalette.Highlight).name()
            mid = pal.color(QtGui.QPalette.Mid).name()
            alt = pal.color(QtGui.QPalette.AlternateBase).name()
            self.setStyleSheet(f"""
                QToolButton#sechead {{ border: none; border-bottom: 1px solid {mid}; padding: 7px 6px;
                                       font-weight: 600; text-align: left; }}
                QToolButton#sechead:hover {{ background: {alt}; }}
                QToolButton#sechead[notes="true"] {{ font-weight: 400; color: {dim}; }}
                QFrame#row:hover {{ background: {alt}; }}
                QLabel[dim="true"] {{ color: {dim}; }}
                QLineEdit[dim="true"], QComboBox[dim="true"], QCheckBox[dim="true"] {{ color: {dim}; }}
                QLineEdit[warn="true"] {{ border: 1px solid #c77d00; }}
                QLabel[changed="true"] {{ background: {acc}; }}
                QLabel#hint {{ color: {dim}; }}
                QLabel#note {{ color: {dim}; }}
                QLabel#sub {{ font-weight: 600; padding-top: 8px; }}
                QTextBrowser#help {{ border: none; border-left: 1px solid {mid}; padding: 10px; }}
            """)

        def populate(self):
            host = QtWidgets.QWidget()
            v = QtWidgets.QVBoxLayout(host)
            v.setContentsMargins(8, 6, 8, 12)
            v.setSpacing(0)
            self.boxes = []
            self.rows = {}
            for s in self.doc.sections:
                box = SectionBox(self, s)
                self.boxes.append(box)
                for r in box.rows:
                    self.rows[id(r.item)] = r
                v.addWidget(box)
            v.addStretch(1)
            self.scroll.setWidget(host)
            for box in self.boxes:                                  # open the plan; the rest start folded
                if box.sec.steps:
                    box.head.setChecked(True)
            self.help.setHtml("<p style='color:gray'>Point at a knob, a section heading or a ¶ note to read its "
                              "explanation here.</p>")
            self.retitle()

        # ── help pane ────────────────────────────────────────────────────────
        def show_help(self, item):
            esc = lambda s: (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))   # noqa: E731
            raw = lambda s: f"<pre style='white-space:pre-wrap; font-family:{mono.family()}'>{esc(s)}</pre>"   # noqa: E731
            pre = lambda s: "".join(raw(t) if kind == "pre" else f"<p>{esc(t)}</p>" for kind, t in reflow(s))   # noqa: E731
            q = lambda v: esc(v) if v != "" else '""'                                           # noqa: E731
            if isinstance(item, Knob):
                k = item
                rows = [("value", f"<b>{q(k.value)}</b>" + (f" &nbsp;(file: {q(k.orig_value)})" if k.changed else ""))]
                if k.default is not None:
                    rows.append(("default", q(str(k.default))))
                if k.kind == "enum":
                    rows.append(("choices", " · ".join(q(c) for c in k.choices)))
                elif k.range:
                    rows.append(("range", esc(k.range)))
                g = self.doc.gate_of(k)
                if g is not None:
                    rows.append(("inactive", f"{g.name} is off"))
                elif k.is_off():
                    rows.append(("state", "off"))
                if k.out_of_range:
                    rows.append(("warning", "outside the documented range"))
                tab = "".join(f"<tr><td style='color:gray; padding-right:14px'>{a}</td><td>{b}</td></tr>" for a, b in rows)
                html = (f"<h3 style='font-family:{mono.family()}'>{k.name}</h3><table>{tab}</table>{pre(k.help)}"
                        f"<p style='color:gray'>{esc(k.section.title)}</p>")
            elif isinstance(item, Step):
                st = item
                html = (f"<h3>plan step {esc(st.fields.get('name', ''))}</h3>"
                        f"<p>{'on' if st.enabled else 'off (commented out)'}</p>{raw(st.body)}"
                        + (pre(_decomment(st.tail.strip())) if st.tail.strip() else "")
                        + "<p style='color:gray'>The tick comments the step in or out.  To add, reorder or rewire "
                          "steps use fiber-plan-edit.</p>")
            elif isinstance(item, Note):
                html = pre(item.text)
            else:
                html = f"<h3>{esc(item.title)}</h3>{pre(item.header)}"
            self.help.setHtml(html)

        # ── state ────────────────────────────────────────────────────────────
        def changed(self, item):
            for r in self.rows.values():                             # gating can reach other rows
                if isinstance(r.item, Knob) and (r.item is item or r.item.name.startswith(getattr(item, "name", "\0") + "_")):
                    r.refresh()
            if isinstance(item, Step):
                self.rows[id(item)].refresh()
            for b in self.boxes:
                b.retitle()
            self.show_help(item)
            self.retitle()
            if self.only_changed.isChecked() or self.hide_off.isChecked():
                self.refilter()

        def retitle(self):
            n = len(self.doc.changes())
            self.setWindowTitle(f"{'● ' if n else ''}{os.path.basename(self.path) or 'fiber-kit.yaml'} — fiber-config-edit")
            self.status.setText(f"{self.path}    {len(self.doc.knobs)} knobs    "
                                + (f"{n} unsaved change{'s' if n != 1 else ''}" if n else "no unsaved changes"))
            for a in (self.act_save, self.act_review, self.act_revert):
                a.setEnabled(bool(n))

        def refilter(self, *_):
            needle = self.search.text().strip().lower()
            for b in self.boxes:
                b.apply_filter(needle, self.only_changed.isChecked(), self.hide_off.isChecked())

        def expand(self, on):
            for b in self.boxes:
                b.head.setChecked(on)

        def revert_all(self):
            for name, *_ in [c for c in self.doc.changes() if c[0] in self.doc.knobs]:
                self.doc.reset(name)
            for s in self.doc.sections:
                for st in s.steps:
                    self.doc.set_step(st, st.orig_enabled)
            for r in self.rows.values():
                r.refresh()
            for b in self.boxes:
                b.retitle()
            self.retitle()
            self.refilter()

        def review(self):
            d = QtWidgets.QDialog(self)
            d.setWindowTitle("Changes not yet saved")
            d.resize(980, 520)
            lay = QtWidgets.QVBoxLayout(d)
            lines = [f"{n}:  {o if o != '' else chr(34) * 2}  →  {v if v != '' else chr(34) * 2}" for n, o, v in self.doc.changes()]
            te = QtWidgets.QPlainTextEdit("\n".join(lines) + "\n\n" + self.doc.diff(os.path.basename(self.path)))
            te.setFont(mono)
            te.setReadOnly(True)
            te.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
            lay.addWidget(te)
            bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
            bb.rejected.connect(d.reject)
            lay.addWidget(bb)
            d.exec()

        def save(self):
            return self._write(self.path)

        def save_as(self):
            p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save config as", self.path, "YAML (*.yaml *.yml)")
            return self._write(p) if p else False

        def _write(self, p):
            try:
                self.doc.save(p)
            except (OSError, ValueError) as e:
                QtWidgets.QMessageBox.critical(self, "Not saved", f"{p}\n\n{e}")
                return False
            self.path = p
            for r in self.rows.values():
                r.refresh()
            for b in self.boxes:
                b.retitle()
            self.retitle()
            self.statusBar().showMessage(f"saved {p}", 4000)
            return True

        def confirm_discard(self):
            if not self.doc.changes():
                return True
            r = QtWidgets.QMessageBox.question(
                self, "Unsaved changes", f"{len(self.doc.changes())} change(s) are not saved.",
                QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel)
            if r == QtWidgets.QMessageBox.Save:
                return self.save()
            return r == QtWidgets.QMessageBox.Discard

        def open_file(self):
            if not self.confirm_discard():
                return
            p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open config", os.path.dirname(self.path) or ".",
                                                         "YAML (*.yaml *.yml);;All files (*)")
            if p:
                self.path = p
                self.doc = ConfigDoc.load(p)
                self.populate()
                self.refilter()

        def closeEvent(self, e):
            e.accept() if self.confirm_discard() else e.ignore()

    return Window(path)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("-h", "--help"):
        print("usage: fiber-config-edit [fiber-kit.yaml]      open the form editor ($FK_CONFIG, else ./fiber-kit.yaml)\n"
              "       fiber-config-edit --check FILE          print what the editor reads from FILE (no Qt)")
        return 0
    if argv and argv[0] == "--check":
        if len(argv) < 2:
            raise SystemExit("fiber-config-edit --check needs a file")
        return check_main(argv[1])
    path = argv[0] if argv else default_path()
    if not os.path.exists(path):
        raise SystemExit(f"fiber-config-edit: {path} not found (pass a config file, or set FK_CONFIG; "
                         "fiber-kit-init writes one)")
    try:
        from PySide6 import QtWidgets
    except ImportError:
        raise SystemExit("fiber-config-edit needs PySide6 >= 6.8  (pip install 'fiber-kit[editor]')") from None
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    w = _build_window(path)
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
