#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════════════════════
#  morpho_chan_ca1.py — per-cell-type CA1 channel kinetics and biophysics.
#
#  Transcribed from the mod files and cell templates of the Bezaire, Raikov,
#  Burk, Vyas & Soltesz (2016) full-scale CA1 model (github.com/mbezaire/ca1),
#  the same source this repo already uses for afferent topology.
#
#  WHY A SECOND CHANNEL MODULE.  morpho_chan carries the Migliore CA1 PYRAMIDAL
#  kinetics and applies them to every morphology.  That is fine for asking how
#  geometry alone reshapes a waveform, and wrong for asking how much SHAPE can
#  vary within and between real hippocampal cell types -- because a PV basket
#  cell is not a pyramidal cell with different dendrites.  It has ~5x the
#  somatic sodium density, a four-gate fast delayed rectifier instead of a
#  first-order one, and almost no A-current.  Those differences are the reason
#  its spike is narrow, and they cannot be reached by re-parameterizing a
#  pyramidal model.
#
#  Usefully, the model's own pyramidal cell (`poolosyncell`) uses ch_Navp /
#  ch_Kdrp / ch_KvAproxp / ch_KvAdistp, which ARE the Migliore na3 / kdrca1 /
#  kap / kad already in morpho_chan.  So the two modules agree where they
#  overlap rather than offering two rival pyramidal cells, and CA1_TYPES routes
#  the pyramidal types to morpho_chan deliberately.
#
#  ═══ SCOPE, stated up front because it bounds every number this produces ═══
#  Transcribed here: the Nav family (Nav, Navbis, Navcck, Navngf), the fast
#  delayed rectifiers (Kdrfast, Kdrfastngf) and the Boltzmann A-types (KvA,
#  KvAngf).  NOT transcribed: CavL, CavN, KCaS, KvCaB, iconc_Ca, HCN, KvM,
#  KvGroup, KvAolm.
#
#  The omission is deliberate and it is defensible for THIS measurement but not
#  for every measurement.  Calcium and the calcium-dependent potassium channels
#  act on the afterhyperpolarization over tens of milliseconds; the extracted
#  window is 42 samples at 32552 Hz = 1.29 ms, and the band-pass starts at
#  300 Hz, so almost none of that is inside the recorded footprint.  The spike
#  itself -- which is what the footprint is -- is set by Nav, Kdr and KvA.
#
#  What this omission DOES cost: calcium accumulates across a burst, so
#  Ca-dependent conductances are a genuine source of firing-state-dependent
#  SHAPE variance, and that component is missing from any envelope built here.
#  An envelope from this module is therefore a LOWER bound on within-cell shape
#  variance.  For the intra-chunk over-merge problem a lower bound is the
#  conservative direction (it makes the gate stricter, so it never licenses a
#  fusion it should not); for cross-chunk linking it is the risky direction.
#  Do not reuse this envelope across that boundary without saying so.
#
#  Units follow NEURON: mV, ms, S/cm^2, degC.
# ════════════════════════════════════════════════════════════════════════════
import numpy as np

try:
    from . import morpho_chan as mch
except ImportError:
    import morpho_chan as mch


def _vtrap(x, y):
    """x/(exp(x/y)-1) with the removable singularity handled, as in the mod files."""
    x = np.asarray(x, float)
    small = np.abs(x / y) < 1e-6
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        out = x / (np.exp(x / y) - 1.0)
    return np.where(small, y * (1.0 - x / y / 2.0), out)


class NavBez:
    """ch_Nav family — fast sodium, m^3 h, with a per-variant rate parameterization.

    All four variants share one functional form and differ only in the eight
    constants below, which is why they are one class rather than four: a
    separate class per variant would be four copies of the same policy, and the
    thing that would go stale is the shared form, not the constants.

        alpha_m = mAlphC * vtrap(v + mAlphV, -5)
        beta_m  = mBetaC * vtrap(v + mBetaV,  5)
        alpha_h = hAlphC / exp((v + hAlphV) / 20)
        beta_h  = hBetaC / (1 + exp(-(v + hBetaV) / 10))

    q10 = 3^((celsius-34)/10) divides the time constants (the mod files apply it
    through the exponential-Euler increment, which is the same thing).
    """
    name = "nav"
    VARIANTS = {
        # variant        mAlphC mAlphV mBetaC mBetaV hAlphC  hAlphV  hBetaC hBetaV
        "nav":       (-0.30000, 43.0, 0.30000, 15.0, 0.23000, 65.0000, 3.3300, 12.5000),
        "navbis":    (-0.20000, 38.0, 0.50000, 10.0, 0.23000, 62.0000, 2.0000, 9.5000),
        "navcck":    (-0.50000, 42.0, 0.30000, 13.0, 0.60000, 65.0000, 1.3000, 12.5000),
        "navngf":    (-0.34133, 24.0, 0.28483, -4.0, 0.29648, 64.4184, 3.0931, 12.1463),
    }

    def __init__(self, variant="nav", celsius=34.0):
        if variant not in self.VARIANTS:
            raise ValueError(f"Nav variant must be one of {sorted(self.VARIANTS)}, got {variant!r}")
        (self.mAlphC, self.mAlphV, self.mBetaC, self.mBetaV,
         self.hAlphC, self.hAlphV, self.hBetaC, self.hBetaV) = self.VARIANTS[variant]
        self.variant = variant; self.name = variant
        self.qt = 3.0 ** ((celsius - 34.0) / 10.0)

    def rates(self, v):
        a = self.mAlphC * _vtrap(v + self.mAlphV, -5.0)
        b = self.mBetaC * _vtrap(v + self.mBetaV, 5.0)
        sm = a + b
        minf = a / sm; mtau = 1.0 / sm / self.qt
        with np.errstate(over="ignore"):
            a = self.hAlphC / np.exp((v + self.hAlphV) / 20.0)
            b = self.hBetaC / (1.0 + np.exp(-(v + self.hBetaV) / 10.0))
        sh = a + b
        hinf = a / sh; htau = 1.0 / sh / self.qt
        return (minf, np.maximum(mtau, 1e-3)), (hinf, np.maximum(htau, 1e-3))

    def g(self, st):
        m, h = st
        return m * m * m * h


class KdrFast:
    """ch_Kdrfast / ch_Kdrfastngf — fast delayed rectifier, n^4.

    Fourth order, unlike the pyramidal kdrca1's first order.  That is the
    single largest reason an interneuron spike repolarizes faster than a
    pyramidal one at comparable conductance, and therefore the largest reason
    the two occupy different regions of trough-to-peak width.

    'ngf' shifts both half-activations by +10 mV (offset5 / offset6 in the mod
    file), which is applied where the mod file applies it: inside vtrap's
    argument and inside the beta exponent, not to v.
    """
    name = "kdrfast"

    def __init__(self, variant="kdrfast", celsius=34.0):
        self.off = 10.0 if variant.endswith("ngf") else 0.0
        self.slope5, self.slope6 = 0.07, 0.264
        self.qt = 3.0 ** ((celsius - 34.0) / 10.0)
        self.name = variant

    def rates(self, v):
        a = -self.slope5 * _vtrap(v + 65.0 - 47.0 - self.off, -6.0)
        with np.errstate(over="ignore"):
            b = self.slope6 / np.exp((v + 65.0 - 22.0 - self.off) / 40.0)
        s = a + b
        return ((a / s, np.maximum(1.0 / s / self.qt, 1e-3)),)

    def g(self, st):
        return st[0] ** 4


class KvABez:
    """ch_KvA / ch_KvAngf — A-type, n*l, Boltzmann-rate form.

    Same algebraic family as the pyramidal kap/kad in morpho_chan but with the
    interneuron half-activations (-33.6 mV, or -23.6 for the neurogliaform
    variant) and its own zeta/gm set.
    """
    name = "kva"
    vhalfl, a0l, a0n, zetan, zetal, gmn, gml = -83.0, 0.08, 0.02, -3.0, 4.0, 0.6, 1.0

    def __init__(self, variant="kva", celsius=34.0):
        self.vhalfn = -23.6 if variant.endswith("ngf") else -33.6
        self.zf = mch._zfac(celsius)
        self.qt = 3.0 ** ((celsius - 30.0) / 10.0)
        self.name = variant

    def rates(self, v):
        with np.errstate(over="ignore"):
            an = np.exp(self.zetan * self.zf * (v - self.vhalfn))
            bn = np.exp(self.zetan * self.gmn * self.zf * (v - self.vhalfn))
            al = np.exp(self.zetal * self.zf * (v - self.vhalfl))
            bl = np.exp(self.zetal * self.gml * self.zf * (v - self.vhalfl))
        ninf = 1.0 / (1.0 + an)
        taun = np.maximum(bn / (self.qt * self.a0n * (1.0 + an)), 0.1)
        linf = 1.0 / (1.0 + al)
        taul = np.maximum(bl / (self.qt * self.a0l * (1.0 + al)), 2.0)
        return (ninf, taun), (linf, taul)

    def g(self, st):
        n, l = st
        return n * l


# ── per-cell-type presets ───────────────────────────────────────────────────
# Passive constants and conductance densities read from each template's
# mechinit().  `nav`/`kdr`/`kva` name the variant; `family` selects which
# kinetics module builds the channel list ("bezaire" here, "migliore" routes to
# morpho_chan, which is what the model's own pyramidal cells use).
CA1_TYPES = {
    "pyramidal": dict(family="migliore", Rm=28000.0, cm=1.0, Ra=150.0, Ra_axon=50.0,
                      v_rest=-65.0, celsius=34.0, gna=0.032, axon_na_mult=2.0,
                      gkdr=0.003, gka=0.008),
    "pvbasket": dict(family="bezaire", hcn=0.00015, kv3=0.05, nav="nav", kdr="kdrfast", kva="kva",
                     Rm=5555.0, cm=1.4, Ra=100.0, v_rest=-65.0, celsius=34.0,
                     gna=0.15, gkdr=0.013, gka=0.00015),
    "axoaxonic": dict(family="bezaire", hcn=0.00015, kv3=0.05, nav="nav", kdr="kdrfast", kva="kva",
                      Rm=5555.0, cm=1.4, Ra=100.0, v_rest=-65.0, celsius=34.0,
                      gna=0.15, gkdr=0.013, gka=0.00015),
    "bistratified": dict(family="bezaire", hcn=0.0002, kv3=0.01, nav="navbis", kdr="kdrfast", kva="kva",
                         Rm=11110.0, cm=1.4, Ra=100.0, v_rest=-67.0, celsius=34.0,
                         gna=0.07, gkdr=0.016, gka=0.00005),
    "cck": dict(family="bezaire", nav="navcck", kdr="kdrfast", kva="kva",
                Rm=27000.0, cm=1.0, Ra=150.0, v_rest=-61.0, celsius=34.0,
                gna=0.009, gkdr=0.003, gka=0.0010),
    "ngf": dict(family="bezaire", nav="navngf", kdr="kdrfastngf", kva="kvangf",
                Rm=5555.0, cm=1.4, Ra=100.0, v_rest=-65.0, celsius=34.0,
                gna=0.15, gkdr=0.013, gka=5.2203905e-06),
    "ivy": dict(family="bezaire", nav="navngf", kdr="kdrfastngf", kva="kvangf",
                Rm=5555.0, cm=1.4, Ra=100.0, v_rest=-65.0, celsius=34.0,
                gna=0.15, gkdr=0.013, gka=5.2203905e-06),
    "sca": dict(family="bezaire", nav="navcck", kdr="kdrfast", kva="kva",
                Rm=45000.0, cm=1.0, Ra=150.0, v_rest=-57.0, celsius=34.0,
                gna=0.004, gkdr=7.2e-06, gka=0.00010),
}

# Types whose defining conductances are NOT transcribed here.  Listed rather
# than silently approximated: OLM's identity rests on ch_HCNolm and ch_KvAolm,
# and CCK/SCA on ch_KvM + ch_KvGroup, none of which exist in this module.  They
# are still runnable (the entries above give their Nav/Kdr/KvA), but their spike
# shape is an approximation and must not be reported as that cell type's.
INCOMPLETE = {"cck": ("KvM", "KvGroup", "Ca", "KCa"),
              "sca": ("KvM", "KvGroup", "HCN", "Ca", "KCa"),
              "ngf": ("Ca", "KCa"), "ivy": ("Ca", "KCa"),
              "pvbasket": ("Ca", "KCa"), "axoaxonic": ("Ca", "KCa"),
              "bistratified": ("Ca", "KCa")}


def channels(kind, celsius=None):
    """[(channel, density_key, erev_key), ...] for a CA1 cell type."""
    if kind not in CA1_TYPES:
        raise ValueError(f"unknown CA1 type {kind!r}; have {sorted(CA1_TYPES)}")
    p = CA1_TYPES[kind]
    cel = float(celsius if celsius is not None else p["celsius"])
    if p["family"] == "migliore":
        return [(mch.Na(cel), "na", "ena"), (mch.Kdr(cel), "kdr", "ek"),
                (mch.KA("prox", cel), "ka_prox", "ek"), (mch.KA("dist", cel), "ka_dist", "ek")]
    out = [(NavBez(p["nav"], cel), "na", "ena"), (KdrFast(p["kdr"], cel), "kdr", "ek"),
           (KvABez(p["kva"], cel), "ka_prox", "ek"), (KvABez(p["kva"], cel), "ka_dist", "ek")]
    if p.get("kv3", 0.0):
        out.append((Kv3(cel), "kv3", "ek"))
    if p.get("hcn", 0.0):
        out.append((HCN(cel), "hcn", "eh"))
    if p.get("narsg", 0.0):
        out.append((NaRsg(cel), "narsg", "ena"))
    return out


def biophys(kind, **over):
    """A morpho_cable.Biophys preset for a CA1 cell type."""
    try:
        from . import morpho_cable as mc
    except ImportError:
        import morpho_cable as mc
    p = dict(CA1_TYPES[kind]); p.pop("family", None)
    for k in ("nav", "kdr", "kva"):
        p.pop(k, None)
    if "kv3" in p:
        p["gkv3"] = p.pop("kv3")
    if "hcn" in p:
        p["ghcn"] = p.pop("hcn")
    if "narsg" in p:
        p["gnarsg"] = p.pop("narsg")
    p.setdefault("Ra_axon", p.get("Ra", 150.0))
    p.update(over)
    b = mc.Biophys(**p)
    b.ca1_type = kind
    return b


class Kv3:
    """Kv3 — the fast-spiking potassium channel, n^4, high threshold.

    Transcribed from Akemann et al. (2009) as implemented by Zang & De Schutter
    (2021), whose rate constants are least-squares fits to the interneuron K
    current data of Martina et al. (2007, J Neurophysiol 97:563).

        alpha = 0.22 * exp(-(v + 16) / -26.5)      beta = 0.22 * exp(-(v + 16) / 26.5)
        g = gbar * n^4                              q10 = 2.7 from 22 degC

    WHY IT HAD TO BE ADDED, rather than tuning what was already here.  The
    Bezaire-derived Kdrfast is the only fast rectifier in this module, and
    sweeping its density over 7.7x (0.013 to 0.100 S/cm^2) on a reconstructed
    basket cell moved the extracellular trough-to-peak only 0.614 -> 0.461 ms
    and then saturated.  The real g5 interneuron measures 0.271 ms.  Conductance
    density cannot buy a time constant: Kdrfast's activation midpoint and tau
    are fixed, whereas Kv3 activates near +16 mV with tau under a millisecond at
    spike potentials, which is what terminates a fast-spiking action potential.

    Its absence is why patch 0330 produced basket cells BROADER than pyramidal
    cells -- a result flagged there as backwards and unexplained.  The model had
    no mechanism for fast spiking at all.
    """
    name = "kv3"
    ca, cva, cka = 0.22, 16.0, -26.5
    cb, cvb, ckb = 0.22, 16.0, 26.5
    q10 = 2.7

    def __init__(self, celsius=34.0):
        self.qt = self.q10 ** ((celsius - 22.0) / 10.0)

    def rates(self, v):
        with np.errstate(over="ignore"):
            a = self.qt * self.ca * np.exp(-(v + self.cva) / self.cka)
            b = self.qt * self.cb * np.exp(-(v + self.cvb) / self.ckb)
        s = a + b
        return ((a / s, np.maximum(1.0 / s, 1e-3)),)

    def g(self, st):
        return st[0] ** 4


class HCN:
    """ch_HCN — the interneuron hyperpolarization-activated current, g = gmax*h^2.

    Transcribed from the Bezaire et al. (2016) mod file that the basket,
    axo-axonic and bistratified templates insert.

        hinf = 1 / (1 + exp((v + 91) / 10))
        tau  = (120 + 129.5 / (1 + exp((v + 59.3) / 0.83))) / q10,  q10 = 3^((T-34)/10)

    WHY IT IS WORTH HAVING: the time constant runs 120 ms at -50 mV to 250 ms at
    -70 mV.  That is the best match in this module to the measured adaptation
    profile, whose marginal R^2 peaks at tau = 50-200 ms and which also has an
    independent slow (0.5-10 s) component -- one channel spanning both, rather
    than two mechanisms.  Second-order gating makes the effective onset slower
    still.  HCN was listed in INCOMPLETE for the fast-spiking types and is now
    filled.

    WHAT IT PROBABLY DOES NOT EXPLAIN, stated so it is not oversold: the
    measured +11% spike-amplitude INCREASE at short ISI.  With a half-activation
    of -91 mV and a fast-spiking duty cycle near 2%, a train leaves the cell
    hyperpolarized far more than depolarized, so Ih should ACTIVATE, depolarize
    between spikes, and reduce Na availability -- predicting smaller spikes, the
    same wrong sign as Kv3 and Na inactivation.  Ih is also cAMP-modulated and so
    ought to be brain-state dependent, whereas the measured state component is
    invariant across theta/non-theta and across 162 minutes.  Both are arguments
    against Ih being the dominant term, and neither is settled here.
    """
    name = "hcn"
    vhalf, slope = -91.0, 10.0
    tau_a, tau_b, tau_vh, tau_k = 120.0, 129.5, -59.3, 0.83
    q10 = 3.0

    def __init__(self, celsius=34.0):
        self.qt = self.q10 ** ((celsius - 34.0) / 10.0)

    def rates(self, v):
        with np.errstate(over="ignore"):
            hinf = 1.0 / (1.0 + np.exp((v - self.vhalf) / self.slope))
            tau = (self.tau_a + self.tau_b /
                   (1.0 + np.exp((v - self.tau_vh) / self.tau_k))) / self.qt
        return ((hinf, np.maximum(tau, 1e-3)),)

    def g(self, st):
        return st[0] ** 2


class NaRsg:
    """naRsg — RESURGENT sodium, the 13-state Raman & Bean scheme.

    Transcribed from narsg.mod (Khaliq, Gouwens & Raman 2003, as distributed with
    Zang & De Schutter 2021).  Unlike every other channel in this module it is
    not a product of independent gates, so it cannot be expressed as (inf, tau)
    pairs and needs the Markov path in morpho_cable.

        C1-C2-C3-C4-C5-O          activation chain, then opening
        O <-> B                   open-channel BLOCK -- the resurgent state
        O <-> I6, I1-...-I5-I6    inactivation, coupled to each closed state

    WHY IT IS HERE.  The measured cell shows spikes ~11% LARGER after short
    interspike intervals, decaying to baseline over ~200 ms.  Kv3, Na
    inactivation and Ih all predict the opposite sign: each of them depresses the
    spike during a train.  Resurgent sodium is the one candidate that facilitates
    -- during repolarization the blocked state B unbinds back THROUGH the open
    state rather than through inactivation, so a preceding spike leaves channels
    poised to reopen instead of inactivated.  That is the mechanism the data has
    been pointing at since the recovery curve of patch 0328 came back with the
    wrong sign.

    Rates are the published ones; alfac and btfac are the microscopic-
    reversibility factors (Oon/Con)^(1/4) and (Ooff/Coff)^(1/4), not free
    parameters.
    """
    name = "narsg"
    markov = True
    NSTATE = 13
    LABEL = ("C1", "C2", "C3", "C4", "C5", "O", "B", "I1", "I2", "I3", "I4", "I5", "I6")
    iO, iB = 5, 6
    Con, Coff, Oon, Ooff = 0.005, 0.5, 0.75, 0.005
    alpha, beta, gamma, delta = 150.0, 3.0, 150.0, 40.0
    epsilon, zeta = 1.75, 0.03
    x1, x2, x6 = 20.0, -20.0, -25.0
    q10 = 2.7

    def __init__(self, celsius=34.0):
        self.qt = self.q10 ** ((celsius - 22.0) / 10.0)
        self.alfac = (self.Oon / self.Con) ** 0.25
        self.btfac = (self.Ooff / self.Coff) ** 0.25

    def _rates(self, v):
        q, af, bf = self.qt, self.alfac, self.btfac
        with np.errstate(over="ignore"):
            ea = np.exp(v / self.x1) * q
            eb = np.exp(v / self.x2) * q
            f = [k * self.alpha * ea for k in (4, 3, 2, 1)]              # f01..f04
            b = [k * self.beta * eb for k in (1, 2, 3, 4)]               # b01..b04
            f1 = [k * self.alpha * af * ea for k in (4, 3, 2, 1)]        # f11..f14
            b1 = [k * self.beta * bf * eb for k in (1, 2, 3, 4)]         # b11..b14
            bip = self.zeta * np.exp(v / self.x6) * q
        fi = [self.Con * af ** k * q for k in range(5)]                  # fi1..fi5
        bi = [self.Coff * bf ** k * q for k in range(5)]                 # bi1..bi5
        return dict(f=f, b=b, f1=f1, b1=b1, fi=fi, bi=bi,
                    f0O=self.gamma * q, b0O=self.delta * q,
                    fip=self.epsilon * q, bip=bip,
                    fin=self.Oon * q, bin=self.Ooff * q,
                    f1n=self.gamma * q, b1n=self.delta * q)

    def generator(self, v):
        """(N, 13, 13) matrix A with dP/dt = A P, columns summing to zero."""
        v = np.atleast_1d(np.asarray(v, float))
        n = len(v)
        r = self._rates(v)
        A = np.zeros((n, 13, 13))

        def link(i, j, fwd, bwd):
            fwd = np.broadcast_to(np.asarray(fwd, float), (n,))
            bwd = np.broadcast_to(np.asarray(bwd, float), (n,))
            A[:, j, i] += fwd; A[:, i, i] -= fwd
            A[:, i, j] += bwd; A[:, j, j] -= bwd

        for k in range(4):                       # C1..C5 chain
            link(k, k + 1, r["f"][k], r["b"][k])
        link(4, self.iO, r["f0O"], r["b0O"])     # C5 <-> O
        link(self.iO, self.iB, r["fip"], r["bip"])   # O <-> B  (resurgent block)
        link(self.iO, 12, r["fin"], r["bin"])    # O <-> I6
        for k in range(4):                       # I1..I5 chain
            link(7 + k, 8 + k, r["f1"][k], r["b1"][k])
        link(11, 12, r["f1n"], r["b1n"])         # I5 <-> I6
        for k in range(5):                       # Ck <-> Ik
            link(k, 7 + k, r["fi"][k], r["bi"][k])
        return A

    def steady(self, v):
        """(N, 13) equilibrium occupancy — the null space with sum = 1."""
        A = self.generator(v).copy()
        A[:, -1, :] = 1.0                        # replace one row by conservation
        rhs = np.zeros((len(A), 13)); rhs[:, -1] = 1.0
        return np.linalg.solve(A, rhs[..., None])[..., 0]

    def step(self, P, v, dt):
        """One implicit step: (I - dt*A) P_new = P_old, batched over compartments.

        Implicit rather than explicit because the fastest rates here reach
        ~10^4 /ms at spike potentials, which an explicit step at dt = 0.02 ms
        would not merely blur but diverge.
        """
        A = self.generator(v)
        M = np.eye(13)[None] - dt * A
        Pn = np.linalg.solve(M, np.asarray(P, float)[..., None])[..., 0]
        np.clip(Pn, 0.0, 1.0, out=Pn)
        s = Pn.sum(1, keepdims=True)
        return Pn / np.maximum(s, 1e-30)

    def g(self, P):
        return np.asarray(P)[..., self.iO]
