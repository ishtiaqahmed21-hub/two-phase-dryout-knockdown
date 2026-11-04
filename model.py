"""
Reduced-order model of flow boiling in N parallel heated microchannels.

Physics (steady 1D per channel, common plenums, fixed total flow):
  - Single-phase liquid region up to x_e = 0 (subcooled boiling neglected).
  - Saturated two-phase region: Kim & Mudawar (2013) frictional pressure drop
    for saturated flow boiling (C-parameter from their 2012 adiabatic paper),
    Zivi void fraction for the acceleration term.
  - If a channel reaches x = 1, the rest of it carries vapor only.
  - Dryout incipience: Kim & Mudawar (2013) Part I correlation.
  - Optional inlet restrictor: orifice of fixed area (same for every channel).
  - Channel dynamics: I_i dm_i/dt = dP_plenum - dP_i(m_i), sum(m_i) = M.
    Ramping power and integrating to steady state picks the stable branch,
    which is how Ledinegg maldistribution shows up.

Simplifications (state them whenever results are shared):
  - Saturated properties at one fixed pressure, no local flashing.
  - No lateral (channel-to-channel) wall conduction: thermally isolated channels.
  - Same heat input per channel.
  - Correlation coefficients typed from the papers: check against the
    originals before publishing (VERIFY tags).
"""
import numpy as np
from CoolProp.CoolProp import PropsSI
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


class Water:
    """Saturated water properties at one pressure."""

    def __init__(self, P=101325.0):
        q = lambda k, Q: PropsSI(k, "P", P, "Q", Q, "Water")
        self.P = P
        self.Tsat = q("T", 0)
        self.rf, self.rg = q("D", 0), q("D", 1)
        self.mf, self.mg = q("V", 0), q("V", 1)
        self.sig = q("I", 0)
        self.hfg = q("H", 1) - q("H", 0)
        self.cpf = q("C", 0)
        self.kf, self.kg = q("L", 0), q("L", 1)
        self.Prf, self.Prg = q("Prandtl", 0), q("Prandtl", 1)
        self.M = 18.015                                  # kg/kmol
        self.PR = P / PropsSI("Pcrit", "Water")


def poiseuille(beta):
    """Fanning f*Re, laminar, rectangular duct (Shah & London)."""
    return 24.0 * (1 - 1.3553 * beta + 1.9467 * beta**2 - 1.7012 * beta**3
                   + 0.9564 * beta**4 - 0.2537 * beta**5)


def fanning(Re, Po):
    Re = np.maximum(Re, 1e-9)
    turb = np.where(Re < 2e4, 0.079 * Re**-0.25, 0.046 * Re**-0.2)
    return np.where(Re < 2000, Po / Re, turb)


class Channels:
    """N channels; width may differ channel to channel (the imperfection)."""

    def __init__(self, w, d, L, fl, dT_sub, restrictor_area=None, K_r=1.5):
        self.w = np.atleast_1d(np.asarray(w, float))
        self.N = self.w.size
        self.d, self.L, self.fl, self.dT_sub = d, L, fl, dT_sub
        self.A = self.w * d
        self.Dh = 2 * self.A / (self.w + d)
        beta = np.minimum(self.w, d) / np.maximum(self.w, d)
        self.Po = poiseuille(beta)
        self.PF = 2 * (self.w + d)          # wetted perimeter
        self.PH = self.w + 2 * d            # heated: base + 2 walls (fin eff. = 1)
        self.HF = self.PH / self.PF
        self.I = L / self.A                 # inertance
        self.Ar, self.Kr = restrictor_area, K_r
        self.L_uh = 0.0                     # unheated inlet length (liquid friction), m
        self.tp_scale = 1.0                 # multiplier on two-phase friction (sensitivity)
        self.sub_boil = False               # subcooled boiling (Saha-Zuber OSV + Levy profile)
        self.C_model = "kim"                # two-phase C: kim (2013) | mishima | qu (2003)
        self.C_lat = 0.0                    # lateral conductance to each neighbour, W/(m K)
        self.Cw = 8.2e-4                    # wall heat capacity per channel, J/K (Rahman & Weibel Table 1)

    def h_bertsch(self, m, Q, nx=24):
        """Length-averaged wall-to-fluid HTC (Bertsch et al. 2009 form, as used by
        Rahman & Weibel 2024): liquid-only region + flow-boiling region."""
        fl = self.fl
        G, Dh, L = m / self.A, self.Dh, self.L
        gz = lambda Re, Pr: Re * Pr * Dh / L
        hc = lambda Re, Pr, k: (3.66 + 0.0668 * gz(Re, Pr) / (1 + 0.04 * gz(Re, Pr)**(2 / 3))) * k / Dh
        hf = hc(G * Dh / fl.mf, fl.Prf, fl.kf)
        hg = hc(G * Dh / fl.mg, fl.Prg, fl.kg)
        qH = Q / (self.PH * L)
        hnb = 55 * fl.PR**0.12 * (-np.log10(fl.PR))**-0.55 * fl.M**-0.5 * qH**0.67
        Co = (9.81 * (fl.rf - fl.rg) * Dh**2 / fl.sig)**-0.5
        xo = np.clip(self.x_eq(m, Q), 0, 1)
        x = xo[:, None] * np.linspace(0, 1, nx)[None, :]
        c = lambda a: a[:, None]
        hfb = c(hnb) * (1 - x) + ((1 - x) * c(hf) + x * c(hg)) * (1 + 80 * (x**2 - x**6) * np.exp(-0.6 * c(Co)))
        hfb_avg = np.trapezoid(hfb, x, axis=1) / np.maximum(xo, 1e-9)
        hfb_avg = np.where(xo > 1e-6, hfb_avg, hf)
        f_sp = np.clip(m * fl.cpf * self.dT_sub / Q, 0, 1)          # liquid-only fraction
        return f_sp * hf + (1 - f_sp) * hfb_avg, f_sp

    def q_fluid(self, m, Q):
        """Heat reaching each channel's fluid when walls conduct to neighbours
        (quasi-steady wall; chain of channels). Equals Q when C_lat = 0."""
        Q = np.broadcast_to(np.asarray(Q, float), np.shape(m)).astype(float)
        if self.C_lat <= 0 or self.N < 2:
            return Q
        fl = self.fl
        h, f_sp = self.h_bertsch(m, Q)
        hA = h * self.PH * self.L
        Tsat = fl.Tsat
        Tf = f_sp * (Tsat - self.dT_sub / 2) + (1 - f_sp) * Tsat      # length-weighted
        K = self.C_lat * self.L
        A = np.diag(hA + K * (np.r_[0, np.ones(self.N - 1)] + np.r_[np.ones(self.N - 1), 0]))
        A -= K * (np.eye(self.N, k=1) + np.eye(self.N, k=-1))
        Tw = np.linalg.solve(A, Q + hA * Tf)
        return hA * (Tw - Tf)

    # ----- correlations -----------------------------------------------------
    def x_eq(self, m, Q):
        """Thermodynamic equilibrium quality at the exit."""
        fl = self.fl
        return (Q - m * fl.cpf * self.dT_sub) / (m * fl.hfg)

    def x_osv(self, m, Q):
        """Equilibrium quality at onset of significant void (Saha & Zuber 1974);
        0 when subcooled boiling is off. Never earlier than the inlet."""
        if not self.sub_boil:
            return np.zeros(np.shape(m))
        fl = self.fl
        G = m / self.A
        qH = Q / (self.PH * self.L)
        Pe = G * self.Dh * fl.cpf / fl.kf
        dT = np.where(Pe < 7e4, qH * self.Dh / (455 * fl.kf), qH / (0.0065 * G * fl.cpf))
        return -fl.cpf * np.minimum(dT, self.dT_sub) / fl.hfg

    @staticmethod
    def x_true(xe, xosv):
        """Flow quality from equilibrium quality (Levy-type profile fit)."""
        xe, xosv = np.broadcast_arrays(xe, xosv)
        safe = np.where(xosv < 0, xosv, -1.0)
        prof = xe - safe * np.exp(np.minimum(xe / safe - 1, 0))
        return np.where(xosv < 0, np.where(xe >= xosv, prof, 0.0), np.maximum(xe, 0.0))

    def x_out(self, m, Q):
        """Flow quality at the exit (equals x_eq when subcooled boiling is off)."""
        return self.x_true(self.x_eq(m, Q), self.x_osv(m, Q))

    def x_dryout(self, m, Q):
        """Kim & Mudawar 2013 Part I dryout incipience quality (checked vs paper eq. 5)."""
        fl = self.fl
        G = m / self.A
        Bo = Q / (self.PH * self.L) / (G * fl.hfg)
        We = G**2 * self.Dh / (fl.rf * fl.sig)
        Ca = fl.mf * G / (fl.rf * fl.sig)
        return (1.4 * We**0.03 * fl.PR**0.08
                - 15.0 * (Bo * self.HF)**0.15 * Ca**0.35 * (fl.rg / fl.rf)**0.06)

    @staticmethod
    def _C(G, x, Bo, Dh, HF, fl):
        """Kim & Mudawar 2013 C for saturated boiling (checked vs paper Table 5)."""
        Refo = G * Dh / fl.mf
        Ref = G * (1 - x) * Dh / fl.mf
        Reg = G * x * Dh / fl.mg
        Su = fl.rg * fl.sig * Dh / fl.mg**2
        dr = fl.rf / fl.rg
        lf, lg = Ref < 2000, Reg < 2000
        Cnb = np.select(
            [~lf & ~lg, ~lf & lg, lf & ~lg],
            [0.39 * Refo**0.03 * Su**0.10 * dr**0.35,     # tt
             8.7e-4 * Refo**0.17 * Su**0.50 * dr**0.14,   # tv
             1.5e-3 * Refo**0.59 * Su**0.19 * dr**0.36],  # vt
            3.5e-5 * Refo**0.44 * Su**0.50 * dr**0.48)    # vv
        We = G**2 * Dh / (fl.rf * fl.sig)
        BoH = Bo * HF
        boil = np.where(Ref >= 2000, 1 + 60 * We**0.32 * BoH**0.78,
                        1 + 530 * We**0.52 * BoH**1.09)
        return Cnb * boil

    # ----- pressure drop ----------------------------------------------------
    def dP(self, m, Q, nz=48):
        """Inlet-to-outlet pressure drop of every channel. m: (N,), Q scalar or (N,)."""
        fl = self.fl
        m = np.maximum(m, 1e-9)
        Q = np.broadcast_to(np.asarray(Q, float), m.shape)
        G, Dh, Po, L = m / self.A, self.Dh, self.Po, self.L
        qlin = Q / L
        Bo = Q / (self.PH * L) / (G * fl.hfg)

        xo_sv = self.x_osv(m, Q)                       # 0 without subcooled boiling
        L_sp = np.clip(m * (fl.cpf * self.dT_sub + fl.hfg * xo_sv) / qlin, 0, L)
        dP_sp = 2 * fanning(G * Dh / fl.mf, Po) * G**2 * L_sp / (fl.rf * Dh)

        xe_end = np.clip(self.x_eq(m, Q), xo_sv, 1)    # equilibrium quality range of the
        s = np.linspace(0, 1, nz)[None, :]             # two-phase region: xo_sv -> xe_end
        c = lambda a: a[:, None]
        xeq = c(xo_sv) + c(xe_end - xo_sv) * s
        x = np.clip(self.x_true(xeq, c(xo_sv)), 1e-6, 1 - 1e-6)
        Ref = c(G) * (1 - x) * c(Dh) / fl.mf
        Reg = c(G) * x * c(Dh) / fl.mg
        dPf = 2 * fanning(Ref, c(Po)) * c(G)**2 * (1 - x)**2 / (fl.rf * c(Dh))
        dPg = 2 * fanning(Reg, c(Po)) * c(G)**2 * x**2 / (fl.rg * c(Dh))
        X = np.sqrt(dPf / dPg)
        if self.C_model == "kim":
            C = self._C(c(G), x, c(Bo), c(Dh), c(self.HF), fl)
        else:  # Mishima & Hibiki 1996; Qu & Mudawar 2003 adds a mass-flux factor (Dh in m)
            C = 21 * (1 - np.exp(-319 * c(Dh))) * np.ones_like(x)
            if self.C_model == "qu":
                C = C * (0.00418 * c(G) + 0.0613)
        dzdx = m * fl.hfg / qlin
        dP_tp = self.tp_scale * dzdx * np.trapezoid(dPf * (1 + C / X + 1 / X**2), xeq, axis=1)

        L_v = np.maximum(L - L_sp - np.minimum(dzdx * (xe_end - xo_sv), L - L_sp), 0)
        dP_v = 2 * fanning(G * Dh / fl.mg, Po) * G**2 * L_v / (fl.rg * Dh)

        xe = self.x_true(xe_end, xo_sv)                # flow quality at exit
        xo = np.clip(xe, 1e-9, 1)
        a = np.clip(1 / (1 + (1 - xo) / xo * (fl.rg / fl.rf)**(2 / 3)), 1e-9, 1 - 1e-12)
        v_out = np.where(xe >= 1, 1 / fl.rg,
                         xo**2 / (fl.rg * a) + (1 - xo)**2 / (fl.rf * (1 - a)))
        dP_acc = G**2 * (v_out - 1 / fl.rf)

        dP_r = 0.0 if self.Ar is None else self.Kr * (m / self.Ar)**2 / (2 * fl.rf)
        dP_uh = 2 * fanning(G * Dh / fl.mf, Po) * G**2 * self.L_uh / (fl.rf * Dh)
        return dP_sp + dP_tp + dP_v + dP_acc + dP_r + dP_uh


# ---------------------------------------------------------------- network
def relax_thermal(ch, m0, Q, M):
    """Flow + wall-temperature dynamics (Rahman & Weibel 2024, Eq. 4):
    Cw dTw/dt = Q - hA (Tw - Tf) - C_lat L sum(Tw_i - Tw_j); fluid gets hA (Tw - Tf).
    Wall heat capacity is what lets lateral conduction stabilise the flow."""
    N, invI = ch.N, 1 / ch.I
    wI = invI / invI.sum()
    Q = np.broadcast_to(np.asarray(Q, float), (N,)).astype(float)
    K = ch.C_lat * ch.L
    Lap = (np.diag(np.r_[0, np.ones(N - 1)] + np.r_[np.ones(N - 1), 0])
           - np.eye(N, k=1) - np.eye(N, k=-1))
    fl = ch.fl

    def wall(m):
        h, f_sp = ch.h_bertsch(m, Q)
        return h * ch.PH * ch.L, f_sp * (fl.Tsat - ch.dT_sub / 2) + (1 - f_sp) * fl.Tsat

    m0 = np.asarray(m0, float)
    Tw0 = getattr(ch, "_Tw", None)
    if Tw0 is None or len(Tw0) != N:                      # start from the steady wall field
        hA, Tf = wall(m0)
        Tw0 = np.linalg.solve(np.diag(hA) + K * Lap, Q + hA * Tf)

    def rhs(t, y):
        m, Tw = np.maximum(y[:N], 1e-10), y[N:]
        hA, Tf = wall(m)
        qf = np.maximum(hA * (Tw - Tf), 1e-6)
        dp = ch.dP(m, qf)
        return np.r_[(np.dot(wI, dp) - dp) * invI, (Q - qf - K * Lap @ Tw) / ch.Cw]

    y = np.r_[m0, Tw0]
    ch.unsettled = False
    hA0, _ = wall(m0)
    dt = max(2.0, 20 * ch.Cw / hA0.min()) / 20           # wall time constant sets the horizon
    for k in range(20):                                   # 20 chunks
        n = [0]

        def rhs_c(t, yy):
            n[0] += 1
            if n[0] > 6000:
                raise RuntimeError("eval cap")
            return rhs(t, yy)

        try:
            sol = solve_ivp(rhs_c, (0, dt), y, method="BDF", rtol=1e-6, atol=1e-11)
        except RuntimeError:
            ch.unsettled = True
            break
        y = sol.y[:, -1]
    m, ch._Tw = np.maximum(y[:N], 1e-10), y[N:]
    hA, Tf = wall(m)
    ch._qf = np.maximum(hA * (ch._Tw - Tf), 1e-6)
    return m * M / m.sum()


def relax(ch, m0, Q, M):
    """Integrate I dm/dt = dPp - dP(m) to steady state (picks the stable branch)."""
    if ch.C_lat > 0:
        return relax_thermal(ch, m0, Q, M)
    invI = 1 / ch.I
    wI = invI / invI.sum()

    def rhs(t, m):
        dp = ch.dP(m, Q)
        return (np.dot(wI, dp) - dp) * invI

    def jac(t, m):
        h = 1e-6 * np.maximum(m, 1e-8)
        s = (ch.dP(m + h, Q) - ch.dP(m, Q)) / h
        return invI[:, None] * (np.outer(np.ones_like(m), wI * s) - np.diag(s))

    # ponytail: chunked integration with an eval cap; a step with no steady state
    # (flow oscillation) returns its last state instead of hanging. Flag via ch.unsettled.
    m, ch.unsettled = np.asarray(m0, float), False
    for k in range(20):                                   # 20 chunks x 0.1 s = 2 s
        n = [0]

        def rhs_c(t, y):
            n[0] += 1
            if n[0] > 4000:
                raise RuntimeError("eval cap")
            return rhs(t, y)

        try:
            sol = solve_ivp(rhs_c, (0, 0.1), m, method="BDF", jac=jac, rtol=1e-7, atol=1e-13)
        except RuntimeError:
            ch.unsettled = True
            break
        m = np.maximum(sol.y[:, -1], 1e-10)   # no early exit: an unstable even split
    return m * M / m.sum()                    # needs the full 2 s for noise to grow


def ramp(ch, M, Q_list, seed=0, noise=1e-5, stop=True, m0=None):
    """Ramp per-channel power; return (first dryout power, history)."""
    rng = np.random.default_rng(seed)
    m = np.full(ch.N, M / ch.N) if m0 is None else m0.copy()
    hist, Q_dry = [], np.nan
    for Q in Q_list:
        m = m * (1 + noise * rng.standard_normal(ch.N))      # physical noise
        m = relax(ch, m * M / m.sum(), Q, M)
        qf = ch._qf if ch.C_lat > 0 else np.broadcast_to(Q, m.shape)
        xo, xdi = ch.x_out(m, qf), ch.x_dryout(m, qf)
        hist.append((Q, m.copy(), xo, xdi))
        if np.isnan(Q_dry) and np.any(xo >= xdi):
            Q_dry = Q
            if stop:
                break
    return Q_dry, hist


def dryout_power(ch, M, Q_max, coarse=3.0, fine=0.25, seed=0):
    """Adaptive ramp: coarse steps to the first dryout, then refine from the
    last safe state (keeps the path history that Ledinegg needs)."""
    Q_dry, hist = ramp(ch, M, np.arange(1.0, Q_max, coarse), seed=seed)
    if np.isnan(Q_dry) or len(hist) < 2:
        return Q_dry
    Q_safe, m_safe = hist[-2][0], hist[-2][1]
    Q_fine, _ = ramp(ch, M, np.arange(Q_safe + fine, Q_dry + 1e-9, fine),
                     seed=seed + 1, m0=m_safe)
    return Q_fine if not np.isnan(Q_fine) else Q_dry


def frozen_split_dryout(ch, M, Q_low=1.0):
    """Dryout power if the flow kept its single-phase split (no two-phase
    feedback). The gap to the true value = amplification by the instability."""
    m = relax(ch, np.full(ch.N, M / ch.N), Q_low, M)
    out = []
    for i in range(ch.N):
        sub = Channels([ch.w[i]], ch.d, ch.L, ch.fl, ch.dT_sub)
        out.append(uniform_dryout_power(sub, m[i]))
    return min(out)


def uniform_dryout_power(ch, m_nom):
    """Per-channel dryout power if flow stayed perfectly uniform (ideal reference)."""
    f = lambda Q: (ch.x_out(np.array([m_nom]), Q) - ch.x_dryout(np.array([m_nom]), Q))[0]
    return brentq(f, 1e-3, 1e4)
