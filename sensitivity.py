"""Does the Fig. 2 headline survive a two-phase friction calibrated to Qu & Mudawar?"""
import numpy as np
from multiprocessing import Pool
from CoolProp.CoolProp import PropsSI
import run_study as rs
from model import Channels, Water, ramp, dryout_power, uniform_dryout_power

S = 0.45   # friction scale that fits Qu & Mudawar 2003 Fig. 6 (MAE ~7%)

def v2(scale):
    P = PropsSI("P", "T", 100.8 + 273.15, "Q", 0, "Water"); fl = Water(P)
    c = Channels([1e-3, 1e-3], 1e-3, 55e-3, fl, 12.3); c.L_uh = 55e-3; c.tp_scale = scale
    M = 9.3e-6 / 60 * PropsSI("D", "T", 88.5 + 273.15, "P", P, "Water")
    Qs = np.arange(0.5, 7.01, 0.1); _, h = ramp(c, M, Qs, stop=False)
    eps = np.array([m.max() / M for _, m, _, _ in h])
    Qa = np.array([1.32, 1.81, 3.75, 4.37, 5.24, 5.98])
    return 1 - np.interp(Qa, Qs, eps)

def sample(args):
    N, sig, r, seed, scale = args
    rng = np.random.default_rng(seed)
    eps = np.clip(rng.standard_normal(N) * sig, -3 * sig, 3 * sig)
    ref = Channels([rs.W], rs.D, rs.L, rs.FL, rs.DT_SUB); ref.tp_scale = scale
    qref = uniform_dryout_power(ref, rs.M_NOM)
    dpref = ref.dP(np.array([rs.M_NOM]), qref)[0]
    ar = rs.M_NOM * np.sqrt(1.5 / (2 * rs.FL.rf * r * dpref))
    ch = Channels(rs.W * (1 + eps), rs.D, rs.L, rs.FL, rs.DT_SUB, restrictor_area=ar)
    ch.tp_scale = scale
    return (N, sig, r, scale, dryout_power(ch, N * rs.M_NOM, 1.3 * qref, seed=seed) / qref)

if __name__ == "__main__":
    print("V2 starved share, scale 1.0 :", np.round(v2(1.0), 3))
    print("V2 starved share, scale 0.45:", np.round(v2(S), 3))
    print("V2 measured                 : [0.5 0.5 0.23 0.08 0.04 0.035]")
    jobs = [(N, 0.05, r, 7000 + k, s) for N in [10, 50] for r in [1.0, 0.03]
            for s in [1.0, S] for k in range(10)]
    with Pool(2) as p:
        res = p.map(sample, jobs)
    for N in [10, 50]:
        for r in [1.0, 0.03]:
            for s in [1.0, S]:
                k = [x[4] for x in res if x[:4] == (N, 0.05, r, s)]
                print(f"N={N:2d} sigma=5% r={r:<4} scale={s:<4}: median kappa {np.median(k):.3f} "
                      f"(range {min(k):.2f}-{max(k):.2f})")
