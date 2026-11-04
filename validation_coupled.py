"""V3: Miglani, Weibel & Garimella 2021 (IJMF 103536), isolated vs thermally coupled channels.
Measured (text, absorbed total power): isolated maldistribution 7.4-11.8 W, min starved 3.5 %;
coupled (C_lat ~ 18.5 W/K) 7.4-8.5 W, min starved 10 %."""
import numpy as np
from CoolProp.CoolProp import PropsSI
from model import Channels, Water, ramp

P = 104.4e3
fl = Water(P)
M = 10e-6 / 60 * PropsSI("D", "T", 88.5 + 273.15, "P", P, "Water")
Qtot = np.arange(5.0, 16.01, 0.25)                       # total absorbed power, W
for name, C in [("isolated", 0.0), ("coupled", 18.5 / 55e-3)]:
    ch = Channels([1e-3, 1e-3], 1e-3, 55e-3, fl, 11.6)
    ch.L_uh, ch.C_model, ch.C_lat = 55e-3, "mishima", C
    ch.Cw = 8960 * 385 * 55e-3 * 15e-3 * 15e-3           # copper block per channel, J/K (approx.)
    _, h = ramp(ch, M, Qtot / 2, stop=False)
    starved = np.array([m.min() / M for _, m, _, _ in h])
    mal = Qtot[starved < 0.45]
    rng = f"{mal.min():.2f}-{mal.max():.2f} W" if mal.size else "none"
    print(f"{name:8s}: maldistribution {rng}, min starved {starved.min()*100:.1f}% "
          f"at {Qtot[starved.argmin()]:.2f} W", flush=True)
    np.savetxt(f"v3_{name}.csv", np.c_[Qtot, starved], delimiter=",",
               header="P_total_abs_W,starved_fraction", comments="")
