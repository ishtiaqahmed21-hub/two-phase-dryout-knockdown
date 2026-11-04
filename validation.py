"""V1 (Qu & Mudawar 2003) and V2 (Miglani et al. 2021) validation figure + errors.

Measured points were read from the papers' figures:
  - Qu & Mudawar 2003, Fig. 6 (T_in = 60 C, G = 255 kg/m2s): marker centres found by
    pixel detection on a 400-dpi render; 4 markers hidden by correlation lines dropped.
  - Miglani et al. 2021, Fig. 4a/4c: flow fractions at 13.54 and 18.41 W from the text;
    the rest read by eye (about +-0.01 in fraction, +-1 kW/m2 in heat flux).
Absorbed power per channel (Miglani) = mean heat flux of the two channels x 3 heated
walls (3 mm x 55 mm), because the paper's total power includes heater losses.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from CoolProp.CoolProp import PropsSI

from model import Channels, Water, ramp
from plot import CAT, INK, INK2
import sys
SUB = "sub" in sys.argv          # subcooled boiling on/off
CM = next((a for a in sys.argv if a in ("kim", "mishima", "qu")), "kim")

# ---- V1: Qu & Mudawar 2003 -------------------------------------------------
qu = np.loadtxt("qu2003_fig6_digitized.csv", delimiter=",", skiprows=1)
fl1 = Water(1.17e5)
c1 = Channels([231e-6], 712e-6, 44.8e-3, fl1, fl1.Tsat - 333.15)
c1.sub_boil = SUB
c1.C_model = CM
Qch = qu[:, 0] * 1e4 * (1.0e-2 * 4.48e-2) / 21
dp1 = np.array([c1.dP(255 * c1.A, q)[0] for q in Qch]) / 1e5
mae1 = np.mean(np.abs(dp1 - qu[:, 1]) / qu[:, 1]) * 100

# ---- V2: Miglani et al. 2021 -------------------------------------------------
# P_T [W], flow fraction of the high-flow channel, heat flux ch1, ch2 [kW/m2]
mig = np.array([[8.3, 0.50, 8.0, 8.0], [10.0, 0.50, 11.0, 11.0],
                [13.54, 0.77, 25.0, 20.5], [14.77, 0.92, 29.5, 23.5],
                [16.8, 0.96, 35.5, 28.0], [18.41, 0.965, 40.0, 32.5]])
np.savetxt("miglani2021_fig4_read.csv", mig, delimiter=",",
           header="P_total_W,eps_high,q1_kW_m2,q2_kW_m2", comments="")
Q_abs = mig[:, 2:].mean(1) * 1e3 * 3e-3 * 55e-3
P2 = PropsSI("P", "T", 100.8 + 273.15, "Q", 0, "Water")
fl2 = Water(P2)
c2 = Channels([1e-3, 1e-3], 1e-3, 55e-3, fl2, 12.3)
c2.L_uh = 55e-3
c2.sub_boil = SUB
c2.C_model = CM
M2 = 9.3e-6 / 60 * PropsSI("D", "T", 88.5 + 273.15, "P", P2, "Water")
Qs = np.arange(0.5, 7.01, 0.1)
_, h = ramp(c2, M2, Qs, stop=False)
eps_model = np.array([m.max() / M2 for _, m, _, _ in h])
eps_at = np.interp(Q_abs, Qs, eps_model)

if __name__ == "__main__":
    print(f"V1 Qu & Mudawar: MAE = {mae1:.0f}%  (model over-predicts on average)")
    for q, e, em in zip(Q_abs, mig[:, 1], eps_at):
        print(f"V2 Q_abs = {q:4.2f} W/ch   measured starved share {1-e:.3f}   model {1-em:.3f}")
    print(f"V2 single-phase overall dP: model {np.mean(c2.dP(np.full(2, M2/2), 2.0)):.0f} Pa, measured 81.5 Pa")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    a1.plot(qu[:, 0], qu[:, 1], "^", color=INK, ms=5, label="Measured (Qu & Mudawar 2003)")
    a1.plot(qu[:, 0], dp1, color=CAT[2], label=f"Model (MAE {mae1:.0f}%)")
    a1.set_xlabel("Effective heat flux (W/cm²)")
    a1.set_ylabel("Heat-sink pressure drop (bar)")
    a1.set_title("V1: model over-predicts two-phase\npressure drop at this condition",
                 loc="left", fontsize=9.5, color=INK)
    a1.legend(frameon=False, fontsize=7.5, loc="upper left")
    a1.set_ylim(0, None)

    a2.plot(Qs, (1 - eps_model) * 100, color=CAT[2], label="Model")
    a2.plot(Q_abs, (1 - mig[:, 1]) * 100, "o", color=INK, ms=5,
            label="Measured (Miglani et al. 2021)")
    a2.set_xlabel("Absorbed power per channel (W)")
    a2.set_ylabel("Flow to starved channel (%)")
    a2.set_ylim(0, 55)
    a2.set_title("V2: onset and trend captured,\nseverity under-predicted",
                 loc="left", fontsize=9.5, color=INK)
    a2.legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.text(0.01, -0.06, "V1: T_in = 60 °C, G = 255 kg/m²s, 21 channels 231 × 712 µm. "
             "V2: water, 2 thermally isolated 1 × 1 mm channels, 9.3 ml/min, 12.3 K subcooling.",
             fontsize=7, color=INK2)
    fig.tight_layout()
    fig.savefig(f"fig_validation_{CM}.png")
