"""Figures for the one-page summary. Reads fig1_data.npz and results.csv."""
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys
CM = next((a for a in sys.argv if a in ("kim", "mishima")), "mishima")

INK, INK2, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
CAT = {2: "#2a78d6", 10: "#eb6834", 50: "#1baf7a"}            # validated slots 1-3
ORD = ["#9ec5f4", "#5598e7", "#256abf", "#104281"]             # blue ordinal ramp

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2,
    "figure.dpi": 200, "savefig.bbox": "tight",
})


def fig1():
    d = np.load(f"fig1_data_{CM}.npz")
    qref = float(d["Q_REF"])
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for c, (key, lab) in zip([ORD[0], ORD[1], ORD[3]],
                             [("r0.0", "no restrictor"), ("r0.01", "restrictor r = 0.01"),
                              ("r0.03", "r = 0.02 or 0.03: stays even")]):
        a = d[key]
        x = a[:, 0] / qref
        starved = np.minimum(a[:, 1], a[:, 2])
        ax.plot(x, starved * 100, color=c, label=lab, solid_capstyle="round")
        if np.any(a[:, 3] >= a[:, 4]):
            i = np.argmax(a[:, 3] >= a[:, 4])
            ax.plot(x[i], starved[i] * 100, "o", ms=6, color=c, mec="white", mew=1.5, zorder=5)
    ax.set_xlabel("Power per channel / uniform-flow dryout power")
    ax.set_ylabel("Flow to starved channel (%)")
    ax.set_ylim(0, 55)
    ax.set_xlim(0, 1.05)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.50, 0.84))
    fig.text(0.01, -0.10, "r = restrictor pressure drop / channel pressure drop at nominal flow.",
             fontsize=7, color=INK2)
    ax.set_title("Two identical channels: without inlet restriction,\n"
                 "boiling onset starves one channel (dot = first dryout)",
                 loc="left", fontsize=9.5, color=INK)
    fig.savefig(f"fig1_ledinegg_2ch_{CM}.png")


def load():
    g = defaultdict(list)
    for r in csv.DictReader(open(f"results_{CM}.csv")):
        g[(int(r["N"]), float(r["r"]), float(r["sigma"]))].append(
            (float(r["kappa"]), float(r["kappa_frozen"])))
    return g


def fig2():
    g = load()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
    for ax, r, title in zip(axes, [1.0, 0.03], ["Strong inlet restrictor (r = 1)",
                                               "Marginal inlet restrictor (r = 0.03)"]):
        for N in [2, 10, 50]:
            sig = sorted(s for (n, rr, s) in g if n == N and rr == r)
            if not sig:
                continue
            k = [np.array(g[(N, r, s)])[:, 0] for s in sig]
            x = np.array(sig) * 100
            med = np.array([np.median(v) for v in k])
            ax.fill_between(x, [np.percentile(v, 10) for v in k],
                            [np.percentile(v, 90) for v in k], color=CAT[N], alpha=0.15, lw=0)
            ax.plot(x, med, color=CAT[N], marker="o", ms=4.5, mec="white", mew=1)
            dy = {10: 7, 50: -6}.get(N, 0) if r != 1.0 else 0   # end points crowd at low kappa
            ax.annotate(f"N = {N}", (x[-1], med[-1]), xytext=(5, dy), textcoords="offset points",
                        va="center", fontsize=8, color=INK)
            if N == 50:
                kf = [np.median(np.array(g[(N, r, s)])[:, 1]) for s in sig]
                ax.plot(x, kf, color=INK2, lw=1.2, ls="--")
                if r == 1.0:   # lines bunch near 1: label above the dashed line
                    ax.annotate("N = 50, single-phase flow split only", (x[1], kf[1]),
                                xytext=(0, 7), textcoords="offset points", va="bottom",
                                fontsize=7.5, color=INK2)
                else:
                    ax.annotate("N = 50, single-phase\nflow split only", (x[-1], kf[-1]),
                                xytext=(5, 0), textcoords="offset points", va="center",
                                fontsize=7.5, color=INK2)
        ax.set_title(title, loc="left", fontsize=9.5, color=INK)
        ax.set_xlabel("Channel-width tolerance, 1σ (%)")
        ax.set_xlim(0, 13.5)
    axes[0].set_ylabel("Dryout knockdown factor κ\n(plate dryout power / ideal)")
    axes[0].set_ylim(0, 1.08)
    fig.suptitle("More channels and looser tolerance lower the real dryout limit; "
                 "the two-phase feedback amplifies it",
                 x=0.01, ha="left", fontsize=10, color=INK, y=1.03)
    fig.text(0.01, -0.09, "Lines: median of 20 Monte Carlo samples; bands: 10th-90th percentile. "
             "Water, 1 atm, 30 K subcooling, G = 200 kg/m²s, 0.3 × 0.6 × 25 mm channels.",
             fontsize=7, color=INK2)
    fig.savefig(f"fig2_knockdown_{CM}.png")


if __name__ == "__main__":
    if "fig2" not in sys.argv:
        fig1()
    if "fig1" not in sys.argv:
        fig2()
