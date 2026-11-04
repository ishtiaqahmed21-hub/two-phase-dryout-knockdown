"""R1 study: dryout knockdown vs channel-width tolerance, channel count, restrictor.

Writes results.csv (one row per Monte Carlo sample) and fig1_data.npz.
"""
import csv
import itertools
import sys
import time
from multiprocessing import Pool

import numpy as np

from model import Channels, Water, dryout_power, frozen_split_dryout, ramp, uniform_dryout_power

# ---- base case (state these in any summary) -------------------------------
W, D, L, DT_SUB, G_NOM = 0.3e-3, 0.6e-3, 25e-3, 30.0, 200.0   # m, m, m, K, kg/m2s
FL = Water(101325.0)
C_MODEL = next((a for a in sys.argv if a in ("kim", "mishima")), "kim")
REF = Channels([W], D, L, FL, DT_SUB)
REF.C_model = C_MODEL
M_NOM = G_NOM * REF.A[0]
Q_REF = uniform_dryout_power(REF, M_NOM)                     # W per channel
DP_REF = REF.dP(np.array([M_NOM]), Q_REF)[0]


def restrictor_area(r):
    """Orifice area giving dP_restrictor = r * dP_channel at nominal flow and Q_REF."""
    return None if r == 0 else M_NOM * np.sqrt(1.5 / (2 * FL.rf * r * DP_REF))


def one_sample(args):
    N, sigma, r, seed = args
    rng = np.random.default_rng(seed)
    eps = np.clip(rng.standard_normal(N) * sigma, -3 * sigma, 3 * sigma)
    ch = Channels(W * (1 + eps), D, L, FL, DT_SUB, restrictor_area=restrictor_area(r))
    ch.C_model = C_MODEL
    M = N * M_NOM
    q = dryout_power(ch, M, Q_max=1.3 * Q_REF, seed=seed)
    qf = frozen_split_dryout(ch, M)
    return dict(N=N, sigma=sigma, r=r, seed=seed, kappa=q / Q_REF,
                kappa_frozen=qf / Q_REF, eps_min=eps.min())


def fig1_data():
    """2 identical channels, flow split vs power for several restrictor strengths."""
    out = {}
    Qs = np.arange(1.0, 50.0, 1.0)
    for r in [0.0, 0.01, 0.02, 0.03]:
        ch = Channels([W, W], D, L, FL, DT_SUB, restrictor_area=restrictor_area(r))
        ch.C_model = C_MODEL
        _, h = ramp(ch, 2 * M_NOM, Qs, stop=False)
        out[f"r{r}"] = np.array([[Q, m[0] / m.sum(), m[1] / m.sum(), xo.max(), xd.min()]
                                 for Q, m, xo, xd in h])
    np.savez(f"fig1_data_{C_MODEL}.npz", Q_REF=Q_REF, **out)


if __name__ == "__main__":
    print(f"Q_REF = {Q_REF:.2f} W/channel, dP_REF = {DP_REF/1e3:.1f} kPa", flush=True)
    if "fig1" in sys.argv:
        fig1_data()
        sys.exit()
    n_samp = next((int(a) for a in sys.argv[1:] if a.isdigit()), 20)
    jobs = [(N, s, r, 1000 * i + k)
            for i, (N, s, r) in enumerate(itertools.product(
                [2, 10, 50], [0.01, 0.025, 0.05, 0.075, 0.10], [0.03, 1.0]))
            for k in range(n_samp)]
    t0 = time.time()
    with Pool(2) as pool, open(f"results_{C_MODEL}.csv", "w", newline="") as f:
        wr = None
        for i, row in enumerate(pool.imap_unordered(one_sample, jobs, chunksize=2)):
            if wr is None:
                wr = csv.DictWriter(f, fieldnames=list(row))
                wr.writeheader()
            wr.writerow(row)
            f.flush()
            if i % 50 == 0:
                print(f"{i}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)
    print("done", time.time() - t0)
