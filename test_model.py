"""Smallest checks that fail if the physics breaks. Run: python3 test_model.py"""
import numpy as np

from model import Channels, Water, dryout_power, uniform_dryout_power

fl = Water()
W, D, L, DT = 0.3e-3, 0.6e-3, 25e-3, 30.0
one = Channels([W], D, L, fl, DT)
m0 = 200 * one.A[0]
Q_ref = uniform_dryout_power(one, m0)

# 1. Demand curve has a Ledinegg (negative-slope) region at moderate power
ms = np.linspace(0.5, 3, 60) * m0
dp = np.array([one.dP(np.array([m]), 0.3 * Q_ref)[0] for m in ms])
assert (np.diff(dp) < 0).any(), "no negative-slope region"

# 2. Identical channels + strong restrictor reach the uniform-flow dryout power
ar = m0 * np.sqrt(1.5 / (2 * fl.rf * 1.0 * one.dP(np.array([m0]), Q_ref)[0]))
two = Channels([W, W], D, L, fl, DT, restrictor_area=ar)
k = dryout_power(two, 2 * m0, 1.3 * Q_ref) / Q_ref
assert abs(k - 1) < 0.02, k

# 3. No restrictor: Ledinegg starves a channel and dryout comes far earlier
bare = Channels([W, W], D, L, fl, DT)
assert dryout_power(bare, 2 * m0, 1.3 * Q_ref) / Q_ref < 0.5

print("all checks pass")
