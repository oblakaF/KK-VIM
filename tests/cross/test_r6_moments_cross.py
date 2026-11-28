"""R6 cross checks between kkvim.moments and the references (kkvim_reference.r6_moments); tests only (D-028).

1. The same rule-A operators, implemented twice (kkvim.moments and r6_moments.d2_moments), give the same moments of
   one field (within the rounding model): two implementations of one formula, not an accuracy statement.
2. The reconstruction applied to the EXACT Navier deflection at the nodes converges to the Navier moments with
   second order at the centre (E_der, an observation of the operator order on exact data; review 8.9 - no threshold).
"""

import math

import numpy as np

from kkvim import moments as mo
from kkvim_reference import navier
from kkvim_reference import r5_cccc_d2 as d2
from kkvim_reference import r6_moments as rm


def test_two_implementations_of_rule_a_give_the_same_moments_of_the_d2_field():
    a, b, D, nu = 1.2, 1.0, 1500.0, 0.3
    sol = d2.solve(d2.D2Problem(a, b, 24, 20, D, 1000.0))
    W = sol.field()
    ours = mo.plate_moments(W, a, b, D, nu, "CC")
    ref = rm.d2_moments(W, a, b, D, nu)
    for key in ("Mx", "My", "Mxy"):
        assert np.max(np.abs(getattr(ours, key) - ref[key])) <= ours.rounding_model


def test_reconstruction_of_the_exact_navier_field_converges_at_the_centre():
    a, b, q, E, t, nu = 1.0, 1.0, 1000.0, 2.1e11, 0.01, 0.3
    D = navier.flexural_rigidity(E, t, nu)
    errors = []
    for n in (16, 32, 64):
        plate = navier.NavierRectangle.from_material(a=a, b=b, q=q, E=E, thickness=t, nu=nu)
        W = np.array(plate.at_nodes(n, n, M=1279, diagnostics=False).w, dtype=float)
        W[[0, -1], :] = 0.0
        W[:, [0, -1]] = 0.0
        M = mo.plate_moments(W, a, b, D, nu, "SS")
        ref = rm.navier_moments_points(a, b, q, nu, [0.5], [0.5], M=2559)
        errors.append(abs(M.Mx[n // 2, n // 2] - float(ref["Mx"][0])))
    orders = [math.log2(errors[k] / errors[k + 1]) for k in range(2)]
    assert all(1.8 <= p <= 2.2 for p in orders), orders
