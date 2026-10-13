"""KK-VIM: plate analysis by the Kirichenko-Krysko variational iteration method.

R2 implementation in progress (D-033), following the accepted docs/MATH_SPEC.md,
part A (D-031), and docs/NUMERICAL_SPEC.md (D-032). Implemented so far: the
one-dimensional grid and its difference operators (``kkvim.grid``) and the
one-dimensional linear solver K Y = g (``kkvim.linear_solver``) and the
half-step coefficients A_h, B_h, C_h, F_h (``kkvim.coefficients``) and the
first half-step s = 1 with the field w1 (``kkvim.first_halfstep``) and the
second half-step s = 2 with the field w2 (``kkvim.second_halfstep``), and a
fixed number of alternating half-steps with their history (``kkvim.iteration``),
the discrete energy Pi_h with the energy test of half-steps (``kkvim.energy``),
automatic stopping of the half-steps (``kkvim.controller``), and the
physical input a, t, E, nu, q with the dimensionless field (``kkvim.physical``).
"""
