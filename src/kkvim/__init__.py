"""KK-VIM: plate analysis by the Kirichenko-Krysko variational iteration method.

R2 implementation in progress (D-033), following the accepted docs/MATH_SPEC.md,
part A (D-031), and docs/NUMERICAL_SPEC.md (D-032). Implemented so far: the
one-dimensional grid and its difference operators (``kkvim.grid``) and the
one-dimensional linear solver K Y = g (``kkvim.linear_solver``) and the
half-step coefficients A_h, B_h, C_h, F_h (``kkvim.coefficients``) and the
first half-step s = 1 with the field w1 (``kkvim.first_halfstep``). Later
half-steps and the KK-VIM iterations are not implemented yet.
"""
