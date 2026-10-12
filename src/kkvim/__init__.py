"""KK-VIM: plate analysis by the Kirichenko-Krysko variational iteration method.

R2 implementation in progress (D-033), following the accepted docs/MATH_SPEC.md,
part A (D-031), and docs/NUMERICAL_SPEC.md (D-032). Implemented so far: the
one-dimensional grid and its difference operators (``kkvim.grid``) and the
one-dimensional linear solver K Y = g (``kkvim.linear_solver``). Half-step
coefficients and the KK-VIM iterations are not implemented yet.
"""
