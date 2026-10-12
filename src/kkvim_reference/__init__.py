"""KK-VIM independent reference solutions (D-015, D-028, D-033).

This package holds reference solutions computed independently of the main
KK-VIM solver, used by QA to verify it. It must not import ``kkvim``, and
``kkvim`` must not import it (D-028).

Contents:
- ``kkvim_reference.levy`` - independent analytic reference for the first
  KK-VIM half-step s = 1 by Levy's formula: w1 = sin(pi x/a) * Y1(y),
  square SSSS plate, uniform load, N = 1 (MATH_SPEC.md, В-29, В-30).

The Navier reference and the semi-analytic half-step sequence
(NUMERICAL_SPEC.md, sections 15-16) are not implemented yet.
"""
