"""KK-VIM independent reference solutions (D-015, D-028, D-033).

This package holds reference solutions computed independently of the main
KK-VIM solver, used by QA to verify it. It must not import ``kkvim``, and
``kkvim`` must not import it (D-028).

Contents:
- ``kkvim_reference.levy`` - independent analytic reference for the first
  KK-VIM half-step s = 1 by Levy's formula: w1 = sin(pi x/a) * Y1(y),
  square SSSS plate, uniform load, N = 1 (MATH_SPEC.md, В-29, В-30);
- ``kkvim_reference.sine_halfstep`` - sine-series solution of one 1D
  half-step D(A f'''' - 2B f'' + C f) = F with f = f'' = 0 at both ends and
  its analytic truncation bound (NUMERICAL_SPEC.md, 15.1, 15.4).

The recursion over half-steps (NUMERICAL_SPEC.md, 15.2-15.5) and the Navier
reference (section 16) are not implemented yet.
"""
