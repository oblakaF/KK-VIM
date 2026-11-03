"""Backend of the GUI (GUI_DESIGN.md, 3.4, 3.6). Qt-free.

- ``units`` — SI prefixes of the input fields (D-009, GUI-D14);
- ``request`` — ``RunRequest``: the entered values, their SI values and the method parameters;
- ``outcome`` — ``RunOutcome``: an immutable copy of the solver result for the interface;
- ``kkvim_adapter`` — the only module of the GUI that imports ``kkvim``;
- ``process`` — the computation in a separate process (3.6);
- ``export`` — JSON and CSV files of a result.
"""
