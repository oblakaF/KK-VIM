"""Session model of the setup page: one SetupState shared by the main window's sections (saved and opened as a
project .kkv by ``project_file``)."""

from PySide6.QtCore import QObject, Signal

from kkvim_gui.model.setup_state import SetupState, summary_rows, to_config, validate


class SetupModel(QObject):
    changed = Signal()

    def __init__(self, state=None, parent=None):
        super().__init__(parent)
        self._state = state or SetupState()

    @property
    def state(self):
        return self._state

    def _set(self, state):
        if state != self._state:
            self._state = state
            self.changed.emit()

    def set_text(self, key, text):
        self._set(self._state.with_text(key, text))

    def set_unit(self, key, unit):
        self._set(self._state.with_unit(key, unit))

    def set_choice(self, **choices):
        self._set(self._state.with_choice(**choices))

    def load(self, state):
        """Replace the whole setup by a complete, already checked state (an opened project, «Новый расчёт»).

        The state is taken as it is: no choice is reset or derived here (``project_file`` checked its rules)."""
        if not isinstance(state, SetupState):
            raise TypeError(f"SetupState expected, got {type(state).__name__}")
        self._set(state)

    def errors(self):
        return validate(self._state)

    def config(self):
        return to_config(self._state)

    def summary(self):
        return summary_rows(self._state)
