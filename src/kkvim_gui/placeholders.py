"""Static demonstration content of the P1 skeleton. TEST DATA, not results.

Everything here only fills the layout for design review. The numbers are arbitrary
placeholders: they are not computed, not derived from any formula and must not be
read as plate results (GUI_DESIGN.md, section 9). Stage P3 replaces the run and
result values with DemoBackend output, still marked TEST DATA.
"""

PROJECT_NAME = "plate_ssss_01.kkv"
RUN_TITLE = "plate_ssss_01 · 12.10.2026 10:48"

# Initial values of the setup form as text (an example plate for the solver); units are SI symbols (GUI-D14).
SETUP = {
    "a": ("2,000", "m"),
    "b": ("1,000", "m"),
    "h": ("10,0", "mm"),
    "E": ("210", "GPa"),
    "nu": "0,300",
    "q": ("5,000", "kPa"),
}

RECENT_PROJECTS = (
    ("plate_ssss_01", "2,0 × 1,0 m", "SSSS", "3", "12.10.2026", "TEST DATA"),
    ("plate_cccc_01", "1,0 × 1,0 m", "CCCC", "1", "11.10.2026", "—"),
    ("plate_ssss_sq", "1,0 × 1,0 m", "SSSS", "2", "10.10.2026", "—"),
)

RUNNING = {
    "iteration": "3 / 100",
    "residual": "1,0·10⁻³",
    "elapsed": "00:00:06",
}

FINISHED = {
    "k_stop": 8,
    "extremum": "−1,234 mm",
    "residual": "1,0·10⁻⁶",
    "elapsed": "00:00:11",
    "N": "3",
}

LOG_RUNNING = (
    ("10:48:03", "ИНФО", "Формальная проверка ввода в интерфейсе выполнена"),
    ("10:48:03", "ИНФО", "Демонстрационный бэкенд: решатель не подключён"),
    ("10:48:04", "ПРОЦЕСС", "Итерация 1"),
    ("10:48:05", "ПРОЦЕСС", "Итерация 2"),
    ("10:48:06", "ПРОЦЕСС", "Итерация 3"),
)

LOG_FINISHED = LOG_RUNNING + (
    ("10:48:07", "ПРОЦЕСС", "Итерации 4–8"),
    ("10:48:14", "ИНФО", "Критерий остановки выполнен (демонстрация)"),
    ("10:48:14", "ИНФО", "Результаты подготовлены: TEST DATA"),
)
