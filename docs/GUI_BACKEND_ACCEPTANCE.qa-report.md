# Контрольные расчёты R3/R4 для подключения GUI — отчёт QA

Цели: G3.2, G3.3, G4. Роль: QA. База: `main` @ `e10c6b58fef1aa7ca71a410321ff3e943d8e6b18`. Ветка: `qa/gui-backend-acceptance`.

**Назначение:**
- независимая сквозная проверка расчётного ядра через публичный API;
- контрольные данные, по которым Builder проверит свой GUI-адаптер.

Это **регрессионные эталоны программной интеграции**, а не новые независимые научные решения. Сам GUI не проверялся.
Код `src/`, принятые спецификации, D-049, R5 и `main` не менялись.

## 1. Состав

| Файл | Что это |
|---|---|
| `scripts/gui_backend_cases.py` | инструмент проверки QA (CLI). Вызывает только функции библиотеки, собственного решателя нет |
| `tests/cross/test_gui_backend_acceptance.py` | 101 автоматическая проверка |
| `docs/results/gui_backend/cases.json` | машиночитаемый набор контрольных случаев (16 случаев) |
| `docs/results/gui_backend/cases/` | данные каждого случая: `<case>.json`, `<case>_history.csv`, `<case>_w.csv`, `<case>_w_bar.csv` |

**Используемые функции API:**

| N | Функции |
|---|---|
| N = 1 | `kkvim.rectangle_run.run_rectangle_physical`, `kkvim.rectangle_run.halfstep_report` |
| N = 2–3 | `kkvim.rectangle.validated_rectangle_input` (проверки R3 и D через `kkvim.physical.bending_stiffness`), затем `kkvim.coupled_run.run_coupled` |

**Собственная арифметика инструмента:**
- нормировка w̄ = wD/(qa⁴) для N = 2–3 — у R4 нет функции w̄; определение то же, что у R3 (MATH_SPEC 2.5);
- максимум |w| и его узел.

## 2. Контрольные случаи

Материал «steel»: t = 0,01, E = 2,1·10¹¹, ν = 0,3, q = 1000 в согласованных единицах (м, Па); D = 19 230,769…

| Случай | N | Режим | Плита, сетка | Статус | Класс |
|---|---|---|---|---|---|
| r3_square | 1 | — | 1 × 1, 24 × 24 | CONVERGED, s = 7 | COMPLETED |
| r3_rect_b2a | 1 | — | 1 × 2, 24 × 48 | CONVERGED, s = 7 | COMPLETED |
| r3_rect_a2b | 1 | — | 2 × 1, 48 × 24 | CONVERGED, s = 8 | COMPLETED |
| r3_rect_r5_unequal | 1 | — | 0,5 × 2,5, 16 × 40 (hₓ ≠ h_y) | CONVERGED, s = 7 | COMPLETED |
| r3_negative_q | 1 | — | 1 × 1,5, q = −500 | CONVERGED, s = 7 | COMPLETED |
| r3_error_nu / _ratio / _grid | 1 | — | ν = 1; b/a = 6; nₓ = 3 | ERROR_INPUT, без поля | ERROR |
| r4_n2_normative | 2 | нормативный, half_steps = 5 | 1 × 1, 24 × 24 | NOT_QUALIFIED, s = 1 | NOT_QUALIFIED |
| r4_n2_normative_one | 2 | нормативный, half_steps = 1 | 1 × 1, 24 × 24 | HALF_STEP_LIMIT_REACHED, s = 1 | NOT_QUALIFIED |
| r4_n2_research | 2 | исследовательский (D-048) | 1 × 1, 24 × 24 | PROVISIONAL_FIELD_STABILIZED, s = 10 | PROVISIONAL |
| r4_n3_research_rect | 3 | исследовательский | 1 × 2, 24 × 48 | PROVISIONAL_FIELD_STABILIZED, s = 11 | PROVISIONAL |
| r4_n3_research_fixed | 3 | 12 фиксированных полушагов | 2 × 1, 40 × 20 | RESEARCH_LIMIT_REACHED, s = 12 | PROVISIONAL |
| r4_n2_dense | 2 | исследовательский, `dense` (n ≤ 40, D-048.6) | 1 × 1, 24 × 24 | PROVISIONAL_FIELD_STABILIZED, s = 10 | PROVISIONAL |
| r4_n3_rank_loss | 3 | исследовательский | nₓ = 4 < 2N | STOPPED_RANK_LOSS_PROVEN, без поля | STOPPED_RANK_LOSS |
| r4_error_E | 2 | исследовательский | E < 0, t < 0 (D > 0) | ERROR_INPUT, без поля | ERROR |

**Состав записи каждого случая:**
- вход и единицы;
- точный вызов API;
- ожидаемый статус и класс;
- научный статус;
- D;
- число полушагов;
- сводка поля: форма, оси, max|w| и узел, центр, max|w̄|, граница;
- история диагностических величин;
- происхождение и ограничения;
- версия источника: SHA `79b6d82`, Python, NumPy, SciPy, платформа.

**Классы статусов** только группируют статусы библиотеки, сами статусы не меняются:
- COMPLETED — только R3 CONVERGED;
- R4 никогда не COMPLETED; научный статус R4 — всегда PROVISIONAL / NOT_QUALIFIED;
- неизвестный статус → UNKNOWN, считается ошибкой.

## 3. Проверки и критерии

Используются только принятые критерии, новых допусков нет.

| Проверка | Критерий | Основание |
|---|---|---|
| D | равно `bending_stiffness`; в пределах 4u от точного рационального значения | `test_physical_a9.py` |
| форма и ориентация поля | (nₓ + 1) × (n_y + 1), ось 0 — x на [0, a]; граничные узлы — точные нули; max\|w\|, узел и центр согласованы с массивом; знак — по нагрузке | API (`HalfStepRecord.field`, `field_from_sets`) |
| экспорт = прямой API | побитово (тот же процесс); JSON/CSV по repr, NPZ/NPY — точный обратный ход | AGENTS.md, разд. 4 |
| R3 = R2 при r = 1 | побитово | STATUS, R3, п. Н-6 |
| R3: статус и s_stop | как в принятой таблице `docs/results/r3/solver/qual/levels.csv` (6 конфигураций) | квалификация R3 (D-043) |
| единицы и нормировки | A-9 (R3, n = 320, оба семейства): ≤ 10⁻¹¹ max\|w̄\|; A4-12 (R4, N = 2, 3, r = 1/5, 1, 5, n ≤ 160, s = 6): ≤ 10⁻¹¹ max\|w̄\| | D-039; D-048.7 |
| нормативный режим R4 | `run_coupled` вызывается без `research_mode`; NOT_QUALIFIED или HALF_STEP_LIMIT_REACHED после s = 1, без QR | D-047, D-048.3 |
| dense и fft (n ≤ 40) | каждое поле — в модели `_model` относительно контроля QA; разность dense − fft — в сумме моделей | D-048.6; `test_r4_controller_comparison.py` |
| остановка без поля | поле не экспортируется, `field_available = false` | — |
| ошибка ≠ успех | инъекции ERROR_LINEAR_SOLVE и ERROR_ENERGY_TEST (R3), PROVISIONAL_ENERGY_E1_STOP и ERROR_LINEAR_SOLVE (R4) — никогда не COMPLETED | — |
| воспроизводимость | два прогона в одном процессе и два запуска CLI — тождественны (побайтно) | AGENTS.md, разд. 4 |
| эталонный набор | статус, класс, s, D (≤ 8u), наличие поля — точно. Поле R4 — в удвоенной модели `_model` (как К-1 D-048). Поле R3 — побитово только на эталонной машине (`KKVIM_GUI_BITWISE=1`), на других платформах разность сообщается, принятого межплатформенного критерия для R3 нет | — |

**Результаты:**
- `tests/cross/test_gui_backend_acceptance.py` — 101 passed (Python 3.13); с `KKVIM_GUI_BITWISE=1` — 101 passed;
- полный pytest — 3165 passed, 31 skipped на Python 3.13 и 3.14;
- `uv lock --check`, самотест hook, `git diff --check` — без замечаний.

## 4. Наблюдения для GUI-адаптера (Builder)

1. **Физический вход R4.** `run_coupled` принимает только D и проверяет стороны, D и сетки. При E < 0 и t < 0 получается D > 0, и `run_coupled` такой
   расчёт выполнит (тест `test_r4_physical_input_is_checked_like_r3`). Инструмент QA проверяет E, t, ν библиотечной функцией R3
   `validated_rectangle_input`. GUI должен делать то же или аналогичную проверку. Код библиотеки не менялся.
2. **w̄ для R4.** У `CoupledRun` нет функции безразмерного поля. Определение w̄ = wD/(qa⁴), a — сторона по x, должно совпадать с R3.
3. **Нормативный режим R4** останавливается после s = 1 с NOT_QUALIFIED. Это не результат расчёта. Исследовательский режим включается только
   явно, и его результат — PROVISIONAL / NOT_QUALIFIED. PROVISIONAL_FIELD_STABILIZED не является сходимостью.
4. **Ошибки с частичной историей.** R3 при ошибке после s ≥ 1 сохраняет выполненные полушаги. Поле последнего полушага доступно, но
   `is_error = true` и `completed = false` — показывать его как результат нельзя.
5. **Сравнение GUI с эталоном:**
   - в той же среде — побитово с прямым вызовом API;
   - на другой платформе — по правилам `cases.json → comparison`.

## 5. Воспроизведение

```
uv run python scripts/gui_backend_cases.py list
uv run python scripts/gui_backend_cases.py run r3_square r4_n2_research --out scratch/gui_demo
uv run python scripts/gui_backend_cases.py run --all --out scratch/gui_demo --format npz
uv run python scripts/gui_backend_cases.py custom --N 3 --mode research --a 1.2 --b 1.8 --t 0.012 --E 2.1e11 --nu 0.3 \
    --q 5000 --nx 32 --ny 48 --out scratch/gui_demo
uv run python scripts/gui_backend_cases.py golden        # перегенерация docs/results/gui_backend/
uv run python -m pytest tests/cross/test_gui_backend_acceptance.py
```

## 6. Ограничения

- R3: ACCEPTED (D-043) в своей области; квалификация только на проверенных сетках; сходимость по s и моменты не утверждаются.
- R4: IN_PROGRESS; все результаты — PROVISIONAL / NOT_QUALIFIED; `dense` — только до n = 40.
- Эталоны получены той же библиотекой и не являются независимыми решениями. Межплатформенный численный критерий для полей R3 не принят.
- Проверка GUI через настоящий интерфейс не выполнялась.
