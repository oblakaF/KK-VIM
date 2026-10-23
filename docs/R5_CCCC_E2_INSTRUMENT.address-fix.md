# R5-V CCCC — инструмент Е-2: адресные исправления ИЕ-1…ИЕ-3 (QA)

Цели: G3.4, G4. Роль: QA. Основа: `review/r5-cccc-e2-instrument` @ `907de2266ce306a63bdfa00fc4c826358326a014`.
Ветка: `qa/r5-cccc-e2-instrument-address`. Рецензия: `docs/R5_CCCC_E2_INSTRUMENT.independent-review.md` (вердикт CHANGES_REQUIRED).

**Границы:**
- исправлены ИЕ-1…ИЕ-3, а также неблокирующие Н-1, Н-2, Н-4;
- Н-3 оставлено открытым: штатный путь ‖·‖_a и c_R относятся к Q3; в отчёте уточнена только формулировка «на два порядка»;
- подтверждённое Reviewer ядро не переписывалось: `field_exact`, `norm_h_sq`, `energy_sq`, `_hat_L`, `sqrt_interval`, двойные пути (iv),
  `swap_difference`;
- модуль Q1b (`_float_down` и др.), рабочий `kkvim`, B-C, постановка, спецификации и решения владельца не менялись;
- порогов нет, Q3 не запускался.

**Вердикт: READY_FOR_ADDRESS_REVIEW.**

## 1. Исправления

| № | Замечание | Исправление (`src/kkvim_reference/r5_cccc_e2.py`) | Тесты |
|---|---|---|---|
| **ИЕ-1** | П-3 не проверяется: X = 2⁻⁵⁴⁰, Y = 3·2⁻⁵⁴⁰ дают «4 нарушения ВОК-4, ВОК-5 False» без признака выхода за предпосылку | Функция `p3_premise`; порог 2⁻¹⁰²² — наименьшее нормальное binary64. **(а) Доказательство П-3:** все входы нормальные или 0, и для каждого ненулевого произведения e(Xᵢₖ) + e(Yⱼₖ) ≥ −918 ⇒ `P3_CONFIRMED`. Вывод: точное произведение кратно 2^{e(X)+e(Y)−104} ≥ 2⁻¹⁰²²; суммы и округления кратных G остаются кратными G (результат точен либо его ulp ≥ G). Значит, при любом порядке и с FMA ненулевые значения ≥ 2⁻¹⁰²². Собственный вывод QA, уровень Б. **(б) Точное нарушение ⇒ `OUTSIDE_P3`:** ненулевой субнормальный вход; ненулевой субнормальный ŵᵢⱼ; узел с единственным ненулевым произведением ниже 2⁻¹⁰²² — его операция имеет ненулевой субнормальный точный результат при любом пути. **(в) Иначе ⇒ `UNRESOLVED_PREMISES`**, с подсчётом в точной арифметике ненулевых произведений и возможных частичных сумм (все подмножества, N ≤ 3) ниже порога. Точные сравнения остаются наблюдением (`vok4_violations`, `vok4_max_ratio_up`, `vok5_holds`). Нормативные `vok4_status`, `vok5_status` равны HOLDS или VIOLATED только при `P3_CONFIRMED`, иначе — статусу предпосылки | `test_ie1_reviewer_counterexamples_are_outside_p3` (оба примера Reviewer); `test_ie1_subnormal_input_is_outside_p3`; `test_ie1_unprovable_absence_is_unresolved_not_violation`; `test_ie1_threshold_of_the_granularity_criterion`; `test_ie1_normal_half_step_data_are_confirmed` |
| **ИЕ-2** | Переполнение: исключения в (а) ŵ = inf; (б) частичная сумма; (в) (iv) в double; (г) ‖ŵ‖_a > DBL_MAX | **П-2:** конечное ŵ доказывает отсутствие переполнения при образовании (±∞ не исчезает, inf − inf = NaN). Не конечное ŵ ⇒ `OUTSIDE_P2`: e, ВОК-4, ВОК-5 и (iv) не вычисляются; записываются число не-конечных узлов и max\|w̃\| (вверх). `field_standard` вызывается под `np.errstate(all="ignore")`: подавляется только оповещение, арифметика вызова та же. **Границы double:** локальная `_down` — точная нижняя граница выше DBL_MAX ⇒ DBL_MAX; верхняя — `_float_up` (+∞). **(iv):** `_component_iv` — при шаге сетки в double 0 или не конечном путь не вычисляется (UNRESOLVED); inf или NaN записывается как наблюдение `'inf'`/`'nan'` с ошибкой UNRESOLVED; исключения путей (`ArithmeticError`, `ValueError`) — UNRESOLVED с типом. Точная часть (i) сохраняется | `test_ie2_overflow_of_the_field_is_outside_p2` (а, б); `test_ie2_overflow_only_in_component_iv_keeps_the_exact_measurement` (в); `test_ie2_exact_norm_above_dbl_max` (г); `test_ie2_grid_step_zero_in_double_skips_component_iv` |
| **ИЕ-3** | Область lₓ, l_y, D, nₓ, n_y не проверяется; complex молча усекается | `FieldInputs.__post_init__` **до вычислений**: lₓ, l_y, D — вещественные (int, float, numpy), не bool, не complex, не str, конечные, > 0 (NS 2.2), точно представимые в double; nₓ, n_y — `int` или `numpy.integer`, не bool, ≥ 3; X, Y — вид dtype `f`, `i` или `u`, точный перевод во float64 (иначе `ValueError`: complex, bool, object, str, целые > 2⁵³, неточный longdouble). Формы, N = 1…3 и конечность — как прежде | `test_ie3_parameter_domain` (16 случаев); `test_ie3_admissible_parameter_types`; `test_ie3_arrays_must_be_real_and_exact` (5 случаев) |
| Н-1 | BLAS сборки вместо времени выполнения | В `provenance` добавлены BLAS времени выполнения (через `threadpoolctl`, если установлен; иначе явно «unavailable»), переменные потоков, `os.cpu_count`, признаки CPU numpy и самопроверка FTZ/DAZ одного вызова 1 × 1 с оговоркой «не доказательство режима IEEE». Имя узла не записывается | `test_n1_runtime_provenance_without_host_name` |
| Н-2 | SHA-256 только в конце | sha256 в начале и в конце `measure`; при расхождении — `RuntimeError` | `test_n2_digest_is_checked_at_start_and_end` |
| Н-4 | мало значений в тесте `sqrt_interval` | 5·10⁻³²⁴, его квадрат, 2⁻²¹⁴⁸, (2⁻¹⁰²²)², DBL_MAX, DBL_MAX², 4·DBL_MAX², 3·5·10⁻³²⁴; ширина ≤ 2⁻⁶³ отн. | `test_n4_sqrt_interval_subnormal_and_large` |

## 2. Какие случаи что дают

| Статус | Случаи |
|---|---|
| **`OUTSIDE_P2`** | ŵ не конечно: X = Y = 10²⁰⁰ (произведение); строка (10³⁰⁸, 10³⁰⁸, −10³⁰⁸) × 𝟏 (частичная сумма при конечной точной 10³⁰⁸). ВОК-4, ВОК-5 и (iv) не вычисляются |
| **`OUTSIDE_P3`** | ненулевой субнормальный вход (например, 5·10⁻³²⁴ или 2⁻¹⁰³⁰); ненулевой субнормальный ŵᵢⱼ (2⁻⁵³⁰ × 2⁻⁵³⁰(1 + 2⁻⁴⁰)); узел с единственным ненулевым произведением ниже 2⁻¹⁰²² (2⁻⁵⁴⁰ × 3·2⁻⁵⁴⁰ → 0) |
| **`UNRESOLVED_PREMISES`** | П-3 не доказуема критерием (e(X) + e(Y) < −918, например 2⁻⁵⁰⁰ × 2⁻⁵⁰⁰ или 2⁻⁴⁵⁹ × 2⁻⁴⁶⁰), а точного нарушения нет |
| **`UNRESOLVED`** (величины) | (iv) при шаге сетки 0 или ∞ в double, при переполнении или NaN пути, при исключении пути; `w_a_sq_double_bound` — всегда (ВОК-3 — модель); отношения с нулевым знаменателем (нулевое поле: `max_rel_e`, `rel_e_a`, `kappa_W`, `eps_a_C_rel`, `e_a_over_eps_a_C`) |
| **`P3_CONFIRMED`**, HOLDS | реальные полушаги B-C (N = 1–3), отрицательные значения, уничтожение, поля ~10¹⁵⁰ (переполнение только в (iv)), поля с точной нормой выше DBL_MAX (интервал [DBL_MAX, +∞]) |

## 3. Проверки

| Проверка | Результат |
|---|---|
| `tests/reference/test_r5_cccc_e2.py` | 59 passed (20 исходных без изменений + 39 адресных) |
| полный `pytest`, Python 3.13.15 | 3378 passed, 31 skipped |
| полный `pytest`, Python 3.14.8 (отдельное окружение) | 3378 passed, 31 skipped |
| `uv lock --check`, `git diff --check`, самотест hook | без изменений; чисто; пройден |

## 4. Не подтверждено

- Критерий П-3 консервативен: `UNRESOLVED_PREMISES` возможен там, где субнормальных операций фактически нет. Точный разбор путей BLAS
  не выполняется, потому что порядок операций ядра неизвестен.
- Строгая граница (iv) — UNRESOLVED (ВОК-3, c_R; Н-3 открыт).
- BLAS времени выполнения на этой машине не определяется (`threadpoolctl` не установлен, зависимость не добавлялась).
- Эталон повышенной точности для больших сеток (ОК-7) — вне задания.

## 5. Отчёт агента (AGENTS.md, разд. 5)

1. **Задача:** ИЕ-1…ИЕ-3 (и Н-1, Н-2, Н-4) по рецензии `907de22`; цели G3.4, G4.
2. **Изменено:**
   - `src/kkvim_reference/r5_cccc_e2.py`: проверка входов, `p3_premise`, статусы, `_down`, `_component_iv`, `_check_digest`, `_runtime`;
   - `tests/reference/test_r5_cccc_e2.py`: 39 новых тестов, 20 исходных не менялись;
   - `docs/R5_CCCC_E2_INSTRUMENT_PREP.report.md`;
   - этот файл.
3. **Источники:** рецензия `907de22`; RO @ `96914c9`, 0.4 (П-1…П-5, уточнение по З-3), ВОК-4, ВОК-5; IEEE 754 binary64 (наименьшее
   нормальное 2⁻¹⁰²²); NS 2.2.
4. **Проверено:** разд. 3.
5. **Не подтверждено:** разд. 4.
6. **Следующий шаг:** адресная сверка Reviewer ИЕ-1…ИЕ-3.

## 6. Закрытие ИЕ-3а и уточнений Н-а, Н-б, Н-в (по заключению `f5f85e1`)

Основа: `review/r5-cccc-e2-final-address` @ `f5f85e15a60225ec8f5335b53f190e6aa5aa5445`. Ветка: `qa/r5-cccc-e2-ie3a-address`.
Менялись **только проверки входных данных, docstring, тесты и этот документ**. Не менялись:
- точное UVᵀ, рациональные нормы и интервалы;
- П-2/П-3;
- тесты и модули Q1b и B-C;
- `kkvim`, постановка и спецификации.

| № | Исправление | Тесты |
|---|---|---|
| **ИЕ-3а** | `_positive_real`: после `f = float(v)` и проверки конечности и знака точное значение аргумента сравнивается с `Fraction.from_float(f)`. Точное значение: `Fraction(int(v))` для `int`/`numpy.integer`, `Fraction(*v.as_integer_ratio())` для `float`/`numpy.floating`, включая `numpy.longdouble`. Несовпадение — `ValueError` («not exactly representable as a double (binary64)»); если точное значение определить нельзя — `ValueError` («cannot be determined»). Область допустимых параметров та же: lₓ, l_y, D конечны, > 0 и точно представимы в binary64 | `test_ie3a_extended_longdouble_is_refused` (1 + 2⁻⁶⁰ в long double; выполняется, если `np.finfo(np.longdouble).nmant > 52`, иначе пропуск с объяснением); `test_ie3a_extended_longdouble_out_of_range_or_inexact_is_refused` (2¹¹⁰⁰, 2⁻¹¹⁰⁰, 1/3; то же условие); `test_ie3a_logic_on_every_platform` (подкласс `float` с более широким точным значением и с неопределимым значением — проверка логики на любой платформе); `test_ie3a_exactly_representable_values_are_accepted` (float, float16/32/64, longdouble с binary64-значением, int, numpy-целые, 2⁵³; 30 случаев); `test_ie3a_inexact_integers_are_refused` (2⁵³ + 1, 10⁴⁰⁰, uint64 2⁶³ + 1, int64 2⁶² + 1); `test_ie3a_domain_checks_still_hold` |
| Н-а | Docstring: субнормальный вход классифицируется `OUTSIDE_P3` как **консервативное ограничение области**, даже когда субнормальной операции фактически не было (например, против нулей другого множителя) | — (редакционное) |
| Н-б | Docstring: вывод «конечное ŵ ⇒ нет переполнения» опирается на П-1 (округление к ближайшему); при округлении к нулю переполнение дало бы ±DBL_MAX | — (редакционное) |
| Н-в | `_real_array` отклоняет `numpy.ma.MaskedArray` (`ValueError`): маска иначе терялась бы молча | `test_nc_masked_arrays_are_refused` (X, Y, массив без замаскированных элементов) |

**Платформы.**
- Здесь (Windows, numpy 2.5.4) `np.finfo(np.longdouble).nmant = 52`: long double совпадает с binary64, и 4 теста расширенной точности
  пропускаются с объяснением.
- Логика ИЕ-3а проверяется на любой платформе тестом с подклассом `float`.
- В CI матрица включает `ubuntu-latest` (x86-64, long double 80 бит, nmant = 63): там тесты расширенной точности выполняются.

**Проверки.**

| Проверка | Результат |
|---|---|
| `tests/reference/test_r5_cccc_e2.py` | 100 passed, 4 skipped (59 прежних + 45 новых случаев; пропуски — только расширенный long double на этой платформе) |
| полный `pytest`, Python 3.13.15 | 3419 passed, 35 skipped |
| полный `pytest`, Python 3.14.8 (отдельное окружение) | 3419 passed, 35 skipped |
| `uv lock --check`, `git diff --check`, самотест hook | без изменений; чисто; пройден |

**Вердикт: READY_FOR_FINAL_CONFIRMATION.** Следующий шаг — короткая сверка Reviewer только ИЕ-3а и Н-а…Н-в.
