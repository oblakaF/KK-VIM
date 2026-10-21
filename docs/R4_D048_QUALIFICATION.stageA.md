# R4 D-048 — этап A: адаптация QA-тестов к автоматической остановке и воспроизводимость

Цели: G2, G3.3, G4. Роль: QA.
Проверяемая реализация: Builder `builder/r4-d048-implementation` @ `3da624a74f324ebdb2129ffe4ac3727766bbe7a0`.
План QA К-1…К-9 включён через `merge --no-ff` (`b7dc9fc`; план `e4e1109`, правка обозначений `cb63c16`).

Это **не** квалификация К-1…К-9: здесь только восстановлены прежние проверки. Квалификация — отдельный этап B и отдельный отчёт.

## 1. Причина падений

D-048.3 ввёл автоматическую остановку в исследовательском режиме (PROVISIONAL_FIELD_STABILIZED при s_stop = 9…12). Тесты, которым нужно поле
на фиксированных s = 12, 20, 40, получали меньше полушагов.

| Тест | Случаев (параметризаций) | Падало |
|---|---|---|
| `test_r4_controller_comparison.py::test_fields_energies_and_singular_values_against_the_qa_control` | 56 | 56 |
| `test_r4_controller_comparison.py::test_harness_reproduces_run_coupled_bitwise` | 1 | 1 |
| `test_r4_integrated_solver.py::test_merged_solver_against_the_qa_control_and_the_group_b_tables` | 20 | 12 |

Всего 69 падений. Других падений в `tests/cross` и полном pytest не было.

## 2. Исправление

Диагностический режим дополнения к D-048: `research_mode=True, stop_on_stabilization=False`. Он передаётся явно во всех вызовах, которым нужна
фиксированная длина последовательности, и сопровождается проверкой `len(run.steps) == expected_s`.

| Тест | Вызов |
|---|---|
| три функции выше | s = 40 (S_LIST), 12; для s = 1, 2, 3, 10 — длина уже проверялась |
| `test_independent_energy_quantities_against_the_builder_log` | 20 полушагов. Раньше `zip(run.steps[1:], rec[1:])` при ранней остановке молча проверял бы меньше шагов; теперь `len == 20` |
| `test_eta_n_recomputed_independently` | 6 |
| `test_qr_keeps_field_and_energy` | 10 |
| `test_negative_energy_changes_are_unresolved_and_within_the_rounding_model` | 40 |
| `test_high_precision_accumulated_field_error` | 6 |
| `test_a4_12_reflections_and_scaling` | 10 (обе сравниваемые последовательности) |

**Не менялось:**
- тесты, параметризации, сравниваемые величины, допуски и модели;
- проверка статуса `RESEARCH_LIMIT_REACHED` там, где требуется поле на s = 20 и 40.

Тесты, которым фиксированная длина не нужна (семантика статусов, инъекция E1, S-1, нормативный режим), остаются в режиме по умолчанию.
Новых `skip` и `xfail` нет.

QA-скрипт `scripts/r4_numerical_prequal.py`: тот же явный режим в обвязке (6 и 8 полушагов), B5 (4), B6 (40) и прогоне до s_stop для дрейфа.
Другие скрипты `run_coupled` не используют.

## 3. Результат

- Три функции: 56 + 1 + 20 = 77 параметризаций — **77 passed**. Число параметризаций то же; сравнения полей, энергий и σ при s = 12, 20, 40
  выполняются при длине, подтверждённой явно.
- `tests/cross`: 1440 passed, 30 skipped. До адаптации было 1371 passed, 69 failed, 30 skipped; пропуски те же — `mpmath` и конвейер группы B.
- Полный pytest, hook, `uv lock --check`, `git diff --check` и CI — в сообщении QA.

## 4. Воспроизведение предквалификации `e1b69dd` во временный каталог

`uv run --with mpmath python scripts/r4_numerical_prequal.py --out scratch/repro_prequal` на реализации `3da624a`. Исторические CSV и отчёт
не перезаписывались.

| Таблица | Строк | Различающихся ячеек | Комментарий |
|---|---|---|---|
| b1_delta_f, b2_modal, b4_rank, b5_energy, b6_stopping, b6_summary, b8_zeta, b9_rank_margins | все | **0** | побитово |
| b3_dst_fast | 18 | 18 — только 9 строк `dense` | строки `fft` побитово равны |
| b3_dense_elements | 4 | 12 | ожидаемо: `dense` исправлен по D-048.6 |

Плотная Ψ после исправления D-048.6:
- ошибка элемента / √(2/n) снизилась с 1,6·10⁻¹⁴…6,8·10⁻¹³ до ≤ 1,75·10⁻¹⁶ (n = 40…1280);
- ошибка импульсов при n = 1280 снизилась с ≤ 1,7·10³·u‖x‖ до ≤ 0,63·u‖x‖;
- на гармониках — до ≤ 22·u‖x‖ (это наблюдение; модель проверяется в К-6).

Расчёт B6 до s = 40 и дрейф после s_stop воспроизведены побитово.
