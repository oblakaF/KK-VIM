# R4 D-048 — отчёт о технической интеграции

Роль: REVIEWER в функции Integrator. Цели: G2, G3.3, G4. Дата: 2026-10-19.

- **Ветка:** `integration/r4-d048-acceptance`, worktree `D:\git\Test_Versions\KK-VIM-r4-d048-accept`.
- **Ограничения:** merge в `main` и PR не выполнялись; исходные ветки Builder, QA и Reviewer не изменялись.

## 1. Базовый SHA и слияния

| Шаг | SHA | Содержание |
|---|---|---|
| база | `8c4dbe1f7ac5d9a8bd71d227139459a90544089a` | `main` (PR #22) |
| merge 1 (`--no-ff`) | `8534184` | `qa/r4-d048-qualification` @ `5b809259ec881f5f283eda15e2dfc2df6679fdbe`; включает Builder `00cf453` через `ff75751` |
| merge 2 (`--no-ff`) | `b2a1a82f14efeda420b1aedd7080993cfdab9a93` | `review/r4-d048-address` @ `cd2426c7eb56d3a6d83fea5c74b135fa418bc361` (рецензии `782e4ab` и адресная) |

- Конфликтов не было.
- В истории HEAD присутствуют `00cf453`, `5b80925`, `cd2426c`, `782e4ab`, `ff75751`.
- Squash, rebase, cherry-pick и force push не использовались.

## 2. Состав вошедших изменений (относительно `8c4dbe1`)

- **Builder (D-048, Б-1…Б-6, диагностический режим, исправления Д-1/К-3.3):**
  - `src/kkvim/coupled.py`, `src/kkvim/coupled_run.py` — **тождественны `00cf453`** (`git diff 00cf453 HEAD -- src` пуст);
  - `tests/kkvim/test_coupled_d048.py`, `tests/kkvim/test_coupled_run.py`;
  - датированное дополнение к D-048 в `docs/DECISIONS.md` (Builder `3da624a`). Интегратор его не менял. По тексту дополнения окончательное включение в `main` — после проверки владельцем.
- **QA:**
  - план, отчёт К-1…К-9 (`7bb60e7`), этап A, адресная перепроверка (`5b80925`);
  - `docs/results/r4_d048_qualification/`, `scripts/r4_d048_qualification.py`, `tests/cross/test_r4_d048_qualification.py`;
  - адаптация `tests/cross/test_r4_controller_comparison.py` и `test_r4_integrated_solver.py`;
  - редакционное переобозначение Λ_mn в `docs/R4_GROUP_B_ACCURACY.qa-report.md` (`cb63c16`; формулы и числа не менялись);
  - правка `scripts/r4_numerical_prequal.py`.

  Тесты и скрипты HEAD тождественны `5b80925`.
- **Reviewer:** `docs/R4_D048_IMPLEMENTATION.review.md` (`782e4ab`, CHANGES_REQUIRED — исторический), `docs/R4_D048_IMPLEMENTATION.address-review.md` (`cd2426c`).
- **Исторические документы не переписаны.** `R4_D048_QUALIFICATION.qa-report.md` с FAIL К-3 не менялся после `7bb60e7`; рецензия `782e4ab` вошла без изменений.

## 3. Замечания Д-1 и К-3.3

- **Д-1 — CLOSED:** адресная рецензия `cd2426c` и повторная проверка QA `5b80925`, § 2. Размеры сетки нормализуются к `int`; шесть сеток семейства II с numpy-целыми побитово равны `int`; допуск 4u и список сеток не изменены.
- **К-3.3 — CLOSED:** `5b80925`, § 1 и `cd2426c`, § 3. Флаг и причина обязательного UNRESOLVED записаны в журнале s = 2 и согласованы с `CoupledRun`. Тест К-3.3 выполняется **без** `xfail`: в `tests/` нет ни одной отметки xfail.

## 4. Итог К-1…К-9 (QA `7bb60e7` + адресная перепроверка `5b80925`)

| К | Итог |
|---|---|
| К-1 | PASS — путь `fft` и N = 1 побитово равны базе |
| К-2 | PASS в области D-048.3 |
| К-3 | PASS — критерии 1, 2, 4, 5; критерий 3 после исправления К-3.3 |
| К-4 | PASS |
| К-5 | PASS |
| К-6 | PASS на локальной платформе; на CI-платформах — по выводу теста К-6.0 |
| К-7 | PASS |
| К-8 | PASS — CONVERGED отсутствует; группа B побитово; Q-022 OPEN |
| К-9 | PASS — эталоны без функций Builder |

## 5. Проверки объединённого дерева (локально, Windows 10)

| Проверка | Результат |
|---|---|
| полный pytest, Python 3.13.15 | **3064 passed, 31 skipped**; 0 failed, 0 xfailed, 0 XPASS |
| полный pytest, Python 3.14.8 | **3064 passed, 31 skipped**; 0 failed, 0 xfailed, 0 XPASS |
| `tests/cross` (Python 3.14) | 1469 passed, 30 skipped |
| `uv lock --check` | OK (9 packages) |
| самотест pre-commit hook | ALL PASSED |
| `git diff --check 8c4dbe1..HEAD` | чисто |
| skip-маркеры в `tests/` | 8 в `main` и 8 в HEAD — **новых skip нет** |
| адаптированные QA-проверки (69) | сохранены: добавлены только явный `stop_on_stabilization=False` и проверки длины; допуски не изменены |
| побитовая регрессия `fft` относительно `main` (собственный скрипт рецензента) | нормативный режим 16/16 прогонов; исследовательский режим `main` против фиксированного режима HEAD — 8/8 |
| пороги D-048 в коде | tol_∞ = 10⁻¹⁰, tol_E = 10⁻⁸, s_min = 3, s_max = 100, η_max = 10⁻¹², K_RESEARCH = 10⁶; обязательные сетки D-048.1 и допуск 4u — как в `00cf453` |

Число тестов совпадает с ориентиром повторной проверки QA (3064 / 31).

## 6. CI

Прогон GitHub Actions **CI #39229566778** для `b2a1a82` (оба слияния): https://github.com/oblakaF/KK-VIM/actions/runs/39229566778 — **SUCCESS**.

| Задача | Результат |
|---|---|
| Pytest (ubuntu-latest, Python 3.13) | success |
| Pytest (ubuntu-latest, Python 3.14) | success |
| Pytest (windows-latest, Python 3.13) | success |
| Pytest (windows-latest, Python 3.14) | success |
| Pre-commit hook self-test | success |

Коммит этого отчёта меняет только этот документ; его CI-прогон указан в сообщении интегратора.

## 7. Сохранённые научные ограничения

- R4 — **IN_PROGRESS**; научной приёмки нет.
- Q-020, Q-022, Q-025 — **OPEN**.
- Группа B: `status_cont_*` при s ≥ 2 — **PROVISIONAL**.
- c_DST — **NOT_QUALIFIED**.
- n = 320 (семейство I) — вне принятой обязательной области s = 2.
- Остановка при n = 1280 — наблюдение, не полная квалификация.
- D-048 — ограниченная инженерная квалификация реализации N = 2–3. Это не доказательство точности непрерывного КК-МВИ.
- STATUS, принятые спецификации и численные критерии интегратором не изменялись.

## 8. Вердикт

**INTEGRATION_PASS.**
- Оба слияния выполнены без конфликтов; код тождественен Builder `00cf453`, тесты — QA `5b80925`.
- Д-1 и К-3.3 закрыты; К-1…К-9 — PASS.
- Локальные проверки Python 3.13 и 3.14: 3064 passed, 31 skipped, 0 failed / xfailed / XPASS.
- CI на четырёх конфигурациях — SUCCESS. Новых регрессий нет.

Ветка подготовлена для передачи независимому Auditor. Merge в `main` — только по отдельному разрешению владельца. До него владельцу также надлежит подтвердить датированное дополнение к D-048 (разд. 2).
