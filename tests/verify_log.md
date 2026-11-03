# Verification log

Each section is one full run of `python scripts/verify_all.py`. SUCCESS = two consecutive GREEN runs that rebuilt from raw.

## Iteration 1 - 2026-11-03 07:05:19Z - RED

Rebuilt from raw: yes · build 36.6s · 156 tests in 231.2s · failed tests: 4 · real LLM API calls: 0

PASS 75 / 82 automated items · MANUAL 9

| item | status | detail |
|---|---|---|
| 1 | PASS |  |
| 2 | PASS |  |
| 3 | PASS |  |
| 4 | FAIL | tests/test_e_dashboard.py::test_every_view_loads_without_exception[Ask the Portfolio] [failed] |
| 5 | PASS |  |
| 6 | PASS |  |
| 7 | PASS |  |
| 8 | PASS |  |
| 9 | PASS |  |
| 10 | PASS |  |
| 11 | PASS |  |
| 12 | PASS |  |
| 13 | MANUAL |  |
| 14 | MANUAL |  |
| 15 | PASS |  |
| 16 | PASS |  |
| 17 | FAIL | tests/test_a_repo.py::test_fresh_virtualenv_runs_only_readme_commands_and_every_view_loads_in_browser [failed] |
| 18 | PASS |  |
| 19 | PASS |  |
| 20 | PASS |  |
| 21 | PASS |  |
| 22 | PASS |  |
| 23 | PASS |  |
| 24 | PASS |  |
| 25 | PASS |  |
| 26 | FAIL | tests/test_a_repo.py::test_fresh_virtualenv_runs_only_readme_commands_and_every_view_loads_in_browser [failed] |
| 27 | PASS |  |
| 28 | PASS |  |
| 29 | PASS |  |
| 30 | PASS |  |
| 31 | PASS |  |
| 32 | PASS |  |
| 33 | PASS |  |
| 34 | PASS |  |
| 35 | PASS |  |
| 36 | PASS |  |
| 37 | PASS |  |
| 38 | PASS |  |
| 39 | PASS |  |
| 40 | PASS |  |
| 41 | PASS |  |
| 42 | PASS |  |
| 43 | PASS |  |
| 44 | FAIL | tests/test_e_dashboard.py::test_account_view_shows_score_and_key_drivers_matching_model_output [failed] |
| 45 | PASS |  |
| 46 | PASS |  |
| 47 | PASS |  |
| 48 | FAIL | tests/test_e_dashboard.py::test_every_view_loads_without_exception[Ask the Portfolio] [failed] |
| 49 | PASS |  |
| 50 | FAIL | tests/test_a_repo.py::test_fresh_virtualenv_runs_only_readme_commands_and_every_view_loads_in_browser [failed] |
| 51 | MANUAL |  |
| 52 | MANUAL |  |
| 53 | PASS |  |
| 54 | PASS |  |
| 55 | PASS |  |
| 56 | PASS |  |
| 57 | PASS |  |
| 58 | PASS |  |
| 59 | PASS |  |
| 60 | PASS |  |
| 61 | PASS |  |
| 62 | PASS |  |
| 63 | PASS |  |
| 64 | PASS |  |
| 65 | PASS |  |
| 66 | PASS |  |
| 67 | PASS |  |
| 68 | PASS |  |
| 69 | PASS |  |
| 70 | PASS |  |
| 71 | MANUAL |  |
| 72 | PASS |  |
| 73 | PASS |  |
| 74 | PASS |  |
| 75 | PASS |  |
| 76 | PASS |  |
| 77 | PASS |  |
| 78 | PASS |  |
| 79 | PASS |  |
| 80 | MANUAL |  |
| 81 | MANUAL |  |
| 82 | MANUAL |  |
| 83 | PASS |  |
| 84 | PASS |  |
| 85 | FAIL | tests/test_a_repo.py::test_fresh_virtualenv_runs_only_readme_commands_and_every_view_loads_in_browser [failed] |
| 86 | PASS |  |
| 87 | PASS |  |
| 88 | PASS |  |
| 89 | MANUAL |  |
| 90 | PASS |  |
| 91 | PASS |  |

Failed tests:
- `tests/test_a_repo.py::test_fresh_virtualenv_runs_only_readme_commands_and_every_view_loads_in_browser` (failed)
- `tests/test_e_dashboard.py::test_every_view_loads_without_exception[Ask the Portfolio]` (failed)
- `tests/test_e_dashboard.py::test_account_view_shows_score_and_key_drivers_matching_model_output` (failed)
- `tests/test_k_llm.py::test_manual_checks_document_is_complete` (failed)

## Iteration 2 - 2026-11-03 07:10:41Z - GREEN

Rebuilt from raw: yes · build 34.5s · 156 tests in 325.0s · failed tests: 0 · real LLM API calls: 0

PASS 82 / 82 automated items · MANUAL 9

| item | status | detail |
|---|---|---|
| 1 | PASS |  |
| 2 | PASS |  |
| 3 | PASS |  |
| 4 | PASS |  |
| 5 | PASS |  |
| 6 | PASS |  |
| 7 | PASS |  |
| 8 | PASS |  |
| 9 | PASS |  |
| 10 | PASS |  |
| 11 | PASS |  |
| 12 | PASS |  |
| 13 | MANUAL |  |
| 14 | MANUAL |  |
| 15 | PASS |  |
| 16 | PASS |  |
| 17 | PASS |  |
| 18 | PASS |  |
| 19 | PASS |  |
| 20 | PASS |  |
| 21 | PASS |  |
| 22 | PASS |  |
| 23 | PASS |  |
| 24 | PASS |  |
| 25 | PASS |  |
| 26 | PASS |  |
| 27 | PASS |  |
| 28 | PASS |  |
| 29 | PASS |  |
| 30 | PASS |  |
| 31 | PASS |  |
| 32 | PASS |  |
| 33 | PASS |  |
| 34 | PASS |  |
| 35 | PASS |  |
| 36 | PASS |  |
| 37 | PASS |  |
| 38 | PASS |  |
| 39 | PASS |  |
| 40 | PASS |  |
| 41 | PASS |  |
| 42 | PASS |  |
| 43 | PASS |  |
| 44 | PASS |  |
| 45 | PASS |  |
| 46 | PASS |  |
| 47 | PASS |  |
| 48 | PASS |  |
| 49 | PASS |  |
| 50 | PASS |  |
| 51 | MANUAL |  |
| 52 | MANUAL |  |
| 53 | PASS |  |
| 54 | PASS |  |
| 55 | PASS |  |
| 56 | PASS |  |
| 57 | PASS |  |
| 58 | PASS |  |
| 59 | PASS |  |
| 60 | PASS |  |
| 61 | PASS |  |
| 62 | PASS |  |
| 63 | PASS |  |
| 64 | PASS |  |
| 65 | PASS |  |
| 66 | PASS |  |
| 67 | PASS |  |
| 68 | PASS |  |
| 69 | PASS |  |
| 70 | PASS |  |
| 71 | MANUAL |  |
| 72 | PASS |  |
| 73 | PASS |  |
| 74 | PASS |  |
| 75 | PASS |  |
| 76 | PASS |  |
| 77 | PASS |  |
| 78 | PASS |  |
| 79 | PASS |  |
| 80 | MANUAL |  |
| 81 | MANUAL |  |
| 82 | MANUAL |  |
| 83 | PASS |  |
| 84 | PASS |  |
| 85 | PASS |  |
| 86 | PASS |  |
| 87 | PASS |  |
| 88 | PASS |  |
| 89 | MANUAL |  |
| 90 | PASS |  |
| 91 | PASS |  |

## Iteration 3 - 2026-11-03 07:24:04Z - RED

Rebuilt from raw: yes · build 37.0s · 156 tests in 375.8s · failed tests: 1 · real LLM API calls: 0

PASS 78 / 82 automated items · MANUAL 9

| item | status | detail |
|---|---|---|
| 1 | PASS |  |
| 2 | PASS |  |
| 3 | PASS |  |
| 4 | PASS |  |
| 5 | PASS |  |
| 6 | PASS |  |
| 7 | PASS |  |
| 8 | PASS |  |
| 9 | PASS |  |
| 10 | PASS |  |
| 11 | PASS |  |
| 12 | PASS |  |
| 13 | MANUAL |  |
| 14 | MANUAL |  |
| 15 | PASS |  |
| 16 | PASS |  |
| 17 | FAIL | tests/test_a_repo.py::test_fresh_virtualenv_runs_only_readme_commands_and_every_view_loads_in_browser [failed] |
| 18 | PASS |  |
| 19 | PASS |  |
| 20 | PASS |  |
| 21 | PASS |  |
| 22 | PASS |  |
| 23 | PASS |  |
| 24 | PASS |  |
| 25 | PASS |  |
| 26 | FAIL | tests/test_a_repo.py::test_fresh_virtualenv_runs_only_readme_commands_and_every_view_loads_in_browser [failed] |
| 27 | PASS |  |
| 28 | PASS |  |
| 29 | PASS |  |
| 30 | PASS |  |
| 31 | PASS |  |
| 32 | PASS |  |
| 33 | PASS |  |
| 34 | PASS |  |
| 35 | PASS |  |
| 36 | PASS |  |
| 37 | PASS |  |
| 38 | PASS |  |
| 39 | PASS |  |
| 40 | PASS |  |
| 41 | PASS |  |
| 42 | PASS |  |
| 43 | PASS |  |
| 44 | PASS |  |
| 45 | PASS |  |
| 46 | PASS |  |
| 47 | PASS |  |
| 48 | PASS |  |
| 49 | PASS |  |
| 50 | FAIL | tests/test_a_repo.py::test_fresh_virtualenv_runs_only_readme_commands_and_every_view_loads_in_browser [failed] |
| 51 | MANUAL |  |
| 52 | MANUAL |  |
| 53 | PASS |  |
| 54 | PASS |  |
| 55 | PASS |  |
| 56 | PASS |  |
| 57 | PASS |  |
| 58 | PASS |  |
| 59 | PASS |  |
| 60 | PASS |  |
| 61 | PASS |  |
| 62 | PASS |  |
| 63 | PASS |  |
| 64 | PASS |  |
| 65 | PASS |  |
| 66 | PASS |  |
| 67 | PASS |  |
| 68 | PASS |  |
| 69 | PASS |  |
| 70 | PASS |  |
| 71 | MANUAL |  |
| 72 | PASS |  |
| 73 | PASS |  |
| 74 | PASS |  |
| 75 | PASS |  |
| 76 | PASS |  |
| 77 | PASS |  |
| 78 | PASS |  |
| 79 | PASS |  |
| 80 | MANUAL |  |
| 81 | MANUAL |  |
| 82 | MANUAL |  |
| 83 | PASS |  |
| 84 | PASS |  |
| 85 | FAIL | tests/test_a_repo.py::test_fresh_virtualenv_runs_only_readme_commands_and_every_view_loads_in_browser [failed] |
| 86 | PASS |  |
| 87 | PASS |  |
| 88 | PASS |  |
| 89 | MANUAL |  |
| 90 | PASS |  |
| 91 | PASS |  |

Failed tests:
- `tests/test_a_repo.py::test_fresh_virtualenv_runs_only_readme_commands_and_every_view_loads_in_browser` (failed)

## Iteration 4 - 2026-11-03 07:43:41Z - GREEN

Rebuilt from raw: yes · build 35.3s · 156 tests in 314.8s · failed tests: 0 · real LLM API calls: 0

PASS 82 / 82 automated items · MANUAL 9

| item | status | detail |
|---|---|---|
| 1 | PASS |  |
| 2 | PASS |  |
| 3 | PASS |  |
| 4 | PASS |  |
| 5 | PASS |  |
| 6 | PASS |  |
| 7 | PASS |  |
| 8 | PASS |  |
| 9 | PASS |  |
| 10 | PASS |  |
| 11 | PASS |  |
| 12 | PASS |  |
| 13 | MANUAL |  |
| 14 | MANUAL |  |
| 15 | PASS |  |
| 16 | PASS |  |
| 17 | PASS |  |
| 18 | PASS |  |
| 19 | PASS |  |
| 20 | PASS |  |
| 21 | PASS |  |
| 22 | PASS |  |
| 23 | PASS |  |
| 24 | PASS |  |
| 25 | PASS |  |
| 26 | PASS |  |
| 27 | PASS |  |
| 28 | PASS |  |
| 29 | PASS |  |
| 30 | PASS |  |
| 31 | PASS |  |
| 32 | PASS |  |
| 33 | PASS |  |
| 34 | PASS |  |
| 35 | PASS |  |
| 36 | PASS |  |
| 37 | PASS |  |
| 38 | PASS |  |
| 39 | PASS |  |
| 40 | PASS |  |
| 41 | PASS |  |
| 42 | PASS |  |
| 43 | PASS |  |
| 44 | PASS |  |
| 45 | PASS |  |
| 46 | PASS |  |
| 47 | PASS |  |
| 48 | PASS |  |
| 49 | PASS |  |
| 50 | PASS |  |
| 51 | MANUAL |  |
| 52 | MANUAL |  |
| 53 | PASS |  |
| 54 | PASS |  |
| 55 | PASS |  |
| 56 | PASS |  |
| 57 | PASS |  |
| 58 | PASS |  |
| 59 | PASS |  |
| 60 | PASS |  |
| 61 | PASS |  |
| 62 | PASS |  |
| 63 | PASS |  |
| 64 | PASS |  |
| 65 | PASS |  |
| 66 | PASS |  |
| 67 | PASS |  |
| 68 | PASS |  |
| 69 | PASS |  |
| 70 | PASS |  |
| 71 | MANUAL |  |
| 72 | PASS |  |
| 73 | PASS |  |
| 74 | PASS |  |
| 75 | PASS |  |
| 76 | PASS |  |
| 77 | PASS |  |
| 78 | PASS |  |
| 79 | PASS |  |
| 80 | MANUAL |  |
| 81 | MANUAL |  |
| 82 | MANUAL |  |
| 83 | PASS |  |
| 84 | PASS |  |
| 85 | PASS |  |
| 86 | PASS |  |
| 87 | PASS |  |
| 88 | PASS |  |
| 89 | MANUAL |  |
| 90 | PASS |  |
| 91 | PASS |  |

## Iteration 5 - 2026-11-03 07:54:58Z - GREEN

Rebuilt from raw: yes · build 39.4s · 156 tests in 308.0s · failed tests: 0 · real LLM API calls: 0

PASS 82 / 82 automated items · MANUAL 9

| item | status | detail |
|---|---|---|
| 1 | PASS |  |
| 2 | PASS |  |
| 3 | PASS |  |
| 4 | PASS |  |
| 5 | PASS |  |
| 6 | PASS |  |
| 7 | PASS |  |
| 8 | PASS |  |
| 9 | PASS |  |
| 10 | PASS |  |
| 11 | PASS |  |
| 12 | PASS |  |
| 13 | MANUAL |  |
| 14 | MANUAL |  |
| 15 | PASS |  |
| 16 | PASS |  |
| 17 | PASS |  |
| 18 | PASS |  |
| 19 | PASS |  |
| 20 | PASS |  |
| 21 | PASS |  |
| 22 | PASS |  |
| 23 | PASS |  |
| 24 | PASS |  |
| 25 | PASS |  |
| 26 | PASS |  |
| 27 | PASS |  |
| 28 | PASS |  |
| 29 | PASS |  |
| 30 | PASS |  |
| 31 | PASS |  |
| 32 | PASS |  |
| 33 | PASS |  |
| 34 | PASS |  |
| 35 | PASS |  |
| 36 | PASS |  |
| 37 | PASS |  |
| 38 | PASS |  |
| 39 | PASS |  |
| 40 | PASS |  |
| 41 | PASS |  |
| 42 | PASS |  |
| 43 | PASS |  |
| 44 | PASS |  |
| 45 | PASS |  |
| 46 | PASS |  |
| 47 | PASS |  |
| 48 | PASS |  |
| 49 | PASS |  |
| 50 | PASS |  |
| 51 | MANUAL |  |
| 52 | MANUAL |  |
| 53 | PASS |  |
| 54 | PASS |  |
| 55 | PASS |  |
| 56 | PASS |  |
| 57 | PASS |  |
| 58 | PASS |  |
| 59 | PASS |  |
| 60 | PASS |  |
| 61 | PASS |  |
| 62 | PASS |  |
| 63 | PASS |  |
| 64 | PASS |  |
| 65 | PASS |  |
| 66 | PASS |  |
| 67 | PASS |  |
| 68 | PASS |  |
| 69 | PASS |  |
| 70 | PASS |  |
| 71 | MANUAL |  |
| 72 | PASS |  |
| 73 | PASS |  |
| 74 | PASS |  |
| 75 | PASS |  |
| 76 | PASS |  |
| 77 | PASS |  |
| 78 | PASS |  |
| 79 | PASS |  |
| 80 | MANUAL |  |
| 81 | MANUAL |  |
| 82 | MANUAL |  |
| 83 | PASS |  |
| 84 | PASS |  |
| 85 | PASS |  |
| 86 | PASS |  |
| 87 | PASS |  |
| 88 | PASS |  |
| 89 | MANUAL |  |
| 90 | PASS |  |
| 91 | PASS |  |
