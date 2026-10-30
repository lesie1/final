# Session summary: typeb_v3.2_screen0.02/session

- Domain `memory`, status `stopped`, wall clock 0.94 h
- Baseline `trial_001`, final best `trial_001`, accepted none
- Agent calls 47 {'code': 12, 'feature': 7, 'analysis': 28}, tokens 888,651
- Manual interventions 0, agent decisions 16, controller fallbacks 2, response repairs 14
- Automation (manual = budget changes only): step coverage 100.00% (62 automatic / 0 manual steps), trial coverage 100.00% (12/12), full pipeline automated: yes

| trial | status | accepted | P(better) | selection F1 | CV F1 | CV TP/FP/FN |
|---|---|---|---|---|---|---|
| trial_001 | complete | True |  | 37.50% | 31.58% | 9/14/25 |
| trial_002 | complete | False | 0.258 | 37.50% | 28.57% | 7/8/27 |
| trial_003 | complete | False |  | 33.33% |  | None/None/None |
| trial_004 | complete | False | 0.823 | 37.50% | 35.82% | 12/21/22 |
| trial_005 | complete | False | 0.581 | 38.10% | 32.14% | 9/13/25 |
| trial_006 | complete | False | 0.292 | 47.06% | 27.78% | 10/28/24 |
| trial_007 | complete | False | 0.015 | 38.10% | 26.23% | 8/19/26 |
| trial_008 | complete | False | 0.367 | 37.50% | 29.79% | 7/6/27 |
| trial_009 | complete | False | 0.204 | 42.11% | 28.99% | 10/25/24 |
| trial_010 | complete | False | 0.277 | 47.06% | 29.85% | 10/23/24 |
| trial_011 | complete | False | 0.000 | 42.86% | 29.03% | 9/19/25 |
| trial_012 | complete | False | 0.176 | 42.86% | 27.27% | 9/23/25 |

| evaluation | devices | faults | TP | FP | FN | TN | precision | recall | F1 | accuracy | balanced acc. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| confirmation/trial_001/deployed | 7126 | 7 | 3 | 1 | 4 | 7118 | 75.00% | 42.86% | 54.55% | 99.93% | 71.42% |
| confirmation/trial_001/selection_window | 7126 | 7 | 3 | 2 | 4 | 7117 | 60.00% | 42.86% | 50.00% | 99.92% | 71.41% |
| historical_test/trial_001 | 7119 | 12 | 0 | 2 | 12 | 7105 | 0.00% | 0.00% | 0.00% | 99.80% | 49.99% |

Accuracy counts a device as correct when it is a fault hit in time or a normal device with no alarm. Normal devices dominate, so accuracy stays high even with F1 near zero; read it with F1 and balanced accuracy.

| evaluation | review budget | listed devices | faults caught by review | recall: alarms only -> alarms + review |
|---|---|---|---|---|
| confirmation/trial_001/deployed | 0.25% | 18 | 2 | 42.86% -> 71.43% |
| confirmation/trial_001/deployed | 0.50% | 36 | 2 | 42.86% -> 71.43% |
| confirmation/trial_001/deployed | 1.00% | 71 | 2 | 42.86% -> 71.43% |
| confirmation/trial_001/deployed | 2.00% | 143 | 2 | 42.86% -> 71.43% |
| confirmation/trial_001/selection_window | 0.25% | 18 | 2 | 42.86% -> 71.43% |
| confirmation/trial_001/selection_window | 0.50% | 36 | 2 | 42.86% -> 71.43% |
| confirmation/trial_001/selection_window | 1.00% | 71 | 2 | 42.86% -> 71.43% |
| confirmation/trial_001/selection_window | 2.00% | 143 | 2 | 42.86% -> 71.43% |
| historical_test/trial_001 | 0.25% | 18 | 2 | 0.00% -> 16.67% |
| historical_test/trial_001 | 0.50% | 36 | 2 | 0.00% -> 16.67% |
| historical_test/trial_001 | 1.00% | 71 | 3 | 0.00% -> 25.00% |
| historical_test/trial_001 | 2.00% | 142 | 3 | 0.00% -> 25.00% |

Review list = the highest-scoring devices that did not alarm (tools/review_list.py). The budget should come from how many devices people can review; all four budgets are shown so none is picked after seeing results.

| model | operating point | trees | nodes | parameters | max depth | features | file | 1 row | 10k rows |
|---|---|---|---|---|---|---|---|---|---|
| trial_001 | cv_oof step 250, thr 0.9396 | 250 | 5718 | 5,718 | 4 | 135 | 534 KiB | 0.108 ms | 24.5 ms |

Parameters = split thresholds + leaf values of the deployed (step-truncated) model.
Latency is model-only scoring on this machine (1 CPUs, Python 3.12.3); feature construction and the full validation pass are in `recorded_training_timings_seconds`.
