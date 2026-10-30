# Session summary: typea_v3.2/session

- Domain `memory`, status `trial_budget_complete`, wall clock 5.10 h
- Baseline `trial_001`, final best `trial_008`, accepted ['trial_008']
- Agent calls 45 {'code': 13, 'feature': 7, 'analysis': 25}, tokens 888,852
- Manual interventions 0, agent decisions 20, controller fallbacks 0, response repairs 5
- Automation (manual = budget changes only): step coverage 100.00% (74 automatic / 0 manual steps), trial coverage 100.00% (13/13), full pipeline automated: yes

| trial | status | accepted | P(better) | selection F1 | CV F1 | CV TP/FP/FN |
|---|---|---|---|---|---|---|
| trial_001 | complete | True |  | 10.10% | 14.12% | 62/544/210 |
| trial_002 | complete | False | 0.095 | 12.12% | 12.27% | 46/432/226 |
| trial_003 | complete | False | 0.249 | 12.17% | 13.52% | 57/514/215 |
| trial_004 | complete | False | 0.776 | 18.63% | 15.50% | 52/347/220 |
| trial_005 | complete | False | 0.620 | 11.88% | 14.32% | 63/545/209 |
| trial_006 | complete | False | 0.209 | 10.26% | 13.06% | 47/401/225 |
| trial_007 | complete | False | 0.776 | 18.63% | 15.50% | 52/347/220 |
| trial_008 | complete | True | 0.932 | 18.35% | 16.22% | 73/555/199 |
| trial_009 | complete | False | 0.820 | 19.74% | 17.23% | 56/322/216 |
| trial_010 | complete | False | 0.345 | 18.75% | 15.77% | 53/347/219 |
| trial_011 | complete | False | 0.820 | 19.74% | 17.23% | 56/322/216 |
| trial_012 | complete | False | 0.461 | 19.05% | 16.08% | 64/460/208 |
| trial_013 | complete | False | 0.788 | 19.74% | 16.99% | 78/568/194 |

| evaluation | devices | faults | TP | FP | FN | TN | precision | recall | F1 | accuracy | balanced acc. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| confirmation/trial_001/deployed | 32015 | 40 | 8 | 163 | 32 | 31813 | 4.68% | 20.00% | 7.58% | 99.39% | 59.75% |
| confirmation/trial_001/selection_window | 32015 | 40 | 11 | 144 | 29 | 31833 | 7.10% | 27.50% | 11.28% | 99.47% | 63.53% |
| confirmation/trial_008/deployed | 32015 | 40 | 12 | 103 | 28 | 31873 | 10.43% | 30.00% | 15.48% | 99.59% | 64.84% |
| confirmation/trial_008/selection_window | 32015 | 40 | 9 | 61 | 31 | 31914 | 12.86% | 22.50% | 16.36% | 99.71% | 61.15% |
| historical_test/trial_001/deployed | 34008 | 101 | 20 | 183 | 81 | 33727 | 9.85% | 19.80% | 13.16% | 99.23% | 59.64% |
| historical_test/trial_001/selection_window | 34008 | 101 | 24 | 170 | 77 | 33738 | 12.37% | 23.76% | 16.27% | 99.28% | 61.63% |
| historical_test/trial_008/deployed | 34008 | 101 | 27 | 139 | 74 | 33774 | 16.27% | 26.73% | 20.22% | 99.39% | 63.17% |
| historical_test/trial_008/selection_window | 34008 | 101 | 19 | 76 | 82 | 33834 | 20.00% | 18.81% | 19.39% | 99.54% | 59.30% |

Accuracy counts a device as correct when it is a fault hit in time or a normal device with no alarm. Normal devices dominate, so accuracy stays high even with F1 near zero; read it with F1 and balanced accuracy.

| evaluation | review budget | listed devices | faults caught by review | recall: alarms only -> alarms + review |
|---|---|---|---|---|
| confirmation/trial_001/deployed | 0.25% | 80 | 2 | 20.00% -> 25.00% |
| confirmation/trial_001/deployed | 0.50% | 160 | 3 | 20.00% -> 27.50% |
| confirmation/trial_001/deployed | 1.00% | 320 | 3 | 20.00% -> 27.50% |
| confirmation/trial_001/deployed | 2.00% | 640 | 4 | 20.00% -> 30.00% |
| confirmation/trial_001/selection_window | 0.25% | 80 | 1 | 27.50% -> 30.00% |
| confirmation/trial_001/selection_window | 0.50% | 160 | 2 | 27.50% -> 32.50% |
| confirmation/trial_001/selection_window | 1.00% | 320 | 2 | 27.50% -> 32.50% |
| confirmation/trial_001/selection_window | 2.00% | 640 | 4 | 27.50% -> 37.50% |
| confirmation/trial_008/deployed | 0.25% | 80 | 4 | 30.00% -> 40.00% |
| confirmation/trial_008/deployed | 0.50% | 160 | 5 | 30.00% -> 42.50% |
| confirmation/trial_008/deployed | 1.00% | 320 | 5 | 30.00% -> 42.50% |
| confirmation/trial_008/deployed | 2.00% | 640 | 7 | 30.00% -> 47.50% |
| confirmation/trial_008/selection_window | 0.25% | 80 | 6 | 22.50% -> 37.50% |
| confirmation/trial_008/selection_window | 0.50% | 160 | 7 | 22.50% -> 40.00% |
| confirmation/trial_008/selection_window | 1.00% | 320 | 8 | 22.50% -> 42.50% |
| confirmation/trial_008/selection_window | 2.00% | 640 | 10 | 22.50% -> 47.50% |
| historical_test/trial_001/deployed | 0.25% | 85 | 3 | 19.80% -> 22.77% |
| historical_test/trial_001/deployed | 0.50% | 170 | 8 | 19.80% -> 27.72% |
| historical_test/trial_001/deployed | 1.00% | 340 | 13 | 19.80% -> 32.67% |
| historical_test/trial_001/deployed | 2.00% | 680 | 20 | 19.80% -> 39.60% |
| historical_test/trial_001/selection_window | 0.25% | 85 | 8 | 23.76% -> 31.68% |
| historical_test/trial_001/selection_window | 0.50% | 170 | 16 | 23.76% -> 39.60% |
| historical_test/trial_001/selection_window | 1.00% | 340 | 20 | 23.76% -> 43.56% |
| historical_test/trial_001/selection_window | 2.00% | 680 | 26 | 23.76% -> 49.50% |
| historical_test/trial_008/deployed | 0.25% | 85 | 10 | 26.73% -> 36.63% |
| historical_test/trial_008/deployed | 0.50% | 170 | 12 | 26.73% -> 38.61% |
| historical_test/trial_008/deployed | 1.00% | 340 | 21 | 26.73% -> 47.52% |
| historical_test/trial_008/deployed | 2.00% | 680 | 31 | 26.73% -> 57.43% |
| historical_test/trial_008/selection_window | 0.25% | 85 | 11 | 18.81% -> 29.70% |
| historical_test/trial_008/selection_window | 0.50% | 170 | 17 | 18.81% -> 35.64% |
| historical_test/trial_008/selection_window | 1.00% | 340 | 25 | 18.81% -> 43.56% |
| historical_test/trial_008/selection_window | 2.00% | 680 | 39 | 18.81% -> 57.43% |

Review list = the highest-scoring devices that did not alarm (tools/review_list.py). The budget should come from how many devices people can review; all four budgets are shown so none is picked after seeing results.

| model | operating point | trees | nodes | parameters | max depth | features | file | 1 row | 10k rows |
|---|---|---|---|---|---|---|---|---|---|
| trial_001 | cv_oof step 250, thr 0.9147 | 250 | 7742 | 7,742 | 4 | 164 | 669 KiB | 0.118 ms | 23.7 ms |
| trial_008 | cv_oof step 250, thr 0.9308 | 250 | 5480 | 5,480 | 4 | 168 | 657 KiB | 0.114 ms | 24.9 ms |

Parameters = split thresholds + leaf values of the deployed (step-truncated) model.
Latency is model-only scoring on this machine (1 CPUs, Python 3.12.3); feature construction and the full validation pass are in `recorded_training_timings_seconds`.
