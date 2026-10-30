# Session summary: hdd_v3.1/session

- Domain `hdd`, status `trial_budget_complete`, wall clock 24.56 h
- Baseline `trial_001`, final best `trial_004`, accepted ['trial_004']
- Agent calls 50 {'code': 13, 'feature': 10, 'analysis': 27}, tokens 989,281
- Manual interventions 1, agent decisions 16, controller fallbacks 2, response repairs 13
- Automation (manual = budget changes only): step coverage 98.53% (67 automatic / 1 manual steps), trial coverage 100.00% (13/13), full pipeline automated: no

| trial | status | accepted | P(better) | selection F1 | CV F1 | CV TP/FP/FN |
|---|---|---|---|---|---|---|
| trial_001 | complete | True |  | 23.80% | 25.48% | 716/2555/1634 |
| trial_002 | complete | False | 0.149 | 24.11% | 24.98% | 628/2050/1722 |
| trial_003 | complete | False | 0.813 | 25.10% | 25.99% | 662/2083/1688 |
| trial_004 | complete | True | 1.000 | 25.65% | 27.96% | 861/2948/1489 |
| trial_005 | complete | False | 0.000 | 25.65% | 27.96% | 861/2948/1489 |
| trial_006 | complete | False | 0.855 | 25.41% | 28.47% | 707/1910/1643 |
| trial_007 | complete | False | 0.786 | 25.77% | 28.27% | 745/2175/1605 |
| trial_008 | complete | False | 0.007 | 26.16% | 26.69% | 693/2150/1657 |
| trial_009 | complete | False | 0.540 | 25.52% | 28.01% | 728/2121/1622 |
| trial_010 | complete | False | 0.275 | 26.40% | 27.76% | 812/2688/1538 |
| trial_011 | complete | False | 0.039 | 25.38% | 27.30% | 770/2521/1580 |
| trial_012 | complete | False | 0.183 | 26.52% | 27.53% | 690/1973/1660 |
| trial_013 | complete | False | 0.016 | 25.39% | 27.27% | 803/2737/1547 |

| evaluation | devices | faults | TP | FP | FN | TN | precision | recall | F1 | accuracy | balanced acc. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| historical_test/test | 292103 | 804 | 348 | 1214 | 456 | 290085 | 22.28% | 43.28% | 29.42% | 99.43% | 71.43% |

Accuracy counts a device as correct when it is a fault hit in time or a normal device with no alarm. Normal devices dominate, so accuracy stays high even with F1 near zero; read it with F1 and balanced accuracy.

| evaluation | review budget | listed devices | faults caught by review | recall: alarms only -> alarms + review |
|---|---|---|---|---|
| historical_test/test | 0.25% | 730 | 76 | 43.28% -> 52.74% |
| historical_test/test | 0.50% | 1461 | 122 | 43.28% -> 58.46% |
| historical_test/test | 1.00% | 2921 | 193 | 43.28% -> 67.29% |
| historical_test/test | 2.00% | 5842 | 271 | 43.28% -> 76.99% |

Review list = the highest-scoring devices that did not alarm (tools/review_list.py). The budget should come from how many devices people can review; all four budgets are shown so none is picked after seeing results.

| model | operating point | trees | nodes | parameters | max depth | features | file | 1 row | 10k rows |
|---|---|---|---|---|---|---|---|---|---|
| trial_001 | selection_window step 550, thr 0.9870 | 550 | 15616 | 15,616 | 4 | 186 | 874 KiB | 0.167 ms | 51.3 ms |
| trial_004 | selection_window step 550, thr 0.9875 | 550 | 15640 | 15,640 | 4 | 190 | 875 KiB | 0.157 ms | 52.3 ms |

Parameters = split thresholds + leaf values of the deployed (step-truncated) model.
Latency is model-only scoring on this machine (1 CPUs, Python 3.12.3); feature construction and the full validation pass are in `recorded_training_timings_seconds`.
