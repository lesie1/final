# Session summary: typea_v3.2/session

- Domain `memory`, status `trial_budget_complete`, wall clock 5.10 h
- Baseline `trial_001`, final best `trial_008`, accepted ['trial_008']
- Agent calls 45 {'code': 13, 'feature': 7, 'analysis': 25}, tokens 888,852
- Manual interventions 0, agent decisions 20, controller fallbacks 0, response repairs 5

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

| model | operating point | trees | nodes | max depth | features | file | 1 row | 10k rows |
|---|---|---|---|---|---|---|---|---|
| trial_001 | cv_oof step 250, thr 0.9147 | 250 | 7742 | 4 | 164 | 669 KiB | 0.112 ms | 25.8 ms |
| trial_008 | cv_oof step 250, thr 0.9308 | 250 | 5480 | 4 | 168 | 657 KiB | 0.124 ms | 25.6 ms |

Latency is model-only scoring on this machine (1 CPUs, Python 3.12.3); feature construction and the full validation pass are in `recorded_training_timings_seconds`.
