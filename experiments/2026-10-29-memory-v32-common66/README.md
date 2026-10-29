# v3.2 memory experiments and common-66 controls

- [Results and interpretation](report.md) / [HTML report](report.html)
- [All search trials](search_trials.csv) / [Search TEST results](search_test.csv)
- [User-requested TypeB trial 13 comparison](supplemental_typeb_test.csv)
- [All 48 control evaluations](controlled_results.csv) / [Pooled counts](pooled_results.csv)
- [TypeA events](typea/session/events.jsonl), [TypeB events](typeb/session/events.jsonl), [control events](controlled/events.jsonl), [supplemental events](typeb/supplemental_trial013/events.jsonl)
- [Export hashes and exclusions](export_manifest.json) / [Download verification](download_verification.json)

Completed: 13 TypeA search trials, 13 TypeB search trials, and 12 fixed common-feature controls (four arms, seeds 42/43/44). No budget extension or TEST-based threshold tuning.

Historical TEST was previously exposed. The extra TypeB trial was requested after baseline TEST results were viewed. All three seeds share evaluation cohorts and are not independent test sets. A-only trains trees on A but calibrates the operating point using B TRAIN out-of-fold labels.

The review snapshot preserves exported sessions, events, code, receipts, primary models, and evaluation device tables. Raw data, large caches, training matrices and CV fold models remain on the server; exclusions are explicit. It is not a standalone resumable checkpoint. Older sessions were hash-verified unchanged.

The source runtime is commit `8ebafc8d311e734d2089df9d1106c962e2026b66`; this publication does not merge its code into the repository root.
