"""User-requested, post-hoc TEST evaluation; never selects a new winner."""
import fcntl
import json
import sys
import time
from pathlib import Path

R = Path(__file__).resolve().parent
sys.path[:0] = [str(R), str(R / 'runtime')]
sys.dont_write_bytecode = True

from common import read, save, sha, require, event
from agent import receipt
from operating_point import choose, load
from history_math import verify
from feature_programs import materialize
from prediction import predict
from evaluation import metrics, lead_summary
from supervisor import preserved


def main():
    import pandas as pd

    session = R / 'typeb/session'
    dest = R / 'typeb/supplemental_trial013'
    dest.mkdir(exist_ok=True)
    lock = (dest / 'evaluation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    preserved()
    state = read(session / 'state.json')
    config = read(session / 'config.json')
    require(state['status'] == 'trial_budget_complete', 'TypeB search unfinished')
    trial = next(t for t in state['trials'] if t['id'] == 'trial_013')
    require(trial['status'] == 'complete', 'Trial 13 unfinished')
    folder = session / 'trials/trial_013/attempt_01'
    complete = receipt(folder)
    points = dict(deployed=choose(folder, config['operating_point']),
                  selection_window=choose(folder, 'selection_window'))
    baseline_dir = R / 'typeb/historical_test'
    baseline = read(baseline_dir / 'summary.json')['reports']
    require({r['point'] for r in baseline} == set(points), 'Baseline comparison missing')
    baseline_hashes = {}
    for report in baseline:
        path = baseline_dir / report['trial'] / report['point']
        for name, digest in read(path / 'complete.json')['files'].items():
            require(sha(path / name) == digest, 'Baseline evidence changed')
            baseline_hashes[str((path / name).relative_to(R))] = digest

    frozen = dict(trial='trial_013', recipe=complete['recipe'], features=complete['features'],
                  points=points, state_sha256=sha(session / 'state.json'),
                  config_sha256=sha(session / 'config.json'),
                  trial_receipt_sha256=sha(folder / 'complete.json'),
                  script_sha256=sha(Path(__file__)), baseline_hashes=baseline_hashes,
                  authorization='User requested trial 13 TEST evaluation after seeing baseline TEST results.',
                  post_hoc_trial_selection=True, independent_blind_test=False,
                  test_previously_exposed=True, threshold_reselected=False, agent_feedback=False)
    if (dest / 'frozen.json').exists():
        require(read(dest / 'frozen.json') == frozen, 'Frozen evaluation changed')
    else:
        save(dest / 'frozen.json', frozen)
    event(dest, 'supplemental_test_frozen', trial='trial_013', post_hoc_trial_selection=True,
          threshold_reselected=False, agent_feedback=False)

    cache = Path('/root/autodl-tmp/final_v31_20261028/final_test_20261029/typeb/cache_test')
    manifest = verify(cache)
    cc = dict(config, active_cache=str(cache), num_threads=2, score_workers=4)
    cc, manifest = materialize(cc, complete['recipe']['feature_program'], manifest, final_test=True)
    require(manifest['features'] == complete['features'], 'Feature order drift')
    reports = []
    for point, op in points.items():
        out = dest / point
        out.mkdir(exist_ok=True)
        if (out / 'complete.json').exists():
            for name, digest in read(out / 'complete.json')['files'].items():
                require(sha(out / name) == digest, 'Existing supplemental result changed')
            report = read(out / 'report.json')
        else:
            table, stats = predict(load(folder, op, 2), cc, complete['recipe'], manifest, 'test', op['threshold'])
            table.to_csv(out / 'devices.csv', index=False)
            report = dict(task='typeb', trial='trial_013', point=point, operating_point=op,
                          metrics=metrics(table, op['threshold']), prediction_stats=stats,
                          lead_summary=lead_summary(table), test_window=manifest['windows']['test'],
                          post_hoc_trial_selection=True, independent_blind_test=False,
                          threshold_reselected=False, agent_feedback=False)
            save(out / 'report.json', report)
            save(out / 'complete.json', dict(files={n: sha(out / n) for n in ['report.json', 'devices.csv']}))
        table = pd.read_csv(out / 'devices.csv')
        rescored = metrics(table, op['threshold'])
        require(all(rescored[k] == report['metrics'][k] for k in ['TP', 'FP', 'FN', 'f1']), 'CSV rescore mismatch')
        base_report = next(r for r in baseline if r['point'] == point)
        base_table = pd.read_csv(baseline_dir / base_report['trial'] / point / 'devices.csv')
        require(sorted(table.device.tolist()) == sorted(base_table.device.tolist()), 'Evaluation device cohort changed')
        require(report['test_window'] == base_report['test_window'], 'Evaluation window changed')
        require(report['metrics']['protocol'] == base_report['metrics']['protocol'], 'Evaluation protocol changed')
        reports.append(report)
        event(dest, 'supplemental_test_scored', trial='trial_013', point=point, metrics=report['metrics'])

    comparison = []
    for report in baseline + reports:
        op = report['operating_point']
        comparison.append(dict(trial=report['trial'], point=report['point'], step=op['step'],
                               threshold=op['threshold'], **{k: report['metrics'][k] for k in
                               ['TP', 'FP', 'FN', 'precision', 'recall', 'f1', 'eligible_devices', 'eligible_fault_devices']}))
    pd.DataFrame(comparison).to_csv(dest / 'comparison.csv', index=False)
    require(sha(session / 'state.json') == frozen['state_sha256'], 'Search state changed')
    require(sha(session / 'config.json') == frozen['config_sha256'], 'Search config changed')
    require(sha(folder / 'complete.json') == frozen['trial_receipt_sha256'], 'Trial changed')
    for name, digest in baseline_hashes.items():
        require(sha(R / name) == digest, 'Baseline evidence changed during replay')
    preserved()
    event(dest, 'supplemental_test_verified', search_state_unchanged=True, same_cohort=True,
          csv_rescore_matches=True, old_sessions_unchanged=True)
    save(dest / 'summary.json', dict(reports=reports, comparison=comparison,
         completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
         post_hoc_trial_selection=True, independent_blind_test=False, threshold_reselected=False,
         agent_feedback=False, adopted_winner_unchanged=state['best']))
    files = {str(p.relative_to(dest)): dict(sha256=sha(p), bytes=p.stat().st_size)
             for p in dest.rglob('*') if p.is_file() and p.suffix != '.lock' and p.name != 'export_manifest.json'}
    save(dest / 'export_manifest.json', dict(files=files, old_sessions_unchanged=True,
         search_state_unchanged=True, same_cohort_verified=True, device_csv_rescore_matches=True))
    print(json.dumps(dict(status='complete', comparison=comparison)), flush=True)


if __name__ == '__main__':
    main()
