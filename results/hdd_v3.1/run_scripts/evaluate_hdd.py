"""User-authorized final HDD inference; leave the sealed v3.1 search untouched."""
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

R = Path('/root/autodl-tmp/hdd_final_test_20261029')
SOURCE = Path('/root/autodl-tmp/final_v31_20261028')
SESSION = SOURCE / 'hdd/session'
CACHE = R / 'cache_test'
OUT = R / 'evaluation'
sys.dont_write_bytecode = True
sys.path.insert(0, str(SOURCE / 'runtime'))
from common import read, save, sha, require, event
from history_math import verify
from agent import receipt
from feature_programs import program, materialize
from rerank import load_model
from prediction import predict
from evaluation import metrics, lead_summary


def readiness():
    reasons = []
    if not (R / 'ready.json').exists():
        reasons.append('TEST cache preparation has not finished')
    final_status = SOURCE / 'hdd/result_status.json'
    if not final_status.exists():
        reasons.append('Search supervisor has not finished')
    else:
        result = read(final_status)
        if result.get('status') != 'complete' or result.get('error') or result.get('export_exit') != 0:
            reasons.append('Search/confirmation/export did not finish successfully')
    s = read(SESSION / 'state.json')
    if s['status'] not in ('stopped', 'trial_budget_complete') or s.get('pending'):
        reasons.append('Search has active or pending work')
    if not (SOURCE / 'hdd/confirmation/summary.json').exists():
        reasons.append('Later-VAL confirmation missing')
    pid_file = SOURCE / 'supervisor_pid.json'
    if pid_file.exists():
        pid = read(pid_file)['pid']
        proc = Path('/proc') / str(pid)
        if proc.exists():
            cmd = (proc / 'cmdline').read_bytes().split(b'\0')
            if str(SOURCE / 'final_supervisor.py').encode() in cmd:
                reasons.append('Original supervisor still running')
    return reasons


def main():
    reasons = readiness()
    require(not reasons, 'Final TEST prerequisites missing: ' + '; '.join(reasons))
    OUT.mkdir(exist_ok=True)
    lock = (OUT / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.monotonic()
    def status(phase, **kw):
        save(OUT / 'status.json', dict(phase=phase, at=time.time(), pid=os.getpid(), **kw))
        print(phase, flush=True)
    status('verifying_inputs')
    preserved = read(R / 'source_preservation.json')['files']
    for name, h in preserved.items():
        require(sha(name) == h, 'Sealed source changed: ' + name)
    c = read(SESSION / 'config.json')
    s = read(SESSION / 'state.json')
    require(len(s['trials']) <= 13 and s['best'], 'Invalid search budget or missing winner')
    trial = next(t for t in s['trials'] if t['id'] == s['best'])
    require(trial['status'] == 'complete', 'Winner is incomplete')
    folder = SESSION / 'trials' / trial['id'] / 'attempt_01'
    result = receipt(folder)
    require(result['recipe'] == trial['recipe'], 'Winner recipe differs')
    require(result['recipe'].get('rerank', 'none') == 'none', 'Composite model needs separate audited inference handling')
    model_path = folder / 'model.ubj'
    require(sha(model_path) == result['model_sha256'], 'Winner model changed')
    threshold = result['metrics']['threshold']
    confirmation = read(SOURCE / 'hdd/confirmation/summary.json')
    checked = next(x for x in confirmation['reports'] if x['trial'] == trial['id'])
    require(checked['model_sha256'] == result['model_sha256'] and
            checked['metrics']['threshold'] == threshold, 'Confirmation used a different operating point')
    base = verify(c['active_cache'])
    test = verify(CACHE)
    require(sha(CACHE / 'manifest.json') == read(R / 'ready.json')['manifest_sha256'], 'TEST cache changed after readiness')
    require(test['test_opened'] and test['spec']['test'], 'Wrong cache partition')
    for name in ['features', 'windows', 'coverage', 'source_receipts']:
        require(test[name] == base[name], 'TEST source contract differs: ' + name)
    for name in ['mapping', 'inventory', 'coverage']:
        require(test['spec'][name] == base['spec'][name], 'TEST identity differs: ' + name)
    previous_ids = read(Path(c['active_cache']) / 'devices.json')
    require(read(CACHE / 'devices.json')[:len(previous_ids)] == previous_ids, 'Device codes changed')

    protected = [SESSION / name for name in ['config.json', 'state.json', 'events.jsonl',
        'mapping.json', 'code_seal.json', 'inventory.json']]
    protected += [folder / 'complete.json', folder / 'result.json', model_path,
        SOURCE / 'hdd/confirmation/summary.json', SOURCE / 'hdd/result_status.json',
        CACHE / 'manifest.json', Path(c['active_cache']) / 'manifest.json']
    key = result['recipe']['feature_program']
    chain = []
    while key != 'base':
        require(len(chain) < 12 and key not in chain, 'Invalid feature program ancestry')
        spec = program(c, key)
        chain.append(key)
        for name in ['program.json', 'transform.py']:
            protected.append(SESSION / 'feature_programs' / key / name)
        key = spec['parent']
    entry = dict(trial=trial['id'], model=str(model_path), model_sha256=result['model_sha256'],
        recipe=result['recipe'], features=result['features'], best_step=result['best_step'], threshold=threshold,
        operating_point_source='Original v3.1 selection-window model and threshold, identical to later-VAL confirmation',
        selection_rule='Original paired CV adoption rule; no TEST selection',
        source_seals={str(p): sha(p) for p in protected}, runtime_and_input_seals=preserved,
        test_window=test['windows']['test'], base_feature_count=len(base['features']),
        test_preprocessing_authorized_before_search_finished=True,
        historical_test_exposure='Not independently established as pristine; no fresh-blind-test claim',
        independent_blind_test=False)
    frozen_path = OUT / 'frozen.json'
    if frozen_path.exists():
        require(read(frozen_path) == entry, 'Frozen entry changed')
    else:
        save(frozen_path, entry)
        event(OUT, 'final_model_frozen', trial=trial['id'], threshold=threshold,
              best_step=result['best_step'], model_sha256=result['model_sha256'])

    def unchanged():
        for name, h in dict(preserved, **entry['source_seals']).items():
            require(sha(name) == h, 'Protected source changed during final inference: ' + name)
    unchanged()
    if (OUT / 'complete.json').exists():
        for name, h in read(OUT / 'complete.json')['files'].items():
            require(sha(OUT / name) == h, 'Completed TEST artifact differs')
        status('complete', already_complete=True)
        return
    # Copy only immutable numerical feature programs. Generated TEST views go in OUT,
    # never in the original search session or inside the base-cache manifest tree.
    for key in chain:
        dest = OUT / 'feature_programs' / key
        dest.mkdir(parents=True, exist_ok=True)
        for name in ['program.json', 'transform.py']:
            source = SESSION / 'feature_programs' / key / name
            if not (dest / name).exists():
                shutil.copy2(source, dest / name)
            require(sha(dest / name) == sha(source), 'Feature program copy changed')
    cc = copy.deepcopy(c)
    cc.update(output_root=str(OUT), active_cache=str(CACHE), num_threads=2,
              score_workers=8, min_free_disk_gb=40)
    save(OUT / 'inference_config.json', cc)
    status('materializing_frozen_features')
    cfg, manifest = materialize(cc, entry['recipe']['feature_program'], test, final_test=True)
    require(manifest['features'] == entry['features'], 'Actual model feature order differs')
    model = load_model(model_path, 2)
    require(model.num_features() == len(entry['features']), 'Model feature width differs')
    require(model.num_boosted_rounds() == entry['best_step'], 'Frozen tree count differs')
    unchanged()
    save(OUT / 'input_audit.json', dict(frozen_sha256=sha(frozen_path),
        base_test_manifest_sha256=sha(CACHE / 'manifest.json'),
        model_features=manifest['features'], feature_view_manifest_sha256=sha(Path(cfg['active_cache']) / 'manifest.json'),
        source_seals_verified=True, feature_program_chain=chain))
    status('predicting_frozen_test')
    table, stats = predict(model, cfg, entry['recipe'], manifest, 'test', threshold)
    scored = metrics(table, threshold)
    table.to_csv(OUT / 'devices_test.csv', index=False)
    table.loc[table.status.str.startswith('unknown_')].to_csv(OUT / 'unknown_devices_test.csv', index=False)
    report = dict(trial=entry['trial'], model_sha256=entry['model_sha256'],
        best_step=entry['best_step'], metrics=scored, test_window=entry['test_window'],
        operating_point_source=entry['operating_point_source'], lead_summary=lead_summary(table),
        prediction_stats=stats, independent_blind_test=False,
        historical_test_exposure=entry['historical_test_exposure'], threshold_reselected=False,
        retrained=False, agent_feedback=False, seconds=time.monotonic() - started)
    save(OUT / 'report_test.json', report)
    unchanged()
    event(OUT, 'final_test_complete', trial=entry['trial'], metrics=scored,
          old_search_preserved=True, threshold_reselected=False)
    files = ['frozen.json', 'inference_config.json', 'input_audit.json', 'report_test.json',
             'devices_test.csv', 'unknown_devices_test.csv']
    save(OUT / 'complete.json', dict(files={name: sha(OUT / name) for name in files},
         source_search_preserved=True, sealed_runtime_unchanged=True, publication_pending=True))
    status('complete', metrics=scored, publication_pending=True)


if __name__ == '__main__':
    if '--check-ready' in sys.argv:
        missing = readiness()
        print(json.dumps(dict(ready=not missing, missing=missing)))
    else:
        try:
            main()
        except Exception as error:
            if OUT.exists():
                save(OUT / 'error.json', dict(at=time.time(), error=str(error)))
            traceback.print_exc()
            raise
