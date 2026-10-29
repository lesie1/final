"""New session: same validated raw-B inputs and v3.2 runtime, screen_margin=1.0."""
import copy
import os
import shutil
import sys
import time
from pathlib import Path

R = Path(__file__).resolve().parent
PREVIOUS = Path('/root/autodl-tmp/typeb_raw_v32_20261029')
sys.dont_write_bytecode = True
if not (R / 'runtime').exists():
    shutil.copytree(PREVIOUS / 'runtime', R / 'runtime', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
sys.path.insert(0, str(R / 'runtime'))
from common import read, save, sha, digest, code_hashes, require, event
from history_math import verify
from settings import validate_config
from agent import Loop


def main():
    require(R.name == 'typeb_raw_screen1_v32_20261029', 'Unexpected run destination')
    if (R / 'ready.json').exists():
        Loop(R / 'typeb/session')
        print('ALREADY_READY', flush=True)
        return
    old_session = PREVIOUS / 'typeb/session'
    old_state = read(old_session / 'state.json')
    require(old_state['status'] in ('stopped', 'trial_budget_complete'), 'Prior run still active')
    old_config = read(old_session / 'config.json')
    require(old_config['screen_margin'] == 0.02 and old_config['max_trials'] == 13, 'Unexpected reference settings')
    old_release = read(PREVIOUS / 'release.json')
    require(code_hashes() == old_release['runtime_hashes'], 'Runtime differs from previous run')
    prior = read(PREVIOUS / 'source_preservation.json')
    names = ['config.json', 'state.json', 'events.jsonl', 'mapping.json', 'code_seal.json', 'inventory.json']
    prior['raw_typeb_screen002'] = dict(path=str(old_session), seals={n: sha(old_session / n) for n in names})
    prior['raw_typeb_frozen_test'] = dict(path=str(PREVIOUS / 'typeb/historical_test'),
        seals={n: sha(PREVIOUS / 'typeb/historical_test' / n) for n in ['frozen.json', 'summary.json', 'events.jsonl']})
    save(R / 'source_preservation.json', prior)
    event(R, 'screen_margin_ablation_preparation', previous=0.02, current=1.0, maximum_trials=13)

    cache_receipts = {}
    for name in ['raw_cache', 'raw_cache_test']:
        source = PREVIOUS / name
        manifest = verify(source)
        require(bool(manifest['test_opened']) == (name == 'raw_cache_test'), 'Wrong cache partition')
        if not (R / name).exists():
            shutil.copytree(source, R / name)
        copied = verify(R / name)
        require(copied == manifest, 'Copied cache differs')
        cache_receipts[name] = dict(source=str(source), manifest_sha256=sha(source / 'manifest.json'),
                                   reused_without_rebuilding=True, features=len(manifest['features']))
    save(R / 'cache_reuse.json', cache_receipts)
    for name in ['raw_typeb.py', 'prepare.py', 'test_raw_typeb.py', 'archive_audit.json']:
        shutil.copy2(PREVIOUS / name, R / name)
    task = R / 'typeb'
    session = task / 'session'
    session.mkdir(parents=True, exist_ok=True)
    require(not (session / 'state.json').exists(), 'Session exists; inspect rather than reinitialize')
    c = copy.deepcopy(old_config)
    c.update(screen_margin=1.0, output_root=str(session), workspace=str(task / 'workspace'),
             active_cache=str(R / 'raw_cache'))
    validate_config(c)
    changed = {k: dict(previous=old_config.get(k), current=c.get(k))
               for k in set(old_config) | set(c) if old_config.get(k) != c.get(k)}
    require(set(changed) == {'screen_margin', 'output_root', 'workspace', 'active_cache'}, 'Unrequested configuration change')
    save(R / 'config_diff.json', dict(changes=changed, experimental_changes=['screen_margin'],
         infrastructure_changes=['output_root', 'workspace', 'active_cache'],
         test_previously_exposed=True, test_must_not_inform_search=True,
         note='Fresh bounded adaptive search. Later proposals can differ because feedback differs; this is not a paired replay of identical candidates.'))
    mapping = read(old_session / 'mapping.json')
    save(session / 'config.json', c)
    save(session / 'mapping.json', mapping)
    save(session / 'inventory.json', read(old_session / 'inventory.json'))
    save(session / 'code_seal.json', code_hashes())
    save(session / 'state.json', dict(version=1, status='ready', config_hash=digest(c),
         dataset=old_state['dataset'], mapping_hash=digest(mapping), calls=0, tokens=0,
         trials=[], best=None, pending=None, analyze_next=None, diagnostics={}, feature_jobs=0,
         experience=None, experience_tried=True, last_failure=None))
    save(session / 'bootstrap.json', dict(source=str(old_session), model_weights_imported=False,
         baseline_from_scratch=True, source_cache_reused=True, original_rows_and_labels_unchanged=True,
         modified_factor='screen_margin', previous_value=0.02, new_value=1.0,
         old_test_previously_exposed=True, old_results_not_supplied_to_agent=True))
    isolation = read(PREVIOUS / 'typeb/isolation.json')
    isolation['forbidden_reads'] += [str(PREVIOUS / 'typeb'), str(PREVIOUS / 'publication'),
        str(PREVIOUS / 'raw_cache_test'), str(R / 'raw_cache_test'), str(R / 'typeb/historical_test')]
    isolation['readonly_roots'] += [str(PREVIOUS), str(R / 'raw_cache'), str(R / 'runtime')]
    save(task / 'isolation.json', isolation)
    Loop(session).prepare()
    adapters = ['raw_typeb.py', 'prepare.py', 'test_raw_typeb.py', 'initialize.py',
                'supervisor.py', 'evaluate.py', 'report_export.py', 'launch.py', 'check_screening.py']
    save(R / 'release.json', dict(runtime_commit=old_release['runtime_commit'], runtime_modified=False,
         runtime_hashes=code_hashes(), adapter_hashes={n: sha(R / n) for n in adapters},
         only_experimental_change=dict(screen_margin=dict(previous=0.02, current=1.0))))
    for source in prior.values():
        for name, h in source['seals'].items():
            require(sha(Path(source['path']) / name) == h, 'Old result changed: ' + name)
    save(R / 'ready.json', dict(status='ready', max_trials=13, features=135, raw_features=69,
         screen_margin=1.0, test_opened=False, cached_historical_test_previously_exposed=True,
         automatic_test_after_freeze=True, old_sessions_unchanged=True))
    save(R / 'status.json', dict(status='ready', at=time.time()))
    event(R, 'preparation_complete', screen_margin=1.0, features=135,
          only_experimental_change='screen_margin', old_sessions_unchanged=True)
    print('READY', flush=True)


if __name__ == '__main__':
    main()
