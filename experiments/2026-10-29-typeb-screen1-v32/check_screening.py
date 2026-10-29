"""Behavior check on unmodified runtime with stub CV evidence; no training or API."""
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch
R = Path(__file__).resolve().parent
sys.path.insert(0, str(R / 'runtime'))
sys.dont_write_bytecode = True
from agent import Loop
from common import read, save, require


def main():
    target = R / 'screening_check'
    target.mkdir(exist_ok=True)
    current = read(R / 'typeb/session/config.json')
    loop = object.__new__(Loop)
    loop.c = dict(current)
    loop.s = dict(best='reference')
    base = dict(id='reference', metrics=dict(f1=1.0))
    trial = dict(id='challenger', metrics=dict(f1=0.0), recipe=dict(rerank='none'))
    loop.trial = lambda _: base
    visited = []
    def cross_validate(t):
        visited.append(t['id'])
        return target / t['id']
    loop.cross_validate = cross_validate
    loop.folder = lambda _: target
    evidence = dict(interval_95=[0.01, 0.2], probability_better=0.95, delta_f1=0.1,
        new_hits=3, lost_hits=2, new_false_alarms=1, removed_false_alarms=2, fault_units=34)
    with patch('crossval.paired_cv', return_value=evidence) as paired:
        loop.c['screen_margin'] = 0.02
        require(loop.accept(trial)['rule'] == 'screened_out' and not visited, 'Original screen behavior changed')
        loop.c['screen_margin'] = 1.0
        require(loop.accept(trial)['accepted'] and visited == ['reference', 'challenger'], 'Margin 1.0 still screened the worst possible F1 gap')
        paired.return_value = dict(evidence, probability_better=0.89)
        require(not loop.accept(trial)['accepted'], 'CV acceptance probability was relaxed')
    command = '''import os,sys
sys.dont_write_bytecode=True
sys.path.insert(0,sys.argv[1])
os.environ['TYPEA_SEARCH_ISOLATION']=sys.argv[2]
import search_isolation
search_isolation.install()
open(sys.argv[3]).read(1)
try: open(sys.argv[4]).read(1)
except PermissionError: print('ISOLATED')
else: raise RuntimeError('TEST cache visible to search')
'''
    result = subprocess.run([sys.executable, '-c', command, str(R / 'runtime'),
        str(R / 'typeb/isolation.json'), str(R / 'raw_cache/manifest.json'),
        str(R / 'raw_cache_test/manifest.json')], capture_output=True, text=True, check=True)
    require(result.stdout.strip() == 'ISOLATED', 'Isolation failed')
    # Spawn/import safety: evaluating this module must not start inference.
    import runpy
    require(not (R / 'typeb/historical_test').exists(), 'Unexpected TEST run before search')
    require(callable(runpy.run_path(str(R / 'evaluate.py'), run_name='spawn_import_check')['main']), 'Missing protected evaluation entry')
    require(not (R / 'typeb/historical_test').exists(), 'Evaluation executed while importing')
    save(target / 'checks.json', dict(original_screen_blocks=True, screen1_runs_cv_at_gap_minus1=True,
        acceptance_probability_unchanged=True, test_cache_read_blocked=True, trainval_read_allowed=True,
        evaluation_import_has_no_side_effect=True, synthetic_cv_evidence_only=True))
    print('SCREENING_AND_ISOLATION_CHECKS_PASSED', flush=True)


if __name__ == '__main__':
    main()
