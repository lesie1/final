"""Public entry-point regressions for review findings 2, 3 and 4."""
import copy
import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from agent import Loop, create_session
from common import ROOT, read, save, sha, process, digest
from experiment_contract import initial_plans, source_for
from history_store import checked_manifest, training_batches
from sample_builder import build_training, targets
from search_step import initialize
from tests.test_workflow import fixture as raw_fixture, FakeDeepSeek
from tests.test_v2_loop import fixture as cached_fixture


def frozen_reference(c):
    """Freeze a nonuniform TRAIN-only sample with all positives retained."""
    m = checked_manifest(c); recipe = initial_plans()[0]
    meta = pd.concat([b.to_pandas() for b in training_batches(c, recipe, m, meta_only=True)], ignore_index=True)
    valid, y = targets(meta, 7, m['windows']['train'][1], 900)
    valid &= (meta.time >= m['windows']['train'][0]).to_numpy() & (meta.history_rows > 0).to_numpy()
    eligible = meta.loc[valid].reset_index(drop=True); y = y[valid]
    membership = hashlib.sha256(eligible[['device','time']].to_numpy(dtype='<i8').tobytes()).hexdigest()
    ix = np.flatnonzero((y == 1) | (np.arange(len(y)) % 3 == 0))
    weights = np.ones(len(ix), np.float32)
    weights[y[ix] == 0] = (y == 0).sum() / (y[ix] == 0).sum()
    root = Path(c['output_root']).parent/'frozen_sample'; root.mkdir()
    np.save(root/'row_ids.npy', ix); np.save(root/'weights.npy', weights)
    ref = root/'reference.json'
    save(ref, dict(row_ids=str(root/'row_ids.npy'), weights=str(root/'weights.npy'),
        files={str(root/n):sha(root/n) for n in ('row_ids.npy','weights.npy')},
        full_training_samples=len(y), positive_samples=int(y.sum()), full_membership_sha256=membership,
        selected_membership_sha256=hashlib.sha256(eligible.iloc[ix][['device','time']].to_numpy(dtype='<i8').tobytes()).hexdigest()))
    return ref


class FreshInput(unittest.TestCase):
    def test_cli_init_run_trains_from_raw_without_manual_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);mapping=raw_fixture(root/'data','hdd')
            save(root/'mapping.json',mapping)
            save(root/'runtime.json',dict(device='cpu',num_threads=1,memory_limit_gb=2,rounds=2,
                eval_every=1,patience=2,max_trials=1,min_free_disk_gb=.01,cv_rounds=2,cv_folds=2,
                coverage_start='2024-01-01T00:00:00Z',coverage_end='2024-04-20T00:00:00Z'))
            env={k:v for k,v in os.environ.items() if not any(x in k.upper() for x in ('API_KEY','TOKEN','SECRET','PASSWORD'))}
            def cli(*args):
                p=subprocess.run([sys.executable,str(ROOT/'agent.py'),*map(str,args)],cwd=ROOT,env=env,
                                 capture_output=True,text=True,timeout=90)
                self.assertEqual(p.returncode,0,p.stdout+'\n'+p.stderr)
                return p.stdout
            cli('init','--data',root/'data','--workspace',root/'workspace','--domain','hdd',
                '--mapping',root/'mapping.json','--config',root/'runtime.json')
            session=next((root/'workspace/sessions').iterdir())
            self.assertFalse((session/'cache/manifest.json').exists())
            # A fixed code response avoids paid API calls; all data/training paths are real.
            out=session/'trials/trial_001/attempt_01';out.mkdir(parents=True)
            (out/'candidate.py').write_text(source_for(initial_plans()[0]),encoding='utf-8')
            cli('run','--session',session)
            state=read(session/'state.json')
            self.assertEqual(state['trials'][0]['status'],'complete',state['trials'][0].get('error'))
            self.assertIn('cv',state['trials'][0])
            self.assertEqual(state['calls'],0)
            self.assertFalse(read(session/'cache/manifest.json')['test_opened'])

    def test_explicit_missing_cache_never_rebuilds_from_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            session,c=cached_fixture(Path(tmp));c['active_cache']=str(Path(tmp)/'missing')
            save(session/'config.json',c);s=read(session/'state.json');s['config_hash']=digest(c);save(session/'state.json',s)
            with patch('agent.normalize') as builder,self.assertRaisesRegex(ValueError,'Explicit active_cache'):
                Loop(session).prepare()
            builder.assert_not_called()
            self.assertFalse((session/'cache').exists())

    def test_input_changed_after_init_is_rejected_before_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);mapping=raw_fixture(root/'data','hdd');save(root/'mapping.json',mapping)
            session=create_session(root/'data',root/'workspace','hdd','Input seal',mapping=root/'mapping.json')
            with (root/'data/quarter1.csv').open('a') as f:f.write('\n')
            with self.assertRaisesRegex(ValueError,'Source inventory changed'):Loop(session).prepare()
            self.assertFalse((session/'cache/manifest.json').exists())

    def test_init_prepare_builds_cache_without_manual_normalize(self):
        for domain in ('memory','hdd'):
            with self.subTest(domain=domain), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); mapping=raw_fixture(root/'data', domain)
                save(root/'mapping.json', mapping)
                session=create_session(root/'data', root/'workspace', domain, 'Synthetic input', mapping=root/'mapping.json')
                loop=Loop(session)
                self.assertFalse((session/'cache/manifest.json').exists())
                result=loop.prepare()
                self.assertTrue((session/'cache/manifest.json').exists())
                self.assertFalse(result['test_opened'])
                before=sha(session/'cache/manifest.json')
                self.assertEqual(loop.prepare(), result)
                self.assertEqual(sha(session/'cache/manifest.json'), before)


class FrozenDeviceWeight(unittest.TestCase):
    def test_initialize_then_device_weights_keeps_rows_and_seal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); old,c=cached_fixture(root)
            ref=frozen_reference(c); c['sampling_reference']=str(ref)
            from common import digest
            save(old/'config.json', c); state=read(old/'state.json'); state['config_hash']=digest(c);save(old/'state.json',state)
            protected={p:sha(p) for p in (ref,ref.parent/'row_ids.npy',ref.parent/'weights.npy',old/'config.json',old/'state.json')}
            initialize(old,root/'new')
            cfg=Loop(root/'new').c; m=checked_manifest(cfg)
            recipe=initial_plans()[0]; results=[]
            for name,weight in [('sample','sample'),('device','device')]:
                out=root/name;out.mkdir();r=dict(recipe,base_weight=weight)
                x,y,w,features,pw,audit,devices=build_training(cfg,r,m,out,900,2,return_devices=True)
                results.append((np.array(x),y.copy(),w.copy(),devices.copy(),audit))
                del x
            a,b=results
            np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1]);np.testing.assert_array_equal(a[3],b[3])
            for d in np.unique(b[3]):self.assertAlmostEqual(float(b[2][b[3]==d].sum()),1.,places=5)
            for d in np.unique(b[3]):
                take=b[3]==d
                np.testing.assert_allclose(b[2][take],a[2][take]/a[2][take].sum(),rtol=1e-6)
            self.assertEqual(a[4]['sampling']['selection_seal_sha256'],b[4]['sampling']['selection_seal_sha256'])
            # Exercise the actual grouped-CV training path with the inherited frozen sample.
            from crossval import oof_tables
            result=oof_tables(cfg,dict(recipe,base_weight='device'),root/'cv',rounds=4,k=3)
            self.assertEqual(result['folds'],3)
            for fold in range(3):
                audit=read(root/'cv'/f'fold_{fold}'/'audit.json')['audit']['base_weight_audit']
                self.assertAlmostEqual(audit['min_device_mass'],1.,places=5)
                self.assertAlmostEqual(audit['max_device_mass'],1.,places=5)
            self.assertEqual(protected,{p:sha(p) for p in protected})

    def test_frozen_sample_corruption_and_changed_horizon_still_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);session,c=cached_fixture(root);ref=frozen_reference(c);c['sampling_reference']=str(ref)
            from training_selector import selection
            seal=read(ref);recipe=initial_plans()[0]
            args=(seal['full_training_samples'],seal['positive_samples'],seal['full_membership_sha256'])
            with self.assertRaisesRegex(ValueError,'7-day'):selection(c,dict(recipe,label_horizon_days=15),*args)
            with (ref.parent/'weights.npy').open('ab') as f:f.write(b'changed')
            with self.assertRaisesRegex(ValueError,'Frozen sampling input changed'):selection(c,dict(recipe,base_weight='device'),*args)


class FailureClassification(unittest.TestCase):
    def test_worker_receipts_use_exception_not_path_or_message_keywords(self):
        from worker_errors import WorkerFailure
        with tempfile.TemporaryDirectory(prefix='memory_cuda_disk_') as tmp:
            root=Path(tmp);script=root/'worker.py';log=root/'execution.log'
            cases=[("raise ValueError('memory cuda scratch budget exceeded')",'training'),
                   ("raise MemoryError('allocation failed')",'resource'),
                   ("raise OSError(28, 'No space left on device')",'resource'),
                   ("from worker_errors import ResourceLimitError\nraise ResourceLimitError('matrix too large')",'resource'),
                   ("from xgboost.core import XGBoostError\nraise XGBoostError('cudaErrorMemoryAllocation: out of memory')",'resource'),
                   ("from xgboost.core import XGBoostError\nraise XGBoostError('CUDA input contains invalid labels')",'training')]
            for source,expected in cases:
                with self.subTest(expected=expected,source=source):
                    script.write_text(source,encoding='utf-8')
                    with self.assertRaises(WorkerFailure) as err:process([str(script)],log,30)
                    self.assertEqual(err.exception.category,expected)
                    self.assertEqual(read(log.with_suffix('.failure.json'))['category'],expected)
            # A worker that exits without a receipt must not inherit the previous category.
            save(log.with_suffix('.failure.json'),dict(category='resource'))
            script.write_text('import os\nos._exit(2)',encoding='utf-8')
            with self.assertRaises(WorkerFailure) as err:process([str(script)],log,30)
            self.assertEqual(err.exception.category,'training')

    def test_resource_failure_without_writable_receipt_still_stops(self):
        from worker_errors import WorkerFailure, RESOURCE_EXIT_CODE
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);script=root/'worker.py';log=root/'execution.log'
            script.write_text("from pathlib import Path\ndef fail(*a,**kw): raise OSError(28, 'Disk full')\nPath.write_text=fail\nraise MemoryError('allocation failed')",encoding='utf-8')
            with self.assertRaises(WorkerFailure) as err:process([str(script)],log,30)
            self.assertEqual(err.exception.category,'resource')
            self.assertEqual(err.exception.returncode,RESOURCE_EXIT_CODE)
            self.assertFalse(log.with_suffix('.failure.json').exists())

    def test_real_resource_failure_preserves_pending_and_stops(self):
        from worker_errors import WorkerFailure
        with tempfile.TemporaryDirectory() as tmp:
            session,c=cached_fixture(Path(tmp));loop=Loop(session);trial=loop.queue(initial_plans()[0])
            out=loop.folder(trial);out.mkdir(parents=True)
            (out/'candidate.py').write_text(source_for(trial['recipe']),encoding='utf-8')
            with patch('agent.process',side_effect=WorkerFailure('allocation failed','resource')) as worker:
                with self.assertRaisesRegex(RuntimeError,'Resource/timeout'):loop.train()
                self.assertEqual(worker.call_count,1)
            self.assertEqual(trial['status'],'pending')
            self.assertEqual(loop.s['status'],'needs_attention')
            self.assertEqual(loop.s['pending']['trial'],trial['id'])
    def test_memory_directory_does_not_turn_normal_error_into_resource_retry(self):
        with tempfile.TemporaryDirectory(prefix='memory_review_') as tmp:
            root=Path(tmp);session,c=cached_fixture(root);loop=Loop(session)
            trial=loop.queue(initial_plans()[0]);folder=loop.folder(trial);folder.mkdir(parents=True)
            (folder/'candidate.py').write_text(source_for(trial['recipe']),encoding='utf-8')
            def fail(*args):
                (folder/'execution.log').write_text('ValueError: Invalid feature specification\n',encoding='utf-8')
                raise ValueError('Worker exit=1. See '+str(folder/'execution.log'))
            with patch('agent.process',side_effect=fail) as worker:
                loop.train()
                self.assertEqual(trial['status'],'failed')
                self.assertIsNone(loop.s['pending'])
                self.assertEqual(loop.s['last_failure']['trial'],trial['id'])
                from search_step import step
                with self.assertRaisesRegex(ValueError,'Baseline did not complete'):step(session)
                self.assertEqual(worker.call_count,1)


if __name__=='__main__':unittest.main()
