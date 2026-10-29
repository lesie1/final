import json,os,sys,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,os.environ.get('V32_RUNTIME',str(Path(__file__).resolve().parent/'runtime')))
import publish_bundle as publication
from common import save,read

class Publication(unittest.TestCase):
    def test_complete_snapshot_has_all_seeds_correct_pooled_counts_and_event_receipts(self):
        with tempfile.TemporaryDirectory(prefix='v32_report_') as temp:
            root=Path(temp);arms=['B_only','A_only','A_plus_B','A_small'];seeds=[42,43,44]
            save(root/'status.json',dict(status='complete'))
            save(root/'release.json',dict(commit='synthetic'))
            save(root/'protocol.json',dict(controlled=dict(arms=arms,seeds=seeds)))
            m=dict(TP=1,FP=2,FN=3,f1=2/7,precision=1/3,recall=.25,threshold=.8)
            op=dict(source='cv_oof',step=50,threshold=.8)
            for label in ['typea','typeb']:
                save(root/label/'session/state.json',dict(best='trial_001',trials=[dict(id='trial_001',status='complete',metrics=m,cv=dict(metrics=m),acceptance=dict(accepted=True,rule='baseline'))]))
                save(root/label/'historical_test/summary.json',dict(reports=[dict(trial='trial_001',point='deployed',operating_point=op,metrics=m)]))
                (root/label/'session/events.jsonl').write_text(json.dumps(dict(event='synthetic_complete'))+'\n')
            reports=[dict(arm=arm,seed=seed,target=target,split=split,operating_point=op,metrics=m)
                for arm in arms for seed in seeds for target in ['typea','typeb'] for split in ['confirmation','test']]
            save(root/'controlled/evaluation/summary.json',dict(reports=reports))
            save(root/'typeb/supplemental_trial013/summary.json',dict(post_hoc_trial_selection=True,threshold_reselected=False,
                comparison=[dict(trial='trial_013',point='deployed',step=50,**m)]))
            with patch.object(publication,'R',root),patch.object(publication,'preserved',lambda:None):publication.bundle()
            summary=read(root/'publication/summary.json');self.assertEqual(len(summary['controlled']),48)
            self.assertEqual(len(summary['pooled']),24)
            self.assertEqual([summary['pooled'][0][k] for k in ['TP','FP','FN']],[2,4,6])
            self.assertAlmostEqual(summary['pooled'][0]['f1'],2/7)
            self.assertEqual(summary['supplemental'][0]['role'],'user_requested_post_hoc')
            self.assertEqual(summary['supplemental'][0]['trial'],'trial_013')
            with zipfile.ZipFile(root/'publication/snapshot.zip') as archive:
                manifest=json.loads(archive.read('export_manifest.json'))
                self.assertEqual(len(manifest['events']),2)
                self.assertTrue(all(v['lines']==1 for v in manifest['events'].values()))
                self.assertIn('report.html',manifest['files'])
                self.assertIn('report.md',manifest['files'])
                self.assertIn('supplemental_typeb_test.csv',manifest['files'])
                self.assertIn('typeb/supplemental_trial013/summary.json',manifest['files'])
                self.assertFalse(manifest['independent_blind_test'])

if __name__=='__main__':unittest.main()
