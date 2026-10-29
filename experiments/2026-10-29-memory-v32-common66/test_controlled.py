"""Integration checks for projection, mature-only matrices, device masks and CPU fitting."""
import copy,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import controlled as ctl
from common import read,sha
from tests.test_v2_loop import fixture
from experiment_contract import initial_plans
from sample_builder import targets

class CommonControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='common66_test_');cls.root=Path(cls.temp.name)
        cls.sources={};cls.recipe=dict(initial_plans()[0],base_weight='device')
        for name,n in [('a',60),('b',30)]:
            root=cls.root/name;root.mkdir()
            session,c=fixture(root,n)
            dest=root/'projected';m=ctl.project_cache(c['active_cache'],dest,['x','z'])
            c=dict(c,active_cache=str(dest),sampling_reference=None)
            folder=root/'matrix';ctl.build_matrix(c,m,cls.recipe,folder)
            cls.sources[name]=(c,m,folder)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_projection_preserves_values_missingness_and_registry(self):
        a=self.root/'a'
        source=pq.read_table(a/'cache/rows/part-00000.parquet',columns=['x','z']).to_pandas()
        dest=pq.read_table(a/'projected/rows/part-00000.parquet',columns=['x','z']).to_pandas()
        pd.testing.assert_frame_equal(source,dest)
        self.assertEqual(sha(a/'cache/device_metadata.parquet'),sha(a/'projected/device_metadata.parquet'))
        self.assertEqual(read(a/'projected/manifest.json')['features'],['x','z'])

    def test_matrix_uses_only_mature_train_rows_and_unit_device_weights(self):
        for c,m,folder in self.sources.values():
            rows=pd.concat([pq.read_table(p).to_pandas() for p in sorted((Path(c['active_cache'])/'rows').glob('*.parquet'))],ignore_index=True)
            keep,y=targets(rows,7,m['windows']['train'][1],c['minimum_lead_seconds'])
            keep &= (rows.time>=m['windows']['train'][0]).to_numpy() & (rows.history_rows>0).to_numpy()
            x,labels,w,d=ctl.component_arrays(folder)
            np.testing.assert_equal(x,rows.loc[keep,['x','z']].to_numpy(np.float32))
            np.testing.assert_array_equal(labels,y[keep])
            sums=np.bincount(d,weights=w);np.testing.assert_allclose(sums[sums>0],1.,rtol=2e-6)
            self.assertTrue((rows.loc[keep,'time']+900+7*86400 < m['windows']['train'][1]).all())

    def test_small_a_matches_training_positive_and_other_device_counts(self):
        af=self.sources['a'][2];bf=self.sources['b'][2]
        chosen=ctl.sample_small_a(af,bf,42)
        np.testing.assert_array_equal(chosen,ctl.sample_small_a(af,bf,42))
        ap,an=ctl.eligible_devices(af);bp,bn=ctl.eligible_devices(bf)
        self.assertEqual(len(np.intersect1d(chosen,ap)),len(bp))
        self.assertEqual(len(np.intersect1d(chosen,an)),len(bn))

    def test_joint_fit_excludes_held_b_devices_and_preserves_weight_formula(self):
        ac,am,af=self.sources['a'];bc,bm,bf=self.sources['b']
        registry=pd.read_parquet(Path(bc['active_cache'])/'device_metadata.parquet')
        fold=ctl.folds(registry,bm['windows']['train'],3);mask=fold!=0
        out=self.root/'fit'
        b=ctl.fit([('typea',af,None),('typeb',bf,mask)],self.recipe,out,4,device='cpu',threads=1)
        audit=read(out/'train_audit.json');x,y,w,d=ctl.component_arrays(bf)
        self.assertEqual(audit['components'][1]['rows'],int(mask[d].sum()))
        totals=audit['components'];ratio=sum(t['weighted_negative'] for t in totals)/sum(t['weighted_positive'] for t in totals)
        self.assertAlmostEqual(audit['params']['scale_pos_weight'],ratio,places=5)
        scores=ctl.score_devices(b,[2,4],bc,self.recipe,bm,ctl.periods(bm,bc),np.flatnonzero(fold==0))
        self.assertTrue(set(scores.device)<=set(np.flatnonzero(fold==0)))
        self.assertFalse(scores.duplicated(['device','period','step']).any())
        self.assertEqual(b.num_features(),2)
        self.assertFalse((out/'combined_matrix.npy').exists())

if __name__=='__main__':unittest.main()
