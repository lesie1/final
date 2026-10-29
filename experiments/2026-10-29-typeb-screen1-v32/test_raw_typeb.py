import sys, unittest
from pathlib import Path
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent
runtime=R/'runtime'
if not runtime.exists():runtime=R.parent/'v32_review_8ebafc8'
sys.path.insert(0,str(runtime));sys.path.insert(0,str(R))
import raw_typeb as r

def frame():
    return pd.DataFrame([dict(LogTime=t,burst_info=bits,RankId=0,BankgroupId=0,
        BankId=1,RowId=10,ColumnId=20,deviceID=chip,error_type_full_name='CE.READ')
        for t,bits,chip in [(100,None,np.nan),(200,"{'0':[64], '5':[65]}",16),
                            (300,"{'1':[66]}",16),(100000,"{'1':[68]}",17)]])

class RawB(unittest.TestCase):
    def test_causal_prefix_and_future_patterns(self):
        ev=r.events(frame());a=np.array([90,100,150,200,250,300])
        full=r.features_at(ev,a); prefix=r.features_at(r.events(frame().iloc[:3]),a)
        np.testing.assert_allclose(full,prefix,equal_nan=True)
        for i,t in enumerate(a):
            np.testing.assert_allclose(full[i],r.features_at(ev[ev.time<=t].reset_index(drop=True),[t])[0],equal_nan=True)
    def test_unknown_not_zero_or_fake_chip(self):
        ev=r.events(frame());out=r.features_at(ev,[100,200]);idx={n:i for i,n in enumerate(r.FEATURES)}
        self.assertTrue(np.isnan(out[0,idx['raw_6h_risky_events']]))
        self.assertTrue(np.isnan(out[0,idx['raw_life_distinct_chips']]))
        self.assertTrue(np.isnan(out[0,idx['raw_life_new_patterns_7d']]))
        self.assertEqual(out[0,idx['raw_6h_bit_observed_events']],0)
        self.assertEqual(out[1,idx['raw_life_distinct_chips']],1)
        self.assertEqual(out[1,idx['raw_life_new_patterns_7d']],1)
        self.assertEqual(out[1,idx['raw_6h_events']],2)
    def test_empty_and_names(self):
        self.assertEqual(len(r.FEATURES),69)
        self.assertEqual([x['name'] for x in r.mapping_features()],r.FEATURES)
        self.assertTrue(np.isnan(r.features_at(r.events(frame()).iloc[:0],[1,2])).all())
        self.assertEqual(r.features_at(r.events(frame()),[]).shape,(0,69))

if __name__=='__main__':unittest.main()
