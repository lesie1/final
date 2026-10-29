"""Replay historical frozen models on the current cohort, keeping old thresholds."""
import os,sys,json,zipfile,hashlib,time
from pathlib import Path
R=Path('/root/autodl-tmp/final_v31_20261028');OUT=R/'rejected_comparison_20261029';H=OUT/'historical_typeb';BASE=R/'final_test_20261029'
sys.path.insert(0,str(R/'runtime'));sys.dont_write_bytecode=True
from common import read,save,sha,require
from history_math import verify
from prediction import predict
from evaluation import metrics,lead_summary
from rerank import load_model
import pandas as pd

def main():
 H.mkdir(exist_ok=True);archive=H/'typeb_frozen_test_20261026.zip';z=zipfile.ZipFile(archive)
 oldf=json.loads(z.read('session/frozen.json'));oldm=json.loads(z.read('session/cache_test/manifest.json'))
 frozen=read(OUT/'frozen.json');e=frozen['tasks']['typeb'];cache=Path(e['base_cache']);m=verify(cache)
 require(m['features']==oldm['features'],'Historical feature order differs')
 require(m['windows']==oldm['windows'],'Historical time splits differ')
 oldsrc={Path(x['path']).name:x['sha256'] for x in oldm['source_receipts'] if x['path'].endswith('.feather')}
 newsrc={Path(x['path']).name:x['sha256'] for x in m['source_receipts'] if x['path'].endswith('.feather')}
 require(oldsrc==newsrc,'Source feature files differ')
 c=read(Path(e['session'])/'config.json');c.update(output_root=str(H),active_cache=str(cache),device='cpu',num_threads=2,score_workers=4)
 current=pd.read_csv(BASE/'typeb/devices_test.csv').set_index('sn_name',drop=False)
 oldtable=pd.read_csv(z.open('session/test/trial_001/devices_test.csv')).set_index('sn_name',drop=False)
 require(set(current.index)==set(oldtable.index),'Device inventory differs')
 oldtable=oldtable.loc[current.index]
 require((current.failure==oldtable.failure).all(),'Failure labels differ')
 eligible_old=oldtable.status.eq('eligible');require(current.loc[eligible_old,'status'].eq('eligible').all(),'Old cohort is not nested in current')
 require((current.loc[eligible_old,'positive']==oldtable.loc[eligible_old,'positive']).all(),'Positive status differs')
 seal=dict(old_archive_sha256=sha(archive),current_test_manifest_sha256=sha(cache/'manifest.json'),same_feature_files=len(oldsrc),same_features=True,same_time_windows=True,same_failure_labels=True,old_eligible_devices=int(eligible_old.sum()),current_eligible_devices=int(current.status.eq('eligible').sum()),historical_models=oldf['models'],threshold_reselected=False,retrained=False,independent_blind_test=False)
 require(not (H/'frozen.json').exists(),'Historical replay already frozen')
 save(H/'frozen.json',seal)
 rows=[]
 def add_row(name,table,threshold,scope):
  t=table.set_index('sn_name',drop=False).loc[current.index]
  if scope=='old_eligible_subset':t=t.loc[eligible_old]
  metric=metrics(t,threshold);rows.append(dict(model=name,scope=scope,**metric));return metric
 for old in oldf['models']:
  trial=old['trial'];folder=H/('old_'+trial);folder.mkdir(exist_ok=True)
  raw=z.read('session/trials/'+trial+'/attempt_01/model.ubj')
  require(hashlib.sha256(raw).hexdigest()==old['model_sha256'],'Old frozen model hash differs')
  (folder/'model.ubj').write_bytes(raw)
  model=load_model(folder/'model.ubj',2)
  table,stats=predict(model,c,old['recipe'],m,'test',old['threshold'])
  table.to_csv(folder/'devices_test.csv',index=False)
  full=add_row('old_'+trial,table,old['threshold'],'current_full_cohort')
  subset=add_row('old_'+trial,table,old['threshold'],'old_eligible_subset')
  original=json.loads(z.read('session/test/'+trial+'/metrics_test.json'))
  save(folder/'report_test.json',dict(current_full_cohort=full,old_eligible_subset=subset,original_report=original,prediction_stats=stats,lead_summary=lead_summary(table),model_sha256=old['model_sha256'],threshold_reselected=False))
  print('OLD_RESULT',trial,json.dumps(dict(full=full,subset=subset,original=original)),flush=True)
 for trial,folder in [('trial_001',BASE/'typeb'),('trial_013',OUT/'typeb/trial_013')]:
  table=pd.read_csv(folder/'devices_test.csv');threshold=read(folder/'report_test.json')['metrics']['threshold']
  for scope in ['current_full_cohort','old_eligible_subset']:add_row('v31_'+trial,table,threshold,scope)
 summary=dict(audit=seal,comparisons=rows,note='Comparison retains each model original frozen threshold. Old-cohort subset uses the identical 1908 serial numbers under current coverage; compare replay to original before attributing all changes to cohort. Historical and current sessions also differ in training/validation coverage and selection policy.')
 save(H/'summary.json',summary);pd.DataFrame(rows).to_csv(H/'comparison.csv',index=False,encoding='utf-8-sig')
 save(H/'complete.json',dict(files={str(p.relative_to(H)):sha(p) for p in H.rglob('*') if p.is_file() and p.name!='complete.json'}))
 print('HISTORICAL_COMPARISON',json.dumps(summary),flush=True)

if __name__=='__main__':main()
