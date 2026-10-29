"""Authorized supplementary TEST replay, without changing the sealed release/search."""
import os,sys,time,copy,shutil,json,traceback,fcntl,subprocess
from pathlib import Path
R=Path('/root/autodl-tmp/final_v31_20261028')
OUT=R/'rejected_comparison_20261029'
BASE=R/'final_test_20261029'
sys.path.insert(0,str(R/'runtime'));sys.dont_write_bytecode=True
from common import read,save,sha,require
from history_math import verify
from agent import receipt
from feature_programs import materialize,program
from rerank import load_model
from prediction import predict
from evaluation import metrics,lead_summary

def freeze():
 OUT.mkdir(exist_ok=True)
 require(not (OUT/'frozen.json').exists(),'Freeze already exists; do not overwrite')
 release=read(R/'release.json')
 for n,h in release['runtime_hashes'].items():require(sha(R/'runtime'/n)==h,'Release changed: '+n)
 frozen=dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),selection_rule='All complete rejected candidates whose selection VAL F1 OR CV F1 exceeds trial_001; frozen before inspecting candidate TEST outcomes.',independent_blind_test=False,supplementary_comparison=True,retraining=False,threshold_reselection=False,api_calls=0,tasks={})
 for label in ['typea','typeb']:
  session=(R/label/'session').resolve();c=read(session/'config.json');state=read(session/'state.json')
  base=state['trials'][0];require(base['id']=='trial_001' and state['best']=='trial_001','Unexpected baseline')
  target=OUT/label;target.mkdir(exist_ok=True)
  cache=Path('/root/autodl-tmp/typea_v2_frozen_test_20261028/out/raw_cache_test') if label=='typea' else BASE/label/'cache_test'
  m=verify(cache)
  audit=read(BASE/label/'input_audit.json');require(sha(cache/'manifest.json')==audit['test_manifest_sha256'],'Base TEST cache differs')
  for n,h in read(BASE/label/'complete.json')['files'].items():require(sha(BASE/label/n)==h,'Baseline artifact changed')
  e=dict(session=str(session),source_seals={n:sha(session/n) for n in ['config.json','state.json','mapping.json','code_seal.json']},base_cache=str(cache),base_manifest_sha256=sha(cache/'manifest.json'),test_window=m['windows']['test'],baseline_report=read(BASE/label/'report_test.json'),baseline_files={n:sha(BASE/label/n) for n in ['devices_test.csv','report_test.json','complete.json']},candidates=[],program_files={})
  copied=set()
  def copy_program(key):
   if key=='base' or key in copied:return
   spec=program(c,key);copy_program(spec['parent'])
   dest=target/'feature_programs'/key;dest.mkdir(parents=True,exist_ok=True)
   for n in ['program.json','transform.py']:
    shutil.copy2(session/'feature_programs'/key/n,dest/n)
    e['program_files'][str(Path('feature_programs')/key/n)]=sha(dest/n)
   copied.add(key)
  for tr in state['trials'][1:]:
   if tr['status']!='complete' or tr.get('acceptance',{}).get('accepted',False):continue
   val_gain=tr['metrics']['f1']>base['metrics']['f1']
   cv=tr.get('cv',{}).get('metrics',{});cv_gain=cv.get('f1',-1)>base['cv']['metrics']['f1']
   if not (val_gain or cv_gain):continue
   folder=session/'trials'/tr['id']/'attempt_01';rr=receipt(folder)
   require(rr['metrics']==tr['metrics'],'State/result differs')
   copy_program(rr['recipe']['feature_program'])
   e['candidates'].append(dict(trial=tr['id'],folder=str(folder),model=str(folder/'model.ubj'),model_sha256=sha(folder/'model.ubj'),trial_receipt_sha256=sha(folder/'complete.json'),result_sha256=sha(folder/'result.json'),recipe=rr['recipe'],features=rr['features'],threshold=rr['metrics']['threshold'],validation=rr['metrics'],cv=cv,acceptance=tr['acceptance'],selection_val_improved=val_gain,cv_improved=cv_gain))
  frozen['tasks'][label]=e
  print('FROZEN',label,[x['trial'] for x in e['candidates']],flush=True)
 save(OUT/'frozen.json',frozen)

def unchanged(label,e):
 session=Path(e['session'])
 for n,h in e['source_seals'].items():require(sha(session/n)==h,'Original session changed: '+n)
 for n,h in read(R/'release.json')['runtime_hashes'].items():require(sha(R/'runtime'/n)==h,'Release changed: '+n)
 for n,h in e['baseline_files'].items():require(sha(BASE/label/n)==h,'Baseline changed: '+n)
 for n,h in e['program_files'].items():require(sha(OUT/label/n)==h,'Feature program changed: '+n)
 require(sha(Path(e['base_cache'])/'manifest.json')==e['base_manifest_sha256'],'Base TEST manifest changed')

def compare_tables(table,baseline,threshold,base_threshold):
 import pandas as pd
 from evaluation import classify
 for k in ['device','status','positive','failure','observed_start','observed_end']:
  require(table[k].reset_index(drop=True).equals(baseline[k].reset_index(drop=True)), 'Comparison cohort differs: '+k)
 b=classify(baseline,base_threshold);a=classify(table,threshold)
 use=a.status.eq('eligible');new=use&a.hit&~b.hit;lost=use&b.hit&~a.hit
 return dict(new_hits=int(new.sum()),lost_hits=int(lost.sum()),new_false_alarms=int((use&a.fp&~b.fp).sum()),removed_false_alarms=int((use&b.fp&~a.fp).sum()),new_hit_device_ids=a.loc[new,'device'].tolist(),lost_hit_device_ids=a.loc[lost,'device'].tolist(),same_cohort=True)

def evaluate_group(label,key):
 import pandas as pd
 e=read(OUT/'frozen.json')['tasks'][label];root=OUT/label
 lock=(root/('group_'+key+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 candidates=[t for t in e['candidates'] if t['recipe']['feature_program']==key]
 require(candidates,'Empty group');unchanged(label,e)
 c=read(Path(e['session'])/'config.json');c.update(output_root=str(root),active_cache=e['base_cache'],device='cpu',num_threads=2,score_workers=4)
 m=read(Path(e['base_cache'])/'manifest.json')
 status=root/('group_'+key+'_status.json')
 def stage(s,**kw):
  save(status,dict(stage=s,at=time.time(),pid=os.getpid(),program=key,**kw));print(label,key[:8],s,kw,flush=True)
 stage('materializing_frozen_test_features')
 cc,mm=materialize(c,key,m,final_test=True)
 require(mm['windows']['test']==e['test_window'],'TEST window changed')
 for n in ['device_metadata.parquet','devices.json','excluded.json']:
  require(sha(Path(cc['active_cache'])/n)==sha(Path(e['base_cache'])/n),'Feature replay changed cohort')
 baseline=pd.read_csv(BASE/label/'devices_test.csv');base_metric=e['baseline_report']['metrics']
 # trial_013 first where present, then remaining frozen candidates in numeric order.
 candidates.sort(key=lambda t:(t['trial']!='trial_013',t['trial']))
 for t in candidates:
  target=root/t['trial'];target.mkdir(exist_ok=True)
  if (target/'complete.json').exists():
   for n,h in read(target/'complete.json')['files'].items():require(sha(target/n)==h,'Completed replay changed')
   continue
  tick=time.monotonic();stage('predicting',trial=t['trial'])
  require(sha(Path(t['model']))==t['model_sha256'],'Frozen model changed')
  require(sha(Path(t['folder'])/'complete.json')==t['trial_receipt_sha256'],'Trial receipt changed')
  require(sha(Path(t['folder'])/'result.json')==t['result_sha256'],'Trial result changed')
  require(mm['features']==t['features'],'Trained feature order differs')
  model=load_model(t['model'],2)
  table,stats=predict(model,cc,t['recipe'],mm,'test',t['threshold'])
  scored=metrics(table,t['threshold']);require(scored['threshold']==t['threshold'],'Threshold changed')
  comparison=compare_tables(table,baseline,t['threshold'],base_metric['threshold'])
  comparison['delta']={k:scored[k]-base_metric[k] for k in ['f1','precision','recall','TP','FP','FN']}
  table.to_csv(target/'devices_test.csv',index=False)
  result=dict(task=label,trial=t['trial'],model_sha256=t['model_sha256'],metrics=scored,validation=t['validation'],cv=t['cv'],comparison_vs_trial_001=comparison,lead_summary=lead_summary(table),prediction_stats=stats,threshold_reselected=False,retrained=False,api_calls=0,independent_blind_test=False,test_window=e['test_window'],seconds=time.monotonic()-tick)
  save(target/'report_test.json',result)
  save(target/'input_audit.json',dict(freeze_sha256=sha(OUT/'frozen.json'),cache=cc['active_cache'],manifest_sha256=sha(Path(cc['active_cache'])/'manifest.json'),feature_count=len(mm['features']),recipe=t['recipe'],threshold=t['threshold'],model_sha256=t['model_sha256']))
  unchanged(label,e)
  save(target/'complete.json',dict(files={n:sha(target/n) for n in ['devices_test.csv','report_test.json','input_audit.json']},source_search_preserved=True,release_code_unchanged=True))
  print('RESULT',label,t['trial'],json.dumps(scored),flush=True)
 stage('complete')

def launch():
 frozen=read(OUT/'frozen.json');jobs=[]
 env=os.environ.copy();env.pop('DEEPSEEK_API_KEY',None);env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
 # Every feature program has its own worker; shared parent program groups must be sequential.
 for label in ['typea','typeb']:
  keys=list(dict.fromkeys(t['recipe']['feature_program'] for t in frozen['tasks'][label]['candidates']))
  if label=='typeb':
   # One worker orchestrates the small TypeB programs, avoiding parent-cache races.
   log=OUT/label/'worker.log'
   with log.open('ab') as f:p=subprocess.Popen([sys.executable,__file__,'label',label],stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
   jobs.append(dict(task=label,group='all',pid=p.pid,log=str(log)))
  else:
   for key in keys:
    # TypeA keys are checked for base-only parents below before concurrent launch.
    if key!='base':require(program(dict(output_root=str(OUT/label)),key)['parent']=='base','Dependent TypeA program requires sequential scheduling')
    log=OUT/label/('worker_'+key+'.log')
    with log.open('ab') as f:p=subprocess.Popen([sys.executable,__file__,'group',label,key],stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    jobs.append(dict(task=label,group=key,pid=p.pid,log=str(log)))
 save(OUT/'jobs.json',jobs);print(json.dumps(jobs),flush=True)

if __name__=='__main__':
 try:
  action=sys.argv[1]
  if action=='freeze':freeze()
  elif action=='launch':launch()
  elif action=='group':evaluate_group(sys.argv[2],sys.argv[3])
  elif action=='label':
   label=sys.argv[2];e=read(OUT/'frozen.json')['tasks'][label]
   for key in dict.fromkeys(t['recipe']['feature_program'] for t in e['candidates']):evaluate_group(label,key)
  else:raise ValueError(action)
 except Exception:traceback.print_exc();raise
