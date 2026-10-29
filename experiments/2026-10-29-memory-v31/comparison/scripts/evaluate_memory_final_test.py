"""Authorized frozen inference only; imports the sealed release without modifying it."""
import sys,os,time,copy,shutil,json,traceback,fcntl
from pathlib import Path
R=Path('/root/autodl-tmp/final_v31_20261028');OUT=R/'final_test_20261029'
sys.path.insert(0,str(R/'runtime'));sys.dont_write_bytecode=True
from common import read,save,sha,require
from history_math import verify
from data_ingest import normalize
from agent import receipt
from rerank import load_model
from prediction import predict
from evaluation import metrics,lead_summary

def main(label):
 target=OUT/label;target.mkdir(exist_ok=True)
 lock=(target/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 frozen=read(OUT/'frozen.json');entry=next(e for e in frozen['models'] if e['label']==label)
 session=Path(entry['session']);c=read(session/'config.json');release=read(R/'release.json')
 def unchanged():
  for n,h in entry['source_seals'].items():require(sha(session/n)==h,'Original search session changed: '+n)
  require(sha(Path(entry['model']))==entry['model_sha256'],'Frozen model changed')
  for n,h in release['runtime_hashes'].items():require(sha(R/'runtime'/n)==h,'Release code changed: '+n)
 def status(stage):
  save(target/'status.json',dict(stage=stage,at=time.time(),pid=os.getpid()));print(label,stage,flush=True)
 unchanged()
 if (target/'complete.json').exists():
  for n,h in read(target/'complete.json')['files'].items():require(sha(target/n)==h,'Test artifact changed')
  print('ALREADY_COMPLETE',label,flush=True);return
 started=time.monotonic();status('verifying_frozen_inputs')
 original=verify(Path(entry['trainval_cache']))
 require(sha(Path(entry['trainval_cache'])/'manifest.json')==entry['trainval_manifest_sha256'],'TRAIN/VAL cache changed')
 folder=Path(entry['model']).parent;rr=receipt(folder)
 require(rr['recipe']==entry['recipe'] and rr['metrics']['threshold']==entry['threshold'],'Frozen recipe or threshold changed')
 require(sha(folder/'complete.json')==entry['trial_receipt_sha256'],'Trial receipt changed')
 if label=='typea':
  status('verifying_existing_typea_test_cache')
  cache=Path('/root/autodl-tmp/typea_v2_frozen_test_20261028/out/raw_cache_test');m=verify(cache)
  require(m['raw_feature_spec']['raw_events_module_sha256']==sha(R/'runtime/raw_events.py'),'Raw feature parser differs')
  require(m['source_receipts']==original['source_receipts'],'Source data receipts differ')
  require(m['coverage']==original['coverage'],'Follow-up policy differs')
  preparation='Reused verified TEST cache with matching source receipts, mapping, windows, features, and raw feature parser hash.'
 else:
  status('preparing_typeb_test_cache')
  for item in original['source_receipts']:
   require(sha(Path(c['data_dir'])/item['path'])==item['sha256'],'Source data changed: '+item['path'])
  (target/'cache').mkdir(exist_ok=True)
  shutil.copy2(Path(entry['trainval_cache'])/'devices.json',target/'cache/devices.json')
  nc=copy.deepcopy(c);nc['output_root']=str(target)
  normalize(nc,read(session/'mapping.json'),original['spec']['inventory'],final_test=True)
  cache=target/'cache_test';m=verify(cache)
  preparation='Normalized same source features and tickets up to the frozen TEST end with identical device IDs and mapping.'
 require(m['test_opened'] and m['spec']['test'],'Not a TEST cache')
 require(m['windows']==original['windows'] and m['windows']['test']==entry['test_window'],'Test window changed')
 require(m['features']==entry['features'],'Feature order or definitions changed')
 for k in ['mapping','inventory','coverage']:require(m['spec'][k]==original['spec'][k],'Cache identity differs: '+k)
 previous_ids=read(Path(entry['trainval_cache'])/'devices.json');test_ids=read(cache/'devices.json')
 require(test_ids[:len(previous_ids)]==previous_ids,'Device IDs changed')
 cc=copy.deepcopy(c);cc.update(output_root=str(target),active_cache=str(cache),num_threads=2,score_workers=8)
 save(target/'inference_config.json',cc)
 save(target/'input_audit.json',dict(freeze_sha256=sha(OUT/'frozen.json'),test_cache=str(cache),test_manifest_sha256=sha(cache/'manifest.json'),
  preparation=preparation,feature_count=len(m['features']),test_window=m['windows']['test'],test_cache_historically_used=(label=='typea'),
  historical_test_period_exposure=True,historical_exposure_note='Earlier project evaluations used this same TEST date range, including TypeA frozen test on 2026-10-28 and local TypeB frozen test archive dated 2026-10-26. This is not a pristine blind evaluation.'))
 status('predicting_at_frozen_threshold')
 model=load_model(entry['model'],2);require(model.num_features()==len(m['features']),'Model feature width differs')
 table,stats=predict(model,cc,entry['recipe'],m,'test',entry['threshold'])
 scored=metrics(table,entry['threshold'])
 require(scored['threshold']==entry['threshold'],'Threshold changed')
 table.to_csv(target/'devices_test.csv',index=False)
 table.loc[table.status.str.startswith('unknown_')].to_csv(target/'unknown_devices_test.csv',index=False)
 result=dict(task=label,trial=entry['trial'],model_sha256=entry['model_sha256'],metrics=scored,
  test_window=entry['test_window'],lead_summary=lead_summary(table),prediction_stats=stats,
  threshold_reselected=False,retrained=False,agent_calls=0,test_opened=True,independent_blind_test=False,
  seconds=time.monotonic()-started,note='Frozen selected model replay on reserved TEST dates; historical project use of these dates prevents a claim of a fresh independent blind test. Outcomes must not be used to retune or select another model.')
 save(target/'report_test.json',result);unchanged()
 save(target/'complete.json',dict(files={n:sha(target/n) for n in ['report_test.json','devices_test.csv','unknown_devices_test.csv','input_audit.json','inference_config.json']},
  frozen_sha256=sha(OUT/'frozen.json'),source_search_preserved=True,release_code_unchanged=True))
 status('complete');print('RESULT',json.dumps(scored),flush=True)

if __name__=='__main__':
 label=sys.argv[1];require(label in ['typea','typeb'],'Unknown task')
 try:main(label)
 except Exception as e:
  target=OUT/label;target.mkdir(exist_ok=True);save(target/'error.json',dict(error=str(e),at=time.time()));traceback.print_exc();raise
