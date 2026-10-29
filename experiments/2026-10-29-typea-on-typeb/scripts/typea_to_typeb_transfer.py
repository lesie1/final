"""Frozen TypeA -> TypeB inference, explicitly recording unavailable input features."""
import os,sys,json,time,copy,shutil,hashlib,traceback
from pathlib import Path
R=Path('/root/autodl-tmp/final_v31_20261028');OUT=R/'typea_to_typeb_20261029'
PREVIOUS=R/'rejected_comparison_20261029';BASE=R/'final_test_20261029'
sys.path.insert(0,str(R/'runtime'));sys.dont_write_bytecode=True
from common import read,save,sha,require
from history_math import verify
from feature_programs import materialize,program
from prediction import predict
from rerank import load_model
from evaluation import metrics,lead_summary
from agent import receipt

def event(name,**kw):
 row=dict(at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),event=name,**kw)
 with (OUT/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
 print(json.dumps(row,ensure_ascii=False),flush=True)

def original_unchanged(frozen):
 for label,e in frozen['sessions'].items():
  for n,h in e['seals'].items():require(sha(Path(e['path'])/n)==h,'Original session changed: '+label+'/'+n)
 for n,h in frozen['runtime_hashes'].items():require(sha(R/'runtime'/n)==h,'Release changed: '+n)
 for t in frozen['models']:
  require(sha(Path(t['folder'])/'model.ubj')==t['model_sha256'],'Frozen source model changed')
  require(sha(Path(t['folder'])/'result.json')==t['result_sha256'],'Frozen source result changed')

def freeze():
 OUT.mkdir(exist_ok=True);require(not (OUT/'frozen.json').exists(),'Existing transfer freeze; inspect before rerunning')
 summary=read(PREVIOUS/'summary.json');source=(R/'typea/session').resolve();target=(R/'typeb/session').resolve()
 # Selection uses only previously observed TypeA TEST results, never TypeB transfer scores.
 ranked=sorted(summary['tasks']['typea']['candidates'],key=lambda x:(-x['test_f1'],x['trial']))[:5]
 ids=[x['trial'] for x in ranked]+['trial_001']
 base_result=read(source/'trials/trial_001/attempt_01/result.json');a_features=base_result['features']
 cache=BASE/'typeb/cache_test';m=verify(cache)
 require(set(m['features']).issubset(a_features),'Unexpected unmatched TypeB feature')
 train_a=Path(read(source/'config.json')['active_cache']);a_devices=set(read(train_a/'devices.json'));b_devices=set(read(cache/'devices.json'))
 require(not a_devices.intersection(b_devices),'Source and target device identities overlap')
 original_source_config=read(source/'config.json');target_config=read(target/'config.json')
 for n in ['evaluation_protocol','minimum_lead_seconds','maximum_lead_seconds']:
  require(original_source_config[n]==target_config[n],'Source and target evaluation rules differ: '+n)
 frozen=dict(created_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),study='TypeA frozen models on TypeB under unavailable-feature input mapping',selection_rule='Top five TypeA supplementary TEST F1 candidates, tie by trial ID; additionally include original TypeA baseline.',top_five=ids[:-1],control=ids[-1],independent_blind_test=False,source_models_selected_using_typea_test=True,target_test_previously_exposed=True,retrained=False,threshold_reselected=False,api_calls=0,sessions={},models=[],runtime_hashes=read(R/'release.json')['runtime_hashes'],target_base_cache=str(cache),target_manifest_sha256=sha(cache/'manifest.json'),target_test_window=m['windows']['test'],source_base_features=a_features,target_features=m['features'],common_features=[x for x in a_features if x in m['features']],unavailable_source_features=[x for x in a_features if x not in m['features']],mapping_policy='Exact canonical feature name and time-window match; preserve original TypeB values and missingness. Unavailable TypeA columns are NaN, not zero. No aggregation of 15m/1h into unsupported 6h or raw-event features.',mapping_assumption='Shared official feature names and windows are used as the available correspondence; different hardware distributions and unavailable columns limit transfer interpretation.',source_target_device_overlap=0,baseline_files={})
 for label,folder in [('typea',source),('typeb',target)]:
  frozen['sessions'][label]=dict(path=str(folder),seals={n:sha(folder/n) for n in ['events.jsonl','state.json','config.json','mapping.json','code_seal.json']})
 for label,folder in [('native_typeb_baseline',BASE/'typeb'),('native_typeb_trial_013',PREVIOUS/'typeb/trial_013'),('historical_typeb_trial_015',PREVIOUS/'historical_typeb/old_trial_015')]:
  rep=read(folder/'report_test.json');metric=rep.get('metrics',rep.get('current_full_cohort'))
  frozen['baseline_files'][label]=dict(folder=str(folder),report_sha256=sha(folder/'report_test.json'),table_sha256=sha(folder/'devices_test.csv'),metrics=metric)
 copied=set()
 def copy_program(key):
  if key=='base' or key in copied:return
  spec=program(original_source_config,key);copy_program(spec['parent']);dest=OUT/'feature_programs'/key;dest.mkdir(parents=True,exist_ok=True)
  for n in ['program.json','transform.py']:shutil.copy2(source/'feature_programs'/key/n,dest/n)
  copied.add(key)
 for trial in ids:
  folder=source/'trials'/trial/'attempt_01';rr=receipt(folder);copy_program(rr['recipe']['feature_program'])
  source_score=next(x for x in [summary['tasks']['typea']['baseline']]+summary['tasks']['typea']['candidates'] if x['trial']==trial)
  frozen['models'].append(dict(trial=trial,role='control' if trial=='trial_001' else 'top_five',folder=str(folder),model_sha256=sha(folder/'model.ubj'),result_sha256=sha(folder/'result.json'),receipt_sha256=sha(folder/'complete.json'),threshold=rr['metrics']['threshold'],features=rr['features'],recipe=rr['recipe'],source_typea_test=source_score))
 frozen['feature_program_hashes']={str(p.relative_to(OUT)):sha(p) for p in (OUT/'feature_programs').rglob('*') if p.is_file()}
 original_unchanged(frozen);save(OUT/'frozen.json',frozen)
 event('transfer_frozen',top_five=frozen['top_five'],source_baseline='trial_001',shared_features=len(frozen['common_features']),missing_source_features=len(frozen['unavailable_source_features']),missing_policy='NaN',source_target_device_overlap=0)
 return frozen

def aligned_cache(frozen):
 import numpy as np,pandas as pd,pyarrow.parquet as pq
 source=Path(frozen['target_base_cache']);original=read(source/'manifest.json');out=OUT/'aligned_typeb_cache_test';out.mkdir(exist_ok=True)
 expected=frozen['source_base_features'];missing=frozen['unavailable_source_features'];shared=frozen['target_features'];rows=0
 if (out/'manifest.json').exists():return out,verify(out)
 event('feature_alignment_started',shared=len(shared),unavailable=len(missing))
 files={}
 for rel in original['files']:
  p=source/rel;dest=out/rel;dest.parent.mkdir(parents=True,exist_ok=True)
  if rel.startswith('rows/') and rel.endswith('.parquet'):
   frame=pq.read_table(p).to_pandas();meta=[n for n in frame.columns if n not in shared]
   aligned=frame.reindex(columns=expected+meta)
   for n in missing:aligned[n]=np.full(len(frame),np.nan,dtype=np.float32)
   require(np.array_equal(aligned[shared].to_numpy(),frame[shared].to_numpy(),equal_nan=True),'Shared values changed during alignment')
   require(aligned[meta].equals(frame[meta]),'Metadata changed during alignment')
   require(aligned[missing].isna().all().all(),'Unavailable inputs are not missing')
   aligned.to_parquet(dest,index=False);rows+=len(frame)
  else:shutil.copy2(p,dest)
  files[rel]=sha(dest)
 m=copy.deepcopy(original);m['features']=expected;m['files']=files
 m['transfer_input_spec']=dict(source_manifest_sha256=frozen['target_manifest_sha256'],available_features=shared,unavailable_features=missing,missing_policy='NaN')
 save(out/'manifest.json',m);verify(out)
 event('feature_alignment_complete',rows=rows,shared_values_bitwise_verified=True,metadata_unchanged=True,manifest_sha256=sha(out/'manifest.json'))
 return out,m

def component_coverage(model,names,available):
 if hasattr(model,'primary'):
  return dict(primary=component_coverage(model.primary,names,available),auxiliary=component_coverage(model.auxiliary,names,available))
 if hasattr(model,'stage1'):
  return dict(stage1=component_coverage(model.stage1,names,available),stage2=component_coverage(model.stage2,names,available))
 values=model.get_score(importance_type='total_gain');total=0.;missing=0.;used_missing=[]
 for k,v in values.items():
  name=names[int(k[1:])] if k.startswith('f') and k[1:].isdigit() else k
  require(name in names,'Unknown model split feature: '+k);total+=v
  if name not in available:missing+=v;used_missing.append(name)
 return dict(split_features=len(values),unavailable_split_features=used_missing,unavailable_fraction_of_training_split_gain=missing/total if total else None,note='Training split gain is descriptive importance, not a causal contribution or a target-domain performance estimate.')

def run():
 import pandas as pd,numpy as np
 frozen=freeze();cache,m=aligned_cache(frozen)
 source_config=read(Path(frozen['sessions']['typea']['path'])/'config.json');c=read(Path(frozen['sessions']['typeb']['path'])/'config.json')
 c.update(output_root=str(OUT),active_cache=str(cache),device='cpu',num_threads=2,score_workers=4,max_total_features=source_config['max_total_features'])
 baseline=pd.read_csv(BASE/'typeb/devices_test.csv');bmetric=frozen['baseline_files']['native_typeb_baseline']['metrics']
 reports=[]
 for t in frozen['models']:
  started=time.monotonic();trial=t['trial'];folder=OUT/'trials'/trial;folder.mkdir(parents=True,exist_ok=True)
  event('prediction_started',trial=trial,threshold=t['threshold'],source_model_sha256=t['model_sha256'])
  cc,mm=materialize(c,t['recipe']['feature_program'],m,final_test=True)
  require(mm['features']==t['features'],'Model feature order differs')
  # Report target columns that are actually finite; derived features may also be wholly absent.
  import pyarrow.parquet as pq
  available=set()
  for p in (Path(cc['active_cache'])/'rows').glob('*.parquet'):
   f=pq.read_table(p,columns=mm['features']).to_pandas();available.update(f.columns[f.notna().any()].tolist())
  model=load_model(Path(t['folder'])/'model.ubj',2)
  coverage=component_coverage(model,mm['features'],available)
  table,stats=predict(model,cc,t['recipe'],mm,'test',t['threshold']);scored=metrics(table,t['threshold'])
  for n in ['sn_name','device','failure','status','positive','observed_start','observed_end']:
   require(table[n].reset_index(drop=True).equals(baseline[n].reset_index(drop=True)),'Target cohort changed: '+n)
  require(scored['eligible_devices']==7119 and scored['eligible_fault_devices']==12,'Target denominator changed')
  table.to_csv(folder/'devices_test.csv',index=False)
  table.loc[table.status.eq('eligible')&table.positive].to_csv(folder/'fault_devices_test.csv',index=False)
  use=table.status.eq('eligible');pair=dict(new_hits=int((use&table.hit&~baseline.hit).sum()),lost_hits=int((use&baseline.hit&~table.hit).sum()),new_false_alarms=int((use&table.fp&~baseline.fp).sum()),removed_false_alarms=int((use&baseline.fp&~table.fp).sum()))
  result=dict(source_task='typea',target_task='typeb',source_trial=trial,role=t['role'],source_typea_test=t['source_typea_test'],model_sha256=t['model_sha256'],metrics=scored,delta_vs_native_typeb_baseline={k:scored[k]-bmetric[k] for k in ['f1','precision','recall','TP','FP','FN']},pair_vs_native_typeb_baseline=pair,feature_count=len(mm['features']),actually_available_columns=sorted(available),unavailable_columns=[n for n in mm['features'] if n not in available],model_split_coverage=coverage,lead_summary=lead_summary(table),prediction_stats=stats,test_window=mm['windows']['test'],threshold_reselected=False,retrained=False,api_calls=0,independent_blind_test=False,limitation='Source model input schema exceeds the target available schema; unavailable inputs are NaN. This experiment does not isolate hardware-domain shift from feature loss and threshold calibration.',seconds=time.monotonic()-started)
  save(folder/'report_test.json',result);save(folder/'input_audit.json',dict(freeze_sha256=sha(OUT/'frozen.json'),cache=str(cc['active_cache']),manifest_sha256=sha(Path(cc['active_cache'])/'manifest.json'),recipe=t['recipe'],features=mm['features']))
  original_unchanged(frozen);require(sha(Path(t['folder'])/'complete.json')==t['receipt_sha256'],'Source receipt changed')
  save(folder/'complete.json',dict(files={n:sha(folder/n) for n in ['devices_test.csv','fault_devices_test.csv','report_test.json','input_audit.json']},source_model_unchanged=True,original_sessions_unchanged=True))
  reports.append(result);event('prediction_complete',trial=trial,metrics=scored,seconds=result['seconds'])
 save(OUT/'summary.json',dict(frozen_sha256=sha(OUT/'frozen.json'),reports=reports,native_comparators=frozen['baseline_files'],independent_blind_test=False,unavailable_source_features=frozen['unavailable_source_features'],retrained=False,threshold_reselected=False,api_calls=0))
 event('transfer_complete',models=len(reports),source_sessions_preserved=True)

if __name__=='__main__':
 try:run()
 except Exception as e:
  OUT.mkdir(exist_ok=True);event('transfer_failed',error=str(e));traceback.print_exc();raise
