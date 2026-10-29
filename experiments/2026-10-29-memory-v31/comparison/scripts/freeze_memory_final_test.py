import sys,json,time
from pathlib import Path
R=Path('/root/autodl-tmp/final_v31_20261028');sys.path.insert(0,str(R/'runtime'));sys.dont_write_bytecode=True
from common import read,save,sha,require
from agent import receipt,Loop
out=R/'final_test_20261029';out.mkdir(exist_ok=True)
release=read(R/'release.json')
for name,h in release['runtime_hashes'].items():require(sha(R/'runtime'/name)==h,'Runtime changed: '+name)
entries=[]
for label in ['typea','typeb']:
 session=(R/label/'session').resolve();loop=Loop(session);s=loop.s;c=loop.c
 require(s['status'] in ('trial_budget_complete','stopped') and s['pending'] is None,'Search not finished')
 require(s['best'],'No selected model');trial=loop.trial(s['best']);folder=loop.folder(trial);r=receipt(folder)
 require(r['recipe']['feature_program']=='base','This evaluator must be extended for generated features before scoring')
 m=read(Path(c['active_cache'])/'manifest.json')
 entries.append(dict(label=label,session=str(session),trial=s['best'],baseline=s['trials'][0]['id'],
  model=str(folder/'model.ubj'),model_sha256=r['model_sha256'],threshold=r['metrics']['threshold'],recipe=r['recipe'],
  test_window=m['windows']['test'],features=m['features'],trainval_cache=c['active_cache'],trainval_manifest_sha256=sha(Path(c['active_cache'])/'manifest.json'),
  source_seals={n:sha(session/n) for n in ['state.json','config.json','mapping.json','code_seal.json']},
  trial_receipt_sha256=sha(folder/'complete.json'),confirmation_freeze_sha256=sha(R/label/'confirmation/frozen.json')))
f=dict(authorized_request='2026-10-29: user requested final testing of TypeA and TypeB after training completed',
 frozen_at=time.time(),framework_commit=release['commit'],models=entries,threshold_selection=False,training=False,agent_calls=False)
p=out/'frozen.json'
if p.exists():
 previous=read(p);require(previous['models']==entries,'Existing frozen models differ')
else:save(p,f)
print(json.dumps([dict(task=e['label'],trial=e['trial'],threshold=e['threshold'],test_window=e['test_window']) for e in entries]))
