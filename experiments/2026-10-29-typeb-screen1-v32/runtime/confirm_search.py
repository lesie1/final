"""Post-search later-VAL confirmation. No Agent calls and no TEST access."""
import argparse
import copy
from pathlib import Path
import sys
sys.dont_write_bytecode=True
from common import read,save,sha,require
from history_store import checked_manifest
from feature_programs import materialize
from prediction import predict
from evaluation import metrics
from agent import receipt
from operating_point import choose, load


def confirm(session,out):
    import xgboost as xgb
    session,out=Path(session),Path(out)
    state=read(session/'state.json');c=read(session/'config.json')
    require(state['status'] in ('stopped','trial_budget_complete'),'Search must finish before later-VAL confirmation')
    require(state['best'] and state['trials'][0]['status']=='complete','Need complete baseline and winner')
    names=list(dict.fromkeys([state['trials'][0]['id'],state['best']]))
    entries=[]
    mode=c.get('operating_point','selection_window')
    for name in names:
        folder=session/'trials'/name/'attempt_01';r=receipt(folder)
        # The selection-window point is always scored too, so the two rules are compared on unseen dates.
        points=dict(deployed=choose(folder,mode),selection_window=choose(folder,'selection_window'))
        entries.append(dict(trial=name,model_sha256=r['model_sha256'],threshold=points['deployed']['threshold'],
            recipe=r['recipe'],operating_points=points))
    out.mkdir(parents=True,exist_ok=True)
    if (out/'frozen.json').exists():require(read(out/'frozen.json')==entries,'Frozen entries changed')
    else:save(out/'frozen.json',entries)
    reports=[]
    for e in entries:
        target=out/e['trial'];target.mkdir(exist_ok=True)
        if (target/'report.json').exists():reports.append(read(target/'report.json'));continue
        base=checked_manifest(c);cc,m=materialize(c,e['recipe']['feature_program'],base)
        m=copy.deepcopy(m);m['windows']['val']=c['confirmation_window']
        folder=session/'trials'/e['trial']/'attempt_01';scored={};tables={}
        for label,op in e['operating_points'].items():
            key=(op['model'],op['step'])
            if key not in tables:tables[key]=predict(load(folder,op,2),cc,e['recipe'],m,'val',op['threshold'])[0]
            scored[label]=dict(source=op['source'],step=op['step'],threshold=op['threshold'],reason=op.get('reason'),
                               metrics=metrics(tables[key],op['threshold']))
            tables[key].to_csv(target/f'devices_{label}.csv',index=False)
        result=dict(trial=e['trial'],model_sha256=e['model_sha256'],metrics=scored['deployed']['metrics'],
            operating_points=scored,confirmation_window=c['confirmation_window'],threshold_reselected=False,test_opened=False,
            note='Later VAL was withheld from this search; earlier historical VAL use means it is not a fresh independent test.')
        save(target/'report.json',result);reports.append(result)
    save(out/'summary.json',dict(reports=reports,test_opened=False,agent_feedback=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--session',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();confirm(a.session,a.out)
