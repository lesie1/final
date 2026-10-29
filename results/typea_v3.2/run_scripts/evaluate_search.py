"""Explicitly authorized historical TEST replay at v3.2 frozen operating points."""
import copy, sys, time
from pathlib import Path
R=Path(__file__).resolve().parent
OLD=Path('/root/autodl-tmp/final_v31_20261028')
sys.path.insert(0,str(R/'runtime'));sys.dont_write_bytecode=True
from common import read,save,sha,require,event
from agent import receipt
from operating_point import choose,load
from history_math import verify
from feature_programs import materialize
from prediction import predict
from evaluation import metrics,lead_summary

def evaluate(label):
    assert label in ('typea','typeb')
    session=R/label/'session';c=read(session/'config.json');s=read(session/'state.json')
    require(s['status'] in ('stopped','trial_budget_complete'),'Search unfinished')
    dest=R/label/'historical_test';dest.mkdir(exist_ok=True)
    selected=list(dict.fromkeys([s['trials'][0]['id'],s['best']]))
    entries=[]
    for name in selected:
        folder=session/'trials'/name/'attempt_01';r=receipt(folder)
        entries.append(dict(trial=name,recipe=r['recipe'],features=r['features'],
            receipt_sha256=sha(folder/'complete.json'),points=dict(deployed=choose(folder,c['operating_point']),
            selection_window=choose(folder,'selection_window'))))
    frozen=dict(entries=entries,config_sha256=sha(session/'config.json'),state_sha256=sha(session/'state.json'),
        threshold_reselected=False,independent_blind_test=False,test_previously_exposed=True)
    if (dest/'frozen.json').exists():require(read(dest/'frozen.json')==frozen,'Freeze changed')
    else:save(dest/'frozen.json',frozen)
    event(dest,'historical_test_frozen',trials=selected,operating_point=c['operating_point'])
    cache=(Path('/root/autodl-tmp/typea_v2_frozen_test_20261028/out/raw_cache_test') if label=='typea'
           else OLD/'final_test_20261029/typeb/cache_test')
    m=verify(cache);cc=dict(c,active_cache=str(cache),num_threads=2,score_workers=4)
    results=[]
    for entry in entries:
        ccc,mm=materialize(cc,entry['recipe']['feature_program'],m,final_test=True)
        require(mm['features']==entry['features'],'Feature order drift')
        folder=session/'trials'/entry['trial']/'attempt_01'
        for point,op in entry['points'].items():
            out=dest/entry['trial']/point;out.mkdir(exist_ok=True,parents=True)
            if (out/'report.json').exists():
                done=read(out/'complete.json')
                for name,h in done['files'].items():require(sha(out/name)==h,'Replay artifact changed')
                results.append(read(out/'report.json'));continue
            table,stats=predict(load(folder,op,2),ccc,entry['recipe'],mm,'test',op['threshold'])
            table.to_csv(out/'devices.csv',index=False)
            report=dict(task=label,trial=entry['trial'],point=point,operating_point=op,
                metrics=metrics(table,op['threshold']),lead_summary=lead_summary(table),prediction_stats=stats,
                test_window=mm['windows']['test'],threshold_reselected=False,independent_blind_test=False)
            save(out/'report.json',report)
            save(out/'complete.json',dict(files={n:sha(out/n) for n in ['report.json','devices.csv']}))
            results.append(report);event(dest,'historical_test_scored',trial=entry['trial'],point=point,metrics=report['metrics'])
    save(dest/'summary.json',dict(reports=results,independent_blind_test=False,agent_feedback=False))
    require(sha(session/'state.json')==frozen['state_sha256'],'Search state changed during replay')

if __name__=='__main__':evaluate(sys.argv[1])
