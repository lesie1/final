"""Freeze baseline/adopted model, then evaluate the previously exposed TEST."""
import sys
from pathlib import Path
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'runtime'));sys.path.insert(0,str(R));sys.dont_write_bytecode=True
from common import read,save,sha,require,event
from agent import receipt
from operating_point import choose,load
from feature_programs import materialize
from prediction import predict
from evaluation import metrics,lead_summary
from prepare import build

def main():
    session=R/'typeb/session';c=read(session/'config.json');s=read(session/'state.json')
    require(s['status'] in ('stopped','trial_budget_complete'),'Search must finish before TEST')
    out=R/'typeb/historical_test';out.mkdir(exist_ok=True)
    names=list(dict.fromkeys([s['trials'][0]['id'],s['best']]))
    entries=[]
    for name in names:
        folder=session/'trials'/name/'attempt_01';r=receipt(folder)
        entries.append(dict(trial=name,recipe=r['recipe'],features=r['features'],receipt=sha(folder/'complete.json'),
            operating_point=choose(folder,c['operating_point'])))
    frozen=dict(entries=entries,config=sha(session/'config.json'),state=sha(session/'state.json'),
        test_previously_exposed=True,independent_blind_test=False,threshold_reselected=False)
    if (out/'frozen.json').exists():require(read(out/'frozen.json')==frozen,'Frozen model changed')
    else:save(out/'frozen.json',frozen)
    event(out,'test_models_frozen',trials=names,test_previously_exposed=True)
    source=Path('/root/autodl-tmp/final_v31_20261028/final_test_20261029/typeb/cache_test')
    m=build(source,R/'raw_cache_test',True)
    cc=dict(c,active_cache=str(R/'raw_cache_test'),num_threads=2,score_workers=4)
    reports=[]
    for e in entries:
        dest=out/e['trial'];dest.mkdir(exist_ok=True)
        if (dest/'complete.json').exists():
            for n,h in read(dest/'complete.json')['files'].items():require(sha(dest/n)==h,'Saved test changed')
            reports.append(read(dest/'report.json'));continue
        cfg,manifest=materialize(cc,e['recipe']['feature_program'],m,final_test=True)
        require(manifest['features']==e['features'],'Feature order differs')
        folder=session/'trials'/e['trial']/'attempt_01';op=e['operating_point']
        table,stats=predict(load(folder,op,2),cfg,e['recipe'],manifest,'test',op['threshold'])
        table.to_csv(dest/'devices.csv',index=False)
        report=dict(trial=e['trial'],operating_point=op,metrics=metrics(table,op['threshold']),
            lead_summary=lead_summary(table),prediction_stats=stats,test_window=manifest['windows']['test'],
            independent_blind_test=False,threshold_reselected=False)
        save(dest/'report.json',report);save(dest/'complete.json',dict(files={n:sha(dest/n) for n in ['report.json','devices.csv']}))
        reports.append(report);event(out,'historical_test_scored',trial=e['trial'],metrics=report['metrics'])
    save(out/'summary.json',dict(reports=reports,agent_feedback=False,test_previously_exposed=True))
    require(sha(session/'state.json')==frozen['state'],'Search state changed')

if __name__ == "__main__":main()
