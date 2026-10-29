"""Finish the report-only repair after training and frozen TEST have completed."""
import ast,sys,time
from pathlib import Path
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'runtime'));sys.path.insert(0,str(R));sys.dont_write_bytecode=True
from common import read,save,sha,require,event
from supervisor import preserved

def main():
    preserved()
    original=ast.parse((R/'evaluate_before_main_guard.py').read_text())
    changed=ast.parse((R/'evaluate.py').read_text())
    start=next(i for i,n in enumerate(original.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='session' for t in n.targets))
    mainfn=next(n for n in changed.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    require(ast.dump(ast.Module(body=original.body[start:],type_ignores=[]))==ast.dump(ast.Module(body=mainfn.body,type_ignores=[])),'Evaluation computation changed')
    state=read(R/'typeb/session/state.json');require(state['status'] in ('stopped','trial_budget_complete'),'Search still active')
    frozen=read(R/'typeb/historical_test/frozen.json');require(sha(R/'typeb/session/state.json')==frozen['state'],'Search state drift')
    for x in read(R/'typeb/historical_test/summary.json')['reports']:
        p=R/'typeb/historical_test'/x['trial']
        for n,h in read(p/'complete.json')['files'].items():require(sha(p/n)==h,'Test receipt mismatch')
    save(R/'report_repair.json',dict(at=time.time(),old_script=sha(R/'report_export_before_threshold_fix.py'),new_script=sha(R/'report_export.py'),
        changes=['Remove duplicated threshold keyword; check equality with frozen operating point',
                 'Include CV/confirmation and early-stop reason, data quality manifests, and infrastructure repair provenance'],
        evaluation_ast_guard_only=True,training_and_test_unchanged=True))
    if read(R/'status.json')['status']!='complete':event(R,'report_infrastructure_repair',training_and_test_unchanged=True)
    save(R/'status.json',dict(status='complete',at=time.time(),publication_pending=True,
        trials=len(state['trials']),adopted=state['best'],repairs=['evaluation_startup_repair.json','report_repair.json']))
    from report_export import main as export
    export();print('PUBLICATION_FINALIZED',flush=True)

if __name__=='__main__':main()
