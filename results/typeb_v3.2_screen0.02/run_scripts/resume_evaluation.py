"""Infrastructure-only resume after adding a safe multiprocessing entry guard."""
import os,runpy,sys,time,traceback
from pathlib import Path
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'runtime'));sys.path.insert(0,str(R));sys.dont_write_bytecode=True
from common import read,save,sha,require,event
from supervisor import preserved,run

def main():
    p=Path('/proc/352279')
    deadline=time.time()+40
    while p.exists() and ') Z ' not in (p/'stat').read_text():
        require(time.time()<deadline,'Previous supervisor still active')
        time.sleep(1)
    preserved()
    expected={'typeb/historical_test/frozen.json':'2bb799ef837526a9cb1c7e266f8f6dad65a902573c132ab05da58a1f5ff2630e',
              'typeb/session/state.json':'52fd11129ee091bd657848c7bec76b6f1185f741ebd2edcc70ff7d98d1e49348'}
    for n,h in expected.items():require(sha(R/n)==h,'Frozen artifact changed: '+n)
    before=sha(R/'typeb/historical_test/events.jsonl')
    module=runpy.run_path(str(R/'evaluate.py'),run_name='spawn_import_check')
    require(callable(module['main']),'Missing evaluation entry point')
    require(sha(R/'typeb/historical_test/events.jsonl')==before,'Import unexpectedly executed evaluation')
    save(R/'evaluation_startup_repair.json',dict(reason='Prediction uses spawn; unguarded top-level evaluator recursively started scoring workers',
        old_script_sha256=sha(R/'evaluate_before_main_guard.py'),new_script_sha256=sha(R/'evaluate.py'),
        modification='Wrap existing evaluation statements in main(); execute only under __main__',
        unchanged_artifacts=expected,import_guard_test_passed=True,retraining=False,threshold_reselection=False,
        original_failed_log='historical_test.log',at=time.time()))
    event(R,'evaluation_infrastructure_repair',retraining=False,threshold_reselection=False,import_guard_test_passed=True)
    run([R/'evaluate.py'],'historical_test_resumed')
    for n,h in expected.items():require(sha(R/n)==h,'Frozen artifact changed during resume: '+n)
    preserved();save(R/'status.json',dict(status='complete',at=time.time(),publication_pending=True,infrastructure_repair='evaluation_startup_repair.json'))
    event(R,'raw_typeb_complete',old_sessions_unchanged=True,publication_pending=True)
    from report_export import main as export
    export()

if __name__=='__main__':
    try:main()
    except Exception as e:
        save(R/'status.json',dict(status='needs_attention',error=str(e),at=time.time()));traceback.print_exc();raise
