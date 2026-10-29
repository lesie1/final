"""Bounded queue; one training worker, historical sessions never written."""
import fcntl,json,os,signal,subprocess,sys,time,shutil,traceback
from pathlib import Path
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'runtime'))
from common import save,read,sha,event,require
from sample_builder import container_available

def preserved():
    for label,v in read(R/'source_preservation.json').items():
        for n,h in v['seals'].items():require(sha(Path(v['path'])/n)==h,'Old session changed: '+label+'/'+n)
    for n,h in read(R/'release.json')['runtime_hashes'].items():require(sha(R/'runtime'/n)==h,'Runtime changed: '+n)

def run(task,arguments,phase,timeout=30000,search=False,isolated=False):
    task.mkdir(exist_ok=True,parents=True)
    env=dict(os.environ,PYTHONPATH=str(R/'runtime'),PYTHONDONTWRITEBYTECODE='1')
    if not search:env.pop('DEEPSEEK_API_KEY',None)
    if search or isolated:env['TYPEA_SEARCH_ISOLATION']=str(task/'isolation.json')
    else:env.pop('TYPEA_SEARCH_ISOLATION',None)
    with (task/(phase+'.log')).open('ab') as log:
        p=subprocess.Popen([sys.executable,'-u',*map(str,arguments)],cwd=R/'runtime',env=env,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        start=time.monotonic();why=None
        while p.poll() is None:
            mem=container_available();disk=shutil.disk_usage(R).free
            save(R/'status.json',dict(status='running',task=task.name,phase=phase,pid=p.pid,
                at=time.time(),available_memory_gib=round(mem/2**30,2),disk_free_gib=round(disk/2**30,2)))
            if mem<8*2**30:why='memory below 8 GiB'
            elif disk<12*2**30:why='disk below 12 GiB'
            elif time.monotonic()-start>timeout:why='bounded worker timeout'
            if why:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                break
            time.sleep(10)
    preserved()
    event(R,'worker_finished',task=task.name,phase=phase,exit_code=p.returncode,error=why)
    return p.returncode,why

def main():
    lock=(R/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    preserved(); event(R,'batch_started',maximum_search_trials=26,controlled_trials=12)
    failed=set()
    for step in range(400):
        work=False
        for label in ('typea','typeb'):
            task=R/label;s=read(task/'session/state.json')
            if label in failed or s['status'] in ('stopped','trial_budget_complete'):continue
            work=True
            rc,why=run(task,[R/'runtime/search_step.py','--session',task/'session'],'search',search=True)
            if rc:
                failed.add(label);save(task/'result_status.json',dict(status='needs_attention',error=why or 'Search worker failed; see search.log'))
        if not work:break
    else:raise RuntimeError('Queue step budget exceeded')
    # Finish and freeze searches before opening authorized historical TEST.
    for label in ('typea','typeb'):
        if label in failed:continue
        task=R/label
        rc,why=run(task,[R/'runtime/confirm_search.py','--session',task/'session','--out',task/'confirmation'],'confirmation')
        if rc:failed.add(label);continue
        rc,why=run(task,[R/'evaluate_search.py',label],'historical_test')
        save(task/'result_status.json',dict(status='complete' if not rc else 'needs_attention',error=why))
        if rc:failed.add(label)
    # Extension readiness is deliberately separate from the sealed v3.2 runtime.
    controlled=R/'controlled';controlled.mkdir(exist_ok=True)
    deadline=time.monotonic()+24*3600
    while not (R/'controlled_ready.json').exists():
        save(R/'status.json',dict(status='waiting_for_controlled_adapter',at=time.time()))
        if time.monotonic()>deadline:raise RuntimeError('Controlled adapter was not validated within 24h')
        time.sleep(30)
    rc,why=run(controlled,[R/'controlled.py','prepare'],'prepare',timeout=21600)
    if rc:raise RuntimeError(why or 'Controlled preparation failed')
    protocol=read(R/'protocol.json')['controlled']
    for seed in protocol['seeds']:
        for arm in protocol['arms']:
            name=f'{arm}_seed{seed}'
            rc,why=run(controlled,[R/'controlled.py','train',arm,str(seed)],name,timeout=43200,isolated=True)
            if rc:raise RuntimeError(why or 'Controlled trial failed: '+name)
    rc,why=run(controlled,[R/'controlled.py','evaluate'],'evaluation',timeout=43200)
    if rc:raise RuntimeError(why or 'Controlled evaluation failed')
    preserved()
    save(R/'status.json',dict(status='complete' if not failed else 'needs_attention',search_failures=sorted(failed),
        at=time.time(),publication_pending=True))
    event(R,'batch_finished',search_failures=sorted(failed),publication_pending=True,old_sessions_unchanged=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        save(R/'status.json',dict(status='needs_attention',error=str(e),at=time.time()))
        event(R,'batch_failed',error=str(e));traceback.print_exc();raise
