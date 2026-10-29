"""Single TypeB raw-data search, same v3.2 policy, at most 13 trials."""
import fcntl,json,os,signal,subprocess,sys,time,shutil,traceback
from pathlib import Path
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'runtime'));sys.dont_write_bytecode=True
from common import save,read,sha,event,require
from sample_builder import container_available

def preserved():
    for v in read(R/'source_preservation.json').values():
        for n,h in v['seals'].items():require(sha(Path(v['path'])/n)==h,'Old session changed: '+n)
    release=read(R/'release.json')
    for n,h in release['runtime_hashes'].items():require(sha(R/'runtime'/n)==h,'Runtime changed: '+n)
    for n,h in release['adapter_hashes'].items():require(sha(R/n)==h,'Adapter changed: '+n)

def run(args,phase,search=False):
    env=dict(os.environ,PYTHONPATH=str(R/'runtime'),PYTHONDONTWRITEBYTECODE='1')
    if search:env['TYPEA_SEARCH_ISOLATION']=str(R/'typeb/isolation.json')
    else:env.pop('DEEPSEEK_API_KEY',None);env.pop('TYPEA_SEARCH_ISOLATION',None)
    with (R/(phase+'.log')).open('ab') as log:
        p=subprocess.Popen([sys.executable,'-u',*map(str,args)],cwd=R/'runtime',env=env,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        start=time.monotonic();why=None
        while p.poll() is None:
            available=container_available();disk=shutil.disk_usage(R).free
            save(R/'status.json',dict(status='running',phase=phase,pid=p.pid,at=time.time(),
                available_memory_gib=round(available/2**30,2),disk_free_gib=round(disk/2**30,2)))
            if available<8*2**30:why='Available memory below 8 GiB'
            elif disk<12*2**30:why='Available disk below 12 GiB'
            elif time.monotonic()-start>30000:why='Worker time budget exhausted'
            if why:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                break
            time.sleep(10)
    preserved();event(R,'worker_finished',phase=phase,exit_code=p.returncode,error=why)
    require(p.returncode==0,why or 'Worker failed; inspect '+phase+'.log')

def main():
    lock=(R/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    preserved();require(read(R/'ready.json')['max_trials']==13,'Budget drift')
    require(read(R/'typeb/session/config.json')['screen_margin']==1.0,'Screen margin drift')
    event(R,'screen1_typeb_search_started',max_trials=13,screen_margin=1.0,raw_features=69,base_features=66,
          historical_test_previously_exposed=True)
    for _ in range(200):
        s=read(R/'typeb/session/state.json')
        if s['status'] in ('stopped','trial_budget_complete'):break
        run([R/'runtime/search_step.py','--session',R/'typeb/session'],'search',True)
    else:raise RuntimeError('Scheduler action cap exhausted')
    run([R/'runtime/confirm_search.py','--session',R/'typeb/session','--out',R/'typeb/confirmation'],'confirmation')
    run([R/'evaluate.py'],'historical_test')
    run([R/'report_export.py'],'report_export')
    preserved();save(R/'status.json',dict(status='complete',at=time.time(),publication_pending=True))
    event(R,'raw_typeb_complete',old_sessions_unchanged=True,publication_pending=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        save(R/'status.json',dict(status='needs_attention',error=str(e),at=time.time()))
        event(R,'batch_failed',error=str(e));traceback.print_exc();raise
