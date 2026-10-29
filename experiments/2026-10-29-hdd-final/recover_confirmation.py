"""Credential-free retry of the interrupted final CV, then original confirmation/export."""
import datetime, fcntl, hashlib, json, os, shutil, signal, subprocess, sys, time
from pathlib import Path
import psutil

BASE = Path('/root/autodl-tmp/hdd_final_test_20261029')
ROOT = BASE / 'recovery_20261029'
SOURCE = Path('/root/autodl-tmp/final_v31_20261028')
TASK = SOURCE / 'hdd'
SESSION = TASK / 'session'
RUNTIME = SOURCE / 'runtime'
PREP_PID = 670219
ROOT.mkdir(exist_ok=True)

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text())
def save(p, obj):
    p = Path(p); temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False)); temp.replace(p)
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()
def event(name, **data):
    with (ROOT/'events.jsonl').open('a') as f:
        f.write(json.dumps(dict(at=now(), event=name, **data), ensure_ascii=False)+'\n')
def prep_matches():
    p = Path(f'/proc/{PREP_PID}/cmdline')
    return p.exists() and str(BASE/'prepare_test_cache.py').encode() in p.read_bytes().split(b'\0')
def resources():
    c = Path('/sys/fs/cgroup')
    stat = dict((k,int(v)) for k,v in (line.split() for line in (c/'memory.stat').read_text().splitlines()))
    limit = int((c/'memory.max').read_text()); used = int((c/'memory.current').read_text())
    reclaim = max(0, stat.get('file',0)-stat.get('shmem',0)-stat.get('file_dirty',0)-stat.get('file_writeback',0))
    headroom = min(psutil.virtual_memory().available, limit-used+reclaim)
    return dict(available_memory_gib=round(headroom/2**30,3), cgroup_used_gib=round(used/2**30,3),
                reclaimable_clean_file_gib=round(reclaim/2**30,3), disk_free_gib=round(shutil.disk_usage(BASE).free/2**30,3))
def run(args, phase, timeout=21600):
    env = {k:os.environ[k] for k in ['PATH','HOME','USER','LOGNAME','LANG','LC_ALL','LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES'] if k in os.environ}
    env.update(TYPEA_SEARCH_ISOLATION=str(TASK/'isolation.json'), PYTHONPATH=str(RUNTIME),
               PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',MKL_NUM_THREADS='8')
    with (ROOT/(phase+'.log')).open('a') as log:
        p = subprocess.Popen([sys.executable,'-u',*args], cwd=RUNTIME, env=env, stdin=subprocess.DEVNULL,
                             stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        event('worker_started',phase=phase,pid=p.pid); start=time.monotonic()
        try:
            while p.poll() is None:
                status=dict(status='running',phase=phase,pid=p.pid,supervisor_pid=os.getpid(),at=now(),**resources())
                save(ROOT/'status.json',status)
                if status['available_memory_gib'] < 8: raise RuntimeError('less than 8 GiB reclaimable-aware headroom')
                if status['disk_free_gib'] < 40: raise RuntimeError('less than 40 GiB disk space')
                if time.monotonic()-start > timeout: raise RuntimeError('recovery step time limit')
                time.sleep(10)
            if p.returncode: raise RuntimeError(f'{phase} exited {p.returncode}; inspect recovery log')
        finally:
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try: p.wait(timeout=20)
                except subprocess.TimeoutExpired: os.killpg(p.pid,signal.SIGKILL);p.wait()
        event('worker_finished',phase=phase,exit_code=p.returncode)

def main():
    lock=(ROOT/'recovery.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    original_lock=(SOURCE/'supervisor.lock').open('a');fcntl.flock(original_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'complete.json').exists(), 'Recovery already completed'
    assert not Path('/proc/522855').exists(), 'Original supervisor still exists'
    assert prep_matches(), 'Original cache process missing or PID reused'
    os.kill(PREP_PID,signal.SIGSTOP)
    state=read(SESSION/'state.json');config=read(SESSION/'config.json')
    assert len(state['trials'])==config['max_trials']==13
    assert state['pending']['action']=='train' and state['pending']['trial']=='trial_013'
    budget=(state['calls'],state['tokens']); assert budget==(50,989281)
    attempt=SESSION/'trials/trial_013/attempt_01'
    assert (attempt/'complete.json').exists() and not (attempt/'cv/oof_complete.json').exists()
    backup=ROOT/'before_recovery';backup.mkdir(exist_ok=False)
    for src in [*SESSION.glob('*.json'),*SESSION.glob('*.jsonl'),TASK/'result_status.json',TASK/'live.json',TASK/'search.log',attempt/'cv.log',attempt/'comparison_parent.json',attempt/'cv_plan.json']:
        if src.is_file():
            dest=backup/src.relative_to(SOURCE);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
    shutil.copytree(attempt/'cv',backup/'hdd/session/trials/trial_013/attempt_01/cv')
    seals=dict(read(BASE/'source_preservation.json')['files'])
    for t in state['trials']:
        folder=SESSION/'trials'/t['id']/'attempt_01'
        for name in ['complete.json','model.ubj','result.json','plan.json','candidate.py']:
            p=folder/name
            if p.exists():seals[str(p)]=sha(p)
    for p,h in seals.items(): assert sha(p)==h, f'Original seal mismatch: {p}'
    save(ROOT/'preservation.json',dict(at=now(),files=seals,backup_files={str(p.relative_to(backup)):sha(p) for p in backup.rglob('*') if p.is_file()}))
    event('recovery_started',reason='Original supervisor terminated CV at its available-memory guard during concurrent cache ingestion',
          changes='Pause preprocessing and account for clean reclaimable file cache; retry same completed primary trial CV only',
          initial_trials=13,initial_calls=budget[0],initial_tokens=budget[1],credentials_required=False,primary_model_retrained=False)
    run([str(RUNTIME/'search_step.py'),'--session',str(SESSION)],'retry_trial013_cv')
    state=read(SESSION/'state.json')
    assert len(state['trials'])==13 and (state['calls'],state['tokens'])==budget and state['pending'] is None
    run([str(RUNTIME/'search_step.py'),'--session',str(SESSION)],'close_trial_budget',1800)
    state=read(SESSION/'state.json')
    assert state['status']=='trial_budget_complete' and len(state['trials'])==13 and (state['calls'],state['tokens'])==budget
    run([str(RUNTIME/'confirm_search.py'),'--session',str(SESSION),'--out',str(TASK/'confirmation')],'confirmation')
    run([str(RUNTIME/'agent.py'),'export','--session',str(SESSION)],'export',1800)
    for p,h in seals.items(): assert sha(p)==h, f'Original seal changed: {p}'
    confirmation=read(TASK/'confirmation/summary.json');assert not confirmation['test_opened']
    outcome=dict(status='complete',error=None,export_exit=0,trials=13,best=state['best'],steps=2,test_opened=False,
                 confirmation_is_independent_test=False,recovery=str(ROOT),initial_failed_status=str(backup/'hdd/result_status.json'))
    save(TASK/'result_status.json',outcome)
    save(ROOT/'complete.json',dict(at=now(),**outcome,source_seals_verified=len(seals),calls=budget[0],tokens=budget[1]))
    save(ROOT/'status.json',dict(at=now(),**outcome));event('recovery_complete',best=state['best'],test_opened=False)

if __name__=='__main__':
    try: main()
    except Exception as exc:
        save(ROOT/'status.json',dict(status='needs_attention',at=now(),error=str(exc)))
        event('recovery_failed',error=str(exc));raise
    finally:
        if prep_matches():
            os.kill(PREP_PID,signal.SIGCONT);event('preparation_resumed',pid=PREP_PID)
        else: event('preparation_resume_unavailable',pid=PREP_PID)
