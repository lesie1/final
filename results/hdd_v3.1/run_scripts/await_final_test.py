"""Start the already authorized frozen HDD TEST once preparation fully finishes."""
import datetime, fcntl, hashlib, json, os, shutil, signal, subprocess, sys, time
from pathlib import Path
import psutil

ROOT = Path('/root/autodl-tmp/hdd_final_test_20261029')
WATCH = ROOT/'test_watch'
SOURCE = Path('/root/autodl-tmp/final_v31_20261028')
EVALUATOR = ROOT/'evaluate_hdd.py'
EXPECTED_SHA = '547f8cd1b0879aa46657a9aa52ba6e682d5fe3993970164ee497d9d768229a39'

def read(p): return json.loads(Path(p).read_text())
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(p,obj):
    p=Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,indent=2));t.replace(p)
def event(name,**kw):
    with (WATCH/'events.jsonl').open('a') as f:f.write(json.dumps(dict(at=now(),event=name,**kw))+'\n')
def status(phase,**kw):save(WATCH/'status.json',dict(at=now(),phase=phase,watcher_pid=os.getpid(),**kw))
def process_matches(pid,script):
    p=Path(f'/proc/{pid}/cmdline')
    try:return script.encode() in p.read_bytes().split(b'\0')
    except FileNotFoundError:return False
def resources():
    root=Path('/sys/fs/cgroup');s={k:int(v) for k,v in (l.split() for l in (root/'memory.stat').read_text().splitlines())}
    clean=max(0,s.get('file',0)-s.get('shmem',0)-s.get('file_dirty',0)-s.get('file_writeback',0))
    available=min(psutil.virtual_memory().available,int((root/'memory.max').read_text())-int((root/'memory.current').read_text())+clean)
    return dict(available_memory_gib=round(available/2**30,3),disk_free_gib=round(shutil.disk_usage(ROOT).free/2**30,3))

def main():
    WATCH.mkdir(exist_ok=True)
    lock=(WATCH/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (WATCH/'attempt.json').exists(),'A TEST attempt already exists; inspect it before recovery'
    assert hashlib.sha256(EVALUATOR.read_bytes()).hexdigest()==EXPECTED_SHA,'Evaluator changed'
    assert read(ROOT/'recovery_20261029/complete.json')['status']=='complete'
    assert not process_matches(681383,str(ROOT/'recover_confirmation.py'))
    env={k:os.environ[k] for k in ['PATH','HOME','USER','LOGNAME','LANG','LC_ALL','LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES'] if k in os.environ}
    env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    event('waiting_for_complete_cache',test_authorized=True,additional_training_authorized=False)
    deadline=time.monotonic()+24*3600
    while True:
        prep=read(ROOT/'status.json');r=resources()
        if prep.get('stage')=='failed':raise RuntimeError('Cache preprocessing failed; inspect preparation.log')
        if r['disk_free_gib']<40:raise RuntimeError('Less than 40 GiB free disk; TEST not launched')
        if (ROOT/'ready.json').exists():
            if not process_matches(670219,str(ROOT/'prepare_test_cache.py')):break
            phase='waiting_for_preparation_exit'
        else:
            if not process_matches(670219,str(ROOT/'prepare_test_cache.py')):raise RuntimeError('Preprocessing exited without ready.json')
            phase='waiting_for_cache'
        status(phase,preparation_stage=prep.get('stage'),written_devices=prep.get('written_devices'),**r)
        if time.monotonic()>deadline:raise RuntimeError('Cache readiness wait exceeded 24 hours')
        time.sleep(30)
    assert not (ROOT/'evaluation/error.json').exists(),'Prior TEST error requires explicit recovery'
    check=subprocess.run([sys.executable,str(EVALUATOR),'--check-ready'],env=env,capture_output=True,text=True,timeout=120)
    assert check.returncode==0, 'TEST readiness check failed'
    ready=json.loads(check.stdout);assert ready['ready'],str(ready)
    r=resources();assert r['available_memory_gib']>=8 and r['disk_free_gib']>=40,'Insufficient TEST resources'
    state=read(SOURCE/'hdd/session/state.json');assert state['status']=='trial_budget_complete' and len(state['trials'])==13
    assert state['best']=='trial_004','Winner changed after completed confirmation'
    with (ROOT/'evaluation.log').open('ab') as log:
        child=subprocess.Popen([sys.executable,'-u',str(EVALUATOR)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(WATCH/'attempt.json',dict(at=now(),pid=child.pid,evaluator_sha256=EXPECTED_SHA,best=state['best'],credential_free=True))
        event('test_evaluator_started',pid=child.pid,trial=state['best'])
        start=time.monotonic()
        try:
            while child.poll() is None:
                r=resources();status('evaluating',worker_pid=child.pid,**r)
                if r['disk_free_gib']<40:raise RuntimeError('Less than 40 GiB free disk during TEST')
                if r['available_memory_gib']<8:raise RuntimeError('Less than 8 GiB reclaimable-aware memory headroom')
                if time.monotonic()-start>24*3600:raise RuntimeError('TEST process exceeded 24 hours')
                time.sleep(10)
            if child.returncode:raise RuntimeError(f'TEST evaluator exited {child.returncode}; inspect evaluation.log')
        finally:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    assert (ROOT/'evaluation/complete.json').exists(),'TEST completion receipt missing'
    report=read(ROOT/'evaluation/report_test.json');status('complete',trial=report['trial'],metrics=report['metrics'],publication_pending=True)
    event('test_completed',trial=report['trial'],metrics=report['metrics'],publication_pending=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        if WATCH.exists():status('needs_attention',error=str(e));event('watch_or_test_failed',error=str(e))
        raise
