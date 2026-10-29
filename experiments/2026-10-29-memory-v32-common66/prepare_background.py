"""Prepare common inputs on CPU while the serial GPU search runs."""
import json,os,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parent
if (R/'controlled/ready.json').exists():
    print('ALREADY_PREPARED');raise SystemExit(0)
pidfile=R/'prepare_pid.json'
if pidfile.exists():
    pid=json.loads(pidfile.read_text())['pid'];p=Path('/proc')/str(pid)
    if p.exists() and b'controlled.py' in (p/'cmdline').read_bytes():
        print('PREPARATION_ALREADY_RUNNING',pid);raise SystemExit(0)
env={k:v for k,v in os.environ.items() if not any(t in k.upper() for t in ['API_KEY','TOKEN','SECRET','PASSWORD'])}
env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
with (R/'controlled_prepare.log').open('ab') as log:
    p=subprocess.Popen([sys.executable,'-u',str(R/'controlled.py'),'prepare'],cwd=R,env=env,
                       stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
pidfile.write_text(json.dumps(dict(pid=p.pid,at=time.time())))
print('PREPARATION_STARTED',p.pid)
